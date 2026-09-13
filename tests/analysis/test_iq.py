"""原始 IQ 分类分支（P4，``iq_waveform_v1`` / ``amc_iq_classify_v1``）。

这一组用例守护的是"**换输入口径**"这件事最容易出错的地方：

* ``iq_waveform_v1`` 张量形状、单位 RMS、居中取值、窗口不够时**硬报错不补零**；
* 前段（混频 → 低通 → 抽取）与 A09 特征通路**同一实现**：分析率与抽样比逐位一致，
  否则训练与推理会看到两种不同的输入分布；
* 清单读取器与时频图检测清单读取器**两个方向都拒绝对方**（``read_model_manifest``
  不会因为新增 IQ 分支而被放宽）；
* 清单声明的归一化／每带宽采样点／低通抽头与实现不一致时直接拒绝，而不是静默换口径；
* 结果契约字段冻结、JSON 安全、耗时字段只有三项；
* 服务层／命令行／GUI 的接入点：动作名、真值比对规则、``--samples`` 默认值、
  IQ 清单分流。

**不涉及**训练侧（``training/`` 下的脚本由 ``test_iq_training_tools.py`` 覆盖），
也**不要求**安装 torch：本文件里的"真模型"是一个用 ``onnx.helper`` 现场拼的
极小图，只依赖 ``onnx`` + ``onnxruntime``。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.storage import file_digest  # noqa: E402
from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.ml import (  # noqa: E402
    AMC_CLASSES,
    CLASS_LABELS,
    CLASS_SET_A09,
    CLASS_SET_CUSTOM,
    DEFAULT_IQ_SAMPLES,
    IQModelError,
    IQ_LAYOUT,
    IQ_MANIFEST_SCHEMA_VERSION,
    IQ_NORMALIZATION,
    IQ_ONNX_CONTRACT,
    IQ_RESULT_CONTRACT,
    IQ_WAVEFORM_CONTRACT,
    ManifestError,
    amc_iq_classify,
    class_labels,
    class_set_name,
    extract_features,
    iq_scores,
    iq_waveform,
    load_iq_runner,
    read_iq_manifest,
    write_iq_manifest,
)
from signal_analysis.ml.amc import SAMPLES_PER_BAND  # noqa: E402
from signal_analysis.ml.runtime import (  # noqa: E402
    IQModelRunner,
    RuntimeUnavailable,
    runtime_version,
)
from signal_analysis.storage import Workspace  # noqa: E402
from signal_analysis.tasks import run_job  # noqa: E402

RATE = 200_000.0
WINDOW = 512


def _scene(mode, snr_db=25.0, offset=40_000.0, bandwidth=30_000.0, seed=7, duration=0.08):
    """按生成器样式造一段 IQ，返回 ``(samples, rate)``；``mode=None`` 为纯噪声。"""
    signals = [] if mode is None else [{"mode": mode, "offset": offset, "bandwidth": bandwidth,
                                        "power_dbfs": -6.0}]
    noise = {"enabled": True, "bandwidth": RATE, "snr_db": snr_db} if snr_db else None
    samples, _ = generate_iq(RATE, duration, signals, noise=noise, seed=seed)
    return samples, RATE


# ---------------------------------------------------------------------------
# 现场拼一个极小的合法 ONNX 分类器（只依赖 onnx）
# ---------------------------------------------------------------------------


def _tiny_iq_model(path, classes, samples):
    """``(1,2,N)`` → 均值 → 线性 → softmax，输出 ``(1,C)`` 概率。

    刻意让每个类别的权重不同，这样"预测结果取决于输入"而不是常量，端到端链路
    才真的被走到（常量输出的模型会让形状与名称的检查变得没有意义）。
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weight = (np.arange(len(classes) * 2, dtype=np.float32).reshape(2, len(classes)) + 1.0) / 8.0
    bias = np.linspace(-0.5, 0.5, len(classes)).astype(np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["iq"], ["mean"], axes=[2], keepdims=0),
        helper.make_node("MatMul", ["mean", "W"], ["logits"]),
        helper.make_node("Add", ["logits", "B"], ["biased"]),
        helper.make_node("Softmax", ["biased"], ["scores"], axis=1),
    ]
    graph = helper.make_graph(
        nodes, "tiny-iq",
        [helper.make_tensor_value_info("iq", TensorProto.FLOAT, [1, 2, int(samples)])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, len(classes)])],
        [numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "B")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = min(model.ir_version, 9)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """``(清单路径, 清单, 模型路径, 类别)``，类别沿用 A09 六类。"""
    root = tmp_path_factory.mktemp("iq_model")
    model = _tiny_iq_model(root / "iq.onnx", AMC_CLASSES, WINDOW)
    manifest_path = root / "iq_manifest.json"
    manifest, library = write_iq_manifest(manifest_path, model, identifier="tiny-iq",
                                          version="0.2.0", samples=WINDOW,
                                          training={"dataset": "单测"}, notes="单测模型")
    return manifest_path, manifest, library, list(AMC_CLASSES)


def _stub_runner(classes=AMC_CLASSES, scores=None, manifest=None):
    """注入用的假会话：只实现 ``run`` 与 ``manifest``。"""

    class Stub:
        runtime_version = runtime_version() or "0.0.0"

        def __init__(self):
            self.manifest = dict(manifest or {
                "id": "stub", "version": "0.0.1", "classes": list(classes),
                "output": {"name": "scores", "classes": list(classes)},
                "input": {"name": "iq", "samples": WINDOW, "channels": 2,
                          "layout": IQ_LAYOUT},
                "preprocess": {"normalization": IQ_NORMALIZATION,
                               "samples_per_band": SAMPLES_PER_BAND},
                "class_set": class_set_name(classes), "sha256": "0" * 64,
            })
            self.calls = []

        def run(self, waveform):
            self.calls.append(np.asarray(waveform))
            if scores is not None:
                values = scores
            else:
                # 合法概率向量（和为 1）：iq 入口会原样采信而不是再 softmax 一次
                values = np.full(len(classes), 0.05)
                values[-1] = 1.0 - 0.05 * (len(classes) - 1)
            return np.asarray(values, dtype=np.float32)

    return Stub()


# ---------------------------------------------------------------------------
# 输入张量：iq_waveform_v1
# ---------------------------------------------------------------------------


def test_contracts_are_wire_stable():
    assert IQ_WAVEFORM_CONTRACT == "iq_waveform_v1"
    assert IQ_ONNX_CONTRACT == IQ_WAVEFORM_CONTRACT
    assert IQ_RESULT_CONTRACT == "amc_iq_classify_v1"
    assert IQ_LAYOUT == "iq_channels_first_v1"
    assert IQ_NORMALIZATION == "unit_rms"
    assert IQ_MANIFEST_SCHEMA_VERSION == 1
    assert DEFAULT_IQ_SAMPLES == 1024
    assert CLASS_SET_A09 == "a09" and CLASS_SET_CUSTOM == "custom"
    # 结果契约字段冻结：GUI／报表／CLI 都按这个集合取值
    result = amc_iq_classify(*_scene("qpsk", snr_db=30.0), runner=_stub_runner(), config=None)
    assert set(result) == {"contract", "algorithm", "classes", "class_set", "labels",
                           "waveform", "snr_estimate_db", "prediction", "model", "timing",
                           "pending"}
    assert set(result["timing"]) == {"preprocess_ms", "inference_ms", "total_ms"}


def test_waveform_shape_normalization_and_centering():
    samples, rate = _scene("qpsk")
    tensor, meta = iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=WINDOW)
    assert tensor.shape == (2, WINDOW) and tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))
    # I/Q 通道就是实部／虚部，且窗口按复窗口 RMS 归一化为单位平均功率：
    # mean(|z|²) = mean(I² + Q²) = 1，等价于两个通道平方均值之和为 1
    power = np.mean(tensor.astype(np.float64)[0] ** 2 + tensor.astype(np.float64)[1] ** 2)
    assert float(power) == pytest.approx(1.0, rel=1e-5)
    assert meta["contract"] == IQ_WAVEFORM_CONTRACT and meta["layout"] == IQ_LAYOUT
    assert meta["channels"] == 2 and meta["samples"] == WINDOW
    assert meta["normalization"] == IQ_NORMALIZATION
    # 居中取值：start = (可用样本 - N) // 2，且不补零
    assert meta["window_start"] == (meta["analysis_samples"] - WINDOW) // 2
    assert meta["analysis_samples"] >= WINDOW
    # 归一化不吞掉功率信息：绝对功率另行测量并留档
    assert meta["power_dbfs"] < 0.0 and meta["rms"] > 0.0
    assert meta["crest_factor"] == pytest.approx(meta["peak"] / meta["rms"], rel=1e-6)
    json.dumps(meta, ensure_ascii=False, allow_nan=False)


def test_waveform_uses_the_same_front_end_as_the_feature_path():
    """两条通路的前段必须是同一份实现，否则训练与推理的输入分布会分叉。"""
    samples, rate = _scene("qam16", snr_db=30.0, offset=-40_000.0, bandwidth=10_000.0,
                           duration=0.5)
    _, meta = iq_waveform(samples, rate, -40_000.0, 10_000.0, window_samples=WINDOW)
    _, info = extract_features(samples, rate, -40_000.0, 10_000.0)
    assert meta["analysis_rate_hz"] == info["analysis_rate_hz"]
    assert meta["decimation"] == info["decimation"]
    assert meta["samples_per_band"] == SAMPLES_PER_BAND
    # 分析率由抽样比决定（抽取是整数倍数，因此只能是采样率/整数）
    assert meta["analysis_rate_hz"] == pytest.approx(rate / meta["decimation"], rel=1e-12)
    assert meta["decimation"] == int(np.floor(rate / (SAMPLES_PER_BAND * 10_000.0))) > 1


def test_waveform_is_deterministic():
    samples, rate = _scene("fm")
    first, first_meta = iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=WINDOW)
    second, second_meta = iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=WINDOW)
    assert np.array_equal(first, second) and first_meta == second_meta


def test_waveform_refuses_short_windows_instead_of_padding():
    samples, rate = _scene("qpsk", duration=0.002)  # 很短的记录
    with pytest.raises(ValueError, match="不补零"):
        iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=WINDOW)


@pytest.mark.parametrize("window,message", [
    (32, "64～65536"),
    (65_537, "64～65536"),
])
def test_waveform_rejects_out_of_range_window(window, message):
    samples, rate = _scene("qpsk")
    with pytest.raises(ValueError, match=message):
        iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=window)


def test_waveform_rejects_invalid_bands():
    samples, rate = _scene("qpsk")
    with pytest.raises(ValueError, match="采样率"):
        iq_waveform(samples, 0.0, 0.0, None, window_samples=WINDOW)
    with pytest.raises(ValueError, match="占用带宽"):
        iq_waveform(samples, rate, 0.0, rate * 2, window_samples=WINDOW)
    with pytest.raises(ValueError, match="占用带宽"):
        iq_waveform(samples, rate, 0.0, 0.0, window_samples=WINDOW)
    with pytest.raises(ValueError, match="中心频率"):
        iq_waveform(samples, rate, rate * 3, 10_000.0, window_samples=WINDOW)


def test_full_band_call_has_no_snr_estimate():
    """整段带宽分析时带内信噪比不可估：必须是 ``None``，而不是编一个数。"""
    samples, rate = _scene("qpsk")
    _, meta = iq_waveform(samples, rate, 0.0, None, window_samples=WINDOW)
    assert meta["offset_hz"] == 0.0 and meta["bandwidth_hz"] == rate
    assert meta["snr_estimate_db"] is None


# ---------------------------------------------------------------------------
# 标签集合
# ---------------------------------------------------------------------------


def test_class_set_identifier_and_labels():
    assert class_set_name(AMC_CLASSES) == CLASS_SET_A09
    assert class_set_name(list(reversed(AMC_CLASSES))) == CLASS_SET_CUSTOM
    assert class_set_name(["bpsk", "8psk"]) == CLASS_SET_CUSTOM
    labels = class_labels(list(AMC_CLASSES))
    # A09 六类用 :data:`CLASS_LABELS` 里的中文全称，逐类对齐
    assert labels == {name: CLASS_LABELS[name] for name in AMC_CLASSES}
    assert labels["qpsk"] != "qpsk"
    # 更宽的标签字典没有中文名可查：回落到原始标签，绝不凭空编一个
    assert class_labels(["8psk", "qpsk"]) == {"8psk": "8psk", "qpsk": CLASS_LABELS["qpsk"]}


# ---------------------------------------------------------------------------
# 清单：读写、篡改检测、口径门禁
# ---------------------------------------------------------------------------


def _placeholder_onnx(directory, name="iq.onnx"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"not-a-real-onnx")
    return path


def test_manifest_round_trip_and_digest(tmp_path):
    model = _placeholder_onnx(tmp_path / "run")
    manifest_path = tmp_path / "run" / "iq.json"
    manifest, library = write_iq_manifest(manifest_path, model, identifier="dut",
                                          version="0.3.0", opset=17, samples=WINDOW,
                                          classes=["bpsk", "qpsk"], notes="测试清单")
    assert manifest["id"] == "dut" and manifest["version"] == "0.3.0"
    assert manifest["contract"] == IQ_ONNX_CONTRACT
    assert manifest["input"] == {"name": "iq", "samples": WINDOW, "channels": 2,
                                 "layout": IQ_LAYOUT}
    assert manifest["output"]["classes"] == ["bpsk", "qpsk"]
    assert manifest["class_set"] == CLASS_SET_CUSTOM
    assert manifest["sha256"] == file_digest(model) and library == model.resolve()
    again, again_library = read_iq_manifest(manifest_path)
    assert again["classes"] == ["bpsk", "qpsk"] and again_library == library
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(IQModelError, match="摘要与清单不符"):
        read_iq_manifest(manifest_path)


def test_manifest_defaults_are_a09(tmp_path):
    model = _placeholder_onnx(tmp_path / "run")
    manifest_path = tmp_path / "run" / "iq.json"
    manifest, _ = write_iq_manifest(manifest_path, model)
    assert manifest["output"]["classes"] == list(AMC_CLASSES)
    assert manifest["class_set"] == CLASS_SET_A09
    assert manifest["input"]["samples"] == DEFAULT_IQ_SAMPLES
    # 冻结的 AMC_CLASSES 不会被写清单这件事改动
    assert list(AMC_CLASSES) == manifest["output"]["classes"]


def test_manifest_rejects_bad_paths_and_classes(tmp_path):
    model = _placeholder_onnx(tmp_path / "run")
    other = _placeholder_onnx(tmp_path / "elsewhere")
    with pytest.raises(IQModelError, match="必须位于清单目录内"):
        write_iq_manifest(tmp_path / "run" / "iq.json", other)
    wrong_suffix = tmp_path / "run" / "iq.bin"
    wrong_suffix.write_bytes(b"x")
    with pytest.raises(IQModelError, match=r"\.onnx"):
        write_iq_manifest(tmp_path / "run" / "bad.json", wrong_suffix)
    with pytest.raises(IQModelError, match="重复类别名"):
        write_iq_manifest(tmp_path / "run" / "bad.json", model, classes=["qpsk", "qpsk"])
    with pytest.raises(IQModelError, match="类别数"):
        write_iq_manifest(tmp_path / "run" / "bad.json", model,
                          classes=[f"c{index}" for index in range(65)])
    with pytest.raises(IQModelError, match="窗口采样点数"):
        write_iq_manifest(tmp_path / "run" / "bad.json", model, samples=8)


def test_manifest_rejects_wrong_contract_and_front_end(tmp_path):
    model = _placeholder_onnx(tmp_path / "run")
    path = tmp_path / "run" / "iq.json"
    write_iq_manifest(path, model, samples=WINDOW)
    payload = json.loads(path.read_text(encoding="utf-8"))

    def rewrite(mutate):
        changed = json.loads(json.dumps(payload))
        mutate(changed)
        path.write_text(json.dumps(changed), encoding="utf-8")

    # 时频图检测清单被送到 IQ 入口：必须报错并指路，而不是按错的契约推理
    rewrite(lambda item: item.update(contract="tf_image_v1"))
    with pytest.raises(IQModelError, match="ml-detect"):
        read_iq_manifest(path)
    # 前段三个量：写了就必须等于本实现的口径
    rewrite(lambda item: item["preprocess"].update(normalization="zero_mean"))
    with pytest.raises(IQModelError, match="归一化"):
        read_iq_manifest(path)
    rewrite(lambda item: item["preprocess"].update(samples_per_band=4.0))
    with pytest.raises(IQModelError, match="每带宽采样点数"):
        read_iq_manifest(path)
    rewrite(lambda item: item["preprocess"].update(lowpass_taps=17))
    with pytest.raises(IQModelError, match="低通抽头"):
        read_iq_manifest(path)
    # 通道数与通道排布
    rewrite(lambda item: item["input"].update(channels=1))
    with pytest.raises(IQModelError, match="2 通道"):
        read_iq_manifest(path)
    rewrite(lambda item: item["input"].update(layout="channels_last"))
    with pytest.raises(IQModelError, match="通道排布"):
        read_iq_manifest(path)
    # 摘要、版本、运行时、库路径
    rewrite(lambda item: item.update(sha256="zz"))
    with pytest.raises(IQModelError, match="64 位十六进制"):
        read_iq_manifest(path)
    rewrite(lambda item: item.update(schema_version=99))
    with pytest.raises(IQModelError, match="清单版本"):
        read_iq_manifest(path)
    rewrite(lambda item: item.update(runtime="tensorrt"))
    with pytest.raises(IQModelError, match="推理运行时"):
        read_iq_manifest(path)
    rewrite(lambda item: item.update(library=str((tmp_path / "absolute.onnx").resolve())))
    with pytest.raises(IQModelError, match="相对路径"):
        read_iq_manifest(path)


def test_time_frequency_reader_still_rejects_iq_manifests(tmp_path):
    """两个读取器互不放松：IQ 清单不能喂给时频图检测入口。"""
    from signal_analysis.ml import read_model_manifest

    model = _placeholder_onnx(tmp_path / "run")
    path = tmp_path / "run" / "iq.json"
    write_iq_manifest(path, model, samples=WINDOW)
    with pytest.raises(ValueError, match="tf_image_v1|契约"):
        read_model_manifest(path)


# ---------------------------------------------------------------------------
# 推理：真 ONNX 极小模型 + 注入会话
# ---------------------------------------------------------------------------


def test_iq_scores_with_a_real_onnx_model(tiny_model):
    manifest_path, manifest, _, classes = tiny_model
    samples, rate = _scene("qpsk", snr_db=30.0)
    tensor, _ = iq_waveform(samples, rate, 40_000.0, 30_000.0, window_samples=WINDOW)
    scores, resolved, probabilities = iq_scores(manifest_path, tensor)
    assert list(scores) == classes and resolved["id"] == "tiny-iq"
    assert probabilities.shape == (len(classes),)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    # 概率字典按清单类别顺序排列，且是概率向量的四舍五入备份
    assert [scores[name] for name in classes] == [round(float(value), 6)
                                                  for value in probabilities]
    with pytest.raises(IQModelError, match=r"\(2, 512\)"):
        iq_scores(manifest_path, np.zeros((1, WINDOW), dtype=np.float32))
    with pytest.raises(IQModelError, match=r"\(2, 512\)"):
        iq_scores(manifest_path, np.zeros((2, WINDOW - 1), dtype=np.float32))


def test_amc_iq_classify_end_to_end(tiny_model):
    manifest_path, manifest, _, classes = tiny_model
    samples, rate = _scene("qpsk", snr_db=30.0)
    result = amc_iq_classify(samples, rate, {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0},
                             model=str(manifest_path))
    assert result["contract"] == IQ_RESULT_CONTRACT
    assert result["algorithm"] == "amc_iq_onnx_v1:tiny-iq"
    assert result["classes"] == classes and result["class_set"] == CLASS_SET_A09
    assert result["labels"] == class_labels(classes)
    assert result["waveform"]["samples"] == WINDOW
    assert result["prediction"]["label"] in classes
    assert result["prediction"]["label_text"] == result["labels"][result["prediction"]["label"]]
    assert 0.0 < result["prediction"]["confidence"] <= 1.0
    assert result["model"]["id"] == "tiny-iq" and result["model"]["library"]
    assert result["pending"] and all(isinstance(item, str) and item for item in result["pending"])
    assert set(result["timing"]) == {"preprocess_ms", "inference_ms", "total_ms"}
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    # 可复现：除耗时外逐字段一致
    again = amc_iq_classify(samples, rate, {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0},
                            model=str(manifest_path))
    assert {key: value for key, value in again.items() if key != "timing"} == \
           {key: value for key, value in result.items() if key != "timing"}


def test_classify_uses_declared_defaults_and_rejects_unknown_config(tmp_path):
    model = _tiny_iq_model(tmp_path / "run" / "iq.onnx", ["bpsk", "qpsk"], WINDOW)
    manifest_path = tmp_path / "run" / "iq.json"
    write_iq_manifest(manifest_path, model, samples=WINDOW, classes=["bpsk", "qpsk"],
                      default_offset_hz=40_000.0, default_bandwidth_hz=30_000.0)
    samples, rate = _scene("qpsk", snr_db=30.0)
    # 不给 config：用清单声明的默认频带（这是"调用方不必知道频带"的唯一来源）
    result = amc_iq_classify(samples, rate, None, model=str(manifest_path))
    assert result["waveform"]["offset_hz"] == pytest.approx(40_000.0)
    assert result["waveform"]["bandwidth_hz"] == pytest.approx(30_000.0)
    assert result["class_set"] == CLASS_SET_CUSTOM and result["labels"]["bpsk"] == "bpsk"
    # config 覆盖清单默认值
    override = amc_iq_classify(samples, rate, {"offset_hz": 0.0, "bandwidth_hz": None},
                               model=str(manifest_path))
    assert override["waveform"]["offset_hz"] == 0.0
    # 未声明的字段一律拒绝：不给"静默忽略"的机会
    with pytest.raises(ValueError, match="不支持以下字段"):
        amc_iq_classify(samples, rate, {"nfft": 256}, model=str(manifest_path))
    with pytest.raises(ValueError, match="应为字典"):
        amc_iq_classify(samples, rate, ["offset_hz"], model=str(manifest_path))


def test_classify_requires_model_or_runner():
    samples, rate = _scene("qpsk")
    with pytest.raises(ValueError, match="IQ 分类器清单路径"):
        amc_iq_classify(samples, rate, None)


def test_classify_with_injected_runner_and_output_arity_guard():
    samples, rate = _scene("fm", snr_db=30.0)
    runner = _stub_runner()
    result = amc_iq_classify(samples, rate, {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0},
                             runner=runner)
    # 假会话的分数在最后一类最高：预测落在该类，置信度就是该类的概率
    assert result["prediction"]["label"] == AMC_CLASSES[-1]
    assert result["prediction"]["confidence"] == pytest.approx(0.75, abs=1e-6)
    assert result["prediction"]["margin"] == pytest.approx(0.70, abs=1e-6)
    assert runner.calls and runner.calls[0].shape == (2, WINDOW)
    # 会话输出维度与类别数不符时必须报错，不能截断或补齐
    bad = _stub_runner(scores=[0.1, 0.2, 0.7])
    with pytest.raises(IQModelError, match="输出维度 3 与类别数 6 不符"):
        amc_iq_classify(samples, rate, {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0},
                        runner=bad)


def test_low_snr_and_low_confidence_are_flagged_not_hidden():
    samples, rate = _scene("qpsk", snr_db=-10.0)
    runner = _stub_runner(scores=[0.34, 0.33, 0.33, 0.0, 0.0, 0.0])
    result = amc_iq_classify(samples, rate, {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0},
                             runner=runner)
    assert result["prediction"]["reliable"] is False
    # 低于 0.5 的区分度不足与低信噪比都会被点名，而不是静默给出结论
    assert ("带内信噪比" in result["prediction"]["reason"]
            or "区分度不足" in result["prediction"]["reason"])
    assert result["prediction"]["margin"] >= 0.0
    assert result["prediction"]["snr_note"] and result["snr_estimate_db"] is not None


def test_iq_model_runner_validates_shape_and_runtime(tmp_path, tiny_model):
    manifest_path, manifest, library, _ = tiny_model
    runner = IQModelRunner(manifest, library)
    tensor, _ = iq_waveform(*_scene("qpsk", snr_db=30.0), 40_000.0, 30_000.0,
                            window_samples=WINDOW)
    output = runner.run(tensor)
    assert np.asarray(output).shape == (1, len(manifest["classes"]))
    with pytest.raises(ValueError, match=r"\(2, 512\)"):
        runner.run(np.zeros((2, 8), dtype=np.float32))
    # 版本不满足清单要求时拒绝加载（这里用显式 version 走通分支，不依赖本机安装）
    with pytest.raises(ManifestError, match="版本过低"):
        IQModelRunner(manifest, library, version="1.0.0")
    loaded, resolved, resolved_library = load_iq_runner(manifest_path)
    assert resolved["id"] == manifest["id"] and resolved_library == library
    assert loaded.run(tensor).shape == (1, len(manifest["classes"]))


def test_classify_reports_missing_runtime(tmp_path, monkeypatch):
    """记录缺 onnxruntime 时的行为：给出安装提示而不是栈回溯。"""
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime")
    model = _placeholder_onnx(tmp_path / "run")
    path = tmp_path / "run" / "iq.json"
    write_iq_manifest(path, model)
    samples, rate = _scene("qpsk")
    with pytest.raises(RuntimeUnavailable, match="onnxruntime"):
        amc_iq_classify(samples, rate, None, model=str(path))


# ---------------------------------------------------------------------------
# 服务层
# ---------------------------------------------------------------------------


def _generate(workspace, signals, duration=0.08, snr=25.0, seed=3):
    return run_job({"workspace": str(workspace), "action": "generate", "sample_rate": RATE,
                    "duration": duration, "signals": signals, "seed": seed,
                    "noise": {"enabled": True, "bandwidth": RATE, "snr_db": snr}})


def test_service_amc_iq_classify_attaches_truth(tmp_path, tiny_model):
    manifest_path, _, _, _ = tiny_model
    workspace = tmp_path / "ws"
    asset = _generate(workspace, [{"mode": "qpsk", "offset": 40_000.0, "bandwidth": 30_000.0,
                                   "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": asset["id"], "model": str(manifest_path),
                   "config": {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0}})
    assert run["kind"] == "amc_iq_classify"
    assert run["contract"] == IQ_RESULT_CONTRACT
    assert run["summary"]["contract"] == IQ_RESULT_CONTRACT
    assert run["class_set"] == CLASS_SET_A09
    assert run["waveform"]["contract"] == IQ_WAVEFORM_CONTRACT
    truth = run["truth"]
    assert truth["available"] is True and truth["class"] == "qpsk" and truth["mode"] == "qpsk"
    assert truth["classes"] == list(AMC_CLASSES)
    assert isinstance(run["truth_hit"], bool)
    assert Workspace(workspace).get_run(run["run_id"])["prediction"] == run["prediction"]


def test_service_iq_truth_is_explicitly_not_applicable(tmp_path, tiny_model):
    manifest_path, _, _, _ = tiny_model
    workspace = tmp_path / "ws"
    # 生成样式不在 A09 字典内
    unmapped = _generate(workspace, [{"mode": "am", "offset": 40_000.0, "bandwidth": 20_000.0,
                                      "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": unmapped["id"], "model": str(manifest_path)})
    assert run["truth"]["available"] is False and run["truth_hit"] is None
    assert "不在 A09 六类字典内" in run["truth"]["reason"]
    # 多信号
    multi = _generate(workspace, [{"mode": "fm", "offset": 40_000.0, "bandwidth": 20_000.0,
                                   "power_dbfs": -6.0},
                                  {"mode": "ask2", "offset": -60_000.0, "bandwidth": 20_000.0,
                                   "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": multi["id"], "model": str(manifest_path)})
    assert run["truth"]["available"] is False and run["truth"]["count"] == 2
    assert run["truth_hit"] is None
    # 没有生成器真值
    demo = run_job({"workspace": str(workspace), "action": "demo", "count": 4096})
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": demo["id"], "model": str(manifest_path)})
    assert run["truth"]["available"] is False and "没有生成器真值" in run["truth"]["reason"]
    assert run["truth_hit"] is None


def test_service_iq_truth_respects_the_model_label_set(tmp_path):
    """模型标签集合不含该样式映射到的类别时也必须"不适用"，而不是判成未命中。"""
    model = _tiny_iq_model(tmp_path / "run" / "iq.onnx", ["bpsk", "8psk"], WINDOW)
    manifest_path = tmp_path / "run" / "iq.json"
    write_iq_manifest(manifest_path, model, samples=WINDOW, classes=["bpsk", "8psk"],
                      default_offset_hz=40_000.0, default_bandwidth_hz=30_000.0)
    workspace = tmp_path / "ws"
    asset = _generate(workspace, [{"mode": "qpsk", "offset": 40_000.0, "bandwidth": 30_000.0,
                                   "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": asset["id"], "model": str(manifest_path)})
    truth = run["truth"]
    assert truth["available"] is False and truth["class"] == "qpsk"
    assert "标签集合不含此类" in truth["reason"]
    assert truth["classes"] == ["bpsk", "8psk"] and run["truth_hit"] is None


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def test_cli_amc_iq_manifest_defaults_and_class_order(tmp_path, capsys):
    from signal_analysis.cli import main

    model = _placeholder_onnx(tmp_path / "run")
    code = main(["amc-iq-manifest", str(model), str(tmp_path / "run" / "iq.json"),
                 "--id", "dut", "--notes", "演示清单"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract"] == IQ_ONNX_CONTRACT
    assert summary["input"]["samples"] == DEFAULT_IQ_SAMPLES  # 不写 --samples 也是契约默认值
    assert summary["output"]["classes"] == list(AMC_CLASSES)
    assert summary["class_set"] == CLASS_SET_A09
    resolved, _ = read_iq_manifest(tmp_path / "run" / "iq.json")
    assert resolved["id"] == "dut" and resolved["notes"] == "演示清单"

    # 类别顺序按 --class 的出现顺序保留（错序 = 静默错标）
    ordered = tmp_path / "run" / "ordered.json"
    code = main(["amc-iq-manifest", str(model), str(ordered), "--samples", "256",
                 "--class", "8psk", "--class", "bpsk", "--default-offset-hz", "1000",
                 "--default-bandwidth-hz", "2000", "--dataset", "TorchSig 转写"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["output"]["classes"] == ["8psk", "bpsk"]
    assert summary["class_set"] == CLASS_SET_CUSTOM and summary["samples"] == 256
    resolved, _ = read_iq_manifest(ordered)
    assert resolved["preprocess"]["default_offset_hz"] == 1000.0
    assert resolved["training"]["dataset"] == "TorchSig 转写"


def test_cli_amc_iq_classify_dispatch(tmp_path, capsys, tiny_model):
    from signal_analysis.cli import main

    manifest_path, _, _, _ = tiny_model
    workspace = tmp_path / "ws"
    spec = tmp_path / "scene.json"
    spec.write_text(json.dumps({"sample_rate": RATE, "duration": 0.08, "seed": 4,
                                "noise": {"enabled": True, "bandwidth": RATE, "snr_db": 25},
                                "signals": [{"mode": "qam16", "offset": 40_000.0,
                                             "bandwidth": 30_000.0, "power_dbfs": -6.0}]}),
                    encoding="utf-8")
    assert main(["--workspace", str(workspace), "generate", str(spec)]) == 0
    capsys.readouterr()
    asset_id = Workspace(workspace).list_assets()[0]["id"]
    code = main(["--workspace", str(workspace), "amc-iq-classify", asset_id,
                 "--model", str(manifest_path), "--bandwidth-hz", "30000",
                 "--offset-hz", "40000"])
    assert code == 0
    run = json.loads(capsys.readouterr().out)
    assert run["kind"] == "amc_iq_classify"  # CLI 的连字符子命令映射到下划线动作
    assert run["contract"] == IQ_RESULT_CONTRACT
    assert run["waveform"]["samples"] == WINDOW
    assert run["truth"]["class"] == "qam16"
    assert any("待确认" in item for item in run["pending"])


def test_cli_amc_iq_classify_reports_errors(tmp_path, capsys, tiny_model):
    from signal_analysis.cli import main

    manifest_path, _, _, _ = tiny_model
    workspace = tmp_path / "ws"
    spec = tmp_path / "scene.json"
    spec.write_text(json.dumps({"sample_rate": RATE, "duration": 0.04, "seed": 1,
                                "signals": [{"mode": "qpsk", "offset": 40_000.0,
                                              "bandwidth": 30_000.0, "power_dbfs": -6.0}]}),
                    encoding="utf-8")
    assert main(["--workspace", str(workspace), "generate", str(spec)]) == 0
    capsys.readouterr()
    asset_id = Workspace(workspace).list_assets()[0]["id"]
    # 分析带宽超过采样率：必须报错退出，而不是给出一个看起来正常的结论
    code = main(["--workspace", str(workspace), "amc-iq-classify", asset_id,
                 "--model", str(manifest_path), "--bandwidth-hz", "1000000"])
    assert code == 1
    assert "错误" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# GUI 分流
# ---------------------------------------------------------------------------


def test_gui_routes_iq_manifest_to_the_iq_action(tmp_path, tiny_model):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets
    from signal_analysis.gui import MainWindow

    manifest_path, _, _, _ = tiny_model
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path / "ws")
    try:
        assert window._use_iq_branch() is False
        window.amc_model.setText(str(manifest_path))
        assert window._use_iq_branch() is True
        assert window._amc_model_contract() == IQ_WAVEFORM_CONTRACT
        # 非 JSON / 非法 JSON 一律按"非 IQ 清单"处理，真正的报错留给识别入口
        window.amc_model.setText("")
        assert window._use_iq_branch() is False
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        window.amc_model.setText(str(broken))
        assert window._amc_model_contract() is None and window._use_iq_branch() is False
    finally:
        window.close()
        app.processEvents()


def test_gui_renders_iq_result_without_fake_feature_columns(tmp_path, tiny_model):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets
    from signal_analysis.gui import MainWindow

    manifest_path, _, _, _ = tiny_model
    workspace = tmp_path / "ws"
    asset = _generate(workspace, [{"mode": "qpsk", "offset": 40_000.0, "bandwidth": 30_000.0,
                                  "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_iq_classify",
                   "asset_id": asset["id"], "model": str(manifest_path),
                   "config": {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0}})
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    try:
        window.display_result(run)
        assert window.tabs.currentIndex() == 3
        headers = [window.amc_table.horizontalHeaderItem(column).text()
                   for column in range(window.amc_table.columnCount())]
        assert headers == ["输入口径", "取值"]
        first_column = [window.amc_table.item(row, 0).text()
                        for row in range(window.amc_table.rowCount())]
        assert "窗口采样点" in first_column and "归一化" in first_column
        assert "特征" not in first_column  # 这条通路没有 34 维特征可列
        summary = window.amc_summary.toPlainText()
        assert IQ_RESULT_CONTRACT in summary
        assert "没有传统启发式基线" in summary
        assert "生成器真值" in summary
        compare = window.compare_summary.toPlainText()
        assert "原始 IQ" in compare and "待确认项" in compare
        assert "合格门限尚未确认" in compare
    finally:
        window.close()
        app.processEvents()
