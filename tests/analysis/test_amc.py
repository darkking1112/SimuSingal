"""A09 六类调制识别（P4）：特征契约、线性模型、ONNX 清单、服务/CLI 与训练工具链。

与 ``test_ml_detect.py``（P3 AI 检测）同一套思路：

* 特征向量是**训练与推理之间的唯一契约**（``amc_feature_vector_v1``），这里既检查
  顺序/维度/有限性，也检查同一段数据重复提取逐位一致；
* 线性模型是确定性基线，模型文件与 ONNX 清单都要做**篡改检测**（特征序、类别、
  逐样本摘要），否则训练/推理口径分叉时不会有任何提示；
* 结果必须带"识别准确率合格门限待确认"的说明，且六类之外的样式按"不适用"计数，
  **不丢弃样本**；
* ``training/`` 下的脚本不在包里（不能被 import），按路径加载后跑一遍小数据集，
  覆盖"数据集 → 训练 → 验收"全链路。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# GUI 用例在无显示环境下运行：必须在导入 PySide6 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.ml import (  # noqa: E402
    AMC_CLASSES,
    AMC_FEATURE_CONTRACT,
    AMC_FEATURES,
    AMC_MODEL_CONTRACT,
    AMC_ONNX_CONTRACT,
    AMC_RESULT_CONTRACT,
    CLASS_LABELS,
    ModelError,
    amc_classify,
    evaluate_model,
    extract_features,
    feature_vector,
    fit_model,
    load_default_model,
    load_model,
    mode_to_class,
    predict,
    read_amc_manifest,
    save_model,
    write_amc_manifest,
)
from signal_analysis.ml.runtime import runtime_version  # noqa: E402
from signal_analysis.storage import Workspace  # noqa: E402
from signal_analysis.tasks import run_job  # noqa: E402

RATE = 200_000.0
#: 每类一个可直接生成的场景（真值频带取自生成器汇总，模拟检测器给出的频带估计）
MODES = ("fm", "ssb", "ask2", "qpsk", "qam16", "qam64")


def scene(mode, snr_db=25.0, bandwidth=30_000.0, offset=40_000.0, seed=5, duration=0.08):
    """生成单信号场景，返回 ``(samples, 生成器给出的频带信息)``。"""
    spec = {"mode": mode, "offset": offset, "bandwidth": bandwidth, "power_dbfs": -6.0}
    if mode == "ssb":
        spec["side"] = "usb"
    noise = {"enabled": True, "bandwidth": RATE, "snr_db": snr_db}
    samples, summary = generate_iq(RATE, duration, [spec], noise=noise, seed=seed)
    return samples, summary["signals"][0]


def features_of(mode, **kwargs):
    samples, band = scene(mode, **kwargs)
    return extract_features(samples, RATE, band["offset"], band["bandwidth_actual"])


def _load(name):
    """按路径加载 training/ 下的脚本（它们不在包内，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("build_amc_dataset")


@pytest.fixture(scope="module")
def trainer():
    return _load("train_amc")


@pytest.fixture(scope="module")
def verifier():
    return _load("verify_amc")


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory, builder):
    """小数据集：每类 40 条、带内信噪比 15～30 dB（约 4 秒生成）。"""
    root = tmp_path_factory.mktemp("amc-dataset")
    code = builder.main(["--output", str(root), "--per-class", "40", "--seed", "5",
                         "--snr-range", "15,30", "--duration-range", "0.04,0.08"])
    assert code == 0
    card = json.loads((root / "amc_dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (root / "features.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return root, card, records


# ------------------------------------------------------------------ 特征契约

def test_feature_contract_is_wire_stable():
    assert len(AMC_FEATURES) == 34
    assert len(set(AMC_FEATURES)) == 34
    assert AMC_FEATURES[:3] == ("env_cv", "env_kurtosis", "env_gamma_max_db")
    assert AMC_FEATURES[-1] == "snr_estimate_db"
    assert AMC_CLASSES == ("fm", "ssb", "ask2", "qpsk", "qam16", "qam64")
    assert sorted(CLASS_LABELS) == sorted(AMC_CLASSES)
    assert AMC_FEATURE_CONTRACT == "amc_feature_vector_v1"
    assert {AMC_MODEL_CONTRACT, AMC_ONNX_CONTRACT, AMC_RESULT_CONTRACT} == {
        "amc_model_v1", "amc_feature_vector_v1", "amc_classify_v1"}


def test_mode_to_class_only_covers_the_six_classes():
    for mode in MODES:
        assert mode_to_class(mode) == mode
    # 跳频是"一段会话一个实例"的检测对象，am 只在数据库示例里出现，都不在字典内
    for mode in ("am", "fh_rc", "fh_video", "noise", ""):
        assert mode_to_class(mode) is None


@pytest.mark.parametrize("mode", MODES)
def test_extract_features_is_deterministic_and_json_safe(mode):
    samples, band = scene(mode)
    first, info = extract_features(samples, RATE, band["offset"], band["bandwidth_actual"])
    second, _ = extract_features(samples, RATE, band["offset"], band["bandwidth_actual"])
    assert first == second
    assert list(first) == list(AMC_FEATURES)
    assert len(feature_vector(first)) == len(AMC_FEATURES)
    assert np.isfinite(feature_vector(first)).all()
    assert all(isinstance(value, float) for value in first.values())
    json.dumps({"features": first, "info": info}, allow_nan=False)
    assert info["center_hz"] == pytest.approx(band["offset"])
    assert info["sample_count"] == samples.size
    assert info["analysis_samples"] <= info["sample_count"]
    assert info["decimation"] >= 1
    assert info["peak_samples"] > 0
    assert info["snr_estimate_db"] is None or info["snr_estimate_db"] > 0.0


def test_extract_features_rejects_invalid_input():
    samples, band = scene("qpsk")
    with pytest.raises(ValueError, match="采样率"):
        extract_features(samples, 0.0)
    with pytest.raises(ValueError, match="占用带宽"):
        extract_features(samples, RATE, 0.0, RATE * 2)
    with pytest.raises(ValueError, match="中心频率"):
        extract_features(samples, RATE, RATE * 2.0)
    with pytest.raises(ValueError, match="样本过少"):
        extract_features(samples[:32], RATE, band["offset"], band["bandwidth_actual"])
    with pytest.raises(ValueError, match="功率为零"):
        extract_features(np.zeros(4096, dtype=np.complex64), RATE, 0.0, 20_000.0)
    with pytest.raises(ValueError, match="特征缺少字段"):
        feature_vector({"env_cv": 1.0})
    with pytest.raises(ValueError, match="长度"):
        feature_vector([0.0] * 3)


def test_features_separate_styles_by_physics():
    """包络与高阶累积量的量级关系——它们正是六类可分的物理依据。"""
    fm, _ = features_of("fm")
    ask2, _ = features_of("ask2")
    qam16, _ = features_of("qam16")
    qam64, _ = features_of("qam64")
    # FM 恒包络；2ASK 有 0 符号，包络起伏最大
    assert fm["env_cv"] < 0.2 < ask2["env_cv"]
    assert fm["amp_clusters"] == 1 and ask2["amp_clusters"] >= 2
    # 幅度聚类数随阶数增长；|C63| 随阶数下降（16QAM 与 64QAM 的主要区分量）
    assert ask2["amp_clusters"] <= qam64["amp_clusters"] + 1
    assert qam16["c63_mag"] > qam64["c63_mag"]
    assert qam16["snr_estimate_db"] == pytest.approx(25.0, abs=4.0)


def test_default_model_predicts_the_trained_classes():
    model = load_default_model()
    assert model["contract"] == AMC_MODEL_CONTRACT
    assert model["id"] == "amc-linear-default" and model["source"] == "builtin"
    assert list(model["features"]) == list(AMC_FEATURES)
    assert model["training"]["samples"] > 0
    for mode in MODES:
        features, _ = features_of(mode)
        outcome = predict(model, features)
        assert outcome["label"] in AMC_CLASSES
        assert outcome["label_text"] == CLASS_LABELS[outcome["label"]]
        assert sum(outcome["scores"].values()) == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= outcome["confidence"] <= 1.0
        assert outcome["nearest_centroid"] in AMC_CLASSES
        assert set(outcome["logits"]) == set(AMC_CLASSES)


def test_predict_rejects_tampered_model(tmp_path):
    """模型文件里的特征序/参数被改写时必须早期报错，而不是静默给出错误的概率。"""
    def tampered(name, mutate):
        payload = dict(load_default_model())
        payload.pop("model_path", None)
        mutate(payload)
        path = tmp_path / name
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        return path

    reversed_path = tampered("reversed.json",
                             lambda model: model.update(features=list(reversed(AMC_FEATURES))))
    with pytest.raises(ModelError, match="特征顺序"):
        load_model(reversed_path)
    incomplete_path = tampered("incomplete.json", lambda model: model.pop("weights"))
    with pytest.raises(ModelError, match="参数不完整"):
        load_model(incomplete_path)
    cold_path = tampered("temperature.json", lambda model: model.update(temperature=0.0))
    with pytest.raises(ModelError, match="温度"):
        load_model(cold_path)
    # 写盘前先自校验：非法模型不会落盘
    with pytest.raises(ModelError, match="温度"):
        save_model({"contract": AMC_MODEL_CONTRACT, "features": list(AMC_FEATURES),
                    "classes": list(AMC_CLASSES), "temperature": 0.0,
                    "standardize": {"mean": [0.0] * len(AMC_FEATURES),
                                    "scale": [1.0] * len(AMC_FEATURES)},
                    "weights": [[0.0] * len(AMC_CLASSES)] * len(AMC_FEATURES),
                    "bias": [0.0] * len(AMC_CLASSES)}, tmp_path / "cold.json")
    assert not (tmp_path / "cold.json").exists()


def test_fit_model_round_trip(small_dataset, tmp_path):
    _, _, records = small_dataset
    train = [record for record in records if record["split"] == "train"]
    model = fit_model(train, provenance={"script": "tests"})
    assert model["contract"] == AMC_MODEL_CONTRACT
    assert model["training"]["samples"] == len(train)
    assert model["temperature"] > 0.0
    assert set(model["centroids"]) == set(AMC_CLASSES)
    path = save_model(model, tmp_path / "amc.json")
    loaded = load_model(path)
    assert loaded["model_path"] == str(path)
    sample = records[0]["features"]
    assert predict(loaded, sample) == predict(model, sample)
    metrics = evaluate_model(loaded, records)
    assert metrics["total"] == len(records)
    assert metrics["accuracy"] >= 0.8
    assert {row["label"] for row in metrics["per_class"]} == set(AMC_CLASSES)
    assert set(metrics["per_snr"]) and all(
        value is None or 0.0 <= value <= 1.0 for value in metrics["per_snr"].values())


def test_fit_model_rejects_labels_outside_dictionary(small_dataset):
    _, _, records = small_dataset
    with pytest.raises(ValueError, match="A09 六类"):
        fit_model([{"features": records[0]["features"], "label": "am"}])
    with pytest.raises(ValueError, match="训练记录为空"):
        fit_model([])


# ------------------------------------------------------------------ 统一入口

def test_amc_classify_result_contract():
    samples, band = scene("qpsk")
    result = amc_classify(samples, RATE, {"offset_hz": band["offset"],
                                          "bandwidth_hz": band["bandwidth_actual"]})
    assert set(result) == {"contract", "algorithm", "classes", "labels", "features",
                           "feature_contract", "band", "snr_estimate_db", "prediction",
                           "baseline", "model", "pending"}
    json.dumps(result, allow_nan=False)
    assert result["contract"] == AMC_RESULT_CONTRACT
    assert result["algorithm"] == "amc_linear_v1"
    assert result["feature_contract"] == AMC_FEATURE_CONTRACT
    assert result["classes"] == list(AMC_CLASSES) and result["labels"] == CLASS_LABELS
    assert result["model"]["source"] == "builtin"
    assert result["model"]["sha256"] and len(result["model"]["sha256"]) == 64
    assert result["baseline"]["algorithm"] == "heuristic_digital_analog_v1"
    assert result["baseline"]["classification"] in ("digital", "analog", None)
    # 门限待确认与"多信号需先切分"必须随结果一起给出，不能只在文档里
    assert len(result["pending"]) == 2
    assert any("待确认" in item for item in result["pending"])
    assert "验收口径" in result["prediction"]["snr_note"]
    assert result["prediction"]["label"] == "qpsk"
    assert result["prediction"]["reliable"] is True
    assert result["prediction"]["reason"] is None


def test_amc_classify_flags_low_inband_snr():
    samples, band = scene("qpsk", snr_db=0.0)
    result = amc_classify(samples, RATE, {"offset_hz": band["offset"],
                                          "bandwidth_hz": band["bandwidth_actual"]})
    prediction = result["prediction"]
    assert result["snr_estimate_db"] is not None and result["snr_estimate_db"] < 5.0
    assert prediction["reliable"] is False
    assert "仅供参考" in prediction["reason"]
    # 结果照常给出（不因低信噪比而丢弃）
    assert prediction["label"] in AMC_CLASSES
    assert sum(prediction["scores"].values()) == pytest.approx(1.0, abs=1e-6)


def test_amc_classify_without_bandwidth_reports_no_snr_reference():
    """省略带宽时占用带覆盖整个采样带宽，没有噪声参考区，不做低信噪比标注。"""
    samples, _ = scene("fm")
    result = amc_classify(samples, RATE, None)
    assert result["band"]["bandwidth_hz"] == RATE
    assert result["snr_estimate_db"] is None
    assert result["prediction"]["reliable"] is True
    assert "无法估计" in result["prediction"]["reason"]


def test_amc_classify_validates_config_and_model(tmp_path):
    samples, band = scene("qpsk")
    with pytest.raises(ValueError, match="不支持以下字段"):
        amc_classify(samples, RATE, {"offset_hz": 0.0, "nfft": 256})
    with pytest.raises(ValueError, match="AMC 配置应为字典"):
        amc_classify(samples, RATE, ["offset_hz"])
    with pytest.raises(ValueError, match="占用带宽"):
        amc_classify(samples, RATE, {"bandwidth_hz": RATE * 2})
    with pytest.raises(ModelError, match="模型文件不存在"):
        amc_classify(samples, RATE, None, model=str(tmp_path / "missing.json"))
    # 显式指定模型文件：来源标为 file，并带上摘要
    path = save_model(load_default_model(), tmp_path / "explicit.json")
    result = amc_classify(samples, RATE, None, model=str(path))
    assert result["model"]["source"] == "file"
    assert result["model"]["model_path"] == str(path)


@pytest.mark.gui
def test_gui_amc_tab_renders_result(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    from signal_analysis.gui import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    workspace = tmp_path / "ws"
    window = MainWindow(workspace)
    window.show()
    try:
        assert window.tabs.count() == 6
        labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert labels[:5] == ["数据分析", "IQ 信号生成", "信号检测", "调制识别", "算法对比"]
        assert window.amc_button.text() and window.amc_from_detect.text()
        assert "内置" in window.amc_model_status.text()  # 内置模型随包分发时应给出可用的提示
        # 无检测结果时，“取用检测结果频带”给出明确提示而不是静默无动作
        window.use_detected_band()
        assert "信号检测" in window.status.text()
        asset = run_job({"workspace": str(workspace), "action": "generate",
                         "sample_rate": RATE, "duration": 0.08, "seed": 5,
                         "signals": [{"mode": "qpsk", "offset": 40_000.0,
                                      "bandwidth": 30_000.0, "power_dbfs": -6.0}],
                         "noise": {"enabled": True, "bandwidth": RATE, "snr_db": 25.0}})
        result = run_job({"workspace": str(workspace), "action": "amc_classify",
                          "asset_id": asset["id"],
                          "config": {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0}})
        window.display_result(result)
        assert window.tabs.currentIndex() == 3
        summary = window.amc_summary.toPlainText()
        assert "QPSK" in summary and "amc-linear-default" in summary
        assert "命中" in summary  # 生成数据带真值：命中/未命中必须显示
        assert "待确认" in summary  # 验收门限的待确认项必须显示给用户
        assert window.amc_table.rowCount() == len(AMC_FEATURES)
        assert window.amc_table.item(0, 0).text() == AMC_FEATURES[0]
        # “算法对比”页同步记录识别结果与真值命中（不切页，避免打断当前视图）
        assert window.tabs.currentIndex() == 3
        window.compare_from_detect()
        assert window.tabs.currentIndex() == 4
        table = window.compare_table
        assert table.rowCount() == 8  # 6 项预测字段 + 真值类别/命中两条
        object_column = {table.item(row, 1).text() for row in range(table.rowCount())}
        assert any(text.startswith("调制识别") for text in object_column)
        assert "生成器真值" in object_column
        compare = window.compare_summary.toPlainText()
        assert "amc-linear-default" in compare and "待确认项" in compare
        assert "识别准确率的合格门限尚未确认" in compare
    finally:
        window.close()
        app.processEvents()


# ------------------------------------------------------------------ ONNX 清单

def _placeholder_onnx(directory, name="classifier.onnx"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"not-a-real-onnx")
    return path


def test_manifest_round_trip_and_digest(tmp_path):
    from common.storage import file_digest

    model = _placeholder_onnx(tmp_path / "run")
    manifest_path = tmp_path / "run" / "amc.json"
    manifest, library = write_amc_manifest(
        manifest_path, model, identifier="dut", version="0.3.0", opset=17,
        standardize={"mean": [0.0] * len(AMC_FEATURES), "scale": [1.0] * len(AMC_FEATURES)},
        training={"script": "training/train_amc.py", "arch": "transformer"},
        notes="测试清单")
    assert manifest["id"] == "dut" and manifest["contract"] == AMC_ONNX_CONTRACT
    assert manifest["input"] == {"name": "features", "size": len(AMC_FEATURES)}
    assert manifest["output"]["classes"] == list(AMC_CLASSES)
    assert manifest["sha256"] == file_digest(model)
    assert library == model.resolve()
    resolved, again = read_amc_manifest(manifest_path)
    assert resolved["training"]["arch"] == "transformer" and again == library
    # 逐样本摘要：模型被改写后必须报错，而不是照常推理
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ModelError, match="摘要与清单不符"):
        read_amc_manifest(manifest_path)


def test_manifest_rejects_bad_inputs(tmp_path):
    model = _placeholder_onnx(tmp_path / "run")
    other = _placeholder_onnx(tmp_path / "elsewhere")
    with pytest.raises(ModelError, match="必须位于清单目录内"):
        write_amc_manifest(tmp_path / "run" / "amc.json", other)
    wrong_suffix = tmp_path / "run" / "classifier.bin"
    wrong_suffix.write_bytes(b"x")
    with pytest.raises(ModelError, match=r"\.onnx"):
        write_amc_manifest(tmp_path / "run" / "bad.json", wrong_suffix)
    with pytest.raises(ModelError, match="类别"):
        write_amc_manifest(tmp_path / "run" / "bad.json", model, classes=["fm", "ssb"])
    with pytest.raises(ModelError, match="特征顺序"):
        write_amc_manifest(tmp_path / "run" / "bad.json", model,
                           features=list(reversed(AMC_FEATURES)))
    path = tmp_path / "run" / "amc.json"
    write_amc_manifest(path, model)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["features"] = list(reversed(AMC_FEATURES))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelError, match="特征顺序"):
        read_amc_manifest(path)
    payload["features"] = list(AMC_FEATURES)
    payload["contract"] = "other_contract"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelError, match="契约"):
        read_amc_manifest(path)
    payload["contract"] = AMC_ONNX_CONTRACT
    payload["library"] = "/tmp/absolute.onnx"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelError, match="相对路径"):
        read_amc_manifest(path)


def test_amc_classify_rejects_onnx_manifest_without_runtime(tmp_path):
    """记录缺 onnxruntime 时的行为：给出安装提示，而不是栈回溯。"""
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime")
    from signal_analysis.ml import RuntimeUnavailable

    model = _placeholder_onnx(tmp_path)
    manifest_path = tmp_path / "amc.json"
    write_amc_manifest(manifest_path, model)
    samples, band = scene("qpsk")
    with pytest.raises(RuntimeUnavailable, match="onnxruntime"):
        amc_classify(samples, RATE, {"offset_hz": band["offset"],
                                     "bandwidth_hz": band["bandwidth_actual"]},
                     model=str(manifest_path))


# ------------------------------------------------------------------ 服务层与命令行

def _generate(workspace, signals, rate=RATE, duration=0.08, noise_snr=25.0, seed=3):
    return run_job({"workspace": str(workspace), "action": "generate", "sample_rate": rate,
                    "duration": duration, "signals": signals, "seed": seed,
                    "noise": {"enabled": True, "bandwidth": rate, "snr_db": noise_snr}})


def test_service_amc_classify_attaches_truth(tmp_path):
    workspace = tmp_path / "ws"
    asset = _generate(workspace, [{"mode": "qpsk", "offset": 40_000.0,
                                  "bandwidth": 30_000.0, "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_classify",
                   "asset_id": asset["id"],
                   "config": {"offset_hz": 40_000.0, "bandwidth_hz": 30_000.0}})
    assert run["kind"] == "amc_classify"
    assert run["contract"] == AMC_RESULT_CONTRACT
    assert run["summary"]["contract"] == AMC_RESULT_CONTRACT
    assert run["prediction"]["label"] == "qpsk"
    assert run["feature_contract"] == AMC_FEATURE_CONTRACT
    assert len(run["summary"]["features"]) == len(AMC_FEATURES)
    truth = run["truth"]
    assert truth["available"] is True and truth["class"] == "qpsk"
    assert truth["mode"] == "qpsk" and truth["snr_inband_db"] > 0.0
    assert run["truth_hit"] is True
    # 结果落盘并可按 run_id 取回（GUI 的"运行记录"与报表都依赖这一点）
    assert Workspace(workspace).get_run(run["run_id"])["prediction"] == run["prediction"]


def test_service_amc_truth_is_explicitly_not_applicable(tmp_path):
    workspace = tmp_path / "ws"
    unmapped = _generate(workspace, [{"mode": "am", "offset": 40_000.0,
                                     "bandwidth": 20_000.0, "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_classify",
                   "asset_id": unmapped["id"]})
    assert run["truth"]["available"] is False
    assert "不在 A09 六类字典内" in run["truth"]["reason"]
    assert run["truth_hit"] is None
    assert run["prediction"]["label"] in AMC_CLASSES  # 仍给出六类概率，样本不被丢弃

    multi = _generate(workspace, [{"mode": "fm", "offset": 40_000.0, "bandwidth": 20_000.0,
                                   "power_dbfs": -6.0},
                                  {"mode": "ask2", "offset": -60_000.0, "bandwidth": 20_000.0,
                                   "power_dbfs": -6.0}])
    run = run_job({"workspace": str(workspace), "action": "amc_classify", "asset_id": multi["id"]})
    assert run["truth"]["available"] is False and run["truth"]["count"] == 2

    demo = run_job({"workspace": str(workspace), "action": "demo", "count": 4096})
    run = run_job({"workspace": str(workspace), "action": "amc_classify", "asset_id": demo["id"]})
    assert run["truth"]["available"] is False and "没有生成器真值" in run["truth"]["reason"]


def test_cli_amc_classify_and_manifest(tmp_path, capsys):
    from signal_analysis.cli import main

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
    code = main(["--workspace", str(workspace), "amc-classify", asset_id,
                 "--bandwidth-hz", "30000", "--offset-hz", "40000"])
    assert code == 0
    run = json.loads(capsys.readouterr().out)
    assert run["kind"] == "amc_classify"
    assert run["contract"] == AMC_RESULT_CONTRACT
    assert run["feature_contract"] == AMC_FEATURE_CONTRACT
    assert run["prediction"]["label"] == "qam16"
    assert run["model"]["source"] == "builtin"
    assert run["truth_hit"] is True
    assert any("待确认" in item for item in run["pending"])

    # 清单：占位模型文件即可（清单只做结构与摘要校验）
    model = _placeholder_onnx(tmp_path / "run")
    code = main(["amc-manifest", str(model), str(tmp_path / "run" / "amc.json"),
                 "--id", "dut", "--version", "0.4.0", "--notes", "演示清单"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["contract"] == AMC_ONNX_CONTRACT
    assert summary["input"]["size"] == len(AMC_FEATURES)
    assert summary["output"]["classes"] == list(AMC_CLASSES)
    resolved, _ = read_amc_manifest(tmp_path / "run" / "amc.json")
    assert resolved["id"] == "dut" and resolved["version"] == "0.4.0"
    assert resolved["notes"] == "演示清单"


def test_cli_amc_classify_reports_errors(tmp_path, capsys):
    from signal_analysis.cli import main

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
    code = main(["--workspace", str(workspace), "amc-classify", asset_id,
                 "--bandwidth-hz", "1000000"])
    assert code == 1
    assert "错误" in capsys.readouterr().err


# ------------------------------------------------------------------ 训练工具链

def test_amc_dataset_builder_uses_the_inference_feature_extractor(small_dataset):
    root, card, records = small_dataset
    contract = card["contract"]
    assert contract["feature_contract"] == AMC_FEATURE_CONTRACT
    assert contract["features"] == list(AMC_FEATURES)
    assert contract["classes"] == list(AMC_CLASSES)
    # 数据集卡片的字典就是六类本身：am 与跳频样式不在其中（按“不适用”计数）
    assert set(contract["mode_to_class"]) == set(AMC_CLASSES)
    assert contract["mode_to_class"]["qam64"] == "qam64"
    assert contract["feature_count"] == len(AMC_FEATURES)
    assert card["sample_count"] == len(records) == 240
    assert card["splits"]["strategy"] == "stratified_per_class"
    assert card["splits"]["per_class"]["qam64"] == {"train": 32, "val": 8}
    # 分层：每类都同时出现在训练与验证集里（否则指标会完全失真）
    for mode in MODES:
        assert card["splits"]["per_class"][mode] == {"train": 32, "val": 8}
    # 特征就是推理端会拿到的同一种特征：按 scene 重建波形后逐位复现
    seen = set()
    for record in records:
        if record["mode"] in seen:
            continue
        seen.add(record["mode"])
        scene_spec = record["scene"]
        samples, summary = generate_iq(scene_spec["rate_hz"], scene_spec["duration_s"],
                                      list(scene_spec["signals"]), scene_spec["noise"],
                                      scene_spec["seed"])
        window = record["analysis"]
        # 记录里的分析窗口是四舍五入后的备份，与提取时的实际取值一致（误差 < 1 mHz）
        assert window["offset_hz"] == pytest.approx(record["extraction"]["center_hz"], abs=1e-3)
        assert window["bandwidth_hz"] == pytest.approx(record["extraction"]["bandwidth_hz"],
                                                       abs=1e-3)
        features, info = extract_features(samples, scene_spec["rate_hz"],
                                         record["extraction"]["center_hz"],
                                         record["extraction"]["bandwidth_hz"])
        assert features == record["features"]
        assert info == record["extraction"]
        assert summary["signals"][0]["snr_inband_db"] == pytest.approx(
            record["truth"]["snr_inband_db"], abs=1e-3)
    assert seen == set(MODES)


def test_amc_dataset_builder_rejects_invalid_arguments(builder, tmp_path):
    with pytest.raises(SystemExit) as error:
        builder.main(["--output", str(tmp_path / "bad"), "--per-class", "1", "--modes", "qpsk,am"])
    assert "am" in str(error.value)
    with pytest.raises(SystemExit):
        builder.main(["--output", str(tmp_path / "bad"), "--per-class", "1",
                      "--train-fraction", "1.2"])
    with pytest.raises(SystemExit):
        builder.main(["--output", str(tmp_path / "bad"), "--per-class", "1",
                      "--min-bandwidth-ratio", "0.5", "--max-bandwidth-ratio", "0.2"])


def test_trainer_writes_linear_model_with_validation_metrics(trainer, small_dataset, tmp_path):
    root, card, _ = small_dataset
    output = tmp_path / "amc_default.json"
    assert trainer.main(["--data", str(root), "--output", str(output),
                         "--identifier", "dut", "--version", "0.9.0"]) == 0
    model = load_model(output)
    assert model["contract"] == AMC_MODEL_CONTRACT
    assert model["id"] == "dut" and model["version"] == "0.9.0"
    validation = model["training"]["validation"]
    assert model["training"]["val_samples"] == card["splits"]["val"]
    assert validation["accuracy"] >= 0.8
    assert {row["label"] for row in validation["per_class"]} == set(AMC_CLASSES)
    assert validation["per_snr"]
    assert model["training"]["dataset_seed"] == card["seed"]


def test_trainer_rejects_stale_dataset_contract(trainer, small_dataset, tmp_path):
    root, card, records = small_dataset
    stale = tmp_path / "dataset"
    stale.mkdir()
    card = dict(card)
    card["contract"] = dict(card["contract"], features=list(reversed(AMC_FEATURES)))
    (stale / "amc_dataset.json").write_text(json.dumps(card), encoding="utf-8")
    (stale / "features.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="特征清单"):
        trainer.main(["--data", str(stale), "--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit, match="数据集不完整"):
        trainer.main(["--data", str(tmp_path / "missing"), "--output", str(tmp_path / "o.json")])


def test_trainer_transformer_requires_torch(trainer, small_dataset, tmp_path):
    pytest.importorskip("numpy")
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("本机已安装 torch")
    root, _, _ = small_dataset
    with pytest.raises(SystemExit) as error:
        trainer.main(["--data", str(root), "--arch", "transformer",
                      "--output", str(tmp_path / "out.json")])
    assert "pip install .[train]" in str(error.value)


def test_verify_amc_passes_on_generated_dataset(verifier, small_dataset, tmp_path):
    root, _, _ = small_dataset
    report_path = tmp_path / "verify.json"
    code = verifier.main(["--data", str(root), "--json", str(report_path)])
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert all(check["ok"] for check in report["checks"])
    assert report["linear"]["accuracy"] >= 0.8
    assert report["linear"]["val_samples"] == 48
    assert report["linear"]["per_snr"]
    assert report["dataset"]["train"] + report["dataset"]["val"] == report["dataset"]["samples"]
    if runtime_version() is None:
        assert report.get("onnx") is None


def test_verify_amc_fails_loudly_on_broken_dataset(verifier, small_dataset, tmp_path, capsys):
    root, card, records = small_dataset
    broken = tmp_path / "dataset"
    broken.mkdir()
    (broken / "amc_dataset.json").write_text(json.dumps(card), encoding="utf-8")
    records = [dict(record) for record in records]
    broken_index = next(index for index, record in enumerate(records)
                        if record["split"] == "train")
    stripped = dict(records[broken_index]["features"])
    stripped.pop("c63_mag")
    records[broken_index]["features"] = stripped
    (broken / "features.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    assert verifier.main(["--data", str(broken)]) == 1
    assert "[失败]" in capsys.readouterr().out


def test_verify_amc_skips_onnx_without_runtime(verifier, small_dataset, tmp_path):
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime")
    root, _, _ = small_dataset
    model = _placeholder_onnx(tmp_path)
    manifest_path, _ = write_amc_manifest(tmp_path / "amc.json", model)
    # 退出码 2 = 结构检查通过但缺少 onnxruntime（不是"数据/模型有问题"）
    assert verifier.main(["--data", str(root), "--manifest", str(manifest_path)]) == 2
