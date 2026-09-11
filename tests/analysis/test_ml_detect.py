"""AI 检测（P3）：清单校验、时频图构造、输出解码与端到端推理。

本文件不依赖 ``onnxruntime``：推理会话通过注入的假会话（``FakeRunner``）
提供，因此仓库的默认测试环境即可覆盖除"真实 ONNX 模型加载"以外的全部路径；
需要真实模型的检查放在 ``training/verify_onnx.py``（需安装 ``[ml]`` 与
``[train]`` 依赖）。
"""

import json

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.evaluation import evaluate_detections, signal_truth
from signal_analysis.ml import (
    ManifestError,
    RuntimeUnavailable,
    band_to_box,
    box_to_band,
    boxes_to_bands,
    detection_image,
    measure_band,
    ml_detect,
    non_max_suppression,
    parse_model_output,
    read_model_manifest,
    spectral_context,
    write_model_manifest,
)
from signal_analysis.ml.runtime import load_runner, runtime_version
from signal_analysis.storage import Workspace

RATE = 1_000_000.0
SIZE = 1024
NFFT = 512
CONTRACT_INPUT = {"name": "images", "image_size": SIZE, "channels": 1,
                  "spectrogram_nfft": NFFT, "dynamic_range_db": 60.0,
                  "layout": "time_frequency_grayscale_v1"}
DEFAULT_MANIFEST = {"id": "fake", "version": "0.1.0", "labels": ["emitter"],
                    "input": CONTRACT_INPUT,
                    "output": {"name": "detections", "layout": "normalized_boxes_v1"}}


def scene(mode="qpsk", offset=120_000.0, bandwidth=100_000.0, power_dbfs=-8.0,
          snr_db=18.0, duration=0.3, rate=RATE, seed=3, **extra):
    signals = [{"mode": mode, "offset": offset, "bandwidth": bandwidth,
                "power_dbfs": power_dbfs, **extra}]
    data, generation = generate_iq(rate, duration, signals,
                                   noise={"enabled": True, "snr_db": snr_db}, seed=seed)
    return data, rate, duration, generation


def image_context(data, rate=RATE, nfft=NFFT, size=SIZE):
    summary, arrays = spectral_context(data, rate, {"nfft": nfft})
    image, meta = detection_image(arrays, summary, size)
    return summary, arrays, image, meta


def truth_box(meta, generation, index=0):
    truth = signal_truth(generation)[index]
    return band_to_box(meta, truth["f_low_hz"], truth["f_high_hz"],
                       truth["t_start_s"], truth["t_end_s"])


def rows(*boxes):
    """``(box, score)`` 序列 → 已解析的模型输出 ``(N, 6)``（与 decode 约定一致）。"""
    payload = [[*box, score, 0.0] for box, score in boxes]
    return np.asarray(payload, dtype=np.float64).reshape(-1, 6)


class FakeRunner:
    """按图像尺寸返回预设检测框的假会话（与 ModelRunner 接口一致）。"""

    model_name = "fake@0.1.0"
    runtime_version = "test"

    def __init__(self, boxes, manifest=None):
        self.manifest = dict(DEFAULT_MANIFEST if manifest is None else manifest)
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        self.images = []

    def run(self, image):
        self.images.append(np.array(image, dtype=np.float32))
        return self.boxes.reshape(1, -1, 6)


def manifest_file(folder, overrides=None, model_bytes=b"fake-onnx-bytes"):
    """写一个自洽的清单 + 模型文件，返回 ``(清单路径, 模型路径)``。"""
    model = folder / "detector.onnx"
    model.write_bytes(model_bytes)
    from common.storage import file_digest

    payload = {
        "schema_version": 1, "id": "spectrogram-detector", "version": "0.1.0",
        "runtime": "onnxruntime", "runtime_min_version": "1.17",
        "contract": "tf_image_v1", "library": "detector.onnx",
        "sha256": file_digest(model), "opset": 17,
        "input": dict(CONTRACT_INPUT),
        "output": {"name": "detections", "layout": "normalized_boxes_v1"},
        "labels": ["emitter"],
        "training": {"framework": "yolox", "license": "Apache-2.0"},
    }
    if overrides:
        for key, value in overrides.items():
            if value is None:
                payload.pop(key, None)
            else:
                payload[key] = value
    path = folder / "model.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path, model


# --------------------------------------------------------------------------- 清单

def test_manifest_roundtrip(tmp_path):
    path, model = manifest_file(tmp_path)
    manifest, library = read_model_manifest(path)
    assert library == model.resolve()
    assert manifest["id"] == "spectrogram-detector"
    assert manifest["input"]["image_size"] == SIZE
    assert manifest["input"]["spectrogram_nfft"] == NFFT
    assert manifest["output"]["layout"] == "normalized_boxes_v1"
    assert manifest["labels"] == ["emitter"]
    assert manifest["training"]["license"] == "Apache-2.0"


def test_manifest_write_and_read(tmp_path):
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"onnx-bytes")
    payload = write_model_manifest(tmp_path / "model.json", model, identifier="dut",
                                   version="1.2.3", labels=["emitter"], opset=17,
                                   training={"framework": "rt-detr", "license": "Apache-2.0"},
                                   notes="自建数据")
    assert payload["id"] == "dut" and payload["version"] == "1.2.3"
    manifest, library = read_model_manifest(tmp_path / "model.json")
    assert manifest["id"] == "dut" and library.name == "detector.onnx"
    assert manifest["notes"] == "自建数据"
    assert manifest["input"]["dynamic_range_db"] == 60.0


def test_manifest_write_rejects_foreign_model(tmp_path):
    model = tmp_path / "detector.onnx"
    model.write_bytes(b"onnx")
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with pytest.raises(ManifestError, match="清单目录"):
        write_model_manifest(manifests / "model.json", model, identifier="dut", version="1.0")


def test_manifest_write_rejects_non_onnx(tmp_path):
    model = tmp_path / "weights.pt"
    model.write_bytes(b"torch")
    with pytest.raises(ManifestError, match="onnx"):
        write_model_manifest(tmp_path / "model.json", model, identifier="dut", version="1.0")


@pytest.mark.parametrize("overrides, message", [
    ({"schema_version": 2}, "清单版本"),
    ({"contract": "other_v2"}, "输入契约"),
    ({"runtime": "tflite"}, "推理运行时"),
    ({"id": None}, "id"),
    ({"id": "   "}, "id"),
    ({"version": None}, "version"),
    ({"sha256": "abc"}, "十六进制"),
    ({"sha256": None}, "sha256"),
    ({"library": None}, "library"),
    ({"labels": []}, "labels"),
    ({"labels": ["  "]}, "类别名"),
    ({"output": {"name": "detections", "layout": "yolo_v8"}}, "解码方式"),
    ({"input": {**CONTRACT_INPUT, "image_size": 1000}}, "输入尺寸"),
    ({"input": {**CONTRACT_INPUT, "channels": 3}}, "单通道"),
    ({"input": {**CONTRACT_INPUT, "spectrogram_nfft": 0}}, "nfft"),
    ({"input": {**CONTRACT_INPUT, "dynamic_range_db": 200.0}}, "动态范围"),
    ({"input": {**CONTRACT_INPUT, "layout": "channels_first"}}, "排布"),
    ({"input": "images"}, "input 段"),
])
def test_manifest_rejects_bad_fields(tmp_path, overrides, message):
    path, _ = manifest_file(tmp_path, overrides)
    with pytest.raises(ManifestError, match=message):
        read_model_manifest(path)


def test_manifest_detects_tampered_model(tmp_path):
    path, model = manifest_file(tmp_path)
    model.write_bytes(b"other-bytes")
    with pytest.raises(ManifestError, match="摘要与清单不符"):
        read_model_manifest(path)


def test_manifest_rejects_escaping_and_missing_library(tmp_path):
    path, _ = manifest_file(tmp_path, {"library": "../detector.onnx"})
    with pytest.raises(ManifestError):
        read_model_manifest(path)
    path, _ = manifest_file(tmp_path, {"library": "missing.onnx"})
    with pytest.raises(ManifestError, match="模型文件不可读"):
        read_model_manifest(path)
    absolute, _ = manifest_file(tmp_path, {"library": "/etc/hostname"})
    with pytest.raises(ManifestError, match="相对路径"):
        read_model_manifest(absolute)


def test_manifest_rejects_empty_model_and_bad_json(tmp_path):
    empty, _ = manifest_file(tmp_path, model_bytes=b"")
    with pytest.raises(ManifestError, match="模型文件缺失或为空"):
        read_model_manifest(empty)
    broken = tmp_path / "broken.json"
    broken.write_text("{不是 JSON", encoding="utf-8")
    with pytest.raises(ManifestError, match="合法 JSON"):
        read_model_manifest(broken)
    huge = tmp_path / "huge.json"
    huge.write_text(json.dumps({"pad": "x" * (65 * 1024)}), encoding="utf-8")
    with pytest.raises(ManifestError, match="64 KiB"):
        read_model_manifest(huge)
    with pytest.raises(ManifestError, match="不可读"):
        read_model_manifest(tmp_path / "missing.json")


def test_load_runner_requires_runtime(tmp_path):
    path, _ = manifest_file(tmp_path)
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime，无法复现缺依赖场景")
    with pytest.raises(RuntimeUnavailable, match="onnxruntime"):
        load_runner(path)


# ------------------------------------------------------------------------- 时频图

def test_detection_image_shape_and_range():
    data, rate, _, _ = scene()
    summary, arrays, image, meta = image_context(data, rate)
    assert image.shape == (SIZE, SIZE) and image.dtype == np.float32
    assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
    assert float(image.max()) > 0.3  # 信号应明显高出本底
    assert meta["layout"] == "time_frequency_grayscale_v1"
    assert meta["db_ceiling"] - meta["db_floor"] == 60.0
    assert meta["f_low_hz"] == -rate / 2 and meta["f_high_hz"] == rate / 2
    assert meta["t_end_s"] == summary["duration_s"]


def test_detection_image_is_deterministic():
    data, rate, _, _ = scene()
    summary, arrays, first, meta = image_context(data, rate)
    _, _, second, _ = image_context(data, rate)
    assert np.array_equal(first, second)
    with pytest.raises(ValueError, match="2 的幂"):
        detection_image(arrays, summary, 1000)
    with pytest.raises(ValueError, match="动态范围"):
        detection_image(arrays, summary, SIZE, dynamic_range_db=5.0)
    with pytest.raises(ValueError, match="2 的幂"):
        detection_image(arrays, summary, 32)


def test_image_rows_start_at_positive_frequency():
    """行 0 对应 +fs/2：正频偏移的信号应出现在图像上半部分。"""
    data, rate, _, _ = scene(offset=250_000.0, bandwidth=60_000.0, snr_db=25.0)
    _, _, image, meta = image_context(data, rate)
    profile = image.mean(axis=1)
    peak_row = int(np.argmax(profile))
    expected = (rate / 2 - 250_000.0) / rate * SIZE
    assert abs(peak_row - expected) < 0.05 * SIZE
    # 反归一化后的取值始终落在清单声明的动态范围内
    from signal_analysis.ml import image_db

    db = image_db(image, meta)
    assert float(db.min()) >= meta["db_floor"] - 1e-6
    assert float(db.max()) <= meta["db_ceiling"] + 1e-6
    # 全图最强点落在信号带内（±40 行 ≈ 39 kHz，覆盖 60 kHz 带宽）
    top, bottom = int(expected) - 40, int(expected) + 41
    assert float(db[top:bottom].max()) == pytest.approx(float(db.max()))


def test_box_band_roundtrip():
    data, rate, duration, _ = scene()
    _, _, _, meta = image_context(data, rate)
    box = band_to_box(meta, 70_000.0, 170_000.0, 0.05, 0.25)
    band = box_to_band(meta, *box)
    assert band["f_low_hz"] == pytest.approx(70_000.0, abs=1.0)
    assert band["f_high_hz"] == pytest.approx(170_000.0, abs=1.0)
    assert band["t_start_s"] == pytest.approx(0.05, abs=1e-6)
    assert band["t_end_s"] == pytest.approx(0.25, abs=1e-6)
    assert 0.0 <= box[0] <= 1.0 and 0.0 <= box[1] <= 1.0 - box[3]
    assert box[2] == pytest.approx(0.2 / duration, rel=1e-6)
    assert box[3] == pytest.approx(100_000.0 / rate, rel=1e-6)
    # 越界输入被裁剪而不是外推
    clamped = box_to_band(meta, 1.4, -0.3, 0.5, 0.5)
    assert clamped["t_end_s"] == pytest.approx(duration)
    assert clamped["f_high_hz"] == pytest.approx(rate / 2)
    assert clamped["t_start_s"] >= 0.0 and clamped["f_low_hz"] >= -rate / 2


def test_measure_band_matches_generator_truth():
    data, rate, _, generation = scene(mode="qam16", snr_db=20.0, power_dbfs=-10.0)
    summary, arrays, _, meta = image_context(data, rate)
    truth = signal_truth(generation)[0]
    measured = measure_band(arrays, summary, {"f_low_hz": truth["f_low_hz"],
                                              "f_high_hz": truth["f_high_hz"],
                                              "t_start_s": 0.0, "t_end_s": summary["duration_s"]})
    assert measured["power_dbfs"] == pytest.approx(truth["power_dbfs"], abs=1.5)
    assert measured["snr_db"] == pytest.approx(truth["snr_inband_db"], abs=1.5)
    assert abs(measured["centroid_hz"] - truth["center_hz"]) < 5_000.0
    assert measured["bin_count"] > 10


# --------------------------------------------------------------------------- 解码

@pytest.mark.parametrize("shape", ["rows", "batch", "channels_first"])
def test_parse_model_output_shapes(shape):
    payload = np.arange(12, dtype=np.float64).reshape(2, 6)
    array = {"rows": payload, "batch": payload.reshape(1, 2, 6),
             "channels_first": payload.T.reshape(1, 6, 2)}[shape]
    rows_, shape_ = parse_model_output(array)
    assert rows_.shape == (2, 6)
    assert list(shape_) == list(np.shape(array))


@pytest.mark.parametrize("bad", [np.zeros((2, 5)), np.zeros((2, 7)), np.zeros((2, 2, 6)),
                                 np.array([[np.nan] * 6])])
def test_parse_model_output_rejects_bad_arrays(bad):
    with pytest.raises(ValueError, match="模型"):
        parse_model_output(bad)


def test_non_max_suppression_removes_duplicates():
    rows_ = np.array([[0.5, 0.5, 0.2, 0.2, 0.9, 0.0],
                      [0.51, 0.5, 0.2, 0.2, 0.5, 0.0],
                      [0.9, 0.9, 0.05, 0.05, 0.8, 0.0]])
    kept = non_max_suppression(rows_, 0.5)
    assert len(kept) == 2
    assert [round(float(item[4]), 2) for item in kept] == [0.9, 0.8]
    # 阈值 0 只保留互不重叠的框，阈值 1 相当于关闭去重
    assert len(non_max_suppression(rows_, 0.0)) == 2
    assert len(non_max_suppression(rows_, 1.0)) == 3
    with pytest.raises(ValueError, match="IoU"):
        non_max_suppression(rows_, 1.5)


def test_boxes_to_bands_maps_and_filters():
    data, rate, _, _ = scene()
    _, _, _, meta = image_context(data, rate)
    box = band_to_box(meta, 70_000.0, 170_000.0, 0.02, 0.22)
    candidates = boxes_to_bands(rows((box, 0.9)), meta, score_threshold=0.25)
    assert len(candidates) == 1
    band = candidates[0]["band"]
    assert band["f_low_hz"] == pytest.approx(70_000.0, abs=1.0)
    assert band["f_high_hz"] == pytest.approx(170_000.0, abs=1.0)
    assert candidates[0]["label"] == "emitter"
    assert candidates[0]["confidence"] == pytest.approx(0.9)
    # 置信度阈值
    assert boxes_to_bands(rows((box, 0.2)), meta, score_threshold=0.25) == []
    # 最小带宽/时长
    assert boxes_to_bands(rows((box, 0.9)), meta, min_bandwidth_hz=200_000.0) == []
    assert boxes_to_bands(rows((box, 0.9)), meta, min_duration_s=1.0) == []
    # 数量上限与非法参数
    two = rows((box, 0.9), (band_to_box(meta, -100_000.0, -20_000.0, 0.0, 0.1), 0.8))
    assert len(boxes_to_bands(two, meta, max_detections=1)) == 1
    with pytest.raises(ValueError, match="置信度阈值"):
        boxes_to_bands(two, meta, score_threshold=1.0)
    with pytest.raises(ValueError, match="最多候选数"):
        boxes_to_bands(two, meta, max_detections=0)


# ------------------------------------------------------------------ 端到端（假会话）

def test_ml_detect_matches_truth_and_reports_contract():
    data, rate, _, generation = scene(mode="qam16", snr_db=20.0, power_dbfs=-10.0)
    _, _, _, meta = image_context(data, rate)
    box = truth_box(meta, generation)
    runner = FakeRunner(rows((box, 0.93)))
    summary, arrays = ml_detect(data, rate, {}, runner=runner)
    assert summary["contract"] == "detect_result_v1"
    assert summary["algorithm"] == "ml_detect:fake@0.1.0"
    assert summary["snr_definition"] == "inband_snr_v1"
    assert summary["model"]["id"] == "fake"
    assert summary["image"]["size"] == SIZE
    assert len(summary["detections"]) == 1
    detection = summary["detections"][0]
    assert detection["method"] == "ml"
    assert detection["label"] == "emitter"
    assert detection["confidence"] == pytest.approx(0.93)
    assert detection["session_id"] is None and detection["hopping"] is False
    truth = signal_truth(generation)[0]
    assert detection["center_hz"] == pytest.approx(truth["center_hz"], abs=2_000.0)
    assert detection["bandwidth_hz"] == pytest.approx(truth["bandwidth_hz"], rel=0.1)
    assert detection["snr_db"] == pytest.approx(truth["snr_inband_db"], abs=2.0)
    assert detection["power_dbfs"] == pytest.approx(truth["power_dbfs"], abs=2.0)
    metrics = evaluate_detections(signal_truth(generation), summary["detections"])
    assert metrics["matched"] == 1 and metrics["missed"] == 0 and metrics["false_alarm"] == 0
    assert metrics["center_mae_hz"] < summary["freq_resolution_hz"] + 2_000.0
    assert runner.images[0].shape == (SIZE, SIZE)
    # 冻结契约字段齐全，且可安全序列化（无 inf/nan）
    for key in ("id", "method", "center_hz", "bandwidth_hz", "f_low_hz", "f_high_hz",
                "t_start_s", "t_end_s", "power_dbfs", "snr_db", "session_id", "confidence"):
        assert key in detection
    assert json.dumps(summary, ensure_ascii=False, allow_nan=False)


def test_ml_detect_baseline_included_and_toggleable():
    data, rate, _, generation = scene()
    _, _, _, meta = image_context(data, rate)
    box = truth_box(meta, generation)
    summary, arrays = ml_detect(data, rate, {}, runner=FakeRunner(rows((box, 0.9))))
    baseline = summary["baseline"]
    assert baseline["algorithm"] == "energy_detect_v1"
    assert len(baseline["detections"]) == 1
    assert arrays["baseline_detection_boxes"].shape == (1, 4)
    baseline_metrics = evaluate_detections(signal_truth(generation), baseline["detections"])
    assert baseline_metrics["matched"] == 1 and baseline_metrics["false_alarm"] == 0
    without, arrays2 = ml_detect(data, rate, {}, runner=FakeRunner(rows((box, 0.9))),
                                 with_baseline=False)
    assert "baseline" not in without
    assert arrays2["baseline_detection_boxes"].shape == (0, 4)


def test_ml_detect_honours_score_threshold_and_limits():
    data, rate, _, generation = scene()
    _, _, _, meta = image_context(data, rate)
    box = truth_box(meta, generation)
    noise_box = band_to_box(meta, -400_000.0, -380_000.0, 0.0, 0.1)
    runner = FakeRunner(rows((box, 0.9), (noise_box, 0.3)))
    summary, _ = ml_detect(data, rate, {"score_threshold": 0.5}, runner=runner)
    assert len(summary["detections"]) == 1
    assert summary["raw_boxes"]["candidates"] == 1
    summary, _ = ml_detect(data, rate, {"score_threshold": 0.2, "max_detections": 1},
                           runner=runner)
    assert len(summary["detections"]) == 1
    summary, _ = ml_detect(data, rate, {"iou_threshold": 1.0}, runner=runner)
    assert len(summary["detections"]) == 2  # 关闭去重后低置信度框保留
    summary, _ = ml_detect(data, rate,
                           {"min_bandwidth_hz": 200_000.0}, runner=runner)
    assert summary["detections"] == []


def test_ml_detect_merges_hopping_session():
    """同一次跳频会话的各个信道框应合并为一个实例（与真值口径一致）。"""
    hop_bandwidth = 40_000.0
    # 跳速取 10 Hz：每个信道驻留 100 ms（≈100 帧），跳变边界的重叠帧
    # 对会话判定的影响可忽略；100 Hz 跳速时驻留仅 10 帧，处在
    # ``_SESSION_OVERLAP_RATIO`` 判据的临界点上（1 ms 帧网格的分辨极限）
    data, rate, duration, generation = scene(
        mode="fh_rc", offset=0.0, bandwidth=200_000.0, snr_db=25.0, duration=0.5,
        hop_rate=10.0, hop_points=[-80_000.0, 0.0, 80_000.0],
        hop_bandwidth=hop_bandwidth)
    visited = sorted(set(generation["signals"][0]["hop_points"]))
    assert len(visited) >= 2
    _, _, _, meta = image_context(data, rate)
    boxes = [(band_to_box(meta, channel - hop_bandwidth / 2, channel + hop_bandwidth / 2,
                          0.0, duration), 0.8) for channel in visited]
    summary, _ = ml_detect(data, rate, {}, runner=FakeRunner(rows(*boxes)))
    detections = summary["detections"]
    assert len(detections) == 1
    detection = detections[0]
    assert detection["hopping"] is True
    assert detection["sub_bands"] == len(visited)
    assert detection["session_id"] == 1
    truth = signal_truth(generation)[0]
    assert truth["hopping"] is True
    assert detection["center_hz"] == pytest.approx(truth["center_hz"], abs=2_000.0)
    assert detection["bandwidth_hz"] == pytest.approx(truth["bandwidth_hz"], rel=0.2)
    metrics = evaluate_detections(signal_truth(generation), detections)
    assert metrics["matched"] == 1 and metrics["false_alarm"] == 0


def test_ml_detect_handles_floor_noise_only():
    """只有噪声也应返回契约完整、可序列化的结果（信噪比触底）。"""
    rng = np.random.default_rng(7)
    data = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)) * 0.05
    _, _, _, meta = image_context(data.astype(np.complex128), RATE)
    summary, arrays = ml_detect(data.astype(np.complex128), RATE, {},
                                runner=FakeRunner(rows(((0.5, 0.5, 1.0, 1.0), 0.9))))
    assert len(summary["detections"]) <= 1
    if summary["detections"]:
        assert summary["detections"][0]["snr_db"] <= 3.0
    assert arrays["detection_boxes"].shape[0] == len(summary["detections"])
    assert json.dumps(summary, ensure_ascii=False, allow_nan=False)
    empty, _ = ml_detect(data.astype(np.complex128), RATE, {},
                         runner=FakeRunner(np.zeros((0, 6))))
    assert empty["detections"] == []
    assert empty["raw_boxes"]["rows"] == 0
    assert "baseline" in empty


def test_ml_detect_config_validation():
    data, rate, _, _ = scene()
    runner = FakeRunner(np.zeros((0, 6)))
    with pytest.raises(ValueError, match="不支持的检测配置项"):
        ml_detect(data, rate, {"unknown": 1}, runner=runner)
    with pytest.raises(ValueError, match="应为有限数值"):
        ml_detect(data, rate, {"score_threshold": True}, runner=runner)
    with pytest.raises(ValueError, match="不应大于"):
        ml_detect(data, rate, {"iou_threshold": 2.0}, runner=runner)
    # nfft/图像尺寸/动态范围必须与清单一致（数值一致性前提）
    with pytest.raises(ValueError, match="nfft"):
        ml_detect(data, rate, {"nfft": 1024}, runner=runner)
    with pytest.raises(ValueError, match="输入图像尺寸"):
        ml_detect(data, rate, {"image_size": 512}, runner=runner)
    with pytest.raises(ValueError, match="图像动态范围"):
        ml_detect(data, rate, {"dynamic_range_db": 40.0}, runner=runner)
    with pytest.raises(ValueError, match="图像排布"):
        ml_detect(data, rate, {"image_contract": "other"}, runner=runner)
    without_manifest = FakeRunner(np.zeros((0, 6)), manifest={})
    _, arrays = ml_detect(data, rate, {"nfft": NFFT}, runner=without_manifest)
    assert arrays["detection_boxes"].shape == (0, 4)


def test_ml_detect_requires_model_or_runner():
    data, rate, _, _ = scene()
    with pytest.raises(ValueError, match="模型清单路径"):
        ml_detect(data, rate, {})


def test_ml_detect_validates_samples():
    _, rate, _, _ = scene()
    runner = FakeRunner(np.zeros((0, 6)))
    with pytest.raises(ValueError):
        ml_detect(np.zeros(0, dtype=np.complex128), rate, {}, runner=runner)


def test_ml_detect_arrays_and_determinism():
    data, rate, _, generation = scene()
    _, _, _, meta = image_context(data, rate)
    box = truth_box(meta, generation)
    first, arrays = ml_detect(data, rate, {}, runner=FakeRunner(rows((box, 0.9))))
    second, _ = ml_detect(data, rate, {}, runner=FakeRunner(rows((box, 0.9))))
    assert first["detections"] == second["detections"]
    assert arrays["detection_boxes"].shape == (1, 4)
    assert arrays["model_boxes"].shape == (1, 4)
    assert arrays["model_scores"][0] == pytest.approx(0.9)
    assert arrays["detection_id"][0] == 1
    assert arrays["detection_snr_db"][0] == first["detections"][0]["snr_db"]
    assert arrays["detection_power_dbfs"][0] == first["detections"][0]["power_dbfs"]
    assert "spectrogram_db" in arrays and "spectrum_db" in arrays
    assert arrays["image_size"][0] == SIZE


# ------------------------------------------------------------------ 服务层与命令行

def test_service_ml_detect_requires_runtime(tmp_path):
    """未安装 onnxruntime 时给出可执行的中文提示，而不是栈回溯。"""
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime")
    from signal_analysis.tasks import run_job

    workspace = Workspace(tmp_path)
    asset = run_job({"workspace": str(workspace.root), "action": "demo", "count": 4096})
    path, _ = manifest_file(tmp_path)
    with pytest.raises(Exception) as excinfo:
        run_job({"workspace": str(workspace.root), "action": "ml_detect",
                 "asset_id": asset["id"], "manifest": str(path)})
    assert "onnxruntime" in str(excinfo.value)


def test_cli_ml_detect_reports_missing_runtime(tmp_path, capsys):
    if runtime_version() is not None:
        pytest.skip("本机已安装 onnxruntime")
    from signal_analysis.cli import main

    workspace = tmp_path / "ws"
    assert main(["--workspace", str(workspace), "demo", "--count", "4096"]) == 0
    capsys.readouterr()
    asset_id = Workspace(workspace).list_assets()[0]["id"]
    path, _ = manifest_file(tmp_path)
    # 清单参数先于推理校验：示例模型文件是占位字节，仍应报缺依赖
    code = main(["--workspace", str(workspace), "ml-detect", asset_id, str(path)])
    assert code == 1
    assert "onnxruntime" in capsys.readouterr().err


def test_cli_ml_manifest_writes_and_validates(tmp_path, capsys):
    from signal_analysis.cli import main

    model = tmp_path / "detector.onnx"
    model.write_bytes(b"placeholder")
    code = main(["ml-manifest", str(model), str(tmp_path / "model.json"),
                 "--id", "dut", "--version", "0.2.0", "--framework", "yolox",
                 "--license", "Apache-2.0", "--dataset", "自建 1024² 时频图",
                 "--notes", "演示清单"])
    assert code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["id"] == "dut" and manifest["version"] == "0.2.0"
    assert manifest["input"]["image_size"] == SIZE
    assert manifest["training"]["license"] == "Apache-2.0"
    resolved, library = read_model_manifest(tmp_path / "model.json")
    assert resolved["labels"] == ["emitter"] and library == model.resolve()
