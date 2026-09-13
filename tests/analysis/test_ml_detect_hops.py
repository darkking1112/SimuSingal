"""AI 逐跳参数估计（``ml_detect_hops``）的验收测试。

分工（Route B）：**判决**（一跳落在时频图的哪一块）来自网络，模型只给频带与
粗略时间；**辐射量**（驻留、功率、单跳带宽、带内信噪比）全部在**原始 PSD**
上由 ``hop_track_v1`` 的同一套测量函数重算。因此本文件不比较"模型输出的数字"，
而是断言三件事：

* ``arrays`` / ``summary`` 的键集与 :func:`detect_hops` 对齐（同一界面、同一
  报告都能直接渲染）；
* 馈入逐跳真值框时参数误差为零量级（说明几何通路没有偏移或翻转）；
* 诚实性出口在起作用：会话级模型被直接拒绝，检不到跳时给出原因而不是空表。

测试全部使用注入的假会话（``FakeRunner``），不依赖 ``onnxruntime``。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_analysis.core_api import detect_hops, generate_iq  # noqa: E402
from signal_analysis.evaluation import (HOP_CONTRACT, evaluate_detections,  # noqa: E402
                                        hop_truth)
from signal_analysis.ml import (HOP_KEYS, band_to_box, detection_image,  # noqa: E402
                                ml_detect_hops, spectral_context)

RATE = 1_000_000.0
NFFT = 512
SIZE = 1024
DURATION = 0.2
HOP_RATE = 50.0
CONTRACT_INPUT = {"name": "images", "image_size": SIZE, "channels": 1,
                  "spectrogram_nfft": NFFT, "dynamic_range_db": 60.0,
                  "layout": "time_frequency_grayscale_v1"}

PER_HOP_MANIFEST = {"id": "hop-fake", "version": "0.1.0", "labels": ["emitter"],
                    "label_semantics": "per_hop_v1", "input": dict(CONTRACT_INPUT),
                    "output": {"name": "detections",
                               "layout": "normalized_boxes_v1"}}
SESSION_MANIFEST = {**PER_HOP_MANIFEST, "label_semantics": "session_v1"}


class FakeRunner:
    """按图像尺寸返回预设检测框的假会话（与 ModelRunner 接口一致）。"""

    model_name = "hop-fake@0.1.0"
    runtime_version = "test"

    def __init__(self, boxes, manifest=None):
        self.manifest = dict(PER_HOP_MANIFEST if manifest is None else manifest)
        self.boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
        self.images = []

    def run(self, image):
        self.images.append(np.array(image, dtype=np.float32))
        return self.boxes.reshape(1, -1, 6)


def scene(mode="fh_rc", hops=10, hop_rate=HOP_RATE, snr_db=18.0, seed=5,
          duration=DURATION, **extra):
    """生成一段跳频 IQ，返回 ``(samples, generation)``。"""
    signals = [{"mode": mode, "offset": 0.0, "bandwidth": 200_000.0,
                "power_dbfs": -8.0, "hops": hops, "hop_rate": hop_rate, **extra}]
    return generate_iq(RATE, duration, signals,
                       noise={"enabled": True, "bandwidth": RATE, "snr_db": snr_db},
                       seed=seed)


def truth_boxes(samples, generation, size=SIZE, indices=None):
    """逐跳真值 → 归一化图像框 ``(N, 6)``（分数列固定 0.9）。"""
    summary, arrays = spectral_context(samples, RATE, {"nfft": NFFT})
    image, meta = detection_image(arrays, summary, size)
    truth = hop_truth(generation)
    chosen = truth if indices is None else [truth[index] for index in indices]
    payload = [[*band_to_box(meta, item["f_low_hz"], item["f_high_hz"],
                             item["t_start_s"], item["t_end_s"]), 0.9, 0.0]
               for item in chosen]
    return np.asarray(payload, dtype=np.float64).reshape(-1, 6)


# ---------------------------------------------------------------------------
# 清单门槛：只有逐跳标签的模型能走这条通路
# ---------------------------------------------------------------------------

def test_session_model_is_rejected_with_guidance():
    samples, _ = scene()
    runner = FakeRunner(np.zeros((0, 6)), manifest=SESSION_MANIFEST)
    with pytest.raises(ValueError) as excinfo:
        ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=runner)
    message = str(excinfo.value)
    assert "session_v1" in message and "per_hop_v1" in message
    assert "--labels hop" in message and "ml-detect" in message
    # 拒绝发生在推理之前：模型一次都没有被调用
    assert runner.images == []


def test_session_model_without_declaration_is_rejected():
    """旧清单没有 ``label_semantics`` 字段时按会话级处理，同样拒绝。"""
    samples, _ = scene()
    manifest = {key: value for key, value in PER_HOP_MANIFEST.items()
                if key != "label_semantics"}
    with pytest.raises(ValueError, match="逐跳估计"):
        ml_detect_hops(samples, RATE, {"nfft": NFFT},
                       runner=FakeRunner(np.zeros((0, 6)), manifest=manifest))


def test_unknown_option_is_rejected_but_hop_keys_are_accepted():
    samples, _ = scene()
    runner = FakeRunner(np.zeros((0, 6)))
    with pytest.raises(ValueError, match="不支持的检测配置项"):
        ml_detect_hops(samples, RATE, {"nfft": NFFT, "hop_rate_hz": 50.0}, runner=runner)
    for key in HOP_KEYS:
        value = 8 if key == "max_hops" else None
        if value is None:
            continue
        ml_detect_hops(samples, RATE, {"nfft": NFFT, key: value}, runner=runner)


# ---------------------------------------------------------------------------
# 逐跳真值框 → 逐跳参数（几何通路与测量口径）
# ---------------------------------------------------------------------------

def test_truth_boxes_reproduce_the_hop_plan():
    samples, generation = scene()
    truth = hop_truth(generation)
    runner = FakeRunner(truth_boxes(samples, generation))
    summary, arrays = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=runner)
    assert summary["contract"] == HOP_CONTRACT
    assert summary["algorithm"] == "ml_detect_hops:hop-fake@0.1.0"
    assert summary["resolvable"] is True and summary["reason"] is None
    assert len(summary["hops"]) == len(truth)
    assert [item["id"] for item in summary["hops"]] == list(range(1, len(truth) + 1))
    assert summary["sessions"], "逐跳结果应聚出至少一个会话"
    assert summary["config"]["nfft"] == NFFT
    reference = detect_hops(samples, RATE, {"nfft": NFFT}, False)[0]
    # 逐跳测量必须与 detect_hops 共用同一张 STFT 网格，否则带宽/驻留会整体错位
    assert summary["hop_samples"] == reference["hop_samples"]
    assert summary["frame_count"] == reference["frame_count"]
    assert summary["frame_interval_s"] == pytest.approx(reference["frame_interval_s"])
    assert summary["freq_resolution_hz"] == pytest.approx(reference["freq_resolution_hz"])
    assert summary["noise_floor_dbfs_per_hz"] == pytest.approx(
        reference["noise_floor_dbfs_per_hz"], abs=0.1)

    metrics = evaluate_detections(truth, summary["hops"], contract=HOP_CONTRACT)
    assert metrics["true"] == len(truth) and metrics["missed"] == 0
    assert metrics["false_alarm"] == 0 and metrics["f1"] == pytest.approx(1.0)
    # 模型只给框、不给物理量：这些数字全部来自原始 PSD 上的重测，
    # 因此与真值的偏差必须是“测量误差”量级（几十 Hz / 几毫秒），而不是整体错位。
    assert metrics["center_mae_hz"] < RATE / NFFT
    assert metrics["bandwidth_mape"] < 0.5
    assert abs(summary["hop_rate_limit_hz"] - reference["hop_rate_limit_hz"]) < 1e-6


def test_each_hop_carries_the_score_of_the_box_that_produced_it():
    """逐跳明细带模型分数：弱框写的跳一眼可见（传统通路没有这两个键）。"""
    samples, generation = scene()
    boxes = truth_boxes(samples, generation)
    boxes[2, 4] = 0.31
    boxes[3, 4] = 0.77
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=FakeRunner(boxes))
    scores = [item["model_confidence"] for item in summary["hops"]]
    assert len(scores) == len(summary["hops"])
    # 分数只能来自候选框，不许由 hop 的置信度（带内 SNR 换算）冒充
    assert set(scores) <= set(np.round(boxes[:, 4], 4))
    target = hop_truth(generation)[2]
    closest = min(summary["hops"],
                  key=lambda item: abs(item["center_hz"] - target["center_hz"]))
    assert closest["model_confidence"] == pytest.approx(0.31, abs=1e-3)
    assert closest["model_label"] == "emitter"
    # 传统逐跳通路没有模型，明细里就一行都不该出现这两个键
    assert all("model_confidence" not in item for item in summary["traditional"]["hops"])


def test_partial_model_output_counts_misses_and_false_alarms():
    samples, generation = scene()
    truth = hop_truth(generation)
    runner = FakeRunner(truth_boxes(samples, generation, indices=[0, 1, 2]))
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=runner)
    metrics = evaluate_detections(truth, summary["hops"], contract=HOP_CONTRACT)
    assert metrics["true"] == len(truth)
    assert metrics["matched"] >= 1
    assert metrics["missed"] >= 1
    assert metrics["false_alarm"] == 0


def test_max_hops_trim_is_reported():
    samples, generation = scene(hops=10)
    runner = FakeRunner(truth_boxes(samples, generation))
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT, "max_hops": 3}, runner=runner)
    assert summary["config"]["max_hops"] == 3
    assert len(summary["hops"]) == 3
    assert summary["raw_boxes"]["candidates"] >= 3


def test_low_score_boxes_are_dropped_by_candidate_decoding():
    samples, generation = scene()
    boxes = truth_boxes(samples, generation)
    boxes[:, 4] = 0.05
    runner = FakeRunner(boxes)
    summary, _ = ml_detect_hops(samples, RATE,
                                {"nfft": NFFT, "score_threshold": 0.25}, runner=runner)
    assert summary["hops"] == []
    assert summary["resolvable"] is False
    assert "置信度" in summary["reason"]


def test_empty_output_explains_instead_of_faking_hops():
    samples, _ = scene()
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT},
                                runner=FakeRunner(np.zeros((0, 6))))
    assert summary["hops"] == [] and summary["sessions"] == []
    assert summary["contract"] == HOP_CONTRACT
    assert summary["resolvable"] is False
    assert "没有一跳" in summary["reason"]
    assert summary["raw_boxes"]["rows"] == 0


# ---------------------------------------------------------------------------
# 契约对齐：界面与报告能直接渲染 AI 逐跳结果
# ---------------------------------------------------------------------------

RENDER_KEYS = ("frequency", "frame_time", "spectrogram_db", "spectrum_db",
               "spectrum_median_db", "threshold_db", "noise_floor_db", "hop_boxes",
               "hop_id", "hop_session_id")


def test_arrays_superset_of_detect_hops_contract():
    samples, generation = scene()
    runner = FakeRunner(truth_boxes(samples, generation))
    _, arrays = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=runner)
    _, reference = detect_hops(samples, RATE, {"nfft": NFFT}, False)
    missing = set(reference) - set(arrays)
    assert not missing, f"AI 逐跳结果缺少 detect_hops 的数组：{sorted(missing)}"
    for key in RENDER_KEYS:
        assert key in arrays
    assert arrays["hop_boxes"].shape[1] == 4
    assert arrays["hop_boxes"].shape[0] == len(arrays["hop_id"])
    assert arrays["model_boxes"].shape[1] == 4
    assert arrays["model_boxes"].shape[0] == arrays["model_scores"].shape[0]
    assert int(arrays["image_size"].ravel()[0]) == SIZE
    assert arrays["spectrogram_db"].shape == arrays["spectrogram_raw_db"].shape


def test_summary_covers_every_hop_field_detect_hops_publishes():
    samples, generation = scene()
    runner = FakeRunner(truth_boxes(samples, generation))
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=runner)
    reference = detect_hops(samples, RATE, {"nfft": NFFT}, False)[0]
    missing = set(reference) - set(summary)
    assert not missing, f"AI 逐跳摘要缺少 detect_hops 的字段：{sorted(missing)}"
    for key in ("id", "session_id", "center_hz", "bandwidth_hz", "f_low_hz", "f_high_hz",
                "t_start_s", "t_end_s", "dwell_s", "power_dbfs", "snr_db", "bin_count",
                "band_nfft", "centroid_hz"):
        assert key in summary["hops"][0]
    for key in ("session_id", "hop_count", "hop_rate_hz", "hop_period_s", "dwell_median_s",
                "duty_cycle", "hop_frequencies_hz", "hop_span_hz", "bandwidth_hz", "snr_db"):
        assert key in summary["sessions"][0]
    assert summary["model"]["id"] == "hop-fake"
    assert summary["image"]["size"] == SIZE
    assert set(summary["timing"]) == {"context_ms", "inference_ms", "total_ms"}
    # 逐跳会话不经过 _merge_sessions，检出条数与会话数由信号本身决定
    assert summary["baseline"]["contract"] == "detect_result_v1"
    assert summary["traditional"]["algorithm"] == "hop_track_v1"
    assert summary["traditional"]["contract"] == HOP_CONTRACT


def test_traditional_and_session_baselines_can_be_disabled():
    samples, generation = scene()
    boxes = truth_boxes(samples, generation)
    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=FakeRunner(boxes),
                                with_sessions=False, with_traditional=False)
    assert "baseline" not in summary and "traditional" not in summary
    assert all(item["session_detection_id"] is None for item in summary["sessions"])

    summary, _ = ml_detect_hops(samples, RATE, {"nfft": NFFT}, runner=FakeRunner(boxes),
                                with_traditional=True)
    traditional = summary["traditional"]
    assert traditional["hops"], "同一段数据上传统逐跳基线应有输出"
    assert summary["baseline"]["detections"]
    ids = [item["session_detection_id"] for item in summary["sessions"]]
    assert any(value is not None for value in ids), "会话应与会话级检出互链"
    assert set(ids) <= {item["id"] for item in summary["baseline"]["detections"]} | {None}
    ours = evaluate_detections(hop_truth(generation), summary["hops"], contract=HOP_CONTRACT)
    theirs = evaluate_detections(hop_truth(generation), traditional["hops"],
                                 contract=HOP_CONTRACT)
    assert ours["f1"] >= theirs["f1"], "喂入真值框时 AI 逐跳不应弱于传统逐跳"


def test_service_payload_scores_all_three_granularities(tmp_path):
    """服务层：逐跳真值、会话基线、传统逐跳三个口径各评一次，互不混用。"""
    from signal_analysis.services import _attach_hop_truth
    from signal_analysis.storage import Workspace

    workspace = Workspace(tmp_path / "ws")
    samples, generation = scene()
    asset = workspace.add_samples(samples, RATE, "hop-scene",
                                  source="generated:iq_fh_rc_v1",
                                  metadata={"generation": generation})
    truth = hop_truth(generation)
    hops = [{"id": index + 1, "session_id": 1, "center_hz": item["center_hz"],
             "bandwidth_hz": item["bandwidth_hz"], "f_low_hz": item["f_low_hz"],
             "f_high_hz": item["f_high_hz"], "t_start_s": item["t_start_s"],
             "t_end_s": item["t_end_s"], "dwell_s": item["t_end_s"] - item["t_start_s"],
             "power_dbfs": -20.0, "snr_db": 20.0}
            for index, item in enumerate(truth)]
    payload = {"hops": hops, "sessions": [{"session_id": 1, "session_detection_id": None}]}
    summary = {"hops": hops,
               "baseline": {"detections": [{"id": 1, "f_low_hz": -100e3,
                                            "f_high_hz": 100e3}]},
               "traditional": {"hops": hops[:3]}}
    result = _attach_hop_truth(payload, workspace, asset["id"], summary)
    assert result["truth"] == truth
    assert result["metrics"]["true"] == len(truth)
    assert result["metrics"]["f1"] == pytest.approx(1.0)
    assert result["baseline_metrics"]["true"] == 1
    assert result["traditional_metrics"]["true"] == len(truth)
    assert result["traditional_metrics"]["recall"] < 1.0
    # 真值不适用时（导入或原生插件产出的资产），三个口径都不写指标、只写原因
    blank = workspace.add_samples(samples[:1024], RATE, "imported")
    result = _attach_hop_truth({"hops": hops, "sessions": []}, workspace,
                               blank["id"], summary)
    assert result["metrics"] is None and result["truth"]["available"] is False
    assert "traditional_metrics" not in result and "baseline_metrics" not in result


def test_report_export_renders_three_hop_columns(tmp_path):
    from common.reports import export_report

    payload = {
        "kind": "ml_detect_hops", "run_id": "abc123", "asset_id": "asset",
        "asset_name": "hop-scene", "contract": HOP_CONTRACT,
        "algorithm": "ml_detect_hops:hop-fake@0.1.0", "resolvable": True, "reason": None,
        "model": {"id": "hop-fake", "version": "0.1.0", "runtime_version": "1.30.0"},
        "truth": [{"id": 1}],
        "metrics": {"true": 1, "matched": 1, "missed": 0, "false_alarm": 0, "precision": 1.0,
                    "recall": 1.0, "f1": 1.0, "center_mae_hz": 10.0, "bandwidth_mape": 0.01,
                    "snr_mae_db": 0.5},
        "traditional_metrics": {"true": 1, "matched": 0, "missed": 1, "false_alarm": 0,
                                "precision": None, "recall": 0.0, "f1": 0.0,
                                "center_mae_hz": None, "bandwidth_mape": None,
                                "snr_mae_db": None},
        "baseline_metrics": {"true": 1, "matched": 1, "missed": 0, "false_alarm": 0,
                             "precision": 1.0, "recall": 1.0, "f1": 1.0,
                             "center_mae_hz": 20.0, "bandwidth_mape": 0.05,
                             "snr_mae_db": 1.0},
        "hops": [{"id": 1, "session_id": 1, "center_hz": 1000.0, "bandwidth_hz": 20000.0,
                  "f_low_hz": -9000.0, "f_high_hz": 11000.0, "t_start_s": 0.0,
                  "t_end_s": 0.02, "dwell_s": 0.02, "power_dbfs": -20.0, "snr_db": 20.0,
                  "model_confidence": 0.8125, "model_label": "emitter"}],
        "sessions": [{"session_id": 1, "hop_count": 1, "hop_rate_hz": 50.0,
                      "hop_period_s": 0.02, "dwell_median_s": 0.02, "duty_cycle": 1.0,
                      "hop_frequencies_hz": [1000.0], "hop_span_hz": 0.0,
                      "bandwidth_hz": 20000.0, "snr_db": 20.0,
                      "session_detection_id": 1}],
    }
    path = export_report(payload, tmp_path / "report.html")
    text = Path(path).read_text(encoding="utf-8")
    assert "AI 跳频参数估计" in text
    assert "AI 逐跳口径" in text and "传统逐跳口径" in text and "会话口径（能量检测基线）" in text
    assert "hop-fake@0.1.0" in text and "原始 PSD" in text
    # 有模型分数时明细表多一列；传统逐跳（无该键）不加列，也不显示 "--"
    assert "模型置信度" in text and "0.812" in text
    plain = {**payload,
             "hops": [{key: value for key, value in payload["hops"][0].items()
                       if key != "model_confidence"}]}
    plain_text = Path(export_report(plain, tmp_path / "plain.html")).read_text(
        encoding="utf-8")
    assert "模型置信度" not in plain_text
    json_path = export_report(payload, tmp_path / "report.json")
    assert json.loads(Path(json_path).read_text(encoding="utf-8"))["kind"] == "ml_detect_hops"


def test_cli_maps_ml_detect_hops_options(monkeypatch, tmp_path):
    from signal_analysis import tasks
    from signal_analysis.cli import main

    captured = {}

    def fake_job(request):
        captured.update(request)
        return {"kind": "ml_detect_hops", "hops": []}

    monkeypatch.setattr(tasks, "run_job", fake_job)
    code = main(["--workspace", str(tmp_path / "ws"), "ml-detect-hops", "asset", "model.json",
                 "--score-threshold", "0.4", "--max-hops", "12", "--min-dwell", "0.001",
                 "--smooth-frames", "2", "--threads", "3", "--no-sessions",
                 "--no-traditional"])
    assert code == 0
    assert captured["action"] == "ml_detect_hops"
    assert captured["asset_id"] == "asset" and captured["manifest"] == "model.json"
    assert captured["with_sessions"] is False and captured["with_traditional"] is False
    assert captured["threads"] == 3
    assert captured["config"] == {"score_threshold": 0.4, "max_hops": 12,
                                  "min_dwell_s": 0.001, "smooth_frames": 2}


def test_ml_manifest_cli_can_declare_per_hop_semantics(tmp_path, capsys):
    from signal_analysis.cli import main

    model = tmp_path / "detector.onnx"
    model.write_bytes(b"onnx-bytes")
    code = main(["--workspace", str(tmp_path / "ws"), "ml-manifest", str(model),
                 str(tmp_path / "model.json"), "--label-semantics", "per_hop_v1"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contract"] == "tf_image_v1"


@pytest.mark.gui
def test_gui_ml_hops_entry_builds_per_hop_request(tmp_path):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    from PySide6 import QtWidgets

    from signal_analysis.gui import MainWindow
    from signal_analysis.tasks import run_job

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    workspace = tmp_path / "ws"
    window = MainWindow(workspace)
    window.show()
    try:
        assert window.hops_ml_button.text() == "AI 估计逐跳参数"
        assert window.hops_ml_traditional.isChecked()
        assert "per_hop_v1" in window.hops_manifest.placeholderText()
        window.hops_ml_button.click()
        assert "请先导入并选择数据" in window.status.text()
        run_job({"workspace": str(workspace), "action": "generate", "sample_rate": RATE,
                 "duration": DURATION, "seed": 5,
                 "noise": {"snr_db": 18.0, "bandwidth": RATE},
                 "signals": [{"mode": "fh_rc", "offset": 0.0, "bandwidth": 200_000.0,
                              "power_dbfs": -8.0, "hops": 10, "hop_rate": HOP_RATE}]})
        window.refresh_assets()
        window.assets.setCurrentRow(0)
        window.hops_ml_button.click()
        assert "逐跳模型清单" in window.status.text()
        captured = {}
        window.start_job = lambda action, **kwargs: captured.update({"action": action,
                                                                    **kwargs})
        window.hops_manifest.setText("/tmp/hopmodel/detector.json")
        window.hops_nfft.setCurrentText("1024")
        window.hops_max.setValue(12)
        window.hops_ml_button.click()
        assert captured["action"] == "ml_detect_hops"
        assert captured["manifest"] == "/tmp/hopmodel/detector.json"
        assert captured["asset_id"] == window.selected_asset()["id"]
        assert captured["with_sessions"] is True and captured["with_traditional"] is True
        assert captured["config"]["max_hops"] == 12
        assert captured["config"]["smooth_frames"] == 4
        # nfft 由清单声明（训练口径），界面上的 STFT 点数不参与 AI 逐跳请求
        assert "nfft" not in captured["config"]
    finally:
        window.close()
        app.processEvents()
