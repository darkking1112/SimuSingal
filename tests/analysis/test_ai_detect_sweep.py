"""AI 检测扫描（``benchmarks/ai_detect_sweep.py``）的用例。

覆盖三层：

* 纯函数：分桶、汇总比例、阈值挑选、按分组拆分——都用合成字典，不碰模型；
* 可比性：场景表必须与 ``benchmarks/detect_sweep.py``（传统能量检测）完全一致，
  否则两条路径的数字不能逐格对比；
* 端到端：用一个**桩 runner** 跑完整条链路（``ml_detect`` → 评测 → ROC → 报告），
  因此既不需要 onnxruntime，也不需要模型文件，CI 上可复现。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "benchmarks"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bench_{name}", BENCHMARKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = _load("ai_detect_sweep")
classic = _load("detect_sweep")


class StubRunner:
    """最简模型会话：恒定输出一行高置信框（覆盖整个时长、居中频段）。"""

    def __init__(self, rows, manifest):
        self._rows = np.asarray(rows, dtype=np.float64)
        self.manifest = dict(manifest)
        self.model_name = "stub@0"
        self.runtime_version = "stub"
        self.threads = None

    def run(self, image):  # pragma: no cover - 形状由 ModelRunner 契约约束，这里只做校验
        size = int(self.manifest["input"]["image_size"])
        assert np.asarray(image).shape == (size, size)
        return self._rows


def _manifest(image_size=128):
    return {
        "id": "stub",
        "version": "0",
        "labels": ["emitter"],
        "label_semantics": "session_v1",
        "input": {"image_size": image_size, "spectrogram_nfft": 128,
                  "dynamic_range_db": 60.0},
        "manifest_path": "stub.json",
        "sha256": None,
    }


def _args(**overrides):
    values = {"modes": ["qpsk"], "snr_grid": [10.0], "seeds": 2, "duration": 0.2,
              "power_dbfs": -10.0, "min_score": 0.02, "operating_score": 0.25,
              "score_grid": [0.05, 0.25, 0.5], "far_targets": [1e-2], "far_basis": "frame",
              "noise_trials": 2, "noise_power_dbfs": -10.0, "threads": None,
              "nfft": 0, "amc_data": "", "amc_model": "", "amc_split": "val"}
    values.update(overrides)
    return type("Args", (), values)()


def _run(rows=None, **overrides):
    """跑一次端到端扫掠，返回 ``(report, points)``。"""
    rows = np.array([[0.5, 0.5, 0.5, 0.02, 0.9, 0.0]]) if rows is None else rows
    manifest = _manifest()
    runner = StubRunner(rows, manifest)
    args = _args(**overrides)
    report, points, _ = sweep.run_sweep(args, runner, manifest, emit=lambda *_: None)
    return report, points


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------


def test_bucket_rounds_down_to_five_db_buckets():
    assert sweep._bucket(12.4) == "+10~+15 dB"
    assert sweep._bucket(-0.2) == "-5~+0 dB"
    assert sweep._bucket(-12.0) == "-15~-10 dB"
    assert sweep._bucket(None) is None


def test_ratio_returns_none_for_empty_denominators():
    assert sweep._ratio(3, 0) is None
    assert sweep._ratio(3, 4) == pytest.approx(0.75)


def test_clean_replaces_non_finite_floats_with_none():
    payload = sweep._clean({"a": float("nan"), "b": [float("inf"), 1, "x"],
                            "c": np.arange(2), "d": np.float64(2.5)})
    assert payload == {"a": None, "b": [None, 1, "x"], "c": [0, 1], "d": 2.5}
    json.dumps(payload, allow_nan=False)  # 报告必须能无 nan 写出


def test_percentiles_are_none_for_empty_samples():
    assert sweep._percentiles([]) is None
    stats = sweep._percentiles([1.0, 2.0, 3.0, 4.0])
    assert stats["trials"] == 4 and stats["p50"] == 2.5 and stats["max"] == 4.0


def test_choose_operating_point_picks_the_highest_eligible_threshold():
    # 虚警率随阈值**上升**的假想模型：只能取满足目标的最靠上合法阈值。
    points = [
        {"score": 0.05, "far_per_frame": 1e-5, "far_per_second": 1e-1},
        {"score": 0.25, "far_per_frame": 1e-4, "far_per_second": 1e-3},
        {"score": 0.50, "far_per_frame": 1e-1, "far_per_second": 1e-5},
    ]
    assert sweep.choose_operating_point(points, 1e-2, "frame") == 0.25
    assert sweep.choose_operating_point(points, 1e-4, "frame") == 0.25
    assert sweep.choose_operating_point(points, 1e-5, "frame") == 0.05
    assert sweep.choose_operating_point(points, 1e-9, "frame") is None
    # 换口径（每秒）后合法集合完全不同：最高合法阈值变成 0.5
    assert sweep.choose_operating_point(points, 1e-5, "second") == 0.5
    assert sweep.choose_operating_point(points, 1e-6, "second") is None


def test_roc_points_use_pooled_denominators():
    thresholds = [0.05, 0.5]
    signal_roc = {0.05: {"tp": 8, "fp": 2, "fn": 2}, 0.5: {"tp": 4, "fp": 0, "fn": 6}}
    noise = {0.05: {"fp": 10}, 0.5: {"fp": 1}}
    points = sweep.roc_points(thresholds, signal_roc, noise, frames=100, seconds=5.0)
    assert points[0]["recall"] == pytest.approx(0.8)
    assert points[0]["precision"] == pytest.approx(0.8)
    assert points[0]["far_per_frame"] == pytest.approx(0.1)
    assert points[1]["far_per_second"] == pytest.approx(0.2)


def test_breakdown_reports_recall_and_precision_per_group():
    thresholds = [0.05, 0.5]
    tally = {0.5: {"tp": 3, "fp": 1, "fn": 1}}
    rows = sweep.breakdown(tally, 0.5, thresholds, {"qpsk": tally, "am": tally})
    assert rows["qpsk"]["recall"] == pytest.approx(0.75)
    assert rows["qpsk"]["precision"] == pytest.approx(0.75)
    assert rows["qpsk"]["truth"] == 4


def test_candidate_rows_match_the_network_layout():
    empty = sweep.candidate_rows({"model_boxes": np.zeros((0, 4)),
                                  "model_scores": np.zeros(0)})
    assert empty.shape == (0, 6)
    rows = sweep.candidate_rows({"model_boxes": np.array([[0.5, 0.5, 0.2, 0.1]]),
                                 "model_scores": np.array([0.7])})
    assert rows.tolist() == [[0.5, 0.5, 0.2, 0.1, 0.7, 0.0]]


def test_candidate_rows_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        sweep.candidate_rows({"model_boxes": np.array([[0.5, 0.5, 0.2, 0.1]]),
                              "model_scores": np.zeros(0)})


def test_snr_grid_default_matches_the_classic_benchmark():
    # 字面量镜像 detect_sweep.py 的 ``--snr`` 默认值：两边必须同时改，否则数字不可比。
    assert sweep.SNR_GRID == (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
    assert sweep._parse_args(["--manifest", "x.json"]).snr_grid == list(sweep.SNR_GRID)


def test_negative_snr_grid_is_accepted_in_both_write_forms():
    spaced = sweep._parse_args(["--manifest", "x.json", "--snr", "-5,0,10"])
    equals = sweep._parse_args(["--manifest", "x.json", "--snr=-5,0,10"])
    assert spaced.snr_grid == equals.snr_grid == [-5.0, 0.0, 10.0]
    # 单个负值本来就是合法负数，不应被合并逻辑破坏
    single = sweep._parse_args(["--manifest", "x.json", "--power-dbfs", "-10"])
    assert single.power_dbfs == pytest.approx(-10.0)


def test_unknown_mode_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        sweep._parse_args(["--manifest", "x.json", "--modes", "qpsk,nosuchmode"])


def test_empty_list_option_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        sweep._parse_args(["--manifest", "x.json", "--far-targets", " , "])


# ---------------------------------------------------------------------------
# 可比性：与 benchmarks/detect_sweep.py 的场景表完全一致
# ---------------------------------------------------------------------------


def test_scenarios_are_identical_to_the_classic_benchmark():
    assert sweep.SCENARIOS == classic.SCENARIOS
    assert sweep.MODES == classic.MODES
    assert sweep.RATE_HZ == classic.RATE_HZ


def test_scenario_spec_keeps_frequency_and_bandwidth():
    spec = sweep.scenario_spec("qam16")
    assert spec == {"mode": "qam16", "power_dbfs": -10.0, "offset": -100e3,
                    "bandwidth": 300e3}


# ---------------------------------------------------------------------------
# 端到端：桩 runner 跑完整链路
# ---------------------------------------------------------------------------


def test_sweep_report_shape_and_contract():
    report, points = _run()
    assert report["schema"] == "ai_detect_sweep_v1"
    assert report["truth_source"] == "synthetic"
    assert report["model"]["label_semantics"] == "session_v1"
    assert set(report) >= {"schema", "created", "truth_source", "model", "environment",
                           "settings", "cells", "per_snr", "parameters", "roc",
                           "operating_points", "latency", "noise_only",
                           "candidate_selfcheck", "amc", "notes"}
    assert report["settings"]["modes"] == ["qpsk"]
    assert report["settings"]["scenarios"] == classic.SCENARIOS
    assert set(report["environment"]) >= {"python", "numpy", "onnxruntime", "cpu_count"}
    assert report["amc"] is None
    # ROC 网格自动包含候选门限本身 → 3 个网格点 + 门限 0.02
    assert [point["score"] for point in points] == [0.02, 0.05, 0.25, 0.5]
    assert report["roc"]["gate"] == pytest.approx(0.02)


def test_sweep_measures_recall_on_signal_and_false_alarms_on_noise():
    """恒定居中框：信号记录必被匹配（召回 1），纯噪声记录必是虚警（FAR > 0）。"""
    report, points = _run()
    cell = report["cells"]["qpsk@10"]
    assert cell["records"] == 2
    assert cell["full_path"]["recall"] == pytest.approx(1.0)
    assert cell["full_path"]["precision"] == pytest.approx(1.0)
    assert all(point["recall"] == pytest.approx(1.0) for point in points)
    assert all(point["noise_false_boxes"] > 0 for point in points)
    assert all(point["far_per_frame"] > 0 for point in points)
    assert report["per_snr"]["buckets"], "应至少有一个实测 SNR 分档"
    assert report["per_snr"]["total"]["recall"] == pytest.approx(1.0)
    assert report["deployed_point"]["recall"] == pytest.approx(1.0)


def test_sweep_selfcheck_confirms_the_roc_replay_matches_the_shipped_decode():
    report, _ = _run()
    selfcheck = report["candidate_selfcheck"]
    assert selfcheck["mismatch_records"] == 0
    assert selfcheck["shipped"] == selfcheck["replayed"] > 0


def test_sweep_latency_has_all_stages_and_trial_counts():
    report, _ = _run()
    expected = {"context_ms", "image_ms", "inference_ms", "gate_ms", "other_ms",
                "total_ms"}
    assert set(report["latency"]) == expected
    assert report["latency"]["inference_ms"]["trials"] == 2  # 1 单元 × 2 种子
    assert report["latency"]["total_ms"]["p95"] >= report["latency"]["total_ms"]["p50"]


def test_sweep_operating_point_reports_per_mode_and_per_snr_breakdown():
    report, _ = _run()
    point = report["operating_points"][0]
    assert point["basis"] == "frame"
    assert point["per_mode"]["qpsk"]["recall"] == pytest.approx(1.0)
    assert list(point["per_snr"]) == ["10 dB"]


def test_deployed_point_is_labelled_as_a_roc_threshold_not_the_candidate_gate():
    report, _ = _run()
    assert report["deployed_point"]["score"] == pytest.approx(0.25)
    assert "候选门限 0.02" in report["deployed_point"]["note"]
    assert "0.02" in report["per_snr"]["basis"]


def test_sweep_grid_below_the_candidate_gate_is_rejected():
    with pytest.raises(SystemExit):
        _run(min_score=0.5, score_grid=[0.05, 0.5])


def test_sweep_rejects_per_hop_label_semantics():
    manifest = _manifest()
    manifest["label_semantics"] = "per_hop_v1"
    runner = StubRunner(np.zeros((0, 6)), manifest)
    with pytest.raises(SystemExit):
        sweep.run_sweep(_args(), runner, manifest, emit=lambda *_: None)


def test_sweep_survives_a_model_with_no_detections():
    """模型输出为空时不能崩：召回 0、虚警 0、工作点仍然可解释。"""
    report, points = _run(rows=np.zeros((0, 6)))
    cell = report["cells"]["qpsk@10"]
    assert cell["full_path"]["recall"] == pytest.approx(0.0)
    assert cell["full_path"]["false_alarm"] == 0
    assert all(point["noise_false_boxes"] == 0 for point in points)
    assert report["operating_points"][0]["score"] == 0.5  # 虚警为 0 时取最高阈值
    assert report["candidate_selfcheck"]["shipped"] == 0


def test_cli_json_roundtrip_is_nan_free(tmp_path, monkeypatch, capsys):
    """命令行入口：桩 runner 替代真实会话，报告能无 nan 写出并再次读回。"""
    target = tmp_path / "report.json"
    argv = ["--manifest", "stub.json", "--modes", "qpsk", "--snr", "10", "--seeds", "1",
            "--duration", "0.2", "--noise-trials", "1", "--json", str(target)]

    def fake_load_runner(manifest_path, threads=None):
        manifest = _manifest()
        return StubRunner(np.array([[0.5, 0.5, 0.5, 0.02, 0.9, 0.0]]), manifest), manifest, "stub"

    import signal_analysis.ml.runtime as runtime
    monkeypatch.setattr(runtime, "load_runner", fake_load_runner)
    sweep.main(argv)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "ai_detect_sweep_v1"
    assert payload["cells"]["qpsk@10"]["full_path"]["recall"] == pytest.approx(1.0)
    stdout = capsys.readouterr().out
    assert "模型 stub@0" in stdout
    assert "固定虚警率工作点" in stdout
    assert "候选门限 0.02" in stdout


def test_truth_sigmf_is_refused_explicitly(monkeypatch):
    def fake_load_runner(*_args, **_kwargs):  # pragma: no cover - 不应被调用
        raise AssertionError("--truth sigmf 必须在加载模型之前就报错")

    import signal_analysis.ml.runtime as runtime
    monkeypatch.setattr(runtime, "load_runner", fake_load_runner)
    with pytest.raises(SystemExit):
        sweep.main(["--manifest", "stub.json", "--truth", "sigmf"])
