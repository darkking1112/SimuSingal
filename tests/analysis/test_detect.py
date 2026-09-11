"""信号检测（P1）与评估模块（P0）的验收测试。

断言全部对着生成器真值（``signal_truth``）：中心频率、占用带宽、带内
SNR 三个量都以 ``inband_snr_v1`` 口径互相比对，不用手写魔数。
"""
import numpy as np
import pytest

from signal_analysis.core_api import detect_signals, generate_iq
from signal_analysis.evaluation import (classification_metrics, evaluate_detections,
                                        match_detections, signal_truth)

RATE = 1e6
DURATION = 0.5
NFFT = 512
BIN_HZ = RATE / NFFT


def scene(mode, snr_db, seed=1, bandwidth=None, offset=0.0, **extra):
    """Generate one signal and return ``(detection summary, truth list)``."""
    spec = {"mode": mode, "offset": offset, "bandwidth": bandwidth or 200e3,
            "power_dbfs": -8.0, **extra}
    samples, summary = generate_iq(RATE, DURATION, [spec], noise={"snr_db": snr_db}, seed=seed)
    result, arrays = detect_signals(samples, RATE, {"nfft": NFFT})
    return result, signal_truth(summary), arrays


def centric(pair):
    """Centre gate: 2 FFT bins or 2% of the truth bandwidth, whichever larger."""
    return abs(pair["center_error_hz"]) <= max(2 * BIN_HZ,
                                               0.02 * pair["truth_bandwidth_hz"])


# ---------------------------------------------------------------------------
# 契约与配置
# ---------------------------------------------------------------------------

def test_contract_keys_are_frozen():
    summary, _ = detect_signals(np.zeros(4096, dtype=complex), RATE)
    assert summary["contract"] == "detect_result_v1"
    assert summary["algorithm"] == "energy_detect_v1"
    assert summary["snr_definition"] == "inband_snr_v1"
    assert summary["frequency_reference"] == "baseband_offset"
    for key in ("sample_rate_hz", "sample_count", "duration_s", "nfft", "hop_samples",
                "frame_count", "config", "freq_resolution_hz",
                "noise_floor_dbfs_per_hz", "threshold_dbfs_per_hz", "detections"):
        assert key in summary
    for key in ("nfft", "threshold_db", "band_threshold_db", "min_bandwidth_hz",
                "min_duration_s", "max_detections", "merge_bins"):
        assert key in summary["config"]


def test_detection_keys_are_frozen_and_json_safe():
    result, _, _ = scene("qpsk", 20.0)
    assert result["detections"]
    for detection in result["detections"]:
        for key in ("id", "method", "center_hz", "bandwidth_hz", "f_low_hz", "f_high_hz",
                    "t_start_s", "t_end_s", "power_dbfs", "snr_db", "session_id",
                    "confidence", "hopping", "sub_bands", "bin_count",
                    "occupied_f_low_hz", "occupied_f_high_hz"):
            assert key in detection
        assert detection["method"] == "energy"
        assert 0.0 <= detection["confidence"] <= 1.0
        assert detection["f_low_hz"] < detection["center_hz"] < detection["f_high_hz"]


def test_arrays_are_returned_for_plotting():
    _, _, arrays = scene("fm", 15.0)
    for key in ("frequency", "frame_time", "spectrogram_db", "spectrum_db",
                "threshold_db", "noise_floor_db", "detection_boxes"):
        assert key in arrays
    assert arrays["frequency"].size == NFFT
    assert arrays["spectrum_db"].size == NFFT
    assert arrays["spectrogram_db"].shape[1] == NFFT
    assert np.isfinite(arrays["spectrogram_db"]).all()


@pytest.mark.parametrize("config", [
    {"nfft": 8}, {"nfft": 8192}, {"nfft": 512.5}, {"threshold_db": -1.0},
    {"threshold_db": 90.0}, {"min_bandwidth_hz": -1.0}, {"max_detections": 0},
    {"max_detections": 300}, {"merge_bins": -1}, {"merge_bins": 999},
    {"band_threshold_db": 10.0}, {"unknown_key": 1},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        detect_signals(np.zeros(4096, dtype=complex), RATE, config)


def test_band_threshold_must_not_exceed_detection_threshold():
    with pytest.raises(ValueError):
        detect_signals(np.zeros(4096, dtype=complex), RATE,
                       {"threshold_db": 3.0, "band_threshold_db": 4.0})
    # 允许相等：此时“量带宽门限”退化为“检测门限”
    summary, _ = detect_signals(np.zeros(4096, dtype=complex), RATE,
                                {"threshold_db": 3.0, "band_threshold_db": 3.0})
    assert summary["config"]["band_threshold_db"] == 3.0


# ---------------------------------------------------------------------------
# 纯噪声：虚警率
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_pure_noise_has_no_false_alarm(seed):
    samples, _ = generate_iq(RATE, DURATION, [], noise={"power_dbfs": -10.0}, seed=seed)
    summary, _ = detect_signals(samples, RATE)
    assert summary["detections"] == []


def test_short_input_is_safe_and_empty_input_is_rejected():
    with pytest.raises(ValueError):
        detect_signals([], RATE)
    summary, _ = detect_signals([1e-9 + 0j], RATE)
    assert summary["detections"] == []


# ---------------------------------------------------------------------------
# 单信号：中心/带宽/带内 SNR 精度
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,bandwidth,offset,snr_db", [
    ("fm", 120e3, 150e3, 10.0),
    ("ssb", 60e3, -350e3, 10.0),
    ("ask2", 100e3, 250e3, 10.0),
    ("qpsk", 200e3, 0.0, 10.0),
    ("qam16", 300e3, -100e3, 10.0),
    ("qam64", 300e3, 100e3, 10.0),
    # AM 与 FH 的检出门限更高，理由见 test_am_sidebands_need_moderate_snr
    ("am", 80e3, -200e3, 15.0),
    ("fh_rc", 400e3, 0.0, 5.0),
    ("fh_video", 300e3, 0.0, 5.0),
])
def test_single_signal_estimates_match_truth(mode, bandwidth, offset, snr_db):
    result, truth, _ = scene(mode, snr_db, seed=2, bandwidth=bandwidth, offset=offset)
    metrics = evaluate_detections(truth, result["detections"])
    assert metrics["missed"] == 0
    assert metrics["false_alarm"] == 0
    pair = metrics["pairs"][0]
    assert centric(pair)
    assert abs(pair["bandwidth_relative_error"]) < 0.20
    assert abs(pair["snr_error_db"]) < 2.0


@pytest.mark.parametrize("mode", ["am", "fm", "ssb", "ask2", "qpsk", "qam16", "qam64"])
def test_centre_matches_occupied_interval(mode):
    """center_hz 与生成器真值的占用区间中点一致（SSB 单边带亦然）。"""
    result, truth, _ = scene(mode, 20.0, seed=3, bandwidth=80e3, offset=-180e3,
                             side="usb" if mode == "ssb" else None)
    metrics = evaluate_detections(truth, result["detections"])
    assert metrics["matched"] == 1
    assert abs(metrics["pairs"][0]["center_error_hz"]) < 2 * BIN_HZ


def test_am_sidebands_need_moderate_snr():
    """AM 载波主导：带内 SNR 10 dB 时边带压到噪声底附近，带宽不可测。

    这是能量检测的固有物理限制（载波比边带高约 11 dB，10 dB 带内 SNR 下
    边带 PSD 只比噪声底高 0.4 dB），不是实现缺陷：15 dB 起带宽误差 < 3%。
    该现象在 docs 与 benchmarks/detect_sweep.py 的成功率曲线中都有记录。
    """
    collapsed, truth, _ = scene("am", 10.0, seed=2, bandwidth=80e3, offset=-200e3)
    assert collapsed["detections"]          # 载波仍被检出
    assert truth[0]["bandwidth_hz"] / collapsed["detections"][0]["bandwidth_hz"] > 4

    measured, truth, _ = scene("am", 15.0, seed=2, bandwidth=80e3, offset=-200e3)
    metrics = evaluate_detections(truth, measured["detections"])
    assert metrics["missed"] == 0
    assert abs(metrics["pairs"][0]["bandwidth_relative_error"]) < 0.05


def test_detection_survives_at_low_inband_snr_for_low_papr_modes():
    """低 PAPR 宽带信号在 0 dB 带内 SNR 下仍可检出（窄带 ask2 更早失败）。"""
    result, truth, _ = scene("qpsk", 0.0, seed=4, bandwidth=200e3)
    metrics = evaluate_detections(truth, result["detections"])
    assert metrics["matched"] == 1
    assert metrics["false_alarm"] == 0


# ---------------------------------------------------------------------------
# 多信号与跳频
# ---------------------------------------------------------------------------

def test_three_signals_are_separated():
    specs = [{"mode": "qpsk", "offset": 100e3, "bandwidth": 200e3, "power_dbfs": -8.0},
             {"mode": "am", "offset": -220e3, "bandwidth": 80e3, "power_dbfs": -6.0},
             {"mode": "ssb", "offset": 300e3, "bandwidth": 60e3, "power_dbfs": -12.0,
              "side": "usb"}]
    samples, summary = generate_iq(RATE, 0.2, specs, noise={"snr_db": 18.0}, seed=3)
    result, _ = detect_signals(samples, RATE, {"nfft": NFFT})
    metrics = evaluate_detections(signal_truth(summary), result["detections"])
    assert (metrics["matched"], metrics["missed"], metrics["false_alarm"]) == (3, 0, 0)
    assert metrics["bandwidth_mape"] < 0.20
    assert metrics["snr_mae_db"] < 2.0


def test_simultaneous_adjacent_emitters_stay_separate():
    """同频段同时工作的两台发射机不能被“一段会话”规则并成一路。"""
    specs = [{"mode": "qpsk", "offset": 0.0, "bandwidth": 100e3, "power_dbfs": -8.0},
             {"mode": "qam16", "offset": 120e3, "bandwidth": 100e3, "power_dbfs": -8.0}]
    samples, summary = generate_iq(RATE, 0.3, specs, noise={"snr_db": 20.0}, seed=4)
    result, _ = detect_signals(samples, RATE, {"nfft": NFFT})
    assert len(result["detections"]) == 2
    assert all(not detection["hopping"] for detection in result["detections"])
    metrics = evaluate_detections(signal_truth(summary), result["detections"])
    assert metrics["false_alarm"] == 0 and metrics["missed"] == 0


@pytest.mark.parametrize("mode,bandwidth,offset,snr_db", [
    ("fh_rc", 400e3, 0.0, 5.0),
    ("fh_rc", 400e3, 0.0, 20.0),
    ("fh_video", 300e3, 0.0, 5.0),
    ("fh_video", 300e3, 0.0, 20.0),
])
def test_hopping_session_is_one_instance(mode, bandwidth, offset, snr_db):
    """跳频按“一段会话一实例”输出：一条 detection 覆盖全跳集占用带宽。"""
    result, truth, _ = scene(mode, snr_db, seed=1, bandwidth=bandwidth, offset=offset)
    metrics = evaluate_detections(truth, result["detections"])
    assert metrics["matched"] == 1
    assert metrics["false_alarm"] == 0
    pair = metrics["pairs"][0]
    assert centric(pair)
    assert abs(pair["bandwidth_relative_error"]) < 0.20
    assert abs(pair["snr_error_db"]) < 2.0


def test_hopping_truth_uses_visited_channels_only():
    """真值带宽只覆盖真正跳到的信道（跳序列是随机的）。"""
    _, summary = generate_iq(RATE, 0.2, [{"mode": "fh_rc", "offset": 0.0,
                                          "bandwidth": 400e3, "power_dbfs": -8.0}],
                             noise={"snr_db": 20.0}, seed=1)
    entry = summary["signals"][0]
    truth = signal_truth(summary)[0]
    visited = sorted(set(entry["hop_points"]))
    assert truth["visited_channels"] == len(visited)
    assert truth["f_low_hz"] == pytest.approx(visited[0] - entry["hop_bandwidth"] / 2)
    assert truth["f_high_hz"] == pytest.approx(visited[-1] + entry["hop_bandwidth"] / 2)
    assert truth["bandwidth_hz"] <= entry["bandwidth_actual"] + 1e-6
    assert truth["hopping"] is True


def test_hopping_is_flagged_when_sub_bands_are_resolved():
    """多个子带被并成一路时给出 hopping/sub_bands/session_id 标记。"""
    result, _, _ = scene("fh_video", 20.0, seed=6, bandwidth=300e3)
    detection = result["detections"][0]
    if detection["sub_bands"] > 1:
        assert detection["hopping"] is True
        assert detection["session_id"] == detection["id"]
    else:
        assert detection["hopping"] is False
        assert detection["session_id"] is None


def test_min_bandwidth_suppresses_narrow_spurs():
    samples, _ = generate_iq(RATE, DURATION,
                             [{"mode": "fm", "offset": 0.0, "bandwidth": 200e3,
                               "power_dbfs": -8.0}], noise={"snr_db": 20.0}, seed=5)
    narrow, _ = detect_signals(samples, RATE, {"nfft": NFFT, "min_bandwidth_hz": 400e3})
    assert narrow["detections"] == []
    wide, _ = detect_signals(samples, RATE, {"nfft": NFFT, "min_bandwidth_hz": 50e3})
    assert len(wide["detections"]) == 1


def test_min_duration_and_max_detections_are_respected():
    samples, _ = generate_iq(RATE, DURATION,
                             [{"mode": "qpsk", "offset": 0.0, "bandwidth": 200e3,
                               "power_dbfs": -8.0}], noise={"snr_db": 20.0}, seed=6)
    none, _ = detect_signals(samples, RATE, {"nfft": NFFT, "min_duration_s": DURATION})
    assert none["detections"] == []
    capped, _ = detect_signals(samples, RATE, {"nfft": NFFT, "max_detections": 1})
    assert len(capped["detections"]) <= 1


def test_threshold_controls_sensitivity():
    samples, summary = generate_iq(RATE, DURATION,
                                   [{"mode": "qpsk", "offset": 0.0, "bandwidth": 200e3,
                                     "power_dbfs": -8.0}], noise={"snr_db": 0.0}, seed=7)
    truth = signal_truth(summary)
    low, _ = detect_signals(samples, RATE, {"nfft": NFFT, "threshold_db": 1.0})
    high, _ = detect_signals(samples, RATE, {"nfft": NFFT, "threshold_db": 20.0})
    assert evaluate_detections(truth, low["detections"])["matched"] >= \
        evaluate_detections(truth, high["detections"])["matched"]
    assert high["detections"] == []


def test_detection_is_deterministic():
    samples, _ = generate_iq(RATE, 0.2, [{"mode": "qam16", "offset": 50e3,
                                          "bandwidth": 200e3, "power_dbfs": -8.0}],
                             noise={"snr_db": 12.0}, seed=8)
    first, _ = detect_signals(samples, RATE)
    second, _ = detect_signals(samples, RATE)
    assert first == second


# ---------------------------------------------------------------------------
# evaluation 模块
# ---------------------------------------------------------------------------

def test_signal_truth_of_empty_and_missing_summary():
    assert signal_truth(None) == []
    assert signal_truth({}) == []
    assert signal_truth({"signals": [{"mode": "fm"}]}) == []


def test_signal_truth_scales_are_rounded_floats():
    _, summary = generate_iq(RATE, 0.1, [{"mode": "fm", "offset": 12.5, "bandwidth": 100e3,
                                          "power_dbfs": -6.0}], noise={"snr_db": 10.0}, seed=1)
    entry = signal_truth(summary)[0]
    assert isinstance(entry["center_hz"], float)
    assert entry["t_start_s"] == 0.0 and entry["t_end_s"] == pytest.approx(0.1)
    assert entry["session_id"] == 0


def test_match_detections_is_greedy_and_one_to_one():
    truth = [{"center_hz": 0.0, "bandwidth_hz": 1000.0, "snr_inband_db": 10.0}]
    detections = [{"id": 1, "center_hz": 10e3, "bandwidth_hz": 1000.0, "snr_db": 10.0},
                  {"id": 2, "center_hz": 100.0, "bandwidth_hz": 1000.0, "snr_db": 11.0}]
    assert match_detections(truth, detections) == [(0, 1)]


def test_evaluate_detections_handles_empty_sides():
    metrics = evaluate_detections([], [])
    assert metrics["f1"] is None and metrics["precision"] is None
    assert metrics["center_mae_hz"] is None and metrics["bandwidth_mape"] is None
    only_false = evaluate_detections([], [{"id": 1, "center_hz": 0.0, "bandwidth_hz": 1.0}])
    assert (only_false["false_alarm"], only_false["recall"], only_false["f1"]) == (1, None, 0.0)


def test_evaluate_detections_reports_pairs_with_truth_reference():
    truth = [{"center_hz": 1000.0, "bandwidth_hz": 100.0, "snr_inband_db": 12.0}]
    detections = [{"id": 7, "center_hz": 1005.0, "bandwidth_hz": 110.0, "snr_db": 12.5}]
    metrics = evaluate_detections(truth, detections)
    pair = metrics["pairs"][0]
    assert pair["truth_index"] == 0 and pair["detection_id"] == 7
    assert pair["center_error_hz"] == pytest.approx(5.0)
    assert pair["bandwidth_relative_error"] == pytest.approx(0.1)
    assert pair["snr_error_db"] == pytest.approx(0.5)
    assert pair["truth_bandwidth_hz"] == pytest.approx(100.0)


def test_classification_metrics_confusion_and_macro_f1():
    truth = ["qpsk", "qpsk", "am", "fm"]
    predicted = ["qpsk", "am", "am", "am"]
    metrics = classification_metrics(truth, predicted, labels=["qpsk", "am", "fm"])
    assert metrics["labels"] == ["qpsk", "am", "fm"]
    assert metrics["total"] == 4
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["confusion"][0] == [1, 1, 0]
    per_class = {row["label"]: row for row in metrics["per_class"]}
    assert per_class["qpsk"]["precision"] == pytest.approx(1.0)
    assert per_class["qpsk"]["recall"] == pytest.approx(0.5)
    assert per_class["fm"]["recall"] == pytest.approx(0.0)
    assert metrics["macro_f1"] == pytest.approx(
        np.mean([row["f1"] for row in metrics["per_class"]]))


def test_classification_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        classification_metrics(["am"], ["am", "fm"])


def test_service_detect_run_carries_truth_and_metrics(tmp_path):
    """服务层 detect 任务：跑一次即可回读结果，且带真值与误差指标。"""
    from signal_analysis.services import execute
    from signal_analysis.storage import Workspace

    generated = execute({"workspace": str(tmp_path), "action": "generate",
                         "sample_rate": RATE, "duration": 0.2, "seed": 4,
                         "noise": {"snr_db": 15.0, "bandwidth": RATE},
                         "signals": [{"mode": "qam16", "offset": -150e3,
                                      "bandwidth": 120e3, "power_dbfs": -10.0}]})
    run = execute({"workspace": str(tmp_path), "action": "detect",
                   "asset_id": generated["id"], "config": {"nfft": NFFT}})
    assert run["kind"] == "detect" and run["contract"] == "detect_result_v1"
    assert run["asset_id"] == generated["id"]
    assert run["truth"][0]["mode"] == "qam16"
    assert run["metrics"]["matched"] == 1 and run["metrics"]["false_alarm"] == 0
    assert run["metrics"]["center_mae_hz"] < 2 * BIN_HZ
    # 结果写盘后可以按 run_id 重新读回，数组与检测框一致
    store = Workspace(tmp_path)
    reloaded = store.get_run(run["run_id"])
    boxes = np.load(store.root / reloaded["plots_path"], allow_pickle=False)["detection_boxes"]
    assert boxes.shape == (len(reloaded["summary"]["detections"]), 4)


def test_service_detect_without_generator_truth(tmp_path):
    """导入/演示数据没有真值：只给检测结果，不编造误差指标。"""
    from signal_analysis.services import execute

    demo = execute({"workspace": str(tmp_path), "action": "demo", "sample_rate": 48000.0,
                    "count": 8192})
    run = execute({"workspace": str(tmp_path), "action": "detect", "asset_id": demo["id"]})
    assert "truth" not in run and "metrics" not in run
    assert run["summary"]["detections"]


def test_cli_detect_prints_contract(tmp_path, capsys):
    import json
    from signal_analysis.cli import main

    root = str(tmp_path / "store")
    assert main(["--workspace", root, "demo", "--sample-rate", "48000", "--count", "8192"]) == 0
    asset_id = json.loads(capsys.readouterr().out)["id"]
    assert main(["--workspace", root, "detect", asset_id, "--nfft", "256",
                 "--threshold-db", "6", "--min-bandwidth", "200"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "detect"
    assert payload["summary"]["config"]["threshold_db"] == pytest.approx(6.0)
    assert payload["summary"]["config"]["band_threshold_db"] == pytest.approx(3.0)
    assert payload["summary"]["config"]["min_bandwidth_hz"] == pytest.approx(200.0)
    assert json.dumps(payload, allow_nan=False)
