"""符号率估计：线性数字调制的包络谱线谱与检测服务接线。"""
import numpy as np
import pytest

from signal_analysis.algorithms.detection.symbol_rate import estimate_symbol_rate
from signal_analysis.core_api import detect_signals, generate_iq

RATE = 1e6
DURATION = 0.2
NFFT = 512


def _band(rate=RATE, snr_db=15.0, mode="qpsk", bandwidth=200e3):
    spec = {"mode": mode, "offset": 0.0, "bandwidth": bandwidth,
            "power_dbfs": -8.0, "alpha": 0.35, "pulse": "rrc"}
    samples, summary = generate_iq(rate, DURATION, [spec],
                                   noise={"snr_db": snr_db}, seed=7)
    detection, _ = detect_signals(samples, rate, {"nfft": NFFT})
    return samples, summary, detection["detections"][0]


def _estimate(samples, detection, rate=RATE):
    low = detection.get("occupied_f_low_hz")
    high = detection.get("occupied_f_high_hz")
    bandwidth = (high - low if low is not None and high is not None and high > low
                 else detection["bandwidth_hz"])
    return estimate_symbol_rate(samples, rate, detection["center_hz"], bandwidth,
                                start_s=detection["t_start_s"],
                                end_s=detection["t_end_s"])


@pytest.mark.parametrize("mode", ["ask2", "qpsk", "qam16", "qam64"])
def test_linear_digital_symbol_rate_from_envelope_spectrum(mode):
    samples, summary, detection = _band(mode=mode)
    result = _estimate(samples, detection)
    truth = summary["signals"][0]["symbol_rate"]
    assert result["symbol_rate_applicable"] is True
    assert result["symbol_rate_reason"] is None
    assert result["symbol_rate_method"] == "envelope_spectrum_v1"
    assert result["symbol_rate_baud"] == pytest.approx(truth, rel=0.02)
    assert 0.0 < result["symbol_rate_confidence"] <= 1.0


@pytest.mark.parametrize("mode", ["am", "fm", "ssb"])
def test_analog_signal_is_not_reported_as_applicable(mode):
    samples, _, detection = _band(mode=mode)
    result = _estimate(samples, detection)
    assert result["symbol_rate_applicable"] is False


def test_invalid_or_short_input_is_explicitly_unavailable():
    empty = estimate_symbol_rate(np.zeros(0, dtype=complex), RATE, 0.0, 0.0)
    assert empty["symbol_rate_baud"] is None
    assert empty["symbol_rate_reason"] == "invalid_band"

    short = estimate_symbol_rate(np.zeros(32, dtype=complex), RATE, 0.0, 100e3)
    assert short["symbol_rate_reason"] == "insufficient_samples"

    # 恒包络单音：混频后包络为常数，没有线谱峰
    no_peak = estimate_symbol_rate(np.ones(4096, dtype=complex), RATE, 0.0, 100e3)
    assert no_peak["symbol_rate_baud"] is None
    assert no_peak["symbol_rate_reason"] == "no_peak"


def test_evaluate_detections_scores_only_applicable_symbol_rate():
    from signal_analysis.evaluation import evaluate_detections

    truth = [{"center_hz": 0.0, "bandwidth_hz": 100e3, "snr_inband_db": 10.0,
              "symbol_rate_baud": 50e3}]
    detection = [{"id": 1, "center_hz": 0.0, "bandwidth_hz": 100e3, "snr_db": 10.0,
                  "symbol_rate_baud": 55e3, "symbol_rate_applicable": True}]
    metrics = evaluate_detections(truth, detection)
    assert metrics["symbol_rate_scored"] == 1
    assert metrics["symbol_rate_mape"] == pytest.approx(0.1)
    assert metrics["pairs"][0]["symbol_rate_relative_error"] == pytest.approx(0.1)

    detection[0]["symbol_rate_applicable"] = False
    metrics = evaluate_detections(truth, detection)
    assert metrics["symbol_rate_scored"] == 0
    assert metrics["symbol_rate_mape"] is None


def test_adoption_writes_only_applicable_symbol_rate_and_keeps_previous(tmp_path):
    from signal_analysis.data import Workspace
    from signal_analysis.data.adoption import adopt_detections

    workspace = Workspace(tmp_path / "ws")
    samples, summary = generate_iq(
        RATE, DURATION,
        [{"mode": "qpsk", "offset": 0.0, "bandwidth": 50e3, "power_dbfs": -8.0}],
        noise={"snr_db": 15.0}, seed=11)
    asset = workspace.add_samples(samples.astype(np.complex64), RATE, "符号率采纳",
                                  "generated:iq_qpsk_v1", metadata={"generation": summary})
    target = workspace.list_targets(asset["id"])[0]
    base = {"id": 1, "center_hz": 0.0, "bandwidth_hz": 50e3,
            "f_low_hz": -25e3, "f_high_hz": 25e3,
            "t_start_s": 0.0, "t_end_s": DURATION,
            "snr_db": 15.0, "power_dbfs": -8.0}
    adopt_detections(workspace, asset["id"],
                     [{**base, "symbol_rate_baud": 100e3,
                       "symbol_rate_applicable": True}], "run-a")
    current = workspace.current_target_version(target["id"])
    assert current["symbol_rate_baud"] == pytest.approx(100e3)

    adopt_detections(workspace, asset["id"],
                     [{**base, "symbol_rate_baud": 12e3,
                       "symbol_rate_applicable": False}], "run-b")
    current = workspace.current_target_version(target["id"])
    assert current["symbol_rate_baud"] == pytest.approx(100e3)


def test_annotate_symbol_rates_covers_ai_baseline_too():
    from signal_analysis.services.detection import _annotate_symbol_rates

    samples, summary = generate_iq(
        RATE, DURATION,
        [{"mode": "qam64", "offset": 0.0, "bandwidth": 200e3, "power_dbfs": -8.0}],
        noise={"snr_db": 15.0}, seed=13)
    detection = {"id": 1, "center_hz": 0.0, "bandwidth_hz": 200e3,
                 "occupied_f_low_hz": -100e3, "occupied_f_high_hz": 100e3,
                 "t_start_s": 0.0, "t_end_s": DURATION, "hopping": False}
    payload = {"detections": [dict(detection)],
               "baseline": {"detections": [dict(detection)]}}
    _annotate_symbol_rates(payload, samples, RATE)
    truth = summary["signals"][0]["symbol_rate"]
    for item in (payload["detections"][0], payload["baseline"]["detections"][0]):
        assert item["symbol_rate_applicable"] is True
        assert item["symbol_rate_baud"] == pytest.approx(truth, rel=0.02)


def test_service_detect_annotates_each_detection_and_scores_symbol_rate(tmp_path):
    from signal_analysis.services import execute

    generated = execute({"workspace": str(tmp_path), "action": "generate",
                         "sample_rate": RATE, "duration": DURATION, "seed": 4,
                         "noise": {"snr_db": 15.0, "bandwidth": RATE},
                         "signals": [{"mode": "qam16", "offset": -150e3,
                                      "bandwidth": 120e3, "power_dbfs": -10.0}]})
    run = execute({"workspace": str(tmp_path), "action": "detect",
                   "asset_id": generated["id"], "config": {"nfft": NFFT}})
    detection = run["summary"]["detections"][0]
    truth_rate = run["truth"][0]["symbol_rate_baud"]
    assert detection["symbol_rate_applicable"] is True
    assert detection["symbol_rate_baud"] == pytest.approx(truth_rate, rel=0.02)
    assert run["symbol_rate_method"] == "envelope_spectrum_v1"
    assert run["metrics"]["symbol_rate_scored"] == 1
    assert run["metrics"]["symbol_rate_mape"] < 0.02
