"""逐跳参数估计（契约 ``fh_hops_v1``，算法 ``hop_track_v1``）的验收测试。

与会话级能量检测（``detect_result_v1``）分成两条契约：那里一条链路只给一段频带，
这里给每一跳的中心频率、单跳带宽、驻留时间与带内信噪比，再按时间连续性聚成会话，
补出跳速、跳频点数与占空比。断言全部对着生成器的逐跳真值（``hop_truth``），
只在真值不存在的量（会话聚类）上使用结构性断言。

已知算法限制（不对应缺陷，测试里显式固定其表现）：

* 连续两跳复用同一频点时，两段驻留在时频图上无法与一段长驻留区分；
* ``fh_video`` 每跳末尾的空闲（OFDM 整块填充）会让实测驻留短于跳周期；
* 单跳瞬时带宽超过信道间隔时（``fh_rc`` 矩形脉冲的旁瓣），多条信道会退化成一段宽驻留；
* 漏检较多时会话数只是上界（一部发射机可能被拆成两个会话）。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_analysis.core_api import detect_hops, detect_signals, generate_iq  # noqa: E402
from signal_analysis.evaluation import (CONTRACT_VERSION, HOP_ALGORITHM,  # noqa: E402
                                        HOP_CONTRACT, evaluate_detections, hop_truth,
                                        signal_truth)

RATE = 1e6
NFFT = 512
BIN_HZ = RATE / NFFT
HOP_BW = 20e3
CHANNELS = [-300e3, -200e3, -100e3, 0.0, 100e3]
HOP_RATE = 50.0
DURATION = 0.2
TRUTH_HOPS = int(round(HOP_RATE * DURATION))


def plan(channels, hop_bw=HOP_BW, hop_rate=HOP_RATE, mode="fh_video", **extra):
    """Build a generator spec with an explicit channel set (hop order stays random)."""
    span = max(channels) - min(channels)
    return {"mode": mode, "offset": 0.0, "bandwidth": span + hop_bw, "power_dbfs": -8.0,
            "hop_points": list(channels), "hop_bandwidth": hop_bw, "hop_rate": hop_rate, **extra}


def scene(specs=None, config=None, snr_db=20.0, seed=3, duration=DURATION,
          with_sessions=True):
    """Generate IQ, run the hop tracker, return ``(summary, arrays, generation)``."""
    specs = [plan(CHANNELS)] if specs is None else specs
    samples, generation = generate_iq(RATE, duration, specs,
                                      noise={"enabled": True, "bandwidth": RATE,
                                             "snr_db": snr_db}, seed=seed)
    summary, arrays = detect_hops(samples, RATE, config, with_sessions)
    return summary, arrays, generation


# ---------------------------------------------------------------------------
# 契约与配置
# ---------------------------------------------------------------------------

def test_contract_keys_are_frozen():
    summary, _ = detect_hops(np.zeros(8192, dtype=complex), RATE)
    assert summary["contract"] == "fh_hops_v1"
    assert summary["algorithm"] == "hop_track_v1"
    assert summary["snr_definition"] == "inband_snr_v1"
    assert summary["frequency_reference"] == "baseband_offset"
    for key in ("sample_rate_hz", "sample_count", "duration_s", "nfft", "hop_samples",
                "frame_count", "config", "freq_resolution_hz", "frame_interval_s",
                "noise_floor_dbfs_per_hz", "threshold_dbfs_per_hz", "dwell_limit_s",
                "hop_rate_limit_hz", "resolvable", "reason", "transition_frames",
                "hops", "sessions"):
        assert key in summary
    assert sorted(summary["config"]) == ["max_gap_bins", "max_gap_frames", "max_hops",
                                         "merge_bins", "min_bandwidth_hz", "min_dwell_s",
                                         "nfft", "smooth_frames", "threshold_db",
                                         "transition_ratio"]
    assert summary["hops"] == [] and summary["sessions"] == []
    assert summary["resolvable"] is False
    assert isinstance(summary["reason"], str) and summary["reason"]


def test_hop_and_session_keys_are_frozen_and_json_safe():
    summary, _, _ = scene()
    assert summary["hops"] and summary["sessions"]
    ids = {session["session_id"] for session in summary["sessions"]}
    for hop in summary["hops"]:
        for key in ("id", "session_id", "center_hz", "centroid_hz", "bandwidth_hz",
                    "f_low_hz", "f_high_hz", "t_start_s", "t_end_s", "dwell_s",
                    "power_dbfs", "snr_db", "confidence", "frame_count", "bin_count",
                    "band_nfft", "mask_f_low_hz", "mask_f_high_hz", "mask_bandwidth_hz"):
            assert key in hop
        assert hop["f_low_hz"] < hop["center_hz"] < hop["f_high_hz"]
        assert hop["mask_f_low_hz"] <= hop["f_low_hz"] <= hop["f_high_hz"] <= hop["mask_f_high_hz"]
        assert 0.0 <= hop["confidence"] <= 1.0
        assert hop["t_end_s"] > hop["t_start_s"] > -1e-9
        assert hop["session_id"] in ids
    for session in summary["sessions"]:
        for key in ("session_id", "hop_count", "sequence", "hop_frequencies_hz",
                    "channel_spacing_hz", "hop_span_hz", "hop_bandwidth_hz", "hop_period_s",
                    "hop_rate_hz", "duty_cycle", "dwell_median_s", "dwell_min_s",
                    "dwell_max_s", "center_hz", "bandwidth_hz", "f_low_hz", "f_high_hz",
                    "t_start_s", "t_end_s", "power_dbfs", "snr_db", "session_detection_id"):
            assert key in session
        assert session["hop_count"] == sum(1 for hop in summary["hops"]
                                           if hop["session_id"] == session["session_id"])
        # hop_frequencies_hz 是同一信道多次访问的聚类中心，sequence 是逐跳实测中心：
        # 前者必须落在后者的一个频点邻域内，且去重后的点数不多于访问次数
        assert len(session["hop_frequencies_hz"]) <= len(session["sequence"])
        for channel in session["hop_frequencies_hz"]:
            assert min(abs(channel - value) for value in session["sequence"]) <= 2 * BIN_HZ
        assert session["dwell_min_s"] <= session["dwell_median_s"] <= session["dwell_max_s"]
    assert json.dumps(summary, ensure_ascii=False, allow_nan=False)


def test_arrays_are_returned_for_plotting():
    summary, arrays, _ = scene()
    for key in ("frequency", "frame_time", "spectrogram_db", "spectrogram_raw_db",
                "spectrum_db", "spectrum_median_db", "threshold_db", "noise_floor_db",
                "hop_boxes", "hop_id", "hop_session_id", "hop_snr_db", "hop_power_dbfs"):
        assert key in arrays
    assert arrays["frequency"].size == summary["nfft"]
    assert arrays["spectrogram_db"].shape == (summary["frame_count"], summary["nfft"])
    assert arrays["spectrogram_db"].dtype == np.float32
    assert np.isfinite(arrays["spectrogram_db"]).all()
    assert arrays["hop_boxes"].shape == (len(summary["hops"]), 4)
    assert arrays["hop_id"].size == len(summary["hops"])
    assert arrays["hop_session_id"].size == len(summary["hops"])
    # 框就是逐跳字段本身，界面不需要再算一遍
    for hop, box in zip(summary["hops"], arrays["hop_boxes"]):
        assert tuple(box) == (hop["f_low_hz"], hop["f_high_hz"], hop["t_start_s"], hop["t_end_s"])


@pytest.mark.parametrize("config", [
    {"nfft": 8}, {"nfft": 8192}, {"nfft": 512.5}, {"nfft": True},
    {"threshold_db": -1.0}, {"threshold_db": 90.0},
    {"smooth_frames": 0}, {"smooth_frames": 100},
    {"min_bandwidth_hz": -1.0}, {"min_bandwidth_hz": RATE * 2},
    {"min_dwell_s": -1.0}, {"min_dwell_s": DURATION * 2},
    {"merge_bins": -1}, {"merge_bins": 999},
    {"max_gap_frames": -1}, {"max_gap_frames": 99},
    {"max_gap_bins": -1}, {"max_gap_bins": 999},
    {"transition_ratio": 1.0}, {"transition_ratio": 9.0},
    {"max_hops": 0}, {"max_hops": 999}, {"unknown_key": 1},
])
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        detect_hops(np.zeros(8192, dtype=complex), RATE, config)


def test_config_must_be_a_dict():
    with pytest.raises(ValueError, match="逐跳配置"):
        detect_hops(np.zeros(8192, dtype=complex), RATE, ["nfft"])


def test_defaults_resolve_from_rate_and_duration():
    summary, _ = detect_hops(np.zeros(20000, dtype=complex), RATE, None, False)
    config = summary["config"]
    assert config["nfft"] == NFFT and config["threshold_db"] == pytest.approx(6.0)
    assert config["smooth_frames"] == 4 and config["max_hops"] == 256
    assert config["min_bandwidth_hz"] == pytest.approx(3 * BIN_HZ)
    assert config["min_dwell_s"] == pytest.approx(4 * NFFT / RATE)
    assert config["merge_bins"] == 1
    assert config["max_gap_bins"] == config["merge_bins"]
    # 可分辨门限走**实际帧间距**（记录过长时 _stft_psd 会把步长拉到 nfft 之上），
    # 最短驻留过滤走 nfft；两者在常规记录下由 nfft/2 的默认步长保证不冲突。
    assert summary["hop_samples"] == NFFT // 2
    assert summary["dwell_limit_s"] == pytest.approx(
        4 * summary["hop_samples"] / RATE)
    assert summary["hop_rate_limit_hz"] == pytest.approx(1.0 / summary["dwell_limit_s"])
    assert "baseline" not in summary


# ---------------------------------------------------------------------------
# 逐跳精度：与生成器真值对照
# ---------------------------------------------------------------------------

def test_hop_count_and_parameters_match_truth():
    summary, _, generation = scene()
    truth = hop_truth(generation)
    assert len(truth) == TRUTH_HOPS
    metrics = evaluate_detections(truth, summary["hops"], contract=HOP_CONTRACT)
    assert metrics["contract"] == "fh_hops_v1"
    assert metrics["precision"] >= 0.9
    assert metrics["recall"] >= 0.8
    assert metrics["center_mae_hz"] <= 2 * BIN_HZ
    assert metrics["bandwidth_mape"] < 0.15
    assert metrics["snr_mae_db"] < 3.5
    # 真值一跳一条：单跳带宽与驻留都直接来自生成器计划
    for entry in truth:
        assert entry["bandwidth_hz"] == pytest.approx(HOP_BW)
        assert entry["dwell_s"] == pytest.approx(DURATION / TRUTH_HOPS)


def test_session_hop_rate_and_dwell_match_truth():
    summary, _, _ = scene()
    assert summary["resolvable"] is True and summary["reason"] is None
    assert len(summary["sessions"]) == 1
    session = summary["sessions"][0]
    assert session["hop_rate_hz"] == pytest.approx(HOP_RATE, rel=0.10)
    assert session["dwell_median_s"] == pytest.approx(DURATION / TRUTH_HOPS, rel=0.15)
    assert session["hop_period_s"] == pytest.approx(1 / session["hop_rate_hz"], rel=1e-3)
    assert session["duty_cycle"] == pytest.approx(session["dwell_median_s"]
                                                 / session["hop_period_s"], rel=0.05)
    assert session["t_start_s"] >= 0.0 and session["t_end_s"] <= DURATION + 1e-9


def test_session_channels_are_planned_channels_only():
    summary, _, generation = scene()
    session = summary["sessions"][0]
    used = set(generation["signals"][0]["hop_points"])
    assert 2 <= len(session["hop_frequencies_hz"]) <= len(CHANNELS)
    for value in session["hop_frequencies_hz"]:
        assert min(abs(value - point) for point in used) <= 2 * BIN_HZ
    assert session["channel_spacing_hz"] >= HOP_BW
    assert session["hop_span_hz"] <= max(CHANNELS) - min(CHANNELS) + 2 * BIN_HZ
    assert session["hop_bandwidth_hz"] == pytest.approx(HOP_BW, rel=0.15)
    # 会话带宽 = 跳频跨度 + 单跳带宽（最外侧信道的边缘）
    assert session["bandwidth_hz"] == pytest.approx(session["hop_span_hz"]
                                                   + session["hop_bandwidth_hz"], rel=0.2)


def test_hop_truth_union_equals_session_level_truth_band():
    _, _, generation = scene()
    hops = hop_truth(generation)
    session = signal_truth(generation)[0]
    assert min(entry["f_low_hz"] for entry in hops) == pytest.approx(session["f_low_hz"])
    assert max(entry["f_high_hz"] for entry in hops) == pytest.approx(session["f_high_hz"])
    assert [entry["index"] for entry in hops] == list(range(TRUTH_HOPS))
    assert all(entry["hop_count"] == TRUTH_HOPS for entry in hops)
    assert all(entry["t_start_s"] < entry["t_end_s"] for entry in hops)


def test_hop_truth_is_empty_without_generator_truth():
    assert hop_truth(None) == []
    assert hop_truth({}) == []
    samples, generation = generate_iq(RATE, 0.05, [{"mode": "qpsk", "offset": 0.0,
                                                    "bandwidth": 100e3,
                                                    "power_dbfs": -8.0}],
                                      noise={"snr_db": 20.0}, seed=1)
    assert hop_truth(generation) == []
    summary, _ = detect_hops(samples, RATE)
    # 没有逐跳真值不等于没有结果：连续波在逐跳口径下就是一段长驻留
    assert len(summary["hops"]) == 1
    assert summary["resolvable"] is True  # 不是跳频样式，但时间分辨率本身足够


def test_continuous_signal_becomes_a_single_dwell():
    """非跳频连续波在逐跳口径下就是“一段长驻留”，会话参数退化为单跳。"""
    samples, generation = generate_iq(RATE, DURATION, [{"mode": "qpsk", "offset": 0.0,
                                                        "bandwidth": 200e3,
                                                        "power_dbfs": -8.0}],
                                      noise={"snr_db": 20.0}, seed=2)
    summary, _ = detect_hops(samples, RATE)
    assert len(summary["hops"]) == 1
    hop = summary["hops"][0]
    assert hop["center_hz"] == pytest.approx(0.0, abs=2 * BIN_HZ)
    assert hop["bandwidth_hz"] == pytest.approx(200e3, rel=0.2)
    assert hop["dwell_s"] == pytest.approx(DURATION, rel=0.05)
    session = summary["sessions"][0]
    assert session["hop_count"] == 1 and session["hop_period_s"] is None
    assert session["hop_rate_hz"] is None and session["duty_cycle"] is None
    assert hop_truth(generation) == []


def test_transition_frames_are_counted_without_faking_hops():
    """跳变帧频谱被展宽：计入 transition_frames，且细化后单跳带宽不得被泄漏撑大。"""
    summary, _, _ = scene()
    assert summary["transition_frames"] >= 1
    assert len(summary["hops"]) <= TRUTH_HOPS + 2
    assert summary["hops"][0]["band_nfft"] >= summary["nfft"]
    assert summary["hops"][0]["bandwidth_hz"] < 1.5 * HOP_BW


def test_wide_instantaneous_bandwidth_degrades_to_one_dwell():
    """单跳瞬时带宽超过信道间隔时（fh_rc 矩形脉冲旁瓣）整段退化成少数宽驻留。

    这是能量域跟踪的固有分辨率限制：每跳谱掩码比 hop_bandwidth 宽 1.7～2 倍，
    相邻信道在跳变帧里被焊在一起。测试固定该行为，避免被误判为回归。
    """
    channels = [-60e3, 0.0, 60e3]
    summary, _, _ = scene([plan(channels, hop_bw=40e3, hop_rate=40.0, mode="fh_rc")])
    assert len(summary["hops"]) < 8  # 真值 8 跳，被旁瓣糊成少数几段
    assert len(summary["sessions"]) == 1
    assert summary["sessions"][0]["bandwidth_hz"] > 100e3


def test_narrow_channel_grid_is_not_merged_into_one_band():
    """信道间隔 30 kHz、单跳 10 kHz：闭运算不得把相邻信道焊成一段宽驻留。"""
    channels = [-30e3, 0.0, 30e3]
    summary, _, generation = scene([plan(channels, hop_bw=10e3)], snr_db=25.0)
    truth = hop_truth(generation)
    metrics = evaluate_detections(truth, summary["hops"], contract=HOP_CONTRACT)
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    visited = {round(entry["center_hz"]) for entry in truth}
    assert len(visited) == 2  # 三个频点里本段记录真值只访问了两个
    assert len(summary["sessions"][0]["hop_frequencies_hz"]) == len(visited)
    for hop in summary["hops"]:
        assert hop["bandwidth_hz"] < 1.6 * 10e3  # 未与相邻信道焊接
        assert min(abs(hop["center_hz"] - value) for value in visited) <= 2 * BIN_HZ


# ---------------------------------------------------------------------------
# 会话聚类与可分辨性
# ---------------------------------------------------------------------------

def test_two_emitters_are_split_into_two_sessions():
    """两部同时工作的跳频发射机：网格分离度足够时按频点集合拆成两个会话。

    同时发射会让每一帧都出现多条游程，此时跟踪器只允许「频域严格重叠」续链，
    因此漏检率明显高于单机场景（本用例只要求召回 ≥ 0.75，见算法文档的已知限制）。
    """
    specs = [plan([-400e3, -300e3, -200e3], hop_bw=15e3, hop_rate=40.0, offset=-300e3),
             plan([200e3, 260e3, 320e3], hop_bw=12e3, hop_rate=70.0, offset=260e3)]
    summary, _, generation = scene(specs, snr_db=20.0, seed=7)
    truth = hop_truth(generation)
    assert len(truth) == 22  # 8 跳（40 Hz）+ 14 跳（70 Hz），0.2 s
    metrics = evaluate_detections(truth, summary["hops"], contract=HOP_CONTRACT)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] >= 0.75
    assert len(summary["sessions"]) == 2
    rates = sorted(session["hop_rate_hz"] for session in summary["sessions"])
    assert rates[0] == pytest.approx(40.0, rel=0.15)
    assert rates[1] == pytest.approx(70.0, rel=0.15)
    grids = {tuple(round(value / 1e3) for value in session["hop_frequencies_hz"])
             for session in summary["sessions"]}
    assert grids == {(-400, -300, -200), (200, 260, 320)}
    widths = sorted(session["hop_bandwidth_hz"] for session in summary["sessions"])
    assert widths[0] == pytest.approx(12e3, rel=0.15)
    assert widths[1] == pytest.approx(15e3, rel=0.15)
    assert all(hop["session_id"] is not None for hop in summary["hops"])


def test_session_detection_is_linked_to_the_energy_baseline():
    summary, _, _ = scene()
    baseline = summary["baseline"]
    assert baseline["contract"] == CONTRACT_VERSION == "detect_result_v1"
    assert baseline["algorithm"] == "energy_detect_v1"
    assert len(baseline["detections"]) == 1  # 会话级口径：整条链路只算一个目标
    assert summary["sessions"][0]["session_detection_id"] == baseline["detections"][0]["id"]
    assert "detections" not in summary  # 两条契约不互相污染字段


def test_without_sessions_skips_the_baseline():
    summary, _, _ = scene(with_sessions=False)
    assert "baseline" not in summary
    assert summary["sessions"] and summary["hops"]


def test_pure_noise_reports_no_hops_and_a_reason():
    samples, _ = generate_iq(RATE, DURATION, [], noise={"power_dbfs": -10.0}, seed=7)
    summary, _ = detect_hops(samples, RATE)
    assert summary["hops"] == [] and summary["sessions"] == []
    assert summary["resolvable"] is False
    assert summary["reason"]


def test_fast_hopping_on_a_long_record_is_declared_unresolvable():
    """超长记录里帧间距被拉大到 nfft 之上：驻留不足 4 帧时必须显式报警。"""
    summary, _, _ = scene([plan(CHANNELS, hop_rate=200.0)], duration=1.0)
    assert summary["hop_samples"] > summary["nfft"] // 2
    assert summary["config"]["min_dwell_s"] == pytest.approx(4 * NFFT / RATE)
    assert summary["dwell_limit_s"] == pytest.approx(
        4 * summary["hop_samples"] / RATE)
    assert summary["dwell_limit_s"] > summary["config"]["min_dwell_s"]
    assert summary["hops"]  # 融合后仍有结果，但必须被标记为不可分辨
    assert summary["resolvable"] is False
    assert "分辨" in summary["reason"]


def test_grid_too_coarse_fuses_hops_even_though_resolvable_is_true():
    """粗网格 + 极快跳频：结果被融合成少量长驻留，而 resolvable 仍为 True。

    这是必须写进文档的限制：resolvable 只比较「过滤器允许的最短驻留」与检出驻留，
    并不读取真值。真值 2.5 ms 远短于 16.4 ms 的过滤门限时本该拒答，实际却只报出
    3 段融合驻留，而 resolvable 仍为 True。
    """
    summary, _, generation = scene([plan(CHANNELS, hop_rate=400.0, mode="fh_rc")],
                                   config={"nfft": 4096})
    truth = hop_truth(generation)
    assert len(truth) == 80 and min(entry["dwell_s"] for entry in truth) < 5e-3
    assert summary["resolvable"] is True  # 门限基于帧间距而非真值，故仍判定"可分辨"
    assert len(summary["hops"]) < 0.1 * len(truth)
    for hop in summary["hops"]:
        assert hop["dwell_s"] > 10e-3  # 融合后的长驻留，不是真正的单跳


def test_impossible_bandwidth_floor_returns_nothing_and_says_why():
    """最小带宽门限高于任何实际游程 → 拒答并给出原因，而不是编造结果。"""
    summary, _, _ = scene([plan(CHANNELS)], config={"min_bandwidth_hz": 500e3})
    assert summary["hops"] == [] and summary["sessions"] == []
    assert summary["resolvable"] is False
    assert "没有任何游程" in summary["reason"]


def test_max_hops_truncates_by_power():
    summary, _, _ = scene(config={"max_hops": 3})
    assert len(summary["hops"]) == 3
    assert [hop["id"] for hop in summary["hops"]] == [1, 2, 3]
    assert summary["hops"][0]["t_start_s"] <= summary["hops"][-1]["t_start_s"]
    assert all(hop["power_dbfs"] > -60.0 for hop in summary["hops"])


def test_tracker_is_deterministic():
    samples, _ = generate_iq(RATE, DURATION, [plan(CHANNELS)], noise={"snr_db": 20.0}, seed=3)
    first, _ = detect_hops(samples, RATE)
    second, _ = detect_hops(samples, RATE)
    assert first == second


def test_empty_input_is_rejected_and_tiny_input_is_safe():
    with pytest.raises(ValueError):
        detect_hops([], RATE)
    summary, _ = detect_hops([1e-9 + 0j], RATE)
    assert summary["hops"] == []


# ---------------------------------------------------------------------------
# 与会话级契约的隔离
# ---------------------------------------------------------------------------

def test_session_level_contract_is_untouched():
    """同一条跳频链路跑会话级检测：仍然只给一个目标（冻结契约不得变化）。"""
    samples, generation = generate_iq(RATE, DURATION, [plan(CHANNELS)],
                                      noise={"snr_db": 20.0}, seed=3)
    summary, _ = detect_signals(samples, RATE, {"nfft": NFFT})
    assert summary["contract"] == "detect_result_v1"
    assert len(summary["detections"]) == 1
    metrics = evaluate_detections(signal_truth(generation), summary["detections"])
    assert (metrics["matched"], metrics["false_alarm"]) == (1, 0)
    # 同一份数据在逐跳口径下是多条：这正是新增契约要补的信息
    hops, _ = detect_hops(samples, RATE)
    assert len(hops["hops"]) > 1


def test_evaluate_detections_accepts_a_contract_override():
    truth = [{"center_hz": 0.0, "bandwidth_hz": 1000.0, "snr_inband_db": 10.0}]
    hops = [{"id": 1, "center_hz": 0.0, "bandwidth_hz": 1000.0, "snr_db": 10.0}]
    assert evaluate_detections(truth, hops)["contract"] == CONTRACT_VERSION
    assert evaluate_detections(truth, hops, contract=HOP_CONTRACT)["contract"] == "fh_hops_v1"
    assert HOP_CONTRACT == "fh_hops_v1" and HOP_ALGORITHM == "hop_track_v1"


# ---------------------------------------------------------------------------
# 服务层与命令行
# ---------------------------------------------------------------------------

def test_service_detect_hops_carries_truth_and_metrics(tmp_path):
    from signal_analysis.services import execute
    from signal_analysis.storage import Workspace

    generated = execute({"workspace": str(tmp_path), "action": "generate",
                         "sample_rate": RATE, "duration": DURATION, "seed": 3,
                         "noise": {"snr_db": 20.0, "bandwidth": RATE},
                         "signals": [plan(CHANNELS)]})
    run = execute({"workspace": str(tmp_path), "action": "detect_hops",
                   "asset_id": generated["id"], "config": {"nfft": NFFT}})
    assert run["kind"] == "detect_hops" and run["contract"] == "fh_hops_v1"
    assert run["algorithm"] == "hop_track_v1" and run["resolvable"] is True
    assert run["hops"] and run["sessions"]
    assert run["metrics"]["precision"] >= 0.9 and run["metrics"]["recall"] >= 0.8
    assert run["baseline_metrics"]["contract"] == "detect_result_v1"
    assert run["truth"][0]["bandwidth_hz"] == pytest.approx(HOP_BW)
    store = Workspace(tmp_path)
    reloaded = store.get_run(run["run_id"])
    boxes = np.load(store.root / reloaded["plots_path"], allow_pickle=False)["hop_boxes"]
    assert boxes.shape == (len(reloaded["hops"]), 4)


def test_service_detect_hops_without_generator_truth(tmp_path):
    from signal_analysis.services import execute

    demo = execute({"workspace": str(tmp_path), "action": "demo", "sample_rate": 48000.0,
                    "count": 65536})
    run = execute({"workspace": str(tmp_path), "action": "detect_hops", "asset_id": demo["id"]})
    assert run["truth"] == {"available": False,
                            "reason": "该资产不是跳频生成样式，逐跳真值不适用"}
    assert run["metrics"] is None
    assert run["hops"]  # 双音演示数据同样按“一段驻留”给出结果，不丢弃


def test_cli_detect_hops_prints_contract(tmp_path, capsys):
    from signal_analysis.cli import main
    from signal_analysis.storage import Workspace

    root = tmp_path / "store"
    spec = tmp_path / "scene.json"
    spec.write_text(json.dumps({"sample_rate": RATE, "duration": DURATION, "seed": 3,
                                "noise": {"enabled": True, "bandwidth": RATE, "snr_db": 20},
                                "signals": [plan(CHANNELS)]}), encoding="utf-8")
    assert main(["--workspace", str(root), "generate", str(spec)]) == 0
    capsys.readouterr()
    asset_id = Workspace(root).list_assets()[0]["id"]
    assert main(["--workspace", str(root), "detect-hops", asset_id, "--nfft", "512",
                 "--threshold-db", "6", "--max-hops", "8"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "detect_hops"
    assert payload["summary"]["contract"] == "fh_hops_v1"
    assert payload["summary"]["config"]["max_hops"] == 8
    assert payload["summary"]["config"]["threshold_db"] == pytest.approx(6.0)
    assert 1 <= len(payload["hops"]) <= 8
    assert json.dumps(payload, ensure_ascii=False, allow_nan=False)
    assert main(["--workspace", str(root), "detect-hops", asset_id, "--no-sessions"]) == 0
    without = json.loads(capsys.readouterr().out)
    assert "baseline" not in without["summary"]


def test_cli_rejects_unknown_option():
    from signal_analysis.cli import main

    with pytest.raises(SystemExit):
        main(["detect-hops", "whatever", "--nope"])


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------

@pytest.mark.gui
def test_gui_hops_tab_renders_result(tmp_path):
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
        assert window.tabs.count() == 7
        labels = [window.tabs.tabText(index) for index in range(window.tabs.count())]
        assert labels == ["数据分析", "IQ 信号生成", "信号检测", "调制识别", "算法对比",
                          "跳频参数", "运行记录"]
        assert window.hops_button.text() == "估计逐跳参数"
        assert window.hops_nfft.currentText() == "512"
        assert window.hops_sessions.isChecked()
        run_job({"workspace": str(workspace), "action": "generate",
                 "sample_rate": RATE, "duration": DURATION, "seed": 3,
                 "noise": {"snr_db": 20.0, "bandwidth": RATE},
                 "signals": [plan(CHANNELS)]})
        window.refresh_assets()
        window.assets.setCurrentRow(0)
        asset = window.selected_asset()
        assert asset is not None and asset["source"] == "generated:iq_fh_video_v1"
        window.hops_button.click()
        deadline = time.monotonic() + 30
        while window.active_job is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        app.processEvents()
        assert window.active_job is None, window.status.text()
        assert "失败" not in window.status.text(), window.status.text()
        assert window.last_result["kind"] == "detect_hops"
        assert window.tabs.currentIndex() == 5
        assert window.hops_table.rowCount() == len(window.last_result["hops"])
        assert window.hops_session_table.rowCount() == len(window.last_result["sessions"])
        text = window.hops_summary.toPlainText()
        assert "fh_hops_v1" in text and "hop_track_v1" in text
        assert "逐跳真值" in text and "会话 1" in text
        assert window.hops_tf_image.image is not None
        assert len(window._hops_items) == len(window.last_result["hops"])
        header = [window.hops_table.horizontalHeaderItem(index).text() for index in range(10)]
        assert header[:4] == ["跳号", "会话", "中心频率", "单跳带宽"]
        assert window.hops_table.item(0, 6).text().endswith("ms")
        assert window.hops_table.item(0, 8).text().endswith("dB")
    finally:
        window.close()
        app.processEvents()
