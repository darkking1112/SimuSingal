"""逐跳评估契约 ``per_hop_eval_v1`` 测试（方案文档 §5）。"""
import pytest

from signal_analysis.evaluation import evaluate_hop_tracks, match_hop_tracks


def hop(index, start, stop, center, bandwidth=50_000.0, **extra):
    entry = {"hop_index": index, "t_start_s": start, "t_end_s": stop,
             "center_hz": center, "bandwidth_hz": bandwidth,
             "f_low_hz": center - bandwidth / 2.0, "f_high_hz": center + bandwidth / 2.0}
    entry.update(extra)
    return entry


def test_perfect_match_metrics():
    truth = [hop(0, 0.00, 0.01, 100_000.0, snr_db=12.0, hop_rate_hz=100.0),
             hop(1, 0.01, 0.02, -100_000.0, snr_db=12.0, hop_rate_hz=100.0)]
    predicted = [dict(entry) for entry in truth]
    result = evaluate_hop_tracks(truth, predicted)
    assert result["comparable"] is True and result["reason"] is None
    metrics = result["metrics"]
    assert metrics["matched"] == 2 and metrics["missed"] == 0
    assert metrics["false_alarm"] == 0
    assert metrics["precision"] == 1.0 and metrics["recall"] == 1.0
    assert metrics["t_start_mae_s"] == 0.0 and metrics["center_mae_hz"] == 0.0
    assert metrics["channel_set_consistency"] == 1.0
    assert result["session"]["sequence"]["lcs_ratio"] == 1.0
    assert result["pairs"][0]["iou"] == 1.0


def test_missed_false_alarm_and_timing_errors():
    truth = [hop(0, 0.00, 0.01, 100_000.0, snr_db=5.0),
             hop(1, 0.01, 0.02, 100_000.0, snr_db=5.0)]
    predicted = [hop(0, 0.0005, 0.0108, 101_000.0, snr_db=5.0),
                 hop(5, 0.05, 0.06, 300_000.0, snr_db=25.0)]
    result = evaluate_hop_tracks(truth, predicted)
    metrics = result["metrics"]
    assert metrics["matched"] == 1 and metrics["missed"] == 1
    assert metrics["false_alarm"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["t_start_mae_s"] == pytest.approx(0.0005)
    assert metrics["center_mae_hz"] == pytest.approx(1000.0)
    # 分组报表：命中对的桶按真值 SNR 归类，误检按预测 SNR 归类
    low = next(item for item in result["groups"]["snr_db"] if item["label"] == "[0, 10)")
    high = next(item for item in result["groups"]["snr_db"] if item["label"] == "≥ 20")
    assert low["true"] == 2 and low["matched"] == 1 and low["missed"] == 1
    assert high["false_alarm"] == 1


def test_iou_gate_blocks_far_pairs():
    truth = [hop(0, 0.0, 0.01, 100_000.0)]
    predicted = [hop(0, 0.0, 0.01, 400_000.0)]
    result = evaluate_hop_tracks(truth, predicted)
    assert result["metrics"]["matched"] == 0
    assert result["metrics"]["missed"] == 1 and result["metrics"]["false_alarm"] == 1
    # 门限放宽后可匹配（IoU 仍按实际重叠计算）
    relaxed = evaluate_hop_tracks(truth, predicted, iou_gate=0.0)
    assert relaxed["metrics"]["matched"] == 1


def test_tolerance_filters_pairs():
    truth = [hop(0, 0.0, 0.01, 100_000.0)]
    shifted = [hop(0, 0.002, 0.012, 100_000.0)]
    assert evaluate_hop_tracks(truth, shifted,
                               time_tolerance_s=0.005)["metrics"]["matched"] == 1
    assert evaluate_hop_tracks(truth, shifted,
                               time_tolerance_s=0.001)["metrics"]["matched"] == 0
    moved = [hop(0, 0.0, 0.01, 120_000.0)]
    assert evaluate_hop_tracks(truth, moved,
                               freq_tolerance_hz=30_000)["metrics"]["matched"] == 1
    assert evaluate_hop_tracks(truth, moved,
                               freq_tolerance_hz=10_000)["metrics"]["matched"] == 0


def test_hungarian_prefers_global_best_pairing():
    # 两个真值、两个预测：验证一对一配对（不是两个真值都抢最近的同一个预测）
    truth = [hop(0, 0.0, 0.01, 100_000.0), hop(1, 0.0, 0.01, 130_000.0)]
    predicted = [hop(0, 0.0, 0.01, 110_000.0), hop(1, 0.0, 0.01, 131_000.0)]
    pairs = match_hop_tracks(truth, predicted)
    assert [(item[0], item[1]) for item in pairs] == [(0, 0), (1, 1)]
    # 交换预测顺序：仍应一一配对而不是全部错配
    pairs = match_hop_tracks(truth, [predicted[1], predicted[0]])
    assert sorted((item[0], item[1]) for item in pairs) == [(0, 1), (1, 0)]


def test_incomparable_reasons():
    truth = [hop(0, 0.0, 0.01, 100_000.0)]
    session = [{"t_start_s": 0.0, "t_end_s": 0.01, "center_hz": 100_000.0,
                "bandwidth_hz": 50_000.0, "hop_index": None}]
    result = evaluate_hop_tracks(truth, session, predicted_kind="session")
    assert result["comparable"] is False and "会话级" in result["reason"]
    assert result["metrics"] is None
    result = evaluate_hop_tracks([], session)
    assert result["comparable"] is False and "无逐跳信息" in result["reason"]
    result = evaluate_hop_tracks(truth, session, coverage="partial")
    assert result["comparable"] is False and "覆盖度" in result["reason"]


def test_resolution_warning_and_missing_boxes():
    truth = [hop(0, 0.0, 0.01, 100_000.0)]
    result = evaluate_hop_tracks(truth, list(truth),
                                 time_tolerance_s=0.0001, freq_tolerance_hz=100.0,
                                 resolution={"time_resolution_s": 0.001,
                                             "freq_resolution_hz": 1000.0})
    assert len(result["warnings"]) == 2
    # 缺少时间/频率信息的条目（会话级预测混入）按漏检/多检计，不报异常
    broken = [{"center_hz": 100_000.0, "bandwidth_hz": 50_000.0}]
    result = evaluate_hop_tracks(truth, broken)
    assert result["metrics"]["matched"] == 0
    assert result["warnings"] == []


def test_channel_sequence_and_rate_metrics():
    truth = [hop(0, 0.0, 0.01, 100_000.0, hop_rate_hz=100.0),
             hop(1, 0.01, 0.02, 200_000.0, hop_rate_hz=100.0),
             hop(2, 0.02, 0.03, 100_000.0, hop_rate_hz=100.0)]
    predicted = [hop(0, 0.0, 0.01, 100_000.0, hop_rate_hz=110.0),
                 hop(1, 0.01, 0.02, 100_000.0, hop_rate_hz=110.0),
                 hop(2, 0.02, 0.03, 200_000.0, hop_rate_hz=110.0),
                 hop(3, 0.03, 0.04, 100_000.0, hop_rate_hz=110.0)]
    result = evaluate_hop_tracks(truth, predicted, iou_gate=0.1)
    # 频道序列：A B A vs A A B A → LCS=3（A B A），比例 3/4
    assert result["session"]["sequence"]["lcs_ratio"] == pytest.approx(0.75)
    assert result["metrics"]["hop_rate_mae_hz"] == pytest.approx(10.0)
    assert result["metrics"]["hop_count_error"] == 1
    assert 0.0 < result["metrics"]["channel_set_consistency"] <= 1.0
