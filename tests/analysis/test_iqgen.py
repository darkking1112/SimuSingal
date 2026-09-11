"""Signal IQ generation: modes, power, bandwidth, SNR, hopping, validation."""
import numpy as np
import pytest

from signal_analysis._numeric import MAX_SAMPLES, generate_iq, plan_signal

RATE = 1_000_000.0
BASE = {"offset": 100_000.0, "bandwidth": 100_000.0, "power_dbfs": -10.0}
ALL_MODES = ["am", "fm", "ssb", "ask2", "qpsk", "qam16", "qam64", "fh_rc", "fh_video"]


def measured_power(samples):
    return float(10 * np.log10(np.mean(np.abs(samples) ** 2)))


def occupied_bandwidth(samples, rate):
    """Two-sided 99% power bandwidth in Hz (whole-record periodogram)."""
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    total = np.sum(psd)
    cumulative = np.cumsum(psd)
    low = np.argmax(cumulative >= total * 0.005)
    high = np.argmax(cumulative >= total * 0.995)
    return (high - low) * rate / len(samples)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_each_mode_generates_valid_iq(mode):
    spec = dict(BASE)
    if mode == "ssb":
        spec["side"] = "usb"
    samples, summary = generate_iq(RATE, 0.05, [dict(spec, mode=mode)], seed=1)
    assert samples.dtype == np.complex64
    assert samples.size == 50_000
    assert np.isfinite(samples).all()
    assert summary["sample_rate_hz"] == RATE
    assert summary["seed"] == 1
    assert summary["signals"][0]["mode"] == mode
    assert samples.ndim == 1 and samples.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_power_matches_request(mode):
    spec = dict(BASE, power_dbfs=-13.0)
    if mode == "ssb":
        spec["side"] = "usb"
    samples, summary = generate_iq(RATE, 0.05, [dict(spec, mode=mode)], seed=2)
    assert measured_power(samples) == pytest.approx(-13.0, abs=0.5)
    assert summary["signals"][0]["power_dbfs_actual"] == pytest.approx(-13.0, abs=0.5)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_bandwidth_within_tolerance(mode):
    spec = dict(BASE)
    if mode == "ssb":
        spec["side"] = "usb"
    samples, summary = generate_iq(RATE, 0.2, [dict(spec, mode=mode)], seed=3)
    actual = summary["signals"][0]["bandwidth_actual"]
    # 实际带宽应接近目标带宽（自动推导按目标带宽设计）。
    assert actual == pytest.approx(100_000.0, rel=0.3)
    measured = occupied_bandwidth(samples, RATE)
    assert measured == pytest.approx(100_000.0, rel=0.35)


def test_center_of_gravity_at_offset():
    samples, _ = generate_iq(RATE, 0.2, [dict(BASE, mode="qpsk")], seed=4)
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(samples.size, 1 / RATE))
    band = np.abs(frequencies - BASE["offset"]) <= BASE["bandwidth"] / 2
    center = float(np.sum(frequencies[band] * psd[band]) / np.sum(psd[band]))
    assert center == pytest.approx(BASE["offset"], abs=2_000.0)


def test_ssb_is_single_sided():
    samples, _ = generate_iq(RATE, 0.2, [dict(BASE, mode="ssb", side="usb")], seed=5)
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(samples.size, 1 / RATE))
    lower = psd[frequencies < BASE["offset"] - BASE["bandwidth"] / 4].sum()
    upper = psd[frequencies > BASE["offset"] + BASE["bandwidth"] / 4].sum()
    assert upper > 100 * lower


def test_am_has_carrier_line():
    samples, _ = generate_iq(RATE, 0.2, [dict(BASE, mode="am")], seed=6)
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(samples.size, 1 / RATE))
    carrier = psd[np.argmin(np.abs(frequencies - BASE["offset"]))]
    assert carrier > 10 * np.median(psd)


def test_2ask_rect_has_carrier_line():
    samples, _ = generate_iq(RATE, 0.2, [dict(BASE, mode="ask2", pulse="rect")], seed=7)
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(samples.size, 1 / RATE))
    carrier = psd[np.argmin(np.abs(frequencies - BASE["offset"]))]
    assert carrier > 10 * np.median(psd)


def test_inband_snr_relative_to_strongest_signal():
    strong = dict(BASE, mode="qpsk", power_dbfs=-5.0)
    weak = dict(BASE, mode="qpsk", power_dbfs=-25.0, offset=-150_000.0)
    samples, summary = generate_iq(RATE, 0.5, [strong, weak],
                                   noise={"bandwidth": RATE, "snr_db": 20.0}, seed=8)
    first, second = summary["signals"]
    noise = summary["noise"]
    # 1) 带内 SNR 口径：信号平均功率 ÷ (噪声功率谱密度 × 实际占用带宽)。
    assert noise["snr_definition"] == "inband_snr_v1"
    assert noise["snr_reference_index"] == 0
    assert noise["snr_db"] == 20.0
    assert first["snr_inband_db"] == pytest.approx(20.0, abs=0.2)
    # 弱信号低 20 dB 且占用带宽相同 → 带内 SNR 也低 20 dB。
    assert second["snr_inband_db"] == pytest.approx(0.0, abs=0.2)
    # 噪声总功率 = 功率谱密度 × 噪声带宽。
    psd = 10 ** (noise["power_dbfs_per_hz"] / 10)
    assert noise["power_dbfs"] == pytest.approx(10 * np.log10(psd * RATE), abs=1e-6)
    # 2) 独立验证：总功率 ≈ 各分量功率之和（独立随机过程相加）。
    total = measured_power(samples)
    expected = 10 * np.log10(10 ** (-5.0 / 10) + 10 ** (-2.5) + 10 ** (noise["power_dbfs"] / 10))
    assert total == pytest.approx(expected, abs=0.5)


def test_inband_snr_from_spectral_density():
    """带内 SNR 可由功率谱密度独立复算：SNR = P_signal / (N0 × B_actual)。"""
    spec = dict(BASE, mode="qpsk", offset=0.0, power_dbfs=-10.0)
    _, summary = generate_iq(RATE, 0.2, [spec],
                             noise={"bandwidth": RATE, "snr_db": 12.0}, seed=21)
    entry, noise = summary["signals"][0], summary["noise"]
    psd = 10 ** (noise["power_dbfs_per_hz"] / 10)
    expected = entry["power_dbfs_actual"] - 10 * np.log10(psd * entry["bandwidth_actual"])
    assert entry["snr_inband_db"] == pytest.approx(expected, abs=1e-6)


def test_noise_band_must_cover_strongest_signal():
    # 信号位于 100 kHz 附近、占用带宽约 96 kHz，50 kHz 噪声带宽无法覆盖。
    with pytest.raises(ValueError, match="未覆盖最强信号"):
        generate_iq(RATE, 0.05, [dict(BASE, mode="qpsk")],
                    noise={"bandwidth": 50_000.0, "snr_db": 20.0}, seed=22)


def test_signal_outside_noise_band_has_undefined_snr():
    inside = dict(BASE, mode="qpsk", offset=0.0, power_dbfs=-5.0)
    outside = dict(BASE, mode="qpsk", offset=400_000.0, power_dbfs=-20.0)
    _, summary = generate_iq(RATE, 0.2, [inside, outside],
                             noise={"bandwidth": 200_000.0, "snr_db": 15.0}, seed=23)
    assert summary["noise"]["snr_reference_index"] == 0
    assert summary["signals"][0]["snr_inband_db"] == pytest.approx(15.0, abs=0.2)
    assert summary["signals"][1]["snr_inband_db"] is None


def test_multi_signal_independent_and_deterministic():
    first = dict(BASE, mode="qpsk")
    second = dict(BASE, mode="am", offset=-200_000.0, power_dbfs=-15.0)
    a, _ = generate_iq(RATE, 0.1, [first, second], noise={"bandwidth": RATE, "snr_db": 30}, seed=9)
    b, _ = generate_iq(RATE, 0.1, [first, second], noise={"bandwidth": RATE, "snr_db": 30}, seed=9)
    np.testing.assert_array_equal(a, b)
    # 两个信号各自的功率保持独立（无噪声时各占其指定功率）。
    c, _ = generate_iq(RATE, 0.1, [first, second], seed=9)
    assert measured_power(c) == pytest.approx(10 * np.log10(0.1 + 10 ** -1.5), abs=0.5)


def test_pure_noise_mode():
    samples, summary = generate_iq(RATE, 0.05, [], noise={"power_dbfs": -20.0}, seed=10)
    noise = summary["noise"]
    assert measured_power(samples) == pytest.approx(-20.0, abs=0.5)
    assert noise["enabled"] is True
    assert noise["snr_db"] is None
    assert noise["snr_reference_index"] is None
    # 默认噪声带宽 = 采样率，功率谱密度 = 总功率 / 带宽。
    assert noise["power_dbfs_per_hz"] == pytest.approx(-20.0 - 10 * np.log10(RATE), abs=1e-6)
    assert summary["signals"] == []


def test_noise_bandwidth_limits_spectrum():
    samples, _ = generate_iq(RATE, 0.2, [], noise={"power_dbfs": -20.0, "bandwidth": 200_000.0}, seed=11)
    psd = np.abs(np.fft.fftshift(np.fft.fft(samples))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(samples.size, 1 / RATE))
    inband = psd[np.abs(frequencies) <= 100_000.0].sum()
    outband = psd[np.abs(frequencies) > 150_000.0].sum()
    assert inband > 1000 * outband


def test_fh_remote_control_hops_and_frequencies():
    spec = dict(BASE, mode="fh_rc", hop_rate=200.0, hop_count=8)
    samples, summary = generate_iq(RATE, 0.5, [spec], seed=12)
    entry = summary["signals"][0]
    assert entry["hops"] == 100
    assert len(entry["hop_points"]) == 100
    assert set(entry["hop_points"]).issubset(set(plan_signal(spec, RATE)["hop_points"]))
    # 每跳能量集中在对应频点附近。
    hop_samples = samples.size // entry["hops"]
    for index in range(0, entry["hops"], 25):
        segment = samples[index * hop_samples:(index + 1) * hop_samples]
        psd = np.abs(np.fft.fftshift(np.fft.fft(segment))) ** 2
        frequencies = np.fft.fftshift(np.fft.fftfreq(segment.size, 1 / RATE))
        center = float(frequencies[np.argmax(psd)])
        expected = entry["hop_points"][index]
        assert center == pytest.approx(expected, abs=2 * entry["hop_bandwidth"])


def test_fh_video_link_hops_and_ofdm_params():
    spec = dict(BASE, mode="fh_video", hop_rate=100.0, hop_count=8, subcarriers=64)
    samples, summary = generate_iq(RATE, 0.5, [spec], seed=13)
    entry = summary["signals"][0]
    assert entry["hops"] == 50
    assert entry["fft_size"] > entry["subcarriers"]
    assert entry["cp_samples"] >= 1
    assert entry["occupied_bandwidth"] <= entry["hop_bandwidth"] * 1.1


def test_fh_explicit_hop_points():
    points = [60_000.0, 100_000.0, 140_000.0]
    spec = dict(BASE, mode="fh_rc", offset=0.0, hop_points=points, hop_rate=100.0,
                hop_bandwidth=20_000.0)
    samples, summary = generate_iq(RATE, 0.3, [spec], seed=14)
    entry = summary["signals"][0]
    assert set(entry["hop_points"]).issubset(set(points))
    assert entry["hop_span"] == 80_000.0
    assert entry["bandwidth_actual"] == 100_000.0


def test_fh_three_parameter_relation():
    spec = dict(BASE, mode="fh_rc", offset=0.0, hop_count=8,
                hop_span=80_000.0, hop_bandwidth=10_000.0)
    plan = plan_signal(spec, RATE)
    assert plan["hop_span"] == 80_000.0
    assert plan["hop_bandwidth"] == 10_000.0
    assert plan["bandwidth_actual"] == 90_000.0
    assert plan["hop_points"][0] == -40_000.0
    assert plan["hop_points"][-1] == 40_000.0


def test_fh_span_plus_hop_bandwidth_exceeds_band():
    # 频点间隔 80k ≥ 单跳带宽 30k（无重叠），但 80k + 30k > 整体频带 100k
    spec = dict(BASE, mode="fh_rc", offset=0.0, hop_count=2,
                hop_span=80_000.0, hop_bandwidth=30_000.0)
    with pytest.raises(ValueError, match="整体频带范围"):
        plan_signal(spec, RATE)


def test_fh_overlap_rejected():
    # 频点间隔 20k/7 < 单跳带宽 20k → 频域重叠
    spec = dict(BASE, mode="fh_rc", offset=0.0, hop_count=8,
                hop_span=20_000.0, hop_bandwidth=20_000.0)
    with pytest.raises(ValueError, match="重叠|跨度"):
        plan_signal(spec, RATE)


def test_fh_default_occupies_declared_band():
    for mode in ("fh_rc", "fh_video"):
        spec = dict(BASE, mode=mode, hop_count=8)
        plan = plan_signal(spec, RATE)
        assert plan["hop_span"] + plan["hop_bandwidth"] == pytest.approx(BASE["bandwidth"])
        assert plan["bandwidth_actual"] == pytest.approx(BASE["bandwidth"])
        assert plan["hop_bandwidth"] == pytest.approx(plan["hop_span"] / 8)


def test_same_seed_reproduces_and_differs_otherwise():
    spec = [dict(BASE, mode="qam64")]
    first, _ = generate_iq(RATE, 0.02, spec, seed=99)
    second, _ = generate_iq(RATE, 0.02, spec, seed=99)
    third, _ = generate_iq(RATE, 0.02, spec, seed=100)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)


def test_summary_is_json_serializable():
    import json
    _, summary = generate_iq(RATE, 0.05, [dict(BASE, mode="fh_video", hop_rate=50)],
                             noise={"bandwidth": RATE, "snr_db": 15}, seed=16)
    json.dumps(summary)


def test_count_limit_enforced():
    with pytest.raises(ValueError, match="1～16_000_000|16000000|16,000,000|MAX"):
        generate_iq(RATE, 17.0, [BASE], seed=17)
    assert MAX_SAMPLES == 16_000_000


@pytest.mark.parametrize("bad", [
    dict(BASE, mode="nope"),
    dict(BASE, mode="qpsk", offset=600_000.0),
    dict(BASE, mode="qpsk", bandwidth=1_200_000.0),
    dict(BASE, mode="qpsk", bandwidth=0.0),
    dict(BASE, mode="qpsk", power_dbfs=1.0),
    dict(BASE, mode="qpsk", alpha=2.0),
    dict(BASE, mode="ssb", side="middle"),
    dict(BASE, mode="fm", deviation=200_000.0),
    dict(BASE, mode="fh_rc", hop_points=[500_000.0, -500_000.0]),
    dict(BASE, mode="fh_video", subcarriers=4),
    dict(BASE, mode="fh_video", cp_ratio=1.0),
    dict(BASE, mode="fh_rc", hop_rate=0.0),
])
def test_invalid_signal_parameters_raise(bad):
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.01, [bad], seed=18)


def test_invalid_top_level_arguments_raise():
    with pytest.raises(ValueError):
        generate_iq(0.0, 0.01, [BASE], seed=19)
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.0, [BASE], seed=20)
    with pytest.raises(ValueError):
        generate_iq(RATE, -1.0, [BASE], seed=21)
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.01, [], seed=22)
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.01, [BASE], noise={"bandwidth": 2 * RATE}, seed=23)
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.01, [BASE], noise={"snr_db": 20}, seed=-1)
    with pytest.raises(ValueError):
        generate_iq(RATE, 0.01, [{}], seed=24)


def test_too_many_signals_raise():
    with pytest.raises(ValueError, match="16"):
        generate_iq(RATE, 0.01, [dict(BASE, mode="qpsk")] * 17, seed=25)
