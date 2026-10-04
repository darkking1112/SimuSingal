import numpy as np
import pytest

from signal_analysis.core_api import (analyze, classify_modulation, constellation_points,
                                      generate_iq, spectrum_row, validate_samples)


def test_constant_statistics_and_psd_energy():
    summary, arrays = analyze(np.full(512, 3 + 4j), 1024, 128)
    assert summary["mean_i"] == 3
    assert summary["mean_q"] == 4
    assert summary["rms"] == 5
    assert summary["peak"] == 5
    assert summary["duration_s"] == .5
    power = (10 ** (arrays["spectrum_db"] / 10)).sum() * 1024 / 128
    assert power == pytest.approx(25, rel=1e-6)


def test_complex_negative_frequency_is_preserved():
    rate = 1024
    samples = np.exp(-2j * np.pi * 128 * np.arange(1024) / rate)
    _, arrays = analyze(samples, rate, 256)
    assert arrays["frequency"][np.argmax(arrays["spectrum_db"]) ] == -128
    assert arrays["spectrogram_db"].shape[1] == 256


@pytest.mark.parametrize("samples", [[], [[1, 2]], [float("nan")], [float("inf")], ["1"], [1e100]])
def test_invalid_arrays(samples):
    with pytest.raises(ValueError):
        validate_samples(samples)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        analyze([1], rate)


def test_short_input_and_bounded_preview():
    summary, arrays = analyze([1], 100)
    assert summary["padded_samples"] == 255
    assert np.isfinite(arrays["spectrogram_db"]).all()
    _, large = analyze(np.exp(2j * np.pi * 0.01 * np.arange(100_000)), 48000)
    assert len(large["wave_i"]) <= 4096
    assert len(large["spectrogram_db"]) <= 512


def test_nfft_validation():
    with pytest.raises(ValueError):
        analyze([1], 100, nfft=0)


def _rrc(t, sps, beta=0.35):
    t_sym = np.asarray(t, dtype=float) / sps
    out = np.zeros_like(t_sym)
    special = np.isclose(t_sym, 0.0)
    out[special] = 1 - beta + 4 * beta / np.pi
    non = ~special
    pi_t = np.pi * t_sym[non]
    out[non] = ((np.sin(pi_t * (1 - beta)) + 4 * beta * t_sym[non] * np.cos(pi_t * (1 + beta)))
                / (pi_t * (1 - (4 * beta * t_sym[non]) ** 2)))
    return out


def _pulse_symbols(symbols, sps=16, beta=0.35, span=8):
    taps = _rrc(np.arange(-span * sps, span * sps + 1), sps, beta)
    up = np.zeros(len(symbols) * sps, dtype=complex)
    up[::sps] = symbols
    return np.convolve(up, taps, mode="same")


def test_classify_digital_constellations():
    rng = np.random.default_rng(7)
    n = 2048
    qpsk = np.exp(1j * (2 * np.pi / 4 * rng.integers(0, 4, n) + np.pi / 4))
    cls, est = classify_modulation(_pulse_symbols(qpsk))
    assert cls == "digital" and 2 <= est <= 8
    grid16 = (-3 - 1j * 3 + 2 * (rng.integers(0, 4, n) + 1j * rng.integers(0, 4, n))) / np.sqrt(10)
    cls, est = classify_modulation(_pulse_symbols(grid16))
    assert cls == "digital" and 2 <= est <= 32
    grid64 = (-7 - 1j * 7 + 2 * (rng.integers(0, 8, n) + 1j * rng.integers(0, 8, n))) / np.sqrt(42)
    cls, est = classify_modulation(_pulse_symbols(grid64))
    assert cls == "digital" and 2 <= est <= 256
    ook = rng.integers(0, 2, n).astype(float)
    cls, est = classify_modulation(np.repeat(ook, 16))
    assert cls == "digital" and 2 <= est <= 64


def test_classify_analog_signals():
    rate = 1_000_000.0
    base = {"offset": 100_000.0, "power_dbfs": -10.0, "bandwidth": 100_000.0}
    for mode in ("am", "fm", "ssb"):
        samples, _ = generate_iq(rate, 0.2, [dict(base, mode=mode)], seed=7)
        cls, est = classify_modulation(samples)
        assert (cls, est) == ("analog", 0), mode
    rng = np.random.default_rng(11)
    n = 20000
    cls, est = classify_modulation(rng.normal(size=n) + 1j * rng.normal(size=n))
    assert cls == "analog" and est == 0
    tones = (np.exp(2j * np.pi * 0.03 * np.arange(n))
             + 0.35 * np.exp(-2j * np.pi * 0.07 * np.arange(n)))
    cls, est = classify_modulation(tones)
    assert cls == "analog" and est == 0


def test_classify_deep_sine_am_is_a_known_heuristic_limit():
    """纯正弦消息的深调制 AM 包络起伏大、边缘分布却接近数字信号，启发式会判为数字。

    态势显示页不消费该判定：星座图只按目标参考参数绘制，本用例只固化启发式的已知边界。
    """
    rate = 48000.0
    n = 20000
    t = np.arange(n) / rate
    samples = (1 + 0.8 * np.sin(2 * np.pi * 300 * t)) * np.exp(2j * np.pi * 4000 * t)
    cls, _ = classify_modulation(samples)
    assert cls == "digital"
    assert np.std(np.abs(samples)) / np.mean(np.abs(samples)) > 0.5


def test_real_valued_detection_and_const_arrays():
    rate = 1000.0
    samples = np.sin(2 * np.pi * 50 * np.arange(10000) / rate).astype(np.complex64)
    summary, arrays = analyze(samples, rate, 256)
    assert summary["real_valued"] is True
    assert summary["classification"] == "digital"
    assert 2 <= summary["cluster_estimate"] <= 256
    assert arrays["const_i"].dtype == np.float32
    assert arrays["const_q"].dtype == np.float32
    assert len(arrays["const_i"]) == len(arrays["const_q"]) <= 20000
    np.testing.assert_array_equal(arrays["const_q"], 0.0)


def test_spectrum_row_matches_analyze_window():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=20000) + 1j * rng.normal(size=20000)
    rate = 48000.0
    nfft = 256
    _, arrays = analyze(samples, rate, nfft)
    hop = max(nfft // 2, int(np.ceil((len(samples) - nfft) / 511)))
    starts = np.arange(0, len(samples) - nfft + 1, hop)
    pos = starts[-1] + nfft
    f, row = spectrum_row(samples, pos, nfft, rate)
    np.testing.assert_allclose(row, arrays["spectrogram_db"][-1], atol=1e-4)
    assert f[0] == -rate / 2
    assert f[-1] == rate / 2 - rate / nfft


def _generated_signal(mode, seed=5, snr=30.0, **spec):
    rate = 500_000.0
    base = {"mode": mode, "offset": 120_000.0, "bandwidth": 100_000.0, "power_dbfs": -6.0}
    base.update(spec)
    samples, summary = generate_iq(rate, 0.1, [base], {"enabled": True, "snr_db": snr}, seed)
    return rate, samples, summary["signals"][0]


def test_constellation_points_recover_square_qpsk():
    """按已知频偏与符号率抽取的符号点应贴住四个理想星座点（RMS 归一化后）。"""
    rate, samples, entry = _generated_signal("qpsk")
    points = constellation_points(samples, rate, entry["offset"], entry["symbol_rate"],
                                  bandwidth_hz=entry["bandwidth_actual"])
    assert points["const_i"].dtype == np.float32
    assert points["symbols"] == len(points["const_i"]) == len(points["const_q"])
    assert points["sps"] == pytest.approx(rate / entry["symbol_rate"])
    assert points["symbols"] > 1000
    z = points["const_i"] + 1j * points["const_q"]
    assert float(np.mean(np.abs(z) ** 2)) == pytest.approx(1.0, rel=1e-6)
    ideal = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.0)
    error = np.min(np.abs(z[:, None] - ideal[None, :]), axis=1)
    assert float(np.percentile(error, 99)) < 0.2


def test_constellation_points_show_two_level_ask():
    """2ASK 矩形成形：符号点只有 I 轴两个电平，Q 轴只剩噪声。"""
    rate, samples, entry = _generated_signal("ask2", pulse="rect")
    points = constellation_points(samples, rate, entry["offset"], entry["symbol_rate"],
                                  bandwidth_hz=entry["bandwidth_actual"])
    assert float(np.sqrt(np.mean(points["const_q"] ** 2))) < 0.1
    levels = np.abs(points["const_i"])
    assert float(np.mean(levels < 0.5)) > 0.4
    assert float(np.mean(levels > 0.9)) > 0.4


def test_constellation_points_respect_symbol_limit():
    symbols = np.repeat(np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2.0), 8)
    samples = np.tile(symbols, 200)
    points = constellation_points(samples, 100_000.0, 0.0, 12_500.0, limit=300)
    assert 0 < points["symbols"] <= 300


@pytest.mark.parametrize("call", [
    lambda s: constellation_points(s, 1000.0, 0.0, 600.0),
    lambda s: constellation_points(s, 1000.0, 900.0, 100.0),
    lambda s: constellation_points(s, 1000.0, 0.0, 0.0),
    lambda s: constellation_points(s, 1000.0, 0.0, 100.0, limit=0),
    lambda s: constellation_points(np.ones(8, dtype=np.complex64), 1000.0, 0.0, 100.0),
])
def test_constellation_points_validation(call):
    samples = np.exp(2j * np.pi * 0.01 * np.arange(4096))
    with pytest.raises(ValueError):
        call(samples)
