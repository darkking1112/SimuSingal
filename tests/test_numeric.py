import numpy as np
import pytest

from simusignal.core_api import analyze, make_demo, validate_samples


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
    _, large = analyze(make_demo(count=100_000), 48000)
    assert len(large["wave_i"]) <= 4096
    assert len(large["spectrogram_db"]) <= 512


def test_demo_reproducible_and_nfft_validation():
    np.testing.assert_array_equal(make_demo(), make_demo())
    with pytest.raises(ValueError):
        analyze([1], 100, nfft=0)
