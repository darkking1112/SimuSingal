"""Qt-independent numerical implementation, also compiled with Cython.

Only generic statistics and scientific plots are implemented here.
"""

import numpy as np

MAX_SAMPLES = 1_000_000


def validate_samples(samples):
    data = np.asarray(samples)
    if data.ndim != 1 or not 1 <= data.size <= MAX_SAMPLES:
        raise ValueError("需要一维非空数组，基础版最多支持 1,000,000 个采样点")
    if data.dtype.kind not in "iufc":
        raise ValueError("只支持实数或复数数值数组")
    if not np.isfinite(data).all():
        raise ValueError("数据包含 NaN 或 Inf")
    if np.max(np.abs(data.astype(np.complex128))) > np.finfo(np.float32).max:
        raise ValueError("数据超出 float32 可表示范围")
    return np.ascontiguousarray(data, dtype=np.complex64)


def validate_rate(sample_rate):
    value = float(sample_rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("采样率必须为有限正数，单位 Hz")
    return value


def make_demo(sample_rate=48000.0, count=8192, seed=7):
    rate = validate_rate(sample_rate)
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= MAX_SAMPLES:
        raise ValueError("演示采样点数必须为 1～1,000,000 的整数")
    t = np.arange(int(count)) / rate
    rng = np.random.default_rng(seed)
    # A reproducible pair of mathematical tones, without a communication model.
    x = np.exp(2j * np.pi * rate / 16 * t)
    x += 0.35 * np.exp(-2j * np.pi * rate / 8 * t)
    x += 0.025 * (rng.standard_normal(int(count)) + 1j * rng.standard_normal(int(count)))
    return x.astype(np.complex64)


def analyze(samples, sample_rate, nfft=256):
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= nfft <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    magnitude = np.abs(x.astype(np.complex128))
    summary = {
        "algorithm": "generic_statistics_v1",
        "sample_count": int(x.size), "sample_rate_hz": rate,
        "duration_s": float(x.size / rate),
        "mean_i": float(np.mean(x.real, dtype=np.float64)),
        "mean_q": float(np.mean(x.imag, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(magnitude ** 2))),
        "peak": float(np.max(magnitude)),
        "amplitude_unit": "arbitrary", "frequency_reference": "baseband_offset",
        "classification": "unsupported",
    }
    # Power spectral density is two-sided for complex input; never double it.
    padded = np.pad(x, (0, max(0, nfft - x.size)))
    hop = max(nfft // 2, int(np.ceil(max(0, padded.size - nfft) / 511)))
    starts = np.arange(0, padded.size - nfft + 1, hop)
    window = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[starts]
    transformed = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
    psd = np.abs(transformed) ** 2 / (rate * np.sum(window ** 2))
    indices = np.arange(0, x.size, max(1, int(np.ceil(x.size / 4096))))
    arrays = {
        "wave_time": indices / rate,
        "wave_i": x.real[indices], "wave_q": x.imag[indices],
        "frequency": np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate)),
        "spectrum_db": 10 * np.log10(np.maximum(psd.mean(axis=0), 1e-30)),
        "frame_time": (starts + (nfft - 1) / 2) / rate,
        "spectrogram_db": (10 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
    }
    summary.update({"nfft": nfft, "hop_samples": hop,
                    "psd_unit": "dB relative to 1 arbitrary-unit²/Hz",
                    "preview_decimated": bool(indices.size < x.size),
                    "padded_samples": int(max(0, nfft - x.size))})
    return summary, arrays
