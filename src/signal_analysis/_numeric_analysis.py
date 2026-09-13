"""统计与可视化数组：波形、双边功率谱、时频图及星座。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import numpy as np

from ._numeric_common import (
    _stft_psd,
    validate_rate,
    validate_samples,
)

from ._numeric_modulation import (
    classify_modulation,
)


def analyze(samples, sample_rate, nfft=256):
    """生成统计摘要及波形、频谱、时频图和星座绘图数据。

    参数：samples 为一维 IQ；sample_rate 为 Hz；nfft 为 16～4096 的整数，默认 256。

    返回：(summary, arrays)：摘要遵循 generic_statistics_v1；数组包含波形预览、双边频率、平均 PSD、时频图与 I/Q 星座。

    算法与边界：校验输入后计算均值、RMS、峰值及启发式分类，共用 STFT 标度；波形最多约 4096 点、星座最多 20000 点，统计仍基于全部样本。非法输入抛出
    ValueError，短输入仅在 STFT 内补零。
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= nfft <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    magnitude = np.abs(x.astype(np.complex128))
    real_valued = bool(np.all(x.imag == 0))
    classification, cluster_estimate = classify_modulation(x)
    summary = {
        "algorithm": "generic_statistics_v1",
        "sample_count": int(x.size), "sample_rate_hz": rate,
        "duration_s": float(x.size / rate),
        "mean_i": float(np.mean(x.real, dtype=np.float64)),
        "mean_q": float(np.mean(x.imag, dtype=np.float64)),
        "rms": float(np.sqrt(np.mean(magnitude ** 2))),
        "peak": float(np.max(magnitude)),
        "amplitude_unit": "arbitrary", "frequency_reference": "baseband_offset",
        "classification": classification, "cluster_estimate": int(cluster_estimate),
        "real_valued": real_valued,
    }
    # 复数输入采用双边功率谱密度，不进行单边谱的倍增。
    psd, frequencies, starts, hop = _stft_psd(x, rate, nfft)
    indices = np.arange(0, x.size, max(1, int(np.ceil(x.size / 4096))))
    const_step = max(1, int(np.ceil(x.size / 20000)))
    const_view = x[::const_step]
    arrays = {
        "wave_time": indices / rate,
        "wave_i": x.real[indices], "wave_q": x.imag[indices],
        "frequency": frequencies,
        "spectrum_db": 10 * np.log10(np.maximum(psd.mean(axis=0), 1e-30)),
        "frame_time": (starts + (nfft - 1) / 2) / rate,
        "spectrogram_db": (10 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        "const_i": const_view.real.astype(np.float32),
        "const_q": const_view.imag.astype(np.float32),
    }
    summary.update({"nfft": nfft, "hop_samples": hop,
                    "psd_unit": "dB relative to 1 arbitrary-unit²/Hz",
                    "preview_decimated": bool(indices.size < x.size),
                    "padded_samples": int(max(0, nfft - x.size))})
    return summary, arrays


def spectrum_row(data, pos, nfft=256, rate=1.0):
    """计算截至指定播放位置的单帧双边功率谱。

    参数：data 为采样序列；pos 为右端不含位置（采样点）；nfft 为 16～4096 的整数；rate 为 Hz。

    返回：(frequencies, psd_db)，均为长度 nfft 的一维数组；单位分别为 Hz 和相对输入幅值平方/Hz 的 dB。

    算法与边界：取 data[max(0,int(pos)-nfft):int(pos)]，不足时左补零；使用与 analyze 相同的 Hanning 窗和 PSD
    归一化。FFT 点数、采样率或播放位置无效时抛出 ValueError。
    """
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= nfft <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    rate = validate_rate(rate)
    if isinstance(pos, bool) or not np.isfinite(float(pos)):
        raise ValueError("播放位置必须为有限数值")
    start = max(0, int(pos) - nfft)
    window_data = np.asarray(data[start:int(pos)], dtype=np.complex128)
    if window_data.size < nfft:
        window_data = np.pad(window_data, (nfft - window_data.size, 0))
    window = np.hanning(nfft)
    transformed = np.fft.fftshift(np.fft.fft(window_data * window))
    psd = np.abs(transformed) ** 2 / (rate * np.sum(window ** 2))
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate))
    return frequencies, 10.0 * np.log10(np.maximum(psd, 1e-30))
