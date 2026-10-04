"""统计与可视化数组：波形、双边功率谱、时频图及星座。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import numpy as np

from .base import (
    _finite,
    _stft_psd,
    validate_rate,
    validate_samples,
)

from ..amc.heuristic import (
    classify_modulation,
)

#: 符号级星座抽取的默认符号数上限（与页面抽点预览口径一致）。
MAX_CONSTELLATION_SYMBOLS = 20000
#: 单次星座抽取最多处理的样点数：超长记录取中段，避免一次分析被拖慢。
MAX_CONSTELLATION_SAMPLES = 4_000_000
#: 定时相位折叠的相位桶数（每符号周期内的分档）。
_TIMING_BINS = 128


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


def _matched_filter(values, rate, bandwidth_hz, symbol_rate):
    """按已知符号率与占用带宽做根升余弦匹配滤波。

    参数：values 为已搬回零频的一维复数组；rate、bandwidth_hz 为 Hz；symbol_rate 为 Bd。

    返回：匹配滤波后的等长复数组。

    算法与边界：滚降系数由占用带宽与符号率反推 ``α = 带宽 / 符号率 − 1``（限幅到
    0～1；带宽未知时用生成器缺省 0.35），在频域乘根升余弦响应：接收端一级匹配滤波
    与发射端成形级联后就是无码间干扰的升余弦脉冲，符号采样点因此成簇而不是晕开。
    响应截止覆盖到采样率一半时按上限截断，不做降采样。
    """
    half = rate / 2.0
    if bandwidth_hz is not None:
        alpha = min(max(bandwidth_hz / symbol_rate - 1.0, 0.0), 1.0)
    else:
        alpha = 0.35
    freqs = np.abs(np.fft.fftfreq(values.size, 1.0 / rate))
    if alpha <= 1e-9:
        mask = (freqs <= min(symbol_rate / 2.0, half)).astype(np.float64)
    else:
        low = (1.0 - alpha) * symbol_rate / 2.0
        high = min((1.0 + alpha) * symbol_rate / 2.0, half)
        shape = np.clip((freqs - low) / max(high - low, 1e-12), 0.0, 1.0)
        response = np.sqrt(0.5 * (1.0 + np.cos(np.pi * shape)))
        mask = np.where(freqs <= low, 1.0, np.where(freqs >= high, 0.0, response))
    return np.fft.ifft(np.fft.fft(values) * mask)


def constellation_points(samples, sample_rate, center_hz, symbol_rate, *,
                         bandwidth_hz=None, limit=MAX_CONSTELLATION_SYMBOLS):
    """按已知特征参数把 IQ 还原成符号级星座点。

    参数：samples 为一维 IQ；sample_rate 为 Hz；center_hz 为信号频偏 Hz；symbol_rate
    为符号率 Bd；bandwidth_hz 为占用带宽 Hz（可省略）；limit 为最多抽取的符号数。

    返回：``{"const_i", "const_q", "symbols", "sps", "timing_phase"}``：单位平均功率
    的符号级 I/Q、符号数、每符号采样点数与符号采样相位（样本）。

    算法与边界：取记录中段按已知频偏搬回零频，再按符号率与占用带宽做根升余弦匹配
    滤波；以 ``|y|²`` 在一个符号周期内的相位折叠峰值估计定时相位，按符号率线性插值
    抽取符号并做 RMS 归一化。全程只用已知参数，不做盲载波/定时估计与均衡：频偏或
    符号率有误时星座会旋转、成弧或弥散。符号率高于采样率一半（每符号不足 2 点）、
    记录过短或参数非法时抛出 ValueError。
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    center = _finite(center_hz, "中心频率")
    if abs(center) > rate / 2.0:
        raise ValueError("中心频率超出 ±采样率/2 的基带范围")
    baud = _finite(symbol_rate, "符号率", 0.0)
    if not baud > 0.0:
        raise ValueError("符号率必须大于 0")
    bandwidth = None if bandwidth_hz is None else _finite(bandwidth_hz, "占用带宽", 0.0)
    if isinstance(limit, bool) or int(limit) != limit or limit < 1:
        raise ValueError("符号数上限必须为正整数")
    limit = int(limit)
    sps = rate / baud
    if sps < 2.0:
        raise ValueError(f"符号率 {baud:g} Bd 高于采样率的一半（{rate / 2:g} Hz），"
                         "无法抽取符号")
    margin = int(np.ceil(8.0 * sps)) + 2
    minimum = 2 * margin + int(np.ceil(4.0 * sps))
    if x.size < minimum:
        raise ValueError("记录太短，不足以抽取符号")
    wanted = int(np.ceil(limit * sps)) + 2 * margin
    span = int(min(x.size, MAX_CONSTELLATION_SAMPLES, max(wanted, minimum)))
    start = (x.size - span) // 2
    y = x[start:start + span].astype(np.complex128)
    if center:
        # 按绝对采样序号混频：相位与生成时的搬频基准一致，星座不会被整体旋转。
        y = y * np.exp(-2j * np.pi * center * np.arange(start, start + span) / rate)
    y = _matched_filter(y, rate, bandwidth, baud)

    # 定时相位：把 |y|² 按符号相位折叠成相位剖面，峰值处即符号采样时刻。
    index = np.arange(margin, span - margin, dtype=np.float64)
    bins = np.minimum(((index / sps) % 1.0 * _TIMING_BINS).astype(np.int64),
                      _TIMING_BINS - 1)
    weights = np.bincount(bins, weights=np.abs(y[margin:span - margin]) ** 2,
                          minlength=_TIMING_BINS)
    counts = np.bincount(bins, minlength=_TIMING_BINS)
    profile = weights / np.maximum(counts, 1)
    peak = int(np.argmax(profile))
    left, right = profile[(peak - 1) % _TIMING_BINS], profile[(peak + 1) % _TIMING_BINS]
    curvature = left - 2.0 * profile[peak] + right
    offset = 0.5 * (left - right) / curvature if curvature < 0 else 0.0
    phase = (peak + 0.5 + offset) / _TIMING_BINS * sps

    first = int(np.ceil((margin - phase) / sps))
    last = int(np.floor((span - margin - phase) / sps))
    if last - first + 1 < 2:
        raise ValueError("记录太短，抽不到足够的符号点")
    if last - first + 1 > limit:
        picks = np.unique(np.linspace(first, last, limit).round().astype(np.int64))
    else:
        picks = np.arange(first, last + 1)
    positions = phase + picks * sps
    base = np.minimum(np.floor(positions).astype(np.int64), span - 2)
    fraction = positions - base
    symbols = y[base] * (1.0 - fraction) + y[base + 1] * fraction
    power = float(np.mean(np.abs(symbols) ** 2))
    if power > 0.0:
        symbols = symbols / np.sqrt(power)
    return {"const_i": symbols.real.astype(np.float32),
            "const_q": symbols.imag.astype(np.float32),
            "symbols": int(symbols.size), "sps": float(sps),
            "timing_phase": float(phase)}


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
