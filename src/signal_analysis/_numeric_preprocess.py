"""传统特征与原始 IQ 分类共用的校验、混频、低通、抽取和 SNR 粗估。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import math

import numpy as np

from ._numeric_common import (
    _EPS,
)


#: 每单位占用带宽保留的采样点数（决定"每符号点数"的量级，须与训练一致）
SAMPLES_PER_BAND = 8.0


MIN_WORK_SAMPLES = 256


MAX_ANALYSIS_SAMPLES = 1 << 20


LOWPASS_TAPS = 65


def _validate(samples):
    """校验 AMC／IQ 分类输入并截取中心分析窗口。

    参数：samples 为一维非空数值序列，可为实数或复数。

    返回：一维 complex128 数组，最多 MAX_ANALYSIS_SAMPLES 点。

    算法与边界：先检查形状与类型，再在长度超限时居中截断；仅检查保留窗口的有限性和正功率，失败抛出 ValueError。与通用 validate_samples 的
    complex64、拒绝超长规则不同，不能互换。
    """
    data = np.asarray(samples)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("需要一维非空数组")
    if data.dtype.kind not in "iufc":
        raise ValueError("只支持实数或复数数值数组")
    view = np.asarray(data, dtype=np.complex128)
    if view.size > MAX_ANALYSIS_SAMPLES:
        start = (view.size - MAX_ANALYSIS_SAMPLES) // 2
        view = view[start:start + MAX_ANALYSIS_SAMPLES]
    if not (np.isfinite(view.real).all() and np.isfinite(view.imag).all()):
        raise ValueError("数据包含 NaN 或 Inf")
    if float(np.mean(np.abs(view) ** 2)) <= 0.0:
        raise ValueError("数据功率为零，无法提取调制特征")
    return view


def _lowpass_taps(ratio, count):
    """设计直流增益为 1 的汉明窗 sinc 低通。

    参数：ratio 为截止频率/采样率；count 为抽头数。

    返回：长度 count 的实数抽头数组。

    算法与边界：截止比限于 [1/count,0.5]，计算 2c·sinc(2c·offset) 后乘汉明窗并按系数总和归一化；总和绝对值小于 _EPS 时抛出
    ValueError。
    """
    cutoff = float(min(max(ratio, 1.0 / count), 0.5))
    center = (count - 1) / 2.0
    offsets = np.arange(count, dtype=np.float64) - center
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
    taps *= np.hamming(count)
    total = float(taps.sum())
    if abs(total) < _EPS:
        raise ValueError("低通滤波器设计失败")
    return taps / total


def _mix_and_decimate(view, sample_rate, center_hz, bandwidth_hz):
    """将指定频带移至零频并低通抽取。

    参数：view 为已校验的 IQ；sample_rate、center_hz、bandwidth_hz 均为 Hz。

    返回：(work, work_rate, factor)：裁去滤波边缘的一维复数组、分析率 Hz、整数抽取因子。

    算法与边界：相位按周期取模后混频，65 抽头低通；按 sample_rate/(8·bandwidth_hz) 向下取整，并限制抽取后至少约 256
    点。裁剪卷积瞬态时至少保留 32 点；原输入不足 65 点报错，不补零造样本。传统特征与原始 IQ 网络共用此实现。
    """
    count = view.size
    # ``np.convolve(mode="same")`` 的输出长度取两输入的最大值：记录比滤波器还短时
    # 会被"补齐"成滤波器长度，等于凭空造出样本，因此这里必须先拒绝。
    if count < LOWPASS_TAPS:
        raise ValueError(f"有效样本过少，无法提取调制特征（至少需要 {LOWPASS_TAPS} 个采样点）")
    step = float(center_hz) / sample_rate
    frac = np.mod(step * np.arange(count, dtype=np.float64), 1.0)
    baseband = view * np.exp(-2j * np.pi * frac)
    taps = _lowpass_taps(bandwidth_hz / (2.0 * sample_rate), LOWPASS_TAPS)
    filtered = np.convolve(baseband, taps, mode="same")
    factor = max(1, int(math.floor(sample_rate / (SAMPLES_PER_BAND * bandwidth_hz))))
    factor = max(1, min(factor, max(1, filtered.size // MIN_WORK_SAMPLES)))
    work = filtered[::factor]
    work_rate = sample_rate / factor
    edge = min((taps.size // 2) // factor + 1, work.size // 10)
    if edge and work.size - 2 * edge >= 32:
        work = work[edge:work.size - edge]
    return work, work_rate, factor


def _inband_snr(view, sample_rate, center_hz, bandwidth_hz):
    """用带内外平均谱能量粗估 SNR。

    参数：view 为 IQ 数组；sample_rate、center_hz、bandwidth_hz 均为 Hz。

    返回：估计值 float（dB），无法建立带内／带外参考时返回 None。

    算法与边界：最多取前 2^16 点加 Hanning 窗做 FFT，用环形频率距离选带；计算 10log10((P_in-P_out)/P_out)。少于 64
    样本或任一侧少于 4 个频点返回 None；噪声非正或带内不高于带外返回 0。该工程粗估不同于检测器 N0·B 量测。
    """
    count = int(min(view.size, 1 << 16))
    segment = view[:count]
    if count < 64:
        return None
    window = np.hanning(count)
    spectrum = np.abs(np.fft.fft(segment * window)) ** 2
    frequencies = np.fft.fftfreq(count, 1.0 / float(sample_rate))
    # center_hz 可能落在采样带宽之外，用环形距离衡量
    shift = float(center_hz) - float(sample_rate) / 2.0
    wrapped = np.mod(frequencies - shift, float(sample_rate)) + shift
    distance = np.abs(wrapped - float(center_hz))
    inside = distance <= float(bandwidth_hz) / 2.0
    if int(inside.sum()) < 4 or int((~inside).sum()) < 4:
        return None
    in_band = float(np.mean(spectrum[inside]))
    out_band = float(np.mean(spectrum[~inside]))
    if not out_band > 0.0 or not in_band > out_band:
        return 0.0
    return 10.0 * math.log10((in_band - out_band) / out_band)
