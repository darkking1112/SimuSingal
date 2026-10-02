"""公共数值基础：校验、谱密度、掩膜、噪声底和共享数值约定。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import math

import numpy as np



MAX_SAMPLES = 16_000_000


def _finite(value, name, minimum=None, maximum=None):
    """将参数转换为有限浮点数并检查范围。

    参数：value 为可转换数值；name 为错误提示名称；minimum、maximum 为可选闭区间端点，单位与 value 相同。

    返回：校验后的 float。

    算法与边界：先调用 float，再检查有限性和上下界；NaN、Inf 或越界抛出 ValueError，类型转换异常原样传播。
    """
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} 必须为有限数值")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} 不能大于 {maximum}")
    return number


def validate_samples(samples):
    """校验通用数值分析输入并统一 IQ 存储格式。

    参数：samples 为一维非空实数或复数数值数组，幅值采用输入单位。

    返回：连续内存的一维 complex64 数组，长度与输入一致。

    算法与边界：依次检查维数、MAX_SAMPLES 上限、数值类型、有限性及 float32 幅值范围；失败抛出
    ValueError。保留原错误文本及校验顺序，不截断数据。
    """
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
    """校验采样率。

    参数：sample_rate 为采样率，单位 Hz。

    返回：正有限 float 采样率。

    算法与边界：先转 float，再拒绝非有限值和非正值；失败抛出 ValueError，类型转换异常原样传播。
    """
    value = float(sample_rate)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("采样率必须为有限正数，单位 Hz")
    return value


def _stft_psd(x, rate, nfft):
    """计算分析和检测共用的双边 STFT 功率谱密度。

    参数：x 为一维 IQ；rate 为采样率 Hz；nfft 为每帧 FFT 点数，由上层保证合法。

    返回：(psd, frequencies, starts, hop)：PSD 形状为（帧数, nfft），单位为输入幅值平方/Hz；频率为 Hz，帧起点和步长为采样点。

    算法与边界：短输入右补零；Hanning 窗、至少 50% 帧长的步长，按记录长度调整以限制到 512
    帧。PSD=|FFT(xw)|²/(rate·sum(w²))，每帧频带功率为 PSD 在频点上求和再乘 rate/nfft；复数双边谱不乘 2。见传统能量检测设计
    §3。
    """
    padded = np.pad(x, (0, max(0, nfft - x.size)))
    hop = max(nfft // 2, int(np.ceil(max(0, padded.size - nfft) / 511)))
    starts = np.arange(0, padded.size - nfft + 1, hop)
    window = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[starts]
    transformed = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
    psd = np.abs(transformed) ** 2 / (rate * np.sum(window ** 2))
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate))
    return psd, frequencies, starts, hop


DETECT_SNR_DEFINITION = "inband_snr_v1"


_OCCUPIED_RATIO = 0.99


#: 带内 SNR 估计下限（dB）：低于此值只报“几乎全为噪声”
_SNR_FLOOR_DB = -20.0


def _noise_floor_db(psd_db, threshold_db, iterations=4):
    """用中位数与 MAD 迭代估计噪声本底。

    参数：psd_db 为谱密度的 dB 数组；threshold_db 为相对门限 dB；iterations 为迭代次数，默认 4。

    返回：与 psd_db 同一参考单位的本底 float。

    算法与边界：排除非有限值，以中位数初始化；每轮用 1.4826·MAD 估标准差，仅保留不高于中心+max(3σ, threshold_db)
    的频点并更新中心。无有限输入返回 0；候选少于 4 点停止。
    """
    values = np.asarray(psd_db, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    centre = float(np.median(values))
    for _ in range(iterations):
        deviation = np.abs(values - centre)
        sigma = 1.4826 * float(np.median(deviation))
        window = max(3.0 * sigma, threshold_db)
        keep = values <= centre + window
        if keep.sum() < 4:
            break
        centre = float(np.median(values[keep]))
    return centre


def _binary_dilate(mask, radius):
    """对一维布尔掩膜做有限边界膨胀。

    参数：mask 为一维掩膜；radius 为左右扩展的频点数。

    返回：与输入等长的布尔数组副本。

    算法与边界：将原掩膜向左右移动 1 至 radius 点后按位或；数组外不参与运算，不修改输入。非正半径只返回副本。
    """
    out = np.asarray(mask, dtype=bool).copy()
    for shift in range(1, int(radius) + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _binary_close(mask, radius):
    """通过一维闭运算填补频谱掩膜内部凹口。

    参数：mask 为一维频点掩膜；radius 为膨胀／腐蚀半径，单位频点。

    返回：与输入等长的布尔掩膜。

    算法与边界：先膨胀，再通过补集膨胀实现腐蚀；沿用有限数组边界规则。radius<=0 时只做布尔转换，可能与输入共享内存。
    """
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    filled = _binary_dilate(mask, radius)
    return ~_binary_dilate(~filled, radius)


def _true_runs(mask):
    """提取布尔掩膜中连续为真的游程。

    参数：mask 为一维可转换为布尔值的序列。

    返回：由 (start, stop) 组成的列表，两端均包含，索引从 0 开始。

    算法与边界：两端补 False，查找状态变化并配对；空掩膜或全 False 返回空列表。
    """
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop) - 1) for start, stop in zip(edges[::2], edges[1::2])]


def _occupied_span(psd_linear, first, last, ratio=_OCCUPIED_RATIO):
    """计算指定频带内的等尾功率占用区间。

    参数：psd_linear 为线性 PSD；first、last 为包含两端的频点索引；ratio 为保留功率比例，默认 0.99。

    返回：占用区间的全局频点索引 (low, high)，均包含端点。

    算法与边界：对带内功率累加，两端各舍去 (1-ratio)/2 后定位边界；总功率非有限或非正时返回原区间。上层保证区间合法；本函数不替代检测掩膜的频带边界。
    """
    segment = np.asarray(psd_linear[first:last + 1], dtype=np.float64)
    total = float(segment.sum())
    if not np.isfinite(total) or total <= 0:
        return first, last
    cumulative = np.cumsum(segment)
    tail = (1.0 - ratio) / 2.0 * total
    low = int(np.searchsorted(cumulative, tail, side="left"))
    high = int(np.searchsorted(cumulative, total - tail, side="left"))
    low = min(max(low, 0), segment.size - 1)
    high = min(max(high, low), segment.size - 1)
    return first + low, first + high


_SESSION_GAP_RATIO = 3.0


_EPS = 1e-30


def _rounded(value, digits=6):
    """将有限数值按指定小数位舍入以写入结果。

    参数：value 为数值或 None；digits 为小数位数，默认 6。

    返回：舍入后的 float；输入为空或非有限时返回 None。

    算法与边界：先处理 None，再转换 float 和检查有限性，最后调用 round；不把无效数值伪装成零。
    """
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)
