"""测试 IQ 生成：九种调制样式、参数规划、带限噪声和数学双音。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import numpy as np

from ._numeric_common import (
    MAX_SAMPLES,
    _finite,
    validate_rate,
)


MAX_SIGNALS = 16


MODES = ("am", "fm", "ssb", "ask2", "qpsk", "qam16", "qam64", "fh_rc", "fh_video")


MODE_NAMES = {
    "am": "AM", "fm": "FM", "ssb": "SSB", "ask2": "2ASK", "qpsk": "QPSK",
    "qam16": "16QAM", "qam64": "64QAM", "fh_rc": "跳频·遥控(FH-2FSK)",
    "fh_video": "跳频·图传(FH-OFDM)",
}


_CONSTELLATIONS = {
    "qpsk": (np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2.0)),
    "qam16": (np.array([r + 1j * i for r in (-3, -1, 1, 3) for i in (-3, -1, 1, 3)]) / np.sqrt(10.0)),
    "qam64": (np.array([r + 1j * i for r in (-7, -5, -3, -1, 1, 3, 5, 7)
                        for i in (-7, -5, -3, -1, 1, 3, 5, 7)]) / np.sqrt(42.0)),
}


def _signal_meta(spec, rate):
    """读取生成器单信号的基础参数。

    参数：spec 为参数字典；rate 为采样率 Hz；offset、bandwidth 的单位为 Hz，power_dbfs 为 dBFS。

    返回：(mode, offset, bandwidth, power_dbfs)。

    算法与边界：检查样式是否在 MODES 中，填入默认频偏、带宽和功率，拒绝非字典、未知样式、非有限及越界值，带宽必须为正。
    """
    if not isinstance(spec, dict):
        raise ValueError("每个信号必须用参数字典描述")
    mode = str(spec.get("mode", ""))
    if mode not in MODES:
        raise ValueError(f"不支持的调制样式：{mode or '（未指定）'}，可选：{', '.join(MODES)}")
    offset = _finite(spec.get("offset", 0.0), "频点", -rate / 2, rate / 2)
    bandwidth = _finite(spec.get("bandwidth", rate / 10.0), "带宽", 0.0, rate)
    if bandwidth <= 0:
        raise ValueError("带宽必须大于 0")
    power_dbfs = _finite(spec.get("power_dbfs", -10.0), "功率", -200.0, 0.0)
    return mode, offset, bandwidth, power_dbfs


def _check_band(offset, width, rate, mode):
    """检查信号频带是否位于采样基带内。

    参数：offset、width、rate 的单位均为 Hz；mode 为调制样式。

    返回：校验成功返回 None。

    算法与边界：常规样式检查 abs(offset)+width/2<=rate/2；SSB 沿用 offset±width 两侧保守校验。越界抛出 ValueError。
    """
    if mode == "ssb":
        okay = offset - width >= -rate / 2 and offset + width <= rate / 2
    else:
        okay = abs(offset) + width / 2 <= rate / 2
    if not okay:
        raise ValueError(f"频点 {offset:g} Hz 与带宽 {width:g} Hz 超出 ±{rate / 2:g} Hz 的基带范围")


def occupied_interval(offset, width, mode, side=None):
    """给出生成信号的双边占用频率区间。

    参数：offset 为频偏 Hz，width 为带宽 Hz，mode 为样式，side 为 SSB 边带 usb/lsb。

    返回：(low, high)，单位 Hz，均相对复基带零频。

    算法与边界：SSB 从 offset 单侧延伸 width，lsb 向下，其余 side 向上；其他样式围绕 offset
    对称。只计算区间，不承担参数校验；供噪声覆盖与 SNR 折算复用。
    """
    if mode == "ssb":
        return (offset - width, offset) if side == "lsb" else (offset, offset + width)
    return offset - width / 2.0, offset + width / 2.0


def _hop_points(spec, rate, offset, bandwidth, mode):
    """解析跳频信道集合与单跳带宽。

    参数：spec 为生成参数；rate、offset、bandwidth 为 Hz；mode 为跳频样式。

    返回：(points, hop_bw, span)：频点列表、单跳带宽及中心跨度，单位均为 Hz。

    算法与边界：显式频点要求 2～64 个；自动频点均匀分布于 offset±span/2，默认 hop_bw=bandwidth/(count+1)。检查
    span+hop_bw 不超过总带宽、信道间隔及基带范围；不满足抛出 ValueError。
    """
    explicit = spec.get("hop_points")
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)) or not 2 <= len(explicit) <= 64:
            raise ValueError("跳频频点集合应显式给出 2～64 个频点")
        points = [_finite(point, "跳频频点", -rate / 2, rate / 2) for point in explicit]
        span = max(points) - min(points)
    else:
        count = int(spec.get("hop_count", 8))
        if not 2 <= count <= 64:
            raise ValueError("跳频频点数量必须为 2～64 的整数")
        # 全部自动时取不动点解：hop_bw = bandwidth/(count+1)、
        # span = bandwidth − hop_bw，使 hop_bw = span/count 且
        # span + hop_bw = bandwidth 同时成立。
        tentative = _finite(spec.get("hop_bandwidth", bandwidth / (count + 1.0)),
                            "单跳带宽", 0.0, bandwidth)
        span = _finite(spec.get("hop_span", bandwidth - tentative),
                       "跳频中心跨度", 0.0, bandwidth)
        if span <= 0:
            raise ValueError("跳频中心跨度必须大于 0")
        points = [offset + span / 2.0 * (-1 + 2 * index / (count - 1)) for index in range(count)]
    hop_bw = _finite(spec.get("hop_bandwidth", span / max(2, len(points))),
                     "单跳带宽", 0.0, bandwidth)
    if hop_bw <= 0:
        raise ValueError("单跳带宽必须大于 0")
    if span + hop_bw > bandwidth:
        raise ValueError("跳频中心跨度 + 单跳带宽 不能超过整体频带范围")
    if span / (len(points) - 1) < hop_bw:
        raise ValueError("频点间隔小于单跳带宽（频域重叠），请增大跨度或减小单跳带宽")
    for point in points:
        _check_band(point, hop_bw, rate, mode)
    return points, hop_bw, span


def _rrc_taps(alpha, sps, span_symbols=8):
    """生成单位能量根升余弦滤波器抽头。

    参数：alpha 为滚降系数；sps 为每符号采样点；span_symbols 为单侧符号跨度，默认 8，由上层保证合法。

    返回：长度 2·int(span_symbols·sps)+1 的一维实数抽头。

    算法与边界：时间按符号单位计算 RRC 闭式表达式，对 t=0 和 |t|=1/(4alpha) 用解析极限填充，再除以能量平方根。
    """
    n = int(span_symbols * sps)
    t = np.arange(-n, n + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_t = np.pi * t
        taps = (np.sin(pi_t * (1 - alpha)) + 4 * alpha * t * np.cos(pi_t * (1 + alpha)))
        taps = taps / (pi_t * (1 - (4 * alpha * t) ** 2))
    taps[n] = 1 - alpha + 4 * alpha / np.pi
    edge = np.abs(np.abs(t) - 1 / (4 * alpha)) < 1e-12
    if np.any(edge):
        value = (alpha / np.sqrt(2.0)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                                          + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        taps[edge] = value
    taps = taps / np.sqrt(np.sum(taps ** 2))
    return taps


def _shape_pulses(symbols, taps, sps, count):
    """对符号流上采样、脉冲成形并裁剪。

    参数：symbols 为复符号数组；taps 为滤波抽头；sps 为整数上采样倍数；count 为请求输出点数。

    返回：去除滤波器群延迟后的一维复数采样切片，取前 count 点。

    算法与边界：符号间插零；抽头不超过 256 时直接卷积，否则零填充到足够大的二次幂后 FFT 卷积。上层负责提供足够符号数。
    """
    stream = np.zeros(len(symbols) * sps, dtype=np.complex128)
    stream[::sps] = symbols
    if len(taps) <= 256:
        shaped = np.convolve(stream, taps)
    else:
        full = len(stream) + len(taps) - 1
        fft_n = 1 << (full - 1).bit_length()
        shaped = np.fft.ifft(np.fft.fft(stream, fft_n) * np.fft.fft(taps, fft_n))[:full]
    delay = len(taps) // 2
    return shaped[delay:delay + count]


def _band_noise(rng, count, rate, band, complex_output=True):
    """生成零频附近的单位平均功率带限噪声。

    参数：rng 为 NumPy 随机生成器；count 为点数；rate 为 Hz；band 为保留的最大绝对频率 Hz；complex_output
    控制原始噪声是否含虚部。

    返回：长度 count 的复数数组，平均模平方为 1。

    算法与边界：独立高斯噪声经 FFT 后清除 |f|>band 的频点，逆变换并按实测功率归一化。即使原始噪声为实数也返回逆 FFT 复数组；功率为零抛出
    ValueError。
    """
    if complex_output:
        noise = rng.standard_normal(count) + 1j * rng.standard_normal(count)
    else:
        noise = rng.standard_normal(count)
    spectrum = np.fft.fft(noise)
    frequencies = np.fft.fftfreq(count, 1 / rate)
    spectrum[np.abs(frequencies) > band] = 0.0
    filtered = np.fft.ifft(spectrum)
    power = np.mean(np.abs(filtered) ** 2)
    if not power > 0:
        raise ValueError("带限消息生成失败：带宽过小")
    return filtered / np.sqrt(power)


def _band_noise_at(rng, count, rate, f_low, f_high):
    """生成指定频率区间的单位平均功率复噪声。

    参数：rng 为随机生成器；count 为点数；rate、f_low、f_high 均为 Hz。

    返回：长度 count 的复数数组，占用 [f_low,f_high]。

    算法与边界：生成复高斯噪声，FFT 硬截带后逆变换并归一化；无有效功率时抛出 ValueError。边界频点包含在保留范围内。
    """
    noise = rng.standard_normal(count) + 1j * rng.standard_normal(count)
    spectrum = np.fft.fft(noise)
    frequencies = np.fft.fftfreq(count, 1 / rate)
    spectrum[(frequencies < f_low) | (frequencies > f_high)] = 0.0
    filtered = np.fft.ifft(spectrum)
    power = np.mean(np.abs(filtered) ** 2)
    if not power > 0:
        raise ValueError("带限信号生成失败：带宽过小")
    return filtered / np.sqrt(power)


def _shift_to_offset(baseband, t, offset):
    """将基带波形搬移到指定频偏。

    参数：baseband 为复波形，t 为对应时间数组（秒），offset 为 Hz。

    返回：搬频后波形；offset 为零时直接返回原输入对象。

    算法与边界：非零频偏乘以 exp(2jπ·offset·t)，不做带宽或混叠检查，范围校验由上层完成。
    """
    if offset:
        return baseband * np.exp(2j * np.pi * offset * t)
    return baseband


def _scale_power(signal, power_dbfs):
    """将波形缩放到指定平均功率。

    参数：signal 为波形数组；power_dbfs 为目标平均功率 dBFS。

    返回：缩放后的波形数组。

    算法与边界：幅度系数为 10^(power_dbfs/20)/sqrt(mean(|signal|²))；实测功率非正时抛出 ValueError，不做峰值裁剪。
    """
    measured = np.mean(np.abs(signal) ** 2)
    if not measured > 0:
        raise ValueError("信号功率为零，无法缩放")
    amplitude = 10 ** (power_dbfs / 20.0) / np.sqrt(measured)
    return signal * amplitude


def _amplitude_modulation(rng, t, count, rate, spec):
    """生成带载波 AM 波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：实部带限高斯消息按峰值归一化，以 1+depth·message 调幅后搬频并定标；返回
    depth、message_bandwidth、bandwidth_actual=2·消息带宽。消息无有效功率或参数非法时报错。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    _check_band(offset, bandwidth, rate, mode)
    depth = _finite(spec.get("depth", 0.8), "调制深度", 0.01, 1.0)
    message_bw = _finite(spec.get("message_bandwidth", bandwidth / 2.0), "消息带宽", 0.0, bandwidth)
    if not 0 < message_bw <= bandwidth:
        raise ValueError("AM 消息带宽必须大于 0 且不超过目标带宽")
    message = np.real(_band_noise(rng, count, rate, message_bw, complex_output=False))
    peak = np.max(np.abs(message))
    if not peak > 0:
        raise ValueError("AM 消息生成失败")
    envelope = 1.0 + depth * message / peak
    signal = _shift_to_offset(envelope, t, offset)
    signal = _scale_power(signal, power_dbfs)
    info = {"depth": depth, "message_bandwidth": message_bw,
            "bandwidth_actual": 2.0 * message_bw}
    return signal, info


def _frequency_modulation(rng, t, count, rate, spec):
    """生成恒包络 FM 波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：带限消息按标准差归一化，累加消息形成相位，再搬频定标。deviation 是 RMS 频偏，bandwidth_actual=5.3·deviation
    为既有经验估计；默认 deviation=0.19·带宽。不裁消息峰值；参数非法或消息标准差非正报错。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    _check_band(offset, bandwidth, rate, mode)
    message_bw = _finite(spec.get("message_bandwidth", bandwidth / 4.0), "消息带宽", 0.0, bandwidth)
    if not 0 < message_bw < bandwidth / 2.0:
        raise ValueError("FM 消息带宽必须大于 0 且小于目标带宽的一半")
    # deviation 表示 RMS 频偏：消息按单位 RMS 归一化（高斯消息，无削波），
    # 99% 功率占用带宽 ≈ 5.3 × deviation（经验标定），自动模式取 0.19×带宽。
    deviation = _finite(spec.get("deviation", 0.19 * bandwidth), "频偏", 0.0, bandwidth)
    if not 0 < deviation <= bandwidth:
        raise ValueError("FM 频偏必须大于 0 且不大于目标带宽")
    message = np.real(_band_noise(rng, count, rate, message_bw, complex_output=False))
    rms = float(np.std(message))
    if not rms > 0:
        raise ValueError("FM 消息生成失败")
    phase = 2 * np.pi * deviation * np.cumsum(message / rms) / rate
    signal = _shift_to_offset(np.exp(1j * phase), t, offset)
    signal = _scale_power(signal, power_dbfs)
    info = {"message_bandwidth": message_bw, "deviation": deviation,
            "bandwidth_actual": 5.3 * deviation}
    return signal, info


def _single_sideband(rng, t, count, rate, spec):
    """生成指定边带的 SSB 波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：usb 使用 [offset,offset+bandwidth]，lsb 使用 [offset-bandwidth,offset]
    的带限复噪声，再按平均功率定标；side 非 usb/lsb 或频带非法时报错。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    _check_band(offset, bandwidth, rate, mode)
    side = str(spec.get("side", "usb"))
    if side not in ("usb", "lsb"):
        raise ValueError("SSB 边带必须为 usb 或 lsb")
    if side == "usb":
        signal = _band_noise_at(rng, count, rate, offset, offset + bandwidth)
    else:
        signal = _band_noise_at(rng, count, rate, offset - bandwidth, offset)
    signal = _scale_power(signal, power_dbfs)
    info = {"side": side, "bandwidth_actual": bandwidth}
    return signal, info


def _linear_digital(rng, t, count, rate, spec):
    """生成 2ASK、QPSK、16QAM 或 64QAM 波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：随机选符号，sps=max(2,ceil(rate/(bandwidth/(1+alpha))))；2ASK
    可矩形成形，其余根升余弦成形，再搬频与功率定标。返回滚降、成形方式、sps、符号率和实际带宽；保留星座顺序及随机抽样顺序，参数非法时报错。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    _check_band(offset, bandwidth, rate, mode)
    alpha = _finite(spec.get("alpha", 0.35), "滚降系数", 0.05, 1.0)
    if mode == "ask2":
        pulse = str(spec.get("pulse", "rrc"))
        if pulse not in ("rrc", "rect"):
            raise ValueError("2ASK 脉冲成形必须为 rrc 或 rect")
        symbols = rng.integers(0, 2, size=max(2, int(np.ceil(count * (1 + alpha) / rate * bandwidth) + 1)))
        symbols = symbols.astype(np.complex128)
    else:
        pulse = "rrc"
        constellation = _CONSTELLATIONS[mode]
        needed = int(np.ceil(count * (1 + alpha) / rate * bandwidth) + 1)
        symbols = constellation[rng.integers(0, len(constellation), size=max(2, needed))]
    sps = int(np.ceil(rate / (bandwidth / (1 + alpha))))
    sps = max(2, sps)
    symbol_rate = rate / sps
    if pulse == "rect":
        signal = np.repeat(symbols, sps)[:count]
    else:
        signal = _shape_pulses(symbols, _rrc_taps(alpha, sps), sps, count)
    signal = _shift_to_offset(signal, t, offset)
    signal = _scale_power(signal, power_dbfs)
    info = {"alpha": alpha, "pulse": pulse, "sps": sps,
            "symbol_rate": symbol_rate,
            "bandwidth_actual": symbol_rate * (1 + alpha) if pulse == "rrc" else symbol_rate}
    return signal, info


def _fh_hop_boundaries(count, hops):
    """按整段长度等分并舍入生成逐跳采样边界。

    参数：count 为总采样点数；hops 为跳数，由调用方保证为正整数。

    返回：长度 hops+1 的 int64 数组，表示相邻半开区间的边界。

    算法与边界：计算 round(arange(hops+1)·count/hops)，使用 NumPy 舍入规则；相邻区间无缝衔接，供生成器和真值评估共用。
    """
    return np.round(np.arange(hops + 1) * count / hops).astype(np.int64)


def _fh_remote_control(rng, t, count, rate, spec):
    """生成逐跳 2FSK 遥控波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：按 _fh_hop_boundaries
    划分跳区间，随机选频点；每跳随机二进制消息保持后积分相位，加随机初相并搬至跳频点。按整段功率定标，记录实际访问序列、跳数、频偏和符号率；连续重用同频点允许发生。非法参数报错，空区间跳过。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    points, hop_bw, span = _hop_points(spec, rate, offset, bandwidth, mode)
    hop_rate = _finite(spec.get("hop_rate", 100.0), "跳速", 0.0, rate)
    if not 0 < hop_rate <= rate:
        raise ValueError("跳速必须大于 0 且不超过采样率")
    hops = int(np.round(hop_rate * count / rate))
    hops = max(1, min(hops, count))
    deviation = _finite(spec.get("deviation", hop_bw / 4.0), "频偏", 0.0, hop_bw)
    if not 0 < deviation < hop_bw:
        raise ValueError("每跳 2FSK 频偏必须大于 0 且小于每跳带宽")
    symbol_rate = _finite(spec.get("symbol_rate", hop_bw / 2.0), "符号速率", 0.0, rate)
    if not 0 < symbol_rate <= rate:
        raise ValueError("符号速率必须大于 0 且不超过采样率")
    signal = np.zeros(count, dtype=np.complex128)
    used_points = []
    boundaries = _fh_hop_boundaries(count, hops)
    for hop in range(hops):
        start, end = int(boundaries[hop]), int(boundaries[hop + 1])
        length = end - start
        if length <= 0:
            continue
        point = points[int(rng.integers(0, len(points)))]
        used_points.append(point)
        per_hop = max(2, int(np.round(symbol_rate * length / rate)))
        data = 2.0 * rng.integers(0, 2, size=per_hop) - 1.0
        held = np.repeat(data, max(1, int(np.ceil(length / per_hop))))[:length]
        phase = 2 * np.pi * deviation * np.cumsum(held) / rate
        phase0 = rng.uniform(0, 2 * np.pi)
        signal[start:end] = np.exp(1j * (phase0 + 2 * np.pi * point * t[start:end] + phase))
    signal = _scale_power(signal, power_dbfs)
    info = {"hop_rate": hop_rate, "hops": int(hops), "hop_points": used_points,
            "deviation": deviation, "symbol_rate": symbol_rate,
            "hop_bandwidth": hop_bw, "hop_span": span,
            "bandwidth_actual": span + hop_bw}
    return signal, info


def _fh_video_link(rng, t, count, rate, spec):
    """生成逐跳 OFDM 图传波形。

    参数：rng 为专属随机生成器；t 为秒数组；count 为采样点数；rate 为 Hz；spec 为样式参数字典（频率／带宽 Hz、功率 dBFS）。

    返回：(signal, info)：长度 count 的复波形及样式专用参数字典。

    算法与边界：每跳随机选中心频率，用随机 QPSK 子载波填充频域、IFFT 后加循环前缀；只填完整 OFDM 块，尾部保留空闲，再按整段功率定标。返回访问序列、FFT
    大小、前缀、子载波间隔等；样本不足以形成有效功率或参数非法时报错。
    """
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    points, hop_bw, span = _hop_points(spec, rate, offset, bandwidth, mode)
    hop_rate = _finite(spec.get("hop_rate", 50.0), "跳速", 0.0, rate)
    if not 0 < hop_rate <= rate:
        raise ValueError("跳速必须大于 0 且不超过采样率")
    hops = int(np.round(hop_rate * count / rate))
    hops = max(1, min(hops, count))
    subcarriers = int(spec.get("subcarriers", 64))
    if not 8 <= subcarriers <= 1024:
        raise ValueError("OFDM 子载波数必须为 8～1024 的整数")
    cp_ratio = _finite(spec.get("cp_ratio", 0.25), "循环前缀比例", 0.0, 0.5)
    if not 0 < cp_ratio <= 0.5:
        raise ValueError("循环前缀比例必须大于 0 且不超过 0.5")
    n_fft = int(max(subcarriers + 4, round(rate * subcarriers / hop_bw)))
    n_cp = max(1, int(np.round(n_fft * cp_ratio)))
    block = n_fft + n_cp
    active = np.arange(-subcarriers // 2, subcarriers // 2)
    active = active[active != 0]
    constellation = _CONSTELLATIONS["qpsk"]
    signal = np.zeros(count, dtype=np.complex128)
    used_points = []
    boundaries = _fh_hop_boundaries(count, hops)
    for hop in range(hops):
        start, end = int(boundaries[hop]), int(boundaries[hop + 1])
        length = end - start
        if length <= 0:
            continue
        point = points[int(rng.integers(0, len(points)))]
        used_points.append(point)
        frames = max(1, length // block)
        frame = np.zeros(frames * block, dtype=np.complex128)
        scale = np.sqrt(n_fft / len(active))
        for index in range(frames):
            grid = np.zeros(n_fft, dtype=np.complex128)
            grid[active % n_fft] = constellation[rng.integers(0, len(constellation), size=len(active))]
            body = np.fft.ifft(grid) * scale
            frame[index * block:(index + 1) * block] = np.concatenate((body[n_fft - n_cp:], body))
        segment = np.zeros(length, dtype=np.complex128)
        segment[:len(frame)] = frame
        phase0 = rng.uniform(0, 2 * np.pi)
        signal[start:end] = segment * np.exp(1j * (phase0 + 2 * np.pi * point * t[start:end]))
    signal = _scale_power(signal, power_dbfs)
    subcarrier_spacing = rate / n_fft
    info = {"hop_rate": hop_rate, "hops": int(hops), "hop_points": used_points,
            "hop_bandwidth": hop_bw, "hop_span": span,
            "bandwidth_actual": span + hop_bw,
            "subcarriers": int(subcarriers), "cp_ratio": cp_ratio,
            "fft_size": int(n_fft), "cp_samples": int(n_cp),
            "subcarrier_spacing": subcarrier_spacing,
            "occupied_bandwidth": len(active) * subcarrier_spacing}
    return signal, info


def _synthesize(plan, rate, count, t, rng):
    """按已解析的调制样式分派信号生成器。

    参数：plan 为 plan_signal 的结果；rate 为 Hz；count 为点数；t 为秒数组；rng 为该信号专属随机生成器。

    返回：(signal, info)：复数波形及样式专用参数字典。

    算法与边界：AM、FM、SSB、线性数字调制与遥控跳频分别分派；其余已校验样式进入图传跳频。此内部入口依赖 plan 已合法，不重复总体验证。
    """
    mode = plan["mode"]
    if mode == "am":
        return _amplitude_modulation(rng, t, count, rate, plan)
    if mode == "fm":
        return _frequency_modulation(rng, t, count, rate, plan)
    if mode == "ssb":
        return _single_sideband(rng, t, count, rate, plan)
    if mode in ("ask2", "qpsk", "qam16", "qam64"):
        return _linear_digital(rng, t, count, rate, plan)
    if mode == "fh_rc":
        return _fh_remote_control(rng, t, count, rate, plan)
    return _fh_video_link(rng, t, count, rate, plan)


def plan_signal(spec, sample_rate):
    """校验单信号配置并补齐自动派生参数。

    参数：spec 为包含 mode 的参数字典；sample_rate 为 Hz。频偏和带宽为 Hz，功率为 dBFS，时间节奏为每秒次数。

    返回：标准化参数字典，含样式专用配置及 bandwidth_actual；不生成采样数据。

    算法与边界：GUI 与生成器共用此规则集，逐样式计算符号率、滤波参数或跳频信道与 OFDM 参数。非法配置抛出
    ValueError；保留各样式现有默认值和校验顺序。见分析技术方案的测试信号 IQ 生成章节。
    """
    rate = validate_rate(sample_rate)
    mode, offset, bandwidth, power_dbfs = _signal_meta(spec, rate)
    resolved = {"mode": mode, "offset": offset, "bandwidth": bandwidth,
                "power_dbfs": power_dbfs}
    if mode == "am":
        message_bw = _finite(spec.get("message_bandwidth", bandwidth / 2.0),
                             "消息带宽", 0.0, bandwidth)
        if not 0 < message_bw <= bandwidth:
            raise ValueError("AM 消息带宽必须大于 0 且不超过目标带宽")
        resolved.update(depth=_finite(spec.get("depth", 0.8), "调制深度", 0.01, 1.0),
                        message_bandwidth=message_bw,
                        bandwidth_actual=2.0 * message_bw)
    elif mode == "fm":
        message_bw = _finite(spec.get("message_bandwidth", bandwidth / 4.0),
                             "消息带宽", 0.0, bandwidth)
        # deviation 为 RMS 频偏，99% 占用带宽 ≈ 5.3 × deviation（经验标定），
        # 自动模式取 0.19 × 带宽 使实际占用带宽贴合目标带宽。
        deviation = _finite(spec.get("deviation", 0.19 * bandwidth), "频偏", 0.0, bandwidth)
        if not 0 < message_bw < bandwidth / 2.0:
            raise ValueError("FM 消息带宽必须大于 0 且小于目标带宽的一半")
        if not 0 < deviation <= bandwidth:
            raise ValueError("FM 频偏必须大于 0 且不大于目标带宽")
        resolved.update(message_bandwidth=message_bw, deviation=deviation,
                        bandwidth_actual=5.3 * deviation)
    elif mode == "ssb":
        _check_band(offset, bandwidth, rate, mode)
        side = str(spec.get("side", "usb"))
        if side not in ("usb", "lsb"):
            raise ValueError("SSB 边带必须为 usb 或 lsb")
        resolved.update(side=side, bandwidth_actual=bandwidth)
    elif mode in ("ask2", "qpsk", "qam16", "qam64"):
        _check_band(offset, bandwidth, rate, mode)
        alpha = _finite(spec.get("alpha", 0.35), "滚降系数", 0.05, 1.0)
        pulse = "rrc"
        if mode == "ask2":
            pulse = str(spec.get("pulse", "rrc"))
            if pulse not in ("rrc", "rect"):
                raise ValueError("2ASK 脉冲成形必须为 rrc 或 rect")
        sps = max(2, int(np.ceil(rate / (bandwidth / (1 + alpha)))))
        symbol_rate = rate / sps
        resolved.update(alpha=alpha, pulse=pulse, sps=int(sps),
                        symbol_rate=symbol_rate,
                        bandwidth_actual=symbol_rate * (1 + alpha) if pulse == "rrc" else symbol_rate)
    elif mode == "fh_rc":
        points, hop_bw, span = _hop_points(spec, rate, offset, bandwidth, mode)
        hop_rate = _finite(spec.get("hop_rate", 100.0), "跳速", 0.0, rate)
        deviation = _finite(spec.get("deviation", hop_bw / 4.0), "频偏", 0.0, hop_bw)
        symbol_rate = _finite(spec.get("symbol_rate", hop_bw / 2.0), "符号速率", 0.0, rate)
        if not 0 < hop_rate <= rate:
            raise ValueError("跳速必须大于 0 且不超过采样率")
        if not 0 < deviation < hop_bw:
            raise ValueError("每跳 2FSK 频偏必须大于 0 且小于每跳带宽")
        if not 0 < symbol_rate <= rate:
            raise ValueError("符号速率必须大于 0 且不超过采样率")
        resolved.update(hop_rate=hop_rate, hop_points=list(points),
                        hop_bandwidth=hop_bw, hop_span=span, deviation=deviation,
                        symbol_rate=symbol_rate, bandwidth_actual=span + hop_bw)
    else:  # 图传跳频样式 fh_video。
        points, hop_bw, span = _hop_points(spec, rate, offset, bandwidth, mode)
        hop_rate = _finite(spec.get("hop_rate", 50.0), "跳速", 0.0, rate)
        subcarriers = int(spec.get("subcarriers", 64))
        cp_ratio = _finite(spec.get("cp_ratio", 0.25), "循环前缀比例", 0.0, 0.5)
        if not 0 < hop_rate <= rate:
            raise ValueError("跳速必须大于 0 且不超过采样率")
        if not 8 <= subcarriers <= 1024:
            raise ValueError("OFDM 子载波数必须为 8～1024 的整数")
        if not 0 < cp_ratio <= 0.5:
            raise ValueError("循环前缀比例必须大于 0 且不超过 0.5")
        n_fft = int(max(subcarriers + 4, round(rate * subcarriers / hop_bw)))
        n_cp = max(1, int(np.round(n_fft * cp_ratio)))
        resolved.update(hop_rate=hop_rate, hop_points=list(points),
                        hop_bandwidth=hop_bw, hop_span=span, subcarriers=int(subcarriers),
                        cp_ratio=cp_ratio, fft_size=int(n_fft), cp_samples=int(n_cp),
                        subcarrier_spacing=rate / n_fft,
                        occupied_bandwidth=(subcarriers - 1) * rate / n_fft,
                        bandwidth_actual=span + hop_bw)
    return resolved


def generate_iq(sample_rate, duration, signals, noise=None, seed=0):
    """生成多信号叠加的 IQ 记录及可复核真值摘要。

    参数：sample_rate 为 Hz，duration 为秒；signals 为最多 16 项参数字典列表；noise 可配置带宽 Hz、最强信号带内 snr_db
    或纯噪声 power_dbfs；seed 为 32 位非负整数。

    返回：(samples, summary)：连续 complex64 一维 IQ 与 iq_generator_v1
    摘要，包含逐信号实际功率、占用带宽、随机种子和噪声口径。

    算法与边界：逐信号先规划，按固定顺序派生独立随机流并叠加 complex128 波形，最后转换 complex64。背景噪声按
    N0=P_ref/(B_ref·10^(SNR/10)) 定标；须覆盖最强信号频带。仅启用噪声时允许空信号列表；无信号且无噪声、点数超限或配置无效均报错，不削峰。
    """
    rate = validate_rate(sample_rate)
    duration_s = _finite(duration, "持续时间", 0.0, 3600.0)
    if duration_s <= 0:
        raise ValueError("持续时间必须大于 0")
    count = int(np.round(rate * duration_s))
    if not 1 <= count <= MAX_SAMPLES:
        raise ValueError(f"采样点数 {count} 超出 1～{MAX_SAMPLES} 范围，请减小持续时间或采样率")
    if not isinstance(signals, (list, tuple)) or len(signals) > MAX_SIGNALS:
        raise ValueError(f"信号列表必须包含 0～{MAX_SIGNALS} 个信号")
    has_noise = bool(isinstance(noise, dict) and noise.get("enabled", True))
    if len(signals) == 0 and not has_noise:
        raise ValueError("至少包含 1 个信号，或启用背景噪声")
    seed = int(seed)
    if not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("随机种子必须为 0～4294967295 的整数")
    rng = np.random.default_rng(seed)
    t = np.arange(count) / rate
    record = np.zeros(count, dtype=np.complex128)
    summaries = []
    for index, spec in enumerate(signals):
        plan = plan_signal(spec, rate)
        child = np.random.default_rng(int(rng.integers(0, 2 ** 32)))
        signal, info = _synthesize(plan, rate, count, t, child)
        power = float(np.mean(np.abs(signal) ** 2))
        entry = {key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
                 for key, value in plan.items()}
        entry.update({key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
                      for key, value in info.items()})
        entry["power_dbfs_actual"] = float(10.0 * np.log10(power)) if power > 0 else float("-inf")
        record += signal
        summaries.append(entry)
    noise_enabled = has_noise
    noise_bandwidth = rate
    noise_power = 0.0
    noise_psd = None
    snr_db = None
    reference_index = None
    half_band = rate / 2.0
    if noise_enabled:
        noise_bandwidth = _finite(noise.get("bandwidth", rate), "噪声带宽", 0.0, rate)
        if not 0 < noise_bandwidth <= rate:
            raise ValueError("噪声带宽必须大于 0 且不超过采样率")
        half_band = noise_bandwidth / 2.0
        if summaries:
            # 参考信号取实测平均功率最大者；带内 SNR 定义要求噪声频带
            # 完整覆盖其占用频带，否则无法按功率谱密度折算。
            reference_index = int(np.argmax([entry["power_dbfs_actual"] for entry in summaries]))
            reference = summaries[reference_index]
            snr_db = _finite(noise.get("snr_db", 20.0), "带内信噪比", -60.0, 120.0)
            low, high = occupied_interval(reference["offset"], reference["bandwidth_actual"],
                                          reference["mode"], reference.get("side"))
            if low < -half_band or high > half_band:
                label = MODE_NAMES.get(reference["mode"], reference["mode"])
                raise ValueError(
                    f"噪声带宽 {noise_bandwidth:g} Hz 未覆盖最强信号（{label}）的占用频带 "
                    f"{low:g}～{high:g} Hz，无法折算带内信噪比；请增大噪声带宽或调整该信号")
            # N0 = P_ref / (B_ref · 10^(SNR/10))，噪声总功率 P_n = N0 · B_n。
            noise_psd = (10 ** (reference["power_dbfs_actual"] / 10.0)
                         / (reference["bandwidth_actual"] * 10 ** (snr_db / 10.0)))
            noise_power = noise_psd * noise_bandwidth
        else:
            noise_power = 10 ** (_finite(noise.get("power_dbfs", -20.0), "噪声功率", -200.0, 0.0) / 10.0)
            noise_psd = noise_power / noise_bandwidth
        if noise_power > 0:
            record += _band_noise(rng, count, rate, half_band) * np.sqrt(noise_power)
    for entry in summaries:
        entry["snr_inband_db"] = None
        low, high = occupied_interval(entry["offset"], entry["bandwidth_actual"],
                                      entry["mode"], entry.get("side"))
        overlap = min(high, half_band) - max(low, -half_band)
        # 未被噪声频带覆盖的信号按重叠部分折算，完全落在带外时无定义。
        if noise_psd and overlap > 0:
            entry["snr_inband_db"] = float(
                entry["power_dbfs_actual"] - 10.0 * np.log10(noise_psd * overlap))
    samples = np.ascontiguousarray(record, dtype=np.complex64)
    summary = {
        "algorithm": "iq_generator_v1",
        "sample_rate_hz": rate,
        "sample_count": int(count),
        "duration_s": float(count / rate),
        "seed": seed,
        "signals": summaries,
        "noise": {"enabled": noise_enabled, "bandwidth": noise_bandwidth,
                  "power_dbfs": float(10.0 * np.log10(noise_power)) if noise_power > 0 else None,
                  "power_dbfs_per_hz": float(10.0 * np.log10(noise_psd)) if noise_psd else None,
                  "snr_db": float(snr_db) if snr_db is not None else None,
                  "snr_definition": "inband_snr_v1",
                  "snr_reference_index": reference_index},
        "peak_dbfs": float(10.0 * np.log10(np.max(np.abs(samples) ** 2))),
    }
    return samples, summary


def make_demo(sample_rate=48000.0, count=8192, seed=7):
    """生成可重复的数学双音演示记录。

    参数：sample_rate 为 Hz；count 为 1～MAX_SAMPLES 的整数；seed 为随机种子。

    返回：长度 count 的 complex64 数组。

    算法与边界：叠加 +rate/16 与 -rate/8 两个复音及固定幅度复高斯噪声；不代表通信调制模型。采样率或点数非法时抛出 ValueError。
    """
    rate = validate_rate(sample_rate)
    if isinstance(count, bool) or int(count) != count or not 1 <= count <= MAX_SAMPLES:
        raise ValueError(f"演示采样点数必须为 1～{MAX_SAMPLES:,} 的整数")
    t = np.arange(int(count)) / rate
    rng = np.random.default_rng(seed)
    # 可重复的数学双音与噪声，仅用于演示，不代表通信调制模型。
    x = np.exp(2j * np.pi * rate / 16 * t)
    x += 0.35 * np.exp(-2j * np.pi * rate / 8 * t)
    x += 0.025 * (rng.standard_normal(int(count)) + 1j * rng.standard_normal(int(count)))
    return x.astype(np.complex64)
