"""Qt-independent numerical implementation, also compiled with Cython.

Generic statistics, scientific plots and test-signal IQ generation are
implemented here. Only NumPy is used so the module stays compilable with
Cython for the binary core build.
"""

import numpy as np

MAX_SAMPLES = 16_000_000
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


def _finite(value, name, minimum=None, maximum=None):
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} 必须为有限数值")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} 不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} 不能大于 {maximum}")
    return number


def _signal_meta(spec, rate):
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
    if mode == "ssb":
        okay = offset - width >= -rate / 2 and offset + width <= rate / 2
    else:
        okay = abs(offset) + width / 2 <= rate / 2
    if not okay:
        raise ValueError(f"频点 {offset:g} Hz 与带宽 {width:g} Hz 超出 ±{rate / 2:g} Hz 的基带范围")


def occupied_interval(offset, width, mode, side=None):
    """Two-sided frequency interval (Hz) occupied by one signal.

    SSB is one-sided (upper or lower sideband starting at ``offset``);
    every other mode is centred on ``offset``. Used to verify that the
    background noise band covers the signal, which the in-band SNR
    definition requires.
    """
    if mode == "ssb":
        return (offset - width, offset) if side == "lsb" else (offset, offset + width)
    return offset - width / 2.0, offset + width / 2.0


def _hop_points(spec, rate, offset, bandwidth, mode):
    """Resolve the hop set; return ``(points, hop_bw, span)``.

    三参数模型：单跳带宽 hop_bw、跳频中心跨度 span 与整体频带范围
    bandwidth 满足恒等式 ``bandwidth = span + hop_bw``。频点均布在
    ``offset ± span/2``，最外侧信道边缘恰好落在 ``offset ± bandwidth/2``。
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
    """Unit-energy root raised-cosine filter taps, t in symbol units."""
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
    """Upsample symbols by sps, pulse-shape and trim to count samples."""
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
    """Unit-power noise band-limited to `band` Hz (band may be one-sided)."""
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
    """Unit-power complex noise occupying the one-sided band [f_low, f_high]."""
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
    if offset:
        return baseband * np.exp(2j * np.pi * offset * t)
    return baseband


def _scale_power(signal, power_dbfs):
    measured = np.mean(np.abs(signal) ** 2)
    if not measured > 0:
        raise ValueError("信号功率为零，无法缩放")
    amplitude = 10 ** (power_dbfs / 20.0) / np.sqrt(measured)
    return signal * amplitude


def _amplitude_modulation(rng, t, count, rate, spec):
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
    return np.round(np.arange(hops + 1) * count / hops).astype(np.int64)


def _fh_remote_control(rng, t, count, rate, spec):
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
    """Validate one signal specification and fill in auto-derived parameters.

    This is the shared rule set used by the GUI parameter dialog and by
    :func:`generate_iq`; no samples are produced here.
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
    else:  # fh_video
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
    """Generate a multi-signal IQ record and return ``(samples, summary)``.

    ``signals`` is a non-empty list (max 16) of parameter dicts, each with
    ``mode``, optional ``offset`` (baseband offset in Hz), ``power_dbfs``,
    ``bandwidth`` and mode-specific parameters (see :func:`plan_signal`).
    ``noise`` is an optional dict with ``enabled``, ``bandwidth`` (Hz,
    two-sided, default full band), ``snr_db`` (in-band SNR of the strongest
    signal) or ``power_dbfs`` (absolute total power, used when there are no
    signals). In-band SNR is the signal mean power divided by the noise
    power inside the same occupied bandwidth: the total noise power is
    spread uniformly over ``bandwidth``, giving a power spectral density
    ``power / bandwidth``, and each signal's in-band noise is that density
    times its own ``bandwidth_actual``. The noise band must cover the
    strongest signal's occupied band. A fixed ``seed`` reproduces identical
    samples.
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


def _stft_psd(x, rate, nfft):
    """Shared Hanning STFT producing a two-sided PSD (linear, per Hz).

    ``psd`` has one row per frame with the same scaling as the spectrum
    page, so integrating a band gives its power directly:
    ``P_band = psd[:, lo:hi+1].sum() * (rate / nfft)``. At most 512 frames
    are produced to bound memory and result size.
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


def analyze(samples, sample_rate, nfft=256):
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
    # Power spectral density is two-sided for complex input; never double it.
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


def _marginal_modes(values):
    """Number of isolated modes in the smoothed 1-D histogram of ``values``."""
    span = 3.0 * float(np.std(values)) + 1e-12
    hist, _ = np.histogram(values, bins=64, range=(-span, span))
    hist = hist.astype(np.float64)
    smoothed = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peak_max = float(smoothed.max())
    if peak_max <= 0:
        return 0
    idx = [i for i in range(1, 63)
           if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
           and smoothed[i] >= 0.12 * peak_max]
    order = sorted(idx, key=lambda i: -smoothed[i])
    keep = []
    for i in order:
        value = float(smoothed[i])
        left = max((j for j in keep if j < i), default=None)
        right = min((j for j in keep if j > i), default=None)
        if left is not None and right is not None:
            low = min(float(smoothed[left + 1:i].min()), float(smoothed[i + 1:right].min()))
        elif left is not None:
            low = min(float(smoothed[left + 1:i].min()), float(smoothed[i + 1:64].min()))
        elif right is not None:
            low = min(float(smoothed[0:i].min()), float(smoothed[i + 1:right].min()))
        else:
            low = min(float(smoothed[0:i].min()), float(smoothed[i + 1:64].min()))
        # 峰须与两侧更高峰之间存在低于 75% 峰高的谷，否则视为肩部抖动
        if low <= 0.75 * value:
            keep.append(i)
    return len(keep)


def classify_modulation(samples, max_points=20000):
    """Heuristic digital/analog classification from the raw I/Q samples.

    Returns ``(classification, cluster_estimate)``. Three simple rules are
    applied in order:

    1. Near constant-envelope signals (ring-like scatter: FM/FSK/CW, and the
       default AM whose scatter is a dense disc) are analog.
    2. Platykurtic I or Q marginals (discrete level structure: PSK/QAM/ASK)
       are digital; ``cluster_estimate`` is the product of the marginal mode
       counts (a rough constellation size hint).
    3. Multi-modal I and Q marginals (e.g. on-off keying with a carrier
       line) are digital.

    Continuous clouds (noise, SSB, OFDM-like) fall through to analog. This
    is a display heuristic for the GUI, not a trained recognition model;
    the user can override the decision manually.
    """
    data = np.asarray(samples)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("需要一维非空数组")
    if data.dtype.kind not in "iufc":
        raise ValueError("只支持实数或复数数值数组")
    step = max(1, int(np.ceil(data.size / max_points)))
    view = np.asarray(data[::step], dtype=np.complex128)
    real, imag = view.real, view.imag
    if not (np.isfinite(real).all() and np.isfinite(imag).all()):
        raise ValueError("数据包含 NaN 或 Inf")
    if float(np.std(real)) == 0.0 and float(np.std(imag)) == 0.0:
        return "analog", 0
    magnitude = np.abs(view)
    ring_ratio = float(np.std(magnitude) / max(np.mean(magnitude), 1e-12))
    if ring_ratio < 0.25:
        return "analog", 0

    def kurtosis(values):
        values = values - values.mean()
        return float(np.mean(values ** 4) / max(float(np.mean(values ** 2)) ** 2, 1e-24))

    ki = kurtosis(real) if float(np.std(real)) > 1e-12 else 99.0
    kq = kurtosis(imag) if float(np.std(imag)) > 1e-12 else 99.0
    modes_i = _marginal_modes(real) if float(np.std(real)) > 1e-12 else 1
    modes_q = _marginal_modes(imag) if float(np.std(imag)) > 1e-12 else 1
    estimate = int(max(2, min(modes_i * modes_q, 256)))
    if min(ki, kq) < 2.45:
        return "digital", estimate
    if modes_i >= 2 and modes_q >= 2:
        return "digital", estimate
    return "analog", 0


def spectrum_row(data, pos, nfft=256, rate=1.0):
    """PSD (dB) of the analysis window ending at sample ``pos``.

    Used by real-time playback: the window covers ``data[pos-nfft:pos]``
    (zero-padded on the left when fewer samples are available) with the
    same Hanning window and two-sided scaling as :func:`analyze`, so
    playback frames match the overview spectrogram. Returns
    ``(frequencies, psd_db)``.
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


# ---------------------------------------------------------------------------
# 信号检测与参数估计
#
# 输出遵守冻结契约 ``detect_result_v1``（见 signal_analysis.evaluation）：
# 传统能量检测与后续 AI/ONNX 检测器输出同一结构，可直接比对。
# 频域一律是复基带偏移，不虚构射频参数。
# ---------------------------------------------------------------------------

DETECT_ALGORITHM = "energy_detect_v1"
DETECT_SNR_DEFINITION = "inband_snr_v1"
_OCCUPIED_RATIO = 0.99
#: 带内 SNR 估计下限（dB）：低于此值只报“几乎全为噪声”
_SNR_FLOOR_DB = -20.0
_DETECT_DEFAULTS = {
    "nfft": 512,
    # 平均 PSD 的逐点起伏远小于 0.5 dB（512 帧平均），故 3 dB 门限已很保守
    "threshold_db": 3.0,
    "band_threshold_db": 0.0,
    "min_bandwidth_hz": 0.0,
    "min_duration_s": 0.0,
    "max_detections": 32,
    "merge_bins": 0,
}


def _detect_config(rate, duration, config):
    """Validate and resolve the detector configuration (frozen contract)."""
    if config is None:
        settings = {}
    elif isinstance(config, dict):
        settings = dict(config)
    else:
        raise ValueError("检测配置必须是字典")
    unknown = set(settings) - set(_DETECT_DEFAULTS)
    if unknown:
        raise ValueError(f"未知的检测参数：{', '.join(sorted(unknown))}")
    nfft = settings.get("nfft", _DETECT_DEFAULTS["nfft"])
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= int(nfft) <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    resolution = rate / nfft
    threshold_db = _finite(settings.get("threshold_db", _DETECT_DEFAULTS["threshold_db"]),
                           "检测门限", 0.0, 80.0)
    band_threshold_db = _finite(settings.get("band_threshold_db", 0.0),
                                "带宽测量门限", 0.0, 80.0)
    if band_threshold_db <= 0:
        # “高门限检测、低门限量带宽”：占用带宽按噪声底以上 1.5 dB（默认）
        # 测量，才能看到 AM 这种载波主导信号被压低的边带
        band_threshold_db = max(1.0, 0.5 * threshold_db)
    if band_threshold_db > threshold_db:
        raise ValueError("带宽测量门限不得高于检测门限")
    min_bandwidth = _finite(settings.get("min_bandwidth_hz", 0.0), "最小带宽", 0.0, rate)
    if min_bandwidth <= 0:
        # 默认要求至少 3 个频点宽，抑制单点毛刺
        min_bandwidth = 3.0 * resolution
    min_duration = _finite(settings.get("min_duration_s", 0.0), "最短持续时间", 0.0, duration)
    max_detections = settings.get("max_detections", _DETECT_DEFAULTS["max_detections"])
    if isinstance(max_detections, bool) or int(max_detections) != max_detections \
            or not 1 <= int(max_detections) <= 256:
        raise ValueError("最大目标数必须为 1～256 的整数")
    merge_bins = settings.get("merge_bins", 0)
    if isinstance(merge_bins, bool) or int(merge_bins) != merge_bins or not 0 <= int(merge_bins) <= nfft // 4:
        raise ValueError(f"谱合并宽度必须为 0～{nfft // 4} 的整数")
    merge_bins = int(merge_bins) or max(2, nfft // 128)
    return {
        "nfft": nfft,
        "threshold_db": threshold_db,
        "band_threshold_db": band_threshold_db,
        "min_bandwidth_hz": min_bandwidth,
        "min_duration_s": min_duration,
        "max_detections": int(max_detections),
        "merge_bins": merge_bins,
        "freq_resolution_hz": resolution,
        "min_bandwidth_bins": max(1, int(np.ceil(min_bandwidth / resolution))),
    }


def _noise_floor_db(psd_db, threshold_db, iterations=4):
    """Robust noise floor: iterative median + MAD sigma clipping."""
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
    out = np.asarray(mask, dtype=bool).copy()
    for shift in range(1, int(radius) + 1):
        out[shift:] |= mask[:-shift]
        out[:-shift] |= mask[shift:]
    return out


def _binary_close(mask, radius):
    """1-D morphological closing: fill spectral ripple narrower than ``radius``."""
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    filled = _binary_dilate(mask, radius)
    return ~_binary_dilate(~filled, radius)


def _true_runs(mask):
    """Inclusive ``(start, stop)`` index pairs of consecutive True values."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop) - 1) for start, stop in zip(edges[::2], edges[1::2])]


def _occupied_span(psd_linear, first, last, ratio=_OCCUPIED_RATIO):
    """Bin indices containing ``ratio`` of the band power (99% occupancy).

    Kept as the traceability reference: the band reported by
    :func:`detect_signals` is the noise-referenced detected run (see the
    function docstring), while this helper provides the classic
    power-occupancy interval used when comparing with spectrum-management
    tools.
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
_SESSION_OVERLAP_RATIO = 0.2


def _band_regions(detect_mask, edge_mask):
    """Group detection runs by the measured-band component that contains them.

    ``detect_mask`` (high threshold) provides the *evidence*, ``edge_mask``
    (low threshold) the *band*. Grouping by component is what keeps one
    emitter with an internal dip from being reported twice: both detection
    runs of such an emitter belong to the same low-threshold component, so
    they produce exactly one band.
    """
    regions = []
    for first, last in _true_runs(edge_mask):
        inner = [(a, b) for a, b in _true_runs(detect_mask)
                 if first <= a and b <= last]
        if inner:
            regions.append((first, last, inner))
    return regions


def _overlap_seconds(first, second):
    """Total time overlap (s) between two lists of disjoint intervals."""
    total = 0.0
    for a_start, a_end in first:
        for b_start, b_end in second:
            total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return total


def _same_session(first, second, gap_ratio=_SESSION_GAP_RATIO,
                  overlap_ratio=_SESSION_OVERLAP_RATIO):
    """True when two frequency runs are hops of the same session.

    Hops of one frequency-hopping signal occupy neighbouring bands at
    different times, while two independent emitters are active at the same
    time. Both conditions are required, so time-interleaved emitters in
    distant bands and simultaneous emitters in adjacent bands stay separate.
    Bandwidths are derived from the band edges, never cached, so merged
    items stay consistent.
    """
    gap = max(first["f_low_hz"] - second["f_high_hz"],
              second["f_low_hz"] - first["f_high_hz"])
    first_width = first["f_high_hz"] - first["f_low_hz"]
    second_width = second["f_high_hz"] - second["f_low_hz"]
    if gap > gap_ratio * max(first_width, second_width):
        return False
    overlap = _overlap_seconds(first["intervals"], second["intervals"])
    shorter = min(first["active_seconds"], second["active_seconds"])
    return overlap <= overlap_ratio * shorter


def _combine_sessions(left, right):
    """Power-weighted union of two provisional detections."""
    power = left["power_linear"] + right["power_linear"]
    centroid = ((left["power_linear"] * left["centroid_hz"]
                 + right["power_linear"] * right["centroid_hz"]) / power
                if power > 0 else 0.5 * (left["centroid_hz"] + right["centroid_hz"]))
    intervals = sorted(left["intervals"] + right["intervals"])
    active_seconds = sum(end - start for start, end in intervals)
    f_low = min(left["f_low_hz"], right["f_low_hz"])
    f_high = max(left["f_high_hz"], right["f_high_hz"])
    return {
        "center_hz": 0.5 * (f_low + f_high),
        "centroid_hz": float(centroid),
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "power_linear": float(power),
        "intervals": intervals,
        "active_seconds": float(active_seconds),
        "t_start_s": intervals[0][0],
        "t_end_s": intervals[-1][1],
        "bin_count": left["bin_count"] + right["bin_count"],
        "sub_bands": left["sub_bands"] + right["sub_bands"],
        "occupied_f_low_hz": min(left["occupied_f_low_hz"], right["occupied_f_low_hz"]),
        "occupied_f_high_hz": max(left["occupied_f_high_hz"], right["occupied_f_high_hz"]),
    }


def _merge_sessions(items, max_detections=256):
    """Merge hop runs into one instance per session (user agreed semantics)."""
    groups = [dict(item) for item in items]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            for j in range(len(groups) - 1, i, -1):
                if not _same_session(groups[i], groups[j]):
                    continue
                groups[i] = _combine_sessions(groups[i], groups[j])
                groups.pop(j)
                changed = True
                break
            if changed:
                break
    groups.sort(key=lambda item: -item["power_linear"])
    return groups[:max_detections]


def detect_signals(samples, sample_rate, config=None):
    """Spectrum-energy detector with parameter estimation.

    Pipeline (all NumPy, Cython-compilable):

    1. Hanning STFT shared with :func:`analyze`, then the **time-averaged**
       PSD: averaging keeps band power and in-band SNR unbiased, and because
       only one hop channel is active at a time a hopping session still
       shows every channel above the floor.
    2. Noise floor by iterative median + MAD sigma clipping.
    3. Threshold on the averaged PSD, morphological closing to avoid ripping
       one signal into pieces, then discard runs narrower than
       ``min_bandwidth_hz``.
    4. Per run: the band is the noise-referenced extent of the run (edges
       aligned to bin edges), band power and in-band SNR
       ``10·log10(P_band / (N0·B))`` where ``N0`` is the estimated noise
       power spectral density (``inband_snr_v1``, identical to the
       generator convention). ``center_hz`` is the band midpoint, which is
       exactly how the generator truth defines the centre of the occupied
       interval and is the only definition that stays meaningful for a
       hopping session (the energy centroid wanders with the random hop
       dwell); the power-weighted centroid is still reported as
       ``centroid_hz``. The classic 99% power-occupancy interval is
       reported as ``occupied_f_low_hz``/``occupied_f_high_hz`` for
       traceability; it is not used as ``bandwidth_hz`` because a
       carrier-dominated signal (AM) would collapse onto its carrier line,
       while ``bandwidth_actual`` from the generator is the band-limited
       spectrum extent that the noise-referenced run measures.
    5. Time support from the per-frame band power.

    Hopping signals follow the agreed "one session, one instance" rule: the
    hop runs of one session are merged when they sit in neighbouring bands
    and are active at different times, so a frequency hopping signal is
    reported as one detection whose band equals the whole session band —
    which is exactly what the generator truth (``bandwidth_actual``)
    reports. ``session_id`` is non-null only for such merged sessions, and
    ``hopping`` marks them, so a later multi-instance detector can reuse
    both fields.

    Returns ``(summary, arrays)`` like :func:`analyze`; the summary follows
    the frozen ``detect_result_v1`` contract.
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    duration = float(x.size / rate)
    resolved = _detect_config(rate, duration, config)
    nfft = resolved["nfft"]
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]

    psd, frequencies, starts, hop = _stft_psd(x, rate, nfft)
    psd_average = psd.mean(axis=0)
    psd_db = 10.0 * np.log10(np.maximum(psd_average, 1e-30))
    noise_floor_db = _noise_floor_db(psd_db, threshold_db)
    threshold_dbfs = noise_floor_db + threshold_db
    noise_linear = 10.0 ** (noise_floor_db / 10.0)

    mask = _binary_close(psd_db > threshold_dbfs, resolved["merge_bins"])
    runs = [(start, stop) for start, stop in _true_runs(mask)
            if stop - start + 1 >= resolved["min_bandwidth_bins"]]
    detect_mask = np.zeros(psd_db.size, dtype=bool)
    for start, stop in runs:
        detect_mask[start:stop + 1] = True
    # 带宽测量门限（低于检测门限）：检出用高门限，量带宽用低门限
    if resolved["band_threshold_db"] < threshold_db:
        edge_mask = _binary_close(
            psd_db > noise_floor_db + resolved["band_threshold_db"], resolved["merge_bins"])
    else:
        edge_mask = mask
    regions = _band_regions(detect_mask, edge_mask)

    frame_starts = np.clip(starts / rate, 0.0, duration)
    frame_ends = np.clip((starts + nfft) / rate, 0.0, duration)
    candidates = []
    for first, last, inner in regions:
        occupancy_low, occupancy_high = _occupied_span(psd_average, first, last)
        weights = np.where(detect_mask[first:last + 1], psd_average[first:last + 1], 0.0)
        weight_sum = float(psd_average[first:last + 1].sum())
        if weight_sum <= 0:
            continue
        signal_sum = float(weights.sum())
        band_frequencies = frequencies[first:last + 1]
        centroid = (float((weights * band_frequencies).sum() / signal_sum)
                    if signal_sum > 0 else float(band_frequencies.mean()))
        f_low = float(band_frequencies[0] - resolution / 2.0)
        f_high = float(band_frequencies[-1] + resolution / 2.0)
        bandwidth = f_high - f_low
        band_power = float(weight_sum * resolution)
        # 时间支撑：按帧的带内功率与噪声参考功率比较
        frame_power = np.asarray(psd[:, first:last + 1].sum(axis=1), dtype=np.float64) * resolution
        active = frame_power > noise_linear * bandwidth * 10.0 ** (threshold_db / 10.0)
        if not active.any():
            active = np.ones(frame_power.size, dtype=bool)
        active_index = np.flatnonzero(active)
        t_start = float(frame_starts[active_index[0]])
        t_end = float(frame_ends[active_index[-1]])
        if t_end - t_start < resolved["min_duration_s"]:
            continue
        intervals = [(float(frame_starts[position]), float(frame_ends[position]))
                     for position in active_index]
        candidates.append({
            "center_hz": 0.5 * (f_low + f_high),
            "centroid_hz": centroid,
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "power_linear": band_power,
            "intervals": intervals,
            "active_seconds": float(t_end - t_start),
            "t_start_s": t_start,
            "t_end_s": t_end,
            "bin_count": int(sum(stop - start + 1 for start, stop in inner)),
            "sub_bands": 1,
            "occupied_f_low_hz": float(frequencies[occupancy_low] - resolution / 2.0),
            "occupied_f_high_hz": float(frequencies[occupancy_high] + resolution / 2.0),
        })

    groups = _merge_sessions(candidates, resolved["max_detections"])
    groups.sort(key=lambda item: item["center_hz"])
    detections = []
    for index, item in enumerate(groups, start=1):
        bandwidth = item["f_high_hz"] - item["f_low_hz"]
        # 噪声参考功率：同带宽白噪声功率 N0·B（inband_snr_v1 口径）
        noise_band = max(noise_linear * bandwidth, 1e-30)
        signal_power = max(item["power_linear"] - noise_band, 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        hopping = item["sub_bands"] > 1
        detections.append({
            "id": index,
            "method": "energy",
            "center_hz": round(item["center_hz"], 3),
            "centroid_hz": round(item["centroid_hz"], 3),
            "bandwidth_hz": round(bandwidth, 3),
            "f_low_hz": round(item["f_low_hz"], 3),
            "f_high_hz": round(item["f_high_hz"], 3),
            "t_start_s": round(item["t_start_s"], 6),
            "t_end_s": round(item["t_end_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(item["power_linear"], 1e-30))), 3),
            "snr_db": round(snr_db, 3),
            "session_id": index if hopping else None,
            "confidence": round(float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)), 3),
            "hopping": hopping,
            "sub_bands": int(item["sub_bands"]),
            "bin_count": int(item["bin_count"]),
            "occupied_f_low_hz": round(item["occupied_f_low_hz"], 3),
            "occupied_f_high_hz": round(item["occupied_f_high_hz"], 3),
        })

    summary = {
        "contract": "detect_result_v1",
        "algorithm": DETECT_ALGORITHM,
        "snr_definition": DETECT_SNR_DEFINITION,
        "frequency_reference": "baseband_offset",
        "sample_rate_hz": rate,
        "sample_count": int(x.size),
        "duration_s": duration,
        "nfft": nfft,
        "hop_samples": hop,
        "frame_count": int(psd.shape[0]),
        "config": {
            "nfft": nfft,
            "threshold_db": threshold_db,
            "band_threshold_db": resolved["band_threshold_db"],
            "min_bandwidth_hz": round(resolved["min_bandwidth_hz"], 6),
            "min_duration_s": resolved["min_duration_s"],
            "max_detections": resolved["max_detections"],
            "merge_bins": resolved["merge_bins"],
        },
        "freq_resolution_hz": resolution,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": round(threshold_dbfs, 3),
        "detections": detections,
    }
    box_rows = [[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                for item in detections]
    arrays = {
        "frequency": frequencies,
        "frame_time": (starts + (nfft - 1) / 2.0) / rate,
        "spectrogram_db": (10.0 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        # 检测依据：时间平均 PSD（与频谱页口径一致）；中位数 PSD 作对照
        "spectrum_db": 10.0 * np.log10(np.maximum(psd_average, 1e-30)),
        "spectrum_median_db": 10.0 * np.log10(np.maximum(np.median(psd, axis=0), 1e-30)),
        "threshold_db": np.array([threshold_dbfs], dtype=np.float64),
        "noise_floor_db": np.array([noise_floor_db], dtype=np.float64),
        "detection_boxes": np.array(box_rows, dtype=np.float64).reshape(-1, 4),
        "detection_id": np.array([item["id"] for item in detections], dtype=np.int64),
        "detection_snr_db": np.array([item["snr_db"] for item in detections], dtype=np.float64),
        "detection_power_dbfs": np.array([item["power_dbfs"] for item in detections], dtype=np.float64),
    }
    return summary, arrays
