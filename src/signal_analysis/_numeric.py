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


# ---------------------------------------------------------------------------
# 跳频逐跳参数估计（算法 ``hop_track_v1``，结果契约 ``fh_hops_v1``）
#
# 与 :func:`detect_signals` 的「一段会话一个实例」互补而不替代：那条通路先对
# 时间求平均，只能在平均 PSD 上给出会话级频带；本通路不做时间平均，逐帧提取
# 游程、再沿时间把游程链成「跳」，因此能给出逐跳的跳频点、驻留时长与跳速。
# 冻结契约 ``detect_result_v1`` 与 :func:`detect_signals` 的行为完全不变。
#
# 全程复用同一份 STFT（:func:`_stft_psd`）与同一个带内信噪比口径
# （``inband_snr_v1``），所以逐跳结果可以直接与会话级结果并排比较。
# ---------------------------------------------------------------------------

HOP_ALGORITHM = "hop_track_v1"
HOP_CONTRACT = "fh_hops_v1"
#: 一跳至少占这么多帧，否则驻留过短、无法从时频面上可靠分辨
_HOP_MIN_DWELL_FRAMES = 4
_HOP_DEFAULTS = {
    "nfft": 512,
    # 单帧周期图只有约 2 个自由度，逐点起伏约 5.6 dB，远大于时间平均后的
    # 0.5 dB。故先按 smooth_frames 帧滑动平均把起伏压到约 2.8 dB，再取
    # 6 dB 门限，使单点虚警概率降到 1e-6 量级（否则 512 点 × 数百帧上会
    # 长出大量噪声游程）。逐跳的频域闭运算半径也相应取小（见下）。
    "threshold_db": 6.0,
    "smooth_frames": 4,
    "min_bandwidth_hz": 0.0,
    "min_dwell_s": 0.0,
    "merge_bins": 0,
    "max_gap_frames": 1,
    "max_gap_bins": 0,
    "transition_ratio": 1.5,
    "max_hops": 256,
}


def _hop_config(rate, duration, config):
    """Validate and resolve the per-hop tracker configuration.

    Mirrors :func:`_detect_config`: unknown keys are rejected instead of
    silently ignored, so a typo in the GUI/CLI never changes the algorithm
    without telling the user. The effective minimum dwell is forced up to
    :data:`_HOP_MIN_DWELL_FRAMES` analysis frames — below that a hop simply
    has no time-frequency signature to track.
    """
    if config is None:
        settings = {}
    elif isinstance(config, dict):
        settings = dict(config)
    else:
        raise ValueError("逐跳配置必须是字典")
    unknown = set(settings) - set(_HOP_DEFAULTS)
    if unknown:
        raise ValueError(f"未知的逐跳参数：{', '.join(sorted(unknown))}")
    nfft = settings.get("nfft", _HOP_DEFAULTS["nfft"])
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= int(nfft) <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    resolution = rate / nfft
    threshold_db = _finite(settings.get("threshold_db", _HOP_DEFAULTS["threshold_db"]),
                           "检测门限", 0.0, 80.0)
    smooth_frames = settings.get("smooth_frames", _HOP_DEFAULTS["smooth_frames"])
    if isinstance(smooth_frames, bool) or int(smooth_frames) != smooth_frames \
            or not 1 <= int(smooth_frames) <= 64:
        raise ValueError("时间平滑帧数必须为 1～64 的整数")
    min_bandwidth = _finite(settings.get("min_bandwidth_hz", 0.0), "最小带宽", 0.0, rate)
    if min_bandwidth <= 0:
        # 同一默认：至少 3 个频点宽，抑制单点毛刺
        min_bandwidth = 3.0 * resolution
    min_dwell = _finite(settings.get("min_dwell_s", 0.0), "最短驻留时间", 0.0, duration)
    min_dwell = max(min_dwell, _HOP_MIN_DWELL_FRAMES * nfft / rate)
    merge_bins = settings.get("merge_bins", 0)
    if isinstance(merge_bins, bool) or int(merge_bins) != merge_bins \
            or not 0 <= int(merge_bins) <= nfft // 4:
        raise ValueError(f"谱合并宽度必须为 0～{nfft // 4} 的整数")
    # 逐跳的闭运算半径取 1 个频点（桥接 2 点）：单帧上信道内部已经平坦，
    # 半径取大只会把相邻信道在跳变帧里焊成一条，掩盖真正的跳变。
    merge_bins = int(merge_bins) or max(1, nfft // 512)
    max_gap_frames = settings.get("max_gap_frames", _HOP_DEFAULTS["max_gap_frames"])
    if isinstance(max_gap_frames, bool) or int(max_gap_frames) != max_gap_frames \
            or not 0 <= int(max_gap_frames) <= 16:
        raise ValueError("允许的跟踪间断帧数必须为 0～16 的整数")
    max_gap_bins = settings.get("max_gap_bins", 0)
    if isinstance(max_gap_bins, bool) or int(max_gap_bins) != max_gap_bins \
            or not 0 <= int(max_gap_bins) <= nfft // 4:
        raise ValueError(f"频点跳变门限必须为 0～{nfft // 4} 的整数")
    max_gap_bins = int(max_gap_bins) or merge_bins
    transition_ratio = _finite(settings.get("transition_ratio", _HOP_DEFAULTS["transition_ratio"]),
                               "过渡帧宽度比", 1.1, 4.0)
    max_hops = settings.get("max_hops", _HOP_DEFAULTS["max_hops"])
    if isinstance(max_hops, bool) or int(max_hops) != max_hops or not 1 <= int(max_hops) <= 256:
        raise ValueError("最大跳数必须为 1～256 的整数")
    return {
        "nfft": nfft,
        "threshold_db": threshold_db,
        "smooth_frames": int(smooth_frames),
        "min_bandwidth_hz": min_bandwidth,
        "min_dwell_s": min_dwell,
        "merge_bins": merge_bins,
        "max_gap_frames": int(max_gap_frames),
        "max_gap_bins": max_gap_bins,
        "transition_ratio": transition_ratio,
        "max_hops": int(max_hops),
        "freq_resolution_hz": resolution,
        "min_bandwidth_bins": max(1, int(np.ceil(min_bandwidth / resolution))),
        "min_dwell_frames": max(_HOP_MIN_DWELL_FRAMES, int(np.ceil(min_dwell * rate / nfft))),
    }


def _smooth_psd(psd, window):
    """Sliding mean of ``window`` consecutive PSD frames, aligned to input.

    A single-frame periodogram has about two degrees of freedom, so its
    level fluctuates by roughly 5.6 dB from bin to bin; a fixed threshold
    applied to it would fire on noise everywhere. Averaging ``window``
    frames divides that fluctuation by ``sqrt(window)`` while keeping the
    original frame grid, which is what makes the per-frame mask usable. The
    mean is centred (edge-replicated at both ends) so frame ``i`` of the
    result still refers to frame ``i`` of the input.
    """
    if window <= 1:
        return psd
    half = window // 2
    padded = np.pad(psd, ((half, window - 1 - half), (0, 0)), mode="edge")
    # ``sliding_window_view`` 把窗口维度追加在**末尾**（不是挂在 axis 位置上），
    # 所以这里按最后一维求均值。
    frames = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return frames.mean(axis=-1)


def _frame_runs(psd_db, threshold_dbfs, merge_bins, min_bins):
    """Per-frame spectral runs above the threshold.

    One list of inclusive ``(first_bin, last_bin)`` pairs per frame. The
    frequency-domain closing fills ripples narrower than ``merge_bins``
    (2FSK tone spacing, OFDM in-band dips) before short runs are discarded,
    exactly as :func:`detect_signals` does on the averaged PSD.
    """
    rows = []
    for index in range(psd_db.shape[0]):
        mask = _binary_close(psd_db[index] > threshold_dbfs, merge_bins)
        rows.append([(start, stop) for start, stop in _true_runs(mask)
                     if stop - start + 1 >= min_bins])
    return rows


def _track_band(track):
    """Inclusive ``(first_bin, last_bin)`` of a track's accepted runs."""
    runs = track["runs"]
    return min(start for start, _ in runs), max(stop for _, stop in runs)


def _best_track(tracks, frame, first, last, max_gap_frames, max_gap_bins, excluded,
                allow_gap=True):
    """The open track that best explains one run (``None`` when there is none).

    Prefers the largest spectral overlap; a track whose band does not touch
    the run can still be linked when the gap is no wider than
    ``max_gap_bins`` bins — that is what carries a track across a single
    faded frame without letting it jump onto an unrelated emitter.
    ``allow_gap=False`` restricts the search to genuine spectral overlap,
    which is what a frame carrying several runs requires (see
    :func:`_link_tracks`).
    """
    ranked = []
    for track in tracks:
        if track["id"] in excluded:
            continue
        if frame - track["last_frame"] - 1 > max_gap_frames:
            continue
        low, high = _track_band(track)
        overlap = min(last, high) - max(first, low) + 1
        if overlap > 0:
            ranked.append((0, -overlap, track["id"], track))
            continue
        if not allow_gap or track["last_frame"] == frame:
            # 同帧内已经有一只游程续上了这条轨道，再靠频隙衔接会把
            # 相邻信道（跳变帧）焊在一起，因此只允许严格重叠。
            continue
        distance = max(low - last, first - high)
        if distance <= max_gap_bins:
            ranked.append((1, distance, track["id"], track))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


def _overlaps_any(tracks, frame, first, last, max_gap_frames):
    """Whether a *live* track's band already covers this run.

    A frame can show several runs inside one track's band (a deep spectral
    null of a single wideband emission, an AM carrier with its sidebands).
    The track is claimed by the first of them, so the remaining runs must not
    spawn bogus one-frame tracks.

    Only tracks that are still trackable (last seen no longer than
    ``max_gap_frames`` ago) may veto a run. A stale track is history: an
    emitter that revisits a channel later shows a run in a band that a dead
    track happens to cover, and vetoing it would delete a legitimate hop —
    which is what happens to every revisit in a record where a second
    concurrent emitter forces ``allow_gap=False`` on every frame.
    """
    for track in tracks:
        if frame - track["last_frame"] - 1 > max_gap_frames:
            continue
        low, high = _track_band(track)
        if min(last, high) - max(first, low) + 1 > 0:
            return True
    return False


def _link_tracks(rows, frequencies, resolved, transition_frames):
    """Link per-frame runs into hop tracks (time-frequency ridge tracking).

    A frame that shows a single run is a *geometry* observation: the run may
    extend the track's band, and it may be linked across a small frequency
    gap (a briefly faded bin inside one hop). A frame that shows several
    runs is a transition frame — either a hop boundary straddling the
    analysis window or several emitters working at once. There, a run may
    only continue a track it spectrally overlaps (a gap link is refused
    because the neighbouring hop channel would be welded on), every run
    already inside a claimed track's band is part of that track and is
    dropped, and every *other* run starts a new track — that is what keeps a
    hop boundary from inflating the single-hop bandwidth, while still
    allowing two simultaneous emitters to be tracked side by side. Finally,
    a run much wider than the track's typical width is a fused two-hop
    artefact and is never accepted as geometry.
    """
    ratio = resolved["transition_ratio"]
    max_gap_frames = resolved["max_gap_frames"]
    max_gap_bins = resolved["max_gap_bins"]
    tracks = []
    counter = 0
    for frame, runs in enumerate(rows):
        if not runs:
            continue
        if len(runs) > 1:
            transition_frames[0] += 1
        claimed = set()
        for first, last in runs:
            track = _best_track(tracks, frame, first, last, max_gap_frames,
                                max_gap_bins, claimed, allow_gap=len(runs) == 1)
            if track is None and len(runs) > 1 and _overlaps_any(
                    tracks, frame, first, last, max_gap_frames):
                continue
            if track is None:
                counter += 1
                tracks.append({"id": counter, "first_frame": frame, "last_frame": frame,
                               "frames": [frame], "runs": [(first, last)],
                               "transition_frames": 0})
                claimed.add(counter)
                continue
            claimed.add(track["id"])
            width = last - first + 1
            widths = sorted(stop - start + 1 for start, stop in track["runs"])
            median = widths[len(widths) // 2]
            track["last_frame"] = frame
            track["frames"].append(frame)
            if len(runs) > 1 or (len(track["runs"]) >= 2 and width > ratio * median):
                track["transition_frames"] += 1
                continue
            track["runs"].append((first, last))
    return tracks


def _finalise_hops(tracks, psd, frequencies, starts, nfft, rate, duration,
                   noise_linear, resolved):
    """Turn tracks into hop measurements (band, dwell, power, in-band SNR).

    The frequency band comes from the geometry runs only. The dwell time is
    then re-measured on the **raw** frame power inside that band, which
    removes the smear introduced by the temporal smoothing: the first and
    last frame that genuinely carry the hop are found by threshold crossing,
    exactly like the session-level time support in :func:`detect_signals`.
    Because a channel can be visited twice, the active run overlapping the
    track most is taken — the two visits become two separate hops.
    """
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]
    min_dwell_frames = resolved["min_dwell_frames"]
    hops = []
    for track in tracks:
        first_bin, last_bin = _track_band(track)
        bandwidth = (last_bin - first_bin + 1) * resolution
        f_low = float(frequencies[first_bin] - resolution / 2.0)
        f_high = float(frequencies[last_bin] + resolution / 2.0)
        band_power = np.asarray(psd[:, first_bin:last_bin + 1].sum(axis=1),
                                dtype=np.float64) * resolution
        active = band_power > noise_linear * bandwidth * 10.0 ** (threshold_db / 10.0)
        if not active.any():
            continue
        best, best_overlap = None, 0
        for start, stop in _true_runs(active):
            overlap = min(stop, track["last_frame"]) - max(start, track["first_frame"]) + 1
            if overlap > best_overlap:
                best, best_overlap = (start, stop), overlap
        if best is None:
            best = (track["first_frame"], track["last_frame"])
        start_frame, stop_frame = best
        if stop_frame - start_frame + 1 < min_dwell_frames:
            continue
        t_start = float(starts[start_frame]) / rate
        t_end = float(min((starts[stop_frame] + nfft) / rate, duration))
        dwell = t_end - t_start
        if dwell < resolved["min_dwell_s"]:
            continue
        segment = band_power[start_frame:stop_frame + 1]
        power_linear = float(segment.mean())
        band_frequencies = frequencies[first_bin:last_bin + 1]
        profile = psd[start_frame:stop_frame + 1, first_bin:last_bin + 1].mean(axis=0)
        weight_sum = float(profile.sum())
        centroid = (float((profile * band_frequencies).sum() / weight_sum)
                    if weight_sum > 0 else float(band_frequencies.mean()))
        # 单跳带宽取「扣噪声后的 99% 功率占用带宽」：检测掩膜会被 OFDM/2FSK 的
        # 频谱旁瓣撑宽数个频点（实测约 1.3～2.4 个频点/侧），只有功率占用带宽才代表
        # 信号真正占用的频带。
        corrected = np.maximum(profile.astype(np.float64) - noise_linear, 0.0)
        occupancy_low, occupancy_high = _occupied_span(corrected, 0, corrected.size - 1)
        occupied_low = float(frequencies[first_bin + occupancy_low] - resolution / 2.0)
        occupied_high = float(frequencies[first_bin + occupancy_high] + resolution / 2.0)
        occupied_bandwidth = occupied_high - occupied_low
        # inband_snr_v1：信号功率按检测掩膜频带内的实测功率扣掉该频带噪声，
        # 噪声参考功率按信号自身占用带宽计算（N0·B_hop）。
        signal_power = max(power_linear - noise_linear * bandwidth, 1e-30)
        noise_band = max(noise_linear * occupied_bandwidth, 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        hops.append({
            "center_hz": 0.5 * (occupied_low + occupied_high),
            "centroid_hz": centroid,
            "bandwidth_hz": occupied_bandwidth,
            "f_low_hz": occupied_low,
            "f_high_hz": occupied_high,
            "mask_bandwidth_hz": bandwidth,
            "mask_f_low_hz": f_low,
            "mask_f_high_hz": f_high,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "dwell_s": dwell,
            "power_linear": power_linear,
            "snr_db": snr_db,
            "confidence": float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)),
            "frame_count": int(stop_frame - start_frame + 1),
            "bin_count": int(sum(stop - start + 1 for start, stop in track["runs"])),
            "band_nfft": resolved["nfft"],
        })
    return hops


#: 频带精测的分析点数上限。跟踪需要时间分辨率、带宽需要频率分辨率，一个网格
#: 满足不了两者，所以跟踪用粗网格、频带测量再在细网格上复算一遍。
_HOP_REFINE_NFFT = 2048


def _refine_hop_bands(hops, x, rate, noise_linear, resolved, threshold_db, coarse_nfft):
    """Re-measure each hop's band on a finer grid, in place.

    Only the spectral parameters are recomputed: the hop *times* keep the
    coarse grid (the finer grid would smear the dwell). The refined search
    band is the coarse mask band plus a small guard, the noise reference
    stays the coarse noise floor, and the pass is skipped when the finer
    grid would leave fewer than :data:`_HOP_MIN_DWELL_FRAMES` frames in the
    shortest hop — then nothing would be gained.
    """
    if not hops:
        return
    shortest = min(item["dwell_s"] for item in hops) * rate
    limit = int(2.0 ** np.floor(np.log2(max(shortest / _HOP_MIN_DWELL_FRAMES, 16.0))))
    refine_nfft = int(min(_HOP_REFINE_NFFT, limit))
    if refine_nfft <= coarse_nfft:
        return
    psd, frequencies, starts, _ = _stft_psd(x, rate, refine_nfft)
    resolution = rate / refine_nfft
    centers = (starts + (refine_nfft - 1) / 2.0) / rate
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    guard = 2.0 * resolved["freq_resolution_hz"]
    for item in hops:
        low_bin = int(np.searchsorted(frequencies, item["mask_f_low_hz"] - guard, side="left"))
        high_bin = int(np.searchsorted(frequencies, item["mask_f_high_hz"] + guard, side="right"))
        low_bin = max(0, min(low_bin, frequencies.size - 2))
        high_bin = min(frequencies.size - 1, max(high_bin, low_bin + 1))
        inside = np.flatnonzero((centers >= item["t_start_s"]) & (centers <= item["t_end_s"]))
        if inside.size == 0:
            continue
        profile = psd[inside, low_bin:high_bin + 1].mean(axis=0).astype(np.float64)
        band = frequencies[low_bin:high_bin + 1]
        mask = profile > noise_linear * threshold_linear
        if not mask.any():
            continue
        occupied_low, occupied_high = _occupied_span(
            np.maximum(profile - noise_linear, 0.0), 0, profile.size - 1)
        f_low = float(band[occupied_low]) - resolution / 2.0
        f_high = float(band[occupied_high]) + resolution / 2.0
        mask_bins = np.flatnonzero(mask)
        mask_low = float(band[int(mask_bins[0])]) - resolution / 2.0
        mask_high = float(band[int(mask_bins[-1])]) + resolution / 2.0
        power_linear = float(profile[mask].sum() * resolution)
        weight_sum = float(profile.sum())
        centroid = (float((profile * band).sum() / weight_sum)
                    if weight_sum > 0 else float(band.mean()))
        signal_power = max(power_linear - noise_linear * (mask_high - mask_low), 1e-30)
        noise_band = max(noise_linear * (f_high - f_low), 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        item.update({
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "bandwidth_hz": f_high - f_low,
            "center_hz": 0.5 * (f_low + f_high),
            "centroid_hz": centroid,
            "mask_f_low_hz": mask_low,
            "mask_f_high_hz": mask_high,
            "mask_bandwidth_hz": mask_high - mask_low,
            "power_linear": power_linear,
            "snr_db": snr_db,
            "confidence": float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)),
            "band_nfft": refine_nfft,
        })


#: 同一会话内频点间隔的最大/最小比：超过它说明这个时间组里混进了另一部
#: 发射机（两部发射机的频带之间会留下一道明显更宽的缝）。
_HOP_SESSION_SPLIT_RATIO = 3.0
_HOP_SESSION_OVERLAP_TOL_FRAMES = 1.5


def _cluster_centres(values, resolution):
    """Deduplicate hop centres into visited channels (1-resolution gap).

    The centre of a channel measured on two different visits differs by less
    than one frequency point, so values closer than one bin are the same
    channel and are averaged.
    """
    clusters = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > resolution:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.mean(group)) for group in clusters]


def _split_session(members, resolution):
    """Split a time group where the channel gaps prove two emitters.

    Hops that follow each other in time form a session, but two emitters
    working at the same time also interleave that way. Their frequency plans
    are the only remaining evidence: one emitter visits channels spread over
    *its* hopping span, so an absolute gap outlier (``>``
    :data:`_HOP_SESSION_SPLIT_RATIO` times the **median** gap of the same
    session) means the group holds two emitters and is split there. The
    median is used rather than the narrowest gap so that an irregular but
    single-emitter plan (one channel pair closer than the rest) is kept
    together; fewer than three distinct channels can never show such an
    outlier and are always kept together.
    """
    if len(members) < 2:
        return [members]
    channels = _cluster_centres((item["center_hz"] for item in members), resolution)
    if len(channels) < 3:
        return [members]
    gaps = [(channels[index + 1] - channels[index], index)
            for index in range(len(channels) - 1)]
    widest, index = max(gaps)
    values = sorted(gap for gap, _ in gaps)
    median = values[len(values) // 2]
    if widest <= _HOP_SESSION_SPLIT_RATIO * max(median, resolution):
        return [members]
    limit = 0.5 * (channels[index] + channels[index + 1])
    lower = [item for item in members if item["center_hz"] <= limit]
    upper = [item for item in members if item["center_hz"] > limit]
    return _split_session(lower, resolution) + _split_session(upper, resolution)


def _group_hop_sessions(hops, noise_linear, duration, resolution, frame_interval_s):
    """Group hop measurements into emitter sessions by time contiguity.

    ``_same_session`` (the session-level heuristic used by
    :func:`detect_signals`) cannot be reused here: it compares a new hop
    against the *whole* accumulated session band, so a wide hopping span
    gets split arbitrarily — that is exactly how a four-channel session ends
    up reported with ``sub_bands`` 2. For hop tracks the reliable
    discriminator is the channel plan: every hop of one emitter falls inside
    that emitter's hopping span, so the *nearest* open session wins, with
    time used as an availability gate (independent emitters either overlap in
    time by much more than a transition frame, or are separated by a pause
    far longer than one dwell). Emitters that still interleave are separated
    afterwards by :func:`_split_session`, which uses their channel plans as
    evidence.

    Each session also exposes the session-level aggregate in-band SNR
    (signal power averaged over the record divided by ``N0·B_session``),
    which is directly comparable with ``signal_truth``'s ``snr_inband_db``.
    """
    overlap_tol = _HOP_SESSION_OVERLAP_TOL_FRAMES * frame_interval_s
    ordered = sorted(hops, key=lambda item: (item["t_start_s"], item["center_hz"]))
    groups = []
    for hop in ordered:
        chosen, chosen_score = None, None
        for index, session in enumerate(groups):
            pause = hop["t_start_s"] - session["t_end_s"]
            if pause < -overlap_tol:
                # 与已在进行的会话明显重叠：只能是另一部发射机（不足
                # 两帧的重叠是跳变帧本身的展宽，不算证据）
                continue
            if pause > _SESSION_GAP_RATIO * session["dwell_median_s"]:
                continue
            centres = [item["center_hz"] for item in session["hops"]]
            distance = max(min(centres) - hop["center_hz"],
                           hop["center_hz"] - max(centres), 0.0)
            # 先看频距再看时间：同一部发射机的跳频点落在一段固定的跨度里，
            # 因此「最近的会话」是稳定的判据；两部交错的发射机也由此分开，
            # 而时间只作为可用性门限（跳变帧会带来不足两帧的重叠）。
            score = (distance, pause)
            if chosen_score is None or score < chosen_score:
                chosen, chosen_score = index, score
        if chosen is None:
            groups.append({"t_end_s": hop["t_end_s"],
                           "dwell_median_s": hop["dwell_s"],
                           "hops": [hop]})
            continue
        session = groups[chosen]
        session["hops"].append(hop)
        session["t_end_s"] = max(session["t_end_s"], hop["t_end_s"])
        dwells = sorted(item["dwell_s"] for item in session["hops"])
        session["dwell_median_s"] = dwells[len(dwells) // 2]
    sessions = [{"hops": members}
                for group in groups
                for members in _split_session(group["hops"], resolution)]
    sessions.sort(key=lambda session: session["hops"][0]["t_start_s"])
    for identifier, session in enumerate(sessions, start=1):
        members = session["hops"]
        for item in members:
            item["session_id"] = identifier
        dwells = sorted(item["dwell_s"] for item in members)
        widths = sorted(item["bandwidth_hz"] for item in members)
        # 跳频点聚合：同一信道被多次访问时测得的中心会有不到一个频点的抖动，
        # 先按一个频点宽度聚类再去重，得到「访问过的跳频点集合」。
        channels = _cluster_centres((item["center_hz"] for item in members), resolution)
        spacings = [channels[index + 1] - channels[index]
                    for index in range(len(channels) - 1)]
        # 跳速取「跳起点间隔中位数」的倒数：它是发射机的跳频周期，与驻留时间
        # （信号真正存在的时间）不同——本工程 OFDM 图传样式每跳尾部有空闲，两者
        # 相差正好是一个占空比。
        starts_sorted = sorted(item["t_start_s"] for item in members)
        periods = [starts_sorted[index + 1] - starts_sorted[index]
                   for index in range(len(starts_sorted) - 1)]
        hop_period = sorted(periods)[len(periods) // 2] if periods else None
        dwell_median = dwells[len(dwells) // 2]
        f_low = min(item["f_low_hz"] for item in members)
        f_high = max(item["f_high_hz"] for item in members)
        # 会话级功率：按驻留时长加权摊到整段记录，与生成器的 power_dbfs_actual 同口径
        power_linear = sum(item["power_linear"] * item["dwell_s"]
                           for item in members) / max(duration, 1e-30)
        noise_band = max(noise_linear * (f_high - f_low), 1e-30)
        signal_power = max(power_linear - noise_band, 1e-30)
        session.update({
            "session_id": identifier,
            "hop_count": len(members),
            "sequence": [item["center_hz"] for item in members],
            "hop_frequencies_hz": channels,
            "channel_spacing_hz": (min(spacings) if spacings else None),
            "hop_span_hz": (channels[-1] - channels[0]) if len(channels) > 1 else 0.0,
            "hop_bandwidth_hz": widths[len(widths) // 2],
            "hop_period_s": hop_period,
            "hop_rate_hz": (1.0 / hop_period) if hop_period else None,
            "duty_cycle": ((dwell_median / hop_period) if hop_period else None),
            "dwell_median_s": dwell_median,
            "dwell_min_s": dwells[0],
            "dwell_max_s": dwells[-1],
            "center_hz": 0.5 * (f_low + f_high),
            "bandwidth_hz": f_high - f_low,
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "t_start_s": members[0]["t_start_s"],
            "t_end_s": max(item["t_end_s"] for item in members),
            "power_linear": power_linear,
            "snr_db": float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB)),
            "session_detection_id": None,
        })
    return sessions


def detect_hops(samples, sample_rate, config=None, with_sessions=True):
    """Per-hop frequency-hopping parameter estimation (``hop_track_v1``).

    Pipeline (all NumPy, Cython-compilable), deliberately *not* a variant of
    :func:`detect_signals`:

    1. The same Hanning STFT as :func:`analyze`/:func:`detect_signals`, but
       the per-frame PSD is kept instead of being averaged over time —
       averaging is exactly what fuses neighbouring hop channels on a wide
       hopping span.
    2. Noise floor from the time-averaged PSD (identical convention to the
       session detector, so the two results are directly comparable).
    3. Per-frame spectral mask: threshold, frequency-domain closing, drop
       runs narrower than ``min_bandwidth_hz``.
    4. Ridge tracking: runs are linked from frame to frame by spectral
       overlap (with a small frequency-gap tolerance for single-run frames).
       A frame carrying several runs is a transition frame — a hop boundary
       or a second concurrent emitter — where every run other than the one
       overlapping a known track starts a new track. Runs much wider than
       the track's typical width are fused two-hop artefacts and are also
       refused as geometry.
    5. Per hop: band from the accepted runs, then re-measured on a finer
       second STFT grid (the coarse grid alone inflates a single-hop band by
       two to three bins of Hann leakage); dwell re-measured on the raw frame
       power inside that band, power as the mean in-band power over the
       dwell, in-band SNR ``10·log10(P_band/(N0·B_hop))``
       (``inband_snr_v1``, the per-hop bandwidth being the signal's own
       occupied bandwidth).
    6. Hops are grouped into emitter sessions by time contiguity; each
       session reports the hop count, the visited channel set, the hop rate
       (``1/median`` start-to-start interval, see ``duty_cycle``) and the
       session-level aggregate SNR.

    Everything is expressed as a baseband offset; no RF parameter is
    invented. The frozen ``detect_result_v1`` contract is untouched: this
    function has its own contract ``fh_hops_v1``. ``resolvable`` plus
    ``reason`` state honestly when the record is too short or the hopping is
    too fast for the STFT grid (a hop needs at least
    :data:`_HOP_MIN_DWELL_FRAMES` frames), instead of silently reporting
    fused hops. Two consecutive hops that reuse the same channel are
    physically indistinguishable and are reported as one longer dwell.

    ``with_sessions`` additionally runs :func:`detect_signals` on the same
    record (read-only reuse, no behaviour change) and attaches its summary
    as ``baseline``, cross-linking every session to the matching detection
    through ``session_detection_id``.

    Returns ``(summary, arrays)``.
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    duration = float(x.size / rate)
    resolved = _hop_config(rate, duration, config)
    nfft = resolved["nfft"]
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]

    psd, frequencies, starts, hop = _stft_psd(x, rate, nfft)
    psd_average = psd.mean(axis=0)
    psd_average_db = 10.0 * np.log10(np.maximum(psd_average, 1e-30))
    noise_floor_db = _noise_floor_db(psd_average_db, threshold_db)
    threshold_dbfs = noise_floor_db + threshold_db
    noise_linear = 10.0 ** (noise_floor_db / 10.0)

    smoothed = _smooth_psd(psd, resolved["smooth_frames"])
    smoothed_db = 10.0 * np.log10(np.maximum(smoothed, 1e-30))
    rows = _frame_runs(smoothed_db, threshold_dbfs, resolved["merge_bins"],
                       resolved["min_bandwidth_bins"])
    transition_frames = [0]
    tracks = _link_tracks(rows, frequencies, resolved, transition_frames)
    hops = _finalise_hops(tracks, psd, frequencies, starts, nfft, rate, duration,
                          noise_linear, resolved)
    _refine_hop_bands(hops, x, rate, noise_linear, resolved, threshold_db, nfft)
    if len(hops) > resolved["max_hops"]:
        hops.sort(key=lambda item: -item["power_linear"])
        hops = hops[:resolved["max_hops"]]
    hops.sort(key=lambda item: (item["t_start_s"], item["center_hz"]))
    for index, item in enumerate(hops, start=1):
        item["id"] = index
        item["session_id"] = None
    sessions = _group_hop_sessions(hops, noise_linear, duration, resolution,
                                   hop / rate)

    dwell_limit_s = resolved["min_dwell_frames"] * hop / rate
    resolvable = bool(hops) and min(item["dwell_s"] for item in hops) >= dwell_limit_s
    if not hops:
        reason = (f"没有任何游程同时满足最小带宽 {resolved['min_bandwidth_hz']:.0f} Hz 与"
                  f"最短驻留 {resolved['min_dwell_frames']} 帧；请降低门限或增大分析点数")
    elif not resolvable:
        reason = (f"STFT 帧间距 {hop / rate * 1000.0:.3f} ms、一跳至少"
                  f"{resolved['min_dwell_frames']} 帧，跳速高于"
                  f"{rate / hop / resolved['min_dwell_frames']:.1f} Hz 时驻留不足、逐跳不可分辨")
    else:
        reason = None

    hop_entries = []
    for item in hops:
        hop_entries.append({
            "id": item["id"],
            "session_id": item["session_id"],
            "center_hz": round(item["center_hz"], 3),
            "centroid_hz": round(item["centroid_hz"], 3),
            "bandwidth_hz": round(item["bandwidth_hz"], 3),
            "f_low_hz": round(item["f_low_hz"], 3),
            "f_high_hz": round(item["f_high_hz"], 3),
            "t_start_s": round(item["t_start_s"], 6),
            "t_end_s": round(item["t_end_s"], 6),
            "dwell_s": round(item["dwell_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(item["power_linear"], 1e-30))), 3),
            "snr_db": round(item["snr_db"], 3),
            "confidence": round(item["confidence"], 3),
            "frame_count": int(item["frame_count"]),
            "bin_count": int(item["bin_count"]),
            "band_nfft": int(item["band_nfft"]),
            "mask_f_low_hz": round(item["mask_f_low_hz"], 3),
            "mask_f_high_hz": round(item["mask_f_high_hz"], 3),
            "mask_bandwidth_hz": round(item["mask_bandwidth_hz"], 3),
        })

    session_entries = []
    for session in sessions:
        session_entries.append({
            "session_id": session["session_id"],
            "hop_count": int(session["hop_count"]),
            "sequence": [round(value, 3) for value in session["sequence"]],
            "hop_frequencies_hz": [round(value, 3) for value in session["hop_frequencies_hz"]],
            "channel_spacing_hz": (None if session["channel_spacing_hz"] is None
                                   else round(session["channel_spacing_hz"], 3)),
            "hop_span_hz": round(session["hop_span_hz"], 3),
            "hop_bandwidth_hz": round(session["hop_bandwidth_hz"], 3),
            "hop_period_s": (None if session["hop_period_s"] is None
                             else round(session["hop_period_s"], 6)),
            "hop_rate_hz": (None if session["hop_rate_hz"] is None
                            else round(session["hop_rate_hz"], 3)),
            "duty_cycle": (None if session["duty_cycle"] is None
                           else round(session["duty_cycle"], 4)),
            "dwell_median_s": round(session["dwell_median_s"], 6),
            "dwell_min_s": round(session["dwell_min_s"], 6),
            "dwell_max_s": round(session["dwell_max_s"], 6),
            "center_hz": round(session["center_hz"], 3),
            "bandwidth_hz": round(session["bandwidth_hz"], 3),
            "f_low_hz": round(session["f_low_hz"], 3),
            "f_high_hz": round(session["f_high_hz"], 3),
            "t_start_s": round(session["t_start_s"], 6),
            "t_end_s": round(session["t_end_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(session["power_linear"], 1e-30))), 3),
            "snr_db": round(session["snr_db"], 3),
            "session_detection_id": session["session_detection_id"],
        })

    summary = {
        "contract": HOP_CONTRACT,
        "algorithm": HOP_ALGORITHM,
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
            "smooth_frames": resolved["smooth_frames"],
            "min_bandwidth_hz": round(resolved["min_bandwidth_hz"], 6),
            "min_dwell_s": round(resolved["min_dwell_s"], 9),
            "merge_bins": resolved["merge_bins"],
            "max_gap_frames": resolved["max_gap_frames"],
            "max_gap_bins": resolved["max_gap_bins"],
            "transition_ratio": resolved["transition_ratio"],
            "max_hops": resolved["max_hops"],
        },
        "freq_resolution_hz": resolution,
        "frame_interval_s": hop / rate,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": round(threshold_dbfs, 3),
        "dwell_limit_s": dwell_limit_s,
        "hop_rate_limit_hz": 1.0 / dwell_limit_s,
        "resolvable": resolvable,
        "reason": reason,
        "transition_frames": int(transition_frames[0]),
        "hops": hop_entries,
        "sessions": session_entries,
    }
    if with_sessions:
        baseline, _ = detect_signals(x, rate, {
            "nfft": nfft,
            "threshold_db": threshold_db,
            "min_bandwidth_hz": resolved["min_bandwidth_hz"],
            "merge_bins": resolved["merge_bins"],
        })
        summary["baseline"] = {
            "contract": baseline["contract"],
            "algorithm": baseline["algorithm"],
            "threshold_dbfs_per_hz": baseline["threshold_dbfs_per_hz"],
            "detections": baseline["detections"],
        }
        # 会话与会话级检测互链：取频带重叠最多的一条（跳频会话必然是 hopping 实例）
        for session in summary["sessions"]:
            best, best_overlap = None, 0.0
            for detection in baseline["detections"]:
                overlap = min(session["f_high_hz"], detection["f_high_hz"]) \
                    - max(session["f_low_hz"], detection["f_low_hz"])
                if overlap > best_overlap:
                    best, best_overlap = detection["id"], overlap
            session["session_detection_id"] = best

    hop_boxes = [[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                 for item in hop_entries]
    arrays = {
        "frequency": frequencies,
        "frame_time": (starts + (nfft - 1) / 2.0) / rate,
        "spectrogram_db": (smoothed_db).astype(np.float32),
        "spectrogram_raw_db": (10.0 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        "spectrum_db": psd_average_db,
        "spectrum_median_db": 10.0 * np.log10(np.maximum(np.median(psd, axis=0), 1e-30)),
        "threshold_db": np.array([threshold_dbfs], dtype=np.float64),
        "noise_floor_db": np.array([noise_floor_db], dtype=np.float64),
        "hop_boxes": np.array(hop_boxes, dtype=np.float64).reshape(-1, 4),
        "hop_id": np.array([item["id"] for item in hop_entries], dtype=np.int64),
        "hop_session_id": np.array([item["session_id"] for item in hop_entries], dtype=np.int64),
        "hop_snr_db": np.array([item["snr_db"] for item in hop_entries], dtype=np.float64),
        "hop_power_dbfs": np.array([item["power_dbfs"] for item in hop_entries], dtype=np.float64),
    }
    return summary, arrays
