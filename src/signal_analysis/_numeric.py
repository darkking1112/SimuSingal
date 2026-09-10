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
    two-sided, default full band), ``snr_db`` (relative to the strongest
    signal) or ``power_dbfs`` (absolute, used when there are no signals).
    A fixed ``seed`` reproduces identical samples.
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
    if noise_enabled:
        noise_bandwidth = _finite(noise.get("bandwidth", rate), "噪声带宽", 0.0, rate)
        if not 0 < noise_bandwidth <= rate:
            raise ValueError("噪声带宽必须大于 0 且不超过采样率")
        if summaries:
            strongest = max(10 ** (entry["power_dbfs"] / 10.0) for entry in summaries)
            snr_db = _finite(noise.get("snr_db", 20.0), "信噪比", -60.0, 120.0)
            noise_power = strongest / 10 ** (snr_db / 10.0)
        else:
            noise_power = 10 ** (_finite(noise.get("power_dbfs", -20.0), "噪声功率", -200.0, 0.0) / 10.0)
        if noise_power > 0:
            record += _band_noise(rng, count, rate, noise_bandwidth / 2.0) * np.sqrt(noise_power)
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
                  "snr_db": float(noise.get("snr_db")) if noise_enabled and summaries else None},
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
    padded = np.pad(x, (0, max(0, nfft - x.size)))
    hop = max(nfft // 2, int(np.ceil(max(0, padded.size - nfft) / 511)))
    starts = np.arange(0, padded.size - nfft + 1, hop)
    window = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[starts]
    transformed = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
    psd = np.abs(transformed) ** 2 / (rate * np.sum(window ** 2))
    indices = np.arange(0, x.size, max(1, int(np.ceil(x.size / 4096))))
    const_step = max(1, int(np.ceil(x.size / 20000)))
    const_view = x[::const_step]
    arrays = {
        "wave_time": indices / rate,
        "wave_i": x.real[indices], "wave_q": x.imag[indices],
        "frequency": np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate)),
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
