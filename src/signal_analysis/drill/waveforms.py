"""Deterministic real-IF waveform generation for the drill question bank.

Generation is intentionally independent of the asset/database layer.  Every
generator returns a real-valued 5 s IF record plus a noisy complex baseband copy
used by the constellation and IQ-phase auxiliary views.  The caller controls
the random seed, so a question can be reproduced from its ``SignalSpec``.
"""

from dataclasses import dataclass

import numpy as np

from .catalog import CLASS_BY_KEY, SignalSpec
from .patterns import apply_gate, apply_hop


@dataclass
class Waveform:
    samples: np.ndarray       # real IF, float32
    baseband: np.ndarray      # noisy complex baseband, complex64
    meta: dict


def _rrc_taps(alpha, sps, span_symbols=8):
    """Unit-energy root-raised-cosine taps (same convention as the app)."""
    n = int(span_symbols * sps)
    t = np.arange(-n, n + 1) / sps
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_t = np.pi * t
        taps = (np.sin(pi_t * (1 - alpha))
                + 4 * alpha * t * np.cos(pi_t * (1 + alpha)))
        taps = taps / (pi_t * (1 - (4 * alpha * t) ** 2))
    taps[n] = 1 - alpha + 4 * alpha / np.pi
    edge = np.abs(np.abs(t) - 1 / (4 * alpha)) < 1e-12
    if np.any(edge):
        value = (alpha / np.sqrt(2.0)) * (
            (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
        taps[edge] = value
    return taps / np.sqrt(np.sum(taps ** 2))


def _shape_pulses(symbols, taps, sps, count):
    """Upsample, pulse-shape and trim to ``count`` samples."""
    stream = np.zeros(len(symbols) * sps, dtype=np.complex128)
    stream[::sps] = symbols
    if len(taps) <= 256:
        shaped = np.convolve(stream, taps)
    else:
        full = len(stream) + len(taps) - 1
        fft_n = 1 << (full - 1).bit_length()
        shaped = np.fft.ifft(np.fft.fft(stream, fft_n)
                             * np.fft.fft(taps, fft_n))[:full]
    delay = len(taps) // 2
    return shaped[delay:delay + count]


def _hilbert(values):
    """Analytic signal of a real vector via FFT."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    spectrum = np.fft.fft(values)
    multiplier = np.zeros(n, dtype=np.float64)
    multiplier[0] = 1.0
    if n % 2 == 0:
        multiplier[n // 2] = 1.0
        multiplier[1:n // 2] = 2.0
    else:
        multiplier[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * multiplier)


def _tone_message(t, frequencies, rng):
    """A sum of 1–3 random-phase tones, peak-normalized to 1."""
    frequencies = np.asarray(tuple(frequencies), dtype=np.float64)
    amps = rng.uniform(0.6, 1.0, size=frequencies.size)
    phases = rng.uniform(0, 2 * np.pi, size=frequencies.size)
    message = np.zeros_like(t)
    for amplitude, frequency, phase in zip(amps, frequencies, phases):
        message += amplitude * np.cos(2 * np.pi * frequency * t + phase)
    peak = float(np.max(np.abs(message)))
    if not peak > 0:
        raise ValueError("message generation failed")
    return message / peak


def _constellation(key):
    if key == "ask2":
        return np.asarray([0.0, 1.0], dtype=np.complex128)
    if key == "bpsk":
        return np.asarray([-1.0, 1.0], dtype=np.complex128)
    if key == "qpsk":
        return np.asarray([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j],
                          dtype=np.complex128) / np.sqrt(2.0)
    if key == "psk8":
        return np.exp(2j * np.pi * np.arange(8) / 8.0).astype(np.complex128)
    if key == "qam16":
        points = np.asarray([(2 * i - 3) + 1j * (2 * j - 3)
                             for i in range(4) for j in range(4)],
                            dtype=np.complex128)
        return points / np.sqrt(10.0)
    if key == "qam64":
        points = np.asarray([(2 * i - 7) + 1j * (2 * j - 7)
                             for i in range(8) for j in range(8)],
                            dtype=np.complex128)
        return points / np.sqrt(42.0)
    raise ValueError(f"no constellation for {key}")


def _linear_baseband(rng, fs_hz, count, key, symbol_rate, alpha):
    constellation = _constellation(key)
    sps = max(8, int(round(fs_hz / float(symbol_rate))))
    symbol_rate = fs_hz / sps
    span = 8
    n_symbols = int(np.ceil(count / sps)) + 2 * span
    symbols = constellation[rng.integers(0, len(constellation), size=n_symbols)]
    taps = _rrc_taps(alpha, sps, span)
    baseband = _shape_pulses(symbols, taps, sps, count)
    return baseband, {
        "symbol_rate": float(symbol_rate),
        "sps": int(sps),
        "taps": taps,
        "symbols": symbols.astype(np.complex128),
        "constellation": constellation,
    }


def _cpfsk_baseband(rng, fs_hz, count, key, symbol_rate, spacing):
    if key == "fsk2":
        order = 2
    elif key == "fsk4":
        order = 4
    elif key == "fsk8":
        order = 8
    else:
        raise ValueError(f"no CPFSK order for {key}")
    sps = max(2, int(round(fs_hz / float(symbol_rate))))
    symbol_rate = fs_hz / sps
    n_symbols = int(np.ceil(count / sps)) + 1
    symbols = rng.integers(0, order, size=n_symbols)
    offsets = (symbols - (order - 1) / 2.0) * float(spacing)
    frequencies = np.repeat(offsets, sps)[:count]
    phase = 2 * np.pi * np.cumsum(frequencies) / float(fs_hz)
    baseband = np.exp(1j * phase)
    return baseband, {
        "symbol_rate": float(symbol_rate),
        "sps": int(sps),
        "fsk_order": int(order),
        "tone_spacing_hz": float(spacing),
    }


def _ook_at_bit_rate(t, fs_hz, bit_rate, rng):
    """A rectangular on/off bit envelope at ``bit_rate`` (single-tone style)."""
    n_bits = int(np.ceil(t[-1] * float(bit_rate))) + 2
    bits = (rng.random(n_bits) < 0.5).astype(np.float64)
    if bits.min() == bits.max():
        bits[0] = 1.0 - bits[0]
    indices = np.minimum((t * float(bit_rate)).astype(np.int64), n_bits - 1)
    return bits[indices]


def _audio_channel(t, rng, bandwidth_hz):
    """A small audio band made from two or three tones."""
    count = 3 if rng.random() < 0.5 else 2
    tones = np.linspace(0.18, 0.95, count) * float(bandwidth_hz)
    phases = rng.uniform(0, 2 * np.pi, size=count)
    amps = rng.uniform(0.5, 1.0, size=count)
    signal = np.zeros_like(t)
    for amplitude, frequency, phase in zip(amps, tones, phases):
        signal += amplitude * np.cos(2 * np.pi * frequency * t + phase)
    return signal / max(1e-12, np.max(np.abs(signal)))


def _add_noise_and_if(baseband, fs_hz, f_if_hz, snr_db, rng):
    """Add complex baseband noise, convert to a real IF record and normalize."""
    baseband = np.asarray(baseband, dtype=np.complex128)
    power = float(np.mean(np.abs(baseband) ** 2))
    if power <= 0:
        raise ValueError("baseband signal has no power")
    if np.isfinite(snr_db):
        noise_variance = power / (10.0 ** (float(snr_db) / 10.0))
        noise = np.sqrt(noise_variance / 2.0) * (
            rng.standard_normal(baseband.size) + 1j * rng.standard_normal(baseband.size))
        received = baseband + noise
    else:
        noise_variance = 0.0
        received = baseband
    t = np.arange(baseband.size, dtype=np.float64) / float(fs_hz)
    carrier = np.exp(2j * np.pi * float(f_if_hz) * t)
    samples = np.real(received * carrier)
    peak = float(np.max(np.abs(samples)))
    if not peak > 0:
        raise ValueError("IF signal has no power")
    scale = 0.9 / peak
    return (samples * scale).astype(np.float32), (received * scale).astype(np.complex64), noise_variance * scale ** 2


def generate_waveform(spec: SignalSpec, rng=None) -> Waveform:
    """Generate one deterministic round from ``spec``."""
    info = CLASS_BY_KEY[spec.key]
    rng = rng if rng is not None else np.random.default_rng(spec.seed)
    count = int(round(float(spec.fs_hz) * float(spec.duration_s)))
    if count < 64:
        raise ValueError("duration is too short for a drill signal")
    t = np.arange(count, dtype=np.float64) / float(spec.fs_hz)
    params = dict(spec.params or {})
    key = spec.key
    meta = {"key": key, "class_label": info.label}
    baseband = None
    gate = spec.gate

    if key == "am":
        message = _tone_message(t, params.get("message_freqs_hz", (1000.0,)),
                                rng)
        depth = float(params.get("depth", 0.7))
        baseband = 1.0 + depth * message
        meta.update(depth=depth)
    elif key == "dsb":
        baseband = _tone_message(t, params.get("message_freqs_hz", (1000.0,)),
                                 rng)
    elif key == "ssb":
        message = _tone_message(t, params.get("message_freqs_hz", (1000.0, 1700.0)),
                                rng)
        analytic = _hilbert(message)
        baseband = analytic if params.get("side", "usb") == "usb" else np.conj(analytic)
        meta["side"] = str(params.get("side", "usb"))
    elif key == "fm":
        message = _tone_message(t, params.get("message_freqs_hz", (1000.0,)),
                                rng)
        deviation = float(params.get("deviation_hz", 6000.0))
        baseband = np.exp(2j * np.pi * deviation * np.cumsum(message) / spec.fs_hz)
        meta["deviation_hz"] = deviation
    elif key in ("ask2", "bpsk", "qpsk", "psk8", "qam16", "qam64"):
        baseband, extra = _linear_baseband(
            rng, spec.fs_hz, count, key,
            float(params.get("symbol_rate", 1000.0)),
            float(params.get("alpha", 0.35)))
        meta.update(extra)
    elif key in ("fsk2", "fsk4", "fsk8"):
        baseband, extra = _cpfsk_baseband(
            rng, spec.fs_hz, count, key,
            float(params.get("symbol_rate", 1000.0)),
            float(params.get("spacing_hz", 4000.0)))
        meta.update(extra)
    elif key == "fsk2_burst":
        baseband, extra = _cpfsk_baseband(
            rng, spec.fs_hz, count, "fsk2",
            float(params.get("symbol_rate", 1000.0)),
            float(params.get("spacing_hz", 4000.0)))
        meta.update(extra)
        baseband, envelope = apply_gate(baseband, spec.fs_hz, gate)
        meta["gate_duty"] = float(gate.duty) if gate else 0.0
        meta["gate_burst_count"] = int(gate.burst_count(spec.duration_s)) if gate else 0
    elif key == "fh_fsk2":
        baseband, extra = _cpfsk_baseband(
            rng, spec.fs_hz, count, "fsk2",
            float(params.get("symbol_rate", 1000.0)),
            float(params.get("spacing_hz", 4000.0)))
        meta.update(extra)
        baseband, hop_meta = apply_hop(baseband, spec.fs_hz, spec.hop, rng)
        meta.update(hop_meta)
    elif key == "link11_single":
        bit_rate = float(params.get("key_rate", 1364.0))
        baseband = _ook_at_bit_rate(t, spec.fs_hz, bit_rate, rng).astype(np.complex128)
        meta["key_rate"] = bit_rate
        if gate is not None:
            baseband, envelope = apply_gate(baseband, spec.fs_hz, gate)
            meta["gate_duty"] = float(gate.duty)
            meta["gate_burst_count"] = int(gate.burst_count(spec.duration_s))
    elif key == "audio_multichannel":
        channels = int(params.get("channels", 5))
        spacing = float(params.get("spacing_hz", 7000.0))
        bandwidth = float(params.get("audio_bandwidth_hz", 1800.0))
        offsets = (np.arange(channels) - (channels - 1) / 2.0) * spacing
        baseband = np.zeros(count, dtype=np.complex128)
        for offset in offsets:
            audio = _audio_channel(t, rng, bandwidth)
            baseband += audio * np.exp(2j * np.pi * offset * t)
        meta.update(audio_band_count=channels, audio_spacing_hz=spacing,
                    audio_bandwidth_hz=bandwidth)
    elif key == "cw":
        baseband = np.ones(count, dtype=np.complex128)
        if gate is not None:
            baseband, envelope = apply_gate(baseband, spec.fs_hz, gate)
            meta["gate_duty"] = float(gate.duty)
            meta["gate_burst_count"] = int(gate.burst_count(spec.duration_s))
    elif key == "multitone":
        tones = int(params.get("tones", 4))
        spacing = float(params.get("spacing_hz", 1800.0))
        offsets = (np.arange(tones) - (tones - 1) / 2.0) * spacing
        phases = rng.uniform(0, 2 * np.pi, size=tones)
        amps = rng.uniform(0.7, 1.0, size=tones)
        baseband = np.zeros(count, dtype=np.complex128)
        for amplitude, offset, phase in zip(amps, offsets, phases):
            baseband += amplitude * np.exp(2j * np.pi * offset * t + 1j * phase)
        meta.update(tone_count=tones, tone_spacing_hz=spacing)
    elif key == "noise":
        baseband = (rng.standard_normal(count)
                    + 1j * rng.standard_normal(count)).astype(np.complex128)
        meta["noise_kind"] = "white"
    else:
        raise ValueError(f"unsupported drill signal: {key}")

    if baseband is None:
        raise ValueError(f"signal generator did not produce {key}")
    if key == "noise":
        # Pure noise: no separate SNR calculation, just normalize the record.
        samples = np.real(baseband * np.exp(
            2j * np.pi * spec.f_if_hz * t))
        peak = float(np.max(np.abs(samples)))
        scale = 0.9 / peak if peak > 0 else 1.0
        samples = (samples * scale).astype(np.float32)
        received = (baseband * scale).astype(np.complex64)
        noise_variance = float(np.mean(np.abs(received) ** 2))
    else:
        samples, received, noise_variance = _add_noise_and_if(
            baseband, spec.fs_hz, spec.f_if_hz, spec.snr_db, rng)

    # The constellation is drawn from the known symbol stream plus the same
    # complex noise level.  This is an auxiliary teaching view, not a blind
    # receiver; using the generator's symbol timing keeps the clusters stable.
    if info.constellation_size and "symbols" in meta:
        symbols = np.asarray(meta["symbols"], dtype=np.complex128)
        show = min(4000, symbols.size)
        chosen = rng.choice(symbols.size, size=show, replace=False)
        sigma = np.sqrt(max(noise_variance, 0.0) / 2.0) if key != "noise" else 0.0
        points = symbols[chosen] + sigma * (
            rng.standard_normal(show) + 1j * rng.standard_normal(show))
        # ``received`` is already scaled; symbol values are in the same units
        # before the final peak normalization.  Estimate the scale by matching
        # the received baseband power to the ideal symbol power.
        ideal_power = float(np.mean(np.abs(symbols) ** 2))
        received_power = float(np.mean(np.abs(received) ** 2))
        scale = np.sqrt(max(received_power, 1e-30) / max(ideal_power, 1e-30))
        meta["constellation_points"] = (points * scale).astype(np.complex64)
        meta["constellation_ideal"] = (
            np.unique(np.round(meta["constellation"], 8)) * scale
        ).astype(np.complex64)
        meta["constellation_size"] = int(info.constellation_size)
    else:
        meta["constellation_points"] = None
        meta["constellation_ideal"] = None
        meta["constellation_size"] = 0

    meta.update(
        seed=int(spec.seed),
        fs_hz=float(spec.fs_hz),
        duration_s=float(spec.duration_s),
        f_if_hz=float(spec.f_if_hz),
        snr_db=float(spec.snr_db),
        noise_variance=float(noise_variance),
    )
    return Waveform(samples=samples, baseband=received, meta=meta)
