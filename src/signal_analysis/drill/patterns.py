"""Time gating and frequency-hopping patterns shared by drill signals.

The patterns are intentionally protocol-agnostic: a caller supplies the
baseband waveform and these helpers apply a burst envelope, a hop trajectory,
or both.  Keeping the pattern layer separate makes the burst/hopping question
types reproducible and prevents each modulation generator from growing its own
copy of the schedule.
"""

from dataclasses import dataclass
import math

import numpy as np


def _raised_cosine(values):
    """Map ``values`` in [0, 1] to a smooth 0→1 raised-cosine ramp."""
    x = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


@dataclass(frozen=True)
class GateSpec:
    """Periodic burst gate.

    ``burst_s`` is the on-air part of each period and ``gap_s`` the silent
    part.  ``edge_s`` is the duration of each raised-cosine edge.  The envelope
    is normalized to 1 in the middle of a burst, so the caller can apply it to
    a unit-power baseband signal without changing the on-air level.
    """

    burst_s: float
    gap_s: float
    edge_s: float = 1e-3

    def __post_init__(self):
        if not math.isfinite(self.burst_s) or self.burst_s <= 0:
            raise ValueError("burst_s must be a positive finite number")
        if not math.isfinite(self.gap_s) or self.gap_s < 0:
            raise ValueError("gap_s must be a non-negative finite number")
        if not math.isfinite(self.edge_s) or self.edge_s <= 0:
            raise ValueError("edge_s must be a positive finite number")
        if self.burst_s <= 2 * self.edge_s:
            raise ValueError("burst_s must be larger than two edge durations")

    @property
    def period_s(self):
        return self.burst_s + self.gap_s

    @property
    def duty(self):
        return self.burst_s / self.period_s

    def envelope(self, t):
        """Return the periodic gate envelope for time vector ``t``."""
        t = np.asarray(t, dtype=np.float64)
        phase = np.mod(t, self.period_s)
        rise = _raised_cosine(phase / self.edge_s)
        fall = _raised_cosine((self.burst_s - phase) / self.edge_s)
        envelope = np.minimum(rise, fall)
        envelope = np.where(phase >= self.burst_s, 0.0, envelope)
        return envelope

    def burst_count(self, duration_s):
        if duration_s <= 0:
            return 0
        return int(math.ceil(duration_s / self.period_s))


@dataclass(frozen=True)
class HopSpec:
    """Frequency-hopping schedule.

    ``channels_hz`` are baseband offsets (Hz), not absolute RF frequencies.
    A random permutation is repeated when ``sequence`` is None.  The helper
    applies a short gain dip at hop boundaries so the finite observation does
    not turn into a broadband splatter mask.
    """

    channels_hz: tuple[float, ...]
    dwell_s: float
    sequence: tuple[int, ...] | None = None
    transition_s: float = 5e-4

    def __post_init__(self):
        if len(self.channels_hz) < 2:
            raise ValueError("a hop set must contain at least two channels")
        if not all(math.isfinite(value) for value in self.channels_hz):
            raise ValueError("hop channels must be finite")
        if not math.isfinite(self.dwell_s) or self.dwell_s <= 0:
            raise ValueError("dwell_s must be a positive finite number")
        if not math.isfinite(self.transition_s) or self.transition_s < 0:
            raise ValueError("transition_s must be non-negative")
        if self.transition_s >= self.dwell_s / 2:
            raise ValueError("transition_s must be shorter than half a dwell")
        if self.sequence is not None:
            if not self.sequence:
                raise ValueError("hop sequence must not be empty")
            if any(not 0 <= index < len(self.channels_hz) for index in self.sequence):
                raise ValueError("hop sequence index out of range")

    def channel_sequence(self, n_hops, rng):
        """Return ``n_hops`` channel indices without adjacent repeats."""
        if n_hops <= 0:
            return ()
        if self.sequence is not None:
            base = list(self.sequence)
        else:
            base = [int(value) for value in rng.permutation(len(self.channels_hz))]
        result = []
        while len(result) < n_hops:
            block = list(base)
            if result and block[0] == result[-1] and len(block) > 1:
                block = block[1:] + block[:1]
            result.extend(block)
        return tuple(result[:n_hops])

    def frequency_track(self, count, fs_hz, rng):
        """Return per-sample hop frequencies and the schedule metadata."""
        count = int(count)
        if count <= 0:
            raise ValueError("sample count must be positive")
        dwell_samples = max(1, int(round(self.dwell_s * fs_hz)))
        n_hops = int(math.ceil(count / dwell_samples))
        sequence = self.channel_sequence(n_hops, rng)
        channel_values = np.asarray(self.channels_hz, dtype=np.float64)
        frequencies = np.repeat(channel_values[list(sequence)], dwell_samples)[:count]
        return frequencies, sequence, dwell_samples

    def transition_gain(self, count, fs_hz, dwell_samples):
        """A short raised-cosine dip around every hop boundary."""
        count = int(count)
        gain = np.ones(count, dtype=np.float64)
        edge = max(1, int(round(self.transition_s * fs_hz)))
        if self.transition_s <= 0 or edge <= 1:
            return gain
        for boundary in range(dwell_samples, count, dwell_samples):
            start = max(0, boundary - edge)
            stop = min(count, boundary + edge + 1)
            offs = np.arange(start, stop, dtype=np.float64) - boundary
            local = 0.5 - 0.5 * np.cos(np.pi * offs / edge)
            gain[start:stop] = np.minimum(gain[start:stop], local)
        return gain


def apply_gate(baseband, fs_hz, gate):
    """Apply a periodic burst envelope to a baseband signal."""
    if gate is None:
        return np.asarray(baseband, dtype=np.complex128), np.ones(
            np.asarray(baseband).size, dtype=np.float64)
    baseband = np.asarray(baseband, dtype=np.complex128)
    t = np.arange(baseband.size, dtype=np.float64) / float(fs_hz)
    envelope = gate.envelope(t)
    return baseband * envelope, envelope


def apply_hop(baseband, fs_hz, hop, rng):
    """Frequency-shift a baseband signal according to ``hop``.

    Returns ``(shifted, metadata)`` where metadata contains the channel set,
    sequence and dwell time.  The per-sample frequency track is deliberately
    not retained: the time-frequency view can derive its staircase from the
    metadata without carrying a second million-element array in every round.
    """
    if hop is None:
        return np.asarray(baseband, dtype=np.complex128), {}
    baseband = np.asarray(baseband, dtype=np.complex128)
    frequencies, sequence, dwell_samples = hop.frequency_track(
        baseband.size, fs_hz, rng)
    phase = 2 * np.pi * np.cumsum(frequencies) / float(fs_hz)
    shifted = baseband * np.exp(1j * phase)
    shifted *= hop.transition_gain(baseband.size, fs_hz, dwell_samples)
    metadata = {
        "hop_channels_hz": tuple(float(value) for value in hop.channels_hz),
        "hop_sequence": tuple(int(value) for value in sequence),
        "hop_dwell_s": float(hop.dwell_s),
        "hop_transition_s": float(hop.transition_s),
    }
    return shifted, metadata
