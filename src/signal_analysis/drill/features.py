"""DSP and measurement helpers for drill rounds.

The page only needs three main views plus auxiliary evidence.  All heavy
arrays are computed once when a round is generated: playback then indexes the
pre-computed STFT, so a 30 fps timer never performs an FFT.
"""

from dataclasses import dataclass

import numpy as np


#: STFT frame length at fs=200 kHz: 20.48 ms, enough to resolve 1 kHz sidebands.
DEFAULT_NFFT = 4096
#: 10 ms hop gives about 500 frames over the 5 s record.
DEFAULT_HOP_S = 0.01


@dataclass
class Spectrum:
    frequencies_hz: np.ndarray
    frequencies_khz: np.ndarray
    frame_db: np.ndarray
    frame_times_s: np.ndarray
    # Prefix (frame 0..i) cumulative spectra used by the accumulating UI.
    progressive_max_db: np.ndarray
    progressive_avg_db: np.ndarray
    # Full-record values retained for answer measurements.
    cumulative_max_db: np.ndarray
    cumulative_avg_db: np.ndarray
    nfft: int
    hop_samples: int

    @property
    def frame_count(self):
        return int(self.frame_db.shape[0])

    def frame_index(self, position_s):
        if self.frame_count == 0:
            return 0
        index = int(np.searchsorted(self.frame_times_s, float(position_s),
                                    side="right") - 1)
        return min(max(index, 0), self.frame_count - 1)


def _hilbert(values):
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


def stft_spectrum(samples, fs_hz, nfft=DEFAULT_NFFT, hop_s=DEFAULT_HOP_S):
    """Compute a single-sided STFT and cumulative max-hold/average spectra.

    Both the full-record result and per-frame prefix arrays are retained.  The
    page indexes the prefix arrays so its cumulative view grows with playback
    instead of showing the whole-record result from the first frame.
    """
    x = np.asarray(samples, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("empty signal")
    nfft = int(nfft)
    if nfft < 64 or nfft > 8192:
        raise ValueError("nfft must be between 64 and 8192")
    hop = max(1, int(round(float(hop_s) * float(fs_hz))))
    padded = np.pad(x, (0, max(0, nfft - x.size)))
    starts = np.arange(0, padded.size - nfft + 1, hop, dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)
    window = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(padded, nfft)[starts]
    transformed = np.fft.rfft(frames * window, axis=1)
    psd = np.abs(transformed) ** 2 / (float(fs_hz) * float(np.sum(window ** 2)))
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0
    reference = float(np.max(psd))
    reference_db = 10.0 * np.log10(max(reference, 1e-30))
    frame_db = (10.0 * np.log10(np.maximum(psd, 1e-30))
                - reference_db).astype(np.float32)
    progressive_max = np.empty_like(frame_db)
    progressive_avg = np.empty_like(frame_db)
    running_max = np.zeros(psd.shape[1], dtype=np.float64)
    running_sum = np.zeros(psd.shape[1], dtype=np.float64)
    for index in range(psd.shape[0]):
        np.maximum(running_max, psd[index], out=running_max)
        running_sum += psd[index]
        progressive_max[index] = (10.0 * np.log10(
            np.maximum(running_max, 1e-30)) - reference_db)
        progressive_avg[index] = (10.0 * np.log10(
            np.maximum(running_sum / (index + 1), 1e-30)) - reference_db)
    cumulative_max = progressive_max[-1].copy()
    cumulative_avg = progressive_avg[-1].copy()
    frequencies = np.fft.rfftfreq(nfft, 1.0 / float(fs_hz))
    return Spectrum(
        frequencies_hz=frequencies,
        frequencies_khz=frequencies / 1e3,
        frame_db=frame_db,
        frame_times_s=(starts + nfft / 2.0) / float(fs_hz),
        progressive_max_db=progressive_max,
        progressive_avg_db=progressive_avg,
        cumulative_max_db=cumulative_max,
        cumulative_avg_db=cumulative_avg,
        nfft=nfft,
        hop_samples=hop,
    )


def find_peaks_db(db, frequencies_hz, min_db=12.0, min_sep_hz=300.0,
                  max_peaks=32):
    """Small local-maximum peak detector used by FSK/multi-tone evidence."""
    values = np.asarray(db, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if values.size < 3:
        return []
    threshold = float(np.max(values)) - float(min_db)
    candidates = np.where(
        (values[1:-1] > values[:-2])
        & (values[1:-1] >= values[2:])
        & (values[1:-1] >= threshold)
    )[0] + 1
    if candidates.size == 0:
        return []
    ordered = candidates[np.argsort(values[candidates])[::-1]]
    selected = []
    for index in ordered:
        frequency = float(frequencies[index])
        if all(abs(frequency - float(frequencies[other])) >= float(min_sep_hz)
               for other in selected):
            selected.append(int(index))
        if len(selected) >= int(max_peaks):
            break
    return sorted(selected, key=lambda index: float(frequencies[index]))


def _carrier_ratio(cumulative_db, frequencies_hz, f_if_hz):
    offsets = frequencies_hz - float(f_if_hz)
    center_mask = np.abs(offsets) <= 500.0
    side_mask = (np.abs(offsets) > 1000.0) & (np.abs(offsets) <= 20_000.0)
    if not np.any(center_mask) or not np.any(side_mask):
        return 0.0
    center = float(np.max(cumulative_db[center_mask]))
    side = float(np.median(cumulative_db[side_mask]))
    return center - side


def _spectral_symmetry(cumulative_db, frequencies_hz, f_if_hz,
                       half_span_hz=20_000.0):
    offsets = frequencies_hz - float(f_if_hz)
    grid = np.linspace(0.0, float(half_span_hz), 200)
    left_mask = (offsets < 0) & (offsets >= -half_span_hz)
    right_mask = (offsets > 0) & (offsets <= half_span_hz)
    if not np.any(left_mask) or not np.any(right_mask):
        return 0.0
    left = np.interp(grid, -offsets[left_mask][::-1],
                     cumulative_db[left_mask][::-1])
    right = np.interp(grid, offsets[right_mask], cumulative_db[right_mask])
    left_power = float(np.mean(10.0 ** (left / 10.0)))
    right_power = float(np.mean(10.0 ** (right / 10.0)))
    total = left_power + right_power
    if total <= 0:
        return 0.0
    return 1.0 - abs(left_power - right_power) / total


def _occupied_bandwidth(cumulative_db, frequencies_hz, threshold_db=20.0):
    """Width between the first and last bins within ``threshold_db`` of peak.

    Using a relative threshold instead of a 99% energy integral keeps a weak
    noise floor from widening an AM/SSB measurement while still spanning all
    hop bands of an FH signal.
    """
    values = np.asarray(cumulative_db, dtype=np.float64)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if values.size == 0:
        return 0.0
    mask = values >= (float(np.max(values)) - float(threshold_db))
    if not np.any(mask):
        return 0.0
    indices = np.flatnonzero(mask)
    return float(frequencies[indices[-1]] - frequencies[indices[0]])


def _spectral_flatness(cumulative_avg_db):
    power = 10.0 ** (np.asarray(cumulative_avg_db, dtype=np.float64) / 10.0)
    power = np.maximum(power, 1e-30)
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def envelope_statistics(samples, fs_hz):
    """Return envelope CV, keyed duty cycle and rising-edge count."""
    envelope = np.abs(_hilbert(samples))
    mean = float(np.mean(envelope))
    cv = float(np.std(envelope) / mean) if mean > 0 else 0.0
    threshold = 0.3 * float(np.max(envelope))
    binary = envelope >= threshold
    duty = float(np.mean(binary))
    if binary.size > 1:
        edges = int(np.count_nonzero(binary[1:] & ~binary[:-1]))
    else:
        edges = 0
    return cv, duty, edges


def measure(spec, waveform, spectrum):
    """Compute the small set of evidence values shown after answering."""
    meta = waveform.meta
    samples = waveform.samples
    frequencies = spectrum.frequencies_hz
    cumulative_max = spectrum.cumulative_max_db
    cumulative_avg = spectrum.cumulative_avg_db
    key = spec.key

    env_cv, gate_duty, burst_count = envelope_statistics(
        samples, spec.fs_hz)
    spacing = float(meta.get("tone_spacing_hz")
                    or meta.get("audio_spacing_hz")
                    or 1000.0)
    min_sep = max(300.0, 0.5 * spacing)
    peaks = find_peaks_db(cumulative_max, frequencies,
                          min_db=8.0, min_sep_hz=min_sep)
    peak_frequencies = [float(frequencies[index]) for index in peaks]
    features = {
        "key": key,
        "class_label": meta.get("class_label", key),
        "envelope_cv": env_cv,
        "gate_duty": float(meta.get("gate_duty", 1.0 if key not in
                                    ("fsk2_burst", "link11_single", "cw") else gate_duty)),
        "burst_count": int(meta.get("gate_burst_count", 0 if key not in
                                    ("fsk2_burst", "link11_single", "cw") else burst_count)),
        "peak_count": len(peaks),
        "peak_frequencies_hz": peak_frequencies,
        "carrier_ratio_db": _carrier_ratio(cumulative_max, frequencies,
                                           spec.f_if_hz),
        "symmetry": _spectral_symmetry(cumulative_max, frequencies,
                                       spec.f_if_hz),
        "occupied_bandwidth_hz": _occupied_bandwidth(
            cumulative_max, frequencies),
        "spectral_flatness": _spectral_flatness(cumulative_avg),
        "snr_db": float(spec.snr_db),
        "center_frequency_hz": float(spec.f_if_hz),
        "symbol_rate": float(meta.get("symbol_rate", 0.0)),
        "constellation_size": int(meta.get("constellation_size", 0)),
        "hop_channels": len(meta.get("hop_channels_hz", ()) or ()),
        "hop_dwell_s": float(meta.get("hop_dwell_s", 0.0)),
        "audio_band_count": int(meta.get("audio_band_count", 0)),
        "tone_count": int(meta.get("tone_count", 0)),
        "key_rate": float(meta.get("key_rate", 0.0)),
        "fsk_order": int(meta.get("fsk_order", 0)),
    }
    return features


def iq_window(baseband, fs_hz, position_s, duration_s, window_s=0.05):
    """Return a short baseband I/Q slice centred on the playhead."""
    baseband = np.asarray(baseband)
    count = baseband.size
    window_samples = max(2, int(round(float(window_s) * float(fs_hz))))
    window_samples = min(window_samples, max(2, count))
    start = int(round(float(position_s) * float(fs_hz) - window_samples / 2.0))
    start = min(max(start, 0), max(0, count - window_samples))
    stop = min(count, start + window_samples)
    t = np.arange(stop - start, dtype=np.float64) / float(fs_hz)
    return t, baseband.real[start:stop].astype(np.float64), baseband.imag[start:stop].astype(np.float64)


def hop_track_khz(meta, times_s):
    """Return a hop staircase (kHz) sampled at the requested times."""
    channels = meta.get("hop_channels_hz") or ()
    sequence = meta.get("hop_sequence") or ()
    dwell = float(meta.get("hop_dwell_s", 0.0))
    if not channels or not sequence or dwell <= 0:
        return None
    times = np.asarray(times_s, dtype=np.float64)
    indices = np.floor(times / dwell).astype(np.int64) % len(sequence)
    values = np.asarray(channels, dtype=np.float64)[
        np.asarray(sequence, dtype=np.int64)[indices]]
    return values / 1e3
