"""Core tests for the manual signal-drill module."""

import numpy as np
import pytest

from signal_analysis.drill.catalog import CLASS_BY_KEY, CLASSES, KEYS
from signal_analysis.drill.features import measure, stft_spectrum
from signal_analysis.drill.session import DrillSession, _build_spec
from signal_analysis.drill.waveforms import generate_waveform


def _round_for(key, seed=11, duration=1.0, snr_key=None):
    index = KEYS.index(key)
    rng = np.random.default_rng(1000 + seed * 37 + index)
    spec = _build_spec(key, 7000 + seed * 37 + index, 200_000.0,
                       duration, 45_000.0, rng)
    if snr_key is not None:
        from dataclasses import replace
        spec = replace(spec, snr_db=float(snr_key))
    waveform = generate_waveform(spec, np.random.default_rng(spec.seed))
    spectrum = stft_spectrum(waveform.samples, spec.fs_hz)
    features = measure(spec, waveform, spectrum)
    return spec, waveform, spectrum, features


def test_catalog_is_one_level_and_complete():
    assert len(CLASSES) == 20
    assert len(set(KEYS)) == len(KEYS)
    assert set(KEYS) == set(CLASS_BY_KEY)
    assert all(item.aux_tab in ("constellation", "iq", "timefreq")
               for item in CLASSES)
    assert set(CLASS_BY_KEY[key].aux_tab for key in
               ("bpsk", "qpsk", "psk8", "qam16", "qam64")) == {"constellation"}
    # No basic/advanced attributes in the single bank.
    assert not any(hasattr(item, "difficulty") for item in CLASSES)


def test_every_class_generates_a_finite_real_if_record():
    for key in KEYS:
        _, waveform, spectrum, features = _round_for(key)
        assert waveform.samples.ndim == 1
        assert waveform.samples.dtype == np.float32
        assert waveform.samples.size == 200_000
        assert np.all(np.isfinite(waveform.samples))
        assert np.max(np.abs(waveform.samples)) > 0
        assert spectrum.frame_db.shape[0] > 0
        assert spectrum.frame_db.shape[1] == 2049
        assert features["key"] == key


def test_cumulative_spectra_are_progressive_prefixes():
    _, _, spectrum, _ = _round_for("fh_fsk2", duration=1.0)
    progressive_max = spectrum.progressive_max_db
    progressive_avg = spectrum.progressive_avg_db
    assert progressive_max.shape == spectrum.frame_db.shape
    assert progressive_avg.shape == spectrum.frame_db.shape
    assert not np.array_equal(progressive_max[0], progressive_max[-1])
    assert np.all(np.diff(progressive_max, axis=0) >= -1e-4)
    np.testing.assert_allclose(progressive_max[-1],
                               spectrum.cumulative_max_db, atol=1e-4)
    np.testing.assert_allclose(progressive_avg[-1],
                               spectrum.cumulative_avg_db, atol=1e-4)
    np.testing.assert_allclose(progressive_avg[0],
                               spectrum.frame_db[0], atol=1e-4)


@pytest.mark.parametrize("key, expected", [("fsk2", 2), ("fsk4", 4), ("fsk8", 8)])
def test_fsk_peak_count_matches_order(key, expected):
    for seed in range(3):
        _, _, _, features = _round_for(key, seed=seed)
        assert features["peak_count"] == expected
        frequencies = np.asarray(features["peak_frequencies_hz"])
        assert np.all((frequencies > 20_000) & (frequencies < 70_000))


@pytest.mark.parametrize(
    "key, size",
    [("bpsk", 2), ("qpsk", 4), ("psk8", 8), ("qam16", 16), ("qam64", 64)])
def test_constellation_has_expected_cluster_count(key, size):
    _, waveform, _, features = _round_for(key, snr_key=30.0)
    points = np.asarray(waveform.meta["constellation_points"])
    ideal = np.asarray(waveform.meta["constellation_ideal"])
    assert features["constellation_size"] == size
    assert points.size > 100
    assert ideal.size == size
    # Every received point should be close to one of the ideal points at 30 dB.
    distances = np.abs(points[:, None] - ideal[None, :])
    nearest = np.min(distances, axis=1)
    assert np.percentile(nearest, 95) < 0.35


def test_gate_and_hop_metadata_are_visible_in_features():
    _, _, _, burst = _round_for("fsk2_burst", duration=2.0)
    assert 0.2 < burst["gate_duty"] < 1.0
    assert burst["burst_count"] >= 3

    _, _, _, hop = _round_for("fh_fsk2", duration=2.0)
    assert hop["hop_channels"] == 8
    assert hop["hop_dwell_s"] > 0
    assert hop["peak_count"] >= 8

    _, _, _, link = _round_for("link11_single", duration=2.0)
    assert link["key_rate"] == 1364.0
    assert link["gate_duty"] < 1.0
    assert link["burst_count"] >= 3

    _, _, _, audio = _round_for("audio_multichannel", duration=2.0)
    assert 4 <= audio["audio_band_count"] <= 6
    assert audio["occupied_bandwidth_hz"] > 10_000


def test_noise_has_high_spectral_flatness():
    _, _, _, features = _round_for("noise")
    assert features["spectral_flatness"] > 0.75
    assert features["occupied_bandwidth_hz"] > 50_000


def test_session_playback_state_machine_and_scoring():
    session = DrillSession(seed=42, duration_s=1.0)
    first = session.new_round()
    assert session.phase == "ready"
    session.play()
    assert session.phase == "playing"
    session.advance(0.4)
    assert 0.39 < session.position_s < 0.41
    session.pause()
    assert session.phase == "paused"
    session.play()
    session.advance(2.0)
    assert session.phase == "ended"
    assert session.position_s == 1.0
    # Replay from zero after the end.
    session.play()
    assert session.phase == "playing" and session.position_s == 0.0
    session.pause()
    session.select(first.spec.key)
    result = session.reveal()
    assert result.answered and result.correct
    assert session.correct_count == 1 and session.answered_count == 1
    assert session.streak == 1
    # Revealing is idempotent and cannot change the score.
    assert session.reveal() is result
    assert session.correct_count == 1

    wrong_round = session.new_round()
    wrong_key = next(key for key in KEYS if key != wrong_round.spec.key)
    session.select(wrong_key)
    wrong = session.reveal()
    assert wrong.answered and not wrong.correct
    assert session.streak == 0
    assert session.answered_count == 2 and session.correct_count == 1

    unanswered_round = session.new_round()
    unanswered = session.reveal()
    assert not unanswered.answered and not unanswered.correct
    assert session.answered_count == 2 and session.correct_count == 1
    assert "未作答" in unanswered.explanation

    # Restart clears the answer/selection but keeps the running score.
    session.new_round()
    assert session.phase == "ready" and session.selected_key is None
    assert session.correct_count == 1 and session.answered_count == 2


def test_same_seed_reproduces_first_round():
    left = DrillSession(seed=12345, duration_s=0.6)
    right = DrillSession(seed=12345, duration_s=0.6)
    first = left.new_round()
    second = right.new_round()
    assert first.spec == second.spec
    assert np.array_equal(first.waveform.samples, second.waveform.samples)
    assert np.array_equal(first.spectrum.frame_db, second.spectrum.frame_db)


def test_one_bag_cycle_covers_all_classes():
    session = DrillSession(seed=99, duration_s=0.5)
    seen = {session.new_round().spec.key for _ in range(len(KEYS))}
    assert seen == set(KEYS)


def test_every_class_can_be_answered_correctly():
    session = DrillSession(seed=2024, duration_s=0.6)
    for _ in range(len(KEYS)):
        round_data = session.new_round()
        session.select(round_data.spec.key)
        result = session.reveal()
        assert result.answered and result.correct
        assert result.truth_key == round_data.spec.key
        assert result.metrics_text
        assert result.note
        assert round_data.info.cues[0] in result.explanation
