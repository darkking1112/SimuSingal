"""Question generation and scoring for the manual signal-drill page."""

from dataclasses import dataclass

import numpy as np

from .catalog import CLASS_BY_KEY, KEYS, SignalClass, SignalSpec
from .features import Spectrum, measure, stft_spectrum
from .patterns import GateSpec, HopSpec
from .waveforms import Waveform, generate_waveform


DEFAULT_FS_HZ = 200_000.0
DEFAULT_DURATION_S = 5.0
DEFAULT_F_IF_HZ = 45_000.0

PHASES = ("ready", "playing", "paused", "ended")


@dataclass
class Round:
    spec: SignalSpec
    info: SignalClass
    waveform: Waveform
    spectrum: Spectrum
    features: dict

    @property
    def duration_s(self):
        return float(self.spec.duration_s)

    @property
    def fs_hz(self):
        return float(self.spec.fs_hz)

    @property
    def frame_db(self):
        return self.spectrum.frame_db

    @property
    def frame_times_s(self):
        return self.spectrum.frame_times_s

    @property
    def freqs_khz(self):
        return self.spectrum.frequencies_khz

    def frame_index(self, position_s):
        return self.spectrum.frame_index(position_s)

    def spectrum_at(self, position_s):
        return self.spectrum.frame_db[self.frame_index(position_s)]


@dataclass(frozen=True)
class AnswerResult:
    answered: bool
    correct: bool
    selected_key: str | None
    selected_label: str | None
    truth_key: str
    truth_label: str
    cues: tuple[str, ...]
    metrics_text: str
    explanation: str
    note: str
    aux_tab: str


def _uniform(rng, low, high):
    return float(rng.uniform(float(low), float(high)))


def _message_freqs(rng, count=2, low=700.0, high=2600.0):
    base = _uniform(rng, low, min(high, 1800.0))
    values = [base]
    if count > 1:
        second = base * _uniform(rng, 1.25, 1.9)
        values.append(min(second, high))
    if count > 2:
        third = base * _uniform(rng, 0.55, 0.85)
        values.append(max(third, low * 0.7))
    return tuple(round(value, 1) for value in values)


def _build_spec(key, seed, fs_hz, duration_s, f_if_hz, rng):
    params = {}
    gate = None
    hop = None
    snr = _uniform(rng, 20.0, 28.0)

    if key == "am":
        params = {"message_freqs_hz": _message_freqs(rng, 1),
                  "depth": _uniform(rng, 0.45, 0.9)}
        snr = _uniform(rng, 18.0, 26.0)
    elif key == "dsb":
        params = {"message_freqs_hz": _message_freqs(rng, int(rng.integers(2, 4)))}
        snr = _uniform(rng, 18.0, 26.0)
    elif key == "ssb":
        params = {"message_freqs_hz": _message_freqs(rng, int(rng.integers(2, 4))),
                  "side": "usb" if rng.random() < 0.5 else "lsb"}
        snr = _uniform(rng, 18.0, 26.0)
    elif key == "fm":
        params = {"message_freqs_hz": _message_freqs(rng, 1),
                  "deviation_hz": _uniform(rng, 4000.0, 8000.0)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key in ("ask2", "bpsk", "qpsk", "psk8"):
        params = {"symbol_rate": _uniform(rng, 500.0, 1500.0),
                  "alpha": _uniform(rng, 0.3, 0.45)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "qam16":
        params = {"symbol_rate": _uniform(rng, 500.0, 1200.0),
                  "alpha": _uniform(rng, 0.3, 0.45)}
        snr = _uniform(rng, 22.0, 29.0)
    elif key == "qam64":
        params = {"symbol_rate": _uniform(rng, 400.0, 1000.0),
                  "alpha": _uniform(rng, 0.3, 0.45)}
        snr = _uniform(rng, 24.0, 30.0)
    elif key in ("fsk2", "fsk2_burst", "fh_fsk2"):
        params = {"symbol_rate": _uniform(rng, 500.0, 1500.0),
                  "spacing_hz": _uniform(rng, 3000.0, 6000.0)}
        snr = _uniform(rng, 20.0, 28.0)
        if key == "fsk2_burst":
            gate = GateSpec(
                burst_s=_uniform(rng, 0.08, 0.18),
                gap_s=_uniform(rng, 0.03, 0.09),
                edge_s=0.001,
            )
        elif key == "fh_fsk2":
            channels = tuple(float(value) for value in
                             np.linspace(-28_000.0, 28_000.0, 8))
            hop = HopSpec(
                channels_hz=channels,
                dwell_s=_uniform(rng, 0.012, 0.05),
                transition_s=0.0005,
            )
    elif key == "fsk4":
        params = {"symbol_rate": _uniform(rng, 400.0, 1000.0),
                  "spacing_hz": _uniform(rng, 1500.0, 2300.0)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "fsk8":
        params = {"symbol_rate": _uniform(rng, 300.0, 700.0),
                  "spacing_hz": _uniform(rng, 1200.0, 1800.0)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "link11_single":
        params = {"key_rate": 1364.0}
        gate = GateSpec(
            burst_s=_uniform(rng, 0.07, 0.15),
            gap_s=_uniform(rng, 0.025, 0.07),
            edge_s=0.0005,
        )
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "audio_multichannel":
        params = {"channels": int(rng.integers(4, 7)),
                  "spacing_hz": _uniform(rng, 6500.0, 8000.0),
                  "audio_bandwidth_hz": _uniform(rng, 1000.0, 2200.0)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "cw":
        params = {"keyed": bool(rng.random() < 0.55)}
        if params["keyed"]:
            gate = GateSpec(
                burst_s=_uniform(rng, 0.025, 0.09),
                gap_s=_uniform(rng, 0.015, 0.07),
                edge_s=0.001,
            )
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "multitone":
        params = {"tones": int(rng.integers(3, 6)),
                  "spacing_hz": _uniform(rng, 1200.0, 2300.0)}
        snr = _uniform(rng, 20.0, 28.0)
    elif key == "noise":
        params = {}
        snr = 0.0
    else:
        raise ValueError(f"unsupported drill key: {key}")

    return SignalSpec(
        key=key,
        seed=int(seed),
        fs_hz=float(fs_hz),
        duration_s=float(duration_s),
        f_if_hz=float(f_if_hz),
        snr_db=float(snr),
        params=params,
        gate=gate,
        hop=hop,
    )


def _fmt_bandwidth(value_hz):
    return f"{value_hz / 1e3:.1f} kHz"


class DrillSession:
    """UI-independent round, playback and scoring state machine."""

    def __init__(self, seed=None, fs_hz=DEFAULT_FS_HZ,
                 duration_s=DEFAULT_DURATION_S, f_if_hz=DEFAULT_F_IF_HZ):
        self.seed = int(seed if seed is not None else np.random.SeedSequence().entropy)
        self.fs_hz = float(fs_hz)
        self.duration_s = float(duration_s)
        self.f_if_hz = float(f_if_hz)
        self._rng = np.random.default_rng(self.seed)
        self._bag_rng = np.random.default_rng(self.seed ^ 0x5A17)
        self._bag = []
        self._last_key = None
        self.question_number = 0
        self.round = None
        self._phase = "ready"
        self._position_s = 0.0
        self._selected_key = None
        self._revealed = False
        self._result = None
        self._correct = 0
        self._answered = 0
        self._streak = 0
        self._best_streak = 0
        self.speed = 1.0

    # ------------------------------------------------------------- generation
    def _next_key(self):
        if not self._bag:
            self._bag = list(KEYS)
            self._bag_rng.shuffle(self._bag)
            if self._bag[-1] == self._last_key and len(self._bag) > 1:
                self._bag[0], self._bag[-1] = self._bag[-1], self._bag[0]
        key = self._bag.pop()
        self._last_key = key
        return key

    def new_round(self):
        key = self._next_key()
        self.question_number += 1
        seed = int(self._rng.integers(0, 2 ** 31 - 1))
        spec = _build_spec(key, seed, self.fs_hz, self.duration_s,
                           self.f_if_hz, self._rng)
        waveform = generate_waveform(spec, np.random.default_rng(spec.seed))
        spectrum = stft_spectrum(waveform.samples, spec.fs_hz)
        features = measure(spec, waveform, spectrum)
        self.round = Round(
            spec=spec,
            info=CLASS_BY_KEY[spec.key],
            waveform=waveform,
            spectrum=spectrum,
            features=features,
        )
        self._phase = "ready"
        self._position_s = 0.0
        self._selected_key = None
        self._revealed = False
        self._result = None
        return self.round

    # --------------------------------------------------------------- playback
    @property
    def phase(self):
        return self._phase

    @property
    def position_s(self):
        return self._position_s

    @property
    def selected_key(self):
        return self._selected_key

    @property
    def revealed(self):
        return self._revealed

    @property
    def correct_count(self):
        return self._correct

    @property
    def answered_count(self):
        return self._answered

    @property
    def streak(self):
        return self._streak

    @property
    def best_streak(self):
        return self._best_streak

    def play(self):
        if self.round is None:
            return
        if self._phase == "playing":
            return
        if self._phase == "ended":
            self._position_s = 0.0
        self._phase = "playing"

    def pause(self):
        if self._phase == "playing":
            self._phase = "paused"

    def toggle_play(self):
        if self._phase == "playing":
            self.pause()
        else:
            self.play()

    def seek(self, position_s):
        if self.round is None:
            return
        self._position_s = min(max(float(position_s), 0.0), self.duration_s)
        if self._phase == "ended" and self._position_s < self.duration_s:
            self._phase = "paused"

    def advance(self, dt_s):
        if self.round is None or self._phase != "playing":
            return
        step = max(0.0, float(dt_s)) * float(self.speed)
        self._position_s = min(self.duration_s, self._position_s + step)
        if self._position_s >= self.duration_s:
            self._position_s = self.duration_s
            self._phase = "ended"

    # ---------------------------------------------------------------- scoring
    def select(self, key):
        if self._revealed or self.round is None:
            return
        if key in CLASS_BY_KEY:
            self._selected_key = key

    def _metrics_text(self, round_data, features):
        parts = []
        if features.get("constellation_size"):
            parts.append(f"星座簇数 {features['constellation_size']}")
        if round_data.spec.key in ("fsk2", "fsk4", "fsk8"):
            parts.append(f"累积谱主峰 {features['peak_count']}")
        if round_data.spec.key == "fsk2_burst":
            parts.append(f"门控占空比 {features['gate_duty']:.2f}")
            parts.append(f"突发段 {features['burst_count']}")
        if round_data.spec.key == "fh_fsk2":
            parts.append(f"跳频信道 {features['hop_channels']}")
            parts.append(f"驻留 {features['hop_dwell_s'] * 1e3:.1f} ms")
        if round_data.spec.key == "link11_single":
            parts.append(f"键控速率 {features['key_rate']:.0f} Bd")
            parts.append(f"突发段 {features['burst_count']}")
        if features.get("audio_band_count"):
            parts.append(f"音频子带 {features['audio_band_count']}")
        if features.get("tone_count"):
            parts.append(f"离散单音 {features['tone_count']}")
        if round_data.spec.key == "cw" and features.get("gate_duty", 1.0) < 0.98:
            parts.append(f"占空比 {features['gate_duty']:.2f}")
        if features.get("symbol_rate"):
            parts.append(f"符号率 {features['symbol_rate']:.0f} Bd")
        parts.append(f"占用带宽 {_fmt_bandwidth(features['occupied_bandwidth_hz'])}")
        if round_data.spec.key in ("am", "dsb", "ssb", "fm", "cw", "ask2"):
            parts.append(f"载波/边带比 {features['carrier_ratio_db']:.1f} dB")
        return "；".join(parts)

    def _confusion_note(self, truth, selected, features):
        if selected is None:
            return "未作答，按第一次显示答案记录。"
        if truth.constellation_size and selected.constellation_size:
            return (f"PSK/QAM 阶数必须看星座图：本题星座为 "
                    f"{truth.constellation_size} 簇。")
        if truth.key in ("fsk4", "fsk8"):
            return (f"累积谱主峰数为 {features['peak_count']}，"
                    f"对应 {features.get('fsk_order', truth.key[-1])}FSK。")
        if truth.key == "fh_fsk2" and selected.key == "fsk2_burst":
            return "跳频的中心频率会变，累积谱呈多频带；2FSK 突发是固定双音的通断。"
        if truth.key == "fsk2_burst" and selected.key == "fh_fsk2":
            return "2FSK 突发是固定双音的通断；跳频的中心频率会按图案改变。"
        if truth.key == "link11_single" and selected.key == "cw":
            return "单音 LINK11 有 1364 Bd 突发图案；连续单音没有这种突发包络。"
        if truth.key == "audio_multichannel" and selected.key == "multitone":
            return "多路音频是多个音频子带；多音是若干离散纯音。"
        if truth.key == "multitone" and selected.key == "audio_multichannel":
            return "多音是离散纯音；多路音频的每个子带内还有音频调制边带。"
        return "请对照包络、谱峰数量和时频结构。"

    def reveal(self):
        if self.round is None:
            raise ValueError("no active round")
        if self._revealed:
            return self._result
        self.pause()
        truth = self.round.info
        selected = CLASS_BY_KEY.get(self._selected_key)
        answered = selected is not None
        correct = bool(answered and self._selected_key == truth.key)
        if answered:
            self._answered += 1
            if correct:
                self._correct += 1
                self._streak += 1
                self._best_streak = max(self._best_streak, self._streak)
            else:
                self._streak = 0
        else:
            self._streak = 0
        features = self.round.features
        metrics_text = self._metrics_text(self.round, features)
        cues = truth.cues
        note = self._confusion_note(truth, selected, features)
        lines = []
        if not answered:
            lines.append(f"未作答。正确答案：{truth.label}。")
        elif correct:
            lines.append(f"正确。{truth.label}。")
        else:
            lines.append(f"错误。你选择：{selected.label}；正确答案：{truth.label}。")
        lines.append("判读依据：" + "；".join(cues) + "。")
        lines.append("测量：" + metrics_text + "。")
        lines.append("易混点：" + note)
        result = AnswerResult(
            answered=answered,
            correct=correct,
            selected_key=selected.key if selected else None,
            selected_label=selected.label if selected else None,
            truth_key=truth.key,
            truth_label=truth.label,
            cues=cues,
            metrics_text=metrics_text,
            explanation="\n".join(lines),
            note=note,
            aux_tab=truth.aux_tab,
        )
        self._revealed = True
        self._result = result
        return result

    # ------------------------------------------------------------------ views
    def spectrum_at_position(self):
        if self.round is None:
            return None
        return self.round.spectrum_at(self._position_s)

    def constellation(self):
        if self.round is None:
            return None, None
        meta = self.round.waveform.meta
        return meta.get("constellation_points"), meta.get("constellation_ideal")

    def iq_window(self, window_s=0.05):
        from .features import iq_window
        if self.round is None:
            return None, None, None
        return iq_window(self.round.waveform.baseband, self.round.fs_hz,
                         self._position_s, self.duration_s, window_s)

    def hop_track(self):
        from .features import hop_track_khz
        if self.round is None:
            return None
        return hop_track_khz(self.round.waveform.meta,
                             self.round.spectrum.frame_times_s)
