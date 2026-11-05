"""Single-level question catalog for the manual signal-drill page.

There is deliberately no basic/advanced split: every entry below is available
in the first delivered page.  A ``SignalClass`` carries only presentation and
pedagogical metadata; waveform generation is dispatched by ``key`` in
``waveforms.py``.
"""

from dataclasses import dataclass, field
from typing import Mapping

from .patterns import GateSpec, HopSpec


GROUP_ORDER = ("模拟调制", "数字调制", "突发/跳频", "多路/特殊")

AUX_LABELS = {
    "constellation": "星座图",
    "iq": "IQ 相位",
    "timefreq": "时频 / 跳频",
}
AUX_INDEX = {"constellation": 0, "iq": 1, "timefreq": 2}


@dataclass(frozen=True)
class SignalClass:
    key: str
    group: str
    label: str
    short: str
    aux_tab: str
    cues: tuple[str, ...]
    family: str
    constellation_size: int = 0

    def __post_init__(self):
        if self.aux_tab not in AUX_LABELS:
            raise ValueError(f"unknown auxiliary tab: {self.aux_tab}")
        if not self.label or not self.short:
            raise ValueError("signal class labels must not be empty")


@dataclass(frozen=True)
class SignalSpec:
    """Everything needed to reproduce one generated round."""

    key: str
    seed: int
    fs_hz: float
    duration_s: float
    f_if_hz: float
    snr_db: float
    params: Mapping[str, object] = field(default_factory=dict)
    gate: GateSpec | None = None
    hop: HopSpec | None = None


CLASSES = (
    SignalClass(
        "am", "模拟调制", "AM 常规调幅", "AM", "timefreq",
        ("包络随消息周期起伏", "强载波 + 对称边带", "总带宽约为 2×消息带宽"),
        "analog",
    ),
    SignalClass(
        "dsb", "模拟调制", "DSB 双边带", "DSB", "timefreq",
        ("载波被抑制", "包络有过零点", "对称双边带"),
        "analog",
    ),
    SignalClass(
        "ssb", "模拟调制", "SSB 单边带", "SSB", "timefreq",
        ("只有一个边带", "频谱关于 f_IF 不对称", "包络近恒定"),
        "analog",
    ),
    SignalClass(
        "fm", "模拟调制", "FM 调频", "FM", "timefreq",
        ("恒包络", "频谱宽而连续", "瞬时频率连续变化"),
        "analog",
    ),
    SignalClass(
        "ask2", "数字调制", "2ASK 幅移键控", "2ASK", "timefreq",
        ("包络两电平通断", "载波 + 对称边带", "符号率决定边带间隔"),
        "digital",
    ),
    SignalClass(
        "fsk2", "数字调制", "2FSK 频移键控", "2FSK", "timefreq",
        ("恒包络", "累积谱两个主峰", "时频图两条频率台阶"),
        "digital",
    ),
    SignalClass(
        "fsk4", "数字调制", "4FSK 四频键控", "4FSK", "timefreq",
        ("累积谱 4 个主峰", "时频图 4 条频率台阶", "恒包络"),
        "digital",
    ),
    SignalClass(
        "fsk8", "数字调制", "8FSK 八频键控", "8FSK", "timefreq",
        ("累积谱 8 个主峰", "时频图 8 条频率台阶", "恒包络"),
        "digital",
    ),
    SignalClass(
        "bpsk", "数字调制", "BPSK 二相调相", "BPSK", "constellation",
        ("星座 2 簇", "Q 路近似为 0", "相位 180° 跳变"),
        "digital", constellation_size=2,
    ),
    SignalClass(
        "qpsk", "数字调制", "QPSK 四相调相", "QPSK", "constellation",
        ("星座 4 簇", "I/Q 两路均有跳变", "恒包络"),
        "digital", constellation_size=4,
    ),
    SignalClass(
        "psk8", "数字调制", "8PSK 八相调相", "8PSK", "constellation",
        ("星座 8 簇", "相位间隔 45°", "恒包络"),
        "digital", constellation_size=8,
    ),
    SignalClass(
        "qam16", "数字调制", "16QAM 正交幅度", "16QAM", "constellation",
        ("星座 16 簇（4×4）", "包络多电平", "载波抑制"),
        "digital", constellation_size=16,
    ),
    SignalClass(
        "qam64", "数字调制", "64QAM 正交幅度", "64QAM", "constellation",
        ("星座 64 簇（8×8）", "包络多电平", "载波抑制"),
        "digital", constellation_size=64,
    ),
    SignalClass(
        "fsk2_burst", "突发/跳频", "2FSK 突发", "2FSK突发", "timefreq",
        ("时域门控包络", "双音累积谱", "时频图断续"),
        "burst",
    ),
    SignalClass(
        "fh_fsk2", "突发/跳频", "跳频 2FSK", "跳频2FSK", "timefreq",
        ("中心频率跳变", "累积谱多频带", "时频图台阶"),
        "burst",
    ),
    SignalClass(
        "link11_single", "突发/跳频", "单音 LINK11（教学简化）", "单音LINK11",
        "timefreq",
        ("单音窄谱线", "1364 Bd 突发图案", "时频图短促线段"),
        "burst",
    ),
    SignalClass(
        "audio_multichannel", "多路/特殊", "多路音频（FDM）", "多路音频",
        "timefreq",
        ("多个音频子带", "时频图多条平行带", "时域合成包络"),
        "special",
    ),
    SignalClass(
        "cw", "多路/特殊", "单音 / 等幅报", "单音/等幅报", "timefreq",
        ("单根谱线", "连续载波或低速通断", "无宽带调制"),
        "special",
    ),
    SignalClass(
        "multitone", "多路/特殊", "多音", "多音", "timefreq",
        ("多根离散谱线", "时域拍频包络", "无载波抑制特征"),
        "special",
    ),
    SignalClass(
        "noise", "多路/特殊", "噪声", "噪声", "timefreq",
        ("频谱平坦", "无稳定谱线", "无周期结构"),
        "special",
    ),
)

CLASS_BY_KEY = {item.key: item for item in CLASSES}
KEYS = tuple(item.key for item in CLASSES)


def classes_for_group(group):
    return tuple(item for item in CLASSES if item.group == group)


def grouped_classes():
    """Return ``[(group, classes), ...]`` in the catalog's display order."""
    return [(group, classes_for_group(group)) for group in GROUP_ORDER]
