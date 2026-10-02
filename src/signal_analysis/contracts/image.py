"""时频图坐标与布局契约（``tf_image_v1`` / ``time_frequency_grayscale_v1``）。

训练标签（:func:`band_to_box`）与推理解码（:func:`box_to_band`）共用本模块的
唯一约定：图像行自上而下从 +fs/2 递减，``y_center`` 用"距最高频率"的归一化量
表示。图像构造与频段量测等实现见 :mod:`signal_analysis.algorithms.dsp.image`。
"""

import numpy as np

from .manifest import DEFAULT_IMAGE_SIZE


IMAGE_SIZE = DEFAULT_IMAGE_SIZE
DYNAMIC_RANGE_DB = 60.0
MAX_SIZE = 4096
MIN_SIZE = 64


def _validate_size(size):
    value = int(size)
    if value < MIN_SIZE or value > MAX_SIZE or value & (value - 1):
        raise ValueError(f"图像尺寸应为 {MIN_SIZE}～{MAX_SIZE} 之间的 2 的幂：{size!r}")
    return value


def _validate_dynamic_range(value):
    number = float(value)
    if not np.isfinite(number) or not 10.0 <= number <= 120.0:
        raise ValueError("图像动态范围应在 10～120 dB 之间")
    return number


def band_to_box(meta, f_low_hz, f_high_hz, t_start_s, t_end_s):
    """真值频段/时间 → 归一化边框 ``(x, y, w, h)``（训练标签的唯一约定）。"""
    rate = float(meta["sample_rate_hz"])
    origin = float(meta["t_start_s"])
    span = (float(meta["t_end_s"]) - origin) or 1.0
    low = min(float(f_low_hz), float(f_high_hz))
    high = max(float(f_low_hz), float(f_high_hz))
    left = max(float(t_start_s), origin)
    right = min(float(t_end_s), origin + span)
    if right <= left:
        left, right = origin, origin + span
    center_frequency = 0.5 * (low + high)
    # 图像行自上而下从 +fs/2 递减，因此 y 用「距最高频率」的归一化量表示
    y_center = (rate / 2.0 - center_frequency) / rate
    return (float(np.clip(0.5 * (left + right) - origin, 0.0, span)) / span,
            float(np.clip(y_center, 0.0, 1.0)),
            float(np.clip((right - left) / span, 1e-9, 1.0)),
            float(np.clip((high - low) / rate, 1e-9, 1.0)))


def box_to_band(meta, x_center, y_center, width, height):
    """归一化边框 → 频段/时间区间（解码的唯一约定，与 :func:`band_to_box` 互逆）。"""
    rate = float(meta["sample_rate_hz"])
    span = float(meta["t_end_s"] - meta["t_start_s"]) or 1.0
    half_w = float(np.clip(width, 0.0, 1.0)) / 2.0
    half_h = float(np.clip(height, 0.0, 1.0)) / 2.0
    x_center = float(np.clip(x_center, 0.0, 1.0))
    y_center = float(np.clip(y_center, 0.0, 1.0))
    t_start = (x_center - half_w) * span + float(meta["t_start_s"])
    t_end = (x_center + half_w) * span + float(meta["t_start_s"])
    f_high = rate / 2.0 - (y_center - half_h) * rate
    f_low = rate / 2.0 - (y_center + half_h) * rate
    return {
        "f_low_hz": float(np.clip(f_low, -rate / 2.0, rate / 2.0)),
        "f_high_hz": float(np.clip(f_high, -rate / 2.0, rate / 2.0)),
        "t_start_s": float(np.clip(t_start, float(meta["t_start_s"]), float(meta["t_end_s"]))),
        "t_end_s": float(np.clip(t_end, float(meta["t_start_s"]), float(meta["t_end_s"]))),
    }


def band_bins(frequency, f_low_hz, f_high_hz, resolution_hz=0.0):
    """频段覆盖的频点索引；半分辨率外扩保证边缘频点不被漏掉。"""
    frequency = np.asarray(frequency, dtype=np.float64)
    low = float(f_low_hz) - abs(float(resolution_hz)) / 2.0
    high = float(f_high_hz) + abs(float(resolution_hz)) / 2.0
    return np.flatnonzero((frequency >= low) & (frequency <= high))
