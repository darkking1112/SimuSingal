"""时频图张量：图像构造、坐标映射与带内辐射量测量。

本模块把"神经网络的输入/输出"和"项目既有的物理量口径"钉在一起：

* 图像由 :func:`detect_signals` 的同一份 STFT 生成，因此 AI 路径与传统
  能量路径给出的 ``power_dbfs``、``snr_db``（``inband_snr_v1``）可逐项对比。
* 图像行 = 频率偏移，**自上而下从 +fs/2 递减到 -fs/2**（与 ``imshow``
  默认的 ``origin='upper'`` 一致）；列 = 时间，自左向右递增。
* 归一化只用"本底 + 动态范围"，与绝对增益无关：
  ``image = clip((db - 本底) / 动态范围, 0, 1)``。
* 边框坐标是**像素边界**而非像素中心：``x = 0`` 对应 ``-fs/2`` 与 ``t = 0``，
  ``x = 1`` 对应 ``+fs/2`` 与 ``t = duration``。训练脚本必须使用
  :func:`band_to_box` 生成标签，推理解码使用 :func:`box_to_band`，两者
  严格互逆。
"""

from __future__ import annotations

import numpy as np

from .._numeric import _SNR_FLOOR_DB, detect_signals
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


def spectral_context(samples, sample_rate, config=None):
    """与检测一致的时频测量上下文：``(summary, arrays)``。

    直接复用 :func:`detect_signals`，因此 AI 路径的频谱、本底、帧时刻与
    ``power_dbfs``/``snr_db`` 口径与传统路径完全相同（便于同图对比）。
    """
    return detect_signals(samples, sample_rate, config)


def _interp_weights(source, target):
    """线性插值权重矩阵，每行和为 1（越界点吸附到最近端点）。"""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.size < 2:
        return np.ones((target.size, max(source.size, 1)))
    position = np.clip(np.searchsorted(source, target), 1, source.size - 1)
    left = source[position - 1]
    right = source[position]
    span = np.where(right > left, right - left, 1.0)
    upper = np.clip((target - left) / span, 0.0, 1.0)
    matrix = np.zeros((target.size, source.size))
    rows = np.arange(target.size)
    matrix[rows, position - 1] = 1.0 - upper
    matrix[rows, position] = upper
    return matrix


def detection_image(arrays, summary, size=IMAGE_SIZE, dynamic_range_db=DYNAMIC_RANGE_DB):
    """把 STFT 幅度谱重采样成单通道灰度时频图，返回 ``(image, meta)``。

    ``image`` 为 ``float32`` 且取值固定在 ``[0, 1]``，可直接作为网络输入；
    ``meta`` 记录坐标映射与归一化参数，供 :func:`box_to_band`、
    :func:`image_db` 使用（推理与训练必须用同一份 ``meta`` 约定）。
    """
    size = _validate_size(size)
    dynamic_range = _validate_dynamic_range(dynamic_range_db)
    spectrogram_db = np.asarray(arrays["spectrogram_db"], dtype=np.float64)
    frequency = np.asarray(arrays["frequency"], dtype=np.float64)
    frame_time = np.asarray(arrays["frame_time"], dtype=np.float64)
    if spectrogram_db.ndim != 2 or spectrogram_db.shape[1] != frequency.size:
        raise ValueError("时频图数组形状与频率轴不一致")
    if spectrogram_db.shape[0] != frame_time.size:
        raise ValueError("时频图数组形状与时间轴不一致")
    rate = float(summary["sample_rate_hz"])
    duration = float(summary["duration_s"])
    grid_frequency = -rate / 2.0 + (np.arange(size) + 0.5) * (rate / size)
    span = duration if duration > 0 else 1.0
    grid_time = (np.arange(size) + 0.5) * (span / size)
    resampled = _interp_weights(frame_time, grid_time) @ spectrogram_db
    # 第二个重采样矩阵把频点轴搬到行：得到 (频率, 时间) 的图像布局
    resampled = _interp_weights(frequency, grid_frequency) @ resampled.T
    # 行 0 为最高频率：与图像坐标系（y 向下）一致
    flipped = resampled[::-1]
    noise_floor_db = float(np.asarray(arrays["noise_floor_db"]).reshape(-1)[0])
    normalized = np.clip((flipped - noise_floor_db) / dynamic_range, 0.0, 1.0)
    meta = {
        "layout": "time_frequency_grayscale_v1",
        "size": int(size),
        "sample_rate_hz": rate,
        "duration_s": duration,
        "f_low_hz": -rate / 2.0,
        "f_high_hz": rate / 2.0,
        "t_start_s": 0.0,
        "t_end_s": duration,
        "noise_floor_dbfs_per_hz": noise_floor_db,
        "dynamic_range_db": dynamic_range,
        "db_floor": noise_floor_db,
        "db_ceiling": noise_floor_db + dynamic_range,
        "nfft": int(summary["nfft"]),
        "freq_resolution_hz": float(summary["freq_resolution_hz"]),
        "frame_count": int(np.asarray(arrays["frame_time"]).size),
    }
    return normalized.astype(np.float32), meta


def image_db(image, meta):
    """反归一化：灰度图 → dB（本底参考）数组，便于人工核对与报表绘图。"""
    array = np.asarray(image, dtype=np.float64) * float(meta["dynamic_range_db"])
    return array + float(meta["db_floor"])


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


def measure_band(arrays, summary, band, threshold_db=None):
    """按 ``inband_snr_v1`` 口径测量一个频段的功率与带内信噪比。

    带内功率取该频段所有帧的平均（时长内平均功率），噪声参考为
    ``N0·B``（同带宽白噪声），与能量检测器使用的公式和常数完全一致。
    """
    frequency = np.asarray(arrays["frequency"], dtype=np.float64)
    frame_time = np.asarray(arrays["frame_time"], dtype=np.float64)
    resolution = float(summary["freq_resolution_hz"])
    noise_floor_db = float(np.asarray(arrays["noise_floor_db"]).reshape(-1)[0])
    noise_linear = 10.0 ** (noise_floor_db / 10.0)
    bins = band_bins(frequency, band["f_low_hz"], band["f_high_hz"], resolution)
    if bins.size == 0:
        nearest = int(np.argmin(np.abs(frequency - 0.5 * (band["f_low_hz"] + band["f_high_hz"]))))
        bins = np.array([nearest], dtype=np.int64)
    frames = np.flatnonzero((frame_time >= min(band["t_start_s"], band["t_end_s"]))
                            & (frame_time <= max(band["t_start_s"], band["t_end_s"])))
    if frames.size == 0:
        frames = np.array([int(np.argmin(np.abs(frame_time - 0.5 * (band["t_start_s"]
                                                                   + band["t_end_s"]))))],
                          dtype=np.int64)
    spectrogram_db = np.asarray(arrays["spectrogram_db"], dtype=np.float64)
    psd = 10.0 ** (spectrogram_db / 10.0)
    frame_power = psd[:, bins].sum(axis=1) * resolution
    band_power = float(frame_power[frames].mean())
    bandwidth = max(float(band["f_high_hz"]) - float(band["f_low_hz"]), 0.0)
    noise_band = max(noise_linear * bandwidth, 1e-30)
    signal_power = max(band_power - noise_band, 1e-30)
    band_frequencies = frequency[bins]
    band_weights = psd[frames][:, bins].sum(axis=0)
    weight_sum = float(band_weights.sum())
    centroid = (float((band_weights * band_frequencies).sum() / weight_sum)
                if weight_sum > 0 else float(band_frequencies.mean()))
    gate = threshold_db if threshold_db is not None else float(summary["config"]["threshold_db"])
    active = frame_power > noise_linear * bandwidth * 10.0 ** (float(gate) / 10.0)
    if not active.any():
        active = np.zeros(frame_power.size, dtype=bool)
        active[frames] = True
    half_frame = (float(summary["duration_s"]) / max(frame_time.size, 1)) / 2.0
    intervals = [(float(max(frame_time[index] - half_frame, 0.0)),
                  float(min(frame_time[index] + half_frame, float(summary["duration_s"]))))
                 for index in np.flatnonzero(active)]
    return {
        "power_linear": band_power,
        "power_dbfs": float(10.0 * np.log10(max(band_power, 1e-30))),
        "snr_db": float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB)),
        "centroid_hz": centroid,
        "intervals": intervals,
        "active_seconds": float(sum(end - start for start, end in intervals)),
        "bin_count": int(bins.size),
    }
