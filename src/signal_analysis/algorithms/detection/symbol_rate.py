"""传统符号率估计：包络谱与包络微分谱的线谱峰位。

对检测得到的单个目标频带做混频、低通抽取，再在占用带内寻找
``|x|²``、``diff(|x|²)`` 与 ``diff(|x|)`` 三条谱的共同线谱。符号率
候选必须同时满足带宽合理性与多特征一致性，模拟信号/跳频会话/低信噪
比目标会显式标为“不适用”，不猜一个数字。

原因码：``hopping``（跳频会话，由服务层直接写）、``invalid_band``、
``insufficient_samples``、``invalid_samples``、``zero_power``、
``no_peak``、``bandwidth_mismatch``、``low_confidence``。

仅依赖标准库、NumPy 与共享前处理，可由传统与 AI 检测服务共同调用。
"""

import math

import numpy as np

from ..dsp.preprocess import _mix_and_decimate


SYMBOL_RATE_METHOD = "envelope_spectrum_v1"

#: 单目标参与估计的最大采样点数。包络谱只需要数千个符号，限制长度控制耗时。
MAX_ESTIMATE_SAMPLES = 131_072

#: 候选峰相对局部基线的提升门限（dB）。
MIN_PEAK_PROMINENCE_DB = 4.0

#: 判定“可用估计”的最小提升与特征一致数。
APPLICABLE_PROMINENCE_DB = 20.0
APPLICABLE_CONSISTENCY = 2

#: 线性数字调制（RRC/矩形成形）的符号率通常落在占用带宽的该区间内；
#: 检测带宽受噪声/边带影响时允许更宽的兜底区间，但会降低置信度。
_PLAUSIBLE_RATIO = (0.45, 1.05)
_FALLBACK_RATIO = (0.35, 1.15)
_SEARCH_RATIO = (0.30, 1.20)

_MAX_PEAKS_PER_FEATURE = 12


def _rounded(value, digits=3):
    return None if value is None else round(float(value), digits)


def _unavailable(reason):
    """统一的“没有可用符号率”结果；字段齐全，便于直接写进检测结果。"""
    return {
        "symbol_rate_baud": None,
        "symbol_rate_confidence": 0.0,
        "symbol_rate_method": SYMBOL_RATE_METHOD,
        "symbol_rate_applicable": False,
        "symbol_rate_reason": reason,
    }


def _feature_peaks(values, work_rate, bandwidth, nfft):
    """一条实数包络特征的线谱峰。

    参数：values 为实数特征；work_rate 为分析率 Hz；bandwidth 为占用带宽 Hz；
    nfft 为统一 FFT 点数。

    返回：(peaks, df, low, high)：peaks 为 ``(频率 Hz, 相对局部基线 dB)``，
    按提升降序最多保留 12 个；``df`` 为频率分辨率；``low/high`` 为搜索范围。
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size < 64:
        return [], 0.0, 0.0, 0.0
    data = data - float(np.mean(data))
    if not np.isfinite(data).all():
        return [], 0.0, 0.0, 0.0
    window = np.hanning(data.size)
    spectrum = np.abs(np.fft.rfft(data * window, nfft)) ** 2
    frequencies = np.fft.rfftfreq(nfft, 1.0 / work_rate)
    df = float(work_rate / nfft)
    low = max(2.0 * df, _SEARCH_RATIO[0] * bandwidth)
    high = min(0.5 * work_rate, _SEARCH_RATIO[1] * bandwidth)
    if high <= low:
        return [], df, low, high
    indices = np.flatnonzero((frequencies >= low) & (frequencies <= high))
    if indices.size < 7:
        return [], df, low, high
    segment = spectrum[indices]
    epsilon = np.finfo(np.float64).tiny
    # 分块低分位基线：窄线谱只占 1～2 个频点，不会被 16 点块的中位数/低分位
    # 抬高；宽带数据谱会把整块基线抬起来，从而压低“假线谱”的提升量。
    block = 16
    count = segment.size // block
    if count >= 3:
        blocks = segment[:count * block].reshape(count, block)
        medians = np.percentile(blocks, 25.0, axis=1)
        centers = (np.arange(count, dtype=np.float64) + 0.5) * block
        baseline = np.interp(np.arange(segment.size, dtype=np.float64),
                             centers, medians)
    else:
        baseline = np.full(segment.size, float(np.median(segment)))
    prominence = 10.0 * np.log10(np.maximum(segment, epsilon)
                                 / np.maximum(baseline, epsilon))
    # 忽略两端各两个频点，避免卷积边沿造成的假峰
    center = prominence[2:-2]
    is_peak = ((center >= prominence[1:-3]) & (center >= prominence[3:-1])
               & (center > prominence[:-4]) & (center >= prominence[4:]))
    locations = np.flatnonzero(is_peak) + 2
    locations = locations[np.isfinite(prominence[locations])]
    if locations.size == 0:
        return [], df, low, high
    # 先按块基线提升取前 48 个候选，再用“挖掉峰肩”的局部中位数复算
    # 提升量：窄线谱仍很高，而模拟消息/宽带数据谱会被压回去。
    order = np.argsort(-prominence[locations])[:48]
    peaks = []
    for index in locations[order]:
        if index <= 0 or index >= segment.size - 1:
            continue
        left = segment[max(0, index - 17):max(0, index - 2)]
        right = segment[min(segment.size, index + 3):min(segment.size, index + 18)]
        neighbourhood = np.concatenate((left, right))
        local = (float(np.median(neighbourhood)) if neighbourhood.size
                 else float(baseline[index]))
        local = max(local, epsilon)
        sharpness = 10.0 * math.log10(max(float(segment[index]), epsilon) / local)
        if sharpness < MIN_PEAK_PROMINENCE_DB:
            continue
        left_log = math.log(max(float(segment[index - 1]), epsilon))
        middle_log = math.log(max(float(segment[index]), epsilon))
        right_log = math.log(max(float(segment[index + 1]), epsilon))
        curvature = left_log - 2.0 * middle_log + right_log
        offset = 0.5 * (left_log - right_log) / curvature if curvature != 0.0 else 0.0
        offset = float(np.clip(offset, -0.5, 0.5))
        peaks.append((float(low + (index + offset) * df), float(sharpness)))
    peaks.sort(key=lambda item: -item[1])
    return peaks[:_MAX_PEAKS_PER_FEATURE], df, low, high


def _choose_candidate(peaks_by_feature, bandwidth, df):
    """在三条特征的候选峰中选最可信的符号率。

    评分 = 最大提升 + 4 dB × 特征一致数 + 带宽合理性先验。一致性指同频
    附近（2 个频点或 1%）至少 6 dB 的提升；先验对 ``Rs/B`` 落在常见区间
    （0.45～1.05）的候选不加罚，落在兜底区间少量扣分，明显越界则重罚。
    """
    entries = [frequency for peaks in peaks_by_feature.values()
               for frequency, prominence in peaks
               if prominence >= MIN_PEAK_PROMINENCE_DB]
    if not entries:
        return None
    candidates = sorted({round(float(frequency), 6) for frequency in entries})
    best = None
    for frequency in candidates:
        ratio = frequency / bandwidth
        if not _SEARCH_RATIO[0] <= ratio <= _SEARCH_RATIO[1]:
            continue
        scores = []
        for peaks in peaks_by_feature.values():
            tolerance = max(2.0 * df, 0.01 * frequency)
            nearby = [prominence for peak, prominence in peaks
                      if abs(peak - frequency) <= tolerance]
            scores.append(max(nearby, default=0.0))
        consistency = sum(1 for score in scores if score >= 6.0)
        best_prominence = max(scores)
        if _PLAUSIBLE_RATIO[0] <= ratio <= _PLAUSIBLE_RATIO[1]:
            prior = 0.0
        elif _FALLBACK_RATIO[0] <= ratio <= _FALLBACK_RATIO[1]:
            prior = -3.0
        else:
            prior = -8.0
        score = best_prominence + 4.0 * consistency + prior
        # 分数相同时优先较低频率，避免把二次谐波当基频
        key = (score, -frequency)
        if best is None or key > best[0]:
            best = (key, float(frequency), float(best_prominence),
                    int(consistency), float(ratio))
    return best


def estimate_symbol_rate(samples, sample_rate, center_hz, bandwidth_hz, *,
                         start_s=None, end_s=None,
                         max_samples=MAX_ESTIMATE_SAMPLES):
    """估计单目标频带的符号率（线性数字调制的包络谱线谱）。

    参数：samples 为一维 IQ；sample_rate 为 Hz；center_hz 为基带频偏 Hz；
    bandwidth_hz 为占用带宽 Hz；start_s/end_s 为可选时间范围；max_samples
    为参与估计的最大采样点数。

    返回：``symbol_rate_baud``、``symbol_rate_confidence``、
    ``symbol_rate_method``、``symbol_rate_applicable``、
    ``symbol_rate_reason`` 五个字段。无峰、样本不足或带宽非法时数值为
    ``None``；有候选但低于可用门限时保留数值并标 ``applicable=False``。
    """
    rate = float(sample_rate)
    if not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("采样率必须为有限正数，单位 Hz")
    try:
        center = float(center_hz or 0.0)
        bandwidth = float(bandwidth_hz)
    except (TypeError, ValueError):
        return _unavailable("invalid_band")
    if not np.isfinite(center) or not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return _unavailable("invalid_band")
    if bandwidth > rate:
        return _unavailable("invalid_band")
    limit = int(max_samples)
    if limit < 64:
        return _unavailable("insufficient_samples")
    data = np.asarray(samples)
    if data.ndim != 1 or data.size < 64:
        return _unavailable("insufficient_samples")
    start = 0
    if start_s is not None:
        start = int(round(max(0.0, float(start_s)) * rate))
    stop = data.size
    if end_s is not None:
        stop = int(round(min(float(end_s), data.size / rate) * rate))
    start = max(0, min(start, data.size))
    stop = max(start, min(stop, data.size))
    segment = data[start:stop]
    if segment.size < 64:
        return _unavailable("insufficient_samples")
    if segment.size > limit:
        offset = (segment.size - limit) // 2
        segment = segment[offset:offset + limit]
    if not np.isfinite(segment).all():
        return _unavailable("invalid_samples")
    segment = np.asarray(segment, dtype=np.complex128)
    if float(np.mean(np.abs(segment) ** 2)) <= 0.0:
        return _unavailable("zero_power")
    try:
        work, work_rate, _ = _mix_and_decimate(segment, rate, center, bandwidth)
    except ValueError:
        return _unavailable("insufficient_samples")
    if work.size < 64:
        return _unavailable("insufficient_samples")

    magnitude = np.abs(work)
    features = (magnitude ** 2, np.diff(magnitude ** 2), np.diff(magnitude))
    max_length = max(feature.size for feature in features)
    nfft = 1 << int(math.ceil(math.log2(max(2 * max_length, 1024))))
    # 限制到 2^19，避免异常长记录把单次估计拖成秒级
    nfft = min(nfft, 1 << 19)
    peaks_by_feature = {}
    resolution = work_rate / nfft
    for index, feature in enumerate(features):
        peaks, df, _, _ = _feature_peaks(feature, work_rate, bandwidth, nfft)
        if peaks:
            peaks_by_feature[str(index)] = peaks
            resolution = df
    if not peaks_by_feature:
        return _unavailable("no_peak")
    best = _choose_candidate(peaks_by_feature, bandwidth, resolution)
    if best is None:
        return _unavailable("bandwidth_mismatch")
    _, frequency, prominence, consistency, ratio = best
    confidence = float(np.clip((prominence - 8.0) / 24.0, 0.0, 1.0))
    confidence = min(1.0, confidence * (0.7 + 0.1 * consistency))
    applicable = (prominence >= APPLICABLE_PROMINENCE_DB
                  and consistency >= APPLICABLE_CONSISTENCY
                  and _FALLBACK_RATIO[0] <= ratio <= _FALLBACK_RATIO[1])
    if applicable:
        reason = None
    elif not _FALLBACK_RATIO[0] <= ratio <= _FALLBACK_RATIO[1]:
        reason = "bandwidth_mismatch"
    else:
        reason = "low_confidence"
    return {
        "symbol_rate_baud": _rounded(frequency),
        "symbol_rate_confidence": _rounded(confidence),
        "symbol_rate_method": SYMBOL_RATE_METHOD,
        "symbol_rate_applicable": bool(applicable),
        "symbol_rate_reason": reason,
    }
