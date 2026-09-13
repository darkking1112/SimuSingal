"""传统能量检测：双门限频带量测、时间支撑与会话合并。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import numpy as np

from ._numeric_common import (
    DETECT_SNR_DEFINITION,
    _SESSION_GAP_RATIO,
    _SNR_FLOOR_DB,
    _binary_close,
    _finite,
    _noise_floor_db,
    _occupied_span,
    _stft_psd,
    _true_runs,
    validate_rate,
    validate_samples,
)


DETECT_ALGORITHM = "energy_detect_v1"


_DETECT_DEFAULTS = {
    "nfft": 512,
    # 平均 PSD 的逐点起伏远小于 0.5 dB（512 帧平均），故 3 dB 门限已很保守
    "threshold_db": 3.0,
    "band_threshold_db": 0.0,
    "min_bandwidth_hz": 0.0,
    "min_duration_s": 0.0,
    "max_detections": 32,
    "merge_bins": 0,
}


def _detect_config(rate, duration, config):
    """校验并展开会话级能量检测配置。

    参数：rate 为 Hz；duration 为秒；config 为配置字典或 None。

    返回：有效配置字典，含 nfft、门限、频率分辨率、最小频带点数等派生项。

    算法与边界：拒绝未知键及非法范围；默认 nfft=512、检出门限 3 dB，测带门限缺省时取 max(1,threshold/2)，最小带宽 3 个频点。零
    merge_bins 表示自动半径；整数项拒绝布尔值。校验顺序及 ValueError 消息保持原契约。
    """
    if config is None:
        settings = {}
    elif isinstance(config, dict):
        settings = dict(config)
    else:
        raise ValueError("检测配置必须是字典")
    unknown = set(settings) - set(_DETECT_DEFAULTS)
    if unknown:
        raise ValueError(f"未知的检测参数：{', '.join(sorted(unknown))}")
    nfft = settings.get("nfft", _DETECT_DEFAULTS["nfft"])
    if isinstance(nfft, bool) or int(nfft) != nfft or not 16 <= int(nfft) <= 4096:
        raise ValueError("FFT 点数必须为 16～4096 的整数")
    nfft = int(nfft)
    resolution = rate / nfft
    threshold_db = _finite(settings.get("threshold_db", _DETECT_DEFAULTS["threshold_db"]),
                           "检测门限", 0.0, 80.0)
    band_threshold_db = _finite(settings.get("band_threshold_db", 0.0),
                                "带宽测量门限", 0.0, 80.0)
    if band_threshold_db <= 0:
        # “高门限检测、低门限量带宽”：占用带宽按噪声底以上 1.5 dB（默认）
        # 测量，才能看到 AM 这种载波主导信号被压低的边带
        band_threshold_db = max(1.0, 0.5 * threshold_db)
    if band_threshold_db > threshold_db:
        raise ValueError("带宽测量门限不得高于检测门限")
    min_bandwidth = _finite(settings.get("min_bandwidth_hz", 0.0), "最小带宽", 0.0, rate)
    if min_bandwidth <= 0:
        # 默认要求至少 3 个频点宽，抑制单点毛刺
        min_bandwidth = 3.0 * resolution
    min_duration = _finite(settings.get("min_duration_s", 0.0), "最短持续时间", 0.0, duration)
    max_detections = settings.get("max_detections", _DETECT_DEFAULTS["max_detections"])
    if isinstance(max_detections, bool) or int(max_detections) != max_detections \
            or not 1 <= int(max_detections) <= 256:
        raise ValueError("最大目标数必须为 1～256 的整数")
    merge_bins = settings.get("merge_bins", 0)
    if isinstance(merge_bins, bool) or int(merge_bins) != merge_bins or not 0 <= int(merge_bins) <= nfft // 4:
        raise ValueError(f"谱合并宽度必须为 0～{nfft // 4} 的整数")
    merge_bins = int(merge_bins) or max(2, nfft // 128)
    return {
        "nfft": nfft,
        "threshold_db": threshold_db,
        "band_threshold_db": band_threshold_db,
        "min_bandwidth_hz": min_bandwidth,
        "min_duration_s": min_duration,
        "max_detections": int(max_detections),
        "merge_bins": merge_bins,
        "freq_resolution_hz": resolution,
        "min_bandwidth_bins": max(1, int(np.ceil(min_bandwidth / resolution))),
    }


_SESSION_OVERLAP_RATIO = 0.2


def _band_regions(detect_mask, edge_mask):
    """将高门限检出游程归入低门限频带。

    参数：detect_mask 为高门限一维掩膜；edge_mask 为同长度低门限掩膜。

    返回：(first,last,inner) 列表：低门限频带的包含端点索引及其内部高门限游程列表。

    算法与边界：逐低门限连通区筛选完全包含的高门限游程，仅保留有检出证据的区间；一个辐射源内部凹口产生的多段检出只形成一个频带。
    """
    regions = []
    for first, last in _true_runs(edge_mask):
        inner = [(a, b) for a, b in _true_runs(detect_mask)
                 if first <= a and b <= last]
        if inner:
            regions.append((first, last, inner))
    return regions


def _overlap_seconds(first, second):
    """计算两组时间区间之间的总重叠时长。

    参数：first、second 为各自内部互不重叠的 (start,end) 列表，单位秒。

    返回：非负 float 秒数。

    算法与边界：逐对累加 max(0,min(end)-max(start))；空列表返回 0。内部不合并重叠区间，调用方负责区间集合符合约定。
    """
    total = 0.0
    for a_start, a_end in first:
        for b_start, b_end in second:
            total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return total


def _same_session(first, second, gap_ratio=_SESSION_GAP_RATIO,
                  overlap_ratio=_SESSION_OVERLAP_RATIO):
    """判断两个检出频带是否属于同一跳频会话。

    参数：first、second 为候选字典，含 Hz 频带边界、秒区间和活动时长；gap_ratio、overlap_ratio 为无量纲门限。

    返回：布尔值。

    算法与边界：频带间距不得大于较宽频带宽度的 gap_ratio 倍，同时重叠时长不得超过较短活动时长的 overlap_ratio
    倍。每次从边界重算宽度，避免合并后的缓存宽度失效。
    """
    gap = max(first["f_low_hz"] - second["f_high_hz"],
              second["f_low_hz"] - first["f_high_hz"])
    first_width = first["f_high_hz"] - first["f_low_hz"]
    second_width = second["f_high_hz"] - second["f_low_hz"]
    if gap > gap_ratio * max(first_width, second_width):
        return False
    overlap = _overlap_seconds(first["intervals"], second["intervals"])
    shorter = min(first["active_seconds"], second["active_seconds"])
    return overlap <= overlap_ratio * shorter


def _combine_sessions(left, right):
    """合并两个候选目标的频带、时间与功率。

    参数：left、right 为包含频带 Hz、功率、时间区间秒及追溯统计的候选字典。

    返回：新的合并字典；不修改两个输入字典。

    算法与边界：功率相加、质心按功率加权，边界取并集，区间排序并累加活动时长、频点数和子带数；功率非正时质心取均值。区间非空由上层保证。
    """
    power = left["power_linear"] + right["power_linear"]
    centroid = ((left["power_linear"] * left["centroid_hz"]
                 + right["power_linear"] * right["centroid_hz"]) / power
                if power > 0 else 0.5 * (left["centroid_hz"] + right["centroid_hz"]))
    intervals = sorted(left["intervals"] + right["intervals"])
    active_seconds = sum(end - start for start, end in intervals)
    f_low = min(left["f_low_hz"], right["f_low_hz"])
    f_high = max(left["f_high_hz"], right["f_high_hz"])
    return {
        "center_hz": 0.5 * (f_low + f_high),
        "centroid_hz": float(centroid),
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "power_linear": float(power),
        "intervals": intervals,
        "active_seconds": float(active_seconds),
        "t_start_s": intervals[0][0],
        "t_end_s": intervals[-1][1],
        "bin_count": left["bin_count"] + right["bin_count"],
        "sub_bands": left["sub_bands"] + right["sub_bands"],
        "occupied_f_low_hz": min(left["occupied_f_low_hz"], right["occupied_f_low_hz"]),
        "occupied_f_high_hz": max(left["occupied_f_high_hz"], right["occupied_f_high_hz"]),
    }


def _merge_sessions(items, max_detections=256):
    """将满足判据的跳频子带合为会话级检出。

    参数：items 为候选字典列表；max_detections 为保留数量上限，默认 256。

    返回：按线性功率降序排列并截断的新候选列表。

    算法与边界：浅拷贝每个候选，按既定遍历顺序反复应用 _same_session 和
    _combine_sessions，直到不能合并；空输入返回空列表，不改输入字典。AI 会话量测也复用此算法。
    """
    groups = [dict(item) for item in items]
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            for j in range(len(groups) - 1, i, -1):
                if not _same_session(groups[i], groups[j]):
                    continue
                groups[i] = _combine_sessions(groups[i], groups[j])
                groups.pop(j)
                changed = True
                break
            if changed:
                break
    groups.sort(key=lambda item: -item["power_linear"])
    return groups[:max_detections]


def detect_signals(samples, sample_rate, config=None):
    """进行会话级能量检测与频带／时间／功率量测。

    参数：samples 为一维 IQ，sample_rate 为 Hz，config 为检测配置字典或 None。

    返回：(summary, arrays)：detect_result_v1 摘要以及频谱、时频图、阈值和目标框数组；频率为基带偏移 Hz，时间为秒，功率为 dBFS。

    算法与边界：共用 STFT 后对时间取平均，MAD 估本底、高门限检出、低门限量带、闭运算与最小宽度过滤，再估计时间支撑并归并会话。带内 SNR
    按扣噪后信号功率/(N0·B) 定义；99% 功率区间另作追溯项。跳频会话按一条记录输出，不把会话数视作跳数；无目标返回空结果，非法输入／配置报错。见传统能量检测设计。
    """
    x = validate_samples(samples)
    rate = validate_rate(sample_rate)
    duration = float(x.size / rate)
    resolved = _detect_config(rate, duration, config)
    nfft = resolved["nfft"]
    resolution = resolved["freq_resolution_hz"]
    threshold_db = resolved["threshold_db"]

    psd, frequencies, starts, hop = _stft_psd(x, rate, nfft)
    psd_average = psd.mean(axis=0)
    psd_db = 10.0 * np.log10(np.maximum(psd_average, 1e-30))
    noise_floor_db = _noise_floor_db(psd_db, threshold_db)
    threshold_dbfs = noise_floor_db + threshold_db
    noise_linear = 10.0 ** (noise_floor_db / 10.0)

    mask = _binary_close(psd_db > threshold_dbfs, resolved["merge_bins"])
    runs = [(start, stop) for start, stop in _true_runs(mask)
            if stop - start + 1 >= resolved["min_bandwidth_bins"]]
    detect_mask = np.zeros(psd_db.size, dtype=bool)
    for start, stop in runs:
        detect_mask[start:stop + 1] = True
    # 带宽测量门限（低于检测门限）：检出用高门限，量带宽用低门限
    if resolved["band_threshold_db"] < threshold_db:
        edge_mask = _binary_close(
            psd_db > noise_floor_db + resolved["band_threshold_db"], resolved["merge_bins"])
    else:
        edge_mask = mask
    regions = _band_regions(detect_mask, edge_mask)

    frame_starts = np.clip(starts / rate, 0.0, duration)
    frame_ends = np.clip((starts + nfft) / rate, 0.0, duration)
    candidates = []
    for first, last, inner in regions:
        occupancy_low, occupancy_high = _occupied_span(psd_average, first, last)
        weights = np.where(detect_mask[first:last + 1], psd_average[first:last + 1], 0.0)
        weight_sum = float(psd_average[first:last + 1].sum())
        if weight_sum <= 0:
            continue
        signal_sum = float(weights.sum())
        band_frequencies = frequencies[first:last + 1]
        centroid = (float((weights * band_frequencies).sum() / signal_sum)
                    if signal_sum > 0 else float(band_frequencies.mean()))
        f_low = float(band_frequencies[0] - resolution / 2.0)
        f_high = float(band_frequencies[-1] + resolution / 2.0)
        bandwidth = f_high - f_low
        band_power = float(weight_sum * resolution)
        # 时间支撑：按帧的带内功率与噪声参考功率比较
        frame_power = np.asarray(psd[:, first:last + 1].sum(axis=1), dtype=np.float64) * resolution
        active = frame_power > noise_linear * bandwidth * 10.0 ** (threshold_db / 10.0)
        if not active.any():
            active = np.ones(frame_power.size, dtype=bool)
        active_index = np.flatnonzero(active)
        t_start = float(frame_starts[active_index[0]])
        t_end = float(frame_ends[active_index[-1]])
        if t_end - t_start < resolved["min_duration_s"]:
            continue
        intervals = [(float(frame_starts[position]), float(frame_ends[position]))
                     for position in active_index]
        candidates.append({
            "center_hz": 0.5 * (f_low + f_high),
            "centroid_hz": centroid,
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "power_linear": band_power,
            "intervals": intervals,
            "active_seconds": float(t_end - t_start),
            "t_start_s": t_start,
            "t_end_s": t_end,
            "bin_count": int(sum(stop - start + 1 for start, stop in inner)),
            "sub_bands": 1,
            "occupied_f_low_hz": float(frequencies[occupancy_low] - resolution / 2.0),
            "occupied_f_high_hz": float(frequencies[occupancy_high] + resolution / 2.0),
        })

    groups = _merge_sessions(candidates, resolved["max_detections"])
    groups.sort(key=lambda item: item["center_hz"])
    detections = []
    for index, item in enumerate(groups, start=1):
        bandwidth = item["f_high_hz"] - item["f_low_hz"]
        # 噪声参考功率：同带宽白噪声功率 N0·B（inband_snr_v1 口径）
        noise_band = max(noise_linear * bandwidth, 1e-30)
        signal_power = max(item["power_linear"] - noise_band, 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        hopping = item["sub_bands"] > 1
        detections.append({
            "id": index,
            "method": "energy",
            "center_hz": round(item["center_hz"], 3),
            "centroid_hz": round(item["centroid_hz"], 3),
            "bandwidth_hz": round(bandwidth, 3),
            "f_low_hz": round(item["f_low_hz"], 3),
            "f_high_hz": round(item["f_high_hz"], 3),
            "t_start_s": round(item["t_start_s"], 6),
            "t_end_s": round(item["t_end_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(item["power_linear"], 1e-30))), 3),
            "snr_db": round(snr_db, 3),
            "session_id": index if hopping else None,
            "confidence": round(float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0)), 3),
            "hopping": hopping,
            "sub_bands": int(item["sub_bands"]),
            "bin_count": int(item["bin_count"]),
            "occupied_f_low_hz": round(item["occupied_f_low_hz"], 3),
            "occupied_f_high_hz": round(item["occupied_f_high_hz"], 3),
        })

    summary = {
        "contract": "detect_result_v1",
        "algorithm": DETECT_ALGORITHM,
        "snr_definition": DETECT_SNR_DEFINITION,
        "frequency_reference": "baseband_offset",
        "sample_rate_hz": rate,
        "sample_count": int(x.size),
        "duration_s": duration,
        "nfft": nfft,
        "hop_samples": hop,
        "frame_count": int(psd.shape[0]),
        "config": {
            "nfft": nfft,
            "threshold_db": threshold_db,
            "band_threshold_db": resolved["band_threshold_db"],
            "min_bandwidth_hz": round(resolved["min_bandwidth_hz"], 6),
            "min_duration_s": resolved["min_duration_s"],
            "max_detections": resolved["max_detections"],
            "merge_bins": resolved["merge_bins"],
        },
        "freq_resolution_hz": resolution,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": round(threshold_dbfs, 3),
        "detections": detections,
    }
    box_rows = [[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                for item in detections]
    arrays = {
        "frequency": frequencies,
        "frame_time": (starts + (nfft - 1) / 2.0) / rate,
        "spectrogram_db": (10.0 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        # 检测依据：时间平均 PSD（与频谱页口径一致）；中位数 PSD 作对照
        "spectrum_db": 10.0 * np.log10(np.maximum(psd_average, 1e-30)),
        "spectrum_median_db": 10.0 * np.log10(np.maximum(np.median(psd, axis=0), 1e-30)),
        "threshold_db": np.array([threshold_dbfs], dtype=np.float64),
        "noise_floor_db": np.array([noise_floor_db], dtype=np.float64),
        "detection_boxes": np.array(box_rows, dtype=np.float64).reshape(-1, 4),
        "detection_id": np.array([item["id"] for item in detections], dtype=np.int64),
        "detection_snr_db": np.array([item["snr_db"] for item in detections], dtype=np.float64),
        "detection_power_dbfs": np.array([item["power_dbfs"] for item in detections], dtype=np.float64),
    }
    return summary, arrays
