"""检测结果后处理：跳频子带的会话判据与合并（传统与 AI 通路共用）。

只依赖 NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

from ..dsp.base import _SESSION_GAP_RATIO


_SESSION_OVERLAP_RATIO = 0.2


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
