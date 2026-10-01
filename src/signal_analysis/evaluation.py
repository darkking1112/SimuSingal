"""Scoring shared by every detection/recognition algorithm in the project.

Pure NumPy with no GUI, storage or workspace imports, so the CLI, the GUI
comparison page, the offline benchmarks and the optional training scripts
all score results with exactly the same code. This module also owns the
**frozen detection result contract** used by the traditional detector
(:func:`signal_analysis._numeric_energy.detect_signals`) and by the optional ONNX
detector in :mod:`signal_analysis.ml`.

Contract (``detect_result_v1``), written into every ``detect`` run::

    {"algorithm": "energy_detect_v1",
     "config": {"nfft": 512, "threshold_db": 6.0, "min_bandwidth_hz": 5859.4},
     "noise_floor_dbfs_per_hz": -71.3,
     "snr_definition": "inband_snr_v1",
     "detections": [{"id": 1, "method": "energy",
                     "center_hz": 100000.0, "bandwidth_hz": 190000.0,
                     "f_low_hz": 5000.0, "f_high_hz": 195000.0,
                     "t_start_s": 0.0, "t_end_s": 0.2,
                     "power_dbfs": -10.1, "snr_db": 18.4,
                     "session_id": 0, "confidence": 0.92}],
     "truth": [...], "metrics": {...}}

Naming convention (unchanged from the rest of the repository): IQ data is
complex baseband, so ``center_hz`` is a **baseband frequency offset**, never
an RF carrier. For SSB the occupied band is one-sided, therefore the
reported ``center_hz`` is the centre of the occupied band and
``nominal_offset_hz`` (present in truth entries) keeps the generator's
nominal point.

Truth entries come straight from the IQ generator summary, which is stored
in ``asset_metadata['generation']``; the generator already reports in-band
SNR with the same ``inband_snr_v1`` convention, so detector output and
truth are directly comparable.
"""

import numpy as np

from .core_api import occupied_interval
from ._numeric_iqgen import (
    _fh_hop_boundaries,
)

CONTRACT_VERSION = "detect_result_v1"
HOP_CONTRACT = "fh_hops_v1"
DETECT_ALGORITHM = "energy_detect_v1"
HOP_ALGORITHM = "hop_track_v1"
SNR_DEFINITION = "inband_snr_v1"


def _finite_or_none(value):
    """Float value or ``None``; results are serialised with allow_nan=False."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _rounded(value, digits=6):
    number = _finite_or_none(value)
    return None if number is None else round(number, digits)


def signal_truth(summary):
    """Truth list for one IQ generator summary (:func:`plan_signal` output).

    One entry per generated signal. Hopping signals follow the agreed
    "one session, one instance" semantics: the reported band is the session
    band of the channels that were **actually visited** (the generator draws
    the hop sequence randomly, so in a short record the outer channels may
    never be used). Falling back to ``bandwidth_actual`` (``span +
    hop_bandwidth``) keeps the entry well defined when the hop set is only
    described nominally.
    """
    if not isinstance(summary, dict):
        return []
    duration = _finite_or_none(summary.get("duration_s"))
    truth = []
    for index, entry in enumerate(summary.get("signals") or []):
        if not isinstance(entry, dict):
            continue
        offset = _finite_or_none(entry.get("offset"))
        width = _finite_or_none(entry.get("bandwidth_actual", entry.get("bandwidth")))
        if offset is None or width is None or width <= 0:
            continue
        mode = str(entry.get("mode", ""))
        hopping = mode.startswith("fh")
        low, high = occupied_interval(offset, width, mode, entry.get("side"))
        visited_channels = None
        if hopping:
            hop_bw = _finite_or_none(entry.get("hop_bandwidth"))
            visited = [value for value in (_finite_or_none(point)
                                          for point in (entry.get("hop_points") or []))
                       if value is not None]
            if hop_bw and hop_bw > 0 and visited:
                channels = sorted(set(visited))
                low = channels[0] - hop_bw / 2.0
                high = channels[-1] + hop_bw / 2.0
                visited_channels = len(channels)
        truth.append({
            "index": index,
            "mode": mode,
            "hopping": hopping,
            "nominal_offset_hz": _rounded(offset),
            "nominal_bandwidth_hz": _rounded(width),
            "center_hz": _rounded((low + high) / 2.0),
            "bandwidth_hz": _rounded(high - low),
            "f_low_hz": _rounded(low),
            "f_high_hz": _rounded(high),
            "t_start_s": 0.0,
            "t_end_s": duration,
            "power_dbfs": _finite_or_none(entry.get("power_dbfs_actual")),
            "snr_inband_db": _finite_or_none(entry.get("snr_inband_db")),
            "session_id": index,
            "visited_channels": visited_channels,
        })
    return truth


def hop_truth(summary):
    """Per-dwell truth for the hopping contract (``fh_hops_v1``).

    One entry per hop the generator actually emitted, i.e. per entry of
    ``hop_points`` — the hop sequence is drawn randomly, so consecutive hops
    may reuse a channel (indistinguishable from a longer dwell), and the
    record may end mid-sequence. Only ``fh*`` modes have this granularity,
    every other mode returns ``[]``: their truth is the whole transmission,
    which :func:`signal_truth` already describes.

    The **band** of every entry is ``hop_bandwidth`` around the visited hop
    point, not the generator's ``occupied_bandwidth`` (which is the measured
    occupancy of the whole signal). With that convention the union of all
    entries is exactly the session band reported by :func:`signal_truth`,
    so the two granularities can be scored side by side.

    ``snr_inband_db`` is converted to the per-hop bandwidth:
    ``snr_inband_db + 10*log10(bandwidth_actual / hop_bandwidth)``. The
    generator reports in-band SNR over its full actual bandwidth, while a
    hop detector measures the power of a single dwell inside one hop
    bandwidth, so the conversion subtracts ``N0·(B_actual - B_hop)``.
    """
    if not isinstance(summary, dict):
        return []
    rate = _finite_or_none(summary.get("sample_rate_hz"))
    duration = _finite_or_none(summary.get("duration_s"))
    truth = []
    for session_index, entry in enumerate(summary.get("signals") or []):
        if not isinstance(entry, dict):
            continue
        mode = str(entry.get("mode", ""))
        if not mode.startswith("fh"):
            continue
        hop_bw = _finite_or_none(entry.get("hop_bandwidth"))
        points = [value for value in (_finite_or_none(point)
                                      for point in (entry.get("hop_points") or []))
                  if value is not None]
        if not hop_bw or hop_bw <= 0 or not points or not rate:
            continue
        count = _finite_or_none(entry.get("sample_count")) or (
            rate * duration if duration else None)
        if not count:
            continue
        width = _finite_or_none(entry.get("bandwidth_actual", entry.get("bandwidth")))
        hop_rate = _finite_or_none(entry.get("hop_rate"))
        power_dbfs = _finite_or_none(entry.get("power_dbfs_actual"))
        session_snr = _finite_or_none(entry.get("snr_inband_db"))
        correction = (10.0 * float(np.log10(width / hop_bw))
                      if width and width > 0 else 0.0)
        boundaries = _fh_hop_boundaries(int(round(count)), len(points))
        for hop_index, point in enumerate(points):
            start = float(boundaries[hop_index]) / rate
            stop = float(boundaries[hop_index + 1]) / rate
            truth.append({
                "index": len(truth),
                "mode": mode,
                "session_index": session_index,
                "hop_index": hop_index,
                "hop_count": len(points),
                "center_hz": _rounded(point),
                "bandwidth_hz": _rounded(hop_bw),
                "f_low_hz": _rounded(point - hop_bw / 2.0),
                "f_high_hz": _rounded(point + hop_bw / 2.0),
                "t_start_s": _rounded(start),
                "t_end_s": _rounded(stop),
                "dwell_s": _rounded(stop - start),
                "hop_rate_hz": _rounded(hop_rate),
                "power_dbfs": power_dbfs,
                "snr_inband_db": (None if session_snr is None
                                  else _rounded(session_snr + correction)),
            })
    return truth


def match_detections(truth, detections, gate_ratio=0.75):
    """Greedy one-to-one matching of detections to truth by centre distance.

    A pair is eligible when the centre distance is at most ``gate_ratio``
    times the wider of the two bands; the closest eligible pair wins first
    and each truth/detection is used at most once. Greedy matching is used
    instead of the Hungarian algorithm on purpose: with a handful of
    targets the results are identical, and no SciPy dependency is allowed
    in the compiled core.

    Returns a sorted list of ``(truth_index, detection_index)`` pairs.
    """
    candidates = []
    for t_index, t in enumerate(truth or []):
        t_center = _finite_or_none(t.get("center_hz"))
        t_band = _finite_or_none(t.get("bandwidth_hz")) or 0.0
        if t_center is None:
            continue
        for d_index, d in enumerate(detections or []):
            d_center = _finite_or_none(d.get("center_hz"))
            d_band = _finite_or_none(d.get("bandwidth_hz")) or 0.0
            if d_center is None:
                continue
            distance = abs(d_center - t_center)
            gate = gate_ratio * max(abs(t_band), abs(d_band))
            if gate <= 0:
                gate = gate_ratio * 1e-9
            if distance <= gate:
                candidates.append((distance, t_index, d_index))
    candidates.sort()
    used_truth, used_detection, matched = set(), set(), []
    for _, t_index, d_index in candidates:
        if t_index in used_truth or d_index in used_detection:
            continue
        used_truth.add(t_index)
        used_detection.add(d_index)
        matched.append((t_index, d_index))
    matched.sort()
    return matched


def evaluate_detections(truth, detections, gate_ratio=0.75, contract=CONTRACT_VERSION):
    """Detection metrics for the detection contracts.

    Used unchanged by both frozen contracts: ``detect_result_v1`` (one
    instance per transmission) and ``fh_hops_v1`` (one instance per hop).
    The only difference is the ``contract`` string written into the result,
    so ``contract=HOP_CONTRACT`` scores hop entries; everything else —
    the matching gate, the reported errors — has the same meaning.

    Reported values: ``true/detected/matched/missed/false_alarm``, precision,
    recall, F1, ``center_mae_hz``, ``center_rmse_hz``, ``bandwidth_mape``
    (fraction, not percent) and ``snr_mae_db``. Every field is a finite float
    or ``None``; no ``inf``/``nan`` ever leaves this function.
    """
    truth = list(truth or [])
    detections = list(detections or [])
    matched = match_detections(truth, detections, gate_ratio)
    center_errors, bandwidth_errors, snr_errors, pairs = [], [], [], []
    for t_index, d_index in matched:
        t, d = truth[t_index], detections[d_index]
        center_error = float(d["center_hz"]) - float(t["center_hz"])
        truth_band = _finite_or_none(t.get("bandwidth_hz")) or 0.0
        detection_band = _finite_or_none(d.get("bandwidth_hz")) or 0.0
        band_relative = (abs(detection_band - truth_band) / truth_band
                         if truth_band > 0 else None)
        truth_snr = _finite_or_none(t.get("snr_inband_db"))
        detection_snr = _finite_or_none(d.get("snr_db"))
        snr_error = (detection_snr - truth_snr
                     if truth_snr is not None and detection_snr is not None else None)
        center_errors.append(center_error)
        if band_relative is not None:
            bandwidth_errors.append(band_relative)
        if snr_error is not None:
            snr_errors.append(snr_error)
        pairs.append({
            "truth_index": t_index,
            "detection_id": d.get("id"),
            "truth_center_hz": _finite_or_none(t.get("center_hz")),
            "truth_bandwidth_hz": _finite_or_none(t.get("bandwidth_hz")),
            "truth_snr_inband_db": truth_snr,
            "detection_center_hz": _finite_or_none(d.get("center_hz")),
            "detection_bandwidth_hz": _finite_or_none(d.get("bandwidth_hz")),
            "detection_snr_db": detection_snr,
            "center_error_hz": _rounded(center_error),
            "bandwidth_relative_error": _rounded(band_relative),
            "snr_error_db": _rounded(snr_error),
        })
    true_count, detection_count = len(truth), len(detections)
    matched_count = len(matched)
    missed = true_count - matched_count
    false_alarm = detection_count - matched_count
    precision = matched_count / detection_count if detection_count else None
    recall = matched_count / true_count if true_count else None
    if precision is None and recall is None:
        f1 = None  # nothing was scored: no truth and no detection
    else:
        safe_precision = 0.0 if precision is None else precision
        safe_recall = 0.0 if recall is None else recall
        f1 = (2 * safe_precision * safe_recall / (safe_precision + safe_recall)
              if (safe_precision + safe_recall) > 0 else 0.0)
    return {
        "contract": contract,
        "snr_definition": SNR_DEFINITION,
        "true": true_count,
        "detected": detection_count,
        "matched": matched_count,
        "missed": missed,
        "false_alarm": false_alarm,
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "f1": _rounded(f1),
        "center_mae_hz": _rounded(np.mean(np.abs(center_errors))) if center_errors else None,
        "center_rmse_hz": _rounded(np.sqrt(np.mean(np.square(center_errors)))) if center_errors else None,
        "bandwidth_mape": _rounded(np.mean(bandwidth_errors)) if bandwidth_errors else None,
        "snr_mae_db": _rounded(np.mean(np.abs(snr_errors))) if snr_errors else None,
        "pairs": pairs,
    }


def classification_metrics(truth_labels, predicted_labels, labels=None):
    """Confusion matrix and macro metrics for modulation recognition (A09).

    ``labels`` defaults to the sorted union of both inputs, so a partially
    covered class still appears (with zero support) in the matrix.
    """
    truth_labels = [str(label) for label in (truth_labels or [])]
    predicted_labels = [str(label) for label in (predicted_labels or [])]
    if len(truth_labels) != len(predicted_labels):
        raise ValueError("真值标签与预测标签数量必须一致")
    if labels is None:
        labels = sorted(set(truth_labels) | set(predicted_labels))
    labels = [str(label) for label in labels]
    index_of = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, predicted in zip(truth_labels, predicted_labels):
        if truth in index_of and predicted in index_of:
            matrix[index_of[truth], index_of[predicted]] += 1
    total = int(matrix.sum())
    per_class = []
    f1_values = []
    for position, label in enumerate(labels):
        true_positive = int(matrix[position, position])
        support = int(matrix[position, :].sum())
        predicted_total = int(matrix[:, position].sum())
        precision = true_positive / predicted_total if predicted_total else None
        recall = true_positive / support if support else None
        if precision is None and recall is None:
            f1 = None  # 该类在真值与预测中都没出现：不可评分，且不拉低 macro F1
        else:
            safe_precision = 0.0 if precision is None else precision
            safe_recall = 0.0 if recall is None else recall
            f1 = (2 * safe_precision * safe_recall / (safe_precision + safe_recall)
                  if (safe_precision + safe_recall) > 0 else 0.0)
        if f1 is not None:
            f1_values.append(f1)
        per_class.append({"label": label, "support": support,
                          "precision": _rounded(precision), "recall": _rounded(recall),
                          "f1": _rounded(f1)})
    accuracy = float(np.trace(matrix) / total) if total else None
    return {
        "labels": labels,
        "confusion": matrix.tolist(),
        "total": total,
        "accuracy": _rounded(accuracy),
        "macro_f1": _rounded(np.mean(f1_values)) if f1_values else None,
        "per_class": per_class,
    }


# --------------------------------------------------------------------------
# 逐跳精确评估（``per_hop_eval_v1``，方案文档 §5）
# --------------------------------------------------------------------------

PER_HOP_CONTRACT = "per_hop_eval_v1"

#: 默认分组报表轴与分箱（None 表示开放端）。
DEFAULT_HOP_GROUPS = {
    "snr_db": [None, 0, 10, 20, None],
    "hop_rate_hz": [None, 50, 150, None],
    "bandwidth_hz": [None, 5e4, 5e5, None],
}

_BIG_COST = 1e6


def _hop_box(entry):
    """跳条目 → ``(t0, t1, f0, f1)``；缺时间或频率信息返回 ``None``。"""
    start = _finite_or_none(entry.get("t_start_s"))
    stop = _finite_or_none(entry.get("t_end_s"))
    low = _finite_or_none(entry.get("f_low_hz"))
    high = _finite_or_none(entry.get("f_high_hz"))
    if low is None or high is None:
        center = _finite_or_none(entry.get("center_hz"))
        bandwidth = _finite_or_none(entry.get("bandwidth_hz"))
        if center is None or bandwidth is None or bandwidth <= 0:
            return None
        low, high = center - bandwidth / 2.0, center + bandwidth / 2.0
    if start is None or stop is None or stop <= start or high <= low:
        return None
    return float(start), float(stop), float(low), float(high)


def _box_iou(left, right):
    """时间-频率二维 IoU；任一维无重叠即为 0。"""
    t_overlap = min(left[1], right[1]) - max(left[0], right[0])
    f_overlap = min(left[3], right[3]) - max(left[2], right[2])
    if t_overlap <= 0 or f_overlap <= 0:
        return 0.0
    intersection = t_overlap * f_overlap
    area_left = (left[1] - left[0]) * (left[3] - left[2])
    area_right = (right[1] - right[0]) * (right[3] - right[2])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def _hungarian_assign(cost):
    """经典 O(n³) 匈牙利算法（最小化方阵代价）；返回 ``row → col``（0 基）。"""
    n = len(cost)
    if n == 0:
        return {}
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for row in range(1, n + 1):
        p[0] = row
        column = 0
        min_values = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[column] = True
            current_row = p[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, n + 1):
                if not used[candidate]:
                    value = cost[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                    if value < min_values[candidate]:
                        min_values[candidate] = value
                        way[candidate] = column
                    if min_values[candidate] < delta:
                        delta = min_values[candidate]
                        next_column = candidate
            for candidate in range(n + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    min_values[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while column:
            previous = way[column]
            p[column] = p[previous]
            column = previous
    return {p[column] - 1: column - 1 for column in range(1, n + 1) if p[column]}


def match_hop_tracks(truth, predicted, *, iou_gate=0.25, time_tolerance_s=None,
                     freq_tolerance_hz=None):
    """逐跳一对一匹配：匈牙利算法最大化时间-频率 IoU 总和。

    只接受同时满足 ``IoU >= iou_gate`` 与时间/频率容差（给了才检查）的配对；
    其余按漏检/多检计入。返回 ``[(truth_index, predicted_index, iou)]``。
    """
    truth_boxes = [_hop_box(entry) for entry in truth]
    predicted_boxes = [_hop_box(entry) for entry in predicted]
    n_truth, n_predicted = len(truth_boxes), len(predicted_boxes)
    size = max(n_truth, n_predicted)
    if size == 0:
        return []
    eligible = np.zeros((size, size), dtype=bool)
    ious = np.zeros((size, size), dtype=float)
    for i in range(n_truth):
        for j in range(n_predicted):
            if truth_boxes[i] is None or predicted_boxes[j] is None:
                continue
            iou = _box_iou(truth_boxes[i], predicted_boxes[j])
            if iou < iou_gate:
                continue
            if time_tolerance_s is not None:
                error = max(abs(predicted_boxes[j][0] - truth_boxes[i][0]),
                            abs(predicted_boxes[j][1] - truth_boxes[i][1]))
                if error > time_tolerance_s:
                    continue
            if freq_tolerance_hz is not None:
                truth_center = (truth_boxes[i][2] + truth_boxes[i][3]) / 2.0
                pred_center = (predicted_boxes[j][2] + predicted_boxes[j][3]) / 2.0
                if abs(pred_center - truth_center) > freq_tolerance_hz:
                    continue
            eligible[i, j] = True
            ious[i, j] = iou
    cost = [[0.0] * size for _ in range(size)]
    for i in range(n_truth):
        for j in range(n_predicted):
            cost[i][j] = (1.0 - ious[i][j]) if eligible[i, j] else _BIG_COST
    assignment = _hungarian_assign(cost)
    pairs = []
    for i in range(n_truth):
        j = assignment.get(i)
        if j is not None and j < n_predicted and eligible[i, j]:
            pairs.append((i, j, float(ious[i, j])))
    pairs.sort()
    return pairs


def _sequence_consistency(truth_channels, predicted_channels):
    """最长公共子序列比例与归一化编辑距离。"""
    n, m = len(truth_channels), len(predicted_channels)
    if n == 0 or m == 0:
        return None, None
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if truth_channels[i - 1] == predicted_channels[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs_ratio = table[n][m] / max(n, m)
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if truth_channels[i - 1] == predicted_channels[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return lcs_ratio, previous[m] / max(n, m)


def _channelize(truth, predicted, channel_bin_hz, freq_tolerance_hz):
    """把跳中心频率量化为频道序号；未知频道返回 ``None``（不计入序列指标）。"""
    truth_centers = [(_hop_box(entry) or (None,) * 4) for entry in truth]
    truth_centers = [None if box is None or box[0] is None else (box[2] + box[3]) / 2.0
                     for box in truth_centers]
    predicted_centers = []
    for entry in predicted:
        box = _hop_box(entry)
        predicted_centers.append(None if box is None else (box[2] + box[3]) / 2.0)
    if any(center is None for center in truth_centers) or \
            any(center is None for center in predicted_centers):
        return None, None
    if channel_bin_hz is None:
        distinct = sorted(set(round(center, 6) for center in truth_centers))
        gaps = [b - a for a, b in zip(distinct, distinct[1:]) if b > a]
        channel_bin_hz = min(gaps) if gaps else None
    if channel_bin_hz is None:
        # 只有一个频道：命中判定用频率容差（未给时用半带宽的较小值）。
        tolerance = freq_tolerance_hz
        if tolerance is None:
            bandwidths = [_finite_or_none(entry.get("bandwidth_hz")) for entry in truth]
            bandwidths = [value for value in bandwidths if value]
            tolerance = min(bandwidths) / 4.0 if bandwidths else 1.0
        channel = round(truth_centers[0], 6)
        return ([0] * len(truth_centers),
                [0 if abs(center - channel) <= tolerance else -1
                 for center in predicted_centers])
    return ([int(round(center / channel_bin_hz)) for center in truth_centers],
            [int(round(center / channel_bin_hz)) for center in predicted_centers])


def _spec_labels(edges):
    specs = []
    for low, high in zip(edges, edges[1:]):
        if low is None and high is None:
            label = "全部"
        elif low is None:
            label = f"< {high:g}"
        elif high is None:
            label = f"≥ {low:g}"
        else:
            label = f"[{low:g}, {high:g})"
        specs.append((low, high, label))
    return specs


def _bucket(value, specs):
    if value is None:
        return "未知"
    number = float(value)
    for low, high, label in specs:
        if (low is None or number >= low) and (high is None or number < high):
            return label
    return "未知"


def _group_report(truth, predicted, matched_pairs, specs):
    """按分组轴统计每档的总数、可评数、漏检与误检。"""
    buckets = {}
    order = [label for _, _, label in specs] + ["未知"]
    matched_truth = {pair[0] for pair in matched_pairs}
    matched_predicted = {pair[1] for pair in matched_pairs}
    for index, entry in enumerate(truth):
        key = _bucket(_finite_or_none(entry.get("snr_db")), specs)
        bucket = buckets.setdefault(key, {"true": 0, "matched": 0, "missed": 0,
                                          "false_alarm": 0})
        bucket["true"] += 1
        if index in matched_truth:
            bucket["matched"] += 1
        else:
            bucket["missed"] += 1
    for index, entry in enumerate(predicted):
        if index in matched_predicted:
            continue
        key = _bucket(_finite_or_none(entry.get("snr_db")), specs)
        bucket = buckets.setdefault(key, {"true": 0, "matched": 0, "missed": 0,
                                          "false_alarm": 0})
        bucket["false_alarm"] += 1
    report = []
    for key in [label for label in order if label in buckets]:
        bucket = buckets[key]
        detected = bucket["matched"] + bucket["false_alarm"]
        precision = bucket["matched"] / detected if detected else None
        recall = (bucket["matched"] / bucket["true"]) if bucket["true"] else None
        report.append({"label": key, **bucket, "precision": _rounded(precision),
                       "recall": _rounded(recall)})
    return report


def evaluate_hop_tracks(truth, predicted, *, iou_gate=0.25, time_tolerance_s=None,
                        freq_tolerance_hz=None, channel_bin_hz=None,
                        resolution=None, predicted_kind="per_hop", coverage="complete",
                        groups=None, reason=None):
    """逐跳评估契约 ``per_hop_eval_v1``（方案文档 §5.2）。

    参数：
        truth/predicted: 跳条目列表（字段同 ``hop_truth`` / 逐跳检测输出）。
        predicted_kind: ``per_hop`` 或 ``session``；会话级预测不可比。
        coverage: 录制覆盖度；非 ``complete`` 不可比。
        resolution: ``{"time_resolution_s":…, "freq_resolution_hz":…}``，
            容差低于分辨率时写入警告（不改变计算）。
        groups: 分组轴 → 分箱边界；默认 :data:`DEFAULT_HOP_GROUPS`。

    返回：评估结果字典；``comparable=False`` 时 ``metrics`` 为 ``None`` 并给出
    ``reason``，不计入任何分母。
    """
    warnings = []
    if resolution:
        time_resolution = resolution.get("time_resolution_s")
        freq_resolution = resolution.get("freq_resolution_hz")
        if (time_tolerance_s is not None and time_resolution is not None
                and time_tolerance_s < time_resolution):
            warnings.append(f"时间容差 {time_tolerance_s:g}s 小于分析时间分辨率 "
                            f"{time_resolution:g}s，匹配结果不可解释")
        if (freq_tolerance_hz is not None and freq_resolution is not None
                and freq_tolerance_hz < freq_resolution):
            warnings.append(f"频率容差 {freq_tolerance_hz:g}Hz 小于分析频率分辨率 "
                            f"{freq_resolution:g}Hz，匹配结果不可解释")
    config = {"iou_gate": iou_gate, "time_tolerance_s": time_tolerance_s,
              "freq_tolerance_hz": freq_tolerance_hz, "channel_bin_hz": channel_bin_hz}
    base = {"contract": PER_HOP_CONTRACT, "config": config,
            "resolution": resolution or None, "warnings": warnings}
    if reason is None:
        if predicted_kind == "session":
            reason = "预测粒度为会话级，不能与逐跳真值匹配"
        elif not truth:
            reason = "真值无逐跳信息（该录制只能做会话级评估）"
        elif coverage != "complete":
            reason = f"录制覆盖度 {coverage}，不计入逐跳分母"
    if reason is not None:
        return {**base, "comparable": False, "reason": reason, "metrics": None,
                "pairs": [], "session": None, "groups": {}}
    matched_pairs = match_hop_tracks(truth, predicted, iou_gate=iou_gate,
                                     time_tolerance_s=time_tolerance_s,
                                     freq_tolerance_hz=freq_tolerance_hz)
    pairs = []
    start_errors, end_errors, dwell_errors = [], [], []
    center_errors, bandwidth_errors = [], []
    for truth_index, predicted_index, iou in matched_pairs:
        t_entry, p_entry = truth[truth_index], predicted[predicted_index]
        t_box, p_box = _hop_box(t_entry), _hop_box(p_entry)
        start_error = p_box[0] - t_box[0]
        end_error = p_box[1] - t_box[1]
        dwell_error = (p_box[1] - p_box[0]) - (t_box[1] - t_box[0])
        truth_center = (t_box[2] + t_box[3]) / 2.0
        pred_center = (p_box[2] + p_box[3]) / 2.0
        center_error = pred_center - truth_center
        truth_band = t_box[3] - t_box[2]
        band_relative = ((p_box[3] - p_box[2]) - truth_band) / truth_band \
            if truth_band > 0 else None
        start_errors.append(start_error)
        end_errors.append(end_error)
        dwell_errors.append(dwell_error)
        center_errors.append(center_error)
        if band_relative is not None:
            bandwidth_errors.append(band_relative)
        pairs.append({
            "truth_index": truth_index, "predicted_index": predicted_index,
            "iou": _rounded(iou), "truth_hop_index": t_entry.get("hop_index"),
            "predicted_hop_index": p_entry.get("hop_index"),
            "t_start_error_s": _rounded(start_error),
            "t_end_error_s": _rounded(end_error),
            "dwell_error_s": _rounded(dwell_error),
            "center_error_hz": _rounded(center_error),
            "bandwidth_relative_error": _rounded(band_relative),
        })
    true_count, predicted_count = len(truth), len(predicted)
    matched_count = len(matched_pairs)
    missed = true_count - matched_count
    false_alarm = predicted_count - matched_count
    precision = matched_count / predicted_count if predicted_count else None
    recall = matched_count / true_count if true_count else None
    if precision is None and recall is None:
        f1 = None
    else:
        safe_precision = 0.0 if precision is None else precision
        safe_recall = 0.0 if recall is None else recall
        f1 = (2 * safe_precision * safe_recall / (safe_precision + safe_recall)
              if (safe_precision + safe_recall) > 0 else 0.0)
    truth_rates = [_finite_or_none(entry.get("hop_rate_hz")) for entry in truth]
    truth_rates = [value for value in truth_rates if value]
    predicted_rates = [_finite_or_none(entry.get("hop_rate_hz")) for entry in predicted]
    predicted_rates = [value for value in predicted_rates if value]
    hop_rate_error = (abs(float(np.mean(predicted_rates)) - float(np.mean(truth_rates)))
                      if truth_rates and predicted_rates else None)
    truth_channels, predicted_channels = _channelize(truth, predicted, channel_bin_hz,
                                                     freq_tolerance_hz)
    sequence = None
    channel_set = None
    if truth_channels is not None:
        truth_set = set(truth_channels)
        predicted_set = {value for value in predicted_channels if value >= 0}
        union = truth_set | predicted_set
        channel_set = (len(truth_set & predicted_set) / len(union)) if union else None
        lcs_ratio, edit_ratio = _sequence_consistency(truth_channels, predicted_channels)
        sequence = {"lcs_ratio": _rounded(lcs_ratio),
                    "normalized_edit_distance": _rounded(edit_ratio),
                    "truth_channels": truth_channels,
                    "predicted_channels": predicted_channels}
    metrics = {
        "true": true_count, "detected": predicted_count, "matched": matched_count,
        "missed": missed, "false_alarm": false_alarm,
        "precision": _rounded(precision), "recall": _rounded(recall), "f1": _rounded(f1),
        "t_start_mae_s": _rounded(np.mean(np.abs(start_errors))) if start_errors else None,
        "t_end_mae_s": _rounded(np.mean(np.abs(end_errors))) if end_errors else None,
        "dwell_mae_s": _rounded(np.mean(np.abs(dwell_errors))) if dwell_errors else None,
        "center_mae_hz": _rounded(np.mean(np.abs(center_errors))) if center_errors else None,
        "center_rmse_hz": _rounded(np.sqrt(np.mean(np.square(center_errors))))
        if center_errors else None,
        "bandwidth_mape": _rounded(np.mean(bandwidth_errors)) if bandwidth_errors else None,
        "hop_count_error": predicted_count - true_count,
        "hop_rate_mae_hz": _rounded(hop_rate_error),
        "channel_set_consistency": _rounded(channel_set),
    }
    group_reports = {}
    for axis, edges in (groups or DEFAULT_HOP_GROUPS).items():
        group_reports[axis] = _group_report(truth, predicted, matched_pairs,
                                            _spec_labels(edges))
    return {**base, "comparable": True, "reason": None, "metrics": metrics,
            "pairs": pairs,
            "session": {"hop_count_error": metrics["hop_count_error"],
                        "hop_rate_mae_hz": metrics["hop_rate_mae_hz"],
                        "channel_set_consistency": metrics["channel_set_consistency"],
                        "sequence": sequence},
            "groups": group_reports}
