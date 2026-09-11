"""Scoring shared by every detection/recognition algorithm in the project.

Pure NumPy with no GUI, storage or workspace imports, so the CLI, the GUI
comparison page, the offline benchmarks and the optional training scripts
all score results with exactly the same code. This module also owns the
**frozen detection result contract** used by the traditional detector
(:func:`signal_analysis._numeric.detect_signals`) and by the optional ONNX
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

CONTRACT_VERSION = "detect_result_v1"
DETECT_ALGORITHM = "energy_detect_v1"
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


def evaluate_detections(truth, detections, gate_ratio=0.75):
    """Detection metrics for the frozen contract.

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
        "contract": CONTRACT_VERSION,
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
