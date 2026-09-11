"""网络输出解码：归一化边框 → 频段/时间区间，并按 IoU 去重。

模型输出契约（``normalized_boxes_v1``）：

* 形状 ``(N, 6)``、``(1, N, 6)`` 或 ``(1, 6, N)`` 的浮点数组；
* 每行 ``[x_center, y_center, width, height, confidence, class]``；
* 前四列按图像宽高归一化到 ``[0, 1]``，坐标是**边框**（不是像素中心），
  且 ``x`` 自左向右、``y`` 自上向下（图像行 0 为 ``+fs/2``）；
* 列数与类别数固定，避免解码方式随模型变化（跨导出的数值一致性检查 C02）。

解码只做几何与去重，辐射量（功率、带内信噪比）由
:func:`signal_analysis.ml.tensor.measure_band` 在同一口径下测量。
"""

from __future__ import annotations

import numpy as np

from .tensor import box_to_band

BOX_COLUMNS = 6
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.5


def parse_model_output(output):
    """校验并整形网络输出，返回 ``(rows, shape)``（``rows`` 为 ``(N, 6)``）。"""
    array = np.asarray(output, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError("模型批次维应为 1")
        array = array[0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != BOX_COLUMNS:
        if array.ndim == 2 and array.shape[0] == BOX_COLUMNS:
            array = array.T
    if array.ndim != 2 or array.shape[1] != BOX_COLUMNS:
        raise ValueError(f"模型输出应为 (N, {BOX_COLUMNS}) 的检测框数组，实际为 {np.shape(output)}")
    if not np.all(np.isfinite(array)):
        raise ValueError("模型输出包含非有限数值")
    return array, np.shape(output)


def _iou(first, second):
    """归一化边框的 IoU；两个边框均为 ``(x, y, w, h)``（中心式）。"""
    first_x1, first_y1 = first[0] - first[2] / 2.0, first[1] - first[3] / 2.0
    first_x2, first_y2 = first[0] + first[2] / 2.0, first[1] + first[3] / 2.0
    second_x1, second_y1 = second[0] - second[2] / 2.0, second[1] - second[3] / 2.0
    second_x2, second_y2 = second[0] + second[2] / 2.0, second[1] + second[3] / 2.0
    overlap_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    overlap_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = overlap_width * overlap_height
    if intersection <= 0.0:
        return 0.0
    union = (first[2] * first[3] + second[2] * second[3] - intersection)
    return float(intersection / union) if union > 0 else 0.0


def non_max_suppression(rows, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """按置信度贪心去重；同类且 IoU 超阈值时只保留高置信框。"""
    threshold = float(iou_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("IoU 阈值应在 0～1 之间")
    order = sorted(range(len(rows)), key=lambda index: -float(rows[index][4]))
    kept = []
    for index in order:
        candidate = rows[index]
        duplicate = False
        for other in kept:
            if int(candidate[5]) != int(other[5]):
                continue
            if _iou(candidate, other) > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def boxes_to_bands(rows, meta, *, score_threshold=DEFAULT_SCORE_THRESHOLD,
                   iou_threshold=DEFAULT_IOU_THRESHOLD, max_detections=32,
                   min_bandwidth_hz=0.0, min_duration_s=0.0, labels=None):
    """解码 + 去重 + 几何筛选，返回候选框字典列表（按频段升序）。"""
    score_gate = float(score_threshold)
    if not 0.0 <= score_gate < 1.0:
        raise ValueError("置信度阈值应在 0～1 之间")
    limit = int(max_detections)
    if limit < 1 or limit > 512:
        raise ValueError("最多候选数应在 1～512 之间")
    definition = list(labels or ["emitter"])
    candidates = []
    for row in non_max_suppression(rows, iou_threshold):
        confidence = float(row[4])
        if confidence < score_gate:
            continue
        band = box_to_band(meta, row[0], row[1], row[2], row[3])
        bandwidth = band["f_high_hz"] - band["f_low_hz"]
        if bandwidth < float(min_bandwidth_hz):
            continue
        if band["t_end_s"] - band["t_start_s"] < float(min_duration_s):
            continue
        class_index = int(round(float(row[5])))
        label = definition[class_index] if 0 <= class_index < len(definition) else definition[-1]
        candidates.append({
            "band": band,
            "confidence": confidence,
            "label": label,
            "class_index": class_index,
            "box": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
        })
    candidates.sort(key=lambda item: -item["confidence"])
    return candidates[:limit]
