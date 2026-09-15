"""Numerical export check and detection scoring against saved annotation boxes."""
import json
from pathlib import Path
import time

import numpy as np


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def verify_native(native_path, contract_path, size):
    """Our two native exporters accept RGB floats in [0,1], independent of the manifest."""
    native, contract = session(native_path), session(contract_path)
    image = np.linspace(0, 1, size * size, dtype=np.float32).reshape(1, 1, size, size)
    raw = native.run(None, {native.get_inputs()[0].name: np.repeat(image, 3, axis=1)})[0]
    actual = contract.run(None, {contract.get_inputs()[0].name: image})[0][0]
    raw = np.asarray(raw)[0]
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError(f"原生输出必须为 pixel_xyxy (1,N,6)，实际 {raw.shape}")
    expected = raw.copy()
    expected[:, 0] = (raw[:, 0] + raw[:, 2]) / (2 * size)
    expected[:, 1] = (raw[:, 1] + raw[:, 3]) / (2 * size)
    expected[:, 2] = (raw[:, 2] - raw[:, 0]) / size
    expected[:, 3] = (raw[:, 3] - raw[:, 1]) / size
    expected[:, :2] = np.clip(expected[:, :2], 0, 1)
    expected[:, 2:4] = np.clip(expected[:, 2:4], 1e-4, 1)
    # Tied scores can yield different TopK orders; compare geometry/class to candidates.
    errors = np.max(np.abs(actual[:, None, :] - expected[None, :, :]), axis=2)
    maximum = float(errors.min(axis=1).max())
    expected_scores = np.sort(expected[:, 4])[::-1][:len(actual)]
    if maximum > 1e-4 or not np.allclose(actual[:, 4], expected_scores, atol=1e-4, rtol=1e-4):
        raise ValueError(f"原生导出与契约模型数值不一致，最大匹配偏差 {maximum:g}；检查预处理和坐标")
    return {"native_input": "RGB float32 [0,1]", "max_error": maximum}


def match_boxes(predictions, truth, threshold=.5):
    from signal_analysis.ml.decode import _iou
    used = set()
    matched = 0
    for prediction in sorted(predictions, key=lambda box: -box[4]):
        candidates = [(i, _iou(prediction, box)) for i, box in enumerate(truth)
                      if i not in used and int(box[5]) == int(prediction[5])]
        if candidates:
            index, overlap = max(candidates, key=lambda item: item[1])
            if overlap >= threshold:
                used.add(index)
                matched += 1
    return matched, len(predictions) - matched, len(truth) - matched


def evaluate(data, model_path, output):
    from detectors.dataset import load_dataset
    from signal_analysis.ml.decode import parse_model_output, non_max_suppression
    _, records = load_dataset(data)
    runtime = session(model_path)
    name = runtime.get_inputs()[0].name
    report = {"score_threshold": .25, "nms_iou": .5, "match_iou": .5,
              "metric": "标注框二维 IoU 评分（独立于生成器真值评分）", "splits": {}}
    for split in ("val", "test"):
        selected = [r for r in records if r.get("split") == split]
        if not selected:
            continue
        tp = fp = fn = noise = noise_false = 0
        latency = []
        for index, record in enumerate(selected):
            image = np.load(Path(data) / record["image"], allow_pickle=False)[None, None].astype(np.float32)
            if index == 0:
                runtime.run(None, {name: image})
            start = time.perf_counter()
            rows, _ = parse_model_output(runtime.run(None, {name: image})[0])
            rows = [row for row in non_max_suppression(rows, .5) if row[4] >= .25][:32]
            latency.append((time.perf_counter() - start) * 1000)
            a, b, c = match_boxes(rows, record.get("boxes", []))
            tp, fp, fn = tp + a, fp + b, fn + c
            if not record.get("boxes"):
                noise += 1
                noise_false += bool(rows)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        report["splits"][split] = {"samples": len(selected), "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "noise_samples": noise, "noise_false_alarm_rate": noise_false / noise if noise else None,
            "latency_ms_p50": float(np.percentile(latency, 50)),
            "latency_ms_p95": float(np.percentile(latency, 95)),
            "timing_scope": "ONNX CPU 推理与框去重；不含 IQ 转时频图"}
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
