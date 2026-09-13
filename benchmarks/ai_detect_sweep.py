#!/usr/bin/env python3
"""AI 检测证据扫描：per-SNR 召回、固定虚警率工作点、参数误差、分级延迟、AMC Macro-F1。

这是 AI 检测路径（``ml_detect``）的**证据生产者**，与 :mod:`benchmarks.detect_sweep`
（传统能量检测）共用同一组场景与同一个 SNR 网格，因此两条路径的数字可以逐格对比。

口径声明（报告里也会原样写进 JSON，避免数字脱离口径被引用）：

* **检测路径**：全部经过 :func:`signal_analysis.ml.detector.ml_detect`，
  即「网络判决 + 传统口径测辐射量」，结果契约 ``detect_result_v1``；
  参数误差与 per-SNR 召回用 :func:`signal_analysis.evaluation.evaluate_detections`
  与 :func:`signal_analysis.evaluation.signal_truth` 计算——**没有任何第二套匹配或误差公式**。
* **SNR 轴**：扫掠轴是**标称** SNR（生成器入参，与 ``detect_sweep.py`` 一致）；
  分档统计用的是每个真值目标的**实测**带内信噪比 ``snr_inband_db``
  （``inband_snr_v1`` 口径，由生成器摘要给出），按 5 dB 分桶。
* **虚警率（FAR）**：在纯噪声记录上统计门限后残留的候选框数，主口径为
  **每帧**（``残留框数 / STFT 帧数``），同时报**每秒**（``/ 记录时长``）与**每条记录**
  的均值/最大值。FAR 一律用**汇总口径**（总框数 ÷ 总帧数），不是逐记录虚警率的平均，
  否则分母不同的比例平均会系统偏移。``--far-basis`` 决定用哪个口径挑工作点。
* **ROC**：所有阈值都在**同一次推理**的输出上重新过门限，不是重跑推理。
  做法是取 ``ml_detect`` 返回的候选框（``model_boxes``/``model_scores``，已过
  ``--min-score`` 门限与 NMS、并按置信度降序），再用**同一个** ``boxes_to_bands``
  逐阈值重放；由于 NMS 与阈值无关、截断发生在按分数排序之后，重放结果与直接以该阈值
  调用解码器一致（脚本会做一次自检并把结果写进 ``candidate_selfcheck``）。
  因此 ``--score-grid`` 的取值必须 ≥ ``--min-score``，否则直接报错。
* **真值来源**：``--truth synthetic`` 用项目生成器造波形（可复现、逐字节一致），
  没有信号记录时走 ``--truth sigmf``；后者需要「SigMF 标注 → 项目真值」的换算，
  该换算**尚未实现**（实采数据到位后再补），因此现在显式报错而不是给出错口径数字。
* **延迟**：``context_ms``/``inference_ms``/``total_ms`` 取自 ``ml_detect`` 自身的计时；
  ``image_ms``（时频图重采样）与 ``gate_ms``（解码/门限）是同参数、同输入的**纯函数**
  重跑计时（确定性、不改变任何判决）；``other_ms`` 为差值口径（合并/测量/结果组装），
  计时噪声下可能略为负。报告同时记录 CPU 型号、核数、onnxruntime 版本与线程数——
  脱离这些元数据的延迟数字不可引用。
* **不做的事**：本脚本不训练、不评估实采数据，也不放松任何冻结契约；
  ``--amc-data`` 只对**已有**的 AMC 数据集做独立评测（内置线性基线），不参与检测扫掠。

用法::

    .venv/bin/python benchmarks/ai_detect_sweep.py --manifest <detector.json>
    .venv/bin/python benchmarks/ai_detect_sweep.py --manifest <detector.json> \\
        --json /tmp/ai_sweep.json --amc-data training/data/amc --modes qpsk,qam16,fh_rc
    .venv/bin/python benchmarks/ai_detect_sweep.py --manifest <detector.json> \\
        --snr -5,0,10 --seeds 2 --noise-trials 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
TRAINING = REPO_ROOT / "training"
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import (  # noqa: E402
    evaluate_detections,
    match_detections,
    signal_truth,
)
from signal_analysis.ml.decode import boxes_to_bands  # noqa: E402
from signal_analysis.ml.detector import ml_detect  # noqa: E402
from signal_analysis.ml.manifest import (  # noqa: E402
    DEFAULT_LABEL_SEMANTICS,
    LABEL_SEMANTICS,
    LABEL_SEMANTICS_FIELD,
)
from signal_analysis.ml.runtime import runtime_version  # noqa: E402
from signal_analysis.ml.tensor import box_to_band, detection_image  # noqa: E402

SCHEMA = "ai_detect_sweep_v1"
RATE_HZ = 1e6
SNR_GRID = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
#: 与 ``benchmarks/detect_sweep.py`` **完全一致**的场景表：同一频点、同一带宽，
#: 便于「传统能量检测 vs AI 检测」逐格对比。改动必须两边同步
#: （``tests/analysis/test_ai_detect_sweep.py`` 会断言两者相等）。
SCENARIOS = {
    "am": {"offset": -200e3, "bandwidth": 80e3},
    "fm": {"offset": 150e3, "bandwidth": 120e3},
    "ssb": {"offset": -350e3, "bandwidth": 60e3, "side": "usb"},
    "ask2": {"offset": 250e3, "bandwidth": 100e3},
    "qpsk": {"offset": 0.0, "bandwidth": 200e3},
    "qam16": {"offset": -100e3, "bandwidth": 300e3},
    "qam64": {"offset": 100e3, "bandwidth": 300e3},
    "fh_rc": {"offset": 0.0, "bandwidth": 400e3},
    "fh_video": {"offset": 0.0, "bandwidth": 300e3},
}
MODES = tuple(SCENARIOS)
#: 延迟分段名（顺序即报告顺序）
STAGES = ("context_ms", "image_ms", "inference_ms", "gate_ms", "other_ms", "total_ms")
#: 实测带内信噪比的分桶宽度（dB），与 ``amc.evaluate_model`` 的 per_snr 分桶一致
SNR_BUCKET_DB = 5.0


# ---------------------------------------------------------------------------
# 小工具：JSON 安全、汇总
# ---------------------------------------------------------------------------


def _clean(value):
    """递归转成 JSON 安全结构：非有限浮点写 ``None``（仓库约定：禁 inf/nan）。"""
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _ratio(numerator, denominator):
    if not denominator:
        return None
    return float(numerator) / float(denominator)


def _percentiles(values):
    """分位数摘要；空样本返回 ``None`` 而不是 0（缺值不写 0）。"""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return None
    return {
        "trials": int(array.size),
        "p50": round(float(np.percentile(array, 50)), 3),
        "p95": round(float(np.percentile(array, 95)), 3),
        "mean": round(float(np.mean(array)), 3),
        "max": round(float(np.max(array)), 3),
    }


def _bucket(snr_db, width=SNR_BUCKET_DB):
    """实测 SNR → 分桶标签（如 ``-5~+0 dB``）。"""
    if snr_db is None:
        return None
    start = int(math.floor(float(snr_db) / width) * width)
    return f"{start:+d}~{start + int(width):+d} dB"


def _new_counts():
    return {"true": 0, "detected": 0, "matched": 0, "missed": 0, "false_alarm": 0}


def _add_counts(total, metrics):
    for key in total:
        total[key] += int(metrics[key])
    return total


def _pooled_metrics(counts, pairs, contract):
    """把逐记录指标汇总成一份指标（含参数误差的合并均值）。"""
    matched = counts["matched"]
    precision = _ratio(matched, counts["detected"])
    recall = _ratio(matched, counts["true"])
    if precision is None and recall is None:
        f1 = None
    else:
        safe_precision = 0.0 if precision is None else precision
        safe_recall = 0.0 if recall is None else recall
        f1 = (2 * safe_precision * safe_recall / (safe_precision + safe_recall)
              if (safe_precision + safe_recall) > 0 else 0.0)
    centers = [abs(pair["center_error_hz"]) for pair in pairs
               if pair.get("center_error_hz") is not None]
    centers_signed = [pair["center_error_hz"] for pair in pairs
                      if pair.get("center_error_hz") is not None]
    bandwidths = [pair["bandwidth_relative_error"] for pair in pairs
                  if pair.get("bandwidth_relative_error") is not None]
    snrs = [abs(pair["snr_error_db"]) for pair in pairs
            if pair.get("snr_error_db") is not None]
    return {
        "contract": contract,
        "true": counts["true"],
        "detected": counts["detected"],
        "matched": matched,
        "missed": counts["missed"],
        "false_alarm": counts["false_alarm"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "center_mae_hz": float(np.mean(centers)) if centers else None,
        "center_rmse_hz": (float(np.sqrt(np.mean(np.square(centers_signed))))
                           if centers_signed else None),
        "bandwidth_mape": float(np.mean(bandwidths)) if bandwidths else None,
        "snr_mae_db": float(np.mean(snrs)) if snrs else None,
        "pairs": matched,
    }


def _new_cell():
    return {"counts": _new_counts(), "records": 0, "pairs": [], "frames": 0,
            "duration_s": 0.0}


def _finish_cell(cell, tally, contract):
    metrics = _pooled_metrics(cell["counts"], cell["pairs"], contract)
    return {
        "records": cell["records"],
        "frames": cell["frames"],
        "duration_s": round(cell["duration_s"], 6),
        "full_path": metrics,
        "roc": tally,
    }


# ---------------------------------------------------------------------------
# 生成与单次推理
# ---------------------------------------------------------------------------


def scenario_spec(mode, power_dbfs=-10.0):
    """单信号场景（与 ``detect_sweep.py`` 同频点、同带宽、同功率）。"""
    return {"mode": mode, "power_dbfs": float(power_dbfs), **SCENARIOS[mode]}


def generate_case(mode, snr_db, seed, duration, power_dbfs=-10.0):
    """按 ``detect_sweep.py`` 的种子约定生成一条信号记录。"""
    return generate_iq(RATE_HZ, duration, [scenario_spec(mode, power_dbfs)],
                       noise={"snr_db": float(snr_db)}, seed=int(seed) + 1)


def generate_noise(seed, duration, power_dbfs=-10.0):
    """纯噪声记录（无信号），用于虚警率分母。"""
    return generate_iq(RATE_HZ, duration, [],
                       noise={"power_dbfs": float(power_dbfs)}, seed=1000 + int(seed))


def candidate_rows(arrays):
    """``ml_detect`` 返回的候选框 → ``(N, 6)`` 行（分数降序，已过门限与 NMS）。

    重放解码器需要**原始行**（``[x, y, w, h, score, class]``），而 ``ml_detect``
    只暴露候选框与分数；行列与类别列按契约补全（单类模型类别恒为 0）。
    """
    boxes = np.asarray(arrays.get("model_boxes", np.zeros((0, 4))), dtype=np.float64)
    boxes = boxes.reshape(-1, 4)
    scores = np.asarray(arrays.get("model_scores", np.zeros(0)), dtype=np.float64).reshape(-1)
    if boxes.shape[0] != scores.size:
        raise ValueError("候选框与置信度数量不一致，无法重放解码")
    if boxes.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float64)
    return np.column_stack([boxes, scores, np.zeros(scores.size, dtype=np.float64)])


def replay_candidates(rows, meta, config, labels):
    """按 ``ml_detect`` 的同一参数重放解码：用于自检与逐阈值 ROC。"""
    return boxes_to_bands(
        rows, meta,
        score_threshold=float(config["score_threshold"]),
        iou_threshold=float(config["iou_threshold"]),
        max_detections=max(64, 4 * int(config["max_detections"])),
        min_bandwidth_hz=float(config["min_bandwidth_hz"]),
        min_duration_s=float(config["min_duration_s"]),
        labels=list(labels),
    )


def roc_tally(truth, bands, scores, thresholds):
    """各阈值下的 ``(tp, fp, fn)``：复用评测器的 :func:`match_detections`。

    ``bands`` 为每个候选框经 :func:`box_to_band` 换算出的 ``(center_hz, bandwidth_hz)``。
    """
    tally = {}
    for threshold in thresholds:
        kept = [index for index in range(scores.size) if scores[index] >= threshold]
        detections = [{"id": index,
                       "center_hz": bands[index][0],
                       "bandwidth_hz": bands[index][1],
                       "confidence": float(scores[index])}
                      for index in kept]
        pairs = match_detections(truth, detections)
        tally[float(threshold)] = {"tp": len(pairs),
                                   "fp": len(detections) - len(pairs),
                                   "fn": len(truth) - len(pairs)}
    return tally


def _merge_tally(total, tally):
    for threshold, counts in tally.items():
        slot = total.setdefault(threshold, {"tp": 0, "fp": 0, "fn": 0})
        for key in ("tp", "fp", "fn"):
            slot[key] += int(counts[key])
    return total


# ---------------------------------------------------------------------------
# 扫掠主体
# ---------------------------------------------------------------------------


def run_signal_pass(args, runner, manifest, config, thresholds, emit=print):
    """信号记录扫掠：逐 (mode, nominal SNR, seed) 一次推理 + 全路径指标 + ROC 计数。"""
    labels = list(manifest.get("labels") or ["emitter"])
    cells, cell_tally, roc_total = {}, {}, {}
    latency = {stage: [] for stage in STAGES}
    per_snr = {}
    pooled = _new_counts()
    pooled_pairs = []
    selfcheck = {"shipped": 0, "replayed": 0, "mismatch_records": 0}
    for mode in args.modes:
        for snr in args.snr_grid:
            name = f"{mode}@{snr:g}"
            cell, raw, cell_total = _new_cell(), {}, []
            for seed in range(args.seeds):
                samples, generation = generate_case(mode, snr, seed, args.duration,
                                                    args.power_dbfs)
                truth = signal_truth(generation)
                started = time.perf_counter()
                summary, arrays = ml_detect(samples, RATE_HZ, config, runner=runner,
                                            with_baseline=False)
                wall_ms = (time.perf_counter() - started) * 1000.0
                metrics = evaluate_detections(truth, summary["detections"])
                _add_counts(cell["counts"], metrics)
                _add_counts(pooled, metrics)
                cell["pairs"].extend(metrics["pairs"])
                pooled_pairs.extend(metrics["pairs"])
                cell["records"] += 1
                cell["frames"] += int(summary["frame_count"])
                cell["duration_s"] += float(summary["duration_s"])
                # 实测带内 SNR 分档（真值口径：生成器摘要的 snr_inband_db）
                matched_truth = {pair["truth_index"] for pair in metrics["pairs"]}
                for index, entry in enumerate(truth):
                    label = _bucket(entry.get("snr_inband_db"))
                    if label is None:
                        continue
                    slot = per_snr.setdefault(label, {"true": 0, "matched": 0})
                    slot["true"] += 1
                    slot["matched"] += 1 if index in matched_truth else 0
                # 分级延迟：image/gate 为同输入纯函数重跑，其余取自 ml_detect 自身计时
                image_started = time.perf_counter()
                _, meta = detection_image(arrays, summary,
                                          int(summary["config"]["image_size"]),
                                          float(summary["config"]["dynamic_range_db"]))
                image_ms = (time.perf_counter() - image_started) * 1000.0
                rows = candidate_rows(arrays)
                gate_started = time.perf_counter()
                replayed = replay_candidates(rows, meta, summary["config"], labels)
                gate_ms = (time.perf_counter() - gate_started) * 1000.0
                selfcheck["shipped"] += int(summary["raw_boxes"]["candidates"])
                selfcheck["replayed"] += len(replayed)
                if len(replayed) != int(summary["raw_boxes"]["candidates"]):
                    selfcheck["mismatch_records"] += 1
                timing = summary["timing"]
                stages = {
                    "context_ms": float(timing["context_ms"]),
                    "image_ms": image_ms,
                    "inference_ms": float(timing["inference_ms"]),
                    "gate_ms": gate_ms,
                    "total_ms": float(timing["total_ms"]),
                }
                stages["other_ms"] = (stages["total_ms"] - sum(
                    stages[key] for key in ("context_ms", "image_ms", "inference_ms",
                                            "gate_ms")))
                for stage, value in stages.items():
                    latency[stage].append(value)
                cell_total.append(stages["total_ms"])
                # ROC：候选框换算频段后逐阈值重放打分
                boxes = rows[:, :4]
                scores = rows[:, 4]
                bands = [(0.5 * (band["f_low_hz"] + band["f_high_hz"]),
                          band["f_high_hz"] - band["f_low_hz"])
                         for band in (box_to_band(meta, *box) for box in boxes)]
                tally = roc_tally(truth, bands, scores, thresholds)
                _merge_tally(roc_total, tally)
                _merge_tally(raw, tally)
                if wall_ms <= 0:  # pragma: no cover - 防御性
                    raise RuntimeError("计时异常：单次推理耗时非正")
            cells[name] = _finish_cell(cell, raw, "detect_result_v1")
            cell_tally[name] = raw
            emit(f"  {name:<16s} 记录 {cell['records']}  检出 "
                 f"{cell['counts']['matched']}/{cell['counts']['true']}  虚警 "
                 f"{cell['counts']['false_alarm']}  平均 {np.mean(cell_total):7.1f} ms")
    return {"cells": cells, "cell_tally": cell_tally, "roc": roc_total,
            "latency": latency, "per_snr": per_snr, "pooled": pooled,
            "pooled_pairs": pooled_pairs, "selfcheck": selfcheck}


def run_noise_pass(args, runner, manifest, config, thresholds, emit=print):
    """纯噪声记录的虚警统计（FAR 分子/分母都以**汇总**口径累加）。"""
    tally = {}
    frames = seconds = 0
    per_record = []
    with_detection = 0
    for seed in range(args.noise_trials):
        samples, _ = generate_noise(seed, args.duration, args.noise_power_dbfs)
        summary, arrays = ml_detect(samples, RATE_HZ, config, runner=runner,
                                    with_baseline=False)
        count = int(summary["raw_boxes"]["candidates"])
        if count:
            with_detection += 1
        per_record.append(count)
        frames += int(summary["frame_count"])
        seconds += float(summary["duration_s"])
        # 纯噪声没有真值，任何门限后残留的候选框都算虚警；因此这里不需要
        # 频段换算（也就不需要再算一遍时频图），只按分数重新过门限即可。
        scores = candidate_rows(arrays)[:, 4]
        for threshold in thresholds:
            kept = int(np.count_nonzero(scores >= threshold))
            slot = tally.setdefault(float(threshold), {"fp": 0})
            slot["fp"] += kept
    emit(f"  纯噪声 {args.noise_trials} 条：{with_detection} 条出现候选框，"
         f"每帧/每秒虚警见报告（帧 {frames}，时长 {seconds:.2f} s）")
    return {"tally": tally, "frames": frames, "seconds": seconds,
            "records": args.noise_trials, "records_with_candidates": with_detection,
            "candidates_per_record": {
                "mean": float(np.mean(per_record)) if per_record else None,
                "max": int(np.max(per_record)) if per_record else None,
            }}


def roc_points(thresholds, signal_roc, noise_tally, frames, seconds):
    """逐阈值 ROC 点表（召回、精确率、每帧/每秒虚警率）。

    FAR 的分母是**整段扫掠**的帧数与时长（汇总口径）；信号侧的 tp/fp/fn 也同样是
    汇总口径，因此点表内部自洽、可直接跨阈值比较。
    """
    points = []
    for threshold in thresholds:
        counts = signal_roc.get(float(threshold), {"tp": 0, "fp": 0, "fn": 0})
        false_boxes = int(noise_tally.get(float(threshold), {"fp": 0})["fp"])
        points.append({
            "score": float(threshold),
            "tp": int(counts["tp"]),
            "fp": int(counts["fp"]),
            "fn": int(counts["fn"]),
            "recall": _ratio(counts["tp"], counts["tp"] + counts["fn"]),
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "noise_false_boxes": false_boxes,
            "far_per_frame": _ratio(false_boxes, frames),
            "far_per_second": _ratio(false_boxes, seconds),
        })
    return points


def choose_operating_point(points, target, basis):
    """给定目标虚警率，取满足 FAR ≤ 目标且**分数最高**的阈值（召回最高的合法工作点）。"""
    key = "far_per_frame" if basis == "frame" else "far_per_second"
    eligible = [point for point in points
                if point[key] is not None and point[key] <= float(target)]
    if not eligible:
        return None
    return max(point["score"] for point in eligible)


def tally_at(tally, score, thresholds):
    """取某个工作点阈值下的 ``(tp, fp, fn, truth)``：分数不在网格上时取就近的合法阈值。"""
    if score is None:
        return None
    threshold = min(thresholds, key=lambda value: (abs(value - score), value))
    counts = tally.get(float(threshold), {"tp": 0, "fp": 0, "fn": 0})
    return {"score": float(threshold), "tp": int(counts["tp"]), "fp": int(counts["fp"]),
            "fn": int(counts["fn"]), "truth": int(counts["tp"]) + int(counts["fn"])}


def breakdown(tally, score, thresholds, groups):
    """按分组（如 mode 或标称 SNR）给出某工作点下的召回与精确率。"""
    rows = {}
    for name, sub_tally in groups.items():
        counts = tally_at(sub_tally, score, thresholds)
        if counts is None:
            continue
        rows[name] = {
            "truth": counts["truth"],
            "tp": counts["tp"],
            "fp": counts["fp"],
            "recall": _ratio(counts["tp"], counts["truth"]),
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
        }
    return rows


def _sum_tally(tallies):
    """合并多份逐阈值计数（分组统计用：同 mode 的所有 SNR 单元等）。"""
    total = {}
    for tally in tallies:
        if tally:
            _merge_tally(total, tally)
    return total


def _group_tallies(cell_tally, modes, snr_grid):
    by_mode, by_snr = {}, {}
    for mode in modes:
        by_mode[mode] = _sum_tally(cell_tally.get(f"{mode}@{snr:g}") for snr in snr_grid)
    for snr in snr_grid:
        by_snr[f"{snr:g} dB"] = _sum_tally(cell_tally.get(f"{mode}@{snr:g}") for mode in modes)
    return by_mode, by_snr


# ---------------------------------------------------------------------------
# 环境与 AMC
# ---------------------------------------------------------------------------


def environment(threads=None):
    """延迟数字的元数据：CPU、核数、运行时版本——脱离这些不可引用。"""
    cpu_model = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover - 非 Linux
        cpu_model = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "onnxruntime": runtime_version(),
        "cpu_model": cpu_model or platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "threads": None if threads is None else int(threads),
    }


def amc_report(directory, split, model_path=None):
    """在**已有** AMC 数据集上评测调制识别（默认内置线性基线），给 Macro-F1 与分 SNR 准确率。

    数据集特征清单必须与 ``AMC_FEATURES`` 一致（由 ``train_amc.load_dataset`` 校验），
    因此这里的数字与训练/推理是同一口径；缺数据集时返回 ``None`` 并在报告里注明
    「未评测」，不写 0 也不猜。
    """
    if not directory:
        return None
    import train_amc  # 训练侧脚本（已在 import 时把 src 加入 sys.path）

    from signal_analysis.ml import amc

    card, records = train_amc.load_dataset(directory)
    subset = [record for record in records
              if (record.get("split") == "train") == (split == "train")]
    if not subset:
        raise SystemExit(f"数据集 {directory} 没有 {split} 划分，无法给出独立指标")
    model = dict(amc.load_model(model_path)) if model_path else dict(amc.load_default_model())
    metrics = amc.evaluate_model(model, subset)
    return {
        "dataset": str(Path(directory)),
        "dataset_created": card.get("created"),
        "dataset_seed": card.get("seed"),
        "split": split,
        "model": metrics.get("model_id"),
        "model_source": "file" if model_path else "builtin",
        "total": metrics.get("total"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "per_snr": metrics.get("per_snr"),
        "per_class": [{"label": row.get("label"), "support": row.get("support"),
                       "recall": row.get("recall"), "precision": row.get("precision"),
                       "f1": row.get("f1")} for row in (metrics.get("per_class") or [])],
        "confusion": metrics.get("confusion"),
    }


# ---------------------------------------------------------------------------
# 组装报告
# ---------------------------------------------------------------------------


def run_sweep(args, runner, manifest, emit=print):
    """跑完整套扫掠并返回报告字典（不写文件，便于测试直接消费）。"""
    semantics = manifest.get(LABEL_SEMANTICS_FIELD)
    if semantics is None:
        training = manifest.get("training")
        if isinstance(training, dict):
            semantics = training.get(LABEL_SEMANTICS_FIELD)
    semantics = semantics or DEFAULT_LABEL_SEMANTICS
    if semantics not in LABEL_SEMANTICS:
        raise SystemExit(f"清单声明的标签语义不被支持：{semantics!r}")
    if semantics != DEFAULT_LABEL_SEMANTICS:
        raise SystemExit(
            f"清单标签语义为 {semantics}，而本扫描按会话级契约 detect_result_v1 评分；\n"
            "逐跳模型请用逐跳口径的评测（fh_hops_v1），否则数字会被误读。")

    thresholds = sorted({float(value) for value in args.score_grid}
                        | {float(args.min_score)})
    below = [value for value in args.score_grid if value < float(args.min_score)]
    if below:
        raise SystemExit(f"--score-grid 中的阈值 {below} 低于 --min-score {args.min_score}；\n"
                         "候选框只有过了 --min-score 才会暴露出来，重放更低阈值没有意义。")

    config = {"score_threshold": float(args.min_score)}
    if args.nfft:
        config["nfft"] = int(args.nfft)
    labels = list(manifest.get("labels") or ["emitter"])
    emit(f"模型 {manifest.get('id', 'injected')}@{manifest.get('version', '')}  "
         f"图像 {manifest.get('input', {}).get('image_size', '?')}  "
         f"类别 {labels}")
    emit(f"扫掠 {len(args.modes)} 模式 × {len(args.snr_grid)} SNR × {args.seeds} 种子"
         f"（时长 {args.duration}s，候选门限 {args.min_score}，工作点阈值 {args.operating_score}）")

    signal = run_signal_pass(args, runner, manifest, config, thresholds, emit=emit)
    noise = run_noise_pass(args, runner, manifest, config, thresholds, emit=emit)
    points = roc_points(thresholds, signal["roc"], noise["tally"],
                        noise["frames"], noise["seconds"])

    by_mode, by_snr = _group_tallies(signal["cell_tally"], args.modes, args.snr_grid)
    per_snr_rows = {label: {"true": slot["true"], "matched": slot["matched"],
                            "recall": _ratio(slot["matched"], slot["true"])}
                    for label, slot in sorted(signal["per_snr"].items())}
    truth_total = sum(slot["true"] for slot in signal["per_snr"].values())
    matched_total = sum(slot["matched"] for slot in signal["per_snr"].values())

    operating = []
    for target in args.far_targets:
        score = choose_operating_point(points, target, args.far_basis)
        if score is None:
            operating.append({"target": float(target), "basis": args.far_basis,
                              "score": None,
                              "reason": "网格内没有满足目标虚警率的阈值："
                                        "请细化 --score-grid 或增加 --noise-trials"})
            continue
        counts = tally_at(signal["roc"], score, thresholds)
        point = next(item for item in points if item["score"] == score)
        operating.append({
            "target": float(target),
            "basis": args.far_basis,
            "score": score,
            "far_per_frame": point["far_per_frame"],
            "far_per_second": point["far_per_second"],
            "recall": _ratio(counts["tp"], counts["truth"]),
            "precision": _ratio(counts["tp"], counts["tp"] + counts["fp"]),
            "tp": counts["tp"], "fp": counts["fp"], "truth": counts["truth"],
            "per_mode": breakdown(signal["roc"], score, thresholds, by_mode),
            "per_snr": breakdown(signal["roc"], score, thresholds, by_snr),
        })

    deployed = tally_at(signal["roc"], float(args.operating_score), thresholds)
    report = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "truth_source": "synthetic",
        "model": {
            "id": manifest.get("id"),
            "version": manifest.get("version"),
            "labels": labels,
            "label_semantics": semantics,
            "image_size": (manifest.get("input") or {}).get("image_size"),
            "spectrogram_nfft": (manifest.get("input") or {}).get("spectrogram_nfft"),
            "dynamic_range_db": (manifest.get("input") or {}).get("dynamic_range_db"),
            "manifest_path": manifest.get("manifest_path"),
            "sha256": manifest.get("sha256"),
        },
        "environment": environment(args.threads),
        "settings": {
            "sample_rate_hz": RATE_HZ,
            "duration_s": float(args.duration),
            "power_dbfs": float(args.power_dbfs),
            "seeds": int(args.seeds),
            "modes": list(args.modes),
            "snr_grid": [float(value) for value in args.snr_grid],
            "snr_axis": "扫掠轴=标称 SNR；per_snr 分档=每目标实测 inband_snr_db（5 dB 桶）",
            "min_score": float(args.min_score),
            "operating_score": float(args.operating_score),
            "score_grid": thresholds,
            "far_targets": [float(value) for value in args.far_targets],
            "far_basis": args.far_basis,
            "noise_trials": int(args.noise_trials),
            "noise_power_dbfs": float(args.noise_power_dbfs),
            "scenarios": SCENARIOS,
        },
        "cells": signal["cells"],
        "per_snr": {
            "buckets": per_snr_rows,
            "total": {"true": truth_total, "matched": matched_total,
                      "recall": _ratio(matched_total, truth_total)},
            "basis": "实测 inband_snr_db（真值目标级）；口径 = cells.full_path，"
                     f"即候选门限 {args.min_score}（不是工作点阈值）",
        },
        "parameters": signal["cells"] and _pooled_metrics(
            signal["pooled"], signal["pooled_pairs"], "detect_result_v1"),
        "roc": {
            "thresholds": thresholds,
            "points": points,
            "gate": float(args.min_score),
            "note": "全部阈值（含候选门限本身）都在同一次推理的候选框上重放门限；"
                    "FAR 分母为整段扫掠的帧数/时长",
        },
        "operating_points": operating,
        "deployed_point": (None if deployed is None else {
            "score": float(args.operating_score),
            "recall": _ratio(deployed["tp"], deployed["truth"]),
            "precision": _ratio(deployed["tp"], deployed["tp"] + deployed["fp"]),
            "tp": deployed["tp"], "fp": deployed["fp"], "truth": deployed["truth"],
            "note": "来自 ROC 重放（阈值 = --operating-score）；与 cells.full_path "
                    f"（候选门限 {args.min_score}）不是同一个门限",
        }),
        "latency": {stage: _percentiles(signal["latency"][stage]) for stage in STAGES},
        "noise_only": {
            "records": noise["records"],
            "records_with_candidates": noise["records_with_candidates"],
            "mean_candidates_per_record": noise["candidates_per_record"]["mean"],
            "max_candidates_per_record": noise["candidates_per_record"]["max"],
            "frames": noise["frames"],
            "seconds": round(noise["seconds"], 6),
        },
        "candidate_selfcheck": {
            "shipped": signal["selfcheck"]["shipped"],
            "replayed": signal["selfcheck"]["replayed"],
            "mismatch_records": signal["selfcheck"]["mismatch_records"],
            "note": "重放解码器得到的候选数应与 ml_detect 内部一致；不一致说明 ROC 重放口径失效",
        },
        "amc": amc_report(args.amc_data, args.amc_split, args.amc_model),
        "notes": [
            "参数误差与 per-SNR 召回来自 evaluate_detections，未使用第二套公式。",
            "FAR 为汇总口径（总虚警框数 ÷ 总帧数/总时长），不是逐记录虚警率的平均。",
            "延迟数字的元数据见 environment；onnxruntime 为 null 表示本机未装（不可能跑到这里）。",
            "实采数据（SigMF）真值换算尚未实现，本报告全部为合成数据。",
        ],
    }
    return report, points, per_snr_rows


def _print_report(report, emit=print):
    settings = report["settings"]
    gate = report["roc"]["gate"]
    emit(f"\n各模式 × 标称 SNR 检出率 [%]（全路径，候选门限 {gate}）\n")
    emit("  mode      " + "".join(f"{snr:>8.0f}" for snr in settings["snr_grid"]))
    for mode in settings["modes"]:
        row = []
        for snr in settings["snr_grid"]:
            cell = report["cells"].get(f"{mode}@{snr:g}")
            recall = cell["full_path"]["recall"] if cell else None
            row.append(f"{recall * 100:>8.0f}" if recall is not None else "     n/a")
        emit(f"  {mode:<9s} " + "".join(row))
    pooled = report["parameters"] or {}
    emit(f"  合计：召回 {pooled.get('recall')}  精确率 {pooled.get('precision')}  "
         f"检出 {pooled.get('detected')}  虚警 {pooled.get('false_alarm')}"
         "  ← 只看召回会被高虚警率蒙蔽，必须连虚警一起看")
    emit("\n实测带内 SNR 分档召回：")
    for label, slot in report["per_snr"]["buckets"].items():
        emit(f"  {label:<14s} {slot['matched']}/{slot['true']}"
             f"  {slot['recall'] * 100 if slot['recall'] is not None else float('nan'):.1f}%")
    emit("\n参数误差（全路径合并）：")
    params = report["parameters"] or {}
    emit(f"  中心 MAE {params.get('center_mae_hz')} Hz  "
         f"带宽 MAPE {params.get('bandwidth_mape')}  "
         f"SNR MAE {params.get('snr_mae_db')} dB  配对 {params.get('pairs')}")
    emit("\n固定虚警率工作点：")
    for point in report["operating_points"]:
        if point.get("score") is None:
            emit(f"  FAR≤{point['target']:g}（{point['basis']}）：{point['reason']}")
            continue
        emit(f"  FAR≤{point['target']:g}（{point['basis']}）：阈值 {point['score']:.2f}  "
             f"每帧 {point['far_per_frame']:.2e}  每秒 {point['far_per_second']:.2e}  "
             f"召回 {(point['recall'] or 0) * 100:.1f}%  精确率 "
             f"{(point['precision'] or 0) * 100:.1f}%")
    emit("\n分级延迟 [ms]（p50 / p95）：")
    for stage in STAGES:
        stats = report["latency"][stage]
        if stats is None:
            continue
        emit(f"  {stage:<13s} {stats['p50']:>9.3f} / {stats['p95']:>9.3f}   "
             f"均值 {stats['mean']:.3f}（{stats['trials']} 次）")
    noise = report["noise_only"]
    emit(f"\n纯噪声 {noise['records']} 条：{noise['records_with_candidates']} 条出现候选框，"
         f"每帧平均 {noise['mean_candidates_per_record']}、最多 {noise['max_candidates_per_record']}")
    amc = report.get("amc")
    if amc:
        emit(f"\nAMC（{amc['split']} 划分，{amc['total']} 样本，模型 {amc['model']}）："
             f"准确率 {amc['accuracy']}  宏平均 F1 {amc['macro_f1']}")
    else:
        emit("\nAMC：未评测（未提供 --amc-data）")


def _number_list(text):
    """``--snr``/``--score-grid``/``--far-targets`` 的解析：直接给出浮点列表。

    参数在解析阶段就定型为列表，避免「同一属性先字符串后列表」的隐蔽陷阱，
    :func:`run_sweep` 也就能直接吃 ``_parse_args`` 的结果（测试可复现）。
    """
    values = [float(item) for item in str(text).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("列表不能为空")
    return values


def _mode_list(text):
    values = [item.strip() for item in str(text).split(",") if item.strip()]
    unknown = [value for value in values if value not in SCENARIOS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"未知模式：{', '.join(unknown)}；可用：{', '.join(MODES)}")
    return values or list(MODES)


#: 取值为「逗号分隔的浮点列表」的选项（负值写法需要预处理，见下）
LIST_OPTIONS = ("--snr", "--score-grid", "--far-targets")


def _absorb_negative_lists(argv):
    """``--snr -5,0,10`` → ``--snr=-5,0,10``。

    argparse 只把 ``-5`` 这种**合法负数**当作值，``-5,0,10`` 会被当成新的选项而报
    「expected one argument」。SNR 网格几乎总是从负值开始，所以这里按选项名显式
    合并（``--snr=-5,0,10`` 的等号写法本来就能用）。
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    merged, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        if (token in LIST_OPTIONS and index + 1 < len(tokens)
                and tokens[index + 1].startswith("-")):
            merged.append(f"{token}={tokens[index + 1]}")
            index += 2
            continue
        merged.append(token)
        index += 1
    return merged


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI 检测证据扫描（per-SNR 召回 / 固定虚警率 / 延迟）")
    parser.add_argument("--manifest", required=True, help="检测模型清单（training/train_yolox.py 输出）")
    parser.add_argument("--modes", dest="modes", type=_mode_list, default=list(MODES),
                        help=f"逗号分隔的模式，默认全部：{','.join(MODES)}")
    parser.add_argument("--snr", dest="snr_grid", type=_number_list, default=list(SNR_GRID),
                        help="逗号分隔的**标称** SNR 网格（dB），默认 "
                             + ",".join(f"{value:g}" for value in SNR_GRID))
    parser.add_argument("--seeds", type=int, default=3, help="每个单元的随机种子数")
    parser.add_argument("--duration", type=float, default=0.5, help="记录时长（秒）")
    parser.add_argument("--power-dbfs", type=float, default=-10.0, help="信号功率（dBFS）")
    parser.add_argument("--nfft", type=int, default=0,
                        help="STFT 点数（0 = 用清单声明的值；与清单冲突会直接报错）")
    parser.add_argument("--min-score", type=float, default=0.02,
                        help="推理时使用的候选门限（ROC 重放的最低阈值，默认 0.02）")
    parser.add_argument("--operating-score", type=float, default=0.25,
                        help="部署工作点阈值（默认 0.25，与 decode.DEFAULT_SCORE_THRESHOLD 一致）")
    parser.add_argument("--score-grid", dest="score_grid", type=_number_list,
                        default=[value / 100 for value in range(5, 100, 5)],
                        help="ROC 的阈值网格（必须 ≥ --min-score，默认 0.05…0.95 步长 0.05）")
    parser.add_argument("--far-targets", dest="far_targets", type=_number_list,
                        default=[1e-2, 1e-3],
                        help="目标虚警率（逗号分隔，单位=框数/帧或框数/秒）")
    parser.add_argument("--far-basis", choices=("frame", "second"), default="frame",
                        help="挑工作点用的虚警率口径，默认每帧（每秒同时报出）")
    parser.add_argument("--noise-trials", type=int, default=20, help="纯噪声记录条数（虚警分母）")
    parser.add_argument("--noise-power-dbfs", type=float, default=-10.0, help="纯噪声记录功率（dBFS）")
    parser.add_argument("--threads", type=int, default=None, help="onnxruntime 线程数（默认运行时自定）")
    parser.add_argument("--truth", choices=("synthetic", "sigmf"), default="synthetic",
                        help="真值来源；sigmf 需要实采数据与标注换算（尚未实现，会直接报错）")
    parser.add_argument("--amc-data", default="", help="可选：AMC 数据集目录（evaluate_model 评测）")
    parser.add_argument("--amc-model", default="", help="可选：AMC 模型 JSON（默认内置线性基线）")
    parser.add_argument("--amc-split", choices=("train", "val"), default="val",
                        help="AMC 评测划分，默认独立验证集")
    parser.add_argument("--json", default="", help="把完整报告写到该路径")
    return parser.parse_args(_absorb_negative_lists(argv))


def main(argv=None):
    args = _parse_args(argv)
    if args.truth == "sigmf":
        raise SystemExit(
            "尚未实现 SigMF 标注 → 项目真值的换算（缺少实采数据与对照样例）。\n"
            "现有记录里 read_sigmf 会返回原始 metadata（含 annotations），但\n"
            "core:freq_lower_edge/upper_edge 是射频绝对频率、还需结合 captures 的\n"
            "core:frequency 与 sample_start 才能落到基带时频坐标；在拿到真实录制之前\n"
            "不写这份没有样例可验证的换算，以免产出错误口径的指标。")
    if args.seeds < 1 or args.noise_trials < 1:
        raise SystemExit("--seeds 与 --noise-trials 都必须 ≥ 1")

    from signal_analysis.ml.runtime import load_runner

    runner, manifest, library = load_runner(args.manifest, threads=args.threads)
    print(f"清单 {args.manifest}\n模型 {library}\n"
          f"onnxruntime {runner.runtime_version}  "
          f"线程 {getattr(runner, 'threads', None)}\n")
    report, _, _ = run_sweep(args, runner, manifest)
    _print_report(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(_clean(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
