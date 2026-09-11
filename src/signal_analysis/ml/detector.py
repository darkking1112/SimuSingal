"""AI 检测：时频图 → ONNX 推理 → 冻结契约 ``detect_result_v1``。

与传统能量检测的分工（这是两者能逐项对比的前提）：

* **判决**（有没有目标、目标在哪个时频块）来自网络；
* **辐射量**（``power_dbfs``、``snr_db``）由 :func:`detect_signals` 的同一份
  STFT 在同一口径（``inband_snr_v1``）下测量；
* **结果契约**与传统路径完全一致，因此 :mod:`signal_analysis.evaluation`
  的匹配与误差统计无需改动就能同时评估两条路径。

同一次调用还会给出**基线**（``summary["baseline"]``，即能量检测的检出），
用于同一份数据上的并排对比（漏警/虚警、中心频率与带宽误差、耗时）。
"""

from __future__ import annotations

import time

import numpy as np

from .._numeric import _SNR_FLOOR_DB, _merge_sessions, _occupied_span
from .decode import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    boxes_to_bands,
    parse_model_output,
)
from .manifest import (
    DEFAULT_DYNAMIC_RANGE_DB,
    DEFAULT_NFFT,
    IMAGE_LAYOUT,
)
from .runtime import load_runner
from .tensor import (
    IMAGE_SIZE,
    _validate_dynamic_range,
    _validate_size,
    band_bins,
    detection_image,
    measure_band,
    spectral_context,
)

DETECTION_METHOD = "ml"
CONTRACT = "detect_result_v1"
SNR_DEFINITION = "inband_snr_v1"
# 传给 STFT/能量基线的配置项（能量检测器会自行校验取值范围）
CONTEXT_KEYS = ("nfft", "threshold_db", "band_threshold_db", "min_bandwidth_hz",
                "min_duration_s", "max_detections", "merge_bins")
# AI 路径特有的配置项
ML_KEYS = ("score_threshold", "iou_threshold", "dynamic_range_db", "image_size",
           "threads", "image_contract")
_NUMERIC_KEYS = ("nfft", "threshold_db", "band_threshold_db", "min_bandwidth_hz",
                 "min_duration_s", "max_detections", "merge_bins",
                 "score_threshold", "iou_threshold", "dynamic_range_db", "image_size")


def _number(settings, key, default=None, minimum=None, maximum=None):
    if key not in settings:
        return default
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"配置项 {key} 应为有限数值")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"配置项 {key} 不应小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"配置项 {key} 不应大于 {maximum}")
    return number


def _contract_input(manifest):
    """清单声明的输入契约；无清单（注入会话）时返回空字典。"""
    if isinstance(manifest, dict) and isinstance(manifest.get("input"), dict):
        return dict(manifest["input"])
    return {}


def _resolve_settings(config, contract):
    """校验配置并与模型清单对齐，返回 ``(设置, 上下文配置)``。"""
    settings = dict(config or {})
    unknown = sorted(set(settings) - set(CONTEXT_KEYS) - set(ML_KEYS))
    if unknown:
        raise ValueError(f"不支持的检测配置项：{'、'.join(unknown)}")
    # 与清单冲突的项直接报错：图像生成方式必须与训练时一致（数值一致性前提）
    for key, field, label in (("nfft", "spectrogram_nfft", "时频图 nfft"),
                              ("image_size", "image_size", "输入图像尺寸"),
                              ("dynamic_range_db", "dynamic_range_db", "图像动态范围")):
        if key in settings and field in contract and float(settings[key]) != float(contract[field]):
            raise ValueError(f"{label}与模型清单不一致（配置 {settings[key]:g}，清单 {contract[field]:g}）；"
                             "请重新生成清单或使用清单声明的取值")
    for key in _NUMERIC_KEYS:
        _number(settings, key)
    size = _validate_size(settings.get("image_size", contract.get("image_size", IMAGE_SIZE)))
    nfft = int(settings.get("nfft", contract.get("spectrogram_nfft", DEFAULT_NFFT)))
    dynamic_range = _validate_dynamic_range(
        settings.get("dynamic_range_db", contract.get("dynamic_range_db", DEFAULT_DYNAMIC_RANGE_DB)))
    if settings.get("image_contract", IMAGE_LAYOUT) != contract.get("layout", IMAGE_LAYOUT):
        raise ValueError("图像排布与模型清单不一致，请检查时频图契约")
    score_threshold = _number(settings, "score_threshold", DEFAULT_SCORE_THRESHOLD,
                              minimum=0.0, maximum=0.999)
    iou_threshold = _number(settings, "iou_threshold", DEFAULT_IOU_THRESHOLD,
                            minimum=0.0, maximum=1.0)
    max_detections = int(_number(settings, "max_detections", 32, minimum=1, maximum=256))
    resolved = {
        "image_size": size,
        "nfft": nfft,
        "dynamic_range_db": dynamic_range,
        "score_threshold": float(score_threshold),
        "iou_threshold": float(iou_threshold),
        "min_bandwidth_hz": float(_number(settings, "min_bandwidth_hz", 0.0, minimum=0.0)),
        "min_duration_s": float(_number(settings, "min_duration_s", 0.0, minimum=0.0)),
        "max_detections": max_detections,
        "threshold_db": float(_number(settings, "threshold_db", 3.0, minimum=0.0, maximum=80.0)),
        "threads": settings.get("threads"),
    }
    context_config = {key: settings[key] for key in CONTEXT_KEYS if key in settings}
    context_config["nfft"] = nfft
    return resolved, context_config


def _model_info(runner, manifest):
    info = {
        "id": manifest.get("id", getattr(runner, "model_name", "injected")),
        "version": manifest.get("version", ""),
        "manifest_path": manifest.get("manifest_path"),
        "sha256": manifest.get("sha256"),
        "library": manifest.get("library"),
        "labels": list(manifest.get("labels") or ["emitter"]),
        "training": manifest.get("training") or {},
        "runtime_version": getattr(runner, "runtime_version", None),
    }
    return info


def _merge_item(candidate, arrays, energy_summary):
    """候选框 → 会话合并所需条目（功率、质心、时间区间、占用区间）。"""
    band = candidate["band"]
    measured = measure_band(arrays, energy_summary, band,
                            threshold_db=energy_summary["config"]["threshold_db"])
    frequency = np.asarray(arrays["frequency"], dtype=np.float64)
    resolution = float(energy_summary["freq_resolution_hz"])
    bins = band_bins(frequency, band["f_low_hz"], band["f_high_hz"], resolution)
    average_linear = 10.0 ** (np.asarray(arrays["spectrum_db"], dtype=np.float64) / 10.0)
    first, last = int(bins[0]), int(bins[-1])
    occupancy_low, occupancy_high = _occupied_span(average_linear, first, last)
    return {
        "center_hz": 0.5 * (band["f_low_hz"] + band["f_high_hz"]),
        "centroid_hz": measured["centroid_hz"],
        "f_low_hz": band["f_low_hz"],
        "f_high_hz": band["f_high_hz"],
        "power_linear": measured["power_linear"],
        "intervals": measured["intervals"],
        "active_seconds": measured["active_seconds"],
        "t_start_s": band["t_start_s"],
        "t_end_s": band["t_end_s"],
        "bin_count": measured["bin_count"],
        "sub_bands": 1,
        "occupied_f_low_hz": float(frequency[occupancy_low] - resolution / 2.0),
        "occupied_f_high_hz": float(frequency[occupancy_high] + resolution / 2.0),
        "confidence": candidate["confidence"],
        "label": candidate["label"],
        "box": candidate["box"],
    }


def _group_sources(groups, items):
    """分组 → 模型输出信息（取组内最高置信度候选），与 ``groups`` 同序。"""
    sources = []
    for group in groups:
        members = [item for item in items
                   if item["f_low_hz"] >= group["f_low_hz"] - 1e-6
                   and item["f_high_hz"] <= group["f_high_hz"] + 1e-6]
        sources.append(max(members, key=lambda item: item["confidence"]) if members
                       else {"confidence": 0.0, "label": "emitter", "box": None})
    return sources


def _finalise(groups, items, noise_floor_db, model_name, labels):
    """合并后的分组 → 冻结契约的检测条目（排序、编号、四舍五入）。

    ``items`` 为合并前的模型输出条目，用于恢复置信度与类别：会话合并
    （``_combine_sessions`` 只保留测量量）后取组内最高置信度候选。
    排序在取元数据之前完成，保证两者一一对应。
    """
    groups.sort(key=lambda item: item["center_hz"])
    sources = _group_sources(groups, items)
    noise_linear = 10.0 ** (noise_floor_db / 10.0)
    detections = []
    boxes = []
    for index, item in enumerate(groups, start=1):
        source = sources[index - 1]
        bandwidth = item["f_high_hz"] - item["f_low_hz"]
        noise_band = max(noise_linear * bandwidth, 1e-30)
        signal_power = max(item["power_linear"] - noise_band, 1e-30)
        snr_db = float(max(10.0 * np.log10(signal_power / noise_band), _SNR_FLOOR_DB))
        hopping = item["sub_bands"] > 1
        detections.append({
            "id": index,
            "method": DETECTION_METHOD,
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
            "confidence": round(float(np.clip(source["confidence"], 0.0, 1.0)), 4),
            "hopping": hopping,
            "sub_bands": int(item["sub_bands"]),
            "bin_count": int(item["bin_count"]),
            "occupied_f_low_hz": round(item["occupied_f_low_hz"], 3),
            "occupied_f_high_hz": round(item["occupied_f_high_hz"], 3),
            "label": source["label"],
            "model": model_name,
        })
        boxes.append([item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]])
    return detections, boxes


def ml_detect(samples, sample_rate, config=None, model=None, runner=None,
              threads=None, with_baseline=True):
    """AI 检测入口，返回 ``(summary, arrays)``。

    * ``model``：模型清单路径（``plugin.json`` 风格），会校验摘要并创建会话；
    * ``runner``：已加载的会话（可注入，便于测试或复用已加载模型）；
    * ``config``：检测配置，见 :data:`CONTEXT_KEYS`/:data:`ML_KEYS`，
      与清单声明的图像生成方式冲突时报错而不是静默改变输入。
    """
    manifest = {}
    if runner is None:
        if not model:
            raise ValueError("请提供模型清单路径（model=...）或已加载的推理会话")
        runner, manifest, _ = load_runner(model, threads=threads)
        manifest = dict(manifest)
    else:
        manifest = dict(getattr(runner, "manifest", {}) or {})
    contract = _contract_input(manifest)
    resolved, context_config = _resolve_settings(config, contract)
    model_name = getattr(runner, "model_name", None) or manifest.get("id", "injected")

    started = time.perf_counter()
    energy_summary, arrays = spectral_context(samples, sample_rate, context_config)
    context_ms = (time.perf_counter() - started) * 1000.0
    image, meta = detection_image(arrays, energy_summary, resolved["image_size"],
                                  resolved["dynamic_range_db"])
    inference_started = time.perf_counter()
    output = runner.run(image)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    rows, output_shape = parse_model_output(output)
    labels = list(manifest.get("labels") or ["emitter"])
    candidates = boxes_to_bands(
        rows, meta,
        score_threshold=resolved["score_threshold"],
        iou_threshold=resolved["iou_threshold"],
        max_detections=max(64, 4 * resolved["max_detections"]),
        min_bandwidth_hz=resolved["min_bandwidth_hz"],
        min_duration_s=resolved["min_duration_s"],
        labels=labels,
    )
    items = [_merge_item(candidate, arrays, energy_summary) for candidate in candidates]
    groups = _merge_sessions(items, resolved["max_detections"])
    noise_floor_db = float(np.asarray(arrays["noise_floor_db"]).reshape(-1)[0])
    detections, boxes = _finalise(groups, items, noise_floor_db, model_name, labels)

    resolution = float(energy_summary["freq_resolution_hz"])
    summary = {
        "contract": CONTRACT,
        "algorithm": f"ml_detect:{model_name}",
        "snr_definition": SNR_DEFINITION,
        "frequency_reference": "baseband_offset",
        "sample_rate_hz": energy_summary["sample_rate_hz"],
        "sample_count": energy_summary["sample_count"],
        "duration_s": energy_summary["duration_s"],
        "nfft": energy_summary["nfft"],
        "hop_samples": energy_summary["hop_samples"],
        "frame_count": energy_summary["frame_count"],
        "config": {
            "nfft": resolved["nfft"],
            "threshold_db": energy_summary["config"]["threshold_db"],
            "band_threshold_db": energy_summary["config"]["band_threshold_db"],
            "min_bandwidth_hz": resolved["min_bandwidth_hz"],
            "min_duration_s": resolved["min_duration_s"],
            "max_detections": resolved["max_detections"],
            "merge_bins": energy_summary["config"]["merge_bins"],
            "score_threshold": resolved["score_threshold"],
            "iou_threshold": resolved["iou_threshold"],
            "image_size": resolved["image_size"],
            "dynamic_range_db": resolved["dynamic_range_db"],
        },
        "freq_resolution_hz": resolution,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": energy_summary["threshold_dbfs_per_hz"],
        "model": _model_info(runner, manifest),
        "image": {
            "layout": meta["layout"],
            "size": meta["size"],
            "db_floor": round(meta["db_floor"], 3),
            "db_ceiling": round(meta["db_ceiling"], 3),
        },
        "raw_boxes": {
            "output_shape": list(output_shape),
            "rows": int(rows.shape[0]),
            "candidates": len(candidates),
            "score_threshold": resolved["score_threshold"],
        },
        "timing": {
            "context_ms": round(context_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "detections": detections,
    }
    if with_baseline:
        summary["baseline"] = {
            "algorithm": energy_summary["algorithm"],
            "threshold_dbfs_per_hz": energy_summary["threshold_dbfs_per_hz"],
            "detections": energy_summary["detections"],
        }
    arrays = dict(arrays)
    arrays["detection_boxes"] = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    arrays["detection_id"] = np.array([item["id"] for item in detections], dtype=np.int64)
    arrays["detection_snr_db"] = np.array([item["snr_db"] for item in detections], dtype=np.float64)
    arrays["detection_power_dbfs"] = np.array([item["power_dbfs"] for item in detections],
                                              dtype=np.float64)
    arrays["model_boxes"] = np.asarray([item["box"] for item in candidates],
                                       dtype=np.float64).reshape(-1, 4)
    arrays["model_scores"] = np.asarray([item["confidence"] for item in candidates],
                                        dtype=np.float64)
    arrays["model_labels"] = np.array([item["label"] for item in candidates], dtype=object)
    baseline_boxes = ([[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                       for item in energy_summary["detections"]] if with_baseline else [])
    arrays["baseline_detection_boxes"] = np.asarray(baseline_boxes, dtype=np.float64).reshape(-1, 4)
    arrays["image_size"] = np.array([meta["size"]], dtype=np.int64)
    return summary, arrays


__all__ = ["CONTEXT_KEYS", "ML_KEYS", "ml_detect"]
