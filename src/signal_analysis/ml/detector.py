"""AI 检测：时频图 → ONNX 推理 → 冻结契约 ``detect_result_v1``。

与传统能量检测的分工（这是两者能逐项对比的前提）：

* **判决**（有没有目标、目标在哪个时频块）来自网络；
* **辐射量与频带几何**（``power_dbfs``、``snr_db``、``bandwidth_hz`` 等）
  由 :func:`detect_signals` 的同一份 STFT 在同一口径（``inband_snr_v1``、
  低门限测带宽）下测量，与能量路径同公式；
* **结果契约**与传统路径完全一致，因此 :mod:`signal_analysis.evaluation`
  的匹配与误差统计无需改动就能同时评估两条路径。

同一次调用还会给出**基线**（``summary["baseline"]``，即能量检测的检出），
用于同一份数据上的并排对比（漏警/虚警、中心频率与带宽误差、耗时）。
"""

from __future__ import annotations

import time

import numpy as np

from .._numeric_common import (
    _SNR_FLOOR_DB,
    _binary_close,
    _occupied_span,
    _true_runs,
    validate_samples,
)

from .._numeric_energy import (
    _merge_sessions,
)

from .._numeric_hops import (
    _finalise_hops,
    _group_hop_sessions,
    _hop_config,
    _refine_hop_bands,
    _smooth_psd,
)
from .decode import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    boxes_to_bands,
    parse_model_output,
)
from .manifest import (
    DEFAULT_DYNAMIC_RANGE_DB,
    DEFAULT_LABEL_SEMANTICS,
    DEFAULT_NFFT,
    IMAGE_LAYOUT,
    LABEL_SEMANTICS,
    LABEL_SEMANTICS_FIELD,
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
#: 逐跳结果契约与算法名（与会话级 ``detect_result_v1`` 完全不同的消费方式：
#: 一跳一个实例，可用于 :func:`signal_analysis.evaluation.evaluate_detections`
#: 的逐跳口径评分）。
HOP_CONTRACT = "fh_hops_v1"
HOP_ALGORITHM = "ml_detect_hops"
#: 模型清单里标签语义的取值（见 :mod:`signal_analysis.ml.manifest`）
PER_HOP_SEMANTICS = "per_hop_v1"
# 传给 STFT/能量基线的配置项（能量检测器会自行校验取值范围）
CONTEXT_KEYS = ("nfft", "threshold_db", "band_threshold_db", "min_bandwidth_hz",
                "min_duration_s", "max_detections", "merge_bins")
# AI 路径特有的配置项
ML_KEYS = ("score_threshold", "iou_threshold", "dynamic_range_db", "image_size",
           "threads", "image_contract")
# 逐跳解码特有的配置项。取值范围不在本模块重复定义，而是交给
# ``_numeric_hops._hop_config`` —— 与 ``detect-hops`` 命令行完全同一套校验，
# 因此“AI 逐跳”与“传统逐跳”的参数含义与边界完全相同。
HOP_KEYS = ("max_hops", "min_dwell_s", "smooth_frames", "merge_bins",
            "min_bandwidth_hz", "max_gap_frames", "max_gap_bins")
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


def _resolve_settings(config, contract, extra_keys=()):
    """校验配置并与模型清单对齐，返回 ``(设置, 上下文配置)``。

    ``extra_keys`` 给出本模块不自行校验、但允许出现在配置里的额外键
    （逐跳解码用 :data:`HOP_KEYS`，由 ``_hop_config`` 校验）。默认空元组，
    :func:`ml_detect` 的行为不受影响。
    """
    settings = dict(config or {})
    unknown = sorted(set(settings) - set(CONTEXT_KEYS) - set(ML_KEYS) - set(extra_keys))
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
    """候选框 → 会话合并所需条目（功率、质心、时间区间、占用区间）。

    网络只负责定位：在框内用与能量路径相同的"低门限测带宽"公式
    （噪声底 + ``band_threshold_db``，闭运算合并后取包含谱峰的游程）
    得到实测频带，随后功率 / 带内信噪比 / 质心 / 占用带都在这条实测带上
    重测（:func:`measure_band`、:func:`_occupied_span` 与能量路径同一实现），
    因此 ``bandwidth_hz`` 等物理量与能量路径同口径（见算法文档 §4.6）。

    ``f_low_hz`` / ``f_high_hz`` 输出为实测频带界（与能量路径字段同义）；
    网络的原始框仍保留在候选与 ``arrays["model_boxes"]`` 中供追溯。
    """
    band = candidate["band"]
    frequency = np.asarray(arrays["frequency"], dtype=np.float64)
    resolution = float(energy_summary["freq_resolution_hz"])
    settings = energy_summary["config"]
    bins = band_bins(frequency, band["f_low_hz"], band["f_high_hz"], resolution)
    first, last = int(bins[0]), int(bins[-1])
    average_db = np.asarray(arrays["spectrum_db"], dtype=np.float64)[first:last + 1]
    noise_floor_db = float(np.asarray(arrays["noise_floor_db"]).reshape(-1)[0])
    edge = _binary_close(average_db > noise_floor_db + settings["band_threshold_db"],
                         settings["merge_bins"])
    runs = _true_runs(edge)
    if runs:
        peak = int(np.argmax(average_db))
        span = next((run for run in runs if run[0] <= peak <= run[1]), runs[0])
        measured_first, measured_last = first + span[0], first + span[1]
    else:
        measured_first, measured_last = first, last
    measured_band = {
        "f_low_hz": float(frequency[measured_first] - resolution / 2.0),
        "f_high_hz": float(frequency[measured_last] + resolution / 2.0),
        "t_start_s": band["t_start_s"],
        "t_end_s": band["t_end_s"],
    }
    measured = measure_band(arrays, energy_summary, measured_band,
                            threshold_db=settings["threshold_db"])
    average_linear = 10.0 ** (np.asarray(arrays["spectrum_db"], dtype=np.float64) / 10.0)
    occupancy_low, occupancy_high = _occupied_span(average_linear, measured_first, measured_last)
    return {
        "center_hz": 0.5 * (measured_band["f_low_hz"] + measured_band["f_high_hz"]),
        "centroid_hz": measured["centroid_hz"],
        "f_low_hz": measured_band["f_low_hz"],
        "f_high_hz": measured_band["f_high_hz"],
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
        # 频带几何已在 _merge_item 中重测为能量路径同款实测带，带宽即其宽度；
        # 网络的原始框只用于候选筛选，不进入物理量输出
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


def _label_semantics(manifest):
    """模型清单声明的标签语义；缺失时按会话级（兼容历史清单）。"""
    value = manifest.get(LABEL_SEMANTICS_FIELD)
    if value is None:
        training = manifest.get("training")
        if isinstance(training, dict):
            value = training.get(LABEL_SEMANTICS_FIELD)
    if value is None:
        return DEFAULT_LABEL_SEMANTICS
    if value not in LABEL_SEMANTICS:
        raise ValueError(f"模型清单的标签语义不被支持：{value!r}")
    return value


def _boxes_to_tracks(candidates, frequency, frame_time, resolution):
    """模型框 → :func:`_finalise_hops` 需要的轨道字典。

    Route B 的全部要点就在这里：网络只回答“一跳在时频图的哪一块”，所以每个
    框只需要映射回 ``_finalise_hops`` 用的两个几何量——``runs``（频点区间）与
    ``first_frame``/``last_frame``（帧区间）。驻留时间、功率、单跳占用带宽、
    带内信噪比随后都由 :func:`_finalise_hops` 与 :func:`_refine_hop_bands` 在
    **原始 PSD** 上重新测量，因此 AI 逐跳与传统逐跳的这些物理量同一口径，
    可以直接并排比较，复用数值模块的物理量测并保持传统判决行为不变。

    帧区间用帧中心落在框内来取（与 :func:`_refine_hop_bands` 同一约定），
    至少保留一帧——窄于一帧的框仍应产生一次测量机会，是否成立交给
    ``min_dwell_frames`` 过滤，而不是在这里静默丢弃。
    """
    tracks = []
    for index, candidate in enumerate(candidates, start=1):
        band = candidate["band"]
        bins = band_bins(frequency, band["f_low_hz"], band["f_high_hz"], resolution)
        if bins.size == 0:
            nearest = int(np.argmin(np.abs(frequency
                                           - 0.5 * (band["f_low_hz"] + band["f_high_hz"]))))
            bins = np.array([nearest], dtype=np.int64)
        first_frame = int(np.searchsorted(frame_time, band["t_start_s"], side="left"))
        last_frame = int(np.searchsorted(frame_time, band["t_end_s"], side="right")) - 1
        first_frame = max(0, min(first_frame, frame_time.size - 1))
        last_frame = max(first_frame, min(last_frame, frame_time.size - 1))
        tracks.append({
            "id": index,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "frames": [first_frame, last_frame],
            "runs": [(int(bins[0]), int(bins[-1]))],
            "transition_frames": 0,
            "confidence": candidate["confidence"],
            "label": candidate["label"],
            "box": candidate["box"],
        })
    return tracks


def _attach_model_scores(hops, tracks, frame_time, frequency):
    """把每个跳标注回产生它的候选框：``model_confidence`` 与 ``model_label``。

    ``_finalise_hops`` 会按最短驻留/最小带宽剔除一部分轨道（弱框更容易被剔除），
    返回的跳与轨道不是一一对应，所以这里按几何量重新配对：取帧区间重叠帧数最多
    的轨道，同分取频带中心最近的那个。这么做只影响“这一跳来自哪个框”的标注，
    物理量仍然全部来自原始 PSD 上的重测。

    ``model_confidence`` 回答的是“这一跳存在吗”（网络给出的分数，可直接用于筛弱
    候选），与 ``confidence``（由带内信噪比换算出来的量测置信度）不是同一件事，
    因此单列一个字段而不是复用 ``confidence``。
    """
    if not hops or not tracks or frame_time.size == 0:
        return
    last_index = frame_time.size - 1
    last_bin = frequency.size - 1
    spans = []
    for track in tracks:
        first_bin, stop_bin = track["runs"][0]
        first_bin = max(0, min(int(first_bin), last_bin))
        stop_bin = max(0, min(int(stop_bin), last_bin))
        spans.append((track["first_frame"], track["last_frame"],
                      0.5 * (frequency[first_bin] + frequency[stop_bin]), track))
    for hop in hops:
        low = int(np.searchsorted(frame_time, hop["t_start_s"], side="left"))
        high = int(np.searchsorted(frame_time, hop["t_end_s"], side="right")) - 1
        low = max(0, min(low, last_index))
        high = max(low, min(high, last_index))
        centre = 0.5 * (hop["f_low_hz"] + hop["f_high_hz"])
        best, best_key = None, None
        for first, stop, band_centre, track in spans:
            overlap = min(high, stop) - max(low, first) + 1
            key = (overlap, -abs(band_centre - centre))
            if best_key is None or key > best_key:
                best, best_key = track, key
        if best is not None:
            hop["model_confidence"] = float(best["confidence"])
            hop["model_label"] = best["label"]


def _multi_hop_frames(hops, frame_time):
    """同一帧上多于一个跳处于活动状态的帧数。

    传统通路里 ``transition_frames`` 统计的是“一帧上多于一条谱游程”的帧——
    即跳变帧或并发发射机。AI 通路不做逐帧游程链接，但同一件事可以照量：
    一跳就是一个时频块，所以“一帧上覆盖了多于一个跳”就是同一含义的统计量，
    两条通路的读数因此仍然可比（而不是留一个没有定义的数字）。
    """
    if frame_time.size == 0 or not hops:
        return 0
    counts = np.zeros(frame_time.size, dtype=np.int64)
    for item in hops:
        low = int(np.searchsorted(frame_time, item["t_start_s"], side="left"))
        high = int(np.searchsorted(frame_time, item["t_end_s"], side="right"))
        if high > low:
            counts[low:high] += 1
    return int(np.count_nonzero(counts > 1))


def _hop_reason(resolved, hops, resolvable, rate, hop_samples):
    """逐跳诚实性出口：可分辨门限的说明文案与 :func:`detect_hops` 相同。"""
    if not hops:
        return (f"没有一跳同时满足最小带宽 {resolved['min_bandwidth_hz']:.0f} Hz 与"
                f"最短驻留 {resolved['min_dwell_frames']} 帧（候选框为空或被驻留过滤全部剔除）；"
                "请放宽门限、降低置信度阈值或增大分析点数")
    if not resolvable:
        return (f"STFT 帧间距 {hop_samples / rate * 1000.0:.3f} ms、一跳至少"
                f"{resolved['min_dwell_frames']} 帧，跳速高于"
                f"{rate / hop_samples / resolved['min_dwell_frames']:.1f} Hz 时驻留不足、逐跳不可分辨")
    return None


def _hop_entries(hops):
    """逐跳测量 → ``fh_hops_v1`` 的 ``hops`` 条目（舍入口径与 :func:`detect_hops` 一致）。

    AI 通路额外带上 ``model_confidence`` / ``model_label``：传统通路没有模型，
    这两个键不出现（而不是填 0 或 None），界面与报告按“缺值”显示。
    """
    entries = []
    for item in hops:
        entry = {
            "id": item["id"],
            "session_id": item["session_id"],
            "center_hz": round(item["center_hz"], 3),
            "centroid_hz": round(item["centroid_hz"], 3),
            "bandwidth_hz": round(item["bandwidth_hz"], 3),
            "f_low_hz": round(item["f_low_hz"], 3),
            "f_high_hz": round(item["f_high_hz"], 3),
            "t_start_s": round(item["t_start_s"], 6),
            "t_end_s": round(item["t_end_s"], 6),
            "dwell_s": round(item["dwell_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(item["power_linear"], 1e-30))), 3),
            "snr_db": round(item["snr_db"], 3),
            "confidence": round(item["confidence"], 3),
            "frame_count": int(item["frame_count"]),
            "bin_count": int(item["bin_count"]),
            "band_nfft": int(item["band_nfft"]),
            "mask_f_low_hz": round(item["mask_f_low_hz"], 3),
            "mask_f_high_hz": round(item["mask_f_high_hz"], 3),
            "mask_bandwidth_hz": round(item["mask_bandwidth_hz"], 3),
        }
        if "model_confidence" in item:
            entry["model_confidence"] = round(float(item["model_confidence"]), 4)
            entry["model_label"] = item.get("model_label")
        entries.append(entry)
    return entries


def _session_entries(sessions):
    """逐跳会话 → ``fh_hops_v1`` 的 ``sessions`` 条目。"""
    entries = []
    for session in sessions:
        entries.append({
            "session_id": session["session_id"],
            "hop_count": int(session["hop_count"]),
            "sequence": [round(value, 3) for value in session["sequence"]],
            "hop_frequencies_hz": [round(value, 3) for value in session["hop_frequencies_hz"]],
            "channel_spacing_hz": (None if session["channel_spacing_hz"] is None
                                   else round(session["channel_spacing_hz"], 3)),
            "hop_span_hz": round(session["hop_span_hz"], 3),
            "hop_bandwidth_hz": round(session["hop_bandwidth_hz"], 3),
            "hop_period_s": (None if session["hop_period_s"] is None
                             else round(session["hop_period_s"], 6)),
            "hop_rate_hz": (None if session["hop_rate_hz"] is None
                            else round(session["hop_rate_hz"], 3)),
            "duty_cycle": (None if session["duty_cycle"] is None
                           else round(session["duty_cycle"], 4)),
            "dwell_median_s": round(session["dwell_median_s"], 6),
            "dwell_min_s": round(session["dwell_min_s"], 6),
            "dwell_max_s": round(session["dwell_max_s"], 6),
            "center_hz": round(session["center_hz"], 3),
            "bandwidth_hz": round(session["bandwidth_hz"], 3),
            "f_low_hz": round(session["f_low_hz"], 3),
            "f_high_hz": round(session["f_high_hz"], 3),
            "t_start_s": round(session["t_start_s"], 6),
            "t_end_s": round(session["t_end_s"], 6),
            "power_dbfs": round(float(10.0 * np.log10(max(session["power_linear"], 1e-30))), 3),
            "snr_db": round(session["snr_db"], 3),
            "session_detection_id": session["session_detection_id"],
        })
    return entries


def ml_detect_hops(samples, sample_rate, config=None, model=None, runner=None,
                   threads=None, with_sessions=True, with_traditional=True):
    """逐跳 AI 估计入口，返回 ``(summary, arrays)``（契约 ``fh_hops_v1``）。

    只接受 ``label_semantics = per_hop_v1`` 的模型。会话级模型（一段传输一个框）
    即使在这里调用也只会得到“一跳等于整段传输”的假结果——一个 8 跳的信号会被
    报成 1 跳——所以本函数直接报错并指出该走哪条路，而不是静默给出对不上的数字。

    与 :func:`ml_detect` 的分工（Route B）：

    * **判决**（一跳在时频图的哪一块）来自网络，模型只提供频带与粗略时间；
    * **辐射量**（驻留时间、功率、单跳占用带宽、带内信噪比）全部由
      :func:`_finalise_hops` / :func:`_refine_hop_bands` 在**原始 PSD** 上重新
      测量，所以 AI 逐跳与传统逐跳的这些量是同一口径的实测值、可直接对比；
    * **不经过会话合并**：``_merge_sessions`` 会把同一发射机的多跳并成一个会话，
      那是会话级契约该有的行为，逐跳结果必须绕开它。

    ``with_sessions`` 与 :func:`detect_hops` 同义：把 :func:`detect_signals` 的
    会话级检出放在 ``summary["baseline"]``，并按频带重叠把每个会话关联到对应检出
    （``session_detection_id``）；关掉就不再跑这段会话级能量检测。
    ``with_traditional`` 用**同一份逐跳配置**再跑一次 :func:`detect_hops`，结果
    放在 ``summary["traditional"]``（传统逐跳基线）。

    结果与 :func:`detect_hops` 的键集逐位对齐，因此渲染、表格与逐跳评分器
    （``evaluate_detections(..., contract=HOP_CONTRACT)``）无需任何改动；只有逐跳
    明细额外多出 ``model_confidence`` / ``model_label`` 两个键（模型分数），
    传统通路没有这两个键，界面与报告按“不适用”显示。
    """
    manifest = {}
    if runner is None:
        if not model:
            raise ValueError("请提供模型清单路径（model=...）或已加载的推理会话")
        runner, manifest, _ = load_runner(model, threads=threads)
        manifest = dict(manifest)
    else:
        manifest = dict(getattr(runner, "manifest", {}) or {})
    semantics = _label_semantics(manifest)
    if semantics != PER_HOP_SEMANTICS:
        raise ValueError(
            f"模型清单的标签语义是 {semantics}（一段传输一个框），不能用于逐跳估计"
            f"（本通路要求 {PER_HOP_SEMANTICS}）；"
            "请用 training/build_dataset.py --labels hop 重建数据集并重训，"
            "或改用 ml-detect（会话级检测）")
    contract = _contract_input(manifest)
    resolved, context_config = _resolve_settings(config, contract, extra_keys=HOP_KEYS)
    model_name = getattr(runner, "model_name", None) or manifest.get("id", "injected")
    settings = dict(config or {})
    hop_config = {key: settings[key] for key in HOP_KEYS if key in settings}

    started = time.perf_counter()
    energy_summary, arrays = spectral_context(samples, sample_rate, context_config)
    context_ms = (time.perf_counter() - started) * 1000.0
    rate = float(energy_summary["sample_rate_hz"])
    duration = float(energy_summary["duration_s"])
    nfft = int(energy_summary["nfft"])
    hop_samples = int(energy_summary["hop_samples"])
    # 逐跳配置的 nfft 必须与产出 PSD 的 STFT 网格一致：``_hop_config`` 会据此
    # 推出频点宽度与帧间距，用另一个值会让带宽/驻留的换算整体错位。
    hop_config["nfft"] = nfft
    hop_resolved = _hop_config(rate, duration, hop_config)
    resolution = float(energy_summary["freq_resolution_hz"])

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
        # 候选上限不能低于 max_hops，否则一跳一个框的模型会在解码阶段被截断
        max_detections=max(64, min(512, hop_resolved["max_hops"])),
        min_bandwidth_hz=resolved["min_bandwidth_hz"],
        min_duration_s=resolved["min_duration_s"],
        labels=labels,
    )

    x = validate_samples(samples)
    frame_time = np.asarray(arrays["frame_time"], dtype=np.float64)
    frequency = np.asarray(arrays["frequency"], dtype=np.float64)
    # 原始 PSD：detect_signals 的 arrays 里存的是 dB，逐跳测量需要线性功率
    psd = 10.0 ** (np.asarray(arrays["spectrogram_db"], dtype=np.float64) / 10.0)
    # 帧起始样点由帧中心反推（arrays 只保留中心时刻，舍入可无损还原整数索引）
    starts = np.rint(frame_time * rate - (nfft - 1) / 2.0).astype(np.int64)
    noise_floor_db = float(np.asarray(arrays["noise_floor_db"]).reshape(-1)[0])
    noise_linear = 10.0 ** (noise_floor_db / 10.0)

    tracks = _boxes_to_tracks(candidates, frequency, frame_time, resolution)
    hops = _finalise_hops(tracks, psd, frequency, starts, nfft, rate, duration,
                          noise_linear, hop_resolved)
    _refine_hop_bands(hops, x, rate, noise_linear, hop_resolved,
                      hop_resolved["threshold_db"], nfft)
    if len(hops) > hop_resolved["max_hops"]:
        hops.sort(key=lambda item: -item["power_linear"])
        hops = hops[:hop_resolved["max_hops"]]
    hops.sort(key=lambda item: (item["t_start_s"], item["center_hz"]))
    for index, item in enumerate(hops, start=1):
        item["id"] = index
        item["session_id"] = None
    _attach_model_scores(hops, tracks, frame_time, frequency)
    sessions = _group_hop_sessions(hops, noise_linear, duration, resolution,
                                   hop_samples / rate)
    transition_frames = _multi_hop_frames(hops, frame_time)
    dwell_limit_s = hop_resolved["min_dwell_frames"] * hop_samples / rate
    resolvable = bool(hops) and min(item["dwell_s"] for item in hops) >= dwell_limit_s

    hop_entries = _hop_entries(hops)
    session_entries = _session_entries(sessions)
    smoothed = _smooth_psd(psd, hop_resolved["smooth_frames"])
    summary = {
        "contract": HOP_CONTRACT,
        "algorithm": f"{HOP_ALGORITHM}:{model_name}",
        "snr_definition": SNR_DEFINITION,
        "frequency_reference": "baseband_offset",
        "sample_rate_hz": rate,
        "sample_count": int(x.size),
        "duration_s": duration,
        "nfft": nfft,
        "hop_samples": hop_samples,
        "frame_count": int(psd.shape[0]),
        "config": {
            "nfft": nfft,
            "threshold_db": hop_resolved["threshold_db"],
            "smooth_frames": hop_resolved["smooth_frames"],
            "min_bandwidth_hz": round(hop_resolved["min_bandwidth_hz"], 6),
            "min_dwell_s": round(hop_resolved["min_dwell_s"], 9),
            "merge_bins": hop_resolved["merge_bins"],
            "max_gap_frames": hop_resolved["max_gap_frames"],
            "max_gap_bins": hop_resolved["max_gap_bins"],
            "transition_ratio": hop_resolved["transition_ratio"],
            "max_hops": hop_resolved["max_hops"],
            "score_threshold": resolved["score_threshold"],
            "iou_threshold": resolved["iou_threshold"],
            "image_size": resolved["image_size"],
            "dynamic_range_db": resolved["dynamic_range_db"],
        },
        "freq_resolution_hz": resolution,
        "frame_interval_s": hop_samples / rate,
        "noise_floor_dbfs_per_hz": round(noise_floor_db, 3),
        "threshold_dbfs_per_hz": energy_summary["threshold_dbfs_per_hz"],
        "dwell_limit_s": dwell_limit_s,
        "hop_rate_limit_hz": 1.0 / dwell_limit_s,
        "resolvable": resolvable,
        "reason": _hop_reason(hop_resolved, hops, resolvable, rate, hop_samples),
        "transition_frames": transition_frames,
        "model": _model_info(runner, manifest),
        "image": {
            "layout": meta["layout"],
            "size": meta["size"],
            "db_floor": round(meta["db_floor"], 3),
            "db_ceiling": round(meta["db_ceiling"], 3),
        },
        "raw_boxes": {
            "output_shape": list(np.shape(output)),
            "rows": int(rows.shape[0]),
            "candidates": len(candidates),
            "score_threshold": resolved["score_threshold"],
        },
        "timing": {
            "context_ms": round(context_ms, 3),
            "inference_ms": round(inference_ms, 3),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "hops": hop_entries,
        "sessions": session_entries,
    }
    if with_sessions:
        summary["baseline"] = {
            "contract": energy_summary["contract"],
            "algorithm": energy_summary["algorithm"],
            "threshold_dbfs_per_hz": energy_summary["threshold_dbfs_per_hz"],
            "detections": energy_summary["detections"],
        }
        # 会话与会话级检出互链（取频带重叠最多的一条），与 detect_hops 同一规则
        for session in summary["sessions"]:
            best, best_overlap = None, 0.0
            for detection in energy_summary["detections"]:
                overlap = min(session["f_high_hz"], detection["f_high_hz"]) \
                    - max(session["f_low_hz"], detection["f_low_hz"])
                if overlap > best_overlap:
                    best, best_overlap = detection["id"], overlap
            session["session_detection_id"] = best
    if with_traditional:
        from ..core_api import detect_hops

        traditional, _ = detect_hops(x, rate, hop_config, False)
        summary["traditional"] = traditional

    hop_boxes = [[item["f_low_hz"], item["f_high_hz"], item["t_start_s"], item["t_end_s"]]
                 for item in hop_entries]
    arrays = dict(arrays)
    arrays.update({
        "spectrogram_db": (10.0 * np.log10(np.maximum(smoothed, 1e-30))).astype(np.float32),
        "spectrogram_raw_db": (10.0 * np.log10(np.maximum(psd, 1e-30))).astype(np.float32),
        "hop_boxes": np.asarray(hop_boxes, dtype=np.float64).reshape(-1, 4),
        "hop_id": np.array([item["id"] for item in hop_entries], dtype=np.int64),
        "hop_session_id": np.array([item["session_id"] for item in hop_entries], dtype=np.int64),
        "hop_snr_db": np.array([item["snr_db"] for item in hop_entries], dtype=np.float64),
        "hop_power_dbfs": np.array([item["power_dbfs"] for item in hop_entries],
                                   dtype=np.float64),
        "model_boxes": np.asarray([item["box"] for item in candidates],
                                  dtype=np.float64).reshape(-1, 4),
        "model_scores": np.asarray([item["confidence"] for item in candidates],
                                   dtype=np.float64),
        "model_labels": np.array([item["label"] for item in candidates], dtype=object),
        "image_size": np.array([meta["size"]], dtype=np.int64),
    })
    return summary, arrays


__all__ = ["CONTEXT_KEYS", "HOP_CONTRACT", "HOP_KEYS", "ML_KEYS", "ml_detect",
           "ml_detect_hops"]
