"""把所选集合落成训练页可直接消费的数据集（方案 §7.4、§8：训练消费数据版本清单）。

训练页不再有任何“生成数据”入口，只剩两个来源：**所选集合**与**已有数据集**。本模块负责
第一个：对集合的检测/AMC 标注集构建一个不可变数据版本（``dataset_manifest_v1``，
按 ``origin_group_id`` 整组划分），再按清单把 IQ 实体化成训练脚本认识的目录：

* 检测 → ``AnnotationDataset`` 目录（``tf_image_v1`` 时频图 + 归一化边框，状态 ``reviewed``），
  标签粒度沿用标注集的 ``label_semantics``（逐跳标注集写 ``per_hop_v1``）；
* AMC → ``iq_dataset.json`` + ``iq_dataset.npz``（``iq_waveform_v1`` 窗口，只取已确认类别
  ``known``；未知与字典外的标签不进入训练，数量写进卡片）。

两者都只依赖 NumPy，与主程序同一个环境，作为后台任务运行。
"""
import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..algorithms.amc.iq_model import iq_waveform
from ..algorithms.dsp.image import detection_image, spectral_context
from ..data.annotations import AnnotationDataset, atomic_json, create_dataset, validate_boxes
from .progress import Reporter
from ..contracts.image import band_to_box
from ..contracts.iq import (CLASS_SET_A09, DEFAULT_IQ_SAMPLES, IQ_INPUT_CHANNELS, IQ_LAYOUT,
                            IQ_NORMALIZATION, IQ_WAVEFORM_CONTRACT)
from ..data.datasets import build_dataset_version, read_manifest

#: 训练用划分：检测/AMC 训练只用 train 与 val，不留 test（需要时由数据版本另建）。
TRAINING_FRACTIONS = {"train": 0.8, "val": 0.2}
TASK_LABELS = {"detection": "检测", "amc": "AMC"}


def _task_set(workspace, collection_id, task):
    workspace.get_collection(collection_id)
    sets = workspace.list_task_sets(collection_id, task)
    if not sets:
        raise ValueError(f"所选集合没有{TASK_LABELS[task]}标注集：请在“信号集合生成”里勾选"
                         "对应标注，或先为集合建立标注集并完成标注")
    return sets[-1]


def _check_splits(counts, what):
    if not counts.get("train") or not counts.get("val"):
        raise ValueError(f"划分后{what}的训练集或验证集为空（train={counts.get('train', 0)}，"
                         f"val={counts.get('val', 0)}）：所选集合样本太少，请先生成更多数据")


def export_detection(workspace, collection_id, *, image_size=1024, nfft=512, seed=0,
                     fractions=None, reporter=None, destination=None):
    """集合 → 检测 ``AnnotationDataset`` 目录；返回导出报告。"""
    task_set = _task_set(workspace, collection_id, "detection")
    built = build_dataset_version(workspace, task_set["id"],
                                  fractions=fractions or TRAINING_FRACTIONS, seed=seed)
    version = built["version"]
    _check_splits(json.loads(version["split_json"] or "{}"), "检测数据")
    by_asset = {}
    for row in read_manifest(workspace, version):
        by_asset.setdefault(row["asset_id"], []).append(row)
    semantics = task_set["label_semantics"] or "session_v1"
    root = Path(destination or Path(workspace.root) / "training" / "datasets" / uuid.uuid4().hex)
    reporter = reporter or Reporter(None)
    created_dataset = False
    try:
        dataset = create_dataset(root, size=image_size, nfft=nfft)
        created_dataset = True
        records, skipped = [], {"missing_band": 0}
        total = len(by_asset)
        for position, (asset_id, rows) in enumerate(by_asset.items()):
            asset, samples = workspace.load_samples(asset_id)
            rate = float(asset["sample_rate"])
            summary, arrays = spectral_context(samples, rate, {"nfft": nfft})
            image, meta = detection_image(arrays, summary, size=image_size,
                                          dynamic_range_db=dataset.card["contract"]
                                          ["dynamic_range_db"])
            boxes = []
            for row in rows:
                if row["negative"] or not row["include"]:
                    continue
                cut = row["extraction"]
                if cut.get("f_low_hz") is None or cut.get("f_high_hz") is None:
                    skipped["missing_band"] += 1
                    continue
                start = cut.get("sample_start")
                end = cut.get("sample_end")
                box = band_to_box(meta, cut["f_low_hz"], cut["f_high_hz"],
                                  0.0 if start is None else start / rate,
                                  meta["duration_s"] if end is None else end / rate)
                boxes.append([*box, 1.0, 0])
            relative = f"images/{position:06d}.npy"
            np.save(root / relative, image)
            records.append({
                "image": relative, "boxes": validate_boxes(boxes), "truth": [],
                "split": rows[0]["split"], "annotation_status": "reviewed",
                "source_asset_id": asset_id,
                "scene": {"rate_hz": rate, "duration_s": float(meta["duration_s"])}})
            reporter.emit(position + 1, total, f"导出检测数据 {position + 1}/{total}")
        card = dict(dataset.card)
        card["contract"] = {**card["contract"], "label_semantics": semantics}
        counts = {name: sum(1 for record in records if record["split"] == name)
                  for name in ("train", "val", "test")}
        card.update(sample_count=len(records), splits=counts,
                    source={"kind": "collection", "collection_id": collection_id,
                            "task_set_id": task_set["id"],
                            "dataset_version_id": version["id"]})
        atomic_json(root / "dataset.json", card)
        (root / "samples.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8")
        AnnotationDataset(root)  # 读回校验契约
    except BaseException:
        if created_dataset:
            shutil.rmtree(root, ignore_errors=True)
        raise
    return {"kind": "training_export", "task": "detection", "path": str(root),
            "dataset_version_id": version["id"], "samples": len(records), "splits": counts,
            "label_semantics": semantics, "skipped": skipped,
            "boxes": sum(len(record["boxes"]) for record in records)}


def export_iq(workspace, collection_id, *, samples=DEFAULT_IQ_SAMPLES, seed=0, fractions=None,
              reporter=None, destination=None):
    """集合 → AMC ``iq_dataset`` 目录（仅已确认的 A09 类别）。"""
    task_set = _task_set(workspace, collection_id, "amc")
    built = build_dataset_version(workspace, task_set["id"],
                                  fractions=fractions or TRAINING_FRACTIONS, seed=seed)
    version = built["version"]
    classes = json.loads(workspace.get_taxonomy(task_set["taxonomy_id"])["classes_json"])
    rows = list(read_manifest(workspace, version))
    reporter = reporter or Reporter(None)
    cache = {}
    waveforms, labels, splits, snrs, offsets, widths, rates = [], [], [], [], [], [], []
    skipped = {}
    for position, row in enumerate(rows):
        if row["asset_id"] not in cache:
            cache.clear()  # 清单按资产聚集不保证，但同一资产的行通常相邻；只缓存最近一个
            cache[row["asset_id"]] = workspace.load_samples(row["asset_id"])
        asset, data = cache[row["asset_id"]]
        cut = row["extraction"]
        start = int(cut.get("sample_start") or 0)
        end = int(cut.get("sample_end") or data.size)
        center = bandwidth = None
        if cut.get("f_low_hz") is not None and cut.get("f_high_hz") is not None:
            center = (cut["f_low_hz"] + cut["f_high_hz"]) / 2.0
            bandwidth = cut["f_high_hz"] - cut["f_low_hz"]
        try:
            tensor, meta = iq_waveform(data[start:end], asset["sample_rate"],
                                       center or 0.0, bandwidth, window_samples=samples)
        except ValueError as exc:
            skipped[str(exc)] = skipped.get(str(exc), 0) + 1
            continue
        version_row = workspace.get_target_version(row["target_version_id"])
        waveforms.append(tensor)
        labels.append(row["class_name"])
        splits.append(row["split"])
        snrs.append(np.nan if version_row.get("snr_db") is None else version_row["snr_db"])
        offsets.append(meta["offset_hz"])
        widths.append(meta["bandwidth_hz"])
        rates.append(float(asset["sample_rate"]))
        reporter.emit(position + 1, len(rows), f"导出 IQ 窗口 {position + 1}/{len(rows)}")
    if not waveforms:
        detail = "；".join(f"{reason}（{count} 条）" for reason, count in skipped.items())
        raise ValueError("没有可导出的 AMC 样本（需要已确认的 A09 类别，且频带内样本足够长）"
                         + (f"：{detail}" if detail else ""))
    counts = {name: splits.count(name) for name in ("train", "val", "test")}
    _check_splits(counts, "AMC 数据")
    root = Path(destination or Path(workspace.root) / "training" / "datasets" / uuid.uuid4().hex)
    root.mkdir(parents=True, exist_ok=False)
    try:
        np.savez(root / "iq_dataset.npz", waveforms=np.stack(waveforms).astype(np.float32),
                 labels=np.asarray(labels), split=np.asarray(splits),
                 source=np.asarray(["collection"] * len(labels)),
                 snr_db=np.asarray(snrs, dtype=np.float32), offset_hz=np.asarray(offsets),
                 bandwidth_hz=np.asarray(widths), sample_rate_hz=np.asarray(rates),
                 index=np.arange(len(labels), dtype=np.int64))
        card = {
            "generator_script": "signal_analysis.training_export",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": int(seed), "sample_count": len(labels),
            "store": {"file": "iq_dataset.npz", "format": "npz",
                      "arrays": ["waveforms", "labels", "split", "source", "snr_db",
                                 "offset_hz", "bandwidth_hz", "sample_rate_hz", "index"]},
            "contract": {"task": "amc_iq", "input_contract": IQ_WAVEFORM_CONTRACT,
                         "layout": IQ_LAYOUT, "channels": IQ_INPUT_CHANNELS,
                         "normalization": IQ_NORMALIZATION, "samples": int(samples),
                         "class_set": CLASS_SET_A09, "classes": list(classes),
                         "class_count": len(classes),
                         "note": "由信号集合的已确认 AMC 标签导出；类别顺序即模型输出下标顺序"},
            "splits": {"strategy": "origin_group", "train": counts["train"],
                       "val": counts["val"], "test": counts["test"]},
            "statistics": {"sources": {"collection": len(labels)},
                           "labels": {name: labels.count(name) for name in classes}},
            "source": {"kind": "collection", "collection_id": collection_id,
                       "task_set_id": task_set["id"], "dataset_version_id": version["id"]},
            "skipped": skipped,
        }
        atomic_json(root / "iq_dataset.json", card)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return {"kind": "training_export", "task": "iq", "path": str(root),
            "dataset_version_id": version["id"], "samples": len(labels), "splits": counts,
            "skipped": skipped}


def export_training_data(workspace, request):
    """``export_training_data`` 服务动作：``task`` 为 ``detection`` 或 ``iq``。"""
    started = time.monotonic()
    reporter = Reporter(request.get("job_dir"))
    task = request.get("task")
    if task == "detection":
        result = export_detection(workspace, request["collection_id"],
                                  image_size=int(request.get("image_size", 1024)),
                                  nfft=int(request.get("nfft", 512)),
                                  seed=int(request.get("seed", 0)), reporter=reporter)
    elif task == "iq":
        result = export_iq(workspace, request["collection_id"],
                           samples=int(request.get("samples", DEFAULT_IQ_SAMPLES)),
                           seed=int(request.get("seed", 0)), reporter=reporter)
    else:
        raise ValueError("训练数据导出任务应为 detection 或 iq")
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result
