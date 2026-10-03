"""把训练集／验证集集合实体化成训练脚本认识的目录（训练页的数据来源）。

GUI 已不再提供「导出训练集」入口：worker 首阶段调用本模块，把
:func:`~signal_analysis.services.training_inputs.inputs_plan` 装配好的样本行写进本次运行
目录，随后由现有训练／验收脚本消费：

* 检测 → ``AnnotationDataset`` 目录（``tf_image_v1`` 时频图 + 归一化边框，状态 ``reviewed``），
  标签粒度沿用训练集标注集的 ``label_semantics``（逐跳标注集写 ``per_hop_v1``）；
* AMC → ``iq_dataset.json`` + ``iq_dataset.npz``（``iq_waveform_v1`` 窗口，只取已确认类别
  ``known``；未知与字典外的标签不进入训练，数量写进卡片）。

训练集行写 ``train``、验证集行写 ``val``，不产生 ``test`` 划分。只依赖 NumPy；产物是
本次运行的内部输入快照，不写入集合／数据版本表，也不作为用户可见的训练数据集。
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..algorithms.amc.iq_model import iq_waveform
from ..algorithms.dsp.image import detection_image, spectral_context
from ..contracts.image import band_to_box
from ..contracts.iq import (CLASS_SET_A09, DEFAULT_IQ_SAMPLES, IQ_INPUT_CHANNELS, IQ_LAYOUT,
                            IQ_NORMALIZATION, IQ_WAVEFORM_CONTRACT)
from ..data.annotations import AnnotationDataset, atomic_json, create_dataset, validate_boxes
from .progress import Reporter


def _source(plan):
    return {"kind": "collection",
            "train_collection_id": plan["train"]["collection_id"],
            "val_collection_id": plan["val"]["collection_id"],
            "task_set_ids": [plan["train"]["task_set"]["id"], plan["val"]["task_set"]["id"]]}


def _check_splits(counts, what):
    if not counts.get("train") or not counts.get("val"):
        raise ValueError(
            f"{what}的训练集或验证集为空（train={counts.get('train', 0)}，"
            f"val={counts.get('val', 0)}）：请检查两个集合的标注与成员")


def _group_by_asset(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["asset_id"], []).append(row)
    return grouped


def detection_snapshot(workspace, plan, *, image_size=1024, nfft=512, seed=0,
                       reporter=None, destination=None):
    """训练集／验证集集合 → 检测 ``AnnotationDataset`` 目录。"""
    semantics = plan["train"]["task_set"]["label_semantics"] or "session_v1"
    by_asset = _group_by_asset(plan["rows"])
    root = Path(destination)
    reporter = reporter or Reporter(None)
    created = False
    try:
        dataset = create_dataset(root, size=image_size, nfft=nfft)
        created = True
        records, skipped = [], {"missing_band": 0}
        total = len(by_asset)
        for position, (asset_id, rows) in enumerate(by_asset.items()):
            asset, samples = workspace.load_samples(asset_id)
            rate = float(asset["sample_rate"])
            summary, arrays = spectral_context(samples, rate, {"nfft": nfft})
            image, meta = detection_image(
                arrays, summary, size=image_size,
                dynamic_range_db=dataset.card["contract"]["dynamic_range_db"])
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
            reporter.emit(position + 1, total, f"渲染检测样本 {position + 1}/{total}")
        counts = {name: sum(1 for record in records if record["split"] == name)
                  for name in ("train", "val", "test")}
        _check_splits(counts, "检测数据")
        card = dict(dataset.card)
        card["contract"] = {**card["contract"], "label_semantics": semantics}
        card.update(sample_count=len(records), splits=counts, seed=int(seed),
                    source=_source(plan),
                    generator_script="signal_analysis.services.training_snapshot")
        atomic_json(root / "dataset.json", card)
        (root / "samples.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8")
        AnnotationDataset(root)  # 读回校验契约
    except BaseException:
        if created:
            shutil.rmtree(root, ignore_errors=True)
        raise
    return {"task": "detection", "path": str(root), "samples": len(records),
            "splits": counts, "label_semantics": semantics, "skipped": skipped,
            "boxes": sum(len(record["boxes"]) for record in records)}


def iq_snapshot(workspace, plan, *, samples=DEFAULT_IQ_SAMPLES, seed=0,
                reporter=None, destination=None):
    """训练集／验证集集合 → AMC ``iq_dataset`` 目录（仅已确认类别 ``known``）。"""
    classes = json.loads(workspace.get_taxonomy(
        plan["train"]["task_set"]["taxonomy_id"])["classes_json"])
    rows = list(plan["rows"])
    reporter = reporter or Reporter(None)
    cache = {}
    waveforms, labels, splits, snrs, offsets, widths, rates = [], [], [], [], [], [], []
    skipped = {}
    for position, row in enumerate(rows):
        if row["asset_id"] not in cache:
            cache.clear()  # 行按资产聚集不保证，只缓存最近一个资产
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
        reporter.emit(position + 1, len(rows), f"生成 IQ 窗口 {position + 1}/{len(rows)}")
    if not waveforms:
        detail = "；".join(f"{reason}（{count} 条）" for reason, count in skipped.items())
        raise ValueError("没有可训练的 AMC 样本（需要已确认的类别，且频带内样本足够长）"
                         + (f"：{detail}" if detail else ""))
    counts = {name: splits.count(name) for name in ("train", "val", "test")}
    _check_splits(counts, "AMC 数据")
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    try:
        np.savez(root / "iq_dataset.npz", waveforms=np.stack(waveforms).astype(np.float32),
                 labels=np.asarray(labels), split=np.asarray(splits),
                 source=np.asarray(["collection"] * len(labels)),
                 snr_db=np.asarray(snrs, dtype=np.float32), offset_hz=np.asarray(offsets),
                 bandwidth_hz=np.asarray(widths), sample_rate_hz=np.asarray(rates),
                 index=np.arange(len(labels), dtype=np.int64))
        card = {
            "generator_script": "signal_analysis.services.training_snapshot",
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
                         "note": "由训练集/验证集集合的已确认 AMC 标签生成；类别顺序即模型输出下标顺序"},
            "splits": {"strategy": "collection", "train": counts["train"],
                       "val": counts["val"], "test": counts["test"]},
            "statistics": {"sources": {"collection": len(labels)},
                           "labels": {name: labels.count(name) for name in classes}},
            "source": _source(plan),
            "skipped": skipped,
        }
        atomic_json(root / "iq_dataset.json", card)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return {"task": "iq", "path": str(root), "samples": len(labels), "splits": counts,
            "skipped": skipped}


def snapshot(workspace, plan, *, destination, image_size=1024, nfft=512,
             samples=DEFAULT_IQ_SAMPLES, seed=0, reporter=None):
    """按计划任务类型实体化训练输入；返回写入报告。"""
    if plan["task"] == "detection":
        return detection_snapshot(workspace, plan, image_size=image_size, nfft=nfft,
                                  seed=seed, reporter=reporter, destination=destination)
    return iq_snapshot(workspace, plan, samples=samples, seed=seed, reporter=reporter,
                       destination=destination)
