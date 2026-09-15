"""Editable detection datasets using the same normalized geometry as inference."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("数据文件路径不能超出数据集目录")
    return path


def validate_boxes(boxes, classes=1):
    result = []
    for box in boxes:
        values = np.asarray(box, dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("检测框应含 6 个有限数值")
        x, y, w, h, score, label = values
        if (w <= 0 or h <= 0 or x - w / 2 < -1e-7 or y - h / 2 < -1e-7
                or x + w / 2 > 1 + 1e-7 or y + h / 2 > 1 + 1e-7
                or not 0 <= score <= 1 or label != int(label) or not 0 <= label < classes):
            raise ValueError("检测框必须位于图内、面积大于零且类别有效")
        result.append([float(x), float(y), float(w), float(h), 1.0, int(label)])
    return result


class AnnotationDataset:
    """Edits are stored in one atomic overlay; source labels stay recoverable."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.card = json.loads((self.root / "dataset.json").read_text(encoding="utf-8"))
        contract = self.card.get("contract", {})
        if (contract.get("input_contract") != "tf_image_v1" or
                contract.get("layout") != "time_frequency_grayscale_v1" or
                contract.get("output_layout") != "normalized_boxes_v1"):
            raise ValueError("请选择本项目时频图检测数据集")
        self.size = int(contract["image_size"])
        self.labels = list(contract["labels"])
        if self.size < 64 or self.size > 4096 or not self.labels:
            raise ValueError("图像尺寸或类别字典无效")
        self.records = [json.loads(line) for line in
                        (self.root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        self.overlay_path = self.root / "annotations.json"
        self.overlay = (json.loads(self.overlay_path.read_text(encoding="utf-8"))
                        if self.overlay_path.exists() else {})

    def image(self, index):
        array = np.load(contained(self.root, self.records[index]["image"]), allow_pickle=False)
        if (array.shape != (self.size, self.size) or not np.isfinite(array).all()
                or array.min() < 0 or array.max() > 1):
            raise ValueError("时频图须为与契约尺寸一致的 [0,1] 有限灰度数组")
        return array

    def record(self, index):
        result = copy.deepcopy(self.records[index])
        result.update(self.overlay.get(str(index), {}))
        return result

    def save(self, index, boxes, split):
        if split not in ("train", "val", "test"):
            raise ValueError("未知数据划分")
        # A source recording is the grouping unit; one source must not cross splits.
        source = self.records[index].get("source_asset_id")
        if source:
            for other, record in enumerate(self.records):
                if other != index and record.get("source_asset_id") == source:
                    if self.record(other).get("split") != split:
                        raise ValueError("同一原始资产的样本必须处于同一数据划分")
        self.overlay[str(index)] = {"boxes": validate_boxes(boxes, len(self.labels)),
                                    "split": split, "annotation_status": "reviewed"}
        atomic_json(self.overlay_path, self.overlay)

    def snapshot(self, destination):
        records = [self.record(i) for i in range(len(self.records))]
        if not records:
            raise ValueError("数据集为空")
        if any(r.get("split") not in ("train", "val", "test") for r in records):
            raise ValueError("数据划分只允许 train / val / test")
        counts = {name: sum(r.get("split") == name for r in records)
                  for name in ("train", "val", "test")}
        if not counts["train"] or not counts["val"]:
            raise ValueError("请至少分配一个训练样本和一个验证样本")
        if any(r.get("annotation_status") == "pending" for r in records):
            raise ValueError("还有未确认的样本；无信号样本也需保存确认")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "images").mkdir()
        digest = hashlib.sha256()
        for index, record in enumerate(records):
            record["boxes"] = validate_boxes(record.get("boxes", []), len(self.labels))
            self.image(index)
            source = contained(self.root, record["image"])
            target = destination / "images" / f"{index:06d}.npy"
            shutil.copyfile(source, target)
            digest.update(target.read_bytes())
            record["image"] = str(target.relative_to(destination))
            # Human labels are distinct from the original generator truth.
            if str(index) in self.overlay:
                record["annotation_source"] = "manual"
            digest.update(json.dumps(record, sort_keys=True).encode())
        card = copy.deepcopy(self.card)
        card.update(sample_count=len(records), splits=counts,
                    annotation_snapshot={"source": str(self.root), "sha256": digest.hexdigest()})
        atomic_json(destination / "dataset.json", card)
        (destination / "samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        return destination


def create_dataset(root, size=1024, nfft=512):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "images").mkdir()
    atomic_json(root / "dataset.json", {"sample_count": 0, "splits": {}, "contract": {
        "input_contract": "tf_image_v1", "layout": "time_frequency_grayscale_v1",
        "output_layout": "normalized_boxes_v1", "image_size": size, "channels": 1,
        "spectrogram_nfft": nfft, "dynamic_range_db": 60.0, "labels": ["emitter"],
        "label_semantics": "session_v1"}})
    (root / "samples.jsonl").write_text("", encoding="utf-8")
    return AnnotationDataset(root)


def append_asset(root, workspace, asset_id):
    from .storage import Workspace
    from .ml.tensor import spectral_context, detection_image
    dataset = AnnotationDataset(root)
    if any(r.get("source_asset_id") == asset_id for r in dataset.records):
        raise ValueError("该资产已在数据集中，不能重复导入")
    asset, samples = Workspace(workspace).load_samples(asset_id)
    contract = dataset.card["contract"]
    summary, arrays = spectral_context(samples, asset["sample_rate"],
                                       {"nfft": contract["spectrogram_nfft"]})
    image, meta = detection_image(arrays, summary, size=dataset.size,
                                  dynamic_range_db=contract["dynamic_range_db"])
    relative = f"images/{uuid.uuid4().hex}.npy"
    np.save(dataset.root / relative, image)
    dataset.records.append({"image": relative, "boxes": [], "truth": [],
        "source_asset_id": asset_id, "annotation_status": "pending", "split": "train",
        "scene": {"rate_hz": asset["sample_rate"], "duration_s": meta["duration_s"]}})
    path = dataset.root / "samples.jsonl"
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r) + "\n" for r in dataset.records), encoding="utf-8")
    temporary.replace(path)
    return relative
