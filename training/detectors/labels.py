"""把项目数据集导出成各检测框架的**原生标注格式**（YOLO txt / COCO json）。

两条硬约束
----------

1. **标签只能来自数据集里已经算好的 ``record["boxes"]``**
   （即 :func:`signal_analysis.ml.tensor.band_to_box` 的结果），
   本模块不允许再写第二份"真值 → 框"的公式。历史教训：一旦在转换脚本里
   顺手写 ``y = 1 - y``，框会整体上下翻转，而形状校验完全查不出来。
2. **图像是 8 位灰度 PNG**。数据集里存的是 ``float32`` 且取值 ``[0, 1]``
   （:func:`signal_analysis.ml.tensor.detection_image` 的"本底 + 动态范围"
   归一化），而 YOLO 系 / COCO 生态只吃常规图片格式。量化误差上限为
   ``0.5 / 255``，在 60 dB 动态范围下折合 **约 0.118 dB**——远小于本底起伏，
   但**必须写进模型清单的 notes**，否则"训练看到的图"与"推理看到的图"
   之间的偏差就成了没人知道的口径差。

两个格式的分工
--------------

``yolo``
    Ultralytics 习惯：``images/<split>/*.png`` + ``labels/<split>/*.txt`` +
    ``data.yaml``（含 ``channels: 1``）。
``coco``
    YOLOX / RT-DETR 习惯：``images/<split>/*.png`` +
    ``annotations/instances_<split>.json``。COCO 的 ``bbox`` 是
    ``[x_min, y_min, w, h]``（左上角原点、``y`` 向下），与本项目的
    ``y`` 方向一致，因此同样**不需要翻转**。

所有 PNG 编码都用标准库（``zlib`` + ``struct``）手写，不引入 Pillow 依赖。
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

#: 8 位量化的最大绝对误差（归一化到 [0, 1] 的灰度单位）
PNG_QUANTIZATION = 0.5 / 255.0

#: 默认写入清单 notes 的量化提示（动态范围 60 dB 时约 0.118 dB）
QUANTIZATION_NOTE = ("数据集图像以 8 位 PNG 交付，量化误差上限约 "
                     f"{PNG_QUANTIZATION * 60.0:.3f} dB（按 60 dB 动态范围折算）")

FORMATS = ("yolo", "coco")


def write_gray_png(path, image):
    """把 ``float32`` ``[0, 1]`` 灰度图写成 8 位灰度 PNG（纯标准库）。"""
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"灰度图应为二维数组，实际为 {array.shape}")
    data = np.ascontiguousarray(np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
    height, width = data.shape

    def chunk(tag, payload):
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + data[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    blob = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _stem(record):
    if not record.get("image"):
        raise SystemExit(f"样本 {record.get('index')} 缺少 image 字段，无法导出原生标注")
    return Path(record["image"]).stem


def _normalized_boxes(record):
    boxes = np.asarray(record.get("boxes") or [], dtype=np.float64)
    if boxes.size == 0:
        return boxes.reshape(0, 6)
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise SystemExit(f"样本 {record.get('index')} 的 boxes 不是 (N, 6)，实际为 {boxes.shape}")
    return boxes


def _to_corner(box, size):
    """归一化中心式 ``(cx, cy, w, h)`` → 像素左上角式 ``(x_min, y_min, w, h)``。"""
    x_min = (float(box[0]) - float(box[2]) / 2.0) * size
    y_min = (float(box[1]) - float(box[3]) / 2.0) * size
    return x_min, y_min, float(box[2]) * size, float(box[3]) * size


def _write_images(root, records, size, output):
    """把 ``.npy`` 灰度图转成 8 位 PNG，按 split 分目录；返回 (stem, split, path) 列表。"""
    root = Path(root)
    items = []
    for record in records:
        split = str(record.get("split", "train")).lower()
        stem = _stem(record)
        source = root / record["image"]
        if not source.is_file():
            raise SystemExit(f"数据集图像缺失：{source}")
        image = np.load(source)
        if image.shape != (size, size):
            raise SystemExit(f"{source} 形状为 {image.shape}，与数据集契约 {size}×{size} 不一致")
        target = Path(output) / "images" / split / f"{stem}.png"
        write_gray_png(target, image)
        items.append((stem, split, target))
    return items


def export_yolo(root, records, card, output, contract=None):
    """导出 Ultralytics 习惯的 YOLO 目录（``images/`` + ``labels/`` + ``data.yaml``）。"""
    from .dataset import contract_of

    contract = contract or contract_of(card)
    size = int(contract["image_size"])
    labels = list(contract["labels"])
    output = Path(output)
    items = _write_images(root, records, size, output)
    by_stem = {_stem(record): record for record in records}
    counts = {"images": 0, "boxes": 0}
    for stem, split, _ in items:
        lines = []
        for box in _normalized_boxes(by_stem[stem]):
            class_index = int(round(float(box[5])))
            lines.append(f"{class_index} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}")
            counts["boxes"] += 1
        path = output / "labels" / split / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts["images"] += 1
    names = ", ".join(f"'{name}'" for name in labels)
    yaml = (
        "# 由 training/detectors/labels.py 生成，请勿手工编辑\n"
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(labels)}\n"
        f"names: [{names}]\n"
        "channels: 1\n"
        f"# {QUANTIZATION_NOTE}\n"
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")
    return {"format": "yolo", "root": str(output), "splits": _split_counts(items), **counts}


def export_coco(root, records, card, output, contract=None):
    """导出 COCO json（YOLOX / RT-DETR 习惯）。"""
    from .dataset import contract_of

    contract = contract or contract_of(card)
    size = int(contract["image_size"])
    labels = list(contract["labels"])
    output = Path(output)
    items = _write_images(root, records, size, output)
    by_stem = {_stem(record): record for record in records}
    annotations_dir = output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    summary = {"format": "coco", "root": str(output), "splits": {}, "images": 0, "boxes": 0}
    for split in sorted({item[1] for item in items}):
        selected = [item for item in items if item[1] == split]
        images = []
        annotations = []
        for index, (stem, _, path) in enumerate(selected, start=1):
            images.append({"id": index, "file_name": str(path.relative_to(output)),
                           "width": size, "height": size, "split": split})
            for box in _normalized_boxes(by_stem[stem]):
                x_min, y_min, width, height = _to_corner(box, size)
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": int(round(float(box[5]))) + 1,
                    "bbox": [round(x_min, 4), round(y_min, 4), round(width, 4), round(height, 4)],
                    "area": round(width * height, 4),
                    "iscrowd": 0,
                })
        payload = {
            "info": {"description": "SimuSingal 检测数据集（由 training/detectors/labels.py 生成）",
                     "note": QUANTIZATION_NOTE},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [{"id": index + 1, "name": name, "supercategory": "signal"}
                           for index, name in enumerate(labels)],
        }
        path = annotations_dir / f"instances_{split}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["splits"][split] = len(images)
        summary["images"] += len(images)
        summary["boxes"] += len(annotations)
    (output / "README.txt").write_text(
        "本目录是 training/detectors/labels.py 生成的原生数据集。\n"
        f"{QUANTIZATION_NOTE}\n"
        "COCO 标注：annotations/instances_<split>.json，bbox 为 [x_min, y_min, w, h]（像素）。\n"
        "训练时把各框架的 --data-dir / data 指向本目录即可；不要再用 .npy 原图。\n",
        encoding="utf-8")
    return summary


def _split_counts(items):
    counts = {}
    for _, split, _ in items:
        counts[split] = counts.get(split, 0) + 1
    return counts


def export_dataset(root, records, card, output, fmt="yolo", contract=None):
    """按 ``fmt`` 导出原生数据集；未知格式时列出可用值。"""
    key = str(fmt).strip().lower()
    if key == "yolo":
        return export_yolo(root, records, card, output, contract)
    if key == "coco":
        return export_coco(root, records, card, output, contract)
    raise SystemExit(f"未知的数据集格式 {fmt!r}；可用取值：{'、'.join(FORMATS)}")
