"""数据集读取与契约校验（``build_dataset.py`` 输出 ↔ 各框架原生格式）。

``train_yolox.py`` 与本包共用同一份校验逻辑，避免出现"训练脚本认这个数据集、
导出脚本不认"的分叉。校验项与推理端严格对齐：

* ``input_contract`` 必须是 ``tf_image_v1``；
* ``layout`` 必须是 ``time_frequency_grayscale_v1``；
* ``output_layout`` 必须是 ``normalized_boxes_v1``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:  # 与 train_yolox.py 的引导保持一致
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from signal_analysis.ml import IMAGE_LAYOUT, INPUT_CONTRACT, OUTPUT_LAYOUT  # noqa: E402


def load_dataset(root):
    """读取数据集卡片与样本索引，并校验与推理端一致的输入/输出契约。"""
    root = Path(root)
    card_path = root / "dataset.json"
    samples_path = root / "samples.jsonl"
    if not card_path.is_file() or not samples_path.is_file():
        raise SystemExit(f"{root} 不是有效的数据集目录（缺少 dataset.json 或 samples.jsonl）")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    contract = card.get("contract") or {}
    if contract.get("input_contract") != INPUT_CONTRACT:
        raise SystemExit(f"数据集输入契约不是 {INPUT_CONTRACT}，请重新构建数据集")
    if contract.get("layout") != IMAGE_LAYOUT:
        raise SystemExit(f"数据集图像排布不是 {IMAGE_LAYOUT}，与推理端不一致")
    if contract.get("output_layout") != OUTPUT_LAYOUT:
        raise SystemExit(f"数据集标签不是 {OUTPUT_LAYOUT}，与推理端不一致")
    records = []
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit("数据集为空")
    return card, records


def contract_of(card):
    """数据集卡片里的契约字段（含图像尺寸 / 训练图像参数三件套）。"""
    contract = dict(card.get("contract") or {})
    for key in ("image_size", "spectrogram_nfft", "dynamic_range_db", "labels"):
        if key not in contract:
            raise SystemExit(f"数据集契约缺少字段 {key}，请重新构建数据集")
    return contract


def scene_meta(record, card):
    """单个样本的坐标映射元数据（供 :func:`signal_analysis.ml.band_to_box` 使用）。

    只重建 ``band_to_box`` 真正用到的三个字段；其余字段（本底、动态范围）与
    标签几何无关，不需要从数据集里回读。
    """
    scene = record.get("scene") or {}
    contract = contract_of(card)
    rate = float(scene["rate_hz"])
    duration = float(scene["duration_s"])
    return {
        "size": int(contract["image_size"]),
        "sample_rate_hz": rate,
        "duration_s": duration,
        "t_start_s": 0.0,
        "t_end_s": duration,
    }


def split_records(records, split):
    """按 ``train`` / ``val`` 过滤样本；``split=None`` 表示全部。"""
    if split is None:
        return list(records)
    wanted = str(split).strip().lower()
    if wanted == "all":
        return list(records)
    selected = [record for record in records if str(record.get("split", "train")).lower() == wanted]
    if not selected:
        raise SystemExit(f"数据集中没有 split={wanted!r} 的样本")
    return selected
