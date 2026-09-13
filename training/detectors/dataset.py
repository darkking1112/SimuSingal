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

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:  # 与 train_yolox.py 的引导保持一致
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from signal_analysis.ml import IMAGE_LAYOUT, INPUT_CONTRACT, OUTPUT_LAYOUT  # noqa: E402

#: 标签语义：``session_v1`` = 一段传输一个框；``per_hop_v1`` = ``fh*`` 一跳一个框。
#: 旧数据集没有该字段，按 ``session_v1`` 处理（向后兼容）。
LABEL_SEMANTICS = ("session_v1", "per_hop_v1")
DEFAULT_LABEL_SEMANTICS = "session_v1"

#: 频率轴重叠判据的容差：与数据集不变量测试
#: （``test_labels_stay_inside_image_and_do_not_overlap``）用的容差一致，
#: 避免"测试说重叠、构建说没重叠"的错位
OVERLAP_EPSILON = 1e-6


def frequency_overlap_pairs(boxes):
    """频率轴上互相重叠的标签框下标对。

    训练脚本把标签网格化时每格只保留最大的一个框，重叠的框会被**静默丢弃**；
    因此"框在频率轴上互不重叠"是训练集必须满足的硬条件。两个产生标签的地方
    （``build_dataset.py`` 的项目生成器、``ingest_torchsig.py`` 的第三方 IQ 转写）
    都用这一个函数判断，判据就不会出现两份。

    框按频率维下沿排序后只需比较相邻两项：任意一对重叠必然存在相邻的一对重叠。
    """
    spans = sorted(((float(box[1]) - float(box[3]) / 2.0,
                     float(box[1]) + float(box[3]) / 2.0, index)
                    for index, box in enumerate(boxes)), key=lambda item: item[0])
    pairs = []
    for (_, previous_high, previous_index), (next_low, _, next_index) in zip(spans, spans[1:]):
        if next_low < previous_high - OVERLAP_EPSILON:
            pairs.append((previous_index, next_index))
    return pairs


#: 两轴各自重叠超过较小框的这个比例，才算"互相顶掉"。
#: 标签按 6 位小数落盘，而合法的相邻跳在时间上首尾相接（跳频回到同一子带时
#: 甚至同频），圆整后会出现 ~1e-7 的"贴边"重叠——实测 14 个 fh_rc 逐跳样本里
#: 有 5 对这样的框。真正的碰撞重叠量比它大好几个数量级，这里用相对比例
#: （与图像尺寸无关）把两者分开。
COLLISION_RATIO = 0.05


def _axis_overlap(left_center, left_extent, right_center, right_extent):
    """两个区间在某一轴上的重叠长度（不重叠时为负值，调用方自行判定）。"""
    left_low = float(left_center) - float(left_extent) / 2.0
    left_high = float(left_center) + float(left_extent) / 2.0
    right_low = float(right_center) - float(right_extent) / 2.0
    right_high = float(right_center) + float(right_extent) / 2.0
    return min(left_high, right_high) - max(left_low, right_low)


def shadowed_pairs(boxes):
    """**两个轴都实质性重叠**的框对：逐跳标签里真正会被网格编码顶掉的那一对。

    与 :func:`frequency_overlap_pairs` 的区别很重要，两个函数各自对应一种
    标签粒度：

    * **会话级**（``session_v1``）每段传输盖满整帧（``width == 1.0``），所有框
      的列相同，于是"频率轴重叠"与"同格"完全等价——用
      :func:`frequency_overlap_pairs` 的严格判据（也是数据集不变量测试用的判据）；
    * **逐跳**（``per_hop_v1``）是一跳一个窄框，同频不同时是**合法**的
      （跳频图案本来就会回到同一子带），甚至时间上首尾相接。用严格判据会把
      合法数据集整批拒掉，所以这里只看"两个轴都实质性重叠"。

    单张图的框数很小，所以直接两两比较（频率轴上的相邻扫描只能保证找到
    "存在重叠的那一对"，对需要精确判定的场合不够）。
    """
    pairs = []
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            time_overlap = _axis_overlap(boxes[left][0], boxes[left][2],
                                         boxes[right][0], boxes[right][2])
            if time_overlap <= 0.0:
                continue
            freq_overlap = _axis_overlap(boxes[left][1], boxes[left][3],
                                         boxes[right][1], boxes[right][3])
            if freq_overlap <= 0.0:
                continue
            if (time_overlap > COLLISION_RATIO * min(float(boxes[left][2]), float(boxes[right][2]))
                    and freq_overlap > COLLISION_RATIO * min(float(boxes[left][3]),
                                                             float(boxes[right][3]))):
                pairs.append((left, right))
    return pairs


def _distribution(values, digits=4):
    """一组数的 min/max/mean/count（空集时 count=0，其余为 None）。"""
    if not values:
        return {"min": None, "max": None, "mean": None, "count": 0}
    return {"min": round(float(np.min(values)), digits),
            "max": round(float(np.max(values)), digits),
            "mean": round(float(np.mean(values)), digits),
            "count": len(values)}


def instance_statistics(records):
    """按**信号实例**汇总的分布统计（两边数据源共用这一份口径）。

    与 :func:`dataset_statistics` 的分工：那个统的是"样本/标签框"层面的量，
    这个统的是"每个信号实例"层面的量，专用于**核对两个来源的分布是否重叠**
    （项目生成器与 TorchSig 混训时最关键的一件事——分布不重叠时混进去的那
    一半数据只会拉低指标，而表面上什么都看不出来）。两边的卡片分别写在
    ``statistics.sources.generator`` 与 ``statistics.sources.torchsig`` 下，
    键名与含义完全一致，可以直接逐项对比。

    ``bandwidth_ratio`` 是**实测占用带宽 / 本场景采样率**（来自 ``truth`` 的
    实测带，不是声明的目标带宽）：项目生成器的 FM、跳频实际占用会明显大于
    声明值，用声明值对比会得出错误结论。采样率取每个样本自己的
    ``scene.rate_hz``，因此 ``--rate-range`` 打开后仍然正确。

    ``instances`` 数的是 ``truth`` 条数（信号实例），与标签个数不总相等：
    ``per_hop_v1`` 下一个信号实例对应多个跳标签框，此时标签总数看
    :func:`dataset_statistics` 的 ``targets``。
    """
    ratios, nominal, inband = [], [], []
    instances = 0
    overlaps = 0
    for record in records:
        scene = record.get("scene") or {}
        rate = float(scene.get("rate_hz") or 0.0)
        overlaps += len(shadowed_pairs(record.get("boxes") or []))
        truth = record.get("truth") or []
        instances += len(truth)
        for entry in truth:
            bandwidth = float(entry.get("bandwidth_hz") or 0.0)
            if bandwidth > 0.0 and rate > 0.0:
                ratios.append(bandwidth / rate)
            nominal_value = entry.get("snr_nominal_db")
            if nominal_value is not None:
                nominal.append(float(nominal_value))
            inband.append(float(entry["snr_inband_db"]))
    statistics = {
        "records": len(records),
        "instances": instances,
        "bandwidth_ratio": _distribution(ratios),
        "snr_inband_db": _distribution(inband, 3),
        "collision_pairs": overlaps,
    }
    # 标称信噪比只有 TorchSig 侧才逐实例记录（项目生成器把它存在场景级
    # 的 scene.noise.snr_db 里），一个都没有时直接不写这个键，
    # 免得 count=0 被当成"统计坏了"
    if nominal:
        statistics["snr_nominal_db"] = _distribution(nominal, 3)
    return statistics


def label_semantics(card):
    """数据集卡片的标签语义（缺失时按会话级，兼容历史数据集）。"""
    contract = card.get("contract") or {}
    value = contract.get("label_semantics", DEFAULT_LABEL_SEMANTICS)
    if value not in LABEL_SEMANTICS:
        raise SystemExit(f"数据集标签语义 {value!r} 不被支持，应为 {LABEL_SEMANTICS} 之一")
    return value


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
    label_semantics(card)  # 语义取值在读取时就校验，避免训练完才发现对不上
    return card, records


def contract_of(card):
    """数据集卡片里的契约字段（含图像尺寸 / 训练图像参数三件套）。"""
    contract = dict(card.get("contract") or {})
    for key in ("image_size", "spectrogram_nfft", "dynamic_range_db", "labels"):
        if key not in contract:
            raise SystemExit(f"数据集契约缺少字段 {key}，请重新构建数据集")
    contract["label_semantics"] = label_semantics(card)
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


def dataset_statistics(records, *, image_size, label_semantics=DEFAULT_LABEL_SEMANTICS, extra=None):
    """从样本记录汇总 ``dataset.json`` 的 ``statistics`` 段。

    ``build_dataset.py``（本项目生成器）与 ``ingest_torchsig.py``（第三方 IQ 转写）
    共用这一份口径：统计量全部由 ``records`` 反推，调用方不需要另开累加器，两份
    卡片也就不可能写出不同的统计含义。

    口径（与历史数据集完全一致，改动等价于破坏既有卡片）：

    * ``targets_per_sample`` 按 ``truth`` 条数分组——会话级/逐跳的差异在标签里，
      不在真值里，因此这里数的是"场景里有几个信号"；
    * ``mean_bandwidth_ratio`` / ``labels.*_px`` 取 ``boxes`` 的第 2、3 列（时间维
      归一化宽度、频率维归一化高度）；
    * ``snr_inband_db`` 取 ``truth[].snr_inband_db``，``None`` 不参与统计。

    ``extra`` 是给调用方补充来源专属统计用的（如 ``{"sources": {...}}``），在标准
    字段之后合并；键名与上述字段冲突时会被覆盖，调用方需自行避让。
    """
    if label_semantics not in LABEL_SEMANTICS:
        raise SystemExit(f"数据集标签语义 {label_semantics!r} 不被支持，应为 {LABEL_SEMANTICS} 之一")
    box_counts = []
    widths = []
    heights = []
    snrs = []
    targets_per_sample = {}
    for record in records:
        boxes = record.get("boxes") or []
        truth = record.get("truth") or []
        box_counts.append(len(boxes))
        targets_per_sample[len(truth)] = targets_per_sample.get(len(truth), 0) + 1
        # box[2] 是"时间维"归一化宽度，box[3] 是"频率维"归一化高度
        widths.extend(float(box[2]) for box in boxes)
        heights.extend(float(box[3]) for box in boxes)
        snrs.extend(float(entry["snr_inband_db"]) for entry in truth
                    if entry.get("snr_inband_db") is not None)
    total = int(sum(box_counts))
    statistics = {
        "targets_per_sample": {str(key): value for key, value in sorted(targets_per_sample.items())},
        "targets": total,
        "mean_bandwidth_ratio": float(np.mean(widths)) if widths else 0.0,
        "snr_inband_db": {
            "min": float(np.min(snrs)) if snrs else None,
            "max": float(np.max(snrs)) if snrs else None,
            "mean": float(np.mean(snrs)) if snrs else None,
        },
        # 标签几何的像素尺度：训练脚本据此拒绝"网格比最小标签还粗"的配置
        # （网格化标签每格只保留最大的一个框，否则逐跳标签会被静默丢弃）
        "labels": {
            "semantics": label_semantics,
            "total": total,
            "boxes_per_sample": {
                "min": int(min(box_counts)) if box_counts else 0,
                "mean": round(float(np.mean(box_counts)), 3) if box_counts else 0.0,
                "max": int(max(box_counts)) if box_counts else 0,
            },
            "min_width_px": round(min(widths) * image_size, 3) if widths else None,
            "min_height_px": round(min(heights) * image_size, 3) if heights else None,
            "median_width_px": round(float(np.median(widths)) * image_size, 3)
                               if widths else None,
            "median_height_px": round(float(np.median(heights)) * image_size, 3)
                                if heights else None,
        },
    }
    if extra:
        statistics.update(extra)
    return statistics
