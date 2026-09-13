#!/usr/bin/env python3
"""把 TorchSig 生成的 IQ bundle 转写成本项目的检测数据集（``tf_image_v1`` 契约）。

为什么要这一层
--------------

TorchSig 是**数据生成器**，不是数据集格式：它给出的是宽带 IQ 与"信号实例元数据"
（中心频率、带宽、起止时间、标称信噪比、类名…），而本项目的训练集格式是
"灰度时频图 + 归一化框 + 真值摘要"。直接把两者对接会有三处口径差：

1. **图像**：TorchSig 的 ``Spectrogram`` 变换与本项目 :func:`detection_image` 的
   本底/动态范围/轴方向都不同。若训练用前者、推理用后者，模型看到的分布不一致。
   因此这里**丢弃** TorchSig 的时频图，只用它的 IQ，重新走本项目的
   :func:`spectral_context` + :func:`detection_image`——训练与推理共用一份图像代码。
2. **标签**：框必须由 :func:`band_to_box` 生成（项目的唯一"真值 → 框"公式）。
   TorchSig 的 ``yolo_label`` 目标**绝不能被消费**：那是另一套 ``y`` 轴约定，
   一旦顺手用它，框会整体翻转而形状校验查不出来。
3. **信噪比**：TorchSig 的 ``snr_db`` 是它自己注入时用的标称值，与本项目
   ``inband_snr_v1``（N0·B 参考、按实际占用带宽重测）不是同一个量。因此每条
   真值的 ``snr_inband_db`` 都用 :func:`measure_band` **重测**，标称值另存
   ``snr_nominal_db`` 供溯源。

标签语义只能是 ``session_v1``
------------------------------

TorchSig 的信号类别里**没有跳频族**，它只能描述"一段信号占一个频段"，因此
``per_hop_v1``（逐跳一框）在这里无从构造。``--labels hop`` 会直接报错退出，
而不是悄悄退化成会话级标签——后者会让逐跳通路把一段 8 跳的传输报成"1 跳"，
数字看着正常但对不上真值。逐跳数据集继续用
``training/build_dataset.py --modes fh_rc --labels hop`` 生成。

拒绝而不是容忍
--------------

数据集里有一条被测试锁定的不变量：**标签框在频率轴上互不重叠**（训练脚本的
网格化标签每格只保留最大的一个框，重叠的框会被静默丢弃）。TorchSig 允许同频
叠加（``cochannel_overlap_probability``），默认 0.2。因此每个信号实例都要过
四道几何守卫：越界/过窄、时间为空、亚像素、框数超限；任一条不过就**整条记录
拒绝**（不是丢掉那个信号——留着信号不标注会变成训练时的假阴性）。
默认情况下拒绝即失败（``SystemExit``，带原因与累计计数）；确实需要"跳过错样本
继续跑"时显式加 ``--skip-rejected``。

bundle 格式（``torchsig_bundle_v1``，读写同源：格式定义在 ``training/torchsig_bundle.py``，
由 ``training/build_torchsig.py`` 写出）::

    <bundle>/manifest.json      # bundle 级：采样率、TorchSig 版本、记录清单
    <bundle>/iq/000000.npy      # complex64 一维序列（也接受 (N, 2) 实部/虚部）
    <bundle>/meta/000000.json   # 记录级：duration_s + components[]

产物（与 ``training/build_dataset.py`` 完全相同的契约，可直接混用/合并训练）::

    <output>/dataset.json    # 契约 + bundle 溯源 + 统计（含 sources 分段）
    <output>/images/*.npy    # float32 (H, W)，取值 [0, 1]
    <output>/samples.jsonl   # 每行一个样本；scene 里记的是 bundle 内定位信息

用法::

    .venv/bin/python training/ingest_torchsig.py \\
        --bundle training/data/torchsig_bundle --output training/data/torchsig \\
        --nfft 512 --image-size 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from detectors.dataset import (  # noqa: E402
    OVERLAP_EPSILON,
    dataset_statistics,
    frequency_overlap_pairs,
    instance_statistics,
)
from signal_analysis.ml import (  # noqa: E402
    BOX_COLUMNS,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    OUTPUT_LAYOUT,
    band_to_box,
    detection_image,
    measure_band,
    spectral_context,
)
from torchsig_bundle import BUNDLE_VERSION, load_bundle  # noqa: E402

#: 类别字典：TorchSig 的 ``class_name`` 不参与本项目标签（见模块文档），
#: 所有信号实例都归到同一个"辐射源"类别（与生成器数据集的 LABELS 一致）
LABELS = ("emitter",)

#: 列名与推理端 ``BOX_COLUMNS``（列数）必须一一对应，这里做一次自校验
BOX_NAMES = ("x_center", "y_center", "width", "height", "confidence", "class")
assert len(BOX_NAMES) == BOX_COLUMNS

#: 记录被拒绝的原因（写进报告，便于按原因定位 bundle 侧的问题）
REJECTION_KINDS = ("malformed_component", "missing_geometry", "out_of_band", "too_narrow",
                   "too_short", "sub_pixel", "overlap_frequency", "too_many_boxes")


class _Rejected(Exception):
    """单条记录无法落成合法标签（携带原因与说明，由 main 决定是失败还是跳过）。"""

    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


def _number(value):
    """把元数据里的数字转成 float；非有限值一律当"没有"。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TorchSig IQ bundle → 本项目检测数据集（tf_image_v1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle", required=True, help="torchsig_bundle_v1 目录（含 manifest.json）")
    parser.add_argument("--output", required=True, help="输出数据集目录")
    parser.add_argument("--image-size", type=int, default=1024, help="时频图边长（2 的幂）")
    parser.add_argument("--nfft", type=int, default=512, help="STFT 点数（必须与推理端一致）")
    parser.add_argument("--dynamic-range", type=float, default=60.0, help="图像动态范围（dB）")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条记录（0 = 全部）")
    parser.add_argument("--max-boxes", type=int, default=32,
                        help="单样本框数上限（超过即拒绝，须 ≤ 训练脚本的 --max-boxes）")
    parser.add_argument("--min-box-px", type=float, default=2.0,
                        help="标签框最小边（像素）；比这更小的框在网格化时会丢")
    parser.add_argument("--labels", choices=("session", "hop"), default="session",
                        help="标签粒度；TorchSig 没有跳频族，hop 会直接报错退出")
    parser.add_argument("--skip-rejected", action="store_true",
                        help="拒绝的记录只计数不中断（默认一旦拒绝就失败）")
    return parser.parse_args(argv)


def _read_iq(path):
    """读一条记录的 IQ：``complex64`` 一维序列；``(N, 2)`` 实数数组按实部/虚部处理。"""
    array = np.load(path)
    if array.ndim == 2 and array.shape[1] == 2:
        array = array[:, 0] + 1j * array[:, 1]
    if array.ndim != 1 or array.size < 2:
        raise SystemExit(f"{path} 应为单通道 IQ 序列，实际形状 {array.shape}")
    if not np.iscomplexobj(array):
        array = array.astype(np.float64).astype(np.complex64)  # 实信号：虚部为 0
    return np.ascontiguousarray(array, dtype=np.complex64)


def _component_band(component, rate, duration):
    """一个 TorchSig 信号实例 → 频段/时间真值；不合法时抛 :class:`_Rejected`。"""
    if not isinstance(component, dict):
        raise _Rejected("malformed_component", f"components 里有 {type(component).__name__}")
    low = _number(component.get("lower_freq"))
    high = _number(component.get("upper_freq"))
    if low is None or high is None:
        # 元数据只给了中心频率与带宽时按对称占用带宽推算（TorchSig 两种写法都存在）
        center = _number(component.get("center_freq"))
        bandwidth = _number(component.get("bandwidth"))
        if bandwidth is None:
            bandwidth = _number(component.get("occupied_bandwidth"))
        if center is None or bandwidth is None:
            raise _Rejected("missing_geometry", "既没有 lower/upper_freq，也没有 center_freq+bandwidth")
        low, high = center - bandwidth / 2.0, center + bandwidth / 2.0
    low, high = min(low, high), max(low, high)
    nyquist = rate / 2.0
    if high <= -nyquist or low >= nyquist:
        raise _Rejected("out_of_band", f"[{low:g}, {high:g}] Hz 完全落在 ±{nyquist:g} Hz 之外")
    low, high = max(low, -nyquist), min(high, nyquist)
    if high - low <= 0:
        raise _Rejected("too_narrow", f"占用带宽 {high - low:g} Hz")
    # 时间轴优先用采样点字段（无歧义）。TorchSig 2.2 的 ``start``/``stop`` 是
    # "占记录长度的比例"，与 bundle 的秒制不同名同义——照搬会把信号整体压到记录
    # 头部；因此只有采样点字段缺失时才退回秒制 ``start``/``stop``，再退到整条记录。
    start = stop = None
    start_samples = _number(component.get("start_in_samples"))
    length_samples = _number(component.get("duration_in_samples"))
    if start_samples is not None and length_samples is not None:
        start = start_samples / rate
        stop = (start_samples + length_samples) / rate
    if start is None or stop is None:
        start = _number(component.get("start")) or 0.0
        stop = _number(component.get("stop"))
        if stop is None:
            stop = duration
    start, stop = max(start, 0.0), min(stop, duration)
    if stop - start <= 0:
        raise _Rejected("too_short", f"时间区间 [{component.get('start')}, {component.get('stop')}] 越出记录")
    return {"f_low_hz": float(low), "f_high_hz": float(high),
            "t_start_s": float(start), "t_end_s": float(stop)}


def _time_overlap_count(boxes):
    """同时在时间轴上重叠的框对数（不构成拒绝理由，只作为分布统计）。"""
    count = 0
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            span = (float(boxes[left][2]) + float(boxes[right][2])) / 2.0
            if abs(float(boxes[left][0]) - float(boxes[right][0])) < span - OVERLAP_EPSILON:
                count += 1
    return count


def _ingest_record(bundle_root, entry, position, args, rate):
    """一条 bundle 记录 → 数据集样本（``record`` 字典）；不合法时抛 :class:`_Rejected`。"""
    iq = _read_iq(bundle_root / entry["iq"])
    duration = float(iq.size) / rate
    meta_path = entry.get("meta")
    meta = json.loads((bundle_root / meta_path).read_text(encoding="utf-8")) if meta_path else {}
    components = meta.get("components")
    if components is None:
        components = []
    if isinstance(components, dict):  # 兼容"以类名为键"的写法
        components = list(components.values())
    declared = _number(meta.get("duration_s"))
    if declared is not None and abs(declared - duration) > 1.0 / rate:
        # 时间轴必须由样本数决定：图像与标签都要用同一份 duration，否则框会整体错位
        raise _Rejected("too_short",
                        f"元数据声明 duration_s={declared:g}，样本数为 {iq.size}（应为 {duration:g}）")

    summary, arrays = spectral_context(iq, rate, {"nfft": args.nfft})
    image, image_meta = detection_image(arrays, summary, args.image_size, args.dynamic_range)

    boxes = []
    truth = []
    for order, component in enumerate(components):
        band = _component_band(component, rate, duration)
        box = band_to_box(image_meta, band["f_low_hz"], band["f_high_hz"],
                          band["t_start_s"], band["t_end_s"])
        if box[2] * args.image_size < args.min_box_px or box[3] * args.image_size < args.min_box_px:
            raise _Rejected("sub_pixel",
                            f"第 {order} 个信号实例折合 {box[2] * args.image_size:.3f} × "
                            f"{box[3] * args.image_size:.3f} 像素，小于 {args.min_box_px:g}")
        measured = measure_band(arrays, summary, band)
        nominal = _number(component.get("snr_db"))
        center = 0.5 * (band["f_low_hz"] + band["f_high_hz"])
        boxes.append([round(float(value), 6) for value in box] + [1.0, 0.0])
        truth.append({
            "index": order,
            "mode": str(component.get("class_name") or component.get("mode") or "unknown"),
            "hopping": False,  # TorchSig 没有跳频族：所有实例都是"一段传输"
            "nominal_offset_hz": round(center, 6),
            "nominal_bandwidth_hz": round(band["f_high_hz"] - band["f_low_hz"], 6),
            "center_hz": round(center, 6),
            "bandwidth_hz": round(band["f_high_hz"] - band["f_low_hz"], 6),
            "f_low_hz": round(band["f_low_hz"], 6),
            "f_high_hz": round(band["f_high_hz"], 6),
            "t_start_s": round(band["t_start_s"], 6),
            "t_end_s": round(band["t_end_s"], 6),
            "power_dbfs": round(float(measured["power_dbfs"]), 6),
            # 项目口径（inband_snr_v1，见 measure_band）——推理端比的就是这个量
            "snr_inband_db": round(float(measured["snr_db"]), 6),
            "snr_nominal_db": None if nominal is None else round(nominal, 6),
            "session_id": order,
            "visited_channels": None,
        })

    if len(boxes) > args.max_boxes:
        raise _Rejected("too_many_boxes", f"该记录有 {len(boxes)} 个框，超过 --max-boxes {args.max_boxes}")
    pairs = frequency_overlap_pairs(boxes)
    if pairs:
        left, right = pairs[0]
        both = int(abs(float(boxes[left][0]) - float(boxes[right][0]))
                   < (float(boxes[left][2]) + float(boxes[right][2])) / 2.0 - OVERLAP_EPSILON)
        raise _Rejected("overlap_frequency",
                        f"第 {left} 与第 {right} 个框在频率轴上重叠"
                        f"（{'同时' if both else '但未'}在时间轴上重叠），共 {len(pairs)} 对")

    name = f"{position:06d}.npy"
    np.save(Path(args.output) / "images" / name, image)
    return {
        "index": position,
        "image": f"images/{name}",
        "boxes": boxes,
        "truth": truth,
        "hops": [],  # 会话级语义：逐跳真值天然为空（TorchSig 无跳频族）
        "source": "torchsig",
        "scene": {
            "rate_hz": rate,
            "duration_s": duration,
            "source": "torchsig_bundle",
            # 与生成器数据集不同，这里**不能**用 generate_iq 复现波形：
            # 复现靠 bundle 本身（路径 + 记录序号），IQ 是外部生成的
            "bundle": entry.get("iq"),
            "bundle_index": position,
            "time_overlap_pairs": _time_overlap_count(boxes),
        },
    }


def main(argv=None):
    args = _parse_args(argv)
    if args.labels == "hop":
        raise SystemExit(
            "--labels hop 无法用于 TorchSig：它的信号类别里没有 fh* 跳频族，"
            "构造不出 per_hop_v1 标签。请用默认的 session；逐跳数据集继续用 "
            "training/build_dataset.py --modes fh_rc --labels hop 生成。")
    if args.max_boxes < 1:
        raise SystemExit("--max-boxes 必须大于 0")
    if not 0.0 < args.train_fraction <= 1.0:
        raise SystemExit("--train-fraction 应在 (0, 1] 之间")

    bundle_root = Path(args.bundle)
    manifest = load_bundle(bundle_root)
    rate = float(manifest["sample_rate_hz"])
    selected = list(manifest["records"])
    if args.limit > 0:
        selected = selected[:args.limit]

    output = Path(args.output)
    (output / "images").mkdir(parents=True, exist_ok=True)

    records = []
    rejected = {}
    for position, entry in enumerate(selected):
        try:
            record = _ingest_record(bundle_root, entry, position, args, rate)
        except _Rejected as problem:
            rejected[problem.kind] = rejected.get(problem.kind, 0) + 1
            if not args.skip_rejected:
                raise SystemExit(
                    f"bundle 第 {position} 条记录无法落成合法标签：{problem.detail}"
                    f"（原因代码 {problem.kind}）。已拒绝 {sum(rejected.values())} 条："
                    f"{_format_rejections(rejected)}。\n"
                    "  这是硬失败而不是静默跳过：被拒绝的信号若留在图像里不标注，"
                    "训练时就是假阴性。请修正 bundle 侧的元数据，或显式加 --skip-rejected。")
            continue
        records.append(record)
        if (position + 1) % 100 == 0 or position + 1 == len(selected):
            print(f"  已转写 {position + 1}/{len(selected)} 条记录", flush=True)

    if not records:
        raise SystemExit(f"没有任何记录可用：全部 {len(selected)} 条都被拒绝"
                         f"（{_format_rejections(rejected)}）")

    # 划分按 bundle 顺序切分（确定性）：TorchSig 的记录之间没有可复现的随机种子，
    # 用位置切分至少保证同一 bundle 每次得到同一份 train/val
    for position, record in enumerate(records):
        record["split"] = "train" if position < round(len(records) * args.train_fraction) else "val"
    train_count = sum(1 for record in records if record["split"] == "train")

    card = {
        "generator_script": "training/ingest_torchsig.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_count": len(records),
        "splits": {"train": train_count, "val": len(records) - train_count},
        "contract": {
            "input_contract": INPUT_CONTRACT,
            "output_layout": OUTPUT_LAYOUT,
            "layout": IMAGE_LAYOUT,
            "image_size": int(args.image_size),
            "channels": 1,
            "spectrogram_nfft": int(args.nfft),
            "dynamic_range_db": float(args.dynamic_range),
            "labels": list(LABELS),
            # TorchSig 只能表达"一段传输一个框"，逐跳语义在这里无从构造
            "label_semantics": "session_v1",
            "box_columns": list(BOX_NAMES),
            "box_column_count": int(BOX_COLUMNS),
            "frequency_reference": "baseband_offset",
            "note": "行 0 = +fs/2，列 0 = t=0；四列几何量按图像宽高归一化",
        },
        "scene": {
            "sample_rate_hz": rate,
            "duration_range_s": [min(record["scene"]["duration_s"] for record in records),
                                 max(record["scene"]["duration_s"] for record in records)],
            "class_names": sorted({entry["mode"] for record in records for entry in record["truth"]}),
            "max_signals": max((len(record["boxes"]) for record in records), default=0),
            "min_box_px": float(args.min_box_px),
            "max_boxes": int(args.max_boxes),
            "train_fraction": float(args.train_fraction),
        },
        "ingest": {
            "source": "torchsig",
            "torchsig_version": manifest.get("torchsig_version"),
            "bundle": str(args.bundle),
            "bundle_version": manifest.get("bundle_version"),
            "metadata_overrides": manifest.get("metadata_overrides") or {},
            # 标签来自信号实例的频段/时间，**不消费** TorchSig 的 yolo_label
            "label_source": "component_band",
            "snr_source": "inband_snr_v1 (measure_band 重测；标称值另存 snr_nominal_db)",
            "record_count": len(selected),
            "accepted": len(records),
            "rejected": dict(sorted(rejected.items())),
            "rejected_total": int(sum(rejected.values())),
            "image_source": "本项目 spectral_context + detection_image（未使用 TorchSig 时频变换）",
            "note": "TorchSig 只提供 IQ 与元数据；图像与标签由本项目同一份公式生成，"
                    "因此训练与推理的口径完全一致。逐跳语义不适用。",
        },
        "statistics": dataset_statistics(
            records,
            image_size=args.image_size,
            label_semantics="session_v1",
            # 与 build_dataset.py 共用 instance_statistics：两边卡片的键名、
            # 含义完全一致，混训前可直接逐项比对分布
            extra={"sources": {"torchsig": instance_statistics(records)}},
        ),
    }
    (output / "dataset.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    print(f"数据集已写入 {output}")
    print(f"  样本 {card['sample_count']}（train {train_count} / val "
          f"{card['sample_count'] - train_count}），目标 {card['statistics']['targets']} 个")
    print(f"  契约：{card['contract']['layout']} {args.image_size}×{args.image_size}，"
          f"nfft {args.nfft}，动态范围 {args.dynamic_range:g} dB，"
          f"标签语义 {card['contract']['label_semantics']}")
    source = card["statistics"]["sources"]["torchsig"]
    print(f"  来源 torchsig {source['records']} 条记录 / {source['instances']} 个实例，"
          f"占用带宽比 {source['bandwidth_ratio']['min']}～{source['bandwidth_ratio']['max']}，"
          f"带内信噪比 {source['snr_inband_db']['min']}～{source['snr_inband_db']['max']} dB，"
          f"两轴碰撞对 {source['collision_pairs']}")
    if card["ingest"]["rejected_total"]:
        print(f"  已跳过被拒绝的记录：{_format_rejections(rejected)}")
    return 0


def _format_rejections(rejected):
    """拒绝计数的可读形式（``原因=条数``，无拒绝时给出明确说明）。"""
    if not rejected:
        return "无"
    return "、".join(f"{kind}={count}" for kind, count in sorted(rejected.items()))


if __name__ == "__main__":
    raise SystemExit(main())
