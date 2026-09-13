#!/usr/bin/env python3
"""构建 AI 检测训练集：把生成器真值直接变成时频图标签。

设计要点（与推理端共用同一份契约，见 ``training/README.md``）：

* 图像由 :func:`signal_analysis.ml.detection_image` 生成、标签由
  :func:`signal_analysis.ml.band_to_box` 生成——训练与推理的坐标定义、
  归一化范围与图像排布因此不可能不一致（这是本项目最重要的一条约定）；
* 场景参数（调制样式、频点、带宽、带内信噪比、持续时间）随机采样，标签取
  :func:`signal_analysis.evaluation.signal_truth`：跳频信号按"一次会话一个
  框"标注，与推理端的会话合并口径一致；
* 只依赖 NumPy 与项目自身代码，不需要 torch；数据集完全由本项目生成器合成，
  无第三方数据集的许可证约束。

场景扩充（标签安全）：

上面三条设计里最关键的一条是"标签由生成本身的真值算出来"——因此**改动采样
分布不会让标签失配**，标签会自动跟到新的频点/带宽/持续时长上。能让数据集
覆盖到更多真实场景的三个旋钮都在下面（默认值都是"关闭"，不改变历史输出）：

* ``--rate-range``：每场景独立采样采样率（多采样率域偏移）。带宽比例因此是
  相对本场景采样率算的，卡片新增 ``scene.rate_hz_range`` 记录该设置；
* ``--cfo-ratio``：槽内载波频偏抖动（占目标带宽）。占用带始终落在槽内，
  因此标签框不会因抖动而互相重叠；
* ``--snr-bias``：把带内信噪比从均匀分布向**低信噪比端**倾斜，补足难例。

另外，生成后仍会校验一次"标签框不会互相顶掉"这条硬不变量，违反就重抽场景
（会话级用 :func:`detectors.dataset.frequency_overlap_pairs`、逐跳用
:func:`detectors.dataset.shadowed_pairs`，两者对应各自的标签粒度）。宁可重抽
也不写出训练脚本会静默丢框的样本——这是本项目最容易被忽视的数据损坏方式。

产物::

    <output>/dataset.json    # 契约 + 生成参数 + 统计（训练脚本据此校验契约）
    <output>/images/*.npy    # float32 (H, W)，取值 [0, 1]
    <output>/samples.jsonl   # 每行一个样本：图像路径、标签框、场景摘要、划分
                             # 其中 scene 字段保存了生成器的原始输入（signals /
                             # noise / seed / duration），可用 generate_iq 逐样本
                             # 复现完全相同的波形，供训练后的端到端评测使用

用法::

    .venv/bin/python training/build_dataset.py --output training/data/detector \\
        --count 2000 --seed 7 --image-size 1024 --nfft 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from detectors.dataset import (  # noqa: E402
    dataset_statistics,
    frequency_overlap_pairs,
    instance_statistics,
    shadowed_pairs,
)
from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import hop_truth, signal_truth  # noqa: E402
from signal_analysis.ml import (  # noqa: E402
    BOX_COLUMNS,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    OUTPUT_LAYOUT,
    band_to_box,
    detection_image,
    spectral_context,
)

DEFAULT_MODES = ("am", "fm", "ssb", "ask2", "qpsk", "qam16", "qam64", "fh_rc", "fh_video")
# 列名与推理端 BOX_COLUMNS（列数）必须一一对应，这里做一次自校验
BOX_NAMES = ("x_center", "y_center", "width", "height", "confidence", "class")
assert len(BOX_NAMES) == BOX_COLUMNS
LABELS = ("emitter",)
# 标签语义：会话级（一段传输一个框）或逐跳（``fh*`` 信号一跳一个框）
LABEL_SEMANTICS = ("session_v1", "per_hop_v1")
MAX_ATTEMPTS = 32


def _pair(text, cast=float):
    """``"最小值,最大值"`` → 二元组（区间必须有限且有序）。"""
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("请给出形如 最小值,最大值 的区间")
    low, high = cast(parts[0]), cast(parts[1])
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise argparse.ArgumentTypeError("区间必须有限且满足 最小值 ≤ 最大值")
    return low, high


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="构建 AI 检测数据集（时频图 + 归一化框标签）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--count", type=int, default=1000, help="样本数，默认 1000")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（决定场景与划分）")
    parser.add_argument("--rate", type=float, default=1_000_000.0, help="采样率（Hz），默认 1e6")
    parser.add_argument("--rate-range", type=_pair, default=None,
                        help="每场景采样率区间（Hz），默认不抖动、恒为 --rate；给出区间后"
                             "带宽比例按本场景采样率算，可用于模拟多采样率域偏移")
    parser.add_argument("--duration-range", type=_pair, default=(0.2, 0.5),
                        help="每场景持续时间区间（s），默认 0.2,0.5")
    parser.add_argument("--image-size", type=int, default=1024,
                        help="时频图边长（与推理清单 input.image_size 一致），默认 1024")
    parser.add_argument("--nfft", type=int, default=512, help="STFT 点数，默认 512")
    parser.add_argument("--dynamic-range", type=float, default=60.0, help="图像动态范围（dB），默认 60")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES),
                        help=f"可选调制样式，默认全部：{', '.join(DEFAULT_MODES)}")
    parser.add_argument("--max-signals", type=int, default=3, help="单场景最多信号数，默认 3")
    parser.add_argument("--snr-range", type=_pair, default=(-5.0, 30.0),
                        help="最强信号带内信噪比区间（dB），默认 -5,30")
    parser.add_argument("--snr-bias", type=float, default=0.0,
                        help="带内信噪比分布向低端倾斜的程度（0～1），默认 0 = 均匀分布；"
                             "越大低信噪比样本越多（补难例）：0.5 时均值约从 12.5 dB 降到 "
                             "9.0 dB、0 dB 以下占比从 14%% 升到 28%%，1.0 时均值约 6.6 dB")
    parser.add_argument("--noise-only-ratio", type=float, default=0.12,
                        help="纯噪声场景比例（无标签，抑制虚警），默认 0.12")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--guard-hz", type=float, default=10_000.0,
                        help="频段两端与信号之间的保护间隔（Hz），默认 10000")
    parser.add_argument("--min-gap-hz", type=float, default=15_000.0,
                        help="相邻信号之间的最小间隔（Hz），默认 15000")
    parser.add_argument("--min-bandwidth-ratio", type=float, default=0.02,
                        help="最小占用带宽（占采样率），默认 0.02；TorchSig 默认分布约为"
                             "0.25～0.33，两个来源混训时应让区间有重叠，例如 0.02～0.33")
    parser.add_argument("--max-bandwidth-ratio", type=float, default=0.12,
                        help="最大占用带宽（占采样率），默认 0.12")
    parser.add_argument("--cfo-ratio", type=float, default=0.0,
                        help="槽内载波频偏抖动幅度（占目标带宽），默认 0 = 关闭；"
                             "占用带始终落在槽内，标签框不会因抖动而重叠")
    parser.add_argument("--hop-rate-range", type=_pair, default=(10.0, 200.0),
                        help="跳频跳速区间（Hz），默认 10,200")
    parser.add_argument("--labels", choices=("session", "hop"), default="session",
                        help="标签粒度：session = 一段传输一个框（默认，会话级模型）；"
                             "hop = fh* 信号一跳一个框（逐跳模型，推理端用 ml-detect-hops 才能对上）")
    return parser.parse_args(argv)


def _plan_slots(rng, args, count, rate, padding=0.0):
    """在 ±fs/2 内随机划分互不重叠的放置区间。

    返回 ``[(low, high, width)]``：``low/high`` 是允许放置的区间，``width`` 是目标
    占用带宽。``padding`` 是槽内抖动余量（占目标带宽），``padding=0`` 时放置区间
    恰好等于占用带宽，与历史行为逐位一致。
    """
    low_limit = -rate / 2.0 + args.guard_hz
    high_limit = rate / 2.0 - args.guard_hz
    slots = []
    cursor = low_limit
    for _ in range(count):
        width = rng.uniform(args.min_bandwidth_ratio, args.max_bandwidth_ratio) * rate
        slot_width = width * (1.0 + 2.0 * padding)
        room = high_limit - cursor
        if room < slot_width + args.min_gap_hz:
            break
        start = cursor + rng.uniform(0.0, room - slot_width)
        low, high = float(start), float(start + slot_width)
        if padding:
            # 槽内余量只用来放载波频偏：返回**抽到的目标带宽**，占用带仍由
            # --min/max-bandwidth-ratio 标定。若这里返回 high - low（槽宽），
            # 目标带宽会被放大成 (1 + 2×cfo_ratio) 倍——混训时用它对齐
            # TorchSig 的带宽分布就会被静默带偏
            slots.append((low, high, width))
        else:
            # padding=0 时槽宽与目标带宽只差 1 ULP，历史实现传下去的是
            # high - low，这里保持一致以保证旧数据集逐位可复现
            slots.append((low, high, high - low))
        cursor = high + args.min_gap_hz
    return slots


def _signal_spec(rng, mode, low, high, width, power_dbfs, args):
    """一个放置区间 → 信号参数（占用频带严格落在区间内，标签因此不会互相重叠）。

    ``width`` 是目标占用带宽。区间比它宽时（``--cfo-ratio`` 给出的槽内余量）先把
    放置区间收窄到占用带宽再随机平移，等效于给信号加一个载波频偏；占用带不会
    出槽，邻道间隔仍然成立。
    """
    if high - low > width:
        center = 0.5 * (low + high)
        center += float(rng.uniform(-1.0, 1.0)) * 0.5 * ((high - low) - width)
        low, high = center - width / 2.0, center + width / 2.0
    else:
        width = high - low
    if mode == "ssb":
        side = "usb" if rng.random() < 0.5 else "lsb"
        offset = low if side == "usb" else high
        spec = {"mode": mode, "offset": float(offset), "bandwidth": float(width),
                "side": side, "power_dbfs": float(power_dbfs)}
    elif mode.startswith("fh"):
        hops = int(rng.integers(2, 9))
        spec = {"mode": mode, "offset": float(0.5 * (low + high)), "bandwidth": float(width),
                "hop_count": hops, "hop_bandwidth": float(width / (hops + 1.0)),
                "hop_rate": float(rng.uniform(*args.hop_rate_range)),
                "power_dbfs": float(power_dbfs)}
        if mode == "fh_video":
            spec["subcarriers"] = int(rng.choice((32, 64, 128)))
    else:
        spec = {"mode": mode, "offset": float(0.5 * (low + high)), "bandwidth": float(width),
                "power_dbfs": float(power_dbfs)}
    if mode == "am":
        spec["depth"] = float(rng.uniform(0.4, 1.0))
    elif mode == "fm":
        # deviation 决定实际占用带宽（5.3×deviation），上限压在槽宽之内
        spec["deviation"] = float(rng.uniform(0.12, 0.19) * width)
        spec["message_bandwidth"] = float(rng.uniform(0.1, 0.3) * width)
    elif mode in ("ask2", "qpsk", "qam16", "qam64"):
        spec["alpha"] = float(rng.choice((0.2, 0.35, 0.5)))
    elif mode == "fh_rc":
        spec["symbol_rate"] = float(rng.uniform(0.25, 0.5) * spec["hop_bandwidth"])
    return spec


def _scene_rate(rng, args):
    """本场景的采样率。

    未给 ``--rate-range`` 时直接返回 ``--rate`` 且**不消耗随机数**，历史数据集因此
    逐位可复现；给出区间后每场景独立采样一个采样率。
    """
    if args.rate_range is None:
        return float(args.rate)
    return float(rng.uniform(*args.rate_range))


def _sample_snr(rng, args):
    """目标的带内信噪比（定义在最强信号上）。

    ``--snr-bias 0``（默认）是与历史一致的均匀分布且不额外消耗随机数；
    ``--snr-bias > 0`` 对均匀分位做幂变换 ``u ** (1 + bias)``：任何阈值上的低端
    样本占比都随之单调增加，``bias → 0`` 时连续退化为均匀分布。

    这里**不用**三角分布（看似自然的“众数移到低端”）：三角分布把众数移向低端的
    同时会抽薄极低端——bias=0.3 时众数落在 19.5 dB，0 dB 以下的样本比均匀分布
    还少一半，均值反而从 12.5 dB 升到 14.8 dB，"补难例"适得其反。
    """
    low, high = args.snr_range
    if args.snr_bias <= 0.0:
        return float(rng.uniform(low, high))
    exponent = 1.0 + min(float(args.snr_bias), 1.0)
    return float(low + (high - low) * rng.random() ** exponent)


def _scene(rng, args, rate):
    """随机生成一个场景，返回 ``(signals, noise)``；``rate`` 是本场景采样率。"""
    if rng.random() < args.noise_only_ratio:
        return [], {"enabled": True, "bandwidth": float(rate),
                    "power_dbfs": float(rng.uniform(-40.0, -10.0))}
    slots = _plan_slots(rng, args, int(args.max_signals), rate, padding=args.cfo_ratio)
    if not slots:
        return [], {"enabled": True, "bandwidth": float(rate),
                    "power_dbfs": float(rng.uniform(-40.0, -10.0))}
    modes = list(args.modes)
    rng.shuffle(modes)
    order = rng.permutation(len(slots))
    signals = []
    for rank, index in enumerate(order):
        low, high, width = slots[int(index)]
        # 最强信号功率定在 0 dBFS，其余按 0.5～6 dB 递减，让 noise.snr_db
        # （定义在最强信号上）就是本场景的目标带内信噪比
        power = 0.0 if rank == 0 else -float(rng.uniform(0.5, 6.0))
        signals.append(_signal_spec(rng, modes[rank % len(modes)], low, high, width, power, args))
    noise = {"enabled": True, "bandwidth": float(rate), "snr_db": _sample_snr(rng, args)}
    return signals, noise


def _label_truth(generation, hop_labels=False):
    """标签用的真值条目：会话级，或跳频信号的逐跳。

    ``hop_labels=False`` 时与历史行为逐位一致（只用 :func:`signal_truth`）。
    打开逐跳后**不是**整体切换：:func:`hop_truth` 只描述 ``fh*`` 信号，连续
    信号的粒度本来就是"整段传输"，继续用会话条目。因此按会话替换——同一场景
    里跳频与连续信号并存时，两类标签都能正确表达。
    """
    session_entries = signal_truth(generation)
    if not hop_labels:
        return list(session_entries)
    per_hop = hop_truth(generation)
    if not per_hop:
        return list(session_entries)
    grouped = {}
    for entry in per_hop:
        grouped.setdefault(entry["session_index"], []).append(entry)
    entries = []
    for entry in session_entries:
        hops = grouped.get(entry["index"])
        if hops:
            entries.extend(hops)
        else:
            entries.append(entry)
    return entries


def _labels(meta, generation, hop_labels=False):
    """真值 → 归一化标签框 ``[x, y, w, h, confidence, class]``。"""
    boxes = []
    for entry in _label_truth(generation, hop_labels):
        box = band_to_box(meta, entry["f_low_hz"], entry["f_high_hz"],
                          entry["t_start_s"], entry["t_end_s"])
        boxes.append([round(float(value), 6) for value in box] + [1.0, 0.0])
    return boxes


def _describe(generation):
    """紧凑的场景摘要（写进样本行，便于按信噪比/样式筛选错误样本）。"""
    return [{"mode": entry["mode"], "hopping": entry["hopping"],
             "center_hz": entry["center_hz"], "bandwidth_hz": entry["bandwidth_hz"],
             "snr_inband_db": entry["snr_inband_db"]}
            for entry in signal_truth(generation)]


def main(argv=None):
    args = _parse_args(argv)
    if args.count < 1:
        raise SystemExit("样本数必须大于 0")
    if not 0.0 <= args.noise_only_ratio < 1.0:
        raise SystemExit("纯噪声场景比例应在 0～1 之间")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("训练集比例应在 0～1 之间")
    if not 0.0 < args.min_bandwidth_ratio <= args.max_bandwidth_ratio < 1.0:
        raise SystemExit("占用带宽比例区间非法")
    if not 0.0 <= args.cfo_ratio <= 1.0:
        raise SystemExit("--cfo-ratio 应在 0～1 之间")
    if not 0.0 <= args.snr_bias <= 1.0:
        raise SystemExit("--snr-bias 应在 0～1 之间")
    rates = (float(args.rate), float(args.rate)) if args.rate_range is None else args.rate_range
    if rates[0] <= 0.0:
        raise SystemExit("采样率必须大于 0")
    # 采样率太低会让 _plan_slots 一个槽都排不下，退化成整批纯噪声数据集（采样分布
    # 类问题最典型的静默失败），所以直接拦住而不是让它安静地跑完
    if rates[0] - 2.0 * args.guard_hz < args.min_bandwidth_ratio * rates[0] + args.min_gap_hz:
        raise SystemExit("采样率相对 --guard-hz/--min-gap-hz 过低，排不下任何信号，请调整参数")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"不支持的调制样式：{', '.join(unknown)}")
    args.modes = modes
    hop_labels = args.labels == "hop"
    if hop_labels and not any(mode.startswith("fh") for mode in modes):
        raise SystemExit(f"逐跳标签需要 fh* 样式参与生成，当前 --modes 不含跳频：{', '.join(modes)}")

    output = Path(args.output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for index in range(args.count):
        record = None
        for _ in range(MAX_ATTEMPTS):
            seed = int(rng.integers(0, 2 ** 32))
            rate = _scene_rate(rng, args)
            duration = float(rng.uniform(*args.duration_range))
            signals, noise = _scene(rng, args, rate)
            try:
                samples, generation = generate_iq(rate, duration, signals,
                                                  noise=noise, seed=seed)
            except ValueError:
                continue  # 参数组合越界：重抽一个场景而不是让整批失败
            summary, arrays = spectral_context(samples, rate, {"nfft": args.nfft})
            image, meta = detection_image(arrays, summary, args.image_size, args.dynamic_range)
            boxes = _labels(meta, generation, hop_labels)
            # 判据跟着标签粒度走：会话级盖满整帧，频率轴重叠就等价于同格；逐跳框
            # 同频不同时是合法的，只有在两个轴上都实质性重叠时才算冲突
            conflicts = (shadowed_pairs(boxes) if hop_labels
                         else frequency_overlap_pairs(boxes))
            if conflicts:
                # 抖动/参数组合把两个信号的占用带挤到一起：重抽场景。宁可重抽也不
                # 写出违反数据集不变量的样本——训练脚本每格只留最大的一个框，
                # 重叠的框会被静默丢弃，那才是真正难以发现的数据损坏
                continue
            name = f"{index:06d}.npy"
            np.save(images / name, image)
            described = _describe(generation)
            record = {
                "index": index,
                "image": f"images/{name}",
                "boxes": boxes,
                "truth": described,
                "hops": hop_truth(generation),
                "scene": {
                    "rate_hz": float(rate),
                    "duration_s": duration,
                    "seed": seed,
                    "signals": signals,
                    "noise": noise,
                },
                "split": "train" if index < round(args.count * args.train_fraction) else "val",
            }
            break
        if record is None:
            raise SystemExit(f"第 {index} 个场景连续 {MAX_ATTEMPTS} 次生成失败，请放宽参数")
        records.append(record)
        if (index + 1) % 100 == 0 or index + 1 == args.count:
            print(f"  已生成 {index + 1}/{args.count} 个样本", flush=True)

    train_count = sum(1 for record in records if record["split"] == "train")
    card = {
        "generator_script": "training/build_dataset.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
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
            "label_semantics": "per_hop_v1" if hop_labels else "session_v1",
            "box_columns": list(BOX_NAMES),
            "box_column_count": int(BOX_COLUMNS),
            "frequency_reference": "baseband_offset",
            "note": "行 0 = +fs/2，列 0 = t=0；四列几何量按图像宽高归一化",
        },
        "scene": {
            "sample_rate_hz": float(args.rate),
            "rate_hz_range": [float(rates[0]), float(rates[1])],
            "duration_range_s": [float(args.duration_range[0]), float(args.duration_range[1])],
            "modes": list(args.modes),
            "max_signals": int(args.max_signals),
            "snr_db_range": [float(args.snr_range[0]), float(args.snr_range[1])],
            "snr_bias": float(args.snr_bias),
            "cfo_ratio": float(args.cfo_ratio),
            "bandwidth_ratio_range": [float(args.min_bandwidth_ratio),
                                      float(args.max_bandwidth_ratio)],
            "noise_only_ratio": float(args.noise_only_ratio),
            "guard_hz": float(args.guard_hz),
            "min_gap_hz": float(args.min_gap_hz),
        },
        # 统计由 dataset_statistics 统一汇总（与 ingest_torchsig.py 共用一份口径），
        # 因此这里不再自己开累加器；sources.generator 与那边的 sources.torchsig
        # 键名一致，混训前可以直接逐项对比两个来源的分布
        "statistics": dataset_statistics(
            records,
            image_size=args.image_size,
            label_semantics="per_hop_v1" if hop_labels else "session_v1",
            extra={"sources": {"generator": instance_statistics(records)}},
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
    label_stats = card["statistics"]["labels"]
    print(f"  标签：共 {label_stats['total']} 个，单样本 {label_stats['boxes_per_sample']['min']}～"
          f"{label_stats['boxes_per_sample']['max']} 个（均值 "
          f"{label_stats['boxes_per_sample']['mean']}），最小标签 "
          f"{label_stats['min_width_px']} × {label_stats['min_height_px']} 像素")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
