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

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import signal_truth  # noqa: E402
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
    parser.add_argument("--noise-only-ratio", type=float, default=0.12,
                        help="纯噪声场景比例（无标签，抑制虚警），默认 0.12")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--guard-hz", type=float, default=10_000.0,
                        help="频段两端与信号之间的保护间隔（Hz），默认 10000")
    parser.add_argument("--min-gap-hz", type=float, default=15_000.0,
                        help="相邻信号之间的最小间隔（Hz），默认 15000")
    parser.add_argument("--min-bandwidth-ratio", type=float, default=0.02,
                        help="最小占用带宽（占采样率），默认 0.02")
    parser.add_argument("--max-bandwidth-ratio", type=float, default=0.12,
                        help="最大占用带宽（占采样率），默认 0.12")
    parser.add_argument("--hop-rate-range", type=_pair, default=(10.0, 200.0),
                        help="跳频跳速区间（Hz），默认 10,200")
    return parser.parse_args(argv)


def _plan_slots(rng, args, count):
    """在 ±fs/2 内随机划分互不重叠的占用频段（保证标签框不重叠）。"""
    low_limit = -args.rate / 2.0 + args.guard_hz
    high_limit = args.rate / 2.0 - args.guard_hz
    slots = []
    cursor = low_limit
    for _ in range(count):
        width = rng.uniform(args.min_bandwidth_ratio, args.max_bandwidth_ratio) * args.rate
        room = high_limit - cursor
        if room < width + args.min_gap_hz:
            break
        start = cursor + rng.uniform(0.0, room - width)
        slots.append((float(start), float(start + width)))
        cursor = slots[-1][1] + args.min_gap_hz
    return slots


def _signal_spec(rng, mode, low, high, power_dbfs, args):
    """一个频段槽 → 信号参数（占用频带严格落在槽内，标签因此不会互相重叠）。"""
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


def _scene(rng, args):
    """随机生成一个场景，返回 ``(signals, noise)``。"""
    if rng.random() < args.noise_only_ratio:
        return [], {"enabled": True, "bandwidth": float(args.rate),
                    "power_dbfs": float(rng.uniform(-40.0, -10.0))}
    slots = _plan_slots(rng, args, int(args.max_signals))
    if not slots:
        return [], {"enabled": True, "bandwidth": float(args.rate),
                    "power_dbfs": float(rng.uniform(-40.0, -10.0))}
    modes = list(args.modes)
    rng.shuffle(modes)
    order = rng.permutation(len(slots))
    signals = []
    for rank, index in enumerate(order):
        low, high = slots[int(index)]
        # 最强信号功率定在 0 dBFS，其余按 0.5～6 dB 递减，让 noise.snr_db
        # （定义在最强信号上）就是本场景的目标带内信噪比
        power = 0.0 if rank == 0 else -float(rng.uniform(0.5, 6.0))
        signals.append(_signal_spec(rng, modes[rank % len(modes)], low, high, power, args))
    noise = {"enabled": True, "bandwidth": float(args.rate),
             "snr_db": float(rng.uniform(*args.snr_range))}
    return signals, noise


def _labels(meta, generation):
    """真值 → 归一化标签框 ``[x, y, w, h, confidence, class]``。"""
    boxes = []
    for entry in signal_truth(generation):
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
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"不支持的调制样式：{', '.join(unknown)}")
    args.modes = modes

    output = Path(args.output)
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    counts = {}
    widths = []
    snrs = []
    for index in range(args.count):
        record = None
        for _ in range(MAX_ATTEMPTS):
            seed = int(rng.integers(0, 2 ** 32))
            duration = float(rng.uniform(*args.duration_range))
            signals, noise = _scene(rng, args)
            try:
                samples, generation = generate_iq(args.rate, duration, signals,
                                                  noise=noise, seed=seed)
            except ValueError:
                continue  # 参数组合越界：重抽一个场景而不是让整批失败
            summary, arrays = spectral_context(samples, args.rate, {"nfft": args.nfft})
            image, meta = detection_image(arrays, summary, args.image_size, args.dynamic_range)
            boxes = _labels(meta, generation)
            name = f"{index:06d}.npy"
            np.save(images / name, image)
            described = _describe(generation)
            counts[len(described)] = counts.get(len(described), 0) + 1
            # box[3] 是"频率维"归一化高度 = 占用带宽 / 采样率
            widths.extend(box[3] for box in boxes)
            snrs.extend(entry["snr_inband_db"] for entry in described
                        if entry["snr_inband_db"] is not None)
            record = {
                "index": index,
                "image": f"images/{name}",
                "boxes": boxes,
                "truth": described,
                "scene": {
                    "rate_hz": float(args.rate),
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
            "box_columns": list(BOX_NAMES),
            "box_column_count": int(BOX_COLUMNS),
            "frequency_reference": "baseband_offset",
            "note": "行 0 = +fs/2，列 0 = t=0；四列几何量按图像宽高归一化",
        },
        "scene": {
            "sample_rate_hz": float(args.rate),
            "duration_range_s": [float(args.duration_range[0]), float(args.duration_range[1])],
            "modes": list(args.modes),
            "max_signals": int(args.max_signals),
            "snr_db_range": [float(args.snr_range[0]), float(args.snr_range[1])],
            "noise_only_ratio": float(args.noise_only_ratio),
            "guard_hz": float(args.guard_hz),
            "min_gap_hz": float(args.min_gap_hz),
        },
        "statistics": {
            "targets_per_sample": {str(key): value for key, value in sorted(counts.items())},
            "targets": sum(len(record["boxes"]) for record in records),
            "mean_bandwidth_ratio": float(np.mean(widths)) if widths else 0.0,
            "snr_inband_db": {
                "min": float(np.min(snrs)) if snrs else None,
                "max": float(np.max(snrs)) if snrs else None,
                "mean": float(np.mean(snrs)) if snrs else None,
            },
        },
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
          f"nfft {args.nfft}，动态范围 {args.dynamic_range:g} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
