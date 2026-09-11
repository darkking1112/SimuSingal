#!/usr/bin/env python3
"""构建 A09 六类调制识别训练集：单信号场景 → 定长特征向量。

设计要点（与推理端共用同一份契约，见 ``training/README.md``）：

* 类别字典**按技术方案 A09 原文主体**固定为六类：FM、SSB、2ASK、QPSK、16QAM、64QAM；
  生成器里的 ``am`` 与跳频样式不在其中（跳频是会话级检测对象，AM 只出现在原文
  数据库示例里），因此不进入本数据集。
* 特征由 :func:`signal_analysis.ml.amc.extract_features` 提取——训练与推理走**同一个
  函数**，不可能出现"训练-推理口径分叉"；特征顺序由 ``AMC_FEATURES`` 冻结。
* 每个场景都是**单信号**：AMC 的输入是检测器切分出来的一个占用频带。分析窗口
  （中心频率 + 带宽）按"检测估计值"模拟：在真值上叠加可配置的相对抖动，
  使模型对检测误差有容忍度（默认中心 ±5%、带宽 ±10%）。
* 只依赖 NumPy 与项目自身代码，不需要 torch；数据完全由本项目生成器合成，
  无第三方数据集的许可证约束。

产物::

    <output>/amc_dataset.json   # 契约 + 生成参数 + 统计（训练脚本据此校验）
    <output>/features.jsonl     # 每行一个样本：特征、类别、SNR、分析窗口、场景

用法::

    .venv/bin/python training/build_amc_dataset.py --output training/data/amc \\
        --per-class 400 --seed 7
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

from signal_analysis._numeric import (  # noqa: E402
    _check_band,
    occupied_interval,
    plan_signal,
)
from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.ml import amc  # noqa: E402

#: A09 六类（`am` 与跳频不在字典内）
DEFAULT_MODES = tuple(amc.AMC_CLASSES)
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
    parser = argparse.ArgumentParser(description="构建 A09 六类调制识别数据集（特征向量）")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--per-class", type=int, default=400, help="每类样本数，默认 400")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（决定场景与划分）")
    parser.add_argument("--rate", type=float, default=200_000.0, help="采样率（Hz），默认 2e5")
    parser.add_argument("--duration-range", type=_pair, default=(0.03, 0.12),
                        help="每场景持续时间区间（s），默认 0.03,0.12")
    parser.add_argument("--snr-range", type=_pair, default=(-5.0, 30.0),
                        help="带内信噪比区间（dB），默认 -5,30")
    parser.add_argument("--min-bandwidth-ratio", type=float, default=0.05,
                        help="最小占用带宽（占采样率），默认 0.05")
    parser.add_argument("--max-bandwidth-ratio", type=float, default=0.30,
                        help="最大占用带宽（占采样率），默认 0.30")
    parser.add_argument("--guard-ratio", type=float, default=0.02,
                        help="信号与采样带宽边缘的保护间隔（占采样率），默认 0.02")
    parser.add_argument("--center-jitter", type=float, default=0.05,
                        help="分析中心频率相对占用带宽的抖动比例，默认 0.05")
    parser.add_argument("--bandwidth-jitter", type=float, default=0.10,
                        help="分析带宽相对真值带宽的抖动比例，默认 0.10")
    parser.add_argument("--power-dbfs", type=float, default=-6.0, help="信号功率（dBFS），默认 -6")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="训练集比例，默认 0.8")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES),
                        help=f"参与的类别，默认全部六类：{', '.join(DEFAULT_MODES)}")
    return parser.parse_args(argv)


def _signal_spec(rng, args, mode, bandwidth, power_dbfs):
    """生成单信号参数（``offset`` 先置 0，稍后按实际占用区间摆放）。"""
    spec = {"mode": mode, "offset": 0.0, "bandwidth": float(bandwidth),
            "power_dbfs": float(power_dbfs)}
    if mode == "ssb":
        spec["side"] = "usb" if rng.random() < 0.5 else "lsb"
    elif mode == "fm":
        # deviation 决定实际占用带宽（5.3×deviation），上限压在声明带宽之内
        spec["deviation"] = float(rng.uniform(0.10, 0.19) * bandwidth)
        spec["message_bandwidth"] = float(rng.uniform(0.1, 0.3) * bandwidth)
    elif mode in ("ask2", "qpsk", "qam16", "qam64"):
        spec["alpha"] = float(rng.choice((0.2, 0.35, 0.5)))
    return spec


def _place_offset(rng, args, spec):
    """按 ``plan_signal`` 得到的**实际占用区间**把信号摆进采样带宽内。

    SSB 的频带校验口径是"两侧各留一个带宽"（见 ``_numeric._check_band``），
    因此这里统一按一个**外包箱**摆放：非 SSB 为占用区间本身，SSB 为
    ``[offset - w, offset + w]``，保证 ``plan_signal`` 不会越界。
    """
    rate = float(args.rate)
    mode = spec["mode"]
    plan = plan_signal(spec, rate)
    low, high = occupied_interval(plan["offset"], plan["bandwidth_actual"], mode,
                                  plan.get("side"))
    need = max(float(high - low), float(plan["bandwidth"]), float(plan["bandwidth_actual"]))
    half = need if mode == "ssb" else need / 2.0
    guard = float(args.guard_ratio) * rate
    room = rate - 2.0 * guard - 2.0 * half
    if room <= 0:
        return None
    start = -rate / 2.0 + guard + float(rng.uniform(0.0, room))
    spec["offset"] = start + need if mode == "ssb" else start + half
    plan = plan_signal(spec, rate)
    _check_band(plan["offset"], plan["bandwidth"], rate, mode)
    low, high = occupied_interval(plan["offset"], plan["bandwidth_actual"], mode,
                                  plan.get("side"))
    return plan, float(low), float(high)


def _analysis_window(rng, args, low, high):
    """真值占用区间 → 带抖动的分析窗口 ``(center_hz, bandwidth_hz)``（模拟检测估计）。"""
    rate = float(args.rate)
    occupied = float(high - low)
    center = 0.5 * (low + high)
    center += float(rng.uniform(-1.0, 1.0)) * float(args.center_jitter) * occupied
    bandwidth = occupied * float(
        rng.uniform(1.0 - float(args.bandwidth_jitter), 1.0 + float(args.bandwidth_jitter)))
    bandwidth = float(min(max(bandwidth, 1.0 / 64.0), rate))
    center = float(min(max(center, -rate / 2.0), rate / 2.0))
    return center, bandwidth


def _scene(rng, args, mode):
    """抽一个场景并提取特征；参数越界时返回 ``None`` 由调用方重抽。"""
    bandwidth = float(rng.uniform(args.min_bandwidth_ratio, args.max_bandwidth_ratio)) * args.rate
    spec = _signal_spec(rng, args, mode, bandwidth, args.power_dbfs)
    placed = _place_offset(rng, args, spec)
    if placed is None:
        return None
    plan, low, high = placed
    center, window = _analysis_window(rng, args, low, high)
    snr_db = float(rng.uniform(*args.snr_range))
    duration = float(rng.uniform(*args.duration_range))
    seed = int(rng.integers(0, 2 ** 32))
    samples, generation = generate_iq(
        args.rate, duration, [spec],
        noise={"enabled": True, "bandwidth": float(args.rate), "snr_db": snr_db},
        seed=seed)
    features, info = amc.extract_features(samples, args.rate, center, window)
    truth = generation["signals"][0]
    return {
        "label": amc.mode_to_class(mode),
        "mode": mode,
        "snr_db": round(snr_db, 3),
        "features": features,
        "analysis": {"offset_hz": round(center, 3), "bandwidth_hz": round(window, 3)},
        "truth": {
            "center_hz": round(0.5 * (low + high), 3),
            "bandwidth_hz": round(high - low, 3),
            "power_dbfs": truth.get("power_dbfs_actual"),
            "snr_inband_db": truth.get("snr_inband_db"),
        },
        "extraction": info,
        "scene": {
            "rate_hz": float(args.rate),
            "duration_s": duration,
            "seed": seed,
            "signals": [spec],
            # 场景备份用于精确复现波形，因此噪声信噪比保留全精度（对比 ``snr_db`` 字段的
            # 三位小数只用于分档统计）；分析窗口则允许 1 mHz 级的四舍五入误差。
            "noise": {"enabled": True, "bandwidth": float(args.rate), "snr_db": snr_db},
        },
    }


def main(argv=None):
    args = _parse_args(argv)
    if args.per_class < 1:
        raise SystemExit("每类样本数必须大于 0")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("训练集比例应在 0～1 之间")
    if not 0.0 < args.min_bandwidth_ratio <= args.max_bandwidth_ratio < 1.0:
        raise SystemExit("占用带宽比例区间非法")
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in DEFAULT_MODES]
    if unknown:
        raise SystemExit(f"不在 A09 六类字典内的样式：{', '.join(unknown)}")
    args.modes = modes

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    total = len(modes) * args.per_class
    per_class_train = round(args.per_class * args.train_fraction)
    index = 0
    for mode in modes:
        for class_index in range(args.per_class):
            record = None
            for _ in range(MAX_ATTEMPTS):
                candidate = _scene(rng, args, mode)
                if candidate is not None:
                    record = candidate
                    break
            if record is None:
                raise SystemExit(f"{mode} 连续 {MAX_ATTEMPTS} 次生成失败，请放宽频带比例区间")
            record["index"] = index
            # 分层划分：每类内部按比例切分，避免出现“某一类全部落在验证集”
            record["split"] = "train" if class_index < per_class_train else "val"
            records.append(record)
            index += 1
            if index % 50 == 0 or index == total:
                print(f"  已生成 {index}/{total} 个样本", flush=True)

    snrs = [record["snr_db"] for record in records]
    bandwidth_ratios = [record["truth"]["bandwidth_hz"] / args.rate for record in records]
    train_count = sum(1 for record in records if record["split"] == "train")
    card = {
        "generator_script": "training/build_amc_dataset.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "sample_count": len(records),
        "splits": {
            "strategy": "stratified_per_class",
            "train_fraction": float(args.train_fraction),
            "train": train_count,
            "val": len(records) - train_count,
            "per_class": {
                mode: {
                    "train": sum(1 for record in records
                                 if record["mode"] == mode and record["split"] == "train"),
                    "val": sum(1 for record in records
                               if record["mode"] == mode and record["split"] == "val"),
                }
                for mode in modes
            },
        },
        "contract": {
            "task": "amc",
            "classes": list(amc.AMC_CLASSES),
            "labels": dict(amc.CLASS_LABELS),
            "feature_contract": amc.AMC_FEATURE_CONTRACT,
            "features": list(amc.AMC_FEATURES),
            "feature_count": len(amc.AMC_FEATURES),
            "mode_to_class": {mode: amc.mode_to_class(mode) for mode in DEFAULT_MODES},
            "note": "特征由 signal_analysis.ml.amc.extract_features 生成，训练与推理同源",
        },
        "scene": {
            "sample_rate_hz": float(args.rate),
            "duration_range_s": [float(args.duration_range[0]), float(args.duration_range[1])],
            "snr_db_range": [float(args.snr_range[0]), float(args.snr_range[1])],
            "modes": list(args.modes),
            "per_class": int(args.per_class),
            "min_bandwidth_ratio": float(args.min_bandwidth_ratio),
            "max_bandwidth_ratio": float(args.max_bandwidth_ratio),
            "guard_ratio": float(args.guard_ratio),
            "noise": "全采样带宽带内信噪比（noise.snr_definition = inband_snr_v1）",
        },
        "analysis_window": {
            "center_jitter_ratio": float(args.center_jitter),
            "bandwidth_jitter_ratio": float(args.bandwidth_jitter),
            "note": "在真值上叠加相对抖动，模拟检测器给出的中心频率/带宽估计误差",
        },
        "statistics": {
            "labels": {mode: sum(1 for record in records if record["mode"] == mode)
                       for mode in modes},
            "snr_db": {"min": float(np.min(snrs)), "max": float(np.max(snrs)),
                       "mean": float(np.mean(snrs))},
            "bandwidth_ratio": {"min": float(np.min(bandwidth_ratios)),
                                "max": float(np.max(bandwidth_ratios)),
                                "mean": float(np.mean(bandwidth_ratios))},
        },
    }
    (output / "amc_dataset.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "features.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"数据集已写入 {output}")
    print(f"  样本 {card['sample_count']}（train {train_count} / val "
          f"{card['sample_count'] - train_count}），特征 {card['contract']['feature_count']} 维")
    print(f"  类别：{card['statistics']['labels']}")
    print(f"  带内信噪比 {card['statistics']['snr_db']['min']:.1f}～"
          f"{card['statistics']['snr_db']['max']:.1f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
