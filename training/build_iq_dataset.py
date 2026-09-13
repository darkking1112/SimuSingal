#!/usr/bin/env python3
"""构建**原始 IQ** 调制识别数据集（``iq_waveform_v1`` 契约，供 :mod:`train_iq` 使用）。

为什么要单独一套数据集
----------------------

``build_amc_dataset.py`` 存的是 :func:`signal_analysis.ml.amc.extract_features` 产出的
34 维特征向量；本脚本存的是 :func:`signal_analysis.ml.iq.iq_waveform` 产出的
``(2, N)`` 单位 RMS 复基带窗口。两者是**两条互不替代的通路**（见 ``training/README.md``）：

* 特征通路：人工设计特征 + 线性/Transformer 分类头，类别字典冻结为 A09 六类；
* IQ 通路：网络自己学波形特征，类别字典由模型清单声明，可以更宽。

因此**不能**把特征向量当成 IQ，也不能把 TorchSig 的"信号类别"直接当成项目的调制类别。

两条数据来源
------------

1. **项目生成器**（默认）：与 ``build_amc_dataset.py`` 共用同一套场景参数化
   （单信号、占用带宽比例、分析窗口抖动、带内信噪比区间），只是把特征换成 IQ 窗口。
   场景抽样函数直接从 ``build_amc_dataset`` 导入——**同一份实现**，不会出现
   "两个数据集对同一场景的摆放规则不同"的隐性分叉。
2. **TorchSig 补充数据**（可选，``--torchsig-bundle``）：消费 ``build_torchsig.py``
   产出的 ``torchsig_bundle_v1``。TorchSig 的 ``class_name`` 是**它自己的**类别体系，
   这里**必须**由 ``--torchsig-map`` 显式给出到项目类别的映射；没映射到的类名原样
   记录在数据卡片的 ``unmapped_classes`` 里并跳过该记录，绝不猜。多条信号实例的记录
   （天生多类）也整条跳过并计数：IQ 分类通路一次只看一个占用频带。

标签与输入同源
--------------

每条样本的波形都来自 :func:`iq_waveform` 本身（推理端入口），因此"窗口长度 /
归一化 / 抽取比 / 窗口取中"这几条口径在训练与推理之间**只有一份实现**。
生成的 ``(2, N)`` 数组、``offset_hz``、``bandwidth_hz`` 也一并落盘，便于复盘。

产物::

    <output>/iq_dataset.json   # 契约 + 类别字典 + 两条来源的分段统计 + 场景参数
    <output>/iq_dataset.npz    # waveforms (M,2,N) float32、labels、snr_db、split、source…

用法::

    # 1) 只用项目生成器（纯 NumPy，不需要 torch）
    .venv/bin/python training/build_iq_dataset.py --output training/data/iq \\
        --per-class 200 --samples 1024 --seed 7

    # 2) 混入 TorchSig 补充数据（先用 build_torchsig.py 生成 bundle）
    .venv/bin/python training/build_iq_dataset.py --output training/data/iq \\
        --torchsig-bundle /data/ts_bundle --torchsig-map training/iq_map.example.json
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
for _extra in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

#: 场景参数化与 ``build_amc_dataset.py`` 共用（同一份实现，避免摆放规则分叉）
from build_amc_dataset import (  # noqa: E402
    MAX_ATTEMPTS,
    _analysis_window,
    _pair,
    _place_offset,
    _signal_spec,
)
from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.ml import amc  # noqa: E402
from signal_analysis.ml.iq import (  # noqa: E402
    CLASS_SET_A09,
    CLASS_SET_CUSTOM,
    DEFAULT_IQ_SAMPLES,
    IQ_INPUT_CHANNELS,
    IQ_LAYOUT,
    IQ_NORMALIZATION,
    IQ_WAVEFORM_CONTRACT,
    MAX_CLASSES,
    MAX_CLASS_NAME,
    MIN_IQ_SAMPLES,
    iq_waveform,
    class_set_name,
)
from torchsig_bundle import load_bundle  # noqa: E402

#: 数据来源标识（落在 npz 的 ``source`` 字段与卡片的 ``sources`` 分段统计里）
SOURCE_GENERATOR = "generator"
SOURCE_TORCHSIG = "torchsig"

#: TorchSig 记录被跳过的原因（卡片里逐项计数，便于定位 bundle 侧的问题）
SKIP_KINDS = ("unmapped_class", "multiple_signals", "missing_band", "too_short",
              "malformed")

OUTPUT_STEM = "iq_dataset"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="构建原始 IQ 调制识别数据集（iq_waveform_v1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--samples", type=int, default=DEFAULT_IQ_SAMPLES,
                        help="窗口长度（采样点）；必须与模型清单的 input.samples 一致")
    parser.add_argument("--per-class", type=int, default=200, help="每类生成器样本数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（决定场景与划分）")
    parser.add_argument("--rate", type=float, default=200_000.0, help="采样率（Hz）")
    parser.add_argument("--duration-range", type=_pair, default=(0.12, 0.30),
                        help="每场景持续时间区间（s）；太短会凑不满窗口")
    parser.add_argument("--snr-range", type=_pair, default=(-5.0, 30.0),
                        help="带内信噪比区间（dB）")
    parser.add_argument("--min-bandwidth-ratio", type=float, default=0.05,
                        help="最小占用带宽（占采样率）")
    parser.add_argument("--max-bandwidth-ratio", type=float, default=0.30,
                        help="最大占用带宽（占采样率）")
    parser.add_argument("--guard-ratio", type=float, default=0.02,
                        help="信号与采样带宽边缘的保护间隔（占采样率）")
    parser.add_argument("--center-jitter", type=float, default=0.05,
                        help="分析中心频率相对占用带宽的抖动比例")
    parser.add_argument("--bandwidth-jitter", type=float, default=0.10,
                        help="分析带宽相对真值带宽的抖动比例")
    parser.add_argument("--power-dbfs", type=float, default=-6.0, help="信号功率（dBFS）")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--modes", default=",".join(amc.AMC_CLASSES),
                        help="参与的生成器样式（默认 A09 六类对应的样式）")
    parser.add_argument("--class-set", choices=(CLASS_SET_A09, CLASS_SET_CUSTOM),
                        default=CLASS_SET_A09,
                        help="类别字典：a09 = 技术方案 A09 六类；custom = 由 --classes 指定")
    parser.add_argument("--classes", default="", help="custom 类别字典，逗号分隔")
    parser.add_argument("--torchsig-bundle", default="",
                        help="torchsig_bundle_v1 目录；给出后才会混入 TorchSig 数据")
    parser.add_argument("--torchsig-map", default="",
                        help="TorchSig class_name → 本项目类别的 JSON 映射文件（给出 bundle 时必填）")
    parser.add_argument("--torchsig-per-class", type=int, default=0,
                        help="TorchSig 每类最多取多少条（0 = 全部可用）")
    return parser.parse_args(argv)


def _classes(args):
    """解析并校验类别字典（顺序即模型输出下标顺序）。"""
    if args.class_set == CLASS_SET_CUSTOM:
        names = [name.strip() for name in str(args.classes).split(",") if name.strip()]
        if not names:
            raise SystemExit("--class-set custom 时必须用 --classes 给出类别名（逗号分隔）")
    else:
        if str(args.classes).strip():
            raise SystemExit("--class-set a09 时不能再用 --classes 覆盖（A09 六类是冻结字典）")
        names = list(amc.AMC_CLASSES)
    if len(names) > MAX_CLASSES:
        raise SystemExit(f"类别数 {len(names)} 超过上限 {MAX_CLASSES}")
    if len(set(names)) != len(names):
        raise SystemExit("类别名重复")
    for name in names:
        if len(name) > MAX_CLASS_NAME:
            raise SystemExit(f"类别名 {name!r} 超过 {MAX_CLASS_NAME} 字符")
    return names


def _read_map(path):
    """读 ``TorchSig class_name → 项目类别`` 映射（缺失/非法一律报错，不猜）。"""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"--torchsig-map 指向的文件不存在：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("映射文件应是一个 JSON 对象：{\"torchsig 类名\": \"项目类别\"}")
    mapping = {}
    for key, value in payload.items():
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"映射 {key!r} 的目标类别必须是非空字符串")
        mapping[str(key)] = value.strip()
    if not mapping:
        raise SystemExit("映射文件为空，无法把 TorchSig 类名接到项目类别上")
    return mapping


def _rounded(value, digits=3):
    return None if value is None else round(float(value), digits)


def _generator_sample(rng, args, mode):
    """抽一个生成器场景 → 一个 IQ 窗口样本（参数越界时返回 ``None`` 由调用方重抽）。"""
    bandwidth = float(rng.uniform(args.min_bandwidth_ratio, args.max_bandwidth_ratio)) * args.rate
    spec = _signal_spec(rng, args, mode, bandwidth, args.power_dbfs)
    placed = _place_offset(rng, args, spec)
    if placed is None:
        return None
    _, low, high = placed
    center, window = _analysis_window(rng, args, low, high)
    snr_db = float(rng.uniform(*args.snr_range))
    duration = float(rng.uniform(*args.duration_range))
    seed = int(rng.integers(0, 2 ** 32))
    samples, generation = generate_iq(
        args.rate, duration, [spec],
        noise={"enabled": True, "bandwidth": float(args.rate), "snr_db": snr_db},
        seed=seed)
    try:
        tensor, meta = iq_waveform(samples, args.rate, center, window,
                                   window_samples=args.samples)
    except ValueError as exc:
        # 记录太短或频带太窄导致凑不满窗口：换一个场景重抽（不补零，见 iq_waveform）
        if "可用样本" in str(exc) or "有效样本" in str(exc):
            return None
        raise
    truth = generation["signals"][0]
    return {
        "label": amc.mode_to_class(mode),
        "mode": mode,
        "source": SOURCE_GENERATOR,
        "snr_db": _rounded(snr_db),
        "snr_estimate_db": meta["snr_estimate_db"],
        "waveform": tensor,
        "analysis": {"offset_hz": _rounded(center), "bandwidth_hz": _rounded(window)},
        "sample_rate_hz": float(args.rate),
        "waveform_info": meta,
        "truth": {
            "center_hz": _rounded(0.5 * (low + high)),
            "bandwidth_hz": _rounded(high - low),
            "power_dbfs": truth.get("power_dbfs_actual"),
            "snr_inband_db": truth.get("snr_inband_db"),
        },
        "scene": {
            "rate_hz": float(args.rate),
            "duration_s": duration,
            "seed": seed,
            "signals": [spec],
            # 噪声信噪比保留全精度（``snr_db`` 的三位小数只用于分档统计）
            "noise": {"enabled": True, "bandwidth": float(args.rate), "snr_db": snr_db},
        },
    }


def _component_band(component):
    """信号实例 → ``(center_hz, bandwidth_hz)``；缺几何信息返回 ``None``。"""
    if not isinstance(component, dict):
        return None
    center = component.get("center_freq")
    width = component.get("bandwidth")
    low, high = component.get("lower_freq"), component.get("upper_freq")
    try:
        if center is not None and width is not None:
            center, width = float(center), float(width)
        elif low is not None and high is not None:
            low, high = float(low), float(high)
            center, width = 0.5 * (low + high), abs(high - low)
        else:
            return None
    except (TypeError, ValueError):
        return None
    if not math.isfinite(center) or not math.isfinite(width) or width <= 0:
        return None
    return center, width


def _torchsig_sample(rng, bundle, entry, meta, args, mapping, skips, unmapped, per_class):
    """bundle 里的一条记录 → 一个 IQ 窗口样本；不可用时返回 ``None`` 并计数。"""
    components = meta.get("components")
    if not isinstance(components, list) or not components:
        skips["malformed"] = skips.get("malformed", 0) + 1
        return None
    names = [str(item.get("class_name", "unknown")) if isinstance(item, dict)
             else "unknown" for item in components]
    unknown = sorted({name for name in names if name not in mapping})
    if unknown:
        for name in unknown:
            unmapped[name] = unmapped.get(name, 0) + 1
        skips["unmapped_class"] = skips.get("unmapped_class", 0) + 1
        return None
    labels = {mapping[name] for name in names}
    if len(labels) != 1:
        # 多信号记录 → 一个频带里不可能只有一类：整条跳过（绝不挑一个当真值）
        skips["multiple_signals"] = skips.get("multiple_signals", 0) + 1
        return None
    label = labels.pop()
    if label not in per_class:
        skips["malformed"] = skips.get("malformed", 0) + 1
        return None
    band = _component_band(components[0])
    if band is None:
        skips["missing_band"] = skips.get("missing_band", 0) + 1
        return None
    center, width = band
    rate = float(bundle["sample_rate_hz"])
    record_rate = meta.get("sample_rate_hz")
    if isinstance(record_rate, (int, float)) and not isinstance(record_rate, bool) and record_rate > 0:
        # 记录自带采样率以记录内声明为准（bundle 级只是默认值）
        rate = float(record_rate)
    samples = np.load(entry["iq"])
    if samples.ndim == 2 and samples.shape[1] == 2:
        samples = samples[:, 0] + 1j * samples[:, 1]
    return {
        "label": label,
        "mode": None,
        "source": SOURCE_TORCHSIG,
        "band": (center, width),
        "samples": samples,
        "rate": rate,
        "snr_db": _rounded(components[0].get("snr_db")),
        "record_index": int(entry.get("index", -1)),
        "class_names": names,
    }


def _torchsig_window(sample, args):
    """把 bundle 记录混频到它的占用频带并取窗口（凑不满窗口则计数跳过）。"""
    try:
        tensor, meta = iq_waveform(sample["samples"], sample["rate"],
                                   sample["band"][0], sample["band"][1],
                                   window_samples=args.samples)
    except ValueError:
        return None, None
    return tensor, meta


def _split_by_index(position, total, fraction):
    """确定性划分：按位置取前 ``fraction`` 比例进训练集（同一 seed/输入必然同结果）。"""
    return "train" if position < max(1, int(round(total * fraction))) else "val"


def _statistics(records, classes):
    snrs = [record["snr_db"] for record in records if record["snr_db"] is not None]
    return {
        "labels": {name: sum(1 for record in records if record["label"] == name)
                   for name in classes},
        "snr_db": ({
            "min": _rounded(min(snrs)), "max": _rounded(max(snrs)),
            "mean": _rounded(sum(snrs) / len(snrs)),
        } if snrs else None),
        "sources": {source: sum(1 for record in records if record["source"] == source)
                    for source in sorted({record["source"] for record in records})},
    }


def main(argv=None):
    args = _parse_args(argv)
    if not MIN_IQ_SAMPLES <= args.samples <= 65536:
        raise SystemExit(f"--samples 应在 {MIN_IQ_SAMPLES}～65536 之间")
    if args.per_class < 1:
        raise SystemExit("每类样本数必须大于 0")
    if not 0.0 < args.train_fraction < 1.0:
        raise SystemExit("训练集比例应在 0～1 之间")
    if not 0.0 < args.min_bandwidth_ratio <= args.max_bandwidth_ratio < 1.0:
        raise SystemExit("占用带宽比例区间非法")
    classes = _classes(args)
    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in amc.AMC_CLASSES]
    if unknown:
        raise SystemExit(f"未知样式：{', '.join(unknown)}；可用：{', '.join(amc.AMC_CLASSES)}")
    covered = {amc.mode_to_class(mode) for mode in modes}
    missing = [name for name in covered if name not in classes]
    if missing:
        raise SystemExit(f"样式映射出的类别不在类别字典内：{', '.join(missing)}")
    args.modes = modes

    mapping, bundle = {}, None
    if args.torchsig_bundle:
        if not args.torchsig_map:
            raise SystemExit(
                "给出 --torchsig-bundle 时必须同时给出 --torchsig-map：\n"
                "TorchSig 的 class_name 属于它自己的类别体系，本项目不会替你猜映射。")
        mapping = _read_map(args.torchsig_map)
        outside = sorted({name for name in mapping.values() if name not in classes})
        if outside:
            raise SystemExit(f"映射目标不在类别字典内：{', '.join(outside)}")
        bundle = load_bundle(args.torchsig_bundle)
    elif args.torchsig_map:
        raise SystemExit("只给了 --torchsig-map 而没有 --torchsig-bundle")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []

    total = len(modes) * args.per_class
    index = 0
    for mode in modes:
        for class_index in range(args.per_class):
            record = None
            for _ in range(MAX_ATTEMPTS):
                candidate = _generator_sample(rng, args, mode)
                if candidate is not None:
                    record = candidate
                    break
            if record is None:
                raise SystemExit(
                    f"{mode} 连续 {MAX_ATTEMPTS} 次生成失败（常见原因：持续时长不足以填满 "
                    f"{args.samples} 点窗口，或频带过窄），请放宽频带比例或加长 --duration-range")
            record["index"] = index
            record["split"] = "train" if class_index < round(args.per_class * args.train_fraction) \
                else "val"
            records.append(record)
            index += 1
            if index % 50 == 0 or index == total:
                print(f"  生成器样本 {index}/{total}", flush=True)

    skips, unmapped = {}, {}
    bundle_summary = None
    if bundle is not None:
        per_class = {name: 0 for name in classes}
        limit = int(args.torchsig_per_class)
        rate = float(bundle["sample_rate_hz"])
        print(f"  混入 TorchSig bundle：{args.torchsig_bundle}（{len(bundle['records'])} 条记录）")
        for entry in bundle["records"]:
            meta_path = Path(args.torchsig_bundle) / str(entry["meta"])
            iq_path = Path(args.torchsig_bundle) / str(entry["iq"])
            if not meta_path.is_file():
                skips["malformed"] = skips.get("malformed", 0) + 1
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sample = _torchsig_sample(rng, bundle, {"index": entry.get("index"),
                                                    "iq": iq_path},
                                      meta, args, mapping, skips, unmapped, per_class)
            if sample is None:
                continue
            if limit and per_class[sample["label"]] >= limit:
                continue
            tensor, wave_meta = _torchsig_window(sample, args)
            if tensor is None:
                skips["too_short"] = skips.get("too_short", 0) + 1
                continue
            position = per_class[sample["label"]]
            per_class[sample["label"]] = position + 1
            records.append({
                "label": sample["label"],
                "mode": None,
                "source": SOURCE_TORCHSIG,
                "snr_db": sample["snr_db"],
                "snr_estimate_db": wave_meta["snr_estimate_db"],
                "waveform": tensor,
                "analysis": {"offset_hz": _rounded(sample["band"][0]),
                             "bandwidth_hz": _rounded(sample["band"][1])},
                "sample_rate_hz": sample["rate"],
                "waveform_info": wave_meta,
                "truth": {"class_names": sample["class_names"],
                          "mapping": {name: mapping[name] for name in sample["class_names"]}},
                "scene": {"bundle": str(args.torchsig_bundle),
                          "record_index": sample["record_index"],
                          "sample_rate_hz": sample["rate"]},
                "index": index,
                "position": position,
                "split": None,
            })
            index += 1
        # TorchSig 各类的实际条数只有跑完才知道，因此划分放到这里统一按类内位置切分
        # （与生成器侧的"每类内部按比例切分"同口径，不会出现某类全落在验证集）
        for record in records:
            if record["source"] == SOURCE_TORCHSIG:
                record["split"] = _split_by_index(
                    record["position"], per_class[record["label"]], args.train_fraction)
        bundle_summary = {
            "path": str(args.torchsig_bundle),
            "bundle_version": bundle.get("bundle_version"),
            "torchsig_version": bundle.get("torchsig_version"),
            "sample_rate_hz": rate,
            "record_count": len(bundle["records"]),
            "mapping": dict(sorted(mapping.items())),
            "per_class_limit": limit,
            "skipped": dict(sorted(skips.items())),
        }

    if not records:
        raise SystemExit("没有任何可用样本，请检查参数或 bundle 内容")

    waveforms = np.stack([record["waveform"] for record in records]).astype(np.float32)
    labels = np.asarray([record["label"] for record in records])
    splits = np.asarray([record["split"] for record in records])
    sources = np.asarray([record["source"] for record in records])
    snrs = np.asarray([np.nan if record["snr_db"] is None else float(record["snr_db"])
                       for record in records], dtype=np.float32)
    offsets = np.asarray([float(record["analysis"]["offset_hz"]) for record in records])
    windows = np.asarray([float(record["analysis"]["bandwidth_hz"]) for record in records])
    rates = np.asarray([float(record["sample_rate_hz"]) for record in records])
    assert waveforms.shape[1:] == (IQ_INPUT_CHANNELS, args.samples), waveforms.shape

    npz_path = output / f"{OUTPUT_STEM}.npz"
    np.savez(npz_path, waveforms=waveforms, labels=labels, split=splits, source=sources,
             snr_db=snrs, offset_hz=offsets, bandwidth_hz=windows, sample_rate_hz=rates,
             index=np.asarray([record["index"] for record in records], dtype=np.int64))

    train_count = int((splits == "train").sum())
    card = {
        "generator_script": "training/build_iq_dataset.py",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "sample_count": len(records),
        "store": {"file": npz_path.name, "format": "npz",
                  "arrays": ["waveforms", "labels", "split", "source", "snr_db",
                             "offset_hz", "bandwidth_hz", "sample_rate_hz", "index"]},
        "contract": {
            "task": "amc_iq",
            "input_contract": IQ_WAVEFORM_CONTRACT,
            "layout": IQ_LAYOUT,
            "channels": IQ_INPUT_CHANNELS,
            "normalization": IQ_NORMALIZATION,
            "samples": int(args.samples),
            "class_set": class_set_name(classes),
            "classes": list(classes),
            "class_count": len(classes),
            "mode_to_class": {mode: amc.mode_to_class(mode) for mode in amc.AMC_CLASSES},
            "note": "波形由 signal_analysis.ml.iq.iq_waveform 生成，训练与推理同源；"
                    "类别顺序即模型输出下标顺序",
        },
        "splits": {
            "strategy": "stratified_per_class",
            "train_fraction": float(args.train_fraction),
            "train": train_count,
            "val": len(records) - train_count,
            "per_class": {
                name: {
                    "train": int(((labels == name) & (splits == "train")).sum()),
                    "val": int(((labels == name) & (splits == "val")).sum()),
                }
                for name in classes
            },
            "note": "TorchSig 记录按各类内部出现顺序划分（无跨来源混批）",
        },
        "sources": {
            SOURCE_GENERATOR: {
                "script": "signal_analysis.core_api.generate_iq（本项目生成器）",
                "sample_rate_hz": float(args.rate),
                "duration_range_s": [float(args.duration_range[0]), float(args.duration_range[1])],
                "snr_db_range": [float(args.snr_range[0]), float(args.snr_range[1])],
                "modes": list(modes),
                "per_class": int(args.per_class),
                "min_bandwidth_ratio": float(args.min_bandwidth_ratio),
                "max_bandwidth_ratio": float(args.max_bandwidth_ratio),
                "guard_ratio": float(args.guard_ratio),
                "noise": "全采样带宽带内信噪比（noise.snr_definition = inband_snr_v1）",
                "analysis_window": {
                    "center_jitter_ratio": float(args.center_jitter),
                    "bandwidth_jitter_ratio": float(args.bandwidth_jitter),
                    "note": "在真值上叠加相对抖动，模拟检测器给出的中心频率/带宽估计误差",
                },
            },
            SOURCE_TORCHSIG: bundle_summary,
            "unmapped_classes": dict(sorted(unmapped.items())),
            "unmapped_note": "TorchSig 类名原样记录、不做猜测；这些记录未进入数据集",
        },
        "statistics": _statistics(records, classes),
    }
    (output / f"{OUTPUT_STEM}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"数据集已写入 {npz_path}（{len(records)} 个样本，窗口 {args.samples} 点）")
    print(f"  类别：{card['statistics']['labels']}")
    print(f"  来源：{card['statistics']['sources']}")
    if bundle_summary:
        print(f"  TorchSig 跳过：{bundle_summary['skipped']}")
        if unmapped:
            print(f"  未映射类名（已跳过）：{card['sources']['unmapped_classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
