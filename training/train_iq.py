#!/usr/bin/env python3
"""训练原始 IQ 分类模型（1D CNN / TCN），导出 ONNX 并写出 ``iq_waveform_v1`` 清单。

用法::

    # 1) 生成数据集（只依赖 NumPy）
    .venv/bin/python training/build_iq_dataset.py --output training/data/iq --per-class 200

    # 2) 训练并导出（需要 pip install .[train]；CNN 在 CPU 上几分钟量级）
    .venv/bin/python training/train_iq.py --data training/data/iq --arch cnn

    # 3) 验收（清单 / ONNX 契约 / 端到端 / 可复现）
    .venv/bin/python training/verify_iq.py --manifest training/data/iq/onnx/iq_manifest.json

约定
----
* 训练与推理共用 :func:`signal_analysis.ml.iq.iq_waveform` 产出的窗口——数据集里存的
  就是推理端会拿到的**同一种** ``(2, N)`` 单位 RMS 张量，窗口长度由数据集契约给出；
* 类别顺序即模型输出下标顺序，写进清单的 ``output.classes``，推理端按同一顺序解读；
* 报告里的指标一律区分"训练集内"与"独立验证集"；本仓库**没有**约定的识别准确率
  合格门限（技术方案待确认项），因此脚本不会用阈值判定通过/失败，只打印事实。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from signal_analysis.evaluation import classification_metrics  # noqa: E402
from signal_analysis.ml.iq import (  # noqa: E402
    IQ_DEFAULT_MODEL_NAME,
    IQ_INPUT_CHANNELS,
    IQ_WAVEFORM_CONTRACT,
    MIN_IQ_SAMPLES,
    write_iq_manifest,
)

#: 训练脚本自身产出的清单文件名
MANIFEST_NAME = "iq_manifest.json"
#: 需要安装的可选依赖
TRAIN_EXTRA = "pip install .[train]"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="训练原始 IQ 分类模型（CNN/TCN）并导出 ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", required=True, help="数据集目录（build_iq_dataset.py 的输出）")
    parser.add_argument("--arch", choices=("cnn", "tcn"), default="cnn", help="网络结构")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", action="store_true", help="输出 GUI 结构化进度")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=8, help="验证准确率不升即早停的轮数")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout 比例")
    parser.add_argument("--channels", default="", help="卷积通道数，逗号分隔（默认按结构取）")
    parser.add_argument("--kernel", type=int, default=0, help="卷积核长度（0 = 按结构取默认）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--min-snr", type=float, default=None,
                        help="只用带内信噪比不低于该值的样本训练（默认不过滤）")
    parser.add_argument("--onnx-dir", default="", help="ONNX 与清单输出目录，默认 <data>/onnx")
    parser.add_argument("--identifier", default="iq-cnn", help="模型标识（写进清单 id）")
    parser.add_argument("--version", default="0.1.0", help="模型版本号")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--threads", type=int, default=None, help="ONNX 校验线程数")
    parser.add_argument("--default-offset-hz", type=float, default=None,
                        help="写进清单的默认分析中心（不给则不带默认值）")
    parser.add_argument("--default-bandwidth-hz", type=float, default=None,
                        help="写进清单的默认分析带宽（不给则不带默认值）")
    parser.add_argument("--note", default="", help="附加到训练信息的备注")
    return parser.parse_args(argv)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise SystemExit(f"训练原始 IQ 模型需要 PyTorch 与 onnx：{TRAIN_EXTRA}\n"
                         f"（原始错误：{exc}）") from exc


def load_dataset(directory):
    """读取 ``iq_dataset.npz`` 并校验契约与形状（不匹配就报错，不静默训练）。

    返回 ``(card, waveforms, labels, splits, snr_db, source)``。
    """
    directory = Path(directory)
    card_path = directory / "iq_dataset.json"
    store_path = directory / "iq_dataset.npz"
    if not card_path.is_file() or not store_path.is_file():
        raise SystemExit(f"数据集不完整：请先运行 training/build_iq_dataset.py 生成 {directory}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    contract = card.get("contract") or {}
    if contract.get("input_contract") != IQ_WAVEFORM_CONTRACT:
        raise SystemExit(f"数据集输入契约 {contract.get('input_contract')!r} 与 "
                         f"{IQ_WAVEFORM_CONTRACT} 不一致，请重新生成数据集")
    classes = list(contract.get("classes") or [])
    if not classes:
        raise SystemExit("数据集类别字典为空，请重新生成数据集")
    samples = contract.get("samples")
    if not isinstance(samples, int) or not MIN_IQ_SAMPLES <= samples <= 65536:
        raise SystemExit(f"数据集声明的窗口长度 {samples!r} 非法，请重新生成数据集")
    if int(contract.get("channels", IQ_INPUT_CHANNELS)) != IQ_INPUT_CHANNELS:
        raise SystemExit("数据集声明的通道数与 iq_waveform_v1 的 2 通道不符，请重新生成")

    with np.load(store_path) as store:
        missing = [name for name in ("waveforms", "labels", "split", "snr_db", "source")
                   if name not in store]
        if missing:
            raise SystemExit(f"数据集缺少字段：{', '.join(missing)}")
        waveforms = np.asarray(store["waveforms"], dtype=np.float32)
        labels = np.asarray(store["labels"]).astype(str)
        splits = np.asarray(store["split"]).astype(str)
        snrs = np.asarray(store["snr_db"], dtype=np.float64)
        sources = np.asarray(store["source"]).astype(str)
    if waveforms.ndim != 3 or waveforms.shape[1:] != (IQ_INPUT_CHANNELS, samples):
        raise SystemExit(f"波形形状 {waveforms.shape} 与契约声明的 "
                         f"({IQ_INPUT_CHANNELS}, {samples}) 不符，请重新生成数据集")
    if not np.isfinite(waveforms).all():
        raise SystemExit("波形里存在非有限值，请检查生成参数")
    outside = sorted(set(labels) - set(classes))
    if outside:
        raise SystemExit(f"数据集里出现了类别字典之外的标签：{', '.join(outside)}")
    if len(waveforms) != len(labels) or len(labels) != len(splits):
        raise SystemExit("数据集各字段长度不一致")
    return card, waveforms, labels, splits, snrs, sources


def _split(waveforms, labels, splits, snrs):
    if not set(splits).issubset({"train", "val", "test"}):
        raise SystemExit("数据划分只允许 train / val / test")
    train_mask = splits == "train"
    val_mask = splits == "val"
    if not train_mask.any():
        raise SystemExit("数据集没有训练划分")
    if not val_mask.any():
        raise SystemExit("数据集没有验证划分，无法给出独立指标")
    return (waveforms[train_mask], labels[train_mask], snrs[train_mask],
            waveforms[val_mask], labels[val_mask], snrs[val_mask])


def _fmt(value, digits=3):
    """指标可能为 ``None``（该类无预测或无支持），此时按“不适用”打印而不是丢掉。"""
    return "  n/a" if value is None else f"{float(value):.{digits}f}"


def _print_metrics(title, metrics):
    print(f"\n== {title} ==")
    print(f"  准确率 {_fmt(metrics['accuracy'], 4)}  宏平均 F1 {_fmt(metrics['macro_f1'], 4)}"
          f"  样本 {metrics['total']}")
    for row in metrics["per_class"]:
        print(f"    {row['label']:6s} 支持 {row['support']:5d}  "
              f"精确率 {_fmt(row['precision'])}  召回 {_fmt(row['recall'])}  "
              f"F1 {_fmt(row['f1'])}")


def _print_confusion(labels, confusion):
    width = max(6, max((len(label) for label in labels), default=6) + 1)
    print("  混淆矩阵（行=真值，列=预测）")
    print("  " + " " * width + "".join(f"{label:>{width}s}" for label in labels))
    for name, row in zip(labels, confusion):
        print(f"  {name:>{width}s}" + "".join(f"{value:>{width}d}" for value in row))


def _per_snr(waveforms, labels, snrs, predict, edges=(-5.0, 0.0, 5.0, 10.0, 20.0)):
    """按带内信噪比分档统计准确率（分档边界与 ``build_amc_dataset`` 的区间同量级）。"""
    rows = []
    for low, high in zip(edges, edges[1:] + (float("inf"),)):
        mask = (snrs >= low) & (snrs < high)
        if not mask.any():
            rows.append({"low_db": low, "high_db": None if high == float("inf") else high,
                         "count": 0, "accuracy": None})
            continue
        predicted = predict(waveforms[mask])
        hit = int((np.asarray(predicted) == labels[mask]).sum())
        rows.append({"low_db": low, "high_db": None if high == float("inf") else high,
                     "count": int(mask.sum()), "accuracy": round(hit / int(mask.sum()), 4)})
    return rows


def _channels(text):
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if not parts:
        return None
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise SystemExit(f"--channels 需要整数，逗号分隔：{text!r}") from exc


def main(argv=None):
    args = _parse_args(argv)
    _require_torch()
    import torch

    from iq_cnn import export_onnx, train_classifier

    card, waveforms, labels, splits, snrs, sources = load_dataset(args.data)
    classes = list((card.get("contract") or {}).get("classes") or [])
    samples = int((card.get("contract") or {})["samples"])
    order = {name: position for position, name in enumerate(classes)}
    train_x, train_labels, train_snr, val_x, val_labels, val_snr = _split(
        waveforms, labels, splits, snrs)
    if args.min_snr is not None:
        keep = train_snr >= args.min_snr
        if not keep.any():
            raise SystemExit(f"按 --min-snr {args.min_snr:g} 过滤后训练集为空")
        print(f"按 --min-snr {args.min_snr:g} 过滤：{len(train_x)} → {int(keep.sum())} 个训练样本")
        train_x, train_labels = train_x[keep], train_labels[keep]
    train_y = np.asarray([order[name] for name in train_labels], dtype=np.int64)
    val_y = np.asarray([order[name] for name in val_labels], dtype=np.int64)
    print(f"数据集 {args.data}：训练 {len(train_x)} / 验证 {len(val_x)}，"
          f"窗口 {samples} 点，类别 {len(classes)} 个")

    channels = _channels(args.channels)
    outcome = train_classifier(
        train_x, train_y, val_x, val_y, classes=classes, arch=args.arch, channels=channels,
        kernel=args.kernel or None, dropout=args.dropout, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, patience=args.patience, seed=args.seed,
        device=args.device, progress=(lambda item: print(
            "TRAIN_EVENT " + json.dumps(item, allow_nan=False), flush=True)) if args.events else None)
    print(f"\n训练完成：最佳验证准确率 {outcome['best_accuracy']:.4f}"
          f"（第 {outcome['best_epoch']} 轮，共跑 {outcome['epochs_run']} 轮）")

    model = outcome["model"]

    def predict(batch):
        with torch.no_grad():
            outputs = model(torch.as_tensor(np.asarray(batch, dtype=np.float32))).argmax(dim=1)
        return [classes[int(index)] for index in outputs]

    in_sample = classification_metrics(
        list(train_labels), predict(train_x), labels=classes)
    validation = classification_metrics(
        list(val_labels), predict(val_x), labels=classes)
    _print_metrics(f"{args.arch.upper()} · 独立验证集", validation)
    _print_confusion(validation["labels"], validation["confusion"])
    print(f"\n  训练集内准确率 {_fmt(in_sample['accuracy'], 4)}（仅自检，不作为指标）")
    validation["per_snr"] = _per_snr(val_x, val_labels, val_snr, predict)

    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else Path(args.data) / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / IQ_DEFAULT_MODEL_NAME
    export_onnx(model, onnx_path, classes=classes, samples=samples, opset=args.opset)

    training_info = {
        "script": "training/train_iq.py",
        "arch": args.arch,
        "dataset": str(Path(args.data)),
        "dataset_seed": card.get("seed"),
        "dataset_samples": card.get("sample_count"),
        "dataset_created": card.get("created"),
        "dataset_contract": card.get("contract"),
        "dataset_sources": card.get("statistics", {}).get("sources"),
        "train_samples": int(len(train_x)),
        "val_samples": int(len(val_x)),
        "epochs": args.epochs,
        "epochs_run": outcome["epochs_run"],
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "channels": list(channels) if channels else None,
        "kernel": args.kernel or None,
        "seed": args.seed,
        "device": args.device,
        "best_epoch": outcome["best_epoch"],
        "best_validation_accuracy": outcome["best_accuracy"],
        "validation": validation,
        "accuracy_in_sample": in_sample["accuracy"],
        "note": args.note or None,
    }
    manifest_path = onnx_dir / MANIFEST_NAME
    manifest, library = write_iq_manifest(
        manifest_path, onnx_path, identifier=args.identifier, version=args.version,
        classes=classes, samples=samples, opset=args.opset,
        default_offset_hz=args.default_offset_hz, default_bandwidth_hz=args.default_bandwidth_hz,
        training=training_info,
        notes="输入为单位 RMS 的 (2, N) 复基带窗口（signal_analysis.ml.iq.iq_waveform 产出）；"
              "softmax 已写入导出图")
    print(f"\nONNX 分类器 {library}\n清单 {manifest_path}（sha256 {manifest['sha256'][:12]}…）")

    if manifest["classes"] != classes:
        raise SystemExit("清单里的类别顺序与训练时不一致，请删除清单后重跑")
    print(f"  清单类别：{manifest['classes']}（标签集合 {manifest['class_set']}）")
    (onnx_dir / "metrics.json").write_text(json.dumps(
        {"validation": validation, "history": outcome["history"],
         "note": "验证集标签评分；独立于资产 generation 真值评分"},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    # 用推理端入口再评一次：确认"训练用的模型"和"清单指向的模型"是同一个
    from signal_analysis.ml.iq import iq_scores

    truth, predicted = [], []
    for waveform, label in zip(val_x, val_labels):
        scores, _, probabilities = iq_scores(manifest_path, waveform, args.threads)
        truth.append(label)
        predicted.append(list(scores)[int(np.argmax(probabilities))])
    _print_metrics("ONNX 分类器（推理端入口）· 独立验证集",
                   classification_metrics(truth, predicted, labels=classes))
    print("\n注意：识别准确率的合格门限尚未确认（技术方案待确认项），"
          "以上指标仅描述本数据集分布下的表现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
