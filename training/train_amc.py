#!/usr/bin/env python3
"""训练 A09 六类调制识别模型（线性判别基线 / 可选 Transformer + ONNX）。

用法::

    # 1) 生成数据集（只依赖 NumPy）
    .venv/bin/python training/build_amc_dataset.py --output training/data/amc --per-class 400

    # 2) 训练线性基线，并写回内置模型（随包分发）
    .venv/bin/python training/train_amc.py --data training/data/amc

    # 3) 可选：训练 Transformer 并导出 ONNX 分类器（需要 pip install .[train]）
    .venv/bin/python training/train_amc.py --data training/data/amc --arch transformer \\
        --onnx-dir training/data/amc/onnx

约定
----
* 训练与推理共用 :func:`signal_analysis.ml.amc.extract_features`，数据集里存的
  就是推理端会拿到的**同一种**特征向量；数据集的特征清单必须与 ``AMC_FEATURES``
  完全一致，否则脚本直接报错而不是悄悄训练。
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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_analysis.ml import amc  # noqa: E402

DEFAULT_OUTPUT = REPO_ROOT / "src" / "signal_analysis" / "ml" / amc.DEFAULT_MODEL_NAME
DEFAULT_ONNX_NAME = "amc_onnx.json"
TRAIN_EXTRA = "pip install .[train]"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="训练 A09 六类调制识别模型")
    parser.add_argument("--data", required=True, help="数据集目录（build_amc_dataset.py 的输出）")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"线性模型输出路径，默认写回内置模型 {DEFAULT_OUTPUT}")
    parser.add_argument("--arch", choices=("linear", "transformer"), default="linear",
                        help="模型结构：linear 为确定性线性判别基线（默认）")
    parser.add_argument("--l2", type=float, default=amc.DEFAULT_L2,
                        help=f"线性模型 L2 正则强度，默认 {amc.DEFAULT_L2:g}")
    parser.add_argument("--min-snr", type=float, default=None,
                        help="只用带内信噪比不低于该值的样本训练（默认不过滤）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（Transformer 用）")
    parser.add_argument("--epochs", type=int, default=60, help="Transformer 训练轮数，默认 60")
    parser.add_argument("--batch-size", type=int, default=128, help="Transformer 批大小，默认 128")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer 隐层宽度，默认 64")
    parser.add_argument("--heads", type=int, default=4, help="Transformer 注意力头数，默认 4")
    parser.add_argument("--layers", type=int, default=2, help="Transformer 编码层数，默认 2")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
                        help="Transformer 学习率，默认 3e-3")
    parser.add_argument("--onnx-dir", default=None,
                        help="Transformer 的 ONNX 输出目录（含清单），默认与 --output 同目录")
    parser.add_argument("--identifier", default=amc.DEFAULT_MODEL_ID,
                        help=f"模型标识，默认 {amc.DEFAULT_MODEL_ID}")
    parser.add_argument("--version", default="0.1.0", help="模型版本号，默认 0.1.0")
    parser.add_argument("--note", default="", help="附加到训练信息的备注")
    parser.add_argument("--threads", type=int, default=None, help="ONNX 校验线程数")
    return parser.parse_args(argv)


def load_dataset(directory):
    """读取 ``features.jsonl`` 并校验契约（特征顺序必须与 ``AMC_FEATURES`` 一致）。"""
    directory = Path(directory)
    card_path = directory / "amc_dataset.json"
    records_path = directory / "features.jsonl"
    if not card_path.is_file() or not records_path.is_file():
        raise SystemExit(f"数据集不完整：请先运行 training/build_amc_dataset.py 生成 {directory}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    features = list((card.get("contract") or {}).get("features") or [])
    if features != list(amc.AMC_FEATURES):
        raise SystemExit("数据集特征清单与当前 AMC_FEATURES 不一致，请重新生成数据集")
    if list((card.get("contract") or {}).get("classes") or []) != list(amc.AMC_CLASSES):
        raise SystemExit("数据集类别字典与当前 A09 六类不一致，请重新生成数据集")
    records = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise SystemExit("数据集为空")
    return card, records


def _split(records, card):
    train = [record for record in records if record.get("split") == "train"]
    val = [record for record in records if record.get("split") != "train"]
    if not train:
        raise SystemExit("数据集没有训练划分")
    if not val:
        raise SystemExit("数据集没有验证划分，无法给出独立指标")
    return train, val


def _fmt(value, digits=3):
    """指标可能为 ``None``（该类无预测或无支持），此时按“不适用”打印而不是丢掉。"""
    if value is None:
        return "  n/a"
    return f"{float(value):.{digits}f}"


def _print_metrics(title, metrics):
    print(f"\n== {title} ==")
    print(f"  准确率 {_fmt(metrics['accuracy'], 4)}  宏平均 F1 {_fmt(metrics['macro_f1'], 4)}"
          f"  样本 {metrics['total']}")
    for row in metrics["per_class"]:
        print(f"    {row['label']:6s} 支持 {row['support']:5d}  "
              f"精确率 {_fmt(row['precision'])}  召回率 {_fmt(row['recall'])}"
              f"  F1 {_fmt(row['f1'])}")
    for bucket, value in (metrics.get("per_snr") or {}).items():
        print(f"    {bucket:>12s}  {_fmt(value)}")


def _print_confusion(labels, confusion):
    print("\n混淆矩阵（行=真值，列=预测）：")
    header = "        " + "".join(f"{label:>8s}" for label in labels)
    print(header)
    for label, row in zip(labels, confusion):
        print(f"  {label:6s}" + "".join(f"{value:>8d}" for value in row))


def _train_linear(args, train, val, provenance):
    model = amc.fit_model(train, l2=args.l2, provenance=provenance)
    model.update({"id": args.identifier, "version": args.version})
    metrics = amc.evaluate_model(model, val)
    _print_metrics(f"线性基线（温度 {model['temperature']:.1f}）· 独立验证集", metrics)
    _print_confusion(metrics["labels"], metrics["confusion"])
    print(f"\n训练集内准确率 {model['training']['accuracy_in_sample']:.4f}"
          f"（仅自检，不作为指标）")
    return model, metrics


def main(argv=None):
    args = _parse_args(argv)
    if args.arch == "transformer":
        return _main_transformer(args)
    card, records = load_dataset(args.data)
    train, val = _split(records, card)
    if args.min_snr is not None:
        kept = [record for record in train if record["snr_db"] >= args.min_snr]
        if not kept:
            raise SystemExit(f"按 --min-snr {args.min_snr:g} 过滤后训练集为空")
        print(f"按 --min-snr {args.min_snr:g} 过滤：{len(train)} → {len(kept)} 个训练样本")
        train = kept
    provenance = {
        "script": "training/train_amc.py",
        "arch": "linear",
        "dataset": str(Path(args.data)),
        "dataset_seed": card.get("seed"),
        "dataset_samples": card.get("sample_count"),
        "dataset_created": card.get("created"),
        "scene": card.get("scene"),
        "analysis_window": card.get("analysis_window"),
        "train_samples": len(train),
        "val_samples": len(val),
        "l2": float(args.l2),
        "note": args.note or None,
    }
    model, metrics = _train_linear(args, train, val, provenance)
    model["training"]["validation"] = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_snr": metrics["per_snr"],
        "per_class": metrics["per_class"],
    }
    model["id"] = args.identifier
    model["version"] = args.version
    path = amc.save_model(model, args.output)
    print(f"\n模型已写入 {path}")
    print("注意：识别准确率的合格门限尚未确认（技术方案待确认项），"
          "以上指标仅描述本数据集分布下的表现。")
    return 0


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise SystemExit(
            f"--arch transformer 需要 PyTorch 与 onnx：{TRAIN_EXTRA}\n（原始错误：{exc}）"
        ) from exc


def _main_transformer(args):
    _require_torch()
    import torch

    from amc_transformer import export_onnx, train_classifier

    card, records = load_dataset(args.data)
    train, val = _split(records, card)
    order = {label: index for index, label in enumerate(amc.AMC_CLASSES)}
    features = np.asarray([[record["features"][name] for name in amc.AMC_FEATURES]
                           for record in records], dtype=np.float32)
    labels = np.asarray([order[record["label"]] for record in records], dtype=np.int64)
    is_train = np.asarray([record.get("split") == "train" for record in records])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    outcome = train_classifier(
        features[is_train], labels[is_train], features[~is_train], labels[~is_train],
        classes=list(amc.AMC_CLASSES), d_model=args.d_model, heads=args.heads,
        layers=args.layers, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed)
    print(f"Transformer 训练完成：最佳验证准确率 {outcome['best_accuracy']:.4f}"
          f"（第 {outcome['best_epoch']} 轮）")
    onnx_dir = Path(args.onnx_dir) if args.onnx_dir else Path(args.output).parent
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "amc_transformer.onnx"
    export_onnx(outcome["model"], onnx_path, standardize=outcome["standardize"],
                feature_count=len(amc.AMC_FEATURES), classes=list(amc.AMC_CLASSES))
    manifest_path = onnx_dir / DEFAULT_ONNX_NAME
    manifest, library = amc.write_amc_manifest(
        manifest_path, onnx_path, identifier=args.identifier, version=args.version,
        standardize=outcome["standardize"], opset=17,
        training={"script": "training/train_amc.py", "arch": "transformer",
                  "dataset": str(Path(args.data)), "dataset_seed": card.get("seed"),
                  "train_samples": int(is_train.sum()),
                  "val_samples": int((~is_train).sum()),
                  "epochs": args.epochs, "batch_size": args.batch_size,
                  "best_validation_accuracy": outcome["best_accuracy"],
                  "note": args.note or None},
        notes="特征向量由 signal_analysis.ml.amc.extract_features 生成，标准化已写入导出图")
    print(f"ONNX 分类器 {library}\n清单 {manifest_path}（sha256 {manifest['sha256'][:12]}…）")

    from signal_analysis.ml.amc import onnx_scores
    from signal_analysis.evaluation import classification_metrics

    truth, predicted = [], []
    for record in val:
        scores, _, probabilities = onnx_scores(manifest_path, record["features"], args.threads)
        truth.append(record["label"])
        predicted.append(list(scores)[int(np.argmax(probabilities))])
    metrics = classification_metrics(truth, predicted, labels=list(amc.AMC_CLASSES))
    _print_metrics("ONNX 分类器 · 独立验证集", metrics)
    _print_confusion(metrics["labels"], metrics["confusion"])
    print("\n注意：识别准确率的合格门限尚未确认（技术方案待确认项）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
