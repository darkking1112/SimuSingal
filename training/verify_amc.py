#!/usr/bin/env python3
"""A09 六类调制识别模型验收：数据集契约、线性基线、ONNX 分类器与一致性。

一次运行完成以下几件事：

1. **数据集契约**：``amc_dataset.json`` 里的特征契约（``amc_feature_vector_v1``）、
   特征顺序（``AMC_FEATURES``）与类别字典（``AMC_CLASSES``）必须与推理端模块一致，
   每条记录的特征必须齐全且为有限值——顺序或维度对不上，训练得再好也会在推理端报错；
2. **线性基线**：加载内置模型或 ``--model`` 指定的 ``amc_model_v1`` JSON，在**独立验证集**
   上给出总体准确率、宏平均 F1、每类指标与分信噪比准确率；
3. **ONNX 分类器**（``--manifest``）：``read_amc_manifest`` 的强校验（契约、特征序、
   类别、sha256 摘要、清单目录内的相对路径），再用 ``onnx_scores`` 跑验证集子集，
   与线性模型比较**逐样本一致率**（两者应当高度一致，差异大说明导出或标准化有问题）；
4. **可复现**：同一批特征重复推理必须得到完全相同的概率（导出图与线性模型都查）；
5. **真值口径**：结果按类计数，**不丢弃**任何样本；``不适用``（六类之外）另行计数。

本脚本**不做**"准确率是否达标"的判定：技术方案里识别准确率的合格门限仍是待确认项，
因此只报告本数据集分布下的实测指标，退出码只反映检查项是否通过。

用法::

    .venv/bin/python training/verify_amc.py --data training/data/amc
    .venv/bin/python training/verify_amc.py --data training/data/amc \\
        --manifest training/runs/amc/amc.json --limit 200 --json training/runs/amc/verify.json

退出码：0 全部检查通过；1 有检查未通过；2 指定了 ``--manifest`` 但缺少 onnxruntime。
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
from signal_analysis.evaluation import classification_metrics  # noqa: E402


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="A09 调制识别模型验收（数据集契约 / 线性基线 / ONNX）")
    parser.add_argument("--data", required=True, help="build_amc_dataset.py 生成的数据集目录")
    parser.add_argument("--model", default="", help="线性模型 JSON；默认使用随包分发的内置基线")
    parser.add_argument("--manifest", default="", help="ONNX 分类器清单（amc-manifest 生成）")
    parser.add_argument("--limit", type=int, default=200, help="ONNX 验证样本数上限，默认 200")
    parser.add_argument("--threads", type=int, default=0, help="ORT 线程数（0 = 默认）")
    parser.add_argument("--json", default="", help="把验收报告写入该 JSON 路径")
    return parser.parse_args(argv)


def _check(report, label, ok, detail=""):
    report["checks"].append({"label": label, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'通过' if ok else '失败'}] {label}" + (f"：{detail}" if detail else ""), flush=True)
    return bool(ok)


def _fmt(value, digits=3):
    """指标可能为 ``None``（该类无预测或无支持），按“不适用”打印而不是丢掉。"""
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _print_metrics(title, metrics):
    print(f"\n== {title} ==")
    print(f"  准确率 {_fmt(metrics['accuracy'], 4)}  宏平均 F1 {_fmt(metrics['macro_f1'], 4)}"
          f"  样本 {metrics['total']}")
    for row in metrics["per_class"]:
        print(f"    {row['label']:6s} 支持 {row['support']:5d}  精确率 {_fmt(row['precision'])}"
              f"  召回率 {_fmt(row['recall'])}  F1 {_fmt(row['f1'])}")
    for bucket, value in (metrics.get("per_snr") or {}).items():
        print(f"    {bucket:>12s}  {_fmt(value)}")


def load_dataset(path):
    """读取数据集卡片与记录（不做契约修正，只报告）。"""
    directory = Path(path)
    card_path = directory / "amc_dataset.json"
    records_path = directory / "features.jsonl"
    if not card_path.is_file() or not records_path.is_file():
        raise SystemExit(f"数据集不完整：缺少 {card_path.name} 或 {records_path.name}")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return card, records


def _check_dataset(report, card, records):
    contract = card.get("contract") or {}
    ok = _check(report, "数据集特征契约",
                contract.get("feature_contract") == amc.AMC_FEATURE_CONTRACT,
                f"{contract.get('feature_contract')}（推理端 {amc.AMC_FEATURE_CONTRACT}）")
    features = list(contract.get("features") or [])
    ok &= _check(report, "数据集特征顺序", features == list(amc.AMC_FEATURES),
                 f"{len(features)} 维" if features == list(amc.AMC_FEATURES)
                 else "与 AMC_FEATURES 不一致，训练/推理特征序已分叉")
    classes = list(contract.get("classes") or [])
    ok &= _check(report, "数据集类别字典", classes == list(amc.AMC_CLASSES),
                 "、".join(classes) if classes == list(amc.AMC_CLASSES)
                 else "与 A09 六类字典不一致")
    broken = [record.get("index") for record in records
              if sorted(record.get("features") or {}) != sorted(amc.AMC_FEATURES)]
    ok &= _check(report, "逐条特征完整性", not broken,
                 f"{len(records)} 条记录" if not broken else f"字段缺失的记录：{broken[:5]}")
    try:
        for record in records:
            amc.feature_vector(record["features"])
    except ValueError as exc:
        ok &= _check(report, "特征向量有限性", False, str(exc))
    else:
        ok &= _check(report, "特征向量有限性", True, f"{len(records)} 条记录均为 34 维有限值")
    train = sum(1 for record in records if record.get("split") == "train")
    val = len(records) - train
    ok &= _check(report, "训练/验证划分", val > 0,
                 f"train {train} / val {val}（分层划分：{card.get('splits', {}).get('strategy')}）")
    labels = sorted({str(record.get("label")) for record in records})
    ok &= _check(report, "类别覆盖", labels == sorted(amc.AMC_CLASSES),
                 "、".join(labels) if labels == sorted(amc.AMC_CLASSES)
                 else f"仅覆盖 {'、'.join(labels)}")
    report["dataset"] = {"samples": len(records), "train": train, "val": val,
                         "seed": card.get("seed"), "created": card.get("created")}
    return ok


def _check_linear(report, records, args):
    if args.model:
        model = amc.load_model(args.model)
        model["source"] = "file"
    else:
        try:
            model = amc.load_default_model()
        except amc.ModelError as exc:
            _check(report, "线性模型可用", False, str(exc))
            return False, None, []
        _check(report, "线性模型可用", True, f"内置 {amc.default_model_path()}")
    val = [record for record in records if record.get("split") != "train"]
    metrics = amc.evaluate_model(model, val)
    _print_metrics(f"线性模型 {model.get('id')}@{model.get('version')}（来源 {model.get('source')}）"
                   " · 独立验证集", metrics)
    report["linear"] = {
        "id": model.get("id"), "version": model.get("version"),
        "source": model.get("source"), "val_samples": len(val),
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"], "per_snr": metrics["per_snr"],
    }
    predicted = [amc.predict(model, record["features"])["label"] for record in val]
    repeat = [amc.predict(model, record["features"])["label"] for record in val]
    ok = _check(report, "线性模型可复现", predicted == repeat,
                "同一批特征重复推理结果一致")
    ok &= _check(report, "验证集覆盖六类", {row["label"] for row in metrics["per_class"]}
                 == set(amc.AMC_CLASSES),
                 f"样本 {metrics['total']}（不适用样本另计，不丢弃）")
    return ok, model, val


def _check_onnx(report, manifest_path, val, model, args):
    from signal_analysis.ml import RuntimeUnavailable, read_amc_manifest, runtime_module

    try:
        runtime = runtime_module()
    except RuntimeUnavailable as exc:
        print(f"[跳过] ONNX 分类器：{exc}", flush=True)
        report["onnx"] = {"skipped": True, "reason": str(exc)}
        return None  # 交给调用方决定退出码
    try:
        manifest, library = read_amc_manifest(manifest_path)
    except amc.ModelError as exc:
        _check(report, "清单与摘要自洽", False, str(exc))
        report["onnx"] = {"skipped": True, "reason": str(exc)}
        return False
    _check(report, "清单与摘要自洽", True,
           f"{manifest['id']}@{manifest['version']} · {library.name} · "
           f"sha256 {manifest['sha256'][:12]}… · opset {manifest['opset']}")

    session_input = None
    try:
        session = runtime.InferenceSession(str(library), providers=["CPUExecutionProvider"])
        session_input = session.get_inputs()[0]
        outputs = session.get_outputs()
        shape = list(session_input.shape)
        ok = _check(report, "ONNX 输入形状", len(shape) == 2 and int(shape[-1]) == len(amc.AMC_FEATURES),
                    f"{session_input.name} {shape}（应为 (N, {len(amc.AMC_FEATURES)})）")
        ok &= _check(report, "ONNX 输入节点名", session_input.name == manifest["input"]["name"],
                     f"{session_input.name}（清单 {manifest['input']['name']}）")
        ok &= _check(report, "ONNX 输出节点", bool(outputs) and outputs[0].name
                     == manifest["output"]["name"],
                     f"{outputs[0].name if outputs else '无'} {list(outputs[0].shape) if outputs else ''}")
    except Exception as exc:  # onnxruntime 的异常类型依版本而异
        _check(report, "ONNX 分类器可加载", False, f"{type(exc).__name__}: {exc}")
        report["onnx"] = {"skipped": True, "reason": str(exc)}
        return False

    subset = val[:max(1, min(args.limit, len(val)))]
    truth, predicted, linear_predicted = [], [], []
    probabilities = None
    for record in subset:
        scores, _, values = amc.onnx_scores(manifest_path, record["features"],
                                            args.threads or None)
        truth.append(record["label"])
        predicted.append(max(scores, key=scores.get))
        linear_predicted.append(amc.predict(model, record["features"])["label"])
        if probabilities is None:
            probabilities = (record["features"], values)
    metrics = classification_metrics(truth, predicted, labels=list(amc.AMC_CLASSES))
    _print_metrics(f"ONNX 分类器 {manifest['id']}@{manifest['version']} · 验证集前 "
                   f"{len(subset)} 条", metrics)
    agree = sum(1 for left, right in zip(predicted, linear_predicted) if left == right)
    ratio = agree / max(len(subset), 1)
    ok &= _check(report, "与线性模型一致率", ratio >= 0.9,
                 f"{agree}/{len(subset)} = {ratio:.3f}（低于 0.9 说明导出图或标准化与训练口径不一致）")
    features, first = probabilities
    _, _, again = amc.onnx_scores(manifest_path, features, args.threads or None)
    ok &= _check(report, "ONNX 推理可复现", np.array_equal(np.asarray(first), np.asarray(again)),
                 "同一特征两次推理概率完全一致")
    ok &= _check(report, "ONNX 输出为概率", bool(
        np.all(np.asarray(first) >= -1e-6) and abs(float(np.sum(first)) - 1.0) < 1e-3),
        f"和 {float(np.sum(first)):.6f}")
    report["onnx"] = {"id": manifest["id"], "version": manifest["version"],
                      "sha256": manifest["sha256"], "opset": manifest["opset"],
                      "samples": len(subset), "accuracy": metrics["accuracy"],
                      "macro_f1": metrics["macro_f1"],
                      "per_class": metrics["per_class"],
                      "agreement_with_linear": {
                          "agree": agree, "total": len(subset), "ratio": ratio}}
    return ok


def main(argv=None):
    args = _parse_args(argv)
    print(f"运行时：onnxruntime {_runtime_version_text()}")
    report = {"checks": [], "data": str(args.data)}
    card, records = load_dataset(args.data)
    ok = _check_dataset(report, card, records)
    linear_ok, model, val = _check_linear(report, records, args)
    ok &= linear_ok
    exit_code = 0 if ok else 1
    if args.manifest:
        if model is None:
            print("[跳过] ONNX 分类器：线性模型不可用（一致率无从比较）", flush=True)
        else:
            outcome = _check_onnx(report, args.manifest, val, model, args)
            if outcome is None:
                exit_code = 2
            else:
                ok &= outcome
                exit_code = 0 if ok else 1
    print("\n注意：识别准确率的合格门限尚未确认（技术方案待确认项）；"
          "以上指标只描述本数据集分布下的表现，六类之外的样式按“不适用”计数。")
    report["ok"] = bool(ok)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"报告已写入 {path}")
    return exit_code


def _runtime_version_text():
    from signal_analysis.ml import runtime_version

    version = runtime_version()
    return version if version else "未安装（ONNX 检查会被跳过，提示 pip install .[ml]）"


if __name__ == "__main__":
    raise SystemExit(main())
