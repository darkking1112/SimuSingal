#!/usr/bin/env python3
"""原始 IQ 分类模型验收：清单自洽、ONNX 输入输出契约、端到端推理、可复现与数据集指标。

一次运行完成以下几件事：

1. **清单自洽**：:func:`read_iq_manifest` 校验版本、契约（``iq_waveform_v1``）、
   类别字典、摘要（sha256）与模型文件位置；
2. **运行时**：``onnxruntime`` 是否可用、版本是否满足清单要求；
3. **接口契约**：ONNX 输入必须是 ``(1, 2, N)``（``N`` = 清单 ``input.samples``）、
   输出必须是 ``(1, C)``（``C`` = 类别数），且节点名与清单一致——这一条只能从图上
   验证，清单里看不出来；
4. **端到端**：用确定性场景（单载波、跳频会话、双信号、纯噪声）跑
   :func:`signal_analysis.ml.iq.amc_iq_classify`，检查结果契约字段齐全、JSON 安全；
5. **可复现**：同一批样本重复推理，除 ``timing`` 外必须逐字节一致；
6. **数据集指标**（``--data``）：在数据集的独立验证划分上算准确率/宏平均 F1 与
   分信噪比表现，用 :func:`classification_metrics` 的口径，**不设合格门限**。

用法::

    .venv/bin/python training/verify_iq.py --manifest training/data/iq/onnx/iq_manifest.json
    .venv/bin/python training/verify_iq.py --manifest training/data/iq/onnx/iq_manifest.json \\
        --data training/data/iq --json training/data/iq/verify.json

退出码：0 全部通过；1 有检查未通过；2 缺少 onnxruntime。
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

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import classification_metrics  # noqa: E402
from signal_analysis.ml import RuntimeUnavailable, runtime_module, runtime_version  # noqa: E402
from signal_analysis.ml.iq import (  # noqa: E402
    IQ_INPUT_CHANNELS,
    IQ_RESULT_CONTRACT,
    amc_iq_classify,
    read_iq_manifest,
)

#: 结果契约字段（少一个都算验收失败）
RESULT_KEYS = ("contract", "algorithm", "classes", "class_set", "labels", "waveform",
               "snr_estimate_db", "prediction", "model", "timing", "pending")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="原始 IQ 分类模型验收（清单 / ONNX 契约 / 端到端 / 可复现）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", required=True, help="IQ 分类器清单路径")
    parser.add_argument("--data", default="", help="数据集目录；给出后额外报验证集指标")
    parser.add_argument("--rate", type=float, default=200_000.0, help="场景采样率（Hz）")
    parser.add_argument("--duration", type=float, default=0.3, help="场景时长（s）")
    parser.add_argument("--seed", type=int, default=20240601, help="场景随机种子")
    parser.add_argument("--threads", type=int, default=0, help="ORT 线程数（0 = 默认）")
    parser.add_argument("--json", default="", help="把验收报告写入该 JSON 路径")
    return parser.parse_args(argv)


def _check(report, label, ok, detail=""):
    report["checks"].append({"label": label, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'通过' if ok else '失败'}] {label}" + (f"：{detail}" if detail else ""), flush=True)
    return bool(ok)


def _scenes(rate, duration):
    """固定场景集：与 ``verify_onnx.py`` 同量级，覆盖单载波 / 跳频 / 双信号 / 纯噪声。"""
    builders = [
        ("单载波数字", [{"mode": "qpsk", "offset": 20_000.0, "bandwidth": 20_000.0,
                         "power_dbfs": 0.0}], {"snr_db": 20.0}, (20_000.0, 20_000.0)),
        ("跳频会话", [{"mode": "fh_rc", "offset": 0.0, "bandwidth": 60_000.0,
                       "hop_count": 6, "hop_bandwidth": 60_000.0 / 7.0, "hop_rate": 20.0,
                       "power_dbfs": 0.0}], {"snr_db": 20.0}, (0.0, 60_000.0)),
        ("双信号（取其一）", [{"mode": "am", "offset": -30_000.0, "bandwidth": 12_000.0,
                               "power_dbfs": 0.0},
                              {"mode": "fm", "offset": 30_000.0, "bandwidth": 16_000.0,
                               "power_dbfs": -3.0}], {"snr_db": 18.0}, (-30_000.0, 12_000.0)),
        ("纯噪声", [], {"power_dbfs": -25.0}, None),
    ]
    scenes = []
    for index, (label, signals, noise, band) in enumerate(builders):
        samples, generation = generate_iq(rate, duration, signals, noise=noise, seed=1000 + index)
        scene = {"label": label, "samples": samples, "generation": generation}
        if band is not None:
            scene["config"] = {"offset_hz": band[0], "bandwidth_hz": band[1]}
        scenes.append(scene)
    return scenes


def _input_contract(session, manifest):
    """检查图上的输入形状是否与清单一致（``(batch, 2, N)``）。"""
    inputs = list(session.get_inputs())
    if len(inputs) != 1:
        return False, f"模型应有 1 个输入，实际 {len(inputs)} 个"
    node = inputs[0]
    shape = list(node.shape)
    samples = int(manifest["input"]["samples"])
    if len(shape) != 3:
        return False, f"输入应为 3 维 (batch,2,N)，实际 {shape}"
    channels, length = shape[1], shape[2]
    if isinstance(channels, int) and channels != IQ_INPUT_CHANNELS:
        return False, f"输入通道维 {channels} 与 {IQ_INPUT_CHANNELS} 不符"
    if isinstance(length, int) and length > 0 and length != samples:
        return False, f"输入窗口长度 {length} 与清单 input.samples {samples} 不一致"
    name = manifest["input"].get("name")
    if name and node.name != name:
        return False, f"输入节点名 {node.name} 与清单 {name} 不一致"
    return True, f"{node.name} {shape} {node.type}"


def _output_contract(session, manifest):
    outputs = list(session.get_outputs())
    if not outputs:
        return False, "模型没有输出节点"
    node = outputs[0]
    shape = list(node.shape)
    classes = len(manifest["classes"])
    if len(shape) != 2:
        return False, f"输出应为 2 维 (batch,C)，实际 {shape}"
    if isinstance(shape[1], int) and shape[1] != classes:
        return False, f"输出类别维 {shape[1]} 与清单类别数 {classes} 不符"
    name = manifest["output"].get("name")
    if name and node.name != name:
        return False, f"输出节点名 {node.name} 与清单 {name} 不一致"
    return True, f"{node.name} {shape} {node.type}"


def _dataset_metrics(directory, manifest_path, classes, threads):
    """在数据集的独立验证划分上评一次（用推理端入口 ``iq_scores``，不碰 torch）。

    数据集里存的就是 ``iq_waveform`` 产出的 ``(2, N)`` 窗口——与推理端拿到的是同一种
    张量，因此这里把落盘波形**直接喂给分类器**，不必再走一遍混频/抽取；分析窗口另存
    在 ``offset_hz``/``bandwidth_hz`` 字段里供回溯。
    """
    from signal_analysis.ml.iq import iq_scores

    directory = Path(directory)
    card_path, store_path = directory / "iq_dataset.json", directory / "iq_dataset.npz"
    if not card_path.is_file() or not store_path.is_file():
        raise SystemExit(f"{directory} 不是 iq_dataset（缺少 iq_dataset.json / .npz）")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if list((card.get("contract") or {}).get("classes") or []) != list(classes):
        raise SystemExit("数据集类别字典与清单类别顺序不一致，指标不可比")
    with np.load(store_path) as store:
        waveforms = np.asarray(store["waveforms"], dtype=np.float32)
        labels = np.asarray(store["labels"]).astype(str)
        splits = np.asarray(store["split"]).astype(str)
        snrs = np.asarray(store["snr_db"], dtype=np.float64)
    if not set(splits).issubset({"train", "val", "test"}):
        raise SystemExit("数据划分只允许 train / val / test")
    rows = np.flatnonzero(splits == "val")
    if not rows.size:
        raise SystemExit("数据集没有验证划分，无法给出独立指标")

    truth, predicted, hit = [], [], []
    for position in rows:
        scores, _, probabilities = iq_scores(manifest_path, waveforms[position], threads or None)
        label = list(scores)[int(np.argmax(probabilities))]
        truth.append(str(labels[position]))
        predicted.append(label)
        hit.append(label == str(labels[position]))
    metrics = classification_metrics(truth, predicted, labels=list(classes))
    snr_values, hit = snrs[rows], np.asarray(hit)
    edges = (-5.0, 0.0, 5.0, 10.0, 20.0)
    metrics["per_snr"] = []
    for low, high in zip(edges, edges[1:] + (float("inf"),)):
        bucket = (snr_values >= low) & (snr_values < high)
        metrics["per_snr"].append({
            "low_db": low, "high_db": None if high == float("inf") else high,
            "count": int(bucket.sum()),
            "accuracy": round(float(hit[bucket].mean()), 4) if bucket.any() else None})
    metrics["card"] = {"sample_count": card.get("sample_count"), "seed": card.get("seed"),
                       "sources": (card.get("statistics") or {}).get("sources")}
    return metrics


def _strip_timing(result):
    return {key: value for key, value in result.items() if key != "timing"}


def main(argv=None):
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    report = {"manifest": str(manifest_path), "checks": [], "scenes": [],
              "runtime": runtime_version()}

    try:
        runtime_module()
    except RuntimeUnavailable as exc:
        print(f"失败：{exc}")
        return 2

    manifest, library = read_iq_manifest(manifest_path)
    classes = list(manifest["classes"])
    print(f"清单：{manifest['id']}@{manifest['version']}  运行时 {report['runtime']}  "
          f"opset {manifest['opset']}  {manifest['sha256'][:16]}…")
    _check(report, "清单校验（版本/契约/摘要/类别字典）", True,
           f"library {library}，{len(classes)} 类，窗口 {manifest['input']['samples']} 点")
    _check(report, "输入波形契约", manifest["input"]["layout"] == "iq_channels_first_v1"
           and int(manifest["input"]["channels"]) == IQ_INPUT_CHANNELS,
           f"{manifest['input']['layout']}，{manifest['input']['channels']} 通道，"
           f"归一化 {manifest['preprocess']['normalization']}")

    session = runtime_module().InferenceSession(str(library), providers=["CPUExecutionProvider"])
    _check(report, "ONNX 输入形状", *_input_contract(session, manifest))
    _check(report, "ONNX 输出形状", *_output_contract(session, manifest))

    failures = sum(1 for check in report["checks"] if not check["ok"])
    for index, scene in enumerate(_scenes(args.rate, args.duration)):
        try:
            result = amc_iq_classify(scene["samples"], args.rate, scene.get("config"),
                                     model=manifest_path, threads=args.threads or None)
        except Exception as exc:  # noqa: BLE001 - 推理错误直接作为验收失败项呈现
            ok = _check(report, f"场景「{scene['label']}」端到端推理", False, f"推理失败：{exc}")
            failures += 0 if ok else 1
            report["scenes"].append({"label": scene["label"], "ok": False, "error": str(exc)})
            continue
        missing = sorted(set(RESULT_KEYS) - set(result))
        extra = sorted(set(result) - set(RESULT_KEYS))
        contract_ok = not missing and not extra and result["contract"] == IQ_RESULT_CONTRACT
        try:
            json.dumps(result, ensure_ascii=False, allow_nan=False)
            safety = True
            error = ""
        except (TypeError, ValueError) as exc:
            safety, error = False, str(exc)
        prediction = result["prediction"]
        ok = _check(report, f"场景「{scene['label']}」端到端推理", contract_ok and safety,
                    f"契约 {result['contract']}（缺 {missing or '无'}／多 {extra or '无'}），"
                    f"预测 {prediction['label']} 置信度 {prediction['confidence']}，"
                    f"可用性 {prediction['reliable']}"
                    + (f"，JSON 不安全：{error}" if error else ""))
        failures += 0 if ok else 1
        report["scenes"].append({"label": scene["label"], "ok": ok,
                                 "contract": result["contract"],
                                 "prediction": prediction,
                                 "snr_estimate_db": result["snr_estimate_db"],
                                 "waveform": result["waveform"]})
        if index == 0:
            repeat = amc_iq_classify(scene["samples"], args.rate, scene.get("config"),
                                     model=manifest_path, threads=args.threads or None)
            same = _strip_timing(repeat) == _strip_timing(result)
            _check(report, "重复推理结果一致（可复现，忽略耗时）", same)
            failures += 0 if same else 1

    if args.data:
        metrics = _dataset_metrics(args.data, manifest_path, classes, args.threads)
        _check(report, "数据集独立验证划分", metrics["accuracy"] is not None,
               f"准确率 {metrics['accuracy']} 宏平均 F1 {metrics['macro_f1']} "
               f"样本 {metrics['total']}")
        report["dataset"] = {"path": str(Path(args.data)), "metrics": metrics}

    print(f"\n验收结论：{'全部通过' if failures == 0 else f'{failures} 项未通过'}"
          f"（{len(report['checks'])} 项检查）")
    print("注意：识别准确率的合格门限尚未确认（技术方案待确认项），"
          "以上数字只描述本次数据集分布。")
    report["failures"] = failures
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")
        print(f"报告已写入 {target}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
