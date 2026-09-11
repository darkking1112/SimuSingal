#!/usr/bin/env python3
"""AI 检测模型验收（C02）：清单/摘要、ONNX 输入输出契约、端到端推理与数值一致性。

一次运行完成以下几件事：

1. **清单自洽**：``read_model_manifest`` 校验版本、契约、标签、摘要（sha256）；
2. **运行时**：``onnxruntime`` 是否可用、版本是否满足清单要求；
3. **接口契约**：ONNX 输入必须是 ``(1, 1, H, W)`` 且 H/W 等于清单
   ``input.image_size``，输出必须能被 :func:`parse_model_output` 解码
   （列数与 :data:`BOX_COLUMNS` 一致）——这一条只能从图上验证，清单里看不出来；
4. **端到端**：用确定性场景（含跳频会话、双信号、纯噪声）跑
   :func:`signal_analysis.ml.ml_detect`，检查结果是 JSON 安全的，并用项目自身
   评测口径（:func:`evaluate_detections`）给出召回/精确率；
5. **可复现**：同一批样本重复推理，检测结果必须逐字节一致；
6. **数值一致性**（``--reference``）：把两个模型放在同一张时频图上比对原始
   输出，报最大绝对偏差（例如 FP32 与 FP16 导出、不同 opset 的回归检查）。

用法::

    .venv/bin/python training/verify_onnx.py --manifest training/runs/tiny/detector.json
    .venv/bin/python training/verify_onnx.py --manifest a/detector.json \\
        --reference a/detector_fp16.onnx --json training/runs/tiny/verify.json

退出码：0 全部通过；1 有检查未通过；2 缺少 onnxruntime。
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

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import evaluate_detections, signal_truth  # noqa: E402
from signal_analysis.ml import (  # noqa: E402
    BOX_COLUMNS,
    RuntimeUnavailable,
    detection_image,
    ml_detect,
    parse_model_output,
    read_model_manifest,
    runtime_module,
    runtime_version,
    spectral_context,
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="AI 检测模型验收（清单 / ONNX 契约 / 端到端 / 一致性）")
    parser.add_argument("--manifest", required=True, help="模型清单路径（detector.json）")
    parser.add_argument("--reference", default="",
                        help="对照模型的 .onnx 路径，用于数值一致性比对（同一图像张量）")
    parser.add_argument("--seed", type=int, default=20240601, help="场景随机种子")
    parser.add_argument("--rate", type=float, default=1_000_000.0, help="场景采样率（Hz）")
    parser.add_argument("--duration", type=float, default=0.4, help="场景时长（s）")
    parser.add_argument("--threads", type=int, default=0, help="ORT 线程数（0 = 默认）")
    parser.add_argument("--json", default="", help="把验收报告写入该 JSON 路径")
    return parser.parse_args(argv)


def _check(report, label, ok, detail=""):
    report["checks"].append({"label": label, "ok": bool(ok), "detail": str(detail)})
    print(f"[{'通过' if ok else '失败'}] {label}" + (f"：{detail}" if detail else ""), flush=True)
    return bool(ok)


def _scenes(rate, duration):
    """固定场景集：覆盖单信号、跳频会话、双信号与纯噪声。"""
    builders = [
        ("单载波数字", [{"mode": "qpsk", "offset": 150_000.0, "bandwidth": 120_000.0,
                         "power_dbfs": 0.0}], {"snr_db": 20.0}),
        ("跳频会话", [{"mode": "fh_rc", "offset": 0.0, "bandwidth": 300_000.0,
                       "hop_count": 6, "hop_bandwidth": 300_000.0 / 7.0, "hop_rate": 20.0,
                       "power_dbfs": 0.0}], {"snr_db": 20.0}),
        ("双信号", [{"mode": "am", "offset": -200_000.0, "bandwidth": 80_000.0, "power_dbfs": 0.0},
                    {"mode": "fm", "offset": 250_000.0, "bandwidth": 100_000.0, "power_dbfs": -3.0}],
         {"snr_db": 18.0}),
        ("纯噪声", [], {"power_dbfs": -25.0}),
    ]
    scenes = []
    for index, (label, signals, noise) in enumerate(builders):
        samples, generation = generate_iq(rate, duration, signals, noise=noise, seed=1000 + index)
        scenes.append({"label": label, "samples": samples, "generation": generation})
    return scenes


def _input_contract(session, manifest):
    """检查图上/图中的输入输出形状是否与清单一致。"""
    inputs = list(session.get_inputs())
    if len(inputs) != 1:
        return False, f"模型应有 1 个输入，实际 {len(inputs)} 个"
    node = inputs[0]
    shape = list(node.shape)
    size = int(manifest["input"]["image_size"])
    expected = shape[-2:] if len(shape) >= 2 else []
    static = [int(value) for value in expected
              if isinstance(value, int) and value > 0]
    if len(shape) != 4:
        return False, f"输入应为 4 维 (1,1,H,W)，实际 {shape}"
    if static and static != [size, size]:
        return False, f"输入 H/W {static} 与清单 image_size {size} 不一致"
    name = manifest["input"].get("name")
    if name and node.name != name:
        return False, f"输入节点名 {node.name} 与清单 {name} 不一致（推理会回退到第一个输入）"
    return True, f"{node.name} {shape} {node.type}"


def _output_contract(session):
    outputs = list(session.get_outputs())
    if not outputs:
        return False, "模型没有输出节点"
    node = outputs[0]
    shape = list(node.shape)
    if len(shape) != 3:
        return False, f"输出应为 3 维 (N,K,{BOX_COLUMNS})，实际 {shape}"
    tail = [value for value in shape[-2:] if isinstance(value, int) and value > 0]
    if tail and BOX_COLUMNS not in tail:
        return False, f"输出 {shape} 未包含 {BOX_COLUMNS} 列检测框"
    return True, f"{node.name} {shape}"


def main(argv=None):
    args = _parse_args(argv)
    manifest_path = Path(args.manifest)
    report = {"manifest": str(manifest_path), "checks": [], "scenes": [], "runtime": runtime_version()}

    try:
        runtime = runtime_module()
    except RuntimeUnavailable as exc:
        print(f"失败：{exc}")
        return 2

    manifest, library = read_model_manifest(manifest_path)
    print(f"清单：{manifest['id']}@{manifest['version']}  运行时 {report['runtime']}  "
          f"opset {manifest['opset']}  {manifest['sha256'][:16]}…")
    _check(report, "清单校验（版本/契约/摘要）", True,
           f"library {manifest['library']}，标签 {manifest['labels']}")
    _check(report, "输入时频图契约",
           manifest["input"]["layout"] == "time_frequency_grayscale_v1",
           f"{manifest['input']['layout']}，nfft {manifest['input']['spectrogram_nfft']}，"
           f"动态范围 {manifest['input']['dynamic_range_db']:g} dB")

    session = runtime.InferenceSession(str(library), providers=["CPUExecutionProvider"],
                                       sess_options=_options(runtime, args.threads))
    _check(report, "ONNX 输入形状", *_input_contract(session, manifest))
    _check(report, "ONNX 输出形状", *_output_contract(session))

    reference = _load_reference(runtime, args.reference) if args.reference else None

    failures = sum(1 for check in report["checks"] if not check["ok"])
    for index, scene in enumerate(_scenes(args.rate, args.duration)):
        summary, _ = ml_detect(scene["samples"], args.rate, model=manifest_path,
                               threads=args.threads or None, with_baseline=True)
        truth = signal_truth(scene["generation"])
        metrics = evaluate_detections(truth, summary["detections"])
        try:
            json.dumps(summary["detections"], allow_nan=False, ensure_ascii=False)
            safety = True
        except (TypeError, ValueError) as exc:
            safety = False
            metrics = {**metrics, "json_error": str(exc)}
        ok = _check(report, f"场景「{scene['label']}」端到端推理", safety,
                    f"真值 {metrics['true']} 检出 {metrics['detected']} 召回 {metrics['recall']} "
                    f"精确率 {metrics['precision']}")
        report["scenes"].append({"label": scene["label"], "metrics": metrics, "ok": ok})
        failures += 0 if ok else 1
        if index == 0:
            repeat = ml_detect(scene["samples"], args.rate, model=manifest_path,
                               threads=args.threads or None, with_baseline=True)[0]
            same = repeat["detections"] == summary["detections"]
            _check(report, "重复推理结果一致（可复现）", same)
            failures += 0 if same else 1
            if reference is not None:
                failures += _compare(report, manifest, scene, reference, session, args)

    print(f"\n验收结论：{'全部通过' if failures == 0 else f'{failures} 项未通过'}"
          f"（{len(report['checks'])} 项检查）")
    report["failures"] = failures
    if args.json:
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")
        print(f"报告已写入 {target}")
    return 0 if failures == 0 else 1


def _options(runtime, threads):
    options = runtime.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
    return options


def _load_reference(runtime, path):
    reference = Path(path).resolve(strict=True)
    return {"path": reference,
            "session": runtime.InferenceSession(str(reference), providers=["CPUExecutionProvider"])}


def _compare(report, manifest, scene, reference, session, args):
    """同一张时频图上比对两个模型的原始输出（数值一致性检查）。"""
    contract = manifest["input"]
    energy, arrays = spectral_context(scene["samples"], args.rate,
                                      {"nfft": int(contract["spectrogram_nfft"])})
    image, _ = detection_image(arrays, energy, int(contract["image_size"]),
                               float(contract["dynamic_range_db"]))
    size = int(image.shape[-1])
    tensor = np.ascontiguousarray(image, dtype=np.float32).reshape(1, 1, size, size)
    try:
        primary = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        other = reference["session"].run(None, {reference["session"].get_inputs()[0].name: tensor})[0]
    except Exception as exc:  # noqa: BLE001 - 会话错误直接作为验收失败项呈现
        ok = _check(report, "数值一致性（对照模型）", False, f"推理失败：{exc}")
        return 0 if ok else 1
    if primary.shape != other.shape:
        ok = _check(report, "数值一致性（对照模型）", False,
                    f"输出形状不同：{primary.shape} vs {other.shape}")
        return 0 if ok else 1
    difference = float(np.max(np.abs(primary.astype(np.float64) - other.astype(np.float64))))
    rows = parse_model_output(primary)[0]
    report["reference"] = {"path": str(reference["path"]), "max_abs_diff": difference,
                           "shape": list(primary.shape), "rows": int(len(rows))}
    ok = _check(report, "数值一致性（对照模型）", np.allclose(primary, other, atol=1e-4),
                f"最大绝对偏差 {difference:.3e}，输出 {list(primary.shape)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
