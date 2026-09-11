#!/usr/bin/env python3
"""信号检测模型训练 + ONNX 导出（P3）。

流程：读取 :mod:`build_dataset` 生成的数据集 → 训练 → **用项目自身的评测口径**
在验证集上打分 → ``torch.onnx.export`` → :func:`signal_analysis.ml.write_model_manifest`
生成模型清单。

``--arch tiny`` 使用 ``training/tiny_detector.py`` 中自研的最小无锚框检测头，
目的是在没有任何第三方检测库的情况下就能产出**契约合规**的 ONNX，把
"构建数据集 → 训练 → 导出 → 清单 → 推理/评测"整条链路先跑通。

``--arch`` 的其他取值（``rtdetr`` / ``yolox`` / ``ultralytics``）由
:mod:`training.detectors` 注册表提供。它们**不在本仓库里训练**：第三方框架用
自己的 trainer 训练、用自己的 exporter 导出，再由
``training/export_contract.py`` 把导出图改写成契约（完整步骤见
``training/README.md`` §7，许可证见 §8）。

用法::

    .venv/bin/python -m pip install -e ".[train]"
    .venv/bin/python training/build_dataset.py --output training/data/detector --count 500
    .venv/bin/python training/train_yolox.py --data training/data/detector \\
        --output training/runs/tiny --epochs 10
    .venv/bin/python training/verify_onnx.py --manifest training/runs/tiny/detector.json

注意：导出的 ``.onnx`` 与模型清单必须位于同一目录（清单里保存的是相对路径）。
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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import evaluate_detections, signal_truth  # noqa: E402
from signal_analysis.ml import ml_detect, write_model_manifest  # noqa: E402


def _arch_choices():
    """注册表里的全部 ``--arch`` 取值（``tiny`` 由 ``detectors/tiny.py`` 注册）。"""
    try:
        import detectors
    except ImportError:  # pragma: no cover - 仅在 detectors 目录缺失时发生
        return ("tiny",)
    return tuple(detectors.names())


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="检测模型训练与 ONNX 导出")
    parser.add_argument("--data", required=True, help="build_dataset.py 的输出目录")
    parser.add_argument("--output", required=True, help="训练输出目录（模型与清单写在此目录）")
    parser.add_argument("--arch", default="tiny", choices=_arch_choices(),
                        help="tiny=内置最小检测头（可跑通全链路）；其他取值见 "
                             "training/export_contract.py --list-frameworks")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-boxes", type=int, default=32,
                        help="单帧输出框数上限，需 ≤ 推理端 max_detections，默认 32")
    parser.add_argument("--width", type=int, default=32, help="主干基础通道数")
    parser.add_argument("--strides", type=int, default=4, help="下采样次数（4 → 步长 16 网格）")
    parser.add_argument("--limit", type=int, default=0, help="仅用前 N 个训练样本（0 = 全部）")
    parser.add_argument("--val-samples", type=int, default=40,
                        help="端到端验证样本数（0 = 跳过验证）")
    parser.add_argument("--image-size", type=int, default=0, help="默认取数据集契约中的尺寸")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--model-id", default="detector-tiny")
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--license", default="Apache-2.0")
    parser.add_argument("--onnx", default="", help="ONNX 路径（默认 <output>/detector.onnx）")
    parser.add_argument("--manifest", default="", help="清单路径（默认 <output>/detector.json）")
    parser.add_argument("--save-state", action="store_true", help="额外保存 torch state_dict")
    parser.add_argument("--allow-copyleft", action="store_true",
                        help="显式确认接受 AGPL 等传染性许可证（仅限内部评测）")
    return parser.parse_args(argv)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 取决于本地环境
        raise SystemExit('训练需要 torch，请先安装：.venv/bin/python -m pip install -e ".[train]"') from exc
    return torch


def _load_dataset(root: Path):
    """读取数据集卡片与样本索引，并校验与推理端一致的输入/输出契约。

    校验规则集中在 :func:`detectors.dataset.load_dataset`，这里只做转发，
    避免训练脚本与 ``export_contract.py`` 出现两套口径。
    """
    from detectors.dataset import load_dataset

    return load_dataset(root)


def _make_loader(torch, records, root, image_size, batch, shuffle, seed):
    """惰性加载图像的数据集（不把整批 float32 图像读进内存）。"""
    import torch as _torch

    class SceneDataset(_torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = list(items)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            record = self.items[index]
            image = np.load(root / record["image"])
            if image.shape != (image_size, image_size):
                raise SystemExit(f"{record['image']} 尺寸 {image.shape} 与契约 "
                                 f"{image_size} 不一致，请重新构建数据集")
            tensor = _torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None]
            boxes = np.asarray(record["boxes"], dtype=np.float32).reshape(-1, 6)[:, :4]
            return tensor, boxes

    def collate(batch_items):
        images = _torch.stack([item[0] for item in batch_items])
        return images, [item[1] for item in batch_items]

    generator = _torch.Generator().manual_seed(int(seed)) if shuffle else None
    return _torch.utils.data.DataLoader(SceneDataset(records), batch_size=batch,
                                        shuffle=shuffle, num_workers=0,
                                        collate_fn=collate, generator=generator)


class TorchRunner:
    """把 torch 模型包装成 ``ml_detect`` 需要的会话接口（与 ModelRunner 同形）。"""

    def __init__(self, model, torch, manifest):
        self._model = model
        self._torch = torch
        self.manifest = dict(manifest)
        self.model_name = str(manifest.get("id", "tiny"))
        self.runtime_version = f"torch {torch.__version__}"

    def run(self, image):
        tensor = self._torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
        with self._torch.no_grad():
            output = self._model(tensor)
        return output.detach().cpu().numpy()


def _pooled(metrics):
    """多场景指标汇总（计数相加，比率按池化重算，避免小场景权重失衡）。"""
    counts = {"true": 0, "detected": 0, "matched": 0, "missed": 0, "false_alarm": 0}
    for item in metrics:
        for key in counts:
            counts[key] += int(item.get(key) or 0)
    precision = counts["matched"] / counts["detected"] if counts["detected"] else None
    recall = counts["matched"] / counts["true"] if counts["true"] else None
    if precision is None and recall is None:
        f1 = None
    else:
        safe_p, safe_r = precision or 0.0, recall or 0.0
        f1 = 2 * safe_p * safe_r / (safe_p + safe_r) if (safe_p + safe_r) > 0 else 0.0
    center = [item["center_mae_hz"] for item in metrics if item.get("center_mae_hz") is not None]
    band = [item["bandwidth_mape"] for item in metrics if item.get("bandwidth_mape") is not None]
    return {
        **counts,
        "scenes": len(metrics),
        "precision": None if precision is None else round(precision, 4),
        "recall": None if recall is None else round(recall, 4),
        "f1": None if f1 is None else round(f1, 4),
        "center_mae_hz": round(float(np.mean(center)), 3) if center else None,
        "bandwidth_mape": round(float(np.mean(band)), 4) if band else None,
    }


def _validate(model, torch, card, records, args):
    """用项目自带的评测口径做端到端验证：重新生成波形 → ml_detect → evaluate_detections。"""
    contract = card["contract"]
    manifest = {
        "id": args.model_id,
        "version": args.version,
        "labels": list(contract["labels"]),
        "input": {
            "name": "images",
            "image_size": int(contract["image_size"]),
            "channels": 1,
            "spectrogram_nfft": int(contract["spectrogram_nfft"]),
            "dynamic_range_db": float(contract["dynamic_range_db"]),
            "layout": contract["layout"],
        },
    }
    runner = TorchRunner(model, torch, manifest)
    model.eval()
    per_scene = []
    val_records = [record for record in records if record["split"] == "val"][:args.val_samples]
    for record in val_records:
        scene = record["scene"]
        samples, generation = generate_iq(scene["rate_hz"], scene["duration_s"], scene["signals"],
                                          noise=scene["noise"], seed=scene["seed"])
        summary, _ = ml_detect(samples, scene["rate_hz"], runner=runner, with_baseline=False)
        per_scene.append(evaluate_detections(signal_truth(generation), summary["detections"]))
        # 结果必须是 JSON 安全的（与运行期契约一致）
        json.dumps(summary["detections"], allow_nan=False, ensure_ascii=False)
    return _pooled(per_scene), per_scene


def _train(model, torch, loader, args, tiny):
    device = torch.device(args.device)
    model.to(device)
    grid = tiny.grid_size(args.image_size, args.strides)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    history = []
    for epoch in range(1, max(args.epochs, 1) + 1):
        total, steps = 0.0, 0
        for images, boxes in loader:
            images = images.to(device)
            targets = [tiny.encode_targets(box, grid, grid) for box in boxes]
            optimizer.zero_grad(set_to_none=True)
            features = model.head(model.backbone(images))
            loss = tiny.detector_loss(features, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
        average = total / max(steps, 1)
        history.append(average)
        print(f"  epoch {epoch:>3}/{args.epochs}  loss {average:.4f}  steps {steps}", flush=True)
    model.eval()
    return history


def _export(model, torch, args, card, metrics, onnx_path, manifest_path):
    """导出 ONNX 并生成模型清单（两者必须同目录）。"""
    try:
        import onnx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 取决于本地环境
        raise SystemExit('导出需要 onnx，请先安装：.venv/bin/python -m pip install -e ".[train]"') from exc

    model.eval()
    dummy = torch.zeros(1, 1, args.image_size, args.image_size,
                        dtype=torch.float32, device=next(model.parameters()).device)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    from detectors.torch_export import export_torch_module

    export_torch_module(model, dummy, onnx_path, opset=args.opset,
                        input_names=["images"], output_names=["detections"], torch=torch)
    contract = card["contract"]
    manifest = write_model_manifest(
        manifest_path, onnx_path,
        identifier=args.model_id, version=args.version,
        image_size=int(contract["image_size"]), opset=args.opset,
        labels=list(contract["labels"]),
        training={"framework": f"torch {torch.__version__}", "arch": args.arch,
                  "license": args.license, "dataset": str(args.data),
                  "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "seed": args.seed, "validation": metrics},
        notes="内置最小检测头（training/tiny_detector.py）；正式精度请接入 YOLOX/RT-DETR",
        spectrogram_nfft=int(contract["spectrogram_nfft"]),
        dynamic_range_db=float(contract["dynamic_range_db"]))
    return manifest


def _dispatch_foreign_arch(args):
    """第三方适配器不在本仓库训练，给出可执行的转交指令。

    这里只做「把活交给对方」的转交说明，不生成任何产物，所以不拦未安装的框架；
    真正的许可证门禁在 :mod:`training.detectors.registry` 里，由
    ``training/export_contract.py`` 在写权重时触发。
    """
    from detectors import registry

    adapter = registry.lookup(args.arch)
    info = adapter.info
    status = []
    if not adapter.is_available():
        status.append(f"    本机未检测到该框架：{adapter.install_hint()}")
    if info.license in registry.COPYLEFT:
        verdict = ("本轮已显式放行 --allow-copyleft"
                   if args.allow_copyleft else "本轮未放行，导出产物时需显式加 --allow-copyleft")
        status.append(f"    {info.license} 是传染性许可证（{verdict}）；产物仅限内部评测，"
                      "不得进入发行包")
    entry = "training/export_contract.py"
    raise SystemExit(
        f"--arch {args.arch}（{info.title}，{info.license}）不在本仓库训练；\n"
        "接入清单与许可证说明见 training/README.md §7 / §8。\n"
        "  1) 先导出原生数据集（这一步不需要安装该框架）：\n"
        f"     .venv/bin/python {entry} --framework {args.arch} --data {args.data} \\\n"
        "         --dataset-only --dataset-output <原生数据集目录>\n"
        f"  2) 用 {info.title} 自己的 trainer 训练、自己的 exporter 导出 ONNX；\n"
        "  3) 再改写成契约图并生成模型清单（这一步才需要 onnx / onnxruntime）：\n"
        f"     .venv/bin/python {entry} --framework {args.arch} --onnx <导出图> \\\n"
        f"         --layout {adapter.layout} --imgsz <图像边长> --output {args.output}\n"
        + ("".join(line + "\n" for line in status)).rstrip()
    )


def main(argv=None):
    args = _parse_args(argv)
    if args.arch != "tiny":
        _dispatch_foreign_arch(args)

    data_root = Path(args.data)
    output = Path(args.output)
    card, records = _load_dataset(data_root)
    contract = card["contract"]
    args.image_size = int(args.image_size or contract["image_size"])
    if args.image_size != int(contract["image_size"]):
        raise SystemExit(f"图像尺寸必须等于数据集契约 {contract['image_size']}，请重建数据集或改用 --image-size 一致取值")
    if not 1 <= args.max_boxes <= 256:
        raise SystemExit("--max-boxes 应在 1～256 之间")

    torch = _import_torch()
    import tiny_detector as tiny

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_records = [record for record in records if record["split"] == "train"]
    if args.limit:
        train_records = train_records[:args.limit]
    if not train_records:
        raise SystemExit("没有可用的训练样本，请检查数据集的划分")

    output.mkdir(parents=True, exist_ok=True)
    model = tiny.TinyDetector(max_boxes=args.max_boxes, classes=len(contract["labels"]),
                              width=args.width, strides=args.strides)
    grid = tiny.grid_size(args.image_size, args.strides)
    print(f"数据集 {data_root}：{card['sample_count']} 个样本，"
          f"训练用 {len(train_records)} 个；输入 {args.image_size}×{args.image_size}，"
          f"网格 {grid}×{grid}")
    print(f"模型：tiny（{tiny.count_parameters(model)} 参数，最多 {args.max_boxes} 框/帧）")

    loader = _make_loader(torch, train_records, data_root, args.image_size,
                          args.batch, True, args.seed)
    history = _train(model, torch, loader, args, tiny)

    metrics = None
    if args.val_samples > 0:
        print("验证：用项目评测口径在验证集上做端到端评测 …")
        metrics, per_scene = _validate(model, torch, card, records, args)
        (output / "validation.json").write_text(json.dumps(
            {"pooled": metrics, "scenes": per_scene, "loss": history},
            ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(f"  召回 {metrics['recall']} 精确率 {metrics['precision']} F1 {metrics['f1']} "
              f"（真值 {metrics['true']} 检出 {metrics['detected']} 虚警 {metrics['false_alarm']}）")

    onnx_path = Path(args.onnx) if args.onnx else output / "detector.onnx"
    manifest_path = Path(args.manifest) if args.manifest else output / "detector.json"
    _export(model, torch, args, card, metrics, onnx_path, manifest_path)
    if args.save_state:
        torch.save(model.state_dict(), output / "detector_state.pt")
    print(f"已导出模型：{onnx_path}")
    print(f"已生成清单：{manifest_path}")
    print(f"下一步：.venv/bin/python training/verify_onnx.py --manifest {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
