#!/usr/bin/env python3
"""External-process entry point for the desktop training workbench."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def event(**values):
    print("TRAIN_EVENT " + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


def execute(config, output):
    from signal_analysis.annotations import AnnotationDataset, append_asset
    output = Path(output).resolve()
    import importlib.metadata
    packages = {}
    for name in ("torch", "torchvision", "onnx", "onnxruntime", "ultralytics", "torchsig"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output / "environment.json").write_text(json.dumps(
        {"python": sys.executable, "version": sys.version, "packages": packages}, indent=2),
        encoding="utf-8")
    task = config.get("task", "iq")
    if task == "asset":
        append_asset(config["data"], config["workspace"], config["asset_id"])
        return
    if task == "generate":
        command = [sys.executable, "-u", str(ROOT / "training/build_dataset.py"),
                   "--output", str(output / "data"), "--count", str(config["count"]),
                   "--image-size", str(config["image_size"]), "--nfft", "512",
                   "--seed", str(config["seed"]), "--min-bandwidth-ratio", "0.05"]
        event(stage="生成时频图数据集")
        subprocess.run(command, check=True)
        return
    if task == "torchsig":
        event(stage="生成 TorchSig bundle")
        subprocess.run([sys.executable, "-u", str(ROOT / "training/build_torchsig.py"),
                        "--output", str(output / "bundle"), "--count", str(config["count"]),
                        "--seed", str(config["seed"])], check=True)
        event(stage="转换时频图与检测框")
        # TorchSig 上游偶有 ~1 Hz 带宽的实例（1024px 下不足 1 像素），撞上最小框规则时
        # 整条记录被拒绝；GUI 由用户驱动、无法"修正 bundle 元数据"，因此显式跳过并
        # 计数——拒绝统计会打印到日志并写入数据卡片的 ingest.rejected，不是静默丢弃。
        subprocess.run([sys.executable, "-u", str(ROOT / "training/ingest_torchsig.py"),
                        "--bundle", str(output / "bundle"), "--output", str(output / "data"),
                        "--image-size", str(config["image_size"]), "--skip-rejected"],
                       check=True)
        return
    if task == "iq":
        from signal_analysis.training_jobs import iq_plan
        if config["source"] == "existing":
            # Snapshot before validation/training; later edits cannot change this experiment.
            import shutil
            shutil.copytree(config["data"], output / "iq_data")
            config = dict(config, data=str(output / "iq_data"))
        for stage in iq_plan(config, output):
            event(stage=stage["name"])
            subprocess.run(stage["argv"], cwd=ROOT, check=True)
        return
    if task != "detection":
        raise ValueError("未知训练任务")
    import torch
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("所选训练环境的 CUDA 不可用，请选择 CPU 或配置 CUDA 环境")
    if config["arch"] == "yolo26s":
        try:
            import ultralytics
        except ImportError as exc:
            raise ValueError("YOLO26s 需要在训练 Python 环境安装 ultralytics、onnx、onnxruntime") from exc
    event(stage="校验标注并保存数据快照")
    data = AnnotationDataset(config["data"]).snapshot(output / "data")
    from detectors.dataset import load_dataset
    from detectors.labels import export_yolo, export_coco
    card, records = load_dataset(data)
    if card["contract"]["labels"] != ["emitter"]:
        raise ValueError("检测训练第一版只支持单类 emitter")
    native = output / "native_data"
    model_dir = output / "model"
    model_dir.mkdir()
    size = card["contract"]["image_size"]
    arch = config["arch"]
    if arch == "yolo26s":
        from ultralytics import YOLO
        from detectors import registry
        import detectors.ultralytics  # registers the adapter
        event(stage="导出 YOLO 数据集")
        export_yolo(data, records, card, native)
        # Native model uses RGB (replicated grayscale), matching our 3-channel adapter.
        # data.yaml 由 labels.py 以 UTF-8 写出（含中文注释），读写都必须显式指定 UTF-8
        yaml_path = native / "data.yaml"
        yaml_path.write_text(
            yaml_path.read_text(encoding="utf-8").replace("channels: 1", "channels: 3"),
            encoding="utf-8")
        model = YOLO(config["weights"] or "yolo26s.pt")

        def progress(trainer):
            values = trainer.loss_items
            if isinstance(values, dict):
                loss = sum(float(value.detach().sum().cpu()) if hasattr(value, "detach")
                           else float(value) for value in values.values())
            else:
                loss = float(values.detach().sum().cpu())
            event(epoch=int(trainer.epoch) + 1,
                  loss=loss)

        model.add_callback("on_train_epoch_end", progress)
        event(stage="YOLO26s 训练")
        model.train(data=str(yaml_path), epochs=config["epochs"], batch=config["batch"],
                    imgsz=size, device=config["device"], lr0=config["lr"], seed=config["seed"],
                    project=str(output), name="native_run", exist_ok=False, workers=0,
                    # Geometry must preserve time/frequency orientation and aspect ratio.
                    fliplr=0, flipud=0, degrees=0, shear=0, perspective=0,
                    mosaic=0, mixup=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0)
        weights = Path(model.trainer.best)
        if not weights.is_file():
            raise ValueError("训练未产出最佳权重")
        adapter = registry.lookup("ultralytics")
        event(stage="导出并适配 ONNX")
        source = adapter.export_native_onnx(weights, image_size=size, output=model_dir)
        adapter.export_onnx(source=source, output=model_dir, data_root=data,
                            image_size=size, layout="pixel_xyxy", max_boxes=32,
                            allow_copyleft=True, weights=str(weights), input_scale=1)
    elif arch == "rtdetr":
        event(stage="导出 COCO 数据集")
        export_coco(data, records, card, native)
        # COCO writer uses 1-based category ids; upstream remap=False requires 0-based ids.
        for path in (native / "annotations").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for category in payload["categories"]:
                category["id"] -= 1
            for annotation in payload["annotations"]:
                annotation["category_id"] -= 1
            path.write_text(json.dumps(payload), encoding="utf-8")
        event(stage="RT-DETRv2 训练与原生导出")
        subprocess.run([sys.executable, "-u", str(ROOT / "training/rtdetr_desktop.py"),
                        "--config", str(output / "experiment.json")], check=True)
        from detectors import registry
        import detectors.rtdetr
        registry.lookup("rtdetr").export_onnx(
            source=model_dir / "native.onnx", output=model_dir, data_root=data,
            image_size=size, layout="pixel_xyxy", max_boxes=32, input_scale=1)
    else:
        raise ValueError("检测模型应为 rtdetr 或 yolo26s")
    from desktop_evaluate import verify_native, evaluate
    event(stage="原生与契约模型数值对账")
    parity = verify_native(model_dir / "native.onnx", model_dir / "detector.onnx", size)
    (output / "native_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    event(stage="按人工/数据集标签评估")
    evaluate(data, model_dir / "detector.onnx", model_dir / "metrics.json")
    event(stage="模型契约验收")
    subprocess.run([sys.executable, "-u", str(ROOT / "training/verify_onnx.py"),
                    "--manifest", str(model_dir / "detector.json"),
                    "--json", str(output / "verification.json")], check=True)


def main():
    if os.name == "posix":
        os.setsid()  # child trainers share this group; desktop cancellation stops all of them
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    args = parser.parse_args()
    path = Path(args.record).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    execute(record["config"], path.parent)


if __name__ == "__main__":
    main()
