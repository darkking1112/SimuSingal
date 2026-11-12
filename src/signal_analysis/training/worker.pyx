# cython: language_level=3
"""编译后的外部训练入口：集合快照、任务分发与检测训练限额。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def event(**values):
    print("TRAIN_EVENT " + json.dumps(values, ensure_ascii=False, allow_nan=False), flush=True)


class StageReporter:
    """把集合快照进度写成 TRAIN_EVENT（外部训练任务没有 job_dir）。"""

    def __init__(self):
        self._last = 0.0

    def emit(self, done, total, message, **_extra):
        now = time.monotonic()
        if int(done) != int(total) and now - self._last < 1.0:
            return
        self._last = now
        event(stage=str(message), message=str(message), done=int(done), total=int(total))


def build_training_input(config, output):
    """从训练集／验证集集合构建本次运行的输入快照；返回配置副本与快照报告。"""
    from signal_analysis.data import Workspace
    from signal_analysis.contracts.iq import DEFAULT_IQ_SAMPLES
    from signal_analysis.services.training_inputs import inputs_plan, plan_summary
    from signal_analysis.services.training_snapshot import snapshot

    workspace = Workspace(config["workspace"])
    event(stage="构建训练输入")
    plan = inputs_plan(workspace, config["task"], config["train_collection_id"],
                       config["val_collection_id"])
    (output / "training_inputs.json").write_text(
        json.dumps(plan_summary(plan), ensure_ascii=False, indent=2), encoding="utf-8")
    report = snapshot(workspace, plan, destination=output / "collection_data",
                      image_size=int(config.get("image_size", 1024)),
                      nfft=int(config.get("nfft", 512)),
                      samples=int(config.get("samples", DEFAULT_IQ_SAMPLES)),
                      seed=int(config.get("seed", 0)), reporter=StageReporter())
    event(stage=f"训练输入就绪：{report['samples']} 个样本", samples=report["samples"])
    return dict(config, data=report["path"]), report


def _package_versions(config):
    """采集训练环境的依赖版本；IQ 任务按模型目录声明的依赖动态追加。"""
    import importlib.metadata

    names = ["torch", "torchvision", "onnx", "onnxruntime", "ultralytics", "torchsig"]
    if config.get("task", "iq") == "iq":
        try:
            from signal_analysis.algorithms.amc.ai_model import model_spec

            names += [name for name in model_spec(config.get("arch", "cnn")).requires
                      if name not in names]
        except ValueError:
            pass
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return packages


def preflight(config):
    """快照之前的最小预检：模型/参数/窗口（纯 Python）+ 训练环境依赖（torch 侧）。"""
    from signal_analysis.training import detection_runtime

    if config.get("task", "iq") != "iq":
        detection_runtime.require_rtdetr_allowed(config.get("arch"))
        detection_runtime.guard_detection_epochs(config.get("epochs"))
        return config
    from signal_analysis.services.training_jobs import preflight_iq

    config = preflight_iq(config)
    from signal_analysis.training import _install_aliases

    _install_aliases()
    from amc_models import missing_requirements

    missing = missing_requirements(config["arch"])
    if missing:
        raise ValueError(f"训练环境缺少依赖：{'、'.join(missing)}"
                         f"（模型 {config['arch']}，请在训练 Python 环境安装）")
    return config


def execute(config, output):
    from signal_analysis.training import _install_aliases, detection_runtime

    _install_aliases()
    output = Path(output).resolve()
    from signal_analysis._training_policy import get_build_info

    build_info = get_build_info()
    (output / "environment.json").write_text(json.dumps(
        {"python": sys.executable, "version": sys.version,
         "edition": build_info["edition"],
         "policy_version": build_info["policy_version"],
         "packages": _package_versions(config)}, indent=2),
        encoding="utf-8")
    task = config.get("task", "iq")
    if task not in ("iq", "detection"):
        raise ValueError("未知训练任务")
    config = preflight(config)
    config, snapshot = build_training_input(config, output)
    if task == "iq":
        from signal_analysis.services.training_jobs import iq_plan
        for stage in iq_plan(config, output):
            event(stage=stage["name"])
            subprocess.run(stage["argv"], cwd=str(output), check=True)
        return
    detection_runtime.guard_detection_samples(int(snapshot["splits"]["train"]))
    import torch
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("所选训练环境的 CUDA 不可用，请选择 CPU 或配置 CUDA 环境")
    if config["arch"] == "yolo26s":
        try:
            import ultralytics  # noqa: F401
        except ImportError as exc:
            raise ValueError("YOLO26s 需要在训练 Python 环境安装 ultralytics、onnx、onnxruntime") from exc
    event(stage="校验标注并保存数据快照")
    from signal_analysis.data.annotations import AnnotationDataset

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
        # 初始模型先检查一次；类别调整/重建后的最终模型在首次优化器更新前再检查。
        detection_runtime.guard_detection_model(model.model)

        def check_limits(trainer):
            detection_runtime.guard_detection_model(trainer.model)

        model.add_callback("on_pretrain_routine_end", check_limits)
        model.add_callback("on_train_start", check_limits)

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
        model.train(data=str(yaml_path), epochs=int(config["epochs"]),
                    batch=int(config["batch"]), imgsz=size, device=config["device"],
                    lr0=config["lr"], seed=config["seed"], resume=False,
                    project=str(output), name="native_run", exist_ok=False, workers=0,
                    # Geometry must preserve time/frequency orientation and aspect ratio.
                    fliplr=0, flipud=0, degrees=0, shear=0, perspective=0,
                    mosaic=0, mixup=0, translate=0, scale=0, hsv_h=0, hsv_s=0, hsv_v=0)
        trainer = getattr(model, "trainer", None)
        if trainer is not None and getattr(trainer, "model", None) is not None:
            detection_runtime.guard_detection_model(trainer.model)
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
        detection_runtime.require_rtdetr_allowed(arch)
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
        subprocess.run([sys.executable, "-u", "-c",
                        "from signal_analysis.training.rtdetr_desktop import main; raise SystemExit(main())",
                        "--config", str(output / "experiment.json")], check=True)
        from detectors import registry
        import detectors.rtdetr
        registry.lookup("rtdetr").export_onnx(
            source=model_dir / "native.onnx", output=model_dir, data_root=data,
            image_size=size, layout="pixel_xyxy", max_boxes=32, input_scale=1)
    else:
        raise ValueError("检测模型应为 rtdetr 或 yolo26s")
    from signal_analysis.training.desktop_evaluate import verify_native, evaluate

    event(stage="原生与契约模型数值对账")
    parity = verify_native(model_dir / "native.onnx", model_dir / "detector.onnx", size)
    (output / "native_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
    event(stage="按人工/数据集标签评估")
    evaluate(data, model_dir / "detector.onnx", model_dir / "metrics.json")
    event(stage="模型契约验收")
    subprocess.run([sys.executable, "-u", "-c",
                    "from signal_analysis.training.verify_onnx import main; raise SystemExit(main())",
                    "--manifest", str(model_dir / "detector.json"),
                    "--json", str(output / "verification.json")], check=True)


def _check_build_identity(expected_edition, expected_policy):
    from signal_analysis._training_policy import get_build_info

    info = get_build_info()
    if expected_edition and expected_edition != info["edition"]:
        raise ValueError(
            f"训练环境构建身份不匹配：界面 {expected_edition} / 训练环境 {info['edition']}；"
            "请安装与界面同版本的编译 wheel")
    if expected_policy is not None and int(expected_policy) != int(info["policy_version"]):
        raise ValueError(
            f"训练环境策略版本不匹配：界面 {expected_policy} / 训练环境 {info['policy_version']}；"
            "请安装与界面同版本的编译 wheel")
    return info


def main(argv=None):
    if os.name == "posix":
        os.setsid()  # child trainers share this group; desktop cancellation stops all of them
    from signal_analysis.training import _install_aliases

    _install_aliases()
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--edition", default=None, choices=("trial", "full", "development"))
    parser.add_argument("--policy-version", type=int, default=None)
    args = parser.parse_args(argv)
    _check_build_identity(args.edition, args.policy_version)
    path = Path(args.record).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    execute(record["config"], path.parent)


if __name__ == "__main__":
    main()
