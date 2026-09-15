"""Isolated lyuwenyu RT-DETRv2 bridge; upstream's `src` never enters the GUI process."""
import argparse
import json
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    record_path = Path(args.config).resolve()
    config = json.loads(record_path.read_text())["config"]
    output = record_path.parent
    upstream = Path(config["framework_path"]).resolve()
    if not (upstream / "src/core/yaml_config.py").is_file():
        raise ValueError("请选择 lyuwenyu RT-DETR 的 rtdetrv2_pytorch 目录")
    sys.path.insert(0, str(upstream))
    import torch
    import yaml
    from src.core import YAMLConfig
    from src.solver import TASKS
    from src.misc import dist_utils
    card = json.loads((output / "data/dataset.json").read_text())
    size = card["contract"]["image_size"]
    base = Path(config["framework_config"]).resolve()
    native = output / "native_data"
    generated = {"__include__": [str(base)], "num_classes": 1,
                 "remap_mscoco_category": False, "epoches": config["epochs"],
                 "output_dir": str(output / "native_run"), "device": config["device"],
                 "optimizer": {"lr": config["lr"]}, "eval_spatial_size": [size, size]}
    for split, loader in (("train", "train_dataloader"), ("val", "val_dataloader")):
        generated[loader] = {"total_batch_size": config["batch"], "num_workers": 0,
            "drop_last": False, "dataset": {
                "img_folder": str(native),
                "ann_file": str(native / "annotations" / f"instances_{split}.json"),
                "transforms": {"ops": [
                    {"type": "Resize", "size": [size, size]},
                    {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]
                    + ([{"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True}]
                       if split == "train" else [])}},
            "collate_fn": {"type": "BatchImageCollateFunction", "scales": None}}
    generated_path = output / "rtdetr_config.yml"
    generated_path.write_text(yaml.safe_dump(generated))
    dist_utils.setup_distributed(seed=config["seed"])
    stop_monitor = threading.Event()
    log_path = output / "native_run/log.txt"

    def monitor():
        seen = 0
        while True:
            if log_path.exists():
                lines = log_path.read_text().splitlines()
                for line in lines[seen:]:
                    try:
                        values = json.loads(line)
                    except ValueError:
                        break
                    print("TRAIN_EVENT " + json.dumps({"epoch": values["epoch"] + 1,
                          "loss": values["train_loss"]}, allow_nan=False), flush=True)
                    seen += 1
            if stop_monitor.wait(.5):
                return

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        cfg = YAMLConfig(str(generated_path), tuning=config["weights"] or None)
        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        solver.fit()
        checkpoints = [output / "native_run" / name for name in
                       ("best_stg2.pth", "best_stg1.pth", "best.pth")]
        weights = next((p for p in checkpoints if p.is_file()), None)
        if weights is None:
            raise ValueError("未找到 RT-DETRv2 最佳权重，请检查上游版本和训练日志")
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        model = cfg.model.cpu()
        model.load_state_dict(state)

        class Export(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                self.register_buffer("sizes", torch.tensor([[size, size]]))

            def forward(self, images):
                labels, boxes, scores = self.postprocessor(self.model(images), self.sizes)
                return torch.cat((boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)), -1)

        wrapped = Export().eval()
        torch.onnx.export(wrapped, torch.zeros(1, 3, size, size),
                          str(output / "model/native.onnx"), input_names=["images"],
                          output_names=["detections"], opset_version=17, dynamo=False)
    finally:
        stop_monitor.set()
        watcher.join(timeout=1)
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
