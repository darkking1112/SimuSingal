"""Training plans and persistent experiment records; no torch or Qt dependency."""
import json
import math
from pathlib import Path
import shutil


def save_record(directory, record):
    path = Path(directory) / "experiment.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def list_experiments(root):
    records = []
    for path in sorted(Path(root).glob("*/experiment.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or not isinstance(record.get("config"), dict)
                    or not all(key in record for key in ("id", "status"))
                    or "task" not in record["config"]):
                continue
            record["directory"] = str(path.parent)
            records.append(record)
        except (OSError, ValueError, TypeError):
            continue
    return records


def iq_plan(config, directory):
    """Validate inputs before creating any output; return argv lists, never shell text."""
    repo = Path(config["repository"]).expanduser().resolve()
    python = shutil.which(config["python"])
    if not python:
        raise ValueError("训练 Python 不存在，请选择训练环境的解释器")
    scripts = repo / "training"
    for name in ("build_iq_dataset.py", "train_iq.py", "verify_iq.py"):
        if not (scripts / name).is_file():
            raise ValueError(f"训练源码目录缺少 training/{name}")
    arch = config["arch"]
    if arch not in ("cnn", "tcn"):
        raise ValueError("IQ 模型只支持 CNN / TCN")
    for key, minimum, maximum in (("epochs", 1, 10000), ("batch", 1, 65536),
                                  ("per_class", 2, 1000000), ("samples", 64, 65536)):
        value = config[key]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} 应为 {minimum}～{maximum} 的整数")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("学习率必须为有限正数")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("设备应为 cpu 或 cuda")
    low, high = config["snr_low"], config["snr_high"]
    if not all(math.isfinite(x) for x in (low, high)) or low > high:
        raise ValueError("SNR 下限不能大于上限，且必须为有限数")
    out = Path(directory).resolve()
    stages = []

    def stage(name, script, *args):
        stages.append({"name": name, "argv": [python, "-u", str(scripts / script),
                                               *map(str, args)]})

    source = config["source"]
    if source == "existing":
        data = Path(config["data"]).expanduser().resolve()
        for name in ("iq_dataset.json", "iq_dataset.npz"):
            if not (data / name).is_file():
                raise ValueError(f"已有数据集缺少 {name}")
    elif source in ("generator", "bundle"):
        data = out / "data"
        extra = []
        if source == "bundle":
            bundle = Path(config["bundle"]).expanduser().resolve()
            mapping_path = Path(config["mapping"]).expanduser().resolve()
            if not (bundle / "manifest.json").is_file() or not mapping_path.is_file():
                raise ValueError("请选择 TorchSig bundle 和显式类别映射 JSON")
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            from .ml.amc import AMC_CLASSES
            if (not isinstance(mapping, dict) or not mapping or
                    any(not isinstance(k, str) or not k or not isinstance(v, str)
                        or v not in AMC_CLASSES for k, v in mapping.items())):
                raise ValueError("本页生成配置要求 TorchSig 映射目标属于 A09 类别字典")
            extra = ["--torchsig-bundle", bundle, "--torchsig-map", mapping_path]
        stage("准备数据", "build_iq_dataset.py", "--output", data,
              "--per-class", config["per_class"], "--samples", config["samples"],
              "--seed", config["seed"], f"--snr-range={low},{high}", *extra)
    else:
        raise ValueError("未知数据来源")
    # Run the existing full data-contract validator in the selected environment first.
    stages.append({"name": "校验环境与数据", "argv": [python, "-u", "-c",
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import torch, onnx, onnxruntime; from train_iq import load_dataset, _split; "
        "card,x,y,s,snr,source=load_dataset(sys.argv[2]); _split(x,y,s,snr); "
        "assert sys.argv[3]!='cuda' or torch.cuda.is_available(), 'CUDA 不可用'; "
        "print('类别:', card['contract']['classes'], flush=True)",
        str(scripts), str(data), config["device"]]})
    stage("训练与导出", "train_iq.py", "--data", data, "--arch", arch,
          "--epochs", config["epochs"], "--batch-size", config["batch"],
          "--learning-rate", config["lr"], "--seed", config["seed"],
          "--device", config["device"], "--events", "--onnx-dir", out / "model")
    stage("模型验收", "verify_iq.py", "--manifest", out / "model/iq_manifest.json",
          "--data", data, "--json", out / "verification.json", "--threads", 1)
    return stages
