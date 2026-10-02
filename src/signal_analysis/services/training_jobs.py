"""Training plans and persistent experiment records; no torch or Qt dependency."""
import json
import math
from pathlib import Path
import shutil


def decode_worker_log(data):
    """解码 worker.log 字节：新实验为 UTF-8；历史实验由管道默认编码（GBK）写出，逐级回退。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


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
    """Validate inputs before creating any output; return argv lists, never shell text.

    数据来源只有一个：已存在的 IQ 数据集目录（来自所选集合的导出也先落成这样的目录）；
    训练阶段不生成任何数据。
    """
    repo = Path(config["repository"]).expanduser().resolve()
    python = shutil.which(config["python"])
    if not python:
        raise ValueError("训练 Python 不存在，请选择训练环境的解释器")
    scripts = repo / "training"
    for name in ("train_iq.py", "verify_iq.py"):
        if not (scripts / name).is_file():
            raise ValueError(f"训练源码目录缺少 training/{name}")
    arch = config["arch"]
    if arch not in ("cnn", "tcn"):
        raise ValueError("IQ 模型只支持 CNN / TCN")
    for key, minimum, maximum in (("epochs", 1, 10000), ("batch", 1, 65536)):
        value = config[key]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} 应为 {minimum}～{maximum} 的整数")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("学习率必须为有限正数")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("设备应为 cpu 或 cuda")
    out = Path(directory).resolve()
    stages = []

    def stage(name, script, *args):
        stages.append({"name": name, "argv": [python, "-u", str(scripts / script),
                                               *map(str, args)]})

    if config["source"] != "existing":
        raise ValueError("训练页不生成数据：请选择已有数据集，或先在“信号集合生成”里生成集合并选用“所选集合”")
    data = Path(config["data"]).expanduser().resolve()
    for name in ("iq_dataset.json", "iq_dataset.npz"):
        if not (data / name).is_file():
            raise ValueError(f"已有数据集缺少 {name}")
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
