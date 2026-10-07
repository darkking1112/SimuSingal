"""Training plans and persistent experiment records; no torch or Qt dependency."""
import json
import math
import subprocess
from pathlib import Path
import shutil

from ..algorithms.amc.ai_model import (CATALOG_VERSION, check_samples, merge_param_sources,
                                       model_spec)
from ..contracts.iq import DEFAULT_IQ_SAMPLES


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


def _model_param_sources(config, arch):
    """取出模型参数的两个来源（旧单参数 + ``model_params`` 字典），只做类型转换。"""
    legacy = {}
    channels = config.get("channels")
    if channels not in (None, ""):
        try:
            widths = [int(part) for part in str(channels).replace("，", ",").split(",")]
        except ValueError as exc:
            raise ValueError(f"channels 需要逗号分隔的整数，收到 {channels!r}") from exc
        legacy["channels"] = widths
    kernel = config.get("kernel")
    if kernel is not None:
        legacy["kernel"] = kernel
    dropout = config.get("dropout")
    if dropout is not None:
        legacy["dropout"] = dropout
    explicit = config.get("model_params") or {}
    if not isinstance(explicit, dict):
        raise ValueError("model_params 应为字典（模型参数 JSON 的解析结果）")
    return legacy, explicit


def model_params(config, arch):
    """解析并校验配置里的模型参数，返回含目录默认值的完整参数字典（非法即报错）。"""
    spec = model_spec(arch)
    legacy, explicit = _model_param_sources(config, spec.id)
    return merge_param_sources(spec, legacy=legacy, explicit=explicit)


def iq_tuning_args(config, arch):
    """校验 IQ 训练的可选项并返回对应的 CLI 参数（未提供的键不产生参数）。

    模型结构参数（channels/kernel/dropout 与 ``model_params``）按模型目录
    （``signal_analysis.algorithms.amc.ai_model``）校验：未知参数、非法取值与
    "同一参数写在两处"都在起任务前拒绝。权重衰减与早停轮数属公共训练配置，
    规则与模型无关；不提供任何键 = 使用目录声明的默认值。
    """
    spec = model_spec(arch)
    legacy, explicit = _model_param_sources(config, spec.id)
    merged = merge_param_sources(spec, legacy=legacy, explicit=explicit)
    args = []
    if "channels" in legacy:
        args += ["--channels", ",".join(str(width) for width in merged["channels"])]
    if "kernel" in legacy:
        args += ["--kernel", str(merged["kernel"])]
    if "dropout" in legacy:
        args += ["--dropout", repr(float(merged["dropout"]))]
    rest = {key: value for key, value in explicit.items() if key not in legacy}
    if rest:
        args += ["--model-params", json.dumps(rest, ensure_ascii=False)]
    weight_decay = config.get("weight_decay")
    if weight_decay is not None:
        if isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float)) \
                or not math.isfinite(float(weight_decay)):
            raise ValueError("weight_decay 应为有限数值")
        if float(weight_decay) < 0:
            raise ValueError("weight_decay 应不小于 0")
        args += ["--weight-decay", repr(float(weight_decay))]
    patience = config.get("patience")
    if patience is not None:
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("早停轮数应为不小于 1 的整数")
        args += ["--patience", str(patience)]
    scheduler = config.get("scheduler")
    if scheduler is not None:
        if scheduler not in ("cosine", "plateau"):
            raise ValueError("学习率调度器应为 cosine（余弦退火）或 plateau（按验证损失降半）")
        args += ["--lr-scheduler", scheduler]
    monitor = config.get("monitor")
    if monitor is not None:
        if monitor not in ("accuracy", "val_loss"):
            raise ValueError("最佳权重判据应为 accuracy（验证准确率）或 val_loss（验证损失）")
        args += ["--monitor", monitor]
    save_checkpoint = config.get("save_checkpoint")
    if save_checkpoint is not None and not isinstance(save_checkpoint, bool):
        raise ValueError("save_checkpoint 应为布尔值（checkpoint 路径由运行时决定）")
    return args


def preflight_iq(config):
    """快照之前的最小预检：模型存在且可导出、参数与窗口约束合法。

    返回补齐窗口长度（``samples``）的配置副本。依赖检查在训练环境内做
    （见 ``training/desktop_worker.py``）：GUI 进程可能没有 torch，不能在这里查。
    """
    arch = config.get("arch", "cnn")
    spec = model_spec(arch)
    if not spec.exportable:
        raise ValueError(f"模型 {spec.id} 尚未通过 ONNX 导出验证，不能作为训练任务运行")
    declared = config.get("catalog_version")
    if declared is not None and int(declared) != CATALOG_VERSION:
        raise ValueError(f"模型目录版本不一致（界面 {declared} / 训练源码 {CATALOG_VERSION}）："
                         "请在「训练配置」里重新点「刷新模型列表」再开始训练")
    iq_tuning_args(config, spec.id)
    samples = config.get("samples")
    samples = DEFAULT_IQ_SAMPLES if samples is None else int(samples)
    check_samples(spec, samples)
    return dict(config, samples=samples)


#: 模型目录查询脚本：在训练环境里读目录并补上每个模型的依赖可用性（界面刷新模型列表用）
_CATALOG_QUERY = (
    "import json, sys; "
    "sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
    "from signal_analysis.algorithms.amc.ai_model import catalog_json; "
    "import amc_models; "
    "payload = json.loads(catalog_json()); "
    "[model.update(missing=list(amc_models.missing_requirements(model['id']))) "
    "for model in payload['models']]; "
    "print(json.dumps(payload, ensure_ascii=False))")


def query_catalog(python, repository, *, timeout=30):
    """用训练环境查询模型目录（含每个模型的依赖可用性）。

    查询失败（Python/仓库路径不对、导入失败、输出无法解析）一律抛 ``ValueError``，
    由界面回退到应用内置目录并标注"未验证"。
    """
    repo = Path(repository).expanduser().resolve()
    for name in ("src", "training"):
        if not (repo / name).is_dir():
            raise ValueError(f"训练源码目录缺少 {name}/：{repo}")
    interpreter = shutil.which(python) or python
    try:
        result = subprocess.run(
            [interpreter, "-c", _CATALOG_QUERY, str(repo / "src"), str(repo / "training")],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"无法在训练环境执行模型目录查询：{exc}") from exc
    if result.returncode != 0:
        lines = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError("训练环境模型目录查询失败：" + (lines[-1] if lines else "未知错误"))
    try:
        payload = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
        int(payload["catalog_version"])
        models = payload["models"]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"训练环境返回的模型目录无法解析：{exc}") from exc
    if not isinstance(models, list) or not models:
        raise ValueError("训练环境返回的模型目录为空")
    return payload


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
    spec = model_spec(arch)
    if not spec.exportable:
        raise ValueError(f"模型 {spec.id} 尚未通过 ONNX 导出验证，不能作为训练任务运行")
    for key, minimum, maximum in (("epochs", 1, 10000), ("batch", 1, 65536)):
        value = config[key]
        if not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{key} 应为 {minimum}～{maximum} 的整数")
    if not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("学习率必须为有限正数")
    if config["device"] not in ("cpu", "cuda"):
        raise ValueError("设备应为 cpu 或 cuda")
    tuning = iq_tuning_args(config, arch)
    out = Path(directory).resolve()
    stages = []

    def stage(name, script, *args):
        stages.append({"name": name, "argv": [python, "-u", str(scripts / script),
                                               *map(str, args)]})

    data = Path(config["data"]).expanduser().resolve()
    for name in ("iq_dataset.json", "iq_dataset.npz"):
        if not (data / name).is_file():
            raise ValueError(f"已有数据集缺少 {name}")
    # 先在选定的训练环境里做完整校验：数据契约 + 依赖 + 按目录参数构建一次模型。
    stages.append({"name": "校验环境与数据", "argv": [python, "-u", "-c",
        "import sys, json; sys.path.insert(0, sys.argv[1]); "
        "import torch, onnx, onnxruntime; from train_iq import load_dataset, _split; "
        "card,x,y,s,snr,source=load_dataset(sys.argv[2]); _split(x,y,s,snr); "
        "assert sys.argv[3]!='cuda' or torch.cuda.is_available(), 'CUDA 不可用'; "
        "from amc_models import build_model; "
        "model=build_model(sys.argv[4], classes=len(card['contract']['classes']), "
        "samples=card['contract']['samples'], params=json.loads(sys.argv[5])); "
        "print('类别:', card['contract']['classes'], "
        "'参数量:', sum(p.numel() for p in model.parameters()), flush=True)",
        str(scripts), str(data), config["device"], arch,
        json.dumps(model_params(config, arch), ensure_ascii=False)]})
    stage("训练与导出", "train_iq.py", "--data", data, "--arch", arch,
          "--epochs", config["epochs"], "--batch-size", config["batch"],
          "--learning-rate", config["lr"], "--seed", config["seed"],
          "--device", config["device"], "--events", "--onnx-dir", out / "model",
          *tuning,
          *(["--save-checkpoint", out / "model" / "iq_checkpoint.pt"]
            if config.get("save_checkpoint") else []))
    stage("模型验收", "verify_iq.py", "--manifest", out / "model/iq_manifest.json",
          "--data", data, "--json", out / "verification.json", "--threads", 1)
    return stages
