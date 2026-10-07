"""训练模型库：命名规则、入库、盘点、重命名与删除。

模型库位于工作区 ``<workspace>/training/models/<模型名>/``，每个模型是一个自包含目录：

    model.json      本模块维护的元数据（名称、类型、用途、来源实验、参数与指标摘要）
    manifest.json   训练产出的检测/分类清单（原样复制，``library`` 仍相对同目录解析）
    <library>       清单声明的 ONNX；优先硬链接训练产物，跨盘或链接失败时复制
    metrics.json    可选：验收/评估摘要副本

默认命名规则 ``<模型类型>-<训练用途>-<UTC 时间>``，例如 ``cnn-amc-20261006T105213``。
用途决定模型出现在哪个页面的下拉里（``detect`` 信号检测 / ``hop`` 跳频参数 / ``amc`` 调制识别）。

只依赖标准库与 :mod:`common.storage`，不导入 Qt、torch 或推理栈：既能在窗口初始化
时调用，也能被单测直接驱动。
"""
import json
import os
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
import shutil

from common.storage import file_digest

MODELS_DIRNAME = "models"
ENTRY_NAME = "model.json"
MANIFEST_NAME = "manifest.json"
METRICS_NAME = "metrics.json"
ENTRY_SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 64

#: 训练用途代号 → 展示名；用途同时决定「在页面中使用」跳到哪个页面。
PURPOSES = ("detect", "hop", "amc")
PURPOSE_TITLES = {"detect": "信号检测", "hop": "跳频参数", "amc": "调制识别"}
#: 训练任务 → 训练脚本产出的清单文件名。
MANIFEST_NAMES = {"iq": "iq_manifest.json", "detection": "detector.json"}

_INVALID_NAME_CHARS = '/\\:*?"<>|'
_META_KEYS = ("schema_version", "name", "model_type", "purpose", "contract", "created_at",
              "source_run", "task", "manifest", "library", "sha256", "size_bytes",
              "params", "metrics")


class ModelError(ValueError):
    """模型名称不合法、清单缺失或入库/重命名/删除被拒绝。"""


def models_root(workspace_root):
    return Path(workspace_root) / "training" / MODELS_DIRNAME


def run_directory(workspace_root, entry):
    """模型来源实验目录；``source_run`` 缺失时返回 ``None``。"""
    identifier = str((entry or {}).get("source_run") or "").strip()
    return None if not identifier else Path(workspace_root) / "training" / "runs" / identifier


# --------------------------------------------------------------------- 名称规则
def validate_name(name):
    """校验并返回规范化的模型名称；不合法时抛 :class:`ModelError`。"""
    text = str(name or "").strip()
    if not text:
        raise ModelError("模型名称不能为空")
    if len(text) > MAX_NAME_LENGTH:
        raise ModelError(f"模型名称不能超过 {MAX_NAME_LENGTH} 个字符")
    if text.startswith(".") or text in (".", ".."):
        raise ModelError("模型名称不能以点开头")
    if any(character in _INVALID_NAME_CHARS or ord(character) < 32 for character in text):
        raise ModelError('模型名称不能包含 / \\ : * ? " < > | 或控制字符')
    return text


def default_name(model_type, purpose, created_at):
    """默认名称 ``<类型>-<用途>-<UTC 时间>``；与运行目录的时间戳同源，便于和实验对齐。"""
    if purpose not in PURPOSES:
        raise ModelError(f"训练用途应为 {PURPOSES} 之一，实际为 {purpose!r}")
    stamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return validate_name(f"{_slug(model_type)}-{purpose}-{stamp}")


def unique_name(workspace_root, name):
    """自动命名冲突时追加 ``-2``、``-3``…；手动命名不走这里（重名直接报错）。"""
    base = models_root(workspace_root)
    if not (base / name).exists():
        return name
    for index in count(2):
        candidate = f"{name}-{index}"
        if not (base / candidate).exists():
            return candidate


def run_purpose(task, manifest):
    """任务的用途：IQ 分类为 ``amc``；检测按清单的标签语义分会话级/逐跳。"""
    if task == "iq":
        return "amc"
    training = manifest.get("training") if isinstance(manifest.get("training"), dict) else {}
    semantics = str(training.get("label_semantics") or "session_v1")
    return "hop" if semantics == "per_hop_v1" else "detect"


# ------------------------------------------------------------------------- 入库
def publish_from_run(workspace_root, directory, name=None, record=None):
    """把一次成功训练的模型登记进模型库，返回登记后的条目。

    ``name`` 为空时用默认命名；显式命名若已被其它模型占用则报错而不静默改名，
    同一实验重复入库（``source_run`` 相同）则原地更新。
    """
    directory = Path(directory)
    if record is None:
        record = _read_json(directory / "experiment.json")
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    task = str(config.get("task") or "")
    manifest_name = MANIFEST_NAMES.get(task)
    if manifest_name is None:
        raise ModelError(f"任务「{task or '未知'}」没有可入库的模型")
    source_manifest = directory / "model" / manifest_name
    if not source_manifest.is_file():
        raise ModelError(f"未找到模型清单：{manifest_name}")
    payload = _load_manifest(source_manifest)
    library, library_relative = _library_path(payload, source_manifest)
    model_type = _slug(str(config.get("arch") or "").strip() or "model")
    purpose = run_purpose(task, payload)
    created_at = _run_created_at(directory, record)
    source_run = str(record.get("id") or directory.name)
    requested = validate_name(name) if name else default_name(model_type, purpose, created_at)
    base = models_root(workspace_root)
    base.mkdir(parents=True, exist_ok=True)
    target = base / requested
    if target.exists():
        existing = _load_entry(target)
        if existing.get("source_run") != source_run:
            if name:
                raise ModelError(f"模型名称已存在：{requested}")
            requested = unique_name(workspace_root, requested)
            target = base / requested
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, target / MANIFEST_NAME)
    _link_or_copy(library, target / library_relative)
    metrics_source = directory / "model" / METRICS_NAME
    if metrics_source.is_file():
        shutil.copy2(metrics_source, target / METRICS_NAME)
    entry = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "name": requested,
        "model_type": model_type,
        "purpose": purpose,
        "contract": str(payload.get("contract") or ""),
        "created_at": created_at.isoformat(),
        "source_run": source_run,
        "task": task,
        "manifest": MANIFEST_NAME,
        "library": library_relative,
        "sha256": file_digest(target / library_relative),
        "size_bytes": (target / library_relative).stat().st_size,
        "params": _manifest_params(task, payload),
        "metrics": _metrics_summary(directory, task),
    }
    _write_entry(target, entry)
    return _entry_view(target, entry)


def reconcile(workspace_root):
    """把已有训练运行里尚未登记的模型补进模型库；按 ``source_run`` 幂等。

    返回 ``{"added": [...], "updated": [...], "failed": [{"run", "error"}, ...]}``。
    """
    from .training_jobs import list_experiments

    runs = Path(workspace_root) / "training" / "runs"
    added, updated, failed = [], [], []
    if not runs.is_dir():
        return {"added": added, "updated": updated, "failed": failed}
    known = {entry.get("source_run"): entry for entry in list_models(workspace_root)
             if entry.get("source_run")}
    for record in list_experiments(runs):
        run_id = str(record.get("id") or "")
        task = (record.get("config") or {}).get("task")
        if record.get("status") != "success" or task not in MANIFEST_NAMES:
            continue
        directory = Path(record.get("directory") or (runs / run_id))
        if not (directory / "model" / MANIFEST_NAMES[task]).is_file():
            continue
        existing = known.get(run_id)
        try:
            digest = _run_model_digest(directory, task)
            if existing is None:
                added.append(publish_from_run(workspace_root, directory,
                                              record=record)["name"])
            elif digest is not None and digest != existing.get("sha256"):
                updated.append(publish_from_run(workspace_root, directory,
                                                name=existing["name"],
                                                record=record)["name"])
        except (ModelError, OSError, ValueError) as exc:
            failed.append({"run": run_id, "error": str(exc)})
    return {"added": added, "updated": updated, "failed": failed}


# ------------------------------------------------------------------------- 盘点
def list_models(workspace_root):
    """模型库条目，按训练时间倒序；损坏条目也会列出（``status != "ok"``）。"""
    base = models_root(workspace_root)
    if not base.is_dir():
        return []
    entries = [load_model(workspace_root, directory.name) for directory in base.iterdir()
               if directory.is_dir() and not directory.name.startswith(".")]
    entries.sort(key=lambda item: (str(item.get("created_at") or ""), item["name"]),
                 reverse=True)
    return entries


def load_model(workspace_root, name):
    """读取单个条目；目录不存在时报错，元数据读不动时返回 ``status="corrupt"`` 的占位。"""
    directory = models_root(workspace_root) / validate_name(name)
    if not directory.is_dir():
        raise ModelError(f"模型不存在：{name}")
    try:
        payload = _read_json(directory / ENTRY_NAME)
    except (OSError, ValueError) as exc:
        return {"name": directory.name, "directory": str(directory), "path": str(directory),
                "status": "corrupt", "reason": f"模型元数据无法读取：{exc}", "purpose": None,
                "purpose_title": "", "params": {}, "metrics": {}, "size_bytes": 0}
    return _entry_view(directory, payload)


def inspect_model(workspace_root, name):
    """详情视图：在盘点的条目上补算实际 SHA-256，用于核对模型是否被改动。"""
    entry = load_model(workspace_root, name)
    library = Path(entry["directory"]) / str(entry.get("library") or "")
    if entry.get("status") != "corrupt" and library.is_file():
        digest = file_digest(library)
        entry["sha256_actual"] = digest
        entry["digest_ok"] = digest == entry.get("sha256")
    return entry


def rename_model(workspace_root, name, new_name):
    entry = load_model(workspace_root, name)
    if entry.get("status") == "corrupt":
        raise ModelError(f"模型元数据损坏，无法重命名：{entry['name']}")
    base = models_root(workspace_root)
    source = Path(entry["directory"])
    target_name = validate_name(new_name)
    if target_name == entry["name"]:
        return entry
    if source.parent != base:
        raise ModelError(f"拒绝重命名：{source} 不在模型库内")
    target = base / target_name
    if target.exists():
        raise ModelError(f"模型名称已存在：{target_name}")
    source.rename(target)
    payload = {key: entry[key] for key in _META_KEYS if key in entry}
    payload["name"] = target_name
    _write_entry(target, payload)
    return _entry_view(target, payload)


def delete_model(workspace_root, name):
    """删除模型库里的一个模型目录；只允许删模型库内、带合法元数据的目录。"""
    entry = load_model(workspace_root, name)
    directory = Path(entry["directory"])
    base = models_root(workspace_root)
    if directory.parent != base or not (directory / ENTRY_NAME).is_file():
        raise ModelError(f"拒绝删除：{directory} 不是模型库内的模型目录")
    shutil.rmtree(directory)
    return entry


# ------------------------------------------------------------------------- 内部
def _slug(value):
    text = str(value or "").strip().lower()
    cleaned = "".join(character if (character.isalnum() or character in "-._")
                      else "-" for character in text)
    cleaned = cleaned.strip("-._")
    return cleaned or "model"


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_manifest(path):
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ModelError(f"模型清单应为 JSON 对象：{Path(path).name}")
    return payload


def _library_path(payload, manifest_path):
    """解析清单声明的 ONNX（相对清单目录），并把越界路径视为失败。"""
    library = payload.get("library")
    if not isinstance(library, str) or not library.strip():
        raise ModelError("模型清单缺少 library 字段")
    base = Path(manifest_path).resolve().parent
    path = (base / library).resolve()
    if not path.is_file() or not path.is_relative_to(base):
        raise ModelError(f"清单声明的模型文件不存在：{library}")
    return path, str((base / library).relative_to(base).as_posix())


def _run_created_at(directory, record):
    """训练时间：优先运行目录名里的 UTC 时间戳，其次记录里的结束时间/修改时间。"""
    stamp = Path(directory).name.split("-", 1)[0]
    try:
        return datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for key in ("finished", "created"):
        value = str(record.get(key) or "")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.fromtimestamp(Path(directory).stat().st_mtime, timezone.utc)


def _manifest_params(task, payload):
    """从清单里抽取展示用参数；缺失的键不写入，避免详情面板出现成片 null。"""
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    source = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    params = {}
    if task == "iq":
        output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        params.update({"window_samples": source.get("samples"), "channels": source.get("channels"),
                       "classes": list(output.get("classes") or []),
                       "model": training.get("arch"),
                       "model_revision": training.get("model_revision"),
                       "num_params": training.get("num_params"),
                       "epochs": training.get("epochs"), "batch_size": training.get("batch_size"),
                       "learning_rate": training.get("learning_rate"),
                       "best_validation_accuracy": training.get("best_validation_accuracy"),
                       "dataset_samples": training.get("dataset_samples")})
        model_params = training.get("model_params")
        if isinstance(model_params, dict) and model_params:
            params["model_params"] = json.dumps(model_params, ensure_ascii=False)
        else:  # 旧清单：目录参数没有记录成一份 JSON，退回 cnn/tcn 的 dropout 字段
            params["dropout"] = training.get("dropout")
    else:
        dataset = training.get("dataset") if isinstance(training.get("dataset"), dict) else {}
        params.update({"image_size": source.get("image_size"),
                       "spectrogram_nfft": source.get("spectrogram_nfft"),
                       "dynamic_range_db": source.get("dynamic_range_db"),
                       "labels": list(payload.get("labels") or []),
                       "framework": training.get("framework"),
                       "dataset_samples": dataset.get("samples")})
    return {key: value for key, value in params.items() if value not in (None, "")}


def _metrics_summary(directory, task):
    """把训练产出的 ``model/metrics.json`` 压成详情面板用的一组标量。"""
    try:
        payload = _read_json(Path(directory) / "model" / METRICS_NAME)
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if task == "iq":
        validation = payload.get("validation") if isinstance(payload.get("validation"), dict) else {}
        return {key: validation.get(key) for key in ("accuracy", "macro_f1")
                if validation.get(key) is not None}
    summary = {}
    for split, metrics in (payload.get("splits") or {}).items():
        if not isinstance(metrics, dict):
            continue
        summary[split] = {key: metrics.get(key) for key in ("precision", "recall", "f1")
                          if metrics.get(key) is not None}
    return {split: values for split, values in summary.items() if values}


def _run_model_digest(directory, task):
    try:
        manifest_path = Path(directory) / "model" / MANIFEST_NAMES[task]
        payload = _load_manifest(manifest_path)
        library, _ = _library_path(payload, manifest_path)
    except (ModelError, OSError, ValueError):
        return None
    return file_digest(library)


def _link_or_copy(source, target):
    """同盘优先硬链接（不占额外空间），否则复制。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_entry(directory, payload):
    path = Path(directory) / ENTRY_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def _load_entry(directory):
    try:
        payload = _read_json(Path(directory) / ENTRY_NAME)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _entry_view(directory, payload):
    """把元数据补成界面直接可用的条目（目录、清单路径、用途中文名、文件状态）。"""
    directory = Path(directory)
    entry = {key: payload[key] for key in _META_KEYS if key in payload}
    entry.setdefault("name", directory.name)
    entry.setdefault("model_type", "")
    entry.setdefault("params", {})
    entry.setdefault("metrics", {})
    entry["directory"] = str(directory)
    entry["manifest"] = str(entry.get("manifest") or MANIFEST_NAME)
    entry["path"] = str(directory / entry["manifest"])
    entry["purpose_title"] = PURPOSE_TITLES.get(entry.get("purpose"), entry.get("purpose") or "")
    library = entry.get("library")
    library_path = directory / library if isinstance(library, str) and library else None
    if library_path is None or not library_path.is_file():
        entry.update(status="missing", reason="模型文件缺失", size_bytes=0)
    else:
        size = library_path.stat().st_size
        if entry.get("size_bytes") != size:
            entry.update(status="changed", reason="模型文件与登记的大小不符", size_bytes=size)
        else:
            entry.update(status="ok", reason=None)
    return entry
