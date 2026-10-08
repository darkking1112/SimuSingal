"""模型包与集合包的导出/导入（跨机器、跨平台迁移用）。

两个包都是 zip：

* **模型包**（``simusignal-model-package.json``）：把模型库条目（``model.json``、
  ``manifest.json``、ONNX、``metrics.json``）打成一个包；导入时逐项校验
  （sha256、契约、类别、窗口，经 ``contracts`` 的清单读取器）；
* **集合包**（``simusignal-collection-package.json``）：把信号集合连同**标注**导出——
  资产采样数据（每条一个 ``.npy``）、资产元数据、目标与参考参数、AMC/检测标签、
  覆盖度、类别字典与标注集定义；导入时经数据层 API 重建（新 id、新集合）。

设计约定：

* 包内一律使用相对路径；zip 只按名读取、不解包到路径，杜绝 zip-slip；导入端逐条校验
  sha256；
* 只导出**当前状态**：每个目标一条当前参考参数版本、每个标注一条当前标签、覆盖度取最新；
  追加式的历史修订留在源机器（包头部 ``state: "current"`` 已声明）；
* 导入是"新建"而不是"合并"：集合/资产/目标/标签全部生成新 id，重名自动加序号后缀，
  已有数据不会被覆盖；**导入后一律为正常状态**（归档与否属于本机管理决定，不随包继承），
  源集合的归档状态记录在包头部；
* 派生资产的父资产关系（``source = "parent:<id>"``）不随包迁移（父资产可能不在包里），
  导入后来源记为 ``derived``；
* 不包含数据版本（``dataset_versions``）、实验与评估记录、模型运行目录（``training/runs``）。
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: 包格式版本：字段含义变更时递增（导入端只接受相同版本）
FORMAT_VERSION = 1
MODEL_PACKAGE_KIND = "simusignal-model-package"
COLLECTION_PACKAGE_KIND = "simusignal-collection-package"
HEADER_NAME = "simusignal-model-package.json"
COLLECTION_HEADER_NAME = "simusignal-collection-package.json"
ASSETS_ENTRY = "assets.jsonl"

#: 安全阀：单个包的文件数、解压后总字节与头部大小上限
MAX_ENTRIES = 200_000
MAX_TOTAL_BYTES = 16 * 1024 ** 3
MAX_HEADER_BYTES = 4 * 1024 * 1024
COPY_CHUNK = 1024 * 1024

#: 随包迁移的目标参考参数字段（与 ``target_versions`` 列一一对应）
VERSION_FIELDS = ("source", "note", "sample_start", "sample_end", "f_low_hz", "f_high_hz",
                  "center_hz", "bandwidth_hz", "nominal_center_hz", "nominal_bandwidth_hz",
                  "signal_type", "waveform_mode", "modulation", "symbol_rate_baud",
                  "hop_rate_hz", "is_hopping", "snr_db", "snr_definition", "power_dbfs",
                  "params_json")


class TransferError(ValueError):
    """包格式非法、校验不通过或路径不可写。"""


# --------------------------------------------------------------------------- 通用工具


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _digest_stream(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(COPY_CHUNK), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _digest_file(path):
    with open(path, "rb") as handle:
        return _digest_stream(handle)


def _clean_entry_name(name):
    """包内条目名必须是不含绝对路径、盘符与回退段的相对 POSIX 路径。"""
    text = str(name)
    if not text or text.startswith("/") or "\\" in text or ":" in text:
        raise TransferError(f"包内条目名非法：{text!r}")
    if any(part in ("", ".", "..") for part in text.split("/")):
        raise TransferError(f"包内条目名非法：{text!r}")
    return text


def _checked_entries(archive):
    """打开包并做安全体检：条目名合法、数量与解压后总大小在上限内。"""
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise TransferError(f"包不可读（不是合法 zip）：{exc}") from exc
    infos = handle.infolist()
    if len(infos) > MAX_ENTRIES:
        raise TransferError(f"包内条目过多（{len(infos)} > {MAX_ENTRIES}）")
    total = 0
    for info in infos:
        _clean_entry_name(info.filename)
        total += int(info.file_size)
    if total > MAX_TOTAL_BYTES:
        raise TransferError(f"包解压后体积过大（{total} 字节 > {MAX_TOTAL_BYTES}）")
    return handle


def _read_header(archive, name, *, kind):
    with _checked_entries(archive) as handle:
        present = set(handle.namelist())
        if name not in present:
            other = COLLECTION_HEADER_NAME if name == HEADER_NAME else HEADER_NAME
            label = "模型包" if other == HEADER_NAME else "集合包"
            hint = f"这看起来是{label}，请改用对应的导入方式" if other in present else ""
            raise TransferError(f"不是一个 {kind} 包：包内缺少 {name}（{hint}）")
        if int(handle.getinfo(name).file_size) > MAX_HEADER_BYTES:
            raise TransferError("包头部文件过大")
        try:
            header = json.loads(handle.read(name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransferError(f"包头部不是合法 JSON：{exc}") from exc
    if not isinstance(header, dict) or header.get("kind") != kind:
        raise TransferError(f"这不是一个 {kind} 包")
    if int(header.get("format_version", 0)) != FORMAT_VERSION:
        raise TransferError(f"包格式版本不受支持：{header.get('format_version')!r}"
                            f"（当前支持 {FORMAT_VERSION}）")
    return header


def _unique_directory(base, name):
    """在 ``base`` 下取一个不冲突的目录名（``名字``、``名字 (2)``…）。"""
    candidate = base / name
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = base / f"{name} ({index})"
        if not candidate.exists():
            return candidate
    raise TransferError(f"名称占用过多：{name}")


def default_package_name(prefix, stamp=None):
    """默认包文件名：``<前缀>-<UTC 时间戳>.zip``（时间戳与训练记录同格式）。

    前缀里的路径分隔符与 Windows 禁用字符换成 ``_``，避免集合/模型名直接落进文件名。
    """
    moment = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(prefix)).strip(" .")
    return f"{safe or 'package'}-{moment}.zip"


def find_collection(workspace, reference):
    """按 id 或名称查集合（界面与 CLI 共用）；找不到时报错并列出可用名称。"""
    text = str(reference or "").strip()
    rows = workspace.list_collections(include_archived=True)
    for row in rows:
        if row["id"] == text:
            return row
    matches = [row for row in rows if row["name"] == text]
    if len(matches) == 1:
        return matches[0]
    names = "、".join(row["name"] for row in rows) or "（当前工作区没有集合）"
    raise TransferError(f"找不到集合：{text or '（空）'}；可用集合：{names}")


def inspect_package(archive):
    """按包头部类型自动分流到模型包/集合包体检（不写任何文件）。"""
    try:
        return inspect_model_package(archive)
    except TransferError as exc:
        if MODEL_PACKAGE_KIND not in str(exc):
            raise
    return inspect_collection_package(archive)


def _optional(value):
    text = None if value is None else str(value).strip()
    return text or None


# --------------------------------------------------------------------------- 模型包


def export_models(workspace_root, destination, *, names=None):
    """把模型库条目导出成一个模型包；``names`` 为空时导出全部。"""
    from . import model_store

    entries = [entry for entry in model_store.list_models(workspace_root)
               if entry.get("status") != "corrupt"]
    if names:
        wanted = list(dict.fromkeys(str(item) for item in names))   # 去重并保持顺序
        picked = []
        for name in wanted:
            match = next((entry for entry in entries if entry["name"] == name), None)
            if match is None:
                raise TransferError(f"模型库里没有模型：{name}")
            picked.append(match)
        entries = picked
    if not entries:
        raise TransferError("模型库为空，没有可导出的模型")

    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    models = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as handle:
        for entry in entries:
            directory = Path(entry["directory"])
            files = []
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.name.startswith("."):
                    continue
                relative = _clean_entry_name(f"models/{entry['name']}/{path.name}")
                handle.write(path, relative)
                files.append({"name": path.name, "size_bytes": path.stat().st_size,
                              "sha256": _digest_file(path)})
            present = {item["name"] for item in files}
            if not {"model.json", "manifest.json"} <= present:
                raise TransferError(f"模型目录不完整（缺 model.json 或 manifest.json）："
                                    f"{entry['name']}")
            models.append({"name": entry["name"], "model_type": entry.get("model_type"),
                           "purpose": entry.get("purpose"), "task": entry.get("task"),
                           "contract": entry.get("contract"),
                           "source_run": entry.get("source_run"),
                           "sha256": entry.get("sha256"), "files": files})
        header = {"kind": MODEL_PACKAGE_KIND, "format_version": FORMAT_VERSION,
                  "created_at": _now(), "state": "current",
                  "source": {"project": "signal_analysis", "package": "model"},
                  "models": models}
        handle.writestr(HEADER_NAME, _json_bytes(header), compress_type=zipfile.ZIP_DEFLATED)
    return {"path": str(destination), "models": [model["name"] for model in models],
            "size_bytes": destination.stat().st_size}


def inspect_model_package(archive):
    """只读体检：列出包里的模型与校验结论（不写任何文件）。"""
    header = _read_header(archive, HEADER_NAME, kind=MODEL_PACKAGE_KIND)
    report = {"kind": header["kind"], "format_version": header["format_version"],
              "created_at": header.get("created_at"), "models": []}
    with _checked_entries(archive) as handle:
        names = set(handle.namelist())
        for model in header.get("models") or []:
            item = {"name": model.get("name"), "model_type": model.get("model_type"),
                    "purpose": model.get("purpose"), "task": model.get("task"),
                    "contract": model.get("contract"), "ok": True, "problems": [],
                    "size_bytes": sum(int(entry.get("size_bytes") or 0)
                                      for entry in model.get("files") or [])}
            for entry in model.get("files") or []:
                archive_name = f"models/{model.get('name')}/{entry.get('name')}"
                if archive_name not in names:
                    item["ok"] = False
                    item["problems"].append(f"缺少文件 {entry.get('name')}")
                    continue
                with handle.open(archive_name) as stream:
                    if _digest_stream(stream) != entry.get("sha256"):
                        item["ok"] = False
                        item["problems"].append(f"{entry.get('name')} 摘要不符")
            report["models"].append(item)
    return report


def import_models(workspace_root, archive, *, name=None):
    """把模型包导入模型库；``name`` 只允许在包内只有一个模型时指定。

    全部校验在临时目录完成（sha256 + 契约/类别/窗口），通过后才移进模型库；
    重名自动加序号后缀；任何一步失败都不留下半个模型。
    """
    from . import model_store

    header = _read_header(archive, HEADER_NAME, kind=MODEL_PACKAGE_KIND)
    models = list(header.get("models") or [])
    if not models:
        raise TransferError("模型包里没有任何模型")
    if name is not None:
        if len(models) != 1:
            raise TransferError("包里有多条模型时不能指定目标名称，请逐个导出后再改名导入")
        try:
            model_store.validate_name(name)
        except model_store.ModelError as exc:
            raise TransferError(str(exc)) from exc

    models_root = model_store.models_root(workspace_root)
    models_root.mkdir(parents=True, exist_ok=True)
    staging = models_root / f".import-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    imported = []
    try:
        with _checked_entries(archive) as handle:
            names = set(handle.namelist())
            for model in models:
                source_name = str(model.get("name") or "")
                if not source_name:
                    raise TransferError("包里的模型条目缺少名称")
                try:
                    model_store.validate_name(source_name)
                except model_store.ModelError as exc:
                    raise TransferError(f"包里的模型名称非法：{exc}") from exc
                for entry in model.get("files") or []:
                    file_name = _clean_entry_name(str(entry.get("name") or ""))
                    archive_name = f"models/{source_name}/{file_name}"
                    if archive_name not in names:
                        raise TransferError(f"模型 {source_name} 缺少文件 {file_name}")
                    target = staging / source_name / file_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with handle.open(archive_name) as stream, open(target, "wb") as sink:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: stream.read(COPY_CHUNK), b""):
                            digest.update(chunk)
                            sink.write(chunk)
                    if digest.hexdigest() != entry.get("sha256"):
                        raise TransferError(f"模型 {source_name} 的 {file_name} 摘要不符")
        for model in models:
            source_name = str(model["name"])
            directory = staging / source_name
            final_name = str(name or source_name)
            _validate_model_directory(directory, expected=model)
            if final_name != source_name:
                _rewrite_model_name(directory, final_name)
            destination = _unique_directory(models_root, final_name)
            directory.replace(destination)
            entry = model_store.load_model(workspace_root, destination.name)
            if entry.get("status") == "corrupt":  # pragma: no cover - 已在上面校验过
                raise TransferError(f"模型 {destination.name} 导入后元数据不可读")
            imported.append({"name": destination.name, "source_name": source_name,
                             "model_type": entry.get("model_type"),
                             "purpose": entry.get("purpose"),
                             "contract": entry.get("contract")})
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {"models": imported, "directory": str(models_root)}


def _rewrite_model_name(directory, new_name):
    """同步 ``model.json`` 里的名称（模型库以目录名为准，元数据必须一致）。"""
    entry_path = Path(directory) / "model.json"
    payload = json.loads(entry_path.read_text(encoding="utf-8"))
    payload["name"] = new_name
    entry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_model_directory(directory, *, expected):
    """按 ``model.json`` 声明的任务选用对应契约读取器校验清单与 ONNX 摘要。"""
    from ..contracts import iq as iq_contracts
    from ..contracts import manifest as image_contracts
    from . import model_store

    entry_path = Path(directory) / "model.json"
    try:
        entry = json.loads(entry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TransferError(f"模型元数据不可读：{exc}") from exc
    if not isinstance(entry, dict) or not str(entry.get("name") or ""):
        raise TransferError("模型元数据缺少 name")
    task = str(entry.get("task") or "")
    if task not in model_store.MANIFEST_NAMES:
        raise TransferError(f"模型元数据里的任务类型不受支持：{task!r}")
    manifest_name = str(entry.get("manifest") or "manifest.json")
    manifest_path = Path(directory) / manifest_name
    if not manifest_path.is_file():
        raise TransferError(f"模型缺少清单文件：{manifest_name}")
    try:
        if task == "iq":
            iq_contracts.read_iq_manifest(manifest_path)
        else:
            image_contracts.read_model_manifest(manifest_path)
    except (iq_contracts.IQModelError, image_contracts.ManifestError) as exc:
        raise TransferError(f"模型清单校验不通过：{exc}") from exc
    if not (Path(directory) / str(entry.get("library") or "")).is_file():
        raise TransferError(f"模型缺少权重文件：{entry.get('library')!r}")
    if expected.get("task") not in (None, task):
        raise TransferError(f"包内登记的 task 与 model.json 不一致："
                            f"{expected.get('task')} / {task}")


# --------------------------------------------------------------------------- 集合包


def _ordered_targets(targets):
    """父目标先建：跳目标的 ``parent_target_id`` 必须已经存在。"""
    return sorted(targets, key=lambda row: (row.get("scope") == "hop",
                                            row.get("hop_index") or 0,
                                            row.get("target_key") or ""))


def _label_record(task_set, row, target_keys):
    record = {"task": task_set["task"], "target": target_keys[row["target_id"]],
              "task_set": task_set["name"], "source": row.get("source"),
              "note": row.get("note"), "revision_no": row.get("revision_no")}
    fields = (("class_state", "class_name", "window_start", "window_end",
               "analysis_center_hz", "analysis_bandwidth_hz")
              if task_set["task"] == "amc"
              else ("class_name", "include", "label_semantics"))
    for field in fields:
        record[field] = row.get(field)
    return record


def export_collection(workspace, collection_id, destination):
    """导出一个信号集合（资产数据、目标与参考参数、标签、覆盖度、字典与标注集）。"""
    collection = workspace.get_collection(collection_id)
    asset_ids = workspace.collection_asset_ids(collection_id)
    if not asset_ids:
        raise TransferError(f"集合「{collection['name']}」没有任何资产，无需导出")

    task_sets = workspace.list_task_sets(collection_id)
    taxonomies, seen, task_set_entries = [], set(), []
    for task_set in task_sets:
        taxonomy = workspace.get_taxonomy(task_set["taxonomy_id"])
        key = (taxonomy["task"], taxonomy["name"], taxonomy["version"])
        if key not in seen:
            seen.add(key)
            taxonomies.append({"task": taxonomy["task"], "name": taxonomy["name"],
                               "version": taxonomy["version"],
                               "classes": json.loads(taxonomy["classes_json"]),
                               "mapping": (json.loads(taxonomy["mapping_json"])
                                           if taxonomy.get("mapping_json") else None)})
        task_set_entries.append({"task": task_set["task"], "name": task_set["name"],
                                 "taxonomy": {"task": taxonomy["task"],
                                              "name": taxonomy["name"],
                                              "version": taxonomy["version"]},
                                 "label_semantics": task_set.get("label_semantics"),
                                 "created_by": task_set.get("created_by")})

    # 标签与覆盖度按标注集一次读出，再按目标归属分发到各资产（标签表里没有资产列）
    labels_by_target = {}
    for task_set in task_sets:
        rows = workspace.current_labels(task_set["task"], task_set["id"], include_all=True)
        labels_by_target[task_set["id"]] = {row["target_id"]: row for row in rows}

    recipe = None
    if collection.get("recipe_id"):
        try:
            row = workspace.get_recipe(collection["recipe_id"])
        except ValueError:
            row = None
        if row:
            recipe = {"name": row["name"], "engine": row["engine"],
                      "recipe": json.loads(row["recipe_json"])}

    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts = {"assets": 0, "targets": 0, "versions": 0, "labels": 0, "coverage": 0}
    lines = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as handle:
        for index, asset_id in enumerate(asset_ids, start=1):
            asset, samples = workspace.load_samples(asset_id)
            array = np.ascontiguousarray(np.asarray(samples, dtype=np.complex64))
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            payload = buffer.getvalue()
            entry_name = _clean_entry_name(f"assets/{index:06d}-{asset_id}.npy")
            handle.writestr(entry_name, payload, compress_type=zipfile.ZIP_STORED)

            record = {"file": entry_name, "name": asset["name"],
                      "sample_rate": asset["sample_rate"], "source": asset.get("source"),
                      "source_kind": asset.get("source_kind"),
                      "origin_group_id": asset.get("origin_group_id"),
                      "rf_center_hz": asset.get("rf_center_hz"),
                      "capture_started_at": asset.get("capture_started_at"),
                      "created_by": asset.get("created_by"),
                      "archived": bool(asset.get("archived_at")),
                      "sample_count": int(array.size), "sha256": _digest_bytes(payload),
                      "metadata": workspace.get_metadata(asset_id),
                      "targets": [], "labels": [], "coverage": []}
            target_keys, asset_targets = {}, set()
            for target in _ordered_targets(workspace.list_targets(asset_id, with_current=False)):
                version = workspace.current_target_version(target["id"])
                target_keys[target["id"]] = target["target_key"]
                asset_targets.add(target["id"])
                record["targets"].append({
                    "key": target["target_key"], "scope": target["scope"],
                    "hop_index": target.get("hop_index"),
                    "parent": target_keys.get(target.get("parent_target_id")),
                    "for_detection": int(target.get("for_detection") or 0),
                    "for_amc": int(target.get("for_amc") or 0),
                    "version": ({field: version.get(field) for field in VERSION_FIELDS}
                                if version else None)})
            counts["versions"] += sum(1 for item in record["targets"] if item["version"])
            for task_set in task_sets:
                for target_id, row in labels_by_target[task_set["id"]].items():
                    if target_id in asset_targets:
                        record["labels"].append(_label_record(task_set, row, target_keys))
                coverage = workspace.get_asset_coverage(task_set["id"], asset_id)
                if coverage:
                    record["coverage"].append({"task": task_set["task"],
                                               "coverage": coverage["coverage"],
                                               "negative_kind": coverage.get("negative_kind"),
                                               "source": coverage.get("source")})
            counts["assets"] += 1
            counts["targets"] += len(record["targets"])
            counts["labels"] += len(record["labels"])
            counts["coverage"] += len(record["coverage"])
            lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
        body = ("\n".join(lines) + "\n").encode("utf-8")
        handle.writestr(ASSETS_ENTRY, body, compress_type=zipfile.ZIP_DEFLATED)
        header = {"kind": COLLECTION_PACKAGE_KIND, "format_version": FORMAT_VERSION,
                  "created_at": _now(), "state": "current",
                  "source": {"project": "signal_analysis", "package": "collection"},
                  "collection": {"name": collection["name"],
                                 "description": collection.get("description") or "",
                                 "source_kind": collection.get("source_kind"),
                                 "created_by": collection.get("created_by"),
                                 "created_at": collection.get("created_at"),
                                 "archived": bool(collection.get("archived_at"))},
                  "recipe": recipe, "taxonomies": taxonomies,
                  "task_sets": task_set_entries,
                  "asset_count": counts["assets"], "counts": counts,
                  "assets_file": ASSETS_ENTRY, "assets_sha256": _digest_bytes(body)}
        handle.writestr(COLLECTION_HEADER_NAME, _json_bytes(header),
                        compress_type=zipfile.ZIP_DEFLATED)
    return {"path": str(destination), "collection": collection["name"],
            "size_bytes": destination.stat().st_size, **counts}


def inspect_collection_package(archive):
    """只读体检：集合概况与计数 + 资产文件完整性（不写任何文件）。"""
    header = _read_header(archive, COLLECTION_HEADER_NAME, kind=COLLECTION_PACKAGE_KIND)
    report = {"kind": header["kind"], "format_version": header["format_version"],
              "created_at": header.get("created_at"),
              "collection": header.get("collection"), "counts": header.get("counts"),
              "task_sets": header.get("task_sets"), "taxonomies": header.get("taxonomies"),
              "ok": True, "problems": []}
    with _checked_entries(archive) as handle:
        if ASSETS_ENTRY not in handle.namelist():
            raise TransferError(f"集合包缺少 {ASSETS_ENTRY}")
        body = handle.read(ASSETS_ENTRY)
        if _digest_bytes(body) != header.get("assets_sha256"):
            report["ok"] = False
            report["problems"].append("资产索引摘要不符")
            return report
        names = set(handle.namelist())
        missing = sum(1 for line in body.decode("utf-8").splitlines() if line.strip()
                      and json.loads(line).get("file") not in names)
        if missing:
            report["ok"] = False
            report["problems"].append(f"缺少 {missing} 个资产文件")
    return report


def import_collection(workspace, archive, *, name=None):
    """把集合包导入当前工作区：新建集合、资产、目标与参考参数、标签与覆盖度。

    资产按包内顺序重建（集合成员顺序一致），全部使用新 id；集合重名自动加序号。
    数据库写入合并成一个事务：任一步失败都会整体回滚，工作区不会留下半个集合；
    已落盘的资产文件会在回滚后一并删除。
    """
    header = _read_header(archive, COLLECTION_HEADER_NAME, kind=COLLECTION_PACKAGE_KIND)
    collection_info = header.get("collection") or {}
    target_name = str(name or collection_info.get("name") or "").strip()
    if not target_name:
        raise TransferError("集合包没有集合名称")

    with _checked_entries(archive) as handle:
        body = handle.read(ASSETS_ENTRY)
        if _digest_bytes(body) != header.get("assets_sha256"):
            raise TransferError("集合包的资产索引摘要不符（包可能被修改或损坏）")
        records = [json.loads(line) for line in body.decode("utf-8").splitlines()
                   if line.strip()]
        names = set(handle.namelist())
        for record in records:
            if record.get("file") not in names:
                raise TransferError(f"集合包缺少资产文件：{record.get('file')}")

        assets_dir = workspace.root / "assets"
        before_files = set(assets_dir.glob("*")) if assets_dir.is_dir() else set()
        counts = {"assets": 0, "targets": 0, "labels": 0, "coverage": 0}
        skipped, asset_ids = [], []
        # 逐条写入会产生上千次提交（每次一个 fsync，实测占总耗时九成以上），因此把配方、
        # 集合、任务集、资产与标注全部合进一个事务：要么整包落库，要么什么都不留。
        # 数据库能回滚，落盘的资产文件不能，所以失败后按差集删掉这次新增的文件。
        try:
            with workspace.batch():
                recipe_id = None
                recipe = header.get("recipe")
                if recipe:
                    row = workspace.ensure_recipe(str(recipe.get("name") or "导入配方"),
                                                  str(recipe.get("engine") or "gen_recipe_v1"),
                                                  recipe.get("recipe"))
                    recipe_id = row["id"]
                collection = workspace.create_collection(
                    _unique_collection_name(workspace, target_name),
                    description=str(collection_info.get("description") or ""),
                    source_kind=str(collection_info.get("source_kind") or "imported"),
                    recipe_id=recipe_id, created_by=_optional(collection_info.get("created_by")))
                collection_id = collection["id"]
                collection_name = collection["name"]

                taxonomy_ids = _ensure_taxonomies(workspace, header.get("taxonomies") or [])
                task_sets = _create_task_sets(workspace, collection_id, header, taxonomy_ids)
                task_by_name = {f"{task_set['task']}|{task_set['name']}": task_set
                                for task_set in task_sets}
                task_by_task = {}
                for task_set in task_sets:
                    task_by_task.setdefault(task_set["task"], task_set)

                for record in records:
                    payload = handle.read(record["file"])
                    if _digest_bytes(payload) != record.get("sha256"):
                        raise TransferError(f"资产 {record.get('name')} 摘要不符，导入中止")
                    array = np.load(io.BytesIO(payload), allow_pickle=False)
                    source = str(record.get("source") or "imported")
                    if source.startswith("parent:"):  # 派生资产的父资产不一定在包里
                        source = "derived"
                    asset = workspace.add_samples(
                        np.asarray(array, dtype=np.complex64), float(record["sample_rate"]),
                        str(record.get("name") or "导入资产"), source=source,
                        metadata=_strip_generation(record.get("metadata")),
                        source_kind=str(record.get("source_kind") or "imported"),
                        origin_group_id=_optional(record.get("origin_group_id")),
                        rf_center_hz=record.get("rf_center_hz"),
                        capture_started_at=_optional(record.get("capture_started_at")),
                        created_by=_optional(record.get("created_by")),
                        storage_format="npy")
                    asset_ids.append(asset["id"])
                    counts["assets"] += 1
                    target_ids = _restore_targets(workspace, asset, record.get("targets") or [])
                    counts["targets"] += len(record.get("targets") or [])
                    written, dropped = _restore_labels(workspace, task_by_name, task_by_task,
                                                       target_ids, record.get("labels") or [])
                    counts["labels"] += written
                    skipped.extend(dropped)
                    counts["coverage"] += _restore_coverage(workspace, task_by_task, asset["id"],
                                                            record.get("coverage") or [])
                if asset_ids:
                    workspace.add_collection_members(collection_id, asset_ids, added_by="import")
        except BaseException:
            for path in assets_dir.glob("*"):
                if path not in before_files:
                    path.unlink(missing_ok=True)
            raise
    return {"collection_id": collection_id, "collection": collection_name,
            "task_sets": len(task_sets), "skipped_labels": skipped, **counts}


def _strip_generation(metadata):
    """``add_samples`` 看到 ``metadata["generation"]`` 会自动登记生成器目标，与包里的目标
    重复（目标键唯一约束会报错），因此导入时把这段摘要摘掉；原始摘要仍留在包里可查。"""
    if not isinstance(metadata, dict):
        return metadata or None
    rest = {key: value for key, value in metadata.items() if key != "generation"}
    return rest or None


def _unique_collection_name(workspace, name):
    existing = {row["name"] for row in workspace.list_collections(include_archived=True)}
    if name not in existing:
        return name
    for index in range(2, 1000):
        candidate = f"{name} ({index})"
        if candidate not in existing:
            return candidate
    raise TransferError(f"集合名称占用过多：{name}")


def _ensure_taxonomies(workspace, entries):
    """按 (task, name, version) 复用已有字典；同名版本但内容不同则报错。"""
    resolved = {}
    for entry in entries:
        task = str(entry.get("task") or "")
        name = str(entry.get("name") or "")
        version = str(entry.get("version") or "")
        classes = [str(item) for item in entry.get("classes") or []]
        if not task or not name or not classes:
            raise TransferError("集合包里的类别字典不完整")
        existing = next((row for row in workspace.list_taxonomies(task)
                         if row["name"] == name and row["version"] == version), None)
        if existing is not None:
            if json.loads(existing["classes_json"]) != classes:
                raise TransferError(f"类别字典 {task}/{name}/{version} 与本地同名版本内容不同，"
                                    "请先处理冲突再导入")
            resolved[(task, name, version)] = existing["id"]
            continue
        created = workspace.create_taxonomy(task, name, version, classes,
                                            mapping=entry.get("mapping"))
        resolved[(task, name, version)] = created["id"]
    return resolved


def _create_task_sets(workspace, collection_id, header, taxonomy_ids):
    created = []
    for entry in header.get("task_sets") or []:
        taxonomy = entry.get("taxonomy") or {}
        key = (str(taxonomy.get("task") or ""), str(taxonomy.get("name") or ""),
               str(taxonomy.get("version") or ""))
        taxonomy_id = taxonomy_ids.get(key)
        if taxonomy_id is None:
            taxonomy_id = workspace.default_taxonomy(str(entry.get("task") or "amc"))["id"]
        created.append(workspace.create_task_set(
            collection_id, str(entry.get("task") or ""),
            name=str(entry.get("name") or ""), taxonomy_id=taxonomy_id,
            label_semantics=entry.get("label_semantics"),
            created_by=_optional(entry.get("created_by"))))
    return created


def _restore_targets(workspace, asset, targets):
    """按导出顺序重建目标与当前参考参数版本；返回 ``{目标键: 新目标 id}``。"""
    target_ids = {}
    for entry in targets:
        parent_key = entry.get("parent")
        target = workspace.add_target(
            asset["id"], str(entry.get("key") or ""), str(entry.get("scope") or "session"),
            parent_target_id=target_ids.get(parent_key) if parent_key else None,
            hop_index=entry.get("hop_index"),
            for_detection=int(entry.get("for_detection") or 0),
            for_amc=int(entry.get("for_amc") or 0))
        target_ids[entry.get("key")] = target["id"]
        version = entry.get("version")
        if version:
            fields = {field: version.get(field) for field in VERSION_FIELDS}
            fields["source"] = str(fields.get("source") or "import")
            workspace.append_target_version(target["id"], **fields)
    return target_ids


def _restore_labels(workspace, task_by_name, task_by_task, target_ids, labels):
    """重建标签；目标或标注集缺失时记入 skipped（返回 ``(写入数, 跳过清单)``）。"""
    written, skipped = 0, []
    for entry in labels:
        task = str(entry.get("task") or "")
        task_set = task_by_name.get(f"{task}|{entry.get('task_set')}") \
            or task_by_task.get(task)
        target_id = target_ids.get(entry.get("target"))
        if task_set is None or target_id is None:
            skipped.append({"target": entry.get("target"), "task": task,
                            "reason": "标注集或目标缺失"})
            continue
        source = str(entry.get("source") or "import")
        if task == "amc":
            workspace.append_amc_label(
                task_set["id"], target_id, source=source,
                class_state=str(entry.get("class_state") or "unknown"),
                class_name=_optional(entry.get("class_name")),
                window_start=entry.get("window_start"), window_end=entry.get("window_end"),
                analysis_center_hz=entry.get("analysis_center_hz"),
                analysis_bandwidth_hz=entry.get("analysis_bandwidth_hz"),
                note=_optional(entry.get("note")))
        else:
            workspace.append_detection_label(
                task_set["id"], target_id, source=source,
                class_name=_optional(entry.get("class_name")),
                include=bool(entry.get("include", 1)),
                label_semantics=entry.get("label_semantics"),
                note=_optional(entry.get("note")))
        written += 1
    return written, skipped


def _restore_coverage(workspace, task_by_task, asset_id, coverage):
    written = 0
    for entry in coverage:
        task_set = task_by_task.get(str(entry.get("task") or ""))
        if task_set is None:
            continue
        workspace.set_asset_coverage(
            task_set["id"], asset_id, str(entry.get("coverage") or "unknown"),
            negative_kind=_optional(entry.get("negative_kind")),
            source=str(entry.get("source") or "import"))
        written += 1
    return written
