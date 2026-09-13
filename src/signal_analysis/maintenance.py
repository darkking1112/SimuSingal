"""工作区数据盘点与安全清理（不写运行记录，只读扫描 + 白名单删除）。

本模块刻意只依赖标准库：它不参与 DSP，也不引入 ``numpy``。删除操作的白名单仅限
``jobs/`` 的过期任务与无主（catalog 无记录）的文件，**永不触碰** ``assets/`` 中已入库
的资产、``runs/`` 中已入库的运行目录与 ``exports/``。

统计口径（与文档 ``docs/数据管理页面.md`` 一致）：

* 单个文件的大小取 ``stat().st_size``，**不重算 SHA-256**（100 MB 资产会拖垮扫描）；
* 目录占用为该目录下所有文件之和，不跟随目录符号链接（避免自指死循环）；
* ``catalog_index_bytes`` 为 ``runs.result_json`` 与 ``asset_metadata.metadata_json``
  的 ``length()`` 之和，即结果 JSON 在 SQLite 里与磁盘上的**重复占用**。
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil

from common.storage import utc_now

VERSION = "storage_report_v1"
SETTINGS_NAME = "maintenance.json"
LOG_NAME = "maintenance.log"
DEFAULT_JOB_RETENTION_DAYS = 30

#: 扫描软上限：额外目录指到巨型目录树时截断，保证页面不挂死。
MAX_FILES = 200_000
#: 表格行上限；总计另有独立的目录遍历口径，不受截断影响。
MAX_ASSET_ROWS = 500
MAX_RUN_ROWS = 1000
MAX_JOB_ROWS = 500
MAX_EXPORT_ROWS = 500
MAX_LIST_ROWS = 200

#: 允许删除的顶层子目录；``exports`` 不在其中（导出物本就不入库，无法判定归属）。
CLEANABLE_ROOTS = ("jobs", "assets", "runs")
#: ``status.json`` 里代表任务已结束的状态。
TERMINAL_STATES = ("success", "failed", "cancelled")

RUN_KIND_LABELS = {
    "analysis": "通用统计",
    "detect": "能量检测",
    "detect_hops": "逐跳参数",
    "ml_detect": "AI 检测",
    "ml_detect_hops": "AI 逐跳",
    "amc_classify": "调制识别",
    "amc_iq_classify": "原始 IQ 识别",
    "native": "原生插件",
    "generate": "IQ 生成",
    "simulation": "仿真",
}

CATEGORY_LABELS = {
    "assets": "信号资产",
    "runs": "运行记录",
    "jobs": "任务工件",
    "exports": "导出文件",
    "catalog": "索引数据库",
    "settings": "页面设置",
}


def format_bytes(size):
    """人类可读的容量文本；负数与非法输入按 0 处理，不产生 NaN。"""
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "0 B"
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return "0 B"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


def _parse_time(text):
    try:
        moment = datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _age_days(text, now):
    moment = _parse_time(text)
    if moment is None:
        return None
    return (now - moment).total_seconds() / 86400.0


def _scan_tree(root, budget=MAX_FILES):
    """返回 ``(字节数, 文件数, 是否截断)``；不跟随目录符号链接。"""
    total = 0
    count = 0
    pending = [Path(root)]
    while pending:
        current = pending.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if count >= budget:
                return total, count, True
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                    continue
                total += entry.stat(follow_symlinks=False).st_size
                count += 1
            except OSError:
                continue
    return total, count, False


def _file_size(path):
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _relative(workspace, path):
    try:
        return str(Path(path).resolve().relative_to(Path(workspace.root).resolve()))
    except (ValueError, OSError):
        return str(path)


def read_settings(workspace):
    """读取页面设置；缺失或损坏时回退默认值（不抛异常）。"""
    path = Path(workspace.root) / SETTINGS_NAME
    settings = {"extra_dirs": [], "job_retention_days": DEFAULT_JOB_RETENTION_DAYS}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings
    if not isinstance(data, dict):
        return settings
    extra = data.get("extra_dirs")
    if isinstance(extra, list):
        settings["extra_dirs"] = [str(item) for item in extra if isinstance(item, str) and item.strip()]
    days = data.get("job_retention_days")
    if isinstance(days, bool):
        days = None
    if isinstance(days, int) and 0 <= days <= 3650:
        settings["job_retention_days"] = days
    return settings


def write_settings(workspace, *, extra_dirs=None, job_retention_days=None):
    """写回页面设置（原子替换）。"""
    current = read_settings(workspace)
    if extra_dirs is not None:
        cleaned = []
        root = str(Path(workspace.root).resolve())
        for item in extra_dirs:
            text = str(item).strip()
            if not text or str(Path(text).expanduser().resolve()) == root:
                continue
            if text not in cleaned:
                cleaned.append(text)
        current["extra_dirs"] = cleaned
    if job_retention_days is not None:
        days = int(job_retention_days)
        if not 0 <= days <= 3650:
            raise ValueError("保留天数应在 0～3650 之间")
        current["job_retention_days"] = days
    path = Path(workspace.root) / SETTINGS_NAME
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return current


def _catalog_rows(workspace):
    with workspace.connect() as conn:
        assets = [dict(row) for row in conn.execute(
            "SELECT id,name,path,sample_rate,sample_count,created_at,source,label "
            "FROM assets ORDER BY created_at DESC,id")]
        metadata = {row[0]: row[1] or 0 for row in conn.execute(
            "SELECT asset_id,length(metadata_json) FROM asset_metadata")}
        runs = [dict(row) for row in conn.execute(
            "SELECT id,kind,created_at,source_id,length(result_json) AS index_bytes "
            "FROM runs ORDER BY created_at DESC,id")]
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    return assets, metadata, runs, tables

def _assets_path_column(path):
    return str(Path(path).as_posix())


def _asset_rows(workspace, assets, metadata):
    rows = []
    known = set()
    for asset in assets:
        relative = asset["path"]
        known.add(_assets_path_column(relative))
        path = Path(workspace.root) / relative
        exists = path.is_file()
        rows.append({
            "id": asset["id"], "name": asset["name"], "path": relative,
            "bytes": _file_size(path) if exists else 0,
            "sample_rate_hz": asset["sample_rate"],
            "sample_count": int(asset["sample_count"]),
            "created_at": asset["created_at"], "source": asset["source"],
            "label": asset["label"] or "", "exists": exists,
            "metadata_bytes": int(metadata.get(asset["id"], 0) or 0),
        })
    return rows, known


def _run_rows(workspace, runs):
    rows = []
    known = set()
    for run in runs:
        known.add(run["id"])
        folder = Path(workspace.root) / "runs" / run["id"]
        exists = folder.is_dir()
        result_bytes = 0
        dir_bytes = 0
        if exists:
            dir_bytes, _, _ = _scan_tree(folder)
            result_bytes = _file_size(folder / "result.json")
            dir_bytes = max(dir_bytes, result_bytes)
        rows.append({
            "id": run["id"], "kind": run["kind"], "created_at": run["created_at"],
            "source_id": run["source_id"], "exists": exists,
            "bytes": int(dir_bytes),
            "sqlite_bytes": int(run["index_bytes"] or 0),
            "plots_bytes": int(max(dir_bytes - result_bytes, 0)),
        })
    return rows, known


def _scandir(path):
    """返回目录项列表；目录不存在或不可读时返回空列表（显式关闭句柄）。"""
    try:
        with os.scandir(path) as handle:
            return list(handle)
    except OSError:
        return []


def _job_rows(workspace, now):
    folder = Path(workspace.root) / "jobs"
    rows = []
    for entry in sorted(_scandir(folder), key=lambda item: item.name):
        if not entry.is_dir(follow_symlinks=False):
            continue
        status = {}
        status_path = Path(entry.path) / "status.json"
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if not isinstance(status, dict):
            status = {}
        size, _, _ = _scan_tree(entry.path)
        finished = status.get("finished") or status.get("started")
        age = _age_days(finished, now)
        if age is None:
            age = (now.timestamp() - entry.stat(follow_symlinks=False).st_mtime) / 86400.0
        rows.append({
            "id": entry.name, "action": str(status.get("action") or "未知"),
            "state": str(status.get("state") or "未知"),
            "started": str(status.get("started") or ""),
            "finished": str(status.get("finished") or ""),
            "error": str(status.get("error") or ""),
            "bytes": int(size), "age_days": round(float(age), 3),
            "has_status": status_path.is_file(),
        })
    return rows


def _export_rows(workspace):
    folder = Path(workspace.root) / "exports"
    rows = []
    for entry in sorted(_scandir(folder), key=lambda item: item.name):
        try:
            if not entry.is_file(follow_symlinks=False):
                continue
            stat = entry.stat(follow_symlinks=False)
            rows.append({"name": entry.name, "bytes": stat.st_size,
                         "modified": datetime.fromtimestamp(
                             stat.st_mtime, timezone.utc).isoformat()})
        except OSError:
            continue
    return rows


def _asset_files_on_disk(workspace):
    folder = Path(workspace.root) / "assets"
    found = {}
    for entry in _scandir(folder):
        try:
            if entry.is_file(follow_symlinks=False):
                found[entry.name] = entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return found


def _run_dirs_on_disk(workspace):
    folder = Path(workspace.root) / "runs"
    found = {}
    for entry in _scandir(folder):
        if entry.is_dir(follow_symlinks=False):
            size, _, _ = _scan_tree(entry.path)
            found[entry.name] = size
    return found


def _consistency(workspace, asset_paths, asset_rows, run_ids):
    asset_files = _asset_files_on_disk(workspace)
    run_dirs = _run_dirs_on_disk(workspace)

    orphan_files = []
    tmp_files = []
    for name, size in sorted(asset_files.items()):
        relative = f"assets/{name}"
        if name.endswith(".tmp"):
            tmp_files.append({"path": relative, "bytes": int(size), "reason": "原子写残留"})
        elif relative not in asset_paths:
            orphan_files.append({"path": relative, "bytes": int(size), "reason": "未入库文件"})

    missing_files = [{"path": item["path"], "bytes": 0, "reason": "入库但文件缺失"}
                     for item in asset_rows if not item["exists"]]

    orphan_run_dirs = [{"path": f"runs/{name}", "bytes": int(size), "reason": "未入库目录"}
                       for name, size in sorted(run_dirs.items()) if name not in run_ids]

    return {
        "orphan_files": orphan_files[:MAX_LIST_ROWS],
        "orphan_files_count": len(orphan_files),
        "missing_files": missing_files[:MAX_LIST_ROWS],
        "missing_files_count": len(missing_files),
        "orphan_run_dirs": orphan_run_dirs[:MAX_LIST_ROWS],
        "orphan_run_dirs_count": len(orphan_run_dirs),
        "tmp_files": tmp_files[:MAX_LIST_ROWS],
        "tmp_files_count": len(tmp_files),
    }


def _candidate_cleanup(workspace, job_rows, orphan_state, retention_days):
    """列出可删除项：过期的已结束任务 + 孤儿文件/目录（不含尚未结束的任务）。"""
    targets = []
    skipped_running = 0
    for job in job_rows:
        if job["state"] not in TERMINAL_STATES:
            skipped_running += 1
            continue
        if job["age_days"] < retention_days:
            continue
        targets.append({"path": f"jobs/{job['id']}", "bytes": job["bytes"], "kind": "job",
                        "reason": f"{job['action']} · {job['state']} · {job['age_days']:.1f} 天前"})
    for item in orphan_state["orphan_files"] + orphan_state["tmp_files"]:
        targets.append({"path": item["path"], "bytes": item["bytes"], "kind": "orphan_file",
                        "reason": item["reason"]})
    for item in orphan_state["orphan_run_dirs"]:
        targets.append({"path": item["path"], "bytes": item["bytes"], "kind": "orphan_run",
                        "reason": item["reason"]})
    targets.sort(key=lambda item: (-item["bytes"], item["path"]))
    return targets, skipped_running


def build_report(workspace, *, extra_dirs=None, job_retention_days=None):
    """扫描工作区并返回 JSON-safe 的盘点报告（只读，不写 ``runs``）。"""
    settings = read_settings(workspace)
    if extra_dirs is None:
        extra_dirs = settings["extra_dirs"]
    if job_retention_days is None:
        job_retention_days = settings["job_retention_days"]
    retention = int(job_retention_days)
    now = datetime.now(timezone.utc)
    root = Path(workspace.root)
    truncated = False
    warnings = []

    categories = []
    for key in ("assets", "runs", "jobs", "exports"):
        size, count, clipped = _scan_tree(root / key)
        truncated = truncated or clipped
        categories.append({"key": key, "label": CATEGORY_LABELS[key], "bytes": size,
                           "files": count, "managed": True})
    catalog_size = _file_size(root / "catalog.sqlite3")
    settings_size = _file_size(root / SETTINGS_NAME) + _file_size(root / LOG_NAME)
    categories.append({"key": "catalog", "label": CATEGORY_LABELS["catalog"],
                       "bytes": catalog_size, "files": 1 if catalog_size else 0, "managed": True})
    categories.append({"key": "settings", "label": CATEGORY_LABELS["settings"],
                       "bytes": settings_size, "files": 2 if settings_size else 0, "managed": True})

    assets, metadata, runs, tables = _catalog_rows(workspace)
    asset_rows, asset_paths = _asset_rows(workspace, assets, metadata)
    run_rows, run_ids = _run_rows(workspace, runs)
    job_rows = _job_rows(workspace, now)
    export_rows = _export_rows(workspace)
    orphan_state = _consistency(workspace, asset_paths, asset_rows, run_ids)

    orphan_runs = [{"path": f"runs/{row['id']}", "bytes": row["bytes"], "reason": "源资产已删除"}
                   for row in run_rows if row["source_id"] and row["source_id"] not in {a["id"] for a in assets}]

    extra_rows = []
    extra_total = 0
    extra_count = 0
    for item in extra_dirs:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if path == root.resolve():
            continue
        try:
            exists = path.is_dir()
        except OSError:
            exists = False
        if not exists:
            extra_rows.append({"path": str(path), "exists": False, "bytes": 0, "files": 0})
            continue
        size, count, clipped = _scan_tree(path)
        truncated = truncated or clipped
        extra_total += size
        extra_count += count
        extra_rows.append({"path": str(path), "exists": True, "bytes": size, "files": count})

    kinds = {}
    for row in run_rows:
        bucket = kinds.setdefault(row["kind"], {"kind": row["kind"],
                                                "label": RUN_KIND_LABELS.get(row["kind"], row["kind"]),
                                                "count": 0, "disk_bytes": 0, "sqlite_bytes": 0})
        bucket["count"] += 1
        bucket["disk_bytes"] += row["bytes"]
        bucket["sqlite_bytes"] += row["sqlite_bytes"]
    runs_by_kind = sorted(kinds.values(), key=lambda item: (-item["disk_bytes"], item["kind"]))

    targets, skipped_running = _candidate_cleanup(workspace, job_rows, orphan_state, retention)

    index_bytes = (sum(row["sqlite_bytes"] for row in run_rows)
                   + sum(row["metadata_bytes"] for row in asset_rows))

    if "asset_metadata" not in tables:
        warnings.append("索引缺少 asset_metadata 表，元数据占用未计入")
    if truncated:
        warnings.append(f"扫描文件数超过 {MAX_FILES:,}，部分目录占用为截断值")

    broken = [item["path"] for item in asset_rows if not item["exists"]]
    if broken:
        warnings.append(f"{len(broken)} 个资产的样本文件缺失（校验会失败）")
    if orphan_runs:
        warnings.append(f"{len(orphan_runs)} 条运行记录的源资产已删除")
    if extra_rows and any(not row["exists"] for row in extra_rows):
        warnings.append("部分额外目录不存在，已按 0 计入")

    totals = sum(item["bytes"] for item in categories) + extra_total
    files = sum(item["files"] for item in categories) + extra_count

    return {
        "kind": "storage_report",
        "version": VERSION,
        "root": str(root),
        "generated_at": utc_now(),
        "settings": {"extra_dirs": list(extra_dirs), "job_retention_days": retention},
        "totals": {"bytes": int(totals), "files": int(files),
                   "workspace_bytes": int(sum(item["bytes"] for item in categories)),
                   "extra_bytes": int(extra_total)},
        "categories": categories,
        "extra_dirs": extra_rows,
        "runs_by_kind": runs_by_kind,
        "tables": {
            "assets": asset_rows[:MAX_ASSET_ROWS],
            "assets_truncated": len(asset_rows) > MAX_ASSET_ROWS,
            "runs": run_rows[:MAX_RUN_ROWS],
            "runs_truncated": len(run_rows) > MAX_RUN_ROWS,
            "jobs": job_rows[:MAX_JOB_ROWS],
            "jobs_truncated": len(job_rows) > MAX_JOB_ROWS,
            "exports": export_rows[:MAX_EXPORT_ROWS],
            "exports_truncated": len(export_rows) > MAX_EXPORT_ROWS,
            "orphan_runs": orphan_runs[:MAX_LIST_ROWS],
        },
        "catalog": {"assets": len(assets), "runs": len(runs),
                    "index_bytes": int(index_bytes)},
        "consistency": {**orphan_state,
                        "orphan_runs_count": len(orphan_runs),
                        "orphan_runs": orphan_runs[:MAX_LIST_ROWS],
                        "assets": len(asset_rows), "runs": len(runs),
                        "truncated": truncated},
        "cleanup": {"retention_days": retention, "targets": targets,
                    "bytes": int(sum(item["bytes"] for item in targets)),
                    "skipped_running": skipped_running},
        "warnings": warnings,
    }


def preview_cleanup(workspace, *, extra_dirs=None, job_retention_days=None):
    """只返回清理候选清单（dry-run），不删除任何文件。"""
    report = build_report(workspace, extra_dirs=extra_dirs,
                          job_retention_days=job_retention_days)
    return {"version": VERSION, "kind": "storage_preview",
            "generated_at": report["generated_at"],
            "root": report["root"], **report["cleanup"]}


def _safe_target(workspace, relative):
    """把候选相对路径规范化；越界或不在白名单内返回 ``None``。"""
    root = Path(workspace.root).resolve()
    text = str(relative).replace("\\", "/").strip()
    if not text or text.startswith("/") or ".." in Path(text).parts:
        return None
    first = Path(text).parts[0] if Path(text).parts else ""
    if first not in CLEANABLE_ROOTS:
        return None
    path = (root / text)
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if resolved != root and not resolved.is_relative_to(root):
        return None
    if resolved == root:
        return None
    return resolved


def apply_cleanup(workspace, targets, *, extra_dirs=None, job_retention_days=None):
    """执行清理：先按当前磁盘与索引**重算**候选，只删命中的路径。

    任何不在最新候选清单里的请求项都会被跳过（防 TOCTOU 与越界）。
    """
    report = build_report(workspace, extra_dirs=extra_dirs,
                          job_retention_days=job_retention_days)
    allowed = {}
    for item in report["cleanup"]["targets"]:
        resolved = _safe_target(workspace, item["path"])
        if resolved is not None:
            allowed[str(resolved)] = item

    removed = []
    skipped = []
    errors = []
    requested = 0
    for raw in targets or ():
        if isinstance(raw, dict):
            raw = raw.get("path")
        if not raw:
            continue
        requested += 1
        resolved = _safe_target(workspace, raw)
        if resolved is None:
            skipped.append({"path": str(raw), "reason": "路径越界或不在可删除范围"})
            continue
        item = allowed.get(str(resolved))
        if item is None:
            skipped.append({"path": str(raw), "reason": "当前已不是清理候选（可能被引用或仍在运行）"})
            continue
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
            removed.append({"path": item["path"], "bytes": item["bytes"], "kind": item["kind"]})
        except OSError as exc:
            errors.append({"path": item["path"], "error": str(exc)})

    log_entry = {"at": utc_now(), "requested": requested, "removed": removed,
                 "skipped": skipped, "errors": errors,
                 "bytes": int(sum(item["bytes"] for item in removed))}
    try:
        with (Path(workspace.root) / LOG_NAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, ensure_ascii=False, allow_nan=False) + "\n")
    except OSError:
        pass

    return {"version": VERSION, "kind": "storage_cleanup",
            "removed": removed, "skipped": skipped, "errors": errors,
            "bytes": log_entry["bytes"], "requested": requested,
            "remaining": report["totals"]["bytes"]}
