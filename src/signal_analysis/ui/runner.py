"""后台任务运行器：按动作选择超时与取消宽限期；批量导入按块编排。"""

from ..tasks import run_job

#: 批量导入的单块大小：与 ``services.imports.import_files`` 的单批上限一致。
IMPORT_CHUNK = 500


def _budgets(action):
    """按动作给出 ``(timeout, cancel_grace)``。

    - 16M 样本的生成可能明显超过默认 30 s 子进程超时；
    - 数据盘点/清理要遍历整棵工作区目录树，同样放宽；
    - 集合生成/批量导入是长任务：取消给 120 s 优雅收尾（封存已写数据并返回部分
      结果），而不是直接终止子进程。
    """
    if action == "generate":
        return 600.0, 0.0
    if action in ("generate_collection", "torchsig_import",
                  "torchsig_probe", "import_files", "import_inspect"):
        return 3600.0, 120.0
    if action in ("storage_report", "storage_cleanup"):
        return 300.0, 0.0
    return 30.0, 0.0


def _run_task(request, cancel=None, progress=None):
    action = request.get("action")
    timeout, grace = _budgets(action)
    if action == "import_files":
        return _run_import_chunks(request, timeout, grace, cancel, progress)
    return run_job(request, timeout=timeout, cancel=cancel, progress=progress,
                   cancel_grace=grace)


def _run_import_chunks(request, timeout, grace, cancel, progress):
    """files 超过单批上限时按块顺序执行：一个任务、一个取消事件、合并结果。

    每块是一个独立 worker 子进程与独立分片；进度回调统一换算成整批口径
    （``done`` 累计、``total`` 为总数，消息带块号）。取消发生在块边界时直接
    停止后续块；发生在块内时由 worker 在文件边界收尾并回报 ``cancelled``。
    """
    files = request.get("files") or []
    if len(files) <= IMPORT_CHUNK:
        return run_job(request, timeout=timeout, cancel=cancel, progress=progress,
                       cancel_grace=grace)
    total = len(files)
    blocks = (total + IMPORT_CHUNK - 1) // IMPORT_CHUNK
    merged = {"kind": "import_files", "results": [], "created": 0, "failed": 0,
              "collection_id": None, "collection_name": None, "shard_id": None,
              "shard_ids": [], "targets_total": 0, "initial_labels": 0,
              "cancelled": False, "unprocessed": 0}
    done_files = 0
    for block, start in enumerate(range(0, total, IMPORT_CHUNK), start=1):
        if cancel is not None and cancel.is_set():
            merged["cancelled"] = True
            break
        chunk = list(files[start:start + IMPORT_CHUNK])
        sub = {key: value for key, value in request.items() if key != "files"}
        sub["files"] = chunk
        if sub.get("batch_shard") and not sub.get("shard_name"):
            sub["shard_name"] = f"导入批次 {block}/{blocks}（{len(chunk)} 个文件）"

        def chunk_progress(info, *, start=start, block=block):
            if progress is None or not isinstance(info, dict):
                return
            report = dict(info)
            report["done"] = start + int(info.get("done") or 0)
            report["total"] = total
            message = info.get("message")
            report["message"] = (f"第 {block}/{blocks} 块 · {message}" if message
                                 else f"第 {block}/{blocks} 块")
            progress(report)

        result = run_job(sub, timeout=timeout, cancel=cancel, progress=chunk_progress,
                         cancel_grace=grace)
        merged["results"].extend(result.get("results") or [])
        merged["created"] += int(result.get("created") or 0)
        merged["failed"] += int(result.get("failed") or 0)
        merged["targets_total"] += int(result.get("targets_total") or 0)
        merged["initial_labels"] += int(result.get("initial_labels") or 0)
        if result.get("collection_id") and not merged["collection_id"]:
            merged["collection_id"] = result["collection_id"]
            merged["collection_name"] = result.get("collection_name")
        if result.get("shard_id"):
            merged["shard_ids"].append(result["shard_id"])
        done_files += len(result.get("results") or [])
        if result.get("cancelled"):
            merged["cancelled"] = True
            break
    if done_files == 0 and cancel is not None and cancel.is_set():
        raise RuntimeError("任务已取消")
    if merged["shard_ids"]:
        merged["shard_id"] = merged["shard_ids"][0]
    merged["unprocessed"] = total - done_files
    return merged
