"""后台任务运行器：按动作选择超时与取消宽限期。"""

from ..tasks import run_job


def _run_task(request, cancel=None, progress=None):
    # 16M 样本的生成、导出与演示（同一点数上限）可能明显超过默认 30 s 子进程超时；
    # 数据盘点/清理要遍历整棵工作区目录树，同样放宽。集合生成是长任务：取消时给
    # worker 优雅收尾的时间（封存已写数据并返回部分结果）。
    action = request.get("action")
    if action in ("generate", "demo"):
        timeout = 600.0
    elif action in ("generate_collection", "torchsig_import", "export_training_data",
                    "torchsig_probe"):
        timeout = 3600.0
    elif action in ("storage_report", "storage_cleanup", "migrate_legacy"):
        timeout = 300.0
    else:
        timeout = 30.0
    grace = 120.0 if action in ("generate_collection", "torchsig_import",
                                "export_training_data") else 0.0
    return run_job(request, timeout=timeout, cancel=cancel, progress=progress,
                   cancel_grace=grace)
