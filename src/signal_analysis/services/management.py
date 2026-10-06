"""存储维护、原生插件与历史迁移服务。"""

from ..integrations.plugins import call_demo_plugin
from ..storage.maintenance import apply_cleanup, build_report


def native(workspace, request):
    asset, data = workspace.load_samples(request["asset_id"])
    manifest, result = call_demo_plugin(request["manifest"], data)
    derived = workspace.add_samples(result, asset["sample_rate"],
                                    asset["name"] + " · 原生复制", f"parent:{asset['id']}")
    return workspace.save_run("native", {"plugin": manifest,
                                         "derived_asset_id": derived["id"]}, asset_id=asset["id"])


def storage_report(workspace, request):
    # 只读盘点：刻意不写入运行记录，否则报告本身会撑大存储。
    return build_report(workspace, job_retention_days=request.get("job_retention_days"))


def storage_cleanup(workspace, request):
    targets = request.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("未指定要清理的条目")
    if len(targets) > 5000:
        raise ValueError("单次清理条目过多，请分批执行")
    return apply_cleanup(workspace, targets, job_retention_days=request.get("job_retention_days"))
