"""数据集与标注生命周期服务：构建/校验/初始标注/采纳运行结果/训练数据导出。"""


def dataset_build(workspace, request):
    from ..data.datasets import build_dataset_version

    return build_dataset_version(
        workspace, request["task_set_id"], fractions=request.get("fractions"),
        seed=int(request.get("seed", 0)), holdout=request.get("holdout"),
        preprocessing=request.get("preprocessing"))


def dataset_verify(workspace, request):
    from ..data.datasets import verify_dataset_version

    version = workspace.get_dataset_version(request["dataset_version_id"])
    if version["status"] != "ready":
        raise ValueError(f"数据版本状态为 {version['status']}，尚不可用于训练")
    return {"kind": "dataset_verify", "dataset_version_id": version["id"],
            "samples": verify_dataset_version(workspace, version),
            "task_set_id": version["task_set_id"]}


def dataset_bootstrap_labels(workspace, request):
    from ..data.datasets import bootstrap_labels_from_versions

    return bootstrap_labels_from_versions(
        workspace, request["task_set_id"], asset_ids=request.get("asset_ids"),
        source=request.get("source", "generator"))


def adopt_result(workspace, request):
    from ..data.adoption import adopt_run_result

    return adopt_run_result(workspace, request["run_id"])


def export_training_data(workspace, request):
    from .training_export import export_training_data as run_export_training_data

    return run_export_training_data(workspace, request)
