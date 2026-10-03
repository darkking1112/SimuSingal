"""训练输入预检与集合装配（训练页直接以信号集合为数据来源）。

训练集集合的样本写 ``split=train``、验证集集合写 ``split=val``；不构建数据版本、不导出
训练数据集，也不按比例重新划分——划分由用户选择的集合决定（见拆分设计第 10 节）。本模块
被 GUI 预检与 worker 首阶段共用，因此「预检通过」与「实际训练输入」始终是同一份装配结果。

只依赖数据层与标准库，不引入 Qt。
"""
from ..data.datasets import build_rows

#: 配置文件里的任务名 → 标注集任务名。
TASK_KEYS = {"detection": "detection", "iq": "amc", "amc": "amc"}
TASK_LABELS = {"detection": "检测", "amc": "AMC"}


def task_label(task):
    return TASK_LABELS.get(task, task)


def task_key(task):
    """把页面任务名（``detection`` / ``iq``）映射到标注集任务名。"""
    if task not in TASK_KEYS:
        raise ValueError("训练任务只能是 detection 或 iq")
    return TASK_KEYS[task]


def collection_task_set(workspace, collection_id, task):
    """集合内该任务的标注集（取最后一个）；缺失时给出可执行提示。"""
    collection = workspace.get_collection(collection_id)
    sets = workspace.list_task_sets(collection_id, task)
    if not sets:
        raise ValueError(
            f"集合「{collection['name']}」没有{task_label(task)}标注集："
            "请到“IQ 信号生成 → 信号集合生成”勾选该任务的标注后重新生成，"
            "或先用该任务的训练页补标注")
    return sets[-1]


def collection_rows(workspace, collection_id, task, split):
    """装配某集合在给定划分下的训练行，返回 ``(task_set, rows, stats)``。"""
    task_set = collection_task_set(workspace, collection_id, task)
    rows, stats = build_rows(workspace, task_set["id"])
    for row in rows:
        row["split"] = split
    return task_set, rows, stats


def _counts(rows):
    negatives = sum(1 for row in rows if row.get("negative"))
    return {"samples": len(rows), "positives": len(rows) - negatives,
            "negatives": negatives, "assets": len({row["asset_id"] for row in rows})}


def inputs_plan(workspace, task, train_collection_id, val_collection_id):
    """校验两个集合可作为训练集／验证集，返回包含样本行的输入计划。

    抛 ``ValueError`` 时直接作为 ``config_status`` 提示文案：集合缺少标注集、没有可训练
    样本、标签粒度不一致、类别字典不一致或两集合同源。
    """
    task = task_key(task)
    if not train_collection_id or not val_collection_id:
        raise ValueError("请选择训练集和验证集信号集合")
    if train_collection_id == val_collection_id:
        raise ValueError("训练集与验证集不能是同一个集合")

    sections = {}
    for name, collection_id, split in (("train", train_collection_id, "train"),
                                       ("val", val_collection_id, "val")):
        task_set, rows, stats = collection_rows(workspace, collection_id, task, split)
        sections[name] = {
            "collection_id": collection_id,
            "collection_name": workspace.get_collection(collection_id)["name"],
            "task_set": task_set, "rows": rows, "stats": stats,
            "counts": _counts(rows)}

    train, val = sections["train"], sections["val"]
    if task == "detection":
        semantics = {train["task_set"]["label_semantics"] or "session_v1",
                     val["task_set"]["label_semantics"] or "session_v1"}
        if len(semantics) > 1:
            raise ValueError("训练集与验证集的检测标签粒度不同（会话级／逐跳级）："
                             "请在“信号集合生成”里统一粒度后重新生成")
    if train["task_set"]["taxonomy_id"] != val["task_set"]["taxonomy_id"]:
        raise ValueError("训练集与验证集的类别字典不同，无法一起训练：请统一后重试")
    for name, section in (("训练集", train), ("验证集", val)):
        if not section["rows"]:
            raise ValueError(_empty_reason(name, section, task))

    groups = ({row["origin_group_id"] for row in train["rows"]}
              & {row["origin_group_id"] for row in val["rows"]})
    if groups:
        raise ValueError(
            f"训练集与验证集有 {len(groups)} 组同源数据（同一次采集或同一生成场景），"
            "会造成数据泄漏：请调整集合成员，把同源信号放在同一个集合里")

    return {"task": task, "train": train, "val": val,
            "rows": [*train["rows"], *val["rows"]]}


def _empty_reason(name, section, task):
    excluded = section["stats"].get("excluded") or []
    if excluded:
        sample = "；".join(f"{row['name']}：{row['reason']}" for row in excluded[:2])
        return (f"{name}没有可训练样本：{len(excluded)} 条录制被排除（{sample}）。"
                "请先完成标注并确认覆盖度为 complete")
    return (f"{name}没有可训练样本：请在集合内完成"
            + ("检测标注" if task == "detection" else "AMC 标注") + "后重试")


def plan_summary(plan):
    """实验记录里的集合溯源信息（不写数据版本）。"""
    return {
        "train_collection": {key: plan["train"][key]
                             for key in ("collection_id", "collection_name")}
        | {"task_set_id": plan["train"]["task_set"]["id"],
           **plan["train"]["counts"]},
        "val_collection": {key: plan["val"][key]
                           for key in ("collection_id", "collection_name")}
        | {"task_set_id": plan["val"]["task_set"]["id"],
           **plan["val"]["counts"]},
    }
