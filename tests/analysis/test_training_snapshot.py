"""训练集／验证集集合 → 训练输入预检与快照（拆分设计第 10 节）。

训练页不再导出训练集：``inputs_plan`` 校验两个集合并装配样本行，``snapshot`` 把它们
实体化成训练脚本认识的目录（检测 ``AnnotationDataset``、AMC ``iq_dataset``）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from signal_analysis.data import Workspace
from signal_analysis.data.annotations import AnnotationDataset
from signal_analysis.services.training_inputs import inputs_plan, plan_summary
from signal_analysis.services.training_snapshot import snapshot

from test_collection_gen import generate, make_recipe

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def two_collections(workspace, count=6, **changes):
    """生成两个独立集合，分别作为训练集与验证集。

    ``workspace`` 由调用方构造（本模块只用到它的根目录与数据层接口）。
    """
    train = generate(workspace, make_recipe(count=count, **changes),
                     collection_name="训练集")
    val = generate(workspace, make_recipe(count=count, **changes),
                   collection_name="验证集")
    assert train["created"] == count and val["created"] == count
    return train["collection_id"], val["collection_id"]


def test_detection_snapshot_uses_train_and_val_collections(tmp_path):
    workspace = Workspace(tmp_path)
    train_id, val_id = two_collections(workspace, count=4)
    plan = inputs_plan(workspace, "detection", train_id, val_id)
    assert plan["train"]["counts"]["samples"] and plan["val"]["counts"]["samples"]
    report = snapshot(workspace, plan, destination=tmp_path / "out", image_size=256)

    dataset = AnnotationDataset(report["path"])
    splits = {record["split"] for record in dataset.records}
    assert splits == {"train", "val"}
    assert report["splits"]["train"] == plan["train"]["counts"]["assets"]
    assert report["splits"]["val"] == plan["val"]["counts"]["assets"]
    assert report["splits"]["test"] == 0
    assert report["boxes"] > 0
    assert dataset.card["contract"]["label_semantics"] == "per_hop_v1"
    assert dataset.card["contract"]["image_size"] == 256
    assert dataset.card["source"] == {
        "kind": "collection", "train_collection_id": train_id,
        "val_collection_id": val_id,
        "task_set_ids": [plan["train"]["task_set"]["id"], plan["val"]["task_set"]["id"]]}
    assert all(record["annotation_status"] == "reviewed" for record in dataset.records)

    # 训练脚本的数据集加载器可直接消费，且不登记数据版本
    from detectors.dataset import load_dataset

    card, records = load_dataset(report["path"])
    assert card["contract"]["label_semantics"] == "per_hop_v1"
    assert len(records) == report["samples"]
    assert workspace.list_dataset_versions() == []


def test_iq_snapshot_keeps_only_confirmed_classes(tmp_path):
    workspace = Workspace(tmp_path)
    train_id, val_id = two_collections(workspace, count=4)
    plan = inputs_plan(workspace, "amc", train_id, val_id)
    report = snapshot(workspace, plan, destination=tmp_path / "iq", samples=256)

    root = Path(report["path"])
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    assert card["contract"]["samples"] == 256 and card["contract"]["class_set"] == "a09"
    assert card["splits"]["strategy"] == "collection"
    assert card["splits"]["train"] and card["splits"]["val"] and not card["splits"]["test"]
    assert card["source"]["train_collection_id"] == train_id
    with np.load(root / "iq_dataset.npz") as store:
        assert store["waveforms"].shape[1:] == (2, 256)
        assert set(store["labels"].astype(str)) <= {"qpsk", "fm"}  # 跳频样式没有 A09 类别
        assert set(store["split"].astype(str)) == {"train", "val"}

    from train_iq import load_dataset

    assert len(load_dataset(root)[1]) == report["samples"]


def test_inputs_plan_reports_missing_sets_and_same_collection(tmp_path):
    workspace = Workspace(tmp_path)
    bare = workspace.create_collection("没有标注集")
    other = workspace.create_collection("另一个")
    with pytest.raises(ValueError, match="没有检测标注集"):
        inputs_plan(workspace, "detection", bare["id"], other["id"])
    with pytest.raises(ValueError, match="不能是同一个集合"):
        inputs_plan(workspace, "detection", bare["id"], bare["id"])
    with pytest.raises(ValueError, match="请选择训练集和验证集"):
        inputs_plan(workspace, "detection", None, other["id"])

    collection = generate(workspace, make_recipe(count=3, labels={"detection": None}),
                          collection_name="只有 AMC 标注")
    with pytest.raises(ValueError, match="没有检测标注集"):
        inputs_plan(workspace, "detection", collection["collection_id"], other["id"])


def test_inputs_plan_rejects_shared_origin_group(tmp_path):
    """同一次采集／生成场景的资产不能分散在训练集与验证集（防泄漏）。"""
    workspace = Workspace(tmp_path)
    train = generate(workspace, make_recipe(count=3), collection_name="集合A")
    twin = generate(workspace, make_recipe(count=3), collection_name="集合B")
    shared = workspace.collection_asset_ids(train["collection_id"])[0]
    group = f"group-{shared}"
    with workspace.connect() as conn:  # 让两个集合共享同源组，模拟同一次采集
        conn.execute("UPDATE assets SET origin_group_id=? WHERE id=?", (group, shared))
        for asset_id in workspace.collection_asset_ids(twin["collection_id"]):
            conn.execute("UPDATE assets SET origin_group_id=? WHERE id=?", (group, asset_id))
    with pytest.raises(ValueError, match="同源"):
        inputs_plan(workspace, "detection", train["collection_id"], twin["collection_id"])


def test_inputs_plan_rejects_mixed_detection_semantics(tmp_path):
    workspace = Workspace(tmp_path)
    per_hop = generate(workspace, make_recipe(count=3, labels={"detection": "per_hop_v1"}),
                       collection_name="逐跳")
    session = generate(workspace, make_recipe(count=3, labels={"detection": "session_v1"}),
                       collection_name="会话")
    with pytest.raises(ValueError, match="标签粒度不同"):
        inputs_plan(workspace, "detection", per_hop["collection_id"],
                    session["collection_id"])


def test_plan_summary_records_collections_without_dataset_version(tmp_path):
    workspace = Workspace(tmp_path)
    train_id, val_id = two_collections(workspace, count=3)
    plan = inputs_plan(workspace, "amc", train_id, val_id)
    summary = plan_summary(plan)
    assert summary["train_collection"]["collection_id"] == train_id
    assert summary["train_collection"]["collection_name"] == "训练集"
    assert summary["train_collection"]["samples"] > 0
    assert summary["val_collection"]["collection_id"] == val_id
    assert "dataset_version_id" not in json.dumps(summary)
