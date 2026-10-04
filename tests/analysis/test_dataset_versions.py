"""数据版本构建测试（方案文档 §8）：分组划分、留出轴、负样本与排除规则。"""
import json

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.data.datasets import (bootstrap_labels_from_versions,
                                           build_dataset_version, read_manifest,
                                           verify_dataset_version)
from signal_analysis.services import execute
from signal_analysis.data import Workspace


def add_generated(workspace, mode="fm", seed=1, name=None, snr=15.0, signals=None):
    if signals is None:
        signals = [{"mode": mode, "offset": 0.0, "bandwidth": 50_000.0,
                    "power_dbfs": -6.0}]
    # 带内 SNR 需要调制信号做参考；纯噪声记录按绝对功率定标（负样本口径）。
    noise = ({"enabled": True, "snr_db": snr} if signals else
             {"enabled": True, "power_dbfs": -20.0})
    samples, summary = generate_iq(200_000.0, 0.05, signals, noise, seed)
    return workspace.add_samples(samples, 200_000.0, name or f"数据{seed}",
                                 f"generated:iq_{mode}_v1",
                                 metadata={"generation": summary})


def make_detection_set(workspace, count=3):
    collection = workspace.create_collection("检测集合")
    assets = [add_generated(workspace, seed=index, name=f"FM{index}")
              for index in range(count)]
    for asset in assets:
        workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "detection", name="检测")
    return collection, assets, task_set


def test_detection_dataset_version_flow(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection, assets, task_set = make_detection_set(workspace)
    boot = bootstrap_labels_from_versions(workspace, task_set["id"])
    assert boot == {"created": 3, "skipped": 0, "unmapped": {}}
    assert bootstrap_labels_from_versions(workspace, task_set["id"]) == \
        {"created": 0, "skipped": 3, "unmapped": {}}  # 幂等
    for asset in assets:
        workspace.set_asset_coverage(task_set["id"], asset["id"], "complete",
                                     source="manual")
    result = build_dataset_version(workspace, task_set["id"],
                                   fractions={"train": 0.6, "val": 0.2, "test": 0.2},
                                   seed=7, preprocessing={"nfft": 512, "image_size": 1024})
    version, card = result["version"], result["card"]
    assert version["status"] == "ready"
    assert version["sample_count"] == 3 and version["asset_count"] == 3
    assert sum(json.loads(version["split_json"]).values()) == 3
    assert json.loads(version["taxonomy_snapshot_json"])["classes"] == ["emitter"]
    assert json.loads(version["preprocessing_json"]) == {"nfft": 512, "image_size": 1024}
    rows = list(read_manifest(workspace, version))
    assert len(rows) == 3
    group_split = {}
    for row in rows:
        assert row["class_name"] == "emitter" and row["include"] is True
        assert row["negative"] is False
        assert row["sample_id"].startswith(row["asset_id"] + ":")
        assert row["extraction"]["sample_start"] == 0
        assert row["extraction"]["sample_end"] > 0
        # 同一 origin_group 只能出现在一个划分
        group_split.setdefault(row["origin_group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in group_split.values())
    assert verify_dataset_version(workspace, version) == 3
    # 清单被篡改：校验失败
    manifest = workspace.root / version["manifest_path"]
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8")
    with pytest.raises(ValueError, match="校验失败"):
        verify_dataset_version(workspace, version)
    # 重新构建产生新版本号，旧版本保持可用
    rebuilt = build_dataset_version(workspace, task_set["id"], seed=8)["version"]
    assert rebuilt["version_no"] == version["version_no"] + 1
    assert workspace.get_dataset_version(version["id"])["status"] == "ready"


def test_partial_coverage_and_unlabeled_targets_are_excluded(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection, assets, task_set = make_detection_set(workspace, count=2)
    bootstrap_labels_from_versions(workspace, task_set["id"])  # 只有前两个有标签
    extra = add_generated(workspace, seed=9, name="未标注")
    partial = add_generated(workspace, seed=10, name="部分覆盖")
    for asset in (extra, partial):
        workspace.add_collection_member(collection["id"], asset["id"])
    for asset in (*assets, extra, partial):
        workspace.set_asset_coverage(task_set["id"], asset["id"], "complete"
                                     if asset is not partial else "partial",
                                     source="manual")
    result = build_dataset_version(workspace, task_set["id"], seed=1)
    card = result["card"]
    assert result["version"]["sample_count"] == 2  # 只有两个已标注且完整覆盖的录制
    reasons = {item["name"]: item["reason"] for item in card["excluded"]}
    assert "覆盖度 partial" in reasons["部分覆盖"]
    assert "尚未标注" in reasons["未标注"]


def test_holdout_forces_groups_to_test_split(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection, assets, task_set = make_detection_set(workspace, count=3)
    bootstrap_labels_from_versions(workspace, task_set["id"])
    for asset in assets:
        workspace.set_asset_coverage(task_set["id"], asset["id"], "complete",
                                     source="manual")
    result = build_dataset_version(
        workspace, task_set["id"], seed=3,
        holdout=[{"axis": "snr_db", "range": [10.0, 20.0], "split": "test"}])
    card = result["card"]
    assert card["splits"]["counts"] == {"test": 3}
    assert card["splits"]["forced_groups"] == {"test": 3}
    assert any("没有样本" in text for text in card["splits"]["warnings"])
    rows = list(read_manifest(workspace, result["version"]))
    assert {row["split"] for row in rows} == {"test"}
    # 留出轴不命中：正常比例分配
    result = build_dataset_version(
        workspace, task_set["id"], seed=3,
        holdout=[{"axis": "snr_db", "range": [60.0, 70.0], "split": "test"}])
    counts = result["card"]["splits"]["counts"]
    assert sum(counts.values()) == 3 and counts != {"test": 3}


def test_negative_noise_recording_is_included(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("含噪声集")
    noise = add_generated(workspace, name="纯噪声", seed=5, signals=[])
    signal = add_generated(workspace, seed=6, name="含信号")
    for asset in (noise, signal):
        workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "detection", name="检测")
    bootstrap_labels_from_versions(workspace, task_set["id"])
    for asset in (noise, signal):
        workspace.set_asset_coverage(task_set["id"], asset["id"], "complete",
                                     source="manual")
    result = build_dataset_version(workspace, task_set["id"], seed=2)
    rows = list(read_manifest(workspace, result["version"]))
    negatives = [row for row in rows if row["negative"]]
    assert len(negatives) == 1
    assert negatives[0]["asset_id"] == noise["id"]
    assert negatives[0]["target_id"] is None
    assert negatives[0]["class_name"] is None


def test_amc_dataset_version_uses_target_windows(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("AMC 集合")
    first = add_generated(workspace, mode="fm", seed=1, name="FM")
    second = add_generated(workspace, mode="qpsk", seed=2, name="QPSK")
    hop = add_generated(workspace, mode="fh_rc", seed=3, name="跳频")
    for asset in (first, second, hop):
        workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "amc", name="AMC")
    boot = bootstrap_labels_from_versions(workspace, task_set["id"])
    assert boot["created"] == 2  # 跳频会话不参与 AMC
    target = workspace.list_targets(first["id"])[0]
    workspace.append_amc_label(task_set["id"], target["id"], source="manual",
                               class_state="known", class_name="fm",
                               window_start=0, window_end=1024)
    result = build_dataset_version(workspace, task_set["id"], seed=1)
    rows = list(read_manifest(workspace, result["version"]))
    assert len(rows) == 2
    classes = {row["class_name"] for row in rows}
    assert classes == {"fm", "qpsk"}
    fm_row = next(row for row in rows if row["class_name"] == "fm")
    assert fm_row["amc"]["class_state"] == "known"
    assert fm_row["extraction"]["window_start"] == 0
    assert fm_row["extraction"]["window_end"] == 1024


def test_build_without_rows_raises(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("空集")
    task_set = workspace.create_task_set(collection["id"], "detection", name="检测")
    with pytest.raises(ValueError, match="没有可纳入"):
        build_dataset_version(workspace, task_set["id"])


def test_service_dataset_and_recipe_actions(tmp_path):
    """服务层动作：引导标注 → 构建 → 校验，以及配方预览/保存。"""
    workspace = Workspace(tmp_path / "ws")
    collection, assets, task_set = make_detection_set(workspace, count=2)
    root = str(workspace.root)
    boot = execute({"workspace": root, "action": "dataset_bootstrap_labels",
                    "task_set_id": task_set["id"]})
    assert boot == {"created": 2, "skipped": 0, "unmapped": {}}
    for asset in assets:
        workspace.set_asset_coverage(task_set["id"], asset["id"], "complete",
                                     source="manual")
    built = execute({"workspace": root, "action": "dataset_build",
                     "task_set_id": task_set["id"], "seed": 4,
                     "preprocessing": {"nfft": 512, "image_size": 1024}})
    assert built["version"]["status"] == "ready"
    verified = execute({"workspace": root, "action": "dataset_verify",
                        "dataset_version_id": built["version"]["id"]})
    assert verified["samples"] == 2
    recipe = {"contract": "gen_recipe_v1", "engine": "project", "base_seed": 1,
              "count": 12, "signals": {"mode": {"balanced": ["fm", "qpsk"]}}}
    preview = execute({"workspace": root, "action": "recipe_preview",
                       "recipe": recipe, "samples": 6})
    assert preview["count"] == 6 and len(preview["preview"]) == 5
    saved = execute({"workspace": root, "action": "recipe_save", "name": "冒烟配方",
                     "recipe": recipe})
    assert saved["engine"] == "project" and len(workspace.list_recipes()) == 1
