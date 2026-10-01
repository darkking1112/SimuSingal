"""标签版本化、重新标注、覆盖度与完成率的存储层测试（方案文档 §3.4、§4）。"""
import json

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.storage import Workspace


def add_generated(workspace, mode="fm", seed=1, name=None):
    samples, summary = generate_iq(
        200_000.0, 0.05,
        [{"mode": mode, "offset": 0.0, "bandwidth": 50_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 15.0}, seed)
    return workspace.add_samples(samples, 200_000.0, name or f"标注{seed}",
                                 f"generated:iq_{mode}_v1",
                                 metadata={"generation": summary})


def make_task_sets(workspace):
    collection = workspace.create_collection("标注集合")
    assets = [add_generated(workspace, seed=index, name=f"资产{index}")
              for index in range(2)]
    hop = add_generated(workspace, mode="fh_rc", seed=9, name="跳频资产")
    for asset in (*assets, hop):
        workspace.add_collection_member(collection["id"], asset["id"])
    detection = workspace.create_task_set(collection["id"], "detection", name="检测标注")
    amc = workspace.create_task_set(collection["id"], "amc", name="AMC 标注")
    return collection, assets, hop, detection, amc


def test_default_taxonomies_and_semantics_validation(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    detection = workspace.default_taxonomy("detection")
    amc = workspace.default_taxonomy("amc")
    assert json.loads(detection["classes_json"]) == ["emitter"]
    assert json.loads(amc["classes_json"]) == ["fm", "ssb", "ask2", "qpsk", "qam16", "qam64"]
    # 默认字典重复获取不新建
    assert workspace.default_taxonomy("amc")["id"] == amc["id"]
    collection = workspace.create_collection("校验集")
    with pytest.raises(ValueError, match="粒度"):
        workspace.create_task_set(collection["id"], "detection",
                                  label_semantics="unknown_v9")
    with pytest.raises(ValueError, match="label_semantics"):
        workspace.create_task_set(collection["id"], "amc", label_semantics="session_v1")
    with pytest.raises(ValueError, match="名称必须唯一"):
        workspace.create_task_set(collection["id"], "detection", name="检测标注")
        workspace.create_task_set(collection["id"], "detection", name="检测标注")


def test_detection_labels_revision_supersede_and_include(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    _, assets, hop, detection, _ = make_task_sets(workspace)
    target = workspace.list_targets(assets[0]["id"])[0]
    first = workspace.append_detection_label(detection["id"], target["id"],
                                             source="generator")
    assert first["revision_no"] == 1 and first["supersedes_id"] is None
    assert first["class_name"] == "emitter"  # 单类别字典自动补名
    assert first["label_semantics"] == "session_v1"
    assert first["target_version_id"] == target["current"]["id"]
    second = workspace.append_detection_label(detection["id"], target["id"],
                                              source="manual", note="重新框选")
    assert second["revision_no"] == 2 and second["supersedes_id"] == first["id"]
    current = workspace.current_label("detection", detection["id"], target["id"])
    assert current["id"] == second["id"]
    revisions = workspace.label_revisions("detection", detection["id"], target["id"])
    assert [item["revision_no"] for item in revisions] == [1, 2]
    # 旧版本行保持原样（追加式，不修改）
    assert workspace.get_label("detection", first["id"]) == first
    # include=0：明确不作为检测目标，当前标签列表默认不含
    hop_target = workspace.list_targets(hop["id"])[1]
    workspace.append_detection_label(detection["id"], hop_target["id"], source="manual",
                                     include=False)
    labels = workspace.current_labels("detection", detection["id"])
    assert {item["target_id"] for item in labels} == {target["id"]}
    labels_all = workspace.current_labels("detection", detection["id"], include_all=True)
    assert {item["target_id"] for item in labels_all} == {target["id"], hop_target["id"]}
    # 粒度不一致必须报错；逐跳任务集与 hop 目标匹配
    per_hop = workspace.create_task_set(
        workspace.get_collection(detection["collection_id"])["id"],
        "detection", name="逐跳标注", label_semantics="per_hop_v1")
    with pytest.raises(ValueError, match="粒度"):
        workspace.append_detection_label(detection["id"], hop_target["id"],
                                         source="manual", label_semantics="per_hop_v1")
    hop_label = workspace.append_detection_label(per_hop["id"], hop_target["id"],
                                                 source="manual")
    assert hop_label["label_semantics"] == "per_hop_v1"


def test_amc_label_states_and_validation(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    _, assets, hop, detection, amc = make_task_sets(workspace)
    target = workspace.list_targets(assets[0]["id"])[0]
    label = workspace.append_amc_label(amc["id"], target["id"], source="generator",
                                       class_state="known", class_name="fm")
    assert label["class_state"] == "known"
    assert workspace.current_label("amc", amc["id"], target["id"])["id"] == label["id"]
    # 重新标注：追加新版本
    again = workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                       class_state="known", class_name="qpsk")
    assert again["revision_no"] == 2
    with pytest.raises(ValueError, match="字典内"):
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="known", class_name="fh_rc")
    with pytest.raises(ValueError, match="原始类名"):
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="out_of_taxonomy")
    with pytest.raises(ValueError, match="应记为 known"):
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="out_of_taxonomy", class_name="fm")
    with pytest.raises(ValueError, match="unknown"):
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="unknown", class_name="fm")
    out = workspace.append_amc_label(amc["id"], hop["id"] and
                                     workspace.list_targets(hop["id"])[0]["id"],
                                     source="manual", class_state="out_of_taxonomy",
                                     class_name="fh_rc")
    assert out["class_name"] == "fh_rc"
    # 检测集不接受 AMC 写入，反之亦然
    with pytest.raises(ValueError, match="不是 AMC"):
        workspace.append_amc_label(detection["id"], target["id"], source="manual",
                                   class_state="known", class_name="fm")
    # 提取窗口边界校验
    with pytest.raises(ValueError, match="提取窗口"):
        workspace.append_amc_label(amc["id"], target["id"], source="manual",
                                   class_state="known", class_name="fm",
                                   window_start=100, window_end=100)
    windowed = workspace.append_amc_label(
        amc["id"], assets[1]["id"] and workspace.list_targets(assets[1]["id"])[0]["id"],
        source="manual", class_state="unknown", window_start=0, window_end=1024)
    assert windowed["window_start"] == 0 and windowed["class_state"] == "unknown"


def test_adopt_algorithm_label(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    _, assets, _, detection, amc = make_task_sets(workspace)
    target = workspace.list_targets(assets[0]["id"])[0]
    adopted = workspace.adopt_label("amc", amc["id"], target["id"],
                                    source_run_id="run123", class_state="known",
                                    class_name="fm")
    assert adopted["source"] == "algorithm"
    assert "run123" in adopted["note"]
    detection_adopted = workspace.adopt_label("detection", detection["id"], target["id"],
                                              source_run_id="run456")
    assert detection_adopted["source"] == "algorithm"
    with pytest.raises(ValueError, match="来源"):
        workspace.adopt_label("amc", amc["id"], target["id"], source="manual",
                              class_state="known", class_name="fm")


def test_coverage_and_progress(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection, assets, hop, detection, amc = make_task_sets(workspace)
    workspace.set_asset_coverage(detection["id"], assets[0]["id"], "complete",
                                 source="manual")
    workspace.set_asset_coverage(detection["id"], assets[1]["id"], "partial",
                                 source="manual")
    workspace.set_asset_coverage(detection["id"], assets[1]["id"], "complete",
                                 negative_kind=None, source="manual")
    # 追加式：同一 (task_set, asset) 取最新一条
    latest = workspace.get_asset_coverage(detection["id"], assets[1]["id"])
    assert latest["coverage"] == "complete" and latest["revision_no"] == 2
    assert workspace.get_asset_coverage(detection["id"], hop["id"]) is None
    with pytest.raises(ValueError, match="negative_kind"):
        workspace.set_asset_coverage(detection["id"], hop["id"], "partial",
                                     negative_kind="noise_only", source="manual")
    with pytest.raises(ValueError, match="覆盖状态"):
        workspace.set_asset_coverage(detection["id"], hop["id"], "maybe", source="manual")
    # 完成率：待标注 / 已标注 / 其中重新标注过；fm 两个会话 + 跳频会话与逐跳
    progress = workspace.task_set_progress(detection["id"])
    hop_count = sum(1 for item in workspace.collection_targets(detection["collection_id"])
                    if item["scope"] == "hop")
    assert progress["targets"] == 2 + 1 + hop_count
    assert progress["labeled"] == 0 and progress["pending"] == progress["targets"]
    target = workspace.list_targets(assets[0]["id"])[0]
    workspace.append_detection_label(detection["id"], target["id"], source="manual")
    workspace.append_detection_label(detection["id"], target["id"], source="manual")
    progress = workspace.task_set_progress(detection["id"])
    assert progress["labeled"] == 1 and progress["reannotated"] == 1
    assert progress["pending"] == progress["targets"] - 1
    assert progress["coverage"]["complete"] == 2
    assert progress["coverage"]["unmarked"] == 1
    # AMC 任务进度互不影响：跳频会话不参与 AMC，只有两个 fm 会话
    amc_progress = workspace.task_set_progress(amc["id"])
    assert amc_progress["targets"] == 2 and amc_progress["labeled"] == 0
