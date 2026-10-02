"""采纳为参数标注（方案 §4.2）：检测/逐跳/AMC 结论 → 目标参考参数 + 标签。"""
import pytest

from signal_analysis.data.adoption import (adopt_classification, adopt_detections,
                                           adopt_hops)
from signal_analysis.core_api import generate_iq
from signal_analysis.services import execute
from signal_analysis.data import Workspace


def add_generated(workspace, mode="fm", seed=1, name=None, snr=15.0):
    samples, summary = generate_iq(
        200_000.0, 0.08,
        [{"mode": mode, "offset": 0.0, "bandwidth": 50_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": snr}, seed)
    return workspace.add_samples(samples, 200_000.0, name or f"采纳{seed}",
                                 f"generated:iq_{mode}_v1",
                                 metadata={"generation": summary})


def test_adopt_detections_updates_and_creates_targets(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = add_generated(workspace, seed=1)
    target = workspace.list_targets(asset["id"])[0]
    detections = [
        # 与现有目标重合（中心偏移 1 kHz，远小于半个带宽）：应追加版本
        {"id": 1, "center_hz": 1_000.0, "bandwidth_hz": 50_000.0,
         "f_low_hz": -24_000.0, "f_high_hz": 26_000.0,
         "t_start_s": 0.0, "t_end_s": 0.08, "snr_db": 14.0, "power_dbfs": -7.0},
        # 远离频带：应新建目标
        {"id": 2, "center_hz": -70_000.0, "bandwidth_hz": 20_000.0,
         "f_low_hz": -80_000.0, "f_high_hz": -60_000.0,
         "t_start_s": 0.0, "t_end_s": 0.04, "snr_db": 9.0, "power_dbfs": -12.0},
    ]
    result = adopt_detections(workspace, asset["id"], detections, "run-1")
    assert result["updated_targets"] == 1 and result["created_targets"] == 1
    # 现目标：追加 source=algorithm 版本，旧版本保留
    versions = workspace.list_target_versions(target["id"])
    assert [item["version_no"] for item in versions] == [1, 2]
    assert versions[1]["source"] == "algorithm"
    assert versions[1]["center_hz"] == pytest.approx(1_000.0)
    assert versions[0]["source"] == "generator"
    assert "run-1" in versions[1]["note"]
    # 继承样式与名义值（算法测不到，不覆盖）
    assert versions[1]["waveform_mode"] == "fm"
    assert versions[1]["modulation"] == "FM"
    assert versions[1]["nominal_center_hz"] is not None
    # 新目标：会话键递增、可检测、时间夹在资产范围内
    fresh = [item for item in workspace.list_targets(asset["id"])
             if item["id"] != target["id"]]
    assert len(fresh) == 1 and fresh[0]["target_key"] == "s1"
    assert fresh[0]["current"]["source"] == "algorithm"
    assert fresh[0]["current"]["sample_start"] == 0
    assert fresh[0]["current"]["sample_end"] == 0.04 * 200_000


def test_adopt_detections_writes_labels_when_task_set_exists(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("采纳集合")
    asset = add_generated(workspace, seed=2)
    workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "detection")
    result = execute({"workspace": str(workspace.root),
                      "action": "detect", "asset_id": asset["id"]})
    adopted = execute({"workspace": str(workspace.root),
                       "action": "adopt_result", "run_id": result["run_id"]})
    assert adopted["kind"] == "adopt_result"
    assert adopted["labels"] >= 1 and adopted["task_sets"] == ["检测标注"]
    labels = workspace.current_labels("detection", task_set["id"])
    assert labels and all(item["source"] == "algorithm" for item in labels)
    version = workspace.get_target_version(labels[0]["target_version_id"])
    assert version["source"] == "algorithm"
    # 重复采纳：追加新版本与新标签（追加式，不覆盖历史）
    execute({"workspace": str(workspace.root), "action": "adopt_result",
             "run_id": result["run_id"]})
    labels = workspace.current_labels("detection", task_set["id"])
    assert labels[0]["revision_no"] == 2
    assert len(workspace.label_revisions("detection", task_set["id"],
                                         labels[0]["target_id"])) == 2


def test_adopt_hops_creates_session_and_hop_versions(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("逐跳集合")
    asset = add_generated(workspace, mode="fh_rc", seed=3)
    workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "detection",
                                         name="逐跳标注", label_semantics="per_hop_v1")
    run = execute({"workspace": str(workspace.root), "action": "detect_hops",
                   "asset_id": asset["id"]})
    adopted = execute({"workspace": str(workspace.root), "action": "adopt_result",
                       "run_id": run["run_id"]})
    hops = [item for item in workspace.list_targets(asset["id"])
            if item["scope"] == "hop"]
    adopted_hops = [item for item in hops if item["current"]["source"] == "algorithm"]
    # 本次采纳写入的逐跳目标（探测器可能合并同频道相邻跳），逐个都应带上逐跳标签
    assert adopted_hops and adopted["labels"] == len(adopted_hops)
    labels = workspace.current_labels("detection", task_set["id"])
    assert {item["target_id"] for item in labels} == {item["id"] for item in adopted_hops}
    # 会话目标同样拿到算法版本（逐跳结果的会话范围）
    session = [item for item in workspace.list_targets(asset["id"])
               if item["scope"] == "session"][0]
    assert session["current"]["source"] == "algorithm"


def test_adopt_classification_updates_modulation_and_labels(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    collection = workspace.create_collection("分类集合")
    asset = add_generated(workspace, mode="qpsk", seed=4)
    workspace.add_collection_member(collection["id"], asset["id"])
    task_set = workspace.create_task_set(collection["id"], "amc")
    target = workspace.list_targets(asset["id"])[0]
    assert target["current"]["modulation"] == "QPSK"
    result = adopt_classification(workspace, asset["id"],
                                  {"label": "fm", "confidence": 0.9}, "run-9",
                                  band={"center_hz": 0.0, "bandwidth_hz": 50_000.0})
    assert result["modulation"] == "FM" and result["updated_targets"] == 1
    current = workspace.current_target_version(target["id"])
    assert current["modulation"] == "FM" and current["source"] == "algorithm"
    labels = workspace.current_labels("amc", task_set["id"])
    assert labels[0]["class_state"] == "known" and labels[0]["class_name"] == "fm"
    # 字典外的类别保留原始类名，不猜测
    result = adopt_classification(workspace, asset["id"],
                                  {"label": "wbfm"}, "run-10",
                                  band={"center_hz": 0.0, "bandwidth_hz": 50_000.0})
    assert result["modulation"] == "WBFM"
    labels = workspace.current_labels("amc", task_set["id"])
    assert labels[0]["class_state"] == "out_of_taxonomy"
    assert labels[0]["class_name"] == "wbfm"


def test_adopt_classification_service_end_to_end(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = add_generated(workspace, mode="qam64", seed=5)
    run = execute({"workspace": str(workspace.root), "action": "amc_classify",
                   "asset_id": asset["id"],
                   "config": {"offset_hz": 0.0, "bandwidth_hz": 50_000.0}})
    adopted = execute({"workspace": str(workspace.root), "action": "adopt_result",
                       "run_id": run["run_id"]})
    assert adopted["source_kind"] == "amc_classify"
    current = workspace.current_target_version(
        workspace.list_targets(asset["id"])[0]["id"])
    assert current["modulation"] == "64QAM"
    # 不支持采纳的结果类型必须报错说明，而不是静默
    analysis = execute({"workspace": str(workspace.root), "action": "analyze",
                        "asset_id": asset["id"]})
    with pytest.raises(ValueError, match="不支持"):
        execute({"workspace": str(workspace.root), "action": "adopt_result",
                 "run_id": analysis["run_id"]})
