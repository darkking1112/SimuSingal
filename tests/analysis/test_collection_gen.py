"""信号集合生成执行器、损伤预留接口与 TorchSig bundle 导入（方案 §6.3、§6.4、§6.5、§7.2）。"""
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from signal_analysis import collection_gen, impairments, torchsig_support
from signal_analysis.datasets import build_dataset_version
from signal_analysis.recipes import (check_generator_support, draw_record, preview_recipe,
                                     sample_seed, validate_recipe)
from signal_analysis.storage import Workspace
from signal_analysis.tasks import run_job

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "training"))


def make_recipe(count=6, **changes):
    recipe = {
        "contract": "gen_recipe_v1", "engine": "project", "base_seed": 11, "count": count,
        "record": {"sample_rate_hz": {"fixed": 200000.0}, "duration_s": {"fixed": 0.05}},
        "signals": {"count": {"choice": [1, 2]},
                    "mode": {"balanced": ["qpsk", "fh_rc", "fm"]},
                    "bandwidth_ratio": {"uniform": [0.05, 0.1]},
                    "snr_db": {"uniform": [10, 25]},
                    "power_dbfs": {"uniform": [-12, -6]}},
        "hopping": {"hop_rate_hz": {"uniform": [200, 400]}},
        "labels": {"detection": "per_hop_v1", "amc": True},
    }
    recipe.update(changes)
    return recipe


def generate(workspace, recipe, **request):
    request = {"workspace": str(workspace.root), "action": "generate_collection",
               "recipe": recipe, "collection_name": "批量集合", **request}
    return run_job(request, timeout=120)


def test_draw_record_is_deterministic_and_balanced():
    recipe = make_recipe()
    validate_recipe(recipe)
    assert draw_record(recipe, 3) == draw_record(recipe, 3)
    assert draw_record(recipe, 3) != draw_record(recipe, 4)
    # 同一条录制的多个信号槽位轮换样式：样式数足够时互不相同
    wide = make_recipe(signals={**recipe["signals"], "count": {"fixed": 3}})
    modes = [item["mode"] for item in draw_record(wide, 0)["signals"]]
    assert sorted(modes) == ["fh_rc", "fm", "qpsk"]
    assert sample_seed(11, 0) == sample_seed(11, 0, None) != sample_seed(11, 0, 0)
    assert preview_recipe(recipe, samples=6)["count"] == 6


def test_unsupported_parameters_fail_before_generation():
    recipe = make_recipe(impairments={"cfo_ratio": {"uniform": [0, 0.001]}})
    with pytest.raises(ValueError, match="频偏.*不可用"):
        check_generator_support(recipe)
    with pytest.raises(ValueError, match="符号率"):
        check_generator_support(make_recipe(signals={**make_recipe()["signals"],
                                                     "symbol_rate_baud": {"fixed": 1e4}}))
    with pytest.raises(ValueError, match="不支持的样式"):
        check_generator_support(make_recipe(signals={**make_recipe()["signals"],
                                                     "mode": {"fixed": "bpsk"}}))
    torchsig = make_recipe(engine="torchsig", signals={"count": {"fixed": 1}},
                           hopping={"hop_rate_hz": {"fixed": 10}},
                           labels={"detection": "session_v1"})
    with pytest.raises(ValueError, match="TorchSig 引擎不支持"):
        check_generator_support(torchsig)
    with pytest.raises(ValueError, match="未知的损伤项"):
        impairments.ensure_supported({"mystery": {"fixed": 1}})
    # 预留接口：没请求损伤时原样返回；请求了就拒绝，不静默跳过
    samples = np.ones(8, dtype=np.complex64)
    assert impairments.apply_impairments(samples, 1e5, {}, None, stage="record") is samples
    assert not any(item.implemented for item in impairments.IMPAIRMENTS.values())
    assert list(impairments.IMPAIRMENTS) == ["cfo", "phase_noise", "multipath", "iq_imbalance"]


def test_project_generation_writes_collection_targets_and_per_hop_labels(tmp_path):
    workspace = Workspace(tmp_path)
    result = generate(workspace, make_recipe(count=6))
    assert result["created"] == 6 and result["failed"] == 0 and result["stopped"] is None
    assert result["hops"] > 0  # 跳频样式默认同时写逐跳目标
    collection = workspace.get_collection(result["collection_id"])
    assert collection["source_kind"] == "generated"
    assert collection["recipe_id"] == result["recipe_id"]
    assert workspace.count_assets(collection_id=collection["id"]) == 6
    # 配方自动存到集合上，内容与提交的一致
    stored = json.loads(workspace.get_recipe(result["recipe_id"])["recipe_json"])
    assert stored["labels"]["detection"] == "per_hop_v1"

    task_sets = {item["task"]: item for item in workspace.list_task_sets(collection["id"])}
    assert task_sets["detection"]["label_semantics"] == "per_hop_v1"
    assert set(task_sets) == {"detection", "amc"}
    targets = workspace.collection_targets(collection["id"], task="detection")
    hop_targets = [item for item in targets if item["scope"] == "hop"]
    assert hop_targets
    labels = {row["target_id"] for row in workspace.current_labels(
        "detection", task_sets["detection"]["id"], include_all=True)}
    for target in targets:
        hopping_parent = target["scope"] != "hop" and target["current"]["is_hopping"]
        assert (target["id"] in labels) == (not hopping_parent)  # 跳频父会话不标，逐跳子目标标
    amc = workspace.current_labels("amc", task_sets["amc"]["id"], include_all=True)
    assert amc and {row["class_state"] for row in amc} == {"known"}
    assert {row["class_name"] for row in amc} <= {"qpsk", "fm"}

    # 生成的集合可直接构建数据版本（检测覆盖度已记为 complete）
    detection = build_dataset_version(workspace, task_sets["detection"]["id"], seed=1)
    assert detection["version"]["status"] == "ready"
    assert detection["version"]["sample_count"] >= 6
    amc_version = build_dataset_version(workspace, task_sets["amc"]["id"], seed=1)
    assert amc_version["version"]["sample_count"] == len(amc)


def test_session_labels_and_append_reuse_the_recipe(tmp_path):
    workspace = Workspace(tmp_path)
    recipe = make_recipe(count=3, labels={"detection": "session_v1", "amc": False})
    first = generate(workspace, recipe)
    second = generate(workspace, recipe)  # 追加到同名集合：复用配方与标注集
    assert first["collection_id"] == second["collection_id"]
    assert first["recipe_id"] == second["recipe_id"]
    assert len(workspace.list_recipes()) == 1
    assert workspace.count_assets(collection_id=first["collection_id"]) == 6
    task_sets = workspace.list_task_sets(first["collection_id"])
    assert [item["task"] for item in task_sets] == ["detection"]
    targets = workspace.collection_targets(first["collection_id"], task="detection")
    labels = workspace.current_labels("detection", task_sets[0]["id"], include_all=True)
    assert {item["id"] for item in targets if item["scope"] != "hop"} == \
        {row["target_id"] for row in labels}
    # 粒度冲突：已有会话级标注集的集合不能再按逐跳追加
    with pytest.raises(Exception, match="不一致"):
        generate(workspace, make_recipe(count=1))


def test_generation_is_reproducible_and_reports_failures(tmp_path):
    workspace = Workspace(tmp_path / "a")
    generate(workspace, make_recipe(count=2, labels={"amc": False}))
    other = Workspace(tmp_path / "b")
    generate(other, make_recipe(count=2, labels={"amc": False}))
    left = [workspace.load_samples(item)[1] for item in workspace.collection_asset_ids(
        workspace.list_collections()[0]["id"])]
    right = [other.load_samples(item)[1] for item in other.collection_asset_ids(
        other.list_collections()[0]["id"])]
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    # 带宽占比过大：放不下的录制被跳过并按原因计数，而不是悄悄改参数
    crowded = make_recipe(count=4, labels={"amc": False},
                          signals={**make_recipe()["signals"], "count": {"fixed": 3},
                                   "bandwidth_ratio": {"fixed": 0.45}})
    result = generate(workspace, crowded, collection_name="拥挤")
    assert result["created"] == 0 and result["failed"] == 4
    assert sum(item["count"] for item in result["failure_reasons"]) == 4


def test_cancel_flag_stops_between_records_and_progress_is_reported(tmp_path):
    workspace = Workspace(tmp_path)
    cancel = threading.Event()
    reports = []

    def progress(info):
        reports.append(info)
        if info.get("done", 0) >= 2:
            cancel.set()

    result = run_job({"workspace": str(workspace.root), "action": "generate_collection",
                      "recipe": make_recipe(count=400, labels={"amc": True}),
                      "collection_name": "可取消"},
                     timeout=120, cancel=cancel, progress=progress, cancel_grace=30.0)
    assert result["stopped"] == "cancelled"
    assert 2 <= result["created"] < 400  # 收尾后返回部分结果，已写数据保留
    assert reports and reports[-1]["total"] == 400
    collection = workspace.get_collection(result["collection_id"])
    assert workspace.count_assets(collection_id=collection["id"]) == result["created"]
    # 已生成的部分同样完成了标注
    assert workspace.list_task_sets(collection["id"], "amc")
    # 部分结果的分片已封存，资产可读
    asset_id = workspace.collection_asset_ids(collection["id"])[0]
    assert workspace.load_samples(asset_id)[1].size == 10000


def test_time_limit_stops_gracefully(tmp_path):
    workspace = Workspace(tmp_path)
    result = generate(workspace, make_recipe(count=50000, labels={"amc": False}),
                      max_seconds=1.0)
    assert result["stopped"] == "time_limit" and 0 < result["created"] < 50000


# ---------------------------------------------------------------------------
# TorchSig
# ---------------------------------------------------------------------------


def write_bundle(root, class_names=("QPSK", "8PSK"), records=3):
    from torchsig_bundle import component, record, write_bundle as write

    rate = 100000.0
    rng = np.random.default_rng(3)
    items = []
    for index in range(records):
        iq = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(np.complex64)
        comps = [component(class_name=class_names[0], center_freq=-20000.0, bandwidth=10000.0,
                           start_in_samples=2000, stop_in_samples=12000,
                           duration_in_samples=10000, snr_db=15.0)]
        if index == 1:
            comps.append(component(class_name=class_names[1], center_freq=20000.0,
                                   bandwidth=8000.0, start_in_samples=0,
                                   duration_in_samples=20000, snr_db=10.0))
        if index == 2:
            comps = []  # 纯噪声记录
        items.append((record(index=index, iq=iq, components=comps), iq))
    write(root, items, sample_rate_hz=rate)
    return root


def test_torchsig_engine_is_linux_only(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path)
    recipe = make_recipe(engine="torchsig", labels={"detection": "session_v1", "amc": True})
    recipe.pop("hopping")
    recipe["signals"] = {"count": {"choice": [0, 3]}, "snr_db": {"uniform": [0, 20]}}
    check_generator_support(recipe)
    monkeypatch.setattr(torchsig_support, "is_linux", lambda: False)
    with pytest.raises(ValueError, match="只能在 Linux"):
        torchsig_support.preflight(workspace, {})
    assert torchsig_support.probe(workspace, {})["ok"] is False
    # 配方压成的命令行参数：分布 → 区间，扰动档位可选
    arguments = torchsig_support.build_arguments(
        {**recipe, "torchsig": {"signal_generators": {"fixed": "qpsk,ook"},
                                "impairment_level": {"fixed": 1}}}, tmp_path / "bundle")
    text = " ".join(arguments)
    assert "--signals-range 0,3" in text and "--snr-range 0.0,20.0" in text
    assert "--signal-generators qpsk,ook" in text and "--impairment-level 1" in text
    assert "--impairment-level" not in " ".join(
        torchsig_support.build_arguments(recipe, tmp_path / "bundle"))


def test_torchsig_bundle_import_writes_targets_with_a09_mapping(tmp_path):
    bundle = write_bundle(tmp_path / "bundle")
    workspace = Workspace(tmp_path / "ws")
    result = run_job({"workspace": str(workspace.root), "action": "torchsig_import",
                      "bundle_path": str(bundle), "collection_name": "TorchSig 集合",
                      "mapping": {"qpsk": "qpsk"}}, timeout=120)
    assert result["created"] == 3 and result["sessions"] == 3 and result["hops"] == 0
    assert result["unmapped"] == {"8PSK": 1}
    collection = workspace.get_collection(result["collection_id"])
    assets = workspace.collection_asset_ids(collection["id"])
    first = workspace.list_targets(assets[0])
    assert len(first) == 1 and first[0]["scope"] == "whole_record"
    version = first[0]["current"]
    assert version["source"] == "external" and version["modulation"] == "QPSK"
    assert (version["sample_start"], version["sample_end"]) == (2000, 12000)
    assert version["f_low_hz"] == pytest.approx(-25000.0)
    assert version["f_high_hz"] == pytest.approx(-15000.0)
    assert version["center_hz"] == pytest.approx(-20000.0)
    assert version["bandwidth_hz"] == pytest.approx(10000.0)
    assert version["snr_db"] is not None
    second = workspace.list_targets(assets[1])
    assert [item["scope"] for item in second] == ["session", "session"]
    assert second[1]["current"]["modulation"] == "8PSK"  # 映射不到：保留原类名
    assert workspace.list_targets(assets[2]) == []  # 纯噪声记录没有目标
    task_sets = {item["task"]: item for item in workspace.list_task_sets(collection["id"])}
    assert task_sets["detection"]["label_semantics"] == "session_v1"
    amc = {row["class_state"] for row in workspace.current_labels(
        "amc", task_sets["amc"]["id"], include_all=True)}
    assert amc == {"known", "out_of_taxonomy"}
    # 纯噪声记录作为负样本进入数据版本
    version = build_dataset_version(workspace, task_sets["detection"]["id"], seed=1)
    assert version["version"]["sample_count"] >= 3


def test_torchsig_mapping_rejects_non_a09_targets(tmp_path):
    with pytest.raises(ValueError, match="A09"):
        torchsig_support.load_mapping({"qpsk": "bpsk"})
    bundle = write_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["bundle_version"] = "v0"
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="格式版本"):
        torchsig_support.read_bundle_manifest(bundle)
