"""旧数据迁移测试（方案文档 §9）：旧标注目录、旧实验记录与分片盘点。"""
import json

import numpy as np
import pytest

from signal_analysis.data.datasets import verify_dataset_version
from signal_analysis.storage.maintenance import (build_report, register_legacy_dataset,
                                                 register_legacy_experiments,
                                                 scan_legacy_datasets)
from signal_analysis.services import execute
from signal_analysis.data import Workspace


def make_legacy_dataset(directory, source_asset_ids=()):
    """构造旧版标注数据集目录（dataset.json + samples.jsonl + images/）。"""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "images").mkdir(exist_ok=True)
    card = {
        "sample_count": len(source_asset_ids) + 1,
        "splits": {"train": 1, "val": 1},
        "contract": {"input_contract": "tf_image_v1",
                     "layout": "time_frequency_grayscale_v1",
                     "output_layout": "normalized_boxes_v1", "image_size": 128,
                     "channels": 1, "spectrogram_nfft": 256,
                     "dynamic_range_db": 60.0, "labels": ["emitter"],
                     "label_semantics": "session_v1"},
    }
    (directory / "dataset.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    records = []
    for index, asset_id in enumerate(source_asset_ids):
        records.append({"image": f"images/{index:06d}.npy",
                        "source_asset_id": asset_id, "split": "train",
                        "boxes": [[0.5, 0.5, 0.2, 0.2, 1.0, 0]]})
        np.save(directory / "images" / f"{index:06d}.npy", np.zeros((4, 4)))
    records.append({"image": "images/999999.npy", "split": "val",
                    "boxes": []})
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8")


def test_register_legacy_dataset_is_idempotent_and_links_assets(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    asset = workspace.add_samples(np.ones(64, dtype=np.complex64), 48_000.0,
                                  "旧资产", "imported:x")
    dataset = tmp_path / "old_dataset"
    make_legacy_dataset(dataset, source_asset_ids=[asset["id"], "不存在的资产"])
    result = register_legacy_dataset(workspace, dataset)
    assert result["already"] is False
    assert result["linked_assets"] == 1 and result["missing_assets"] == 1
    collection = result["collection"]
    assert collection["source_kind"] == "legacy"
    assert collection["description"] == str(dataset.resolve())
    assert workspace.count_assets(collection_id=collection["id"]) == 1
    task_set = result["task_set"]
    assert task_set["task"] == "detection"
    assert task_set["label_semantics"] == "session_v1"
    version = result["version"]
    assert version["status"] == "ready"
    assert version["sample_count"] == 3
    assert json.loads(version["card_json"])["raw_iq"] is False
    preprocessing = json.loads(version["preprocessing_json"])
    assert preprocessing["legacy"] is True and preprocessing["raw_iq"] is False
    assert verify_dataset_version(workspace, version) == 3
    # 重复登记：沿用同一集合、标注集与数据版本，不再新增行
    again = register_legacy_dataset(workspace, dataset)
    assert again["already"] is True and again["version"]["id"] == version["id"]
    assert len(workspace.list_dataset_versions(task_set["id"])) == 1
    assert len(workspace.list_collections(include_archived=True)) == 1


def test_scan_legacy_datasets_registers_and_skips(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    make_legacy_dataset(workspace.root / "datasets" / "aaa")
    make_legacy_dataset(workspace.root / "datasets" / "bbb")
    (workspace.root / "datasets" / "not_a_dataset").mkdir()
    result = scan_legacy_datasets(workspace)
    assert len(result["registered"]) == 2 and result["skipped"] == []
    assert result["errors"] == []
    again = scan_legacy_datasets(workspace)
    assert again["registered"] == [] and len(again["skipped"]) == 2
    with pytest.raises(ValueError):
        register_legacy_dataset(workspace, workspace.root / "datasets" / "not_a_dataset")


def test_register_legacy_experiments_indexes_and_links(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    dataset = tmp_path / "old_dataset"
    make_legacy_dataset(dataset)
    version = register_legacy_dataset(workspace, dataset)["version"]
    runs = workspace.root / "training" / "runs" / "20260101T000000-abcd"
    runs.mkdir(parents=True)
    record = {"id": "20260101T000000-abcd", "status": "success",
              "stage": "done", "started": "2026-01-01T00:00:00+00:00",
              "finished": "2026-01-01T00:10:00+00:00",
              "config": {"task": "detection", "arch": "yolo26s",
                         "data": str(dataset)}}
    (runs / "experiment.json").write_text(json.dumps(record, ensure_ascii=False),
                                          encoding="utf-8")
    broken = workspace.root / "training" / "runs" / "broken"
    broken.mkdir(parents=True)
    (broken / "experiment.json").write_text("{ not json", encoding="utf-8")
    result = register_legacy_experiments(workspace)
    assert len(result["registered"]) == 1 and result["skipped"] == 1
    row = workspace.get_experiment("20260101T000000-abcd")
    assert row["status"] == "success"
    assert row["dataset_version_id"] == version["id"]
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"
    assert row["finished_at"] == "2026-01-01T00:10:00+00:00"
    assert json.loads(row["config_json"])["arch"] == "yolo26s"
    # 幂等：重复扫描不重复登记
    again = register_legacy_experiments(workspace)
    assert again["registered"] == [] and again["skipped"] == 2
    # 无数据目录可关联时 dataset_version_id 为 None，其余照常登记
    orphan = workspace.root / "training" / "runs" / "20260102T000000-ffff"
    orphan.mkdir(parents=True)
    (orphan / "experiment.json").write_text(json.dumps(
        {"id": "20260102T000000-ffff", "status": "failed",
         "config": {"task": "iq"}}), encoding="utf-8")
    result = register_legacy_experiments(workspace)
    row = workspace.get_experiment("20260102T000000-ffff")
    assert result["registered"][0]["dataset_version_id"] is None
    assert row["status"] == "failed"


def test_service_migrate_legacy_is_wired(tmp_path):
    """服务层动作 ``migrate_legacy``：扫描并登记工作区 datasets/ 下的旧数据集。"""
    workspace = Workspace(tmp_path / "ws")
    make_legacy_dataset(workspace.root / "datasets" / "x")
    result = execute({"workspace": str(workspace.root), "action": "migrate_legacy"})
    assert result["kind"] == "legacy_migration"
    assert len(result["datasets"]["registered"]) == 1
    assert result["datasets"]["registered"][0]["samples"] == 1


def test_storage_report_counts_shard_assets_by_share(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    writer = workspace.create_shard(name="批量")
    writer.append(np.ones(100, dtype=np.complex64), 1000.0, "分片一", "imported:x")
    writer.append(np.ones(50, dtype=np.complex64), 1000.0, "分片二", "imported:x")
    writer.seal()
    report = build_report(workspace)
    rows = {item["name"]: item for item in report["tables"]["assets"]}
    assert rows["分片一"]["bytes"] == 100 * 8
    assert rows["分片二"]["bytes"] == 50 * 8
    assert rows["分片一"]["exists"] is True
    # 分片文件在 assets/shards/ 子目录：不能算作无主文件
    assert report["consistency"]["orphan_files_count"] == 0
    # 磁盘总占用仍按整份分片计入“信号资产”类别
    shard_file = next((workspace.root / "assets" / "shards").glob("*.bin"))
    assets_category = next(item for item in report["categories"] if item["key"] == "assets")
    assert assets_category["bytes"] >= shard_file.stat().st_size
