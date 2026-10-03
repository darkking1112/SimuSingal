"""schema 迁移框架测试：旧库升级、备份、幂等与版本上限拒绝。"""
import json
import sqlite3

import numpy as np
import pytest

from signal_analysis.core_api import generate_iq
from signal_analysis.data import Workspace


LEGACY_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT,
    created_at TEXT NOT NULL, result_json TEXT NOT NULL);
CREATE TABLE assets (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
    sha256 TEXT NOT NULL, sample_rate REAL NOT NULL,
    sample_count INTEGER NOT NULL, created_at TEXT NOT NULL,
    source TEXT NOT NULL, label TEXT NOT NULL DEFAULT '');
CREATE INDEX idx_assets_name ON assets(name);
CREATE TABLE asset_metadata (
    asset_id TEXT PRIMARY KEY REFERENCES assets(id), metadata_json TEXT NOT NULL);
PRAGMA user_version=1;
"""


def _legacy_database(root, rows):
    """构造旧版（v1）数据库：内容与历史代码生成的完全一致。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(
        json.dumps({"project": "signal_analysis", "schema": 1}), encoding="utf-8")
    database = root / "catalog.sqlite3"
    conn = sqlite3.connect(database)
    conn.executescript(LEGACY_SCHEMA)
    for row in rows:
        conn.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)", row[:9])
        if len(row) > 9 and row[9] is not None:
            conn.execute("INSERT INTO asset_metadata VALUES (?,?)", (row[0], row[9]))
    conn.commit()
    conn.close()
    return database


def _generation_summary(rate=200_000.0, duration=0.05, seed=11):
    _, summary = generate_iq(
        rate, duration,
        [{"mode": "fm", "offset": 0.0, "bandwidth": 60_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 18.0}, seed)
    return summary


def _asset_row(asset_id, name, source, metadata_json=None, count=10_000, rate=200_000.0):
    return (asset_id, name, f"assets/{asset_id}.npy", f"{asset_id:0<64}"[:64], rate,
            count, "2026-09-30T00:00:00+00:00", source, "", metadata_json)


def test_fresh_workspace_uses_latest_schema(tmp_path):
    workspace = Workspace(tmp_path / "ws")
    assert workspace.schema_version == 4
    marker = json.loads((workspace.root / "project.json").read_text(encoding="utf-8"))
    assert marker == {"project": "signal_analysis", "schema": 4}
    with workspace.connect() as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(assets)")}
    assert version == 4
    # 资产存储格式列（v4）：文件编码 + 字节序
    assert {"storage_format", "endian"} <= columns
    for table in ("assets", "asset_shards", "collections", "collection_members",
                  "targets", "target_versions", "taxonomies", "task_sets",
                  "detection_labels", "amc_labels", "asset_coverage", "recipes",
                  "dataset_versions", "experiments", "evaluations", "coverage_cache"):
        assert table in tables


def test_legacy_database_upgrade_backup_and_idempotence(tmp_path):
    summary = _generation_summary()
    root = tmp_path / "legacy"
    _legacy_database(root, [
        _asset_row("a" * 32, "生成样本", "generated:iq_fm_v1",
                   json.dumps({"generation": summary})),
        _asset_row("b" * 32, "导入样本", str(tmp_path / "记录.bin")),
        _asset_row("c" * 32, "衍生样本", "parent:" + "b" * 32),
    ])
    workspace = Workspace(root)
    assert workspace.schema_version == 4
    # 迁移前自动备份
    backups = list((root / "backups").glob("catalog-v1-*.sqlite3"))
    assert len(backups) == 1
    # 旧 ID / 路径 / 备注保持不变
    assets = {asset["id"]: asset for asset in workspace.list_assets(limit=10)}
    assert set(assets) == {"a" * 32, "b" * 32, "c" * 32}
    assert assets["a" * 32]["path"] == f"assets/{'a' * 32}.npy"
    # 老资产本来就是独立 NPY：新列补上默认值即可，无需回填文件
    assert assets["a" * 32]["storage_format"] == "npy"
    assert assets["a" * 32]["endian"] == "little"
    assert assets["a" * 32]["source_kind"] == "generated"
    assert assets["b" * 32]["source_kind"] == "imported"
    assert assets["c" * 32]["source_kind"] == "derived"
    assert assets["c" * 32]["parent_asset_id"] == "b" * 32
    assert assets["a" * 32]["origin_group_id"] == "a" * 32
    assert assets["a" * 32]["updated_at"] != ""
    # 有生成摘要的资产写了目标与参考参数；无摘要的写未知占位
    targets = workspace.list_targets("a" * 32)
    assert [item["target_key"] for item in targets] == ["s0"]
    assert targets[0]["current"]["source"] == "generator"
    assert targets[0]["for_detection"] == 1 and targets[0]["for_amc"] == 1
    placeholder = workspace.list_targets("b" * 32)
    assert placeholder[0]["scope"] == "whole_record"
    assert placeholder[0]["current"]["source"] == "import"
    assert placeholder[0]["current"]["modulation"] is None
    assert placeholder[0]["current"]["f_low_hz"] is None
    # 再次打开不重复迁移、不重复备份
    reopened = Workspace(root)
    assert reopened.schema_version == 4
    assert len(list((root / "backups").glob("*.sqlite3"))) == 1
    assert len(reopened.list_targets("a" * 32)) == 1
    assert len(reopened.list_targets("b" * 32)) == 1


def test_newer_schema_is_rejected(tmp_path):
    root = tmp_path / "future"
    root.mkdir()
    (root / "project.json").write_text(
        json.dumps({"project": "signal_analysis", "schema": 99}), encoding="utf-8")
    conn = sqlite3.connect(root / "catalog.sqlite3")
    conn.executescript("PRAGMA user_version=99;")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="高于当前程序"):
        Workspace(root)


def test_malformed_metadata_falls_back_to_placeholder(tmp_path):
    root = tmp_path / "broken"
    _legacy_database(root, [_asset_row("d" * 32, "坏数据", "generated:bad")])
    # 生成摘要被截断为非法 JSON：迁移不阻断，按“无摘要”写占位目标
    conn = sqlite3.connect(root / "catalog.sqlite3")
    conn.execute("UPDATE asset_metadata SET metadata_json='{oops' WHERE asset_id=?",
                 ("d" * 32,))
    conn.commit()
    conn.close()
    workspace = Workspace(root)
    assert workspace.schema_version == 4
    assert workspace.list_targets("d" * 32)[0]["current"]["source"] == "import"
    assert len(list((root / "backups").glob("catalog-v1-*.sqlite3"))) == 1
