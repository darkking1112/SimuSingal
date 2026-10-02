"""Workspace 的 schema 迁移与旧数据回填（MigrationMixin）。

由 :class:`signal_analysis.data.workspace.Workspace` 组合使用；
``schema_migrations`` 通过 ``super()`` 衔接 ``common.storage`` 的迁移框架。
"""

import json

from .schema import _ASSET_COLUMNS_V3, _SCHEMA_V3
from .utils import _classify_source, _ensure_columns


class MigrationMixin:
    """schema 版本 2/3 的迁移与回填（实例方法由 Workspace 提供上下文）。"""

    def schema_migrations(self):
        return super().schema_migrations() + [
            (2, "信号资产表", self._migration_assets_v2),
            (3, "信号集合与统一标注表", self._migration_unified_v3),
        ]

    @staticmethod
    def _migration_assets_v2(conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                sha256 TEXT NOT NULL, sample_rate REAL NOT NULL,
                sample_count INTEGER NOT NULL, created_at TEXT NOT NULL,
                source TEXT NOT NULL, label TEXT NOT NULL DEFAULT '');
            CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name);
            CREATE TABLE IF NOT EXISTS asset_metadata (
                asset_id TEXT PRIMARY KEY REFERENCES assets(id), metadata_json TEXT NOT NULL);
        """)

    def _migration_unified_v3(self, conn):
        conn.executescript(_SCHEMA_V3)
        _ensure_columns(conn, "assets", _ASSET_COLUMNS_V3)
        self._backfill_assets_v3(conn)
        self.default_taxonomy("detection", conn=conn)
        self.default_taxonomy("amc", conn=conn)

    def _backfill_assets_v3(self, conn):
        """旧数据补齐：来源分类、更新时刻、分组键、目标与参考参数。"""
        rows = [dict(row) for row in conn.execute("SELECT * FROM assets ORDER BY created_at,id")]
        metadata = {row["asset_id"]: row["metadata_json"]
                    for row in conn.execute("SELECT asset_id, metadata_json FROM asset_metadata")}
        for asset in rows:
            kind, parent = _classify_source(asset["source"])
            if asset.get("source_kind"):
                kind = asset["source_kind"]
            parent_id = asset.get("parent_asset_id")
            if parent_id is None and parent and parent != asset["id"]:
                exists = conn.execute("SELECT 1 FROM assets WHERE id=?", (parent,)).fetchone()
                parent_id = parent if exists else None
            conn.execute(
                "UPDATE assets SET source_kind=?, parent_asset_id=?, origin_group_id=?, "
                "updated_at=CASE WHEN updated_at='' THEN created_at ELSE updated_at END "
                "WHERE id=?",
                (kind, parent_id, asset.get("origin_group_id") or asset["id"], asset["id"]))
            has_target = conn.execute("SELECT 1 FROM targets WHERE asset_id=? LIMIT 1",
                                      (asset["id"],)).fetchone()
            if has_target:
                continue
            summary = None
            try:
                payload = json.loads(metadata.get(asset["id"]) or "{}")
                if isinstance(payload, dict):
                    summary = payload.get("generation")
            except ValueError:
                summary = None
            if isinstance(summary, dict):
                self._register_generation_targets(
                    conn, asset["id"], summary, int(asset["sample_count"]),
                    float(asset["sample_rate"]))
            else:
                self._register_placeholder_target(conn, asset["id"], int(asset["sample_count"]))
