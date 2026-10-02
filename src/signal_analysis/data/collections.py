"""集合与集合统计（CollectionMixin）。

集合是资产的命名分组（多对多、按 position 排序）；统计方法聚合集合内的
目标/参数分布。由 :class:`signal_analysis.data.workspace.Workspace` 组合使用。
"""

import uuid

from common.storage import utc_now
from ..storage.schema import SCOPES
from ..storage.utils import _bin_specs, _clean_text, _enum, _optional_text


class CollectionMixin:
    """集合 CRUD、成员管理与集合级统计。"""

    # ------------------------------------------------------------------ 集合
    def create_collection(self, name, *, description="", source_kind="manual",
                          recipe_id=None, created_by=None):
        name = _clean_text(name, "集合名称", 100)
        source_kind = _enum(source_kind, ("manual", "generated", "imported", "legacy"),
                            "集合来源")
        now = utc_now()
        collection_id = uuid.uuid4().hex
        with self.connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO collections VALUES (?,?,?,?,?,?,?,?,NULL)",
                    (collection_id, name, str(description or ""), source_kind, recipe_id,
                     _optional_text(created_by, "创建者", 100), now, now))
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("集合名称已存在") from exc
                raise
        return self.get_collection(collection_id)

    def get_collection(self, collection_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM collections WHERE id=?",
                               (collection_id,)).fetchone()
        if row is None:
            raise ValueError("集合不存在")
        return dict(row)

    def list_collections(self, *, include_archived=False, limit=500):
        clause = "" if include_archived else "WHERE c.archived_at IS NULL"
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT c.*, (SELECT COUNT(*) FROM collection_members m "
                f"WHERE m.collection_id=c.id) AS asset_count FROM collections c {clause} "
                "ORDER BY c.created_at DESC, c.id LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def update_collection(self, collection_id, *, name=None, description=None):
        self.get_collection(collection_id)
        sets, params = ["updated_at=?"], [utc_now()]
        if name is not None:
            sets.append("name=?")
            params.append(_clean_text(name, "集合名称", 100))
        if description is not None:
            sets.append("description=?")
            params.append(str(description))
        params.append(collection_id)
        with self.connect() as conn:
            try:
                conn.execute(f"UPDATE collections SET {', '.join(sets)} WHERE id=?", params)
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("集合名称已存在") from exc
                raise
        return self.get_collection(collection_id)

    def archive_collection(self, collection_id, archived=True):
        self.get_collection(collection_id)
        with self.connect() as conn:
            conn.execute("UPDATE collections SET archived_at=?, updated_at=? WHERE id=?",
                         (utc_now() if archived else None, utc_now(), collection_id))
        return self.get_collection(collection_id)

    def add_collection_member(self, collection_id, asset_id, *, added_by=None):
        """加入集合成员；已存在时返回 False（幂等，便于批量导入重跑）。"""
        self.get_collection(collection_id)
        self.get_asset(asset_id)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM collection_members WHERE collection_id=? AND asset_id=?",
                (collection_id, asset_id)).fetchone()
            if row:
                return False
            row = conn.execute("SELECT MAX(position) FROM collection_members "
                               "WHERE collection_id=?", (collection_id,)).fetchone()
            # 注意：position 从 0 开始，不能用 ``row[0] or -1``（0 会被当假值）
            position = 0 if row[0] is None else int(row[0]) + 1
            conn.execute("INSERT INTO collection_members VALUES (?,?,?,?,?)",
                         (collection_id, asset_id, position,
                          _optional_text(added_by, "添加者", 100), utc_now()))
            conn.execute("UPDATE collections SET updated_at=? WHERE id=?",
                         (utc_now(), collection_id))
        return True

    def add_collection_members(self, collection_id, asset_ids, *, added_by=None):
        return sum(1 for asset_id in asset_ids
                   if self.add_collection_member(collection_id, asset_id, added_by=added_by))

    def remove_collection_member(self, collection_id, asset_id):
        self.get_collection(collection_id)
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM collection_members WHERE collection_id=? "
                                  "AND asset_id=?", (collection_id, asset_id))
        return cursor.rowcount > 0

    def collection_asset_ids(self, collection_id):
        self.get_collection(collection_id)
        with self.connect() as conn:
            rows = conn.execute("SELECT asset_id FROM collection_members WHERE "
                                "collection_id=? ORDER BY position", (collection_id,)).fetchall()
        return [row[0] for row in rows]

    def collections_of_asset(self, asset_id):
        self.get_asset(asset_id)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT c.* FROM collections c JOIN collection_members m "
                "ON m.collection_id=c.id WHERE m.asset_id=? ORDER BY c.created_at",
                (asset_id,)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 集合统计
    def collection_summary(self, collection_id):
        """集合概览：资产/目标计数、各任务标注进度、覆盖与来源分布。"""
        collection = self.get_collection(collection_id)
        with self.connect() as conn:
            assets = conn.execute("SELECT COUNT(*) FROM collection_members "
                                  "WHERE collection_id=?", (collection_id,)).fetchone()[0]
            targets = conn.execute(
                "SELECT COUNT(*) FROM targets t JOIN collection_members m "
                "ON m.asset_id=t.asset_id WHERE m.collection_id=?",
                (collection_id,)).fetchone()[0]
            hops = conn.execute(
                "SELECT COUNT(*) FROM targets t JOIN collection_members m "
                "ON m.asset_id=t.asset_id WHERE m.collection_id=? AND t.scope='hop'",
                (collection_id,)).fetchone()[0]
            sources = conn.execute(
                "SELECT a.source_kind, COUNT(*) FROM assets a JOIN collection_members m "
                "ON m.asset_id=a.id WHERE m.collection_id=? GROUP BY a.source_kind",
                (collection_id,)).fetchall()
        progress = [self.task_set_progress(item["id"])
                    for item in self.list_task_sets(collection_id)]
        return {**collection, "asset_count": int(assets), "target_count": int(targets),
                "hop_count": int(hops),
                "source_kinds": {str(key): int(value) for key, value in sources},
                "task_sets": progress}

    def target_axis_stats(self, collection_id, axis, *, bins=None, top=None, scope=None):
        """按“当前目标参数”聚合集合内目标数量；未知值归入“未知”。

        ``axis`` 支持枚举字段（modulation / waveform_mode / signal_type /
        is_hopping / source / scope）与数值字段（snr_db / bandwidth_hz /
        center_hz / symbol_rate_baud / hop_rate_hz）。数值轴可给 ``bins``
        边界列表（含 ``None`` 表示 ±∞）。``scope`` 可按目标粒度过滤：
        ``signal`` 表示“信号级”（排除逐跳子目标），或取
        ``session`` / ``hop`` / ``whole_record`` / ``segment`` 之一。
        """
        numeric_defaults = {
            "snr_db": [None, 0, 5, 10, 15, 20, 25, 30, None],
            "bandwidth_hz": [None, 1e4, 5e4, 1e5, 5e5, 1e6, None],
            "center_hz": [None, -5e5, -1e5, 0, 1e5, 5e5, None],
            "symbol_rate_baud": [None, 1e4, 5e4, 1e5, 2e5, None],
            "hop_rate_hz": [None, 10, 50, 100, 200, None],
        }
        clause = ""
        params = [collection_id]
        if scope == "signal":
            clause = "AND t.scope != 'hop'"
        elif scope is not None:
            _enum(scope, SCOPES, "目标粒度")
            clause = "AND t.scope=?"
            params.append(scope)
        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT t.scope, t.for_detection, t.for_amc, t.hop_index, v.* "
                "FROM targets t JOIN collection_members m ON m.asset_id=t.asset_id "
                "JOIN target_versions v ON v.target_id=t.id AND v.version_no = "
                "(SELECT MAX(version_no) FROM target_versions WHERE target_id=t.id) "
                f"WHERE m.collection_id=? {clause}", params)]
        if not rows:
            return []
        counts = {}
        if axis in ("modulation", "waveform_mode", "signal_type", "source"):
            for row in rows:
                value = row.get(axis)
                key = str(value) if value not in (None, "") else "未知"
                counts[key] = counts.get(key, 0) + 1
            items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        elif axis in ("is_hopping", "scope", "for_detection", "for_amc"):
            for row in rows:
                value = row.get(axis)
                if axis in ("is_hopping", "for_detection", "for_amc"):
                    key = "未知" if value is None else ("是" if value else "否")
                else:
                    key = str(value)
                counts[key] = counts.get(key, 0) + 1
            items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        elif axis in numeric_defaults:
            edges = numeric_defaults[axis] if bins is None else bins
            specs = _bin_specs(edges)
            counts = {label: 0 for _, _, label in specs}
            counts["未知"] = 0
            for row in rows:
                value = row.get(axis)
                label = "未知"
                if value is not None:
                    number = float(value)
                    for low, high, name in specs:
                        if (low is None or number >= low) and (high is None or number < high):
                            label = name
                            break
                counts[label] += 1
            return [{"label": name, "count": int(count)} for name, count in counts.items()]
        else:
            raise ValueError(f"不支持的统计轴：{axis}")
        if top:
            items = items[:top]
        return [{"label": name, "count": int(count)} for name, count in items]
