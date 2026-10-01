"""Signal-only asset tables and arrays, extending shared run storage.

本模块实现信号分析工作区的 schema **v3**（数据模型权威文档：docs/数据库设计.md）：

* **v2**：原有 `assets` / `asset_metadata` 两张表（与历史版本逐列一致）；
* **v3**：资产扩展列（来源分类、分片定位、分组键、审计字段）、分片表
  `asset_shards`、集合 `collections` / `collection_members`、目标
  `targets` / `target_versions`、任务字典 `taxonomies` / `task_sets`、
  标签 `detection_labels` / `amc_labels`、覆盖 `asset_coverage`、
  生成配方 `recipes`、数据版本 `dataset_versions`、实验与评估
  `experiments` / `evaluations`、覆盖度缓存 `coverage_cache`。

设计约定（与方案文档一致）：

* 标注**追加式版本化**：重新标注写入更高的 ``revision_no`` 并以
  ``supersedes_id`` 指向旧版本，旧行不修改、不删除；“当前标签”= 同一任务
  标注集内该目标的最大 ``revision_no``；
* 未知一律为 ``NULL``，绝不用 0 代替；
* 资产物理数据两类：独立 NPY（``storage_kind='file'``，现状）与分片内偏移
  （``storage_kind='shard'``，批量生成/导入）；分片封存后不可变；
* 迁移可重复执行（按现有行判断跳过），升级前由 :class:`common.storage.Workspace`
  自动备份数据库。
"""
import hashlib
import json
from pathlib import Path
import re
import uuid

import numpy as np

from common.storage import Workspace as RunWorkspace, file_digest, utc_now
from .core_api import validate_rate, validate_samples

SOURCE_KINDS = ("imported", "generated", "derived", "legacy")
SAMPLE_KINDS = ("complex", "real")
STORAGE_KINDS = ("file", "shard", "recipe")
SCOPES = ("whole_record", "session", "hop", "segment")
VERSION_SOURCES = ("generator", "import", "manual", "external", "algorithm")
COVERAGE_VALUES = ("unknown", "partial", "complete")
CLASS_STATES = ("known", "unknown", "out_of_taxonomy")
DETECTION_SEMANTICS = ("session_v1", "per_hop_v1")
TASKS = ("detection", "amc")
LABEL_TABLES = {"detection": "detection_labels", "amc": "amc_labels"}

#: 生成器样式 → 规范调制名；跳频与演示样式没有单一调制名（NULL = 未知）。
_MODULATION_NAMES = {"am": "AM", "fm": "FM", "ssb": "SSB", "ask2": "2ASK",
                     "qpsk": "QPSK", "qam16": "16QAM", "qam64": "64QAM"}

#: 目标键解析：``s3`` 与 ``s3.h2``（会话 3 的第 2 跳）。
_TARGET_KEY = re.compile(r"^s(\d+)(?:\.h(\d+))?$")

_ASSET_COLUMNS_V3 = {
    "source_kind": "TEXT NOT NULL DEFAULT ''",
    "sample_kind": "TEXT NOT NULL DEFAULT 'complex'",
    "dtype": "TEXT NOT NULL DEFAULT 'complex64'",
    "storage_kind": "TEXT NOT NULL DEFAULT 'file'",
    "shard_id": "TEXT",
    "shard_offset": "INTEGER",
    "shard_length": "INTEGER",
    "origin_group_id": "TEXT",
    "parent_asset_id": "TEXT",
    "rf_center_hz": "REAL",
    "capture_started_at": "TEXT",
    "created_by": "TEXT",
    "updated_at": "TEXT NOT NULL DEFAULT ''",
    "archived_at": "TEXT",
}

_SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS asset_shards (
    id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL DEFAULT '', record_count INTEGER NOT NULL DEFAULT 0,
    size_bytes INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, sealed_at TEXT);
CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, description TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL, recipe_id TEXT, created_by TEXT,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, archived_at TEXT);
CREATE TABLE IF NOT EXISTS collection_members (
    collection_id TEXT NOT NULL REFERENCES collections(id),
    asset_id TEXT NOT NULL REFERENCES assets(id),
    position INTEGER NOT NULL, added_by TEXT, added_at TEXT NOT NULL,
    PRIMARY KEY (collection_id, asset_id));
CREATE INDEX IF NOT EXISTS idx_members_asset ON collection_members(asset_id, collection_id);
CREATE INDEX IF NOT EXISTS idx_members_order ON collection_members(collection_id, position);
CREATE TABLE IF NOT EXISTS targets (
    id TEXT PRIMARY KEY, asset_id TEXT NOT NULL REFERENCES assets(id),
    target_key TEXT NOT NULL, scope TEXT NOT NULL,
    parent_target_id TEXT REFERENCES targets(id), hop_index INTEGER,
    for_detection INTEGER NOT NULL DEFAULT 0, for_amc INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, UNIQUE (asset_id, target_key));
CREATE INDEX IF NOT EXISTS idx_targets_asset ON targets(asset_id);
CREATE TABLE IF NOT EXISTS target_versions (
    id TEXT PRIMARY KEY, target_id TEXT NOT NULL REFERENCES targets(id),
    version_no INTEGER NOT NULL, supersedes_id TEXT, source TEXT NOT NULL,
    created_at TEXT NOT NULL, note TEXT,
    sample_start INTEGER, sample_end INTEGER,
    f_low_hz REAL, f_high_hz REAL, center_hz REAL, bandwidth_hz REAL,
    nominal_center_hz REAL, nominal_bandwidth_hz REAL,
    signal_type TEXT, waveform_mode TEXT, modulation TEXT,
    symbol_rate_baud REAL, hop_rate_hz REAL, is_hopping INTEGER,
    snr_db REAL, snr_definition TEXT, power_dbfs REAL, params_json TEXT,
    UNIQUE (target_id, version_no));
CREATE INDEX IF NOT EXISTS idx_tv_target ON target_versions(target_id, version_no);
CREATE INDEX IF NOT EXISTS idx_tv_modulation ON target_versions(modulation);
CREATE INDEX IF NOT EXISTS idx_tv_waveform ON target_versions(waveform_mode);
CREATE INDEX IF NOT EXISTS idx_tv_snr ON target_versions(snr_db);
CREATE INDEX IF NOT EXISTS idx_tv_bandwidth ON target_versions(bandwidth_hz);
CREATE TABLE IF NOT EXISTS taxonomies (
    id TEXT PRIMARY KEY, task TEXT NOT NULL, name TEXT NOT NULL, version TEXT NOT NULL,
    classes_json TEXT NOT NULL, mapping_json TEXT, UNIQUE (task, name, version));
CREATE TABLE IF NOT EXISTS task_sets (
    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL REFERENCES collections(id),
    task TEXT NOT NULL CHECK (task IN ('detection','amc')),
    name TEXT NOT NULL, taxonomy_id TEXT NOT NULL REFERENCES taxonomies(id),
    label_semantics TEXT, created_by TEXT, created_at TEXT NOT NULL,
    UNIQUE (collection_id, task, name));
CREATE TABLE IF NOT EXISTS detection_labels (
    id TEXT PRIMARY KEY, task_set_id TEXT NOT NULL REFERENCES task_sets(id),
    target_id TEXT NOT NULL REFERENCES targets(id),
    revision_no INTEGER NOT NULL, supersedes_id TEXT, source TEXT NOT NULL,
    created_at TEXT NOT NULL, label_semantics TEXT NOT NULL, class_name TEXT,
    target_version_id TEXT NOT NULL REFERENCES target_versions(id),
    include INTEGER NOT NULL DEFAULT 1, note TEXT,
    UNIQUE (task_set_id, target_id, revision_no));
CREATE INDEX IF NOT EXISTS idx_dlabels_target ON detection_labels(target_id);
CREATE TABLE IF NOT EXISTS amc_labels (
    id TEXT PRIMARY KEY, task_set_id TEXT NOT NULL REFERENCES task_sets(id),
    target_id TEXT NOT NULL REFERENCES targets(id),
    revision_no INTEGER NOT NULL, supersedes_id TEXT, source TEXT NOT NULL,
    created_at TEXT NOT NULL, class_name TEXT, class_state TEXT NOT NULL,
    window_start INTEGER, window_end INTEGER,
    analysis_center_hz REAL, analysis_bandwidth_hz REAL,
    target_version_id TEXT NOT NULL REFERENCES target_versions(id), note TEXT,
    UNIQUE (task_set_id, target_id, revision_no));
CREATE INDEX IF NOT EXISTS idx_alabels_target ON amc_labels(target_id);
CREATE TABLE IF NOT EXISTS asset_coverage (
    id TEXT PRIMARY KEY, task_set_id TEXT NOT NULL REFERENCES task_sets(id),
    asset_id TEXT NOT NULL REFERENCES assets(id),
    revision_no INTEGER NOT NULL, coverage TEXT NOT NULL, negative_kind TEXT,
    source TEXT NOT NULL, created_at TEXT NOT NULL,
    UNIQUE (task_set_id, asset_id, revision_no));
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, engine TEXT NOT NULL,
    recipe_json TEXT NOT NULL, recipe_sha256 TEXT NOT NULL,
    created_by TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dataset_versions (
    id TEXT PRIMARY KEY, task_set_id TEXT NOT NULL REFERENCES task_sets(id),
    version_no INTEGER NOT NULL, manifest_path TEXT, manifest_sha256 TEXT,
    status TEXT NOT NULL, sample_count INTEGER NOT NULL DEFAULT 0,
    asset_count INTEGER NOT NULL DEFAULT 0, split_json TEXT,
    preprocessing_json TEXT, taxonomy_snapshot_json TEXT, card_json TEXT,
    created_at TEXT NOT NULL, UNIQUE (task_set_id, version_no));
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, task TEXT NOT NULL, dataset_version_id TEXT,
    config_json TEXT NOT NULL, status TEXT NOT NULL, output_path TEXT,
    model_sha256 TEXT, created_at TEXT NOT NULL, finished_at TEXT);
CREATE INDEX IF NOT EXISTS idx_experiments_dataset ON experiments(dataset_version_id);
CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY, dataset_version_id TEXT NOT NULL,
    experiment_id TEXT, model_sha256 TEXT, contract TEXT NOT NULL,
    config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
    result_path TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_evaluations_dataset ON evaluations(dataset_version_id);
CREATE TABLE IF NOT EXISTS coverage_cache (
    collection_id TEXT NOT NULL, recipe_id TEXT NOT NULL DEFAULT '',
    axis TEXT NOT NULL, bin TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (collection_id, recipe_id, axis, bin));
"""


# ---------------------------------------------------------------------- 工具


def _clean_text(value, name, maximum=200, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name}不能为空")
    if not isinstance(value, str):
        raise ValueError(f"{name}应为文本")
    text = value.strip()
    if not text and not allow_none:
        raise ValueError(f"{name}不能为空")
    if len(text) > maximum:
        raise ValueError(f"{name}最多 {maximum} 个字符")
    return text


def _optional_text(value, name, maximum=200):
    if value is None:
        return None
    return _clean_text(value, name, maximum, allow_none=True) or None


def _number(value, name, *, minimum=None, maximum=None):
    """有限数值或 ``None``；NaN/Inf 与越界直接报错。"""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name}必须为有限数值")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}不能大于 {maximum}")
    return number


def _integer(value, name, *, minimum=None, maximum=None):
    if value is None:
        return None
    if isinstance(value, bool) or value != int(value):
        raise ValueError(f"{name}应为整数")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}不能大于 {maximum}")
    return number


def _enum(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name}应为 {' / '.join(allowed)} 之一")
    return value


def _json_text(value, name="内容"):
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}不是有效的 JSON：{exc}") from exc


def _json_safe(value):
    """递归去掉非有限浮点；返回可安全序列化的副本。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _classify_source(source):
    """按既有 ``source`` 文本推断来源分类与父资产；不猜测时记为 legacy。"""
    text = str(source or "")
    if text.startswith("parent:"):
        parent = text.split(":", 1)[1].strip()
        return "derived", (parent or None)
    if text.startswith("generated") or text == "generated":
        return "generated", None
    if text.startswith("imported") or re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("/"):
        return "imported", None
    return "legacy", None


def _target_sort_key(row):
    match = _TARGET_KEY.match(str(row.get("target_key", "")))
    if match:
        session = int(match.group(1))
        hop = int(match.group(2)) if match.group(2) is not None else -1
        return (session, hop, str(row.get("target_key", "")))
    return (10 ** 9, -1, str(row.get("target_key", "")))


def _ensure_columns(conn, table, columns):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _fmt_edge(value):
    return f"{float(value):g}"


def _bin_specs(edges):
    """分箱边界 → ``[(low, high, label)]``；``None`` 表示开放端。"""
    specs = []
    for low, high in zip(edges, edges[1:]):
        if low is None and high is None:
            label = "全部"
        elif low is None:
            label = f"< {_fmt_edge(high)}"
        elif high is None:
            label = f"≥ {_fmt_edge(low)}"
        else:
            label = f"[{_fmt_edge(low)}, {_fmt_edge(high)})"
        specs.append((low, high, label))
    return specs


class Workspace(RunWorkspace):
    project = "signal_analysis"

    def __init__(self, root):
        super().__init__(root)
        (self.root / "assets").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 迁移
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

    # ------------------------------------------------------------------ 资产
    def _insert_asset(self, conn, *, asset_id, name, relative, digest, rate, count,
                      source, source_kind, sample_kind, dtype, storage_kind,
                      shard_id=None, shard_offset=None, shard_length=None,
                      origin_group_id=None, parent_asset_id=None, rf_center_hz=None,
                      capture_started_at=None, created_by=None):
        now = utc_now()
        conn.execute(
            "INSERT INTO assets (id, name, path, sha256, sample_rate, sample_count, "
            "created_at, source, label, source_kind, sample_kind, dtype, storage_kind, "
            "shard_id, shard_offset, shard_length, origin_group_id, parent_asset_id, "
            "rf_center_hz, capture_started_at, created_by, updated_at, archived_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            (asset_id, name, relative, digest, rate, count, now, str(source), "",
             source_kind, sample_kind, dtype, storage_kind, shard_id, shard_offset,
             shard_length, origin_group_id or asset_id, parent_asset_id, rf_center_hz,
             capture_started_at, created_by, now))

    def add_samples(self, samples, sample_rate, name, source="generated", *, metadata=None,
                    source_kind=None, origin_group_id=None, parent_asset_id=None,
                    rf_center_hz=None, capture_started_at=None, created_by=None):
        """新增一条独立 NPY 资产；带生成摘要时同时登记目标与参考参数版本。"""
        data = validate_samples(samples)
        rate = validate_rate(sample_rate)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("名称应为 1～200 个字符")
        inferred_kind, inferred_parent = _classify_source(source)
        kind = _enum(source_kind or inferred_kind, SOURCE_KINDS, "来源分类")
        parent = parent_asset_id or inferred_parent
        if parent is not None:
            with self.connect() as conn:
                exists = conn.execute("SELECT 1 FROM assets WHERE id=?", (parent,)).fetchone()
            if not exists:
                raise ValueError("父资产不存在")
        asset_id = uuid.uuid4().hex
        relative = f"assets/{asset_id}.npy"
        destination = self.root / relative
        temporary = destination.with_suffix(".tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, data, allow_pickle=False)
            temporary.replace(destination)
            with self.connect() as conn:
                self._insert_asset(
                    conn, asset_id=asset_id, name=name, relative=relative,
                    digest=file_digest(destination), rate=rate, count=data.size,
                    source=source, source_kind=kind, sample_kind="complex",
                    dtype="complex64", storage_kind="file",
                    origin_group_id=origin_group_id, parent_asset_id=parent,
                    rf_center_hz=_number(rf_center_hz, "射频中心"),
                    capture_started_at=_optional_text(capture_started_at, "采集时间", 64),
                    created_by=_optional_text(created_by, "创建者", 100))
                if metadata is not None:
                    conn.execute("INSERT INTO asset_metadata VALUES (?,?)",
                                 (asset_id, json.dumps(metadata, ensure_ascii=False,
                                                       allow_nan=False)))
                    generation = metadata.get("generation") if isinstance(metadata, dict) else None
                    if isinstance(generation, dict):
                        self._register_generation_targets(conn, asset_id, generation,
                                                          int(data.size), float(rate))
        except BaseException:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return self.get_asset(asset_id)

    def get_metadata(self, asset_id):
        self.get_asset(asset_id)
        with self.connect() as conn:
            row = conn.execute("SELECT metadata_json FROM asset_metadata WHERE asset_id=?",
                               (asset_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def get_asset(self, asset_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            raise ValueError("数据记录不存在")
        return dict(row)

    def _asset_clauses(self, search="", *, collection_id=None, scattered=False,
                       include_archived=False):
        """资产筛选条件列表（不含 WHERE 前缀）与参数。"""
        clauses, params = [], []
        if not include_archived:
            clauses.append("assets.archived_at IS NULL")
        if search:
            clauses.append("instr(assets.name, ?) > 0")
            params.append(str(search))
        if collection_id is not None:
            clauses.append("EXISTS (SELECT 1 FROM collection_members m WHERE "
                           "m.asset_id=assets.id AND m.collection_id=?)")
            params.append(str(collection_id))
        if scattered:
            clauses.append("NOT EXISTS (SELECT 1 FROM collection_members m JOIN collections c "
                           "ON c.id=m.collection_id WHERE m.asset_id=assets.id "
                           "AND c.archived_at IS NULL)")
        return clauses, params

    def list_assets(self, search="", limit=100, offset=0, *, collection_id=None,
                    scattered=False, include_archived=False, after=None):
        """分页列出资产。

        ``collection_id`` 指定时按集合内 ``position`` 顺序；否则按
        ``created_at DESC, id``（键集分页用 ``after=(created_at, id)`` 取下一页）。
        """
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError("分页参数不合法")
        clauses, params = self._asset_clauses(search, collection_id=collection_id,
                                              scattered=scattered,
                                              include_archived=include_archived)
        if collection_id is not None:
            if after is not None:
                raise ValueError("集合内列表不支持键集分页")
            conditions = ["m.collection_id=?", *clauses]
            sql = ("SELECT assets.* FROM assets JOIN collection_members m "
                   "ON m.asset_id=assets.id WHERE " + " AND ".join(conditions) +
                   " ORDER BY m.position, assets.id LIMIT ? OFFSET ?")
            values = [str(collection_id), *params, limit, offset]
        else:
            if after is not None:
                created_at, last_id = after
                clauses.append("(assets.created_at < ? OR "
                               "(assets.created_at = ? AND assets.id > ?))")
                params = [*params, str(created_at), str(created_at), str(last_id)]
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = ("SELECT assets.* FROM assets " + where +
                   " ORDER BY assets.created_at DESC, assets.id LIMIT ? OFFSET ?")
            values = [*params, limit, offset]
        with self.connect() as conn:
            rows = conn.execute(sql, values).fetchall()
        return [dict(row) for row in rows]

    def count_assets(self, search="", *, collection_id=None, scattered=False,
                     include_archived=False):
        clauses, params = self._asset_clauses(search, collection_id=collection_id,
                                              scattered=scattered,
                                              include_archived=include_archived)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            return int(conn.execute(f"SELECT count(*) FROM assets {where}",
                                    params).fetchone()[0])

    def archived_assets(self, limit=500):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM assets WHERE archived_at IS NOT NULL "
                                "ORDER BY archived_at DESC, id LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def archive_asset(self, asset_id, archived=True):
        self.get_asset(asset_id)
        with self.connect() as conn:
            conn.execute("UPDATE assets SET archived_at=?, updated_at=? WHERE id=?",
                         (utc_now() if archived else None, utc_now(), asset_id))
        return self.get_asset(asset_id)

    def resolve_asset(self, asset):
        """独立 NPY 资产的文件路径；分片资产必须走 :meth:`load_samples`。"""
        if asset.get("storage_kind", "file") != "file":
            raise ValueError("分片资产没有独立文件，请通过 load_samples 读取")
        path = (self.root / asset["path"]).resolve()
        if not path.is_relative_to(self.root / "assets"):
            raise ValueError("资产路径越界")
        if not path.is_file() or file_digest(path) != asset["sha256"]:
            raise ValueError("资产文件缺失或校验失败")
        return path

    def _shard_record(self, asset):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM asset_shards WHERE id=?",
                               (asset["shard_id"],)).fetchone()
        if row is None:
            raise ValueError("分片记录缺失")
        shard = dict(row)
        path = (self.root / shard["path"]).resolve()
        if not path.is_relative_to(self.root / "assets"):
            raise ValueError("分片路径越界")
        if not path.is_file():
            raise ValueError("分片文件缺失或校验失败")
        return path, shard

    def load_samples(self, asset_id):
        asset = self.get_asset(asset_id)
        kind = asset.get("storage_kind", "file")
        if kind == "file":
            return asset, np.load(self.resolve_asset(asset), mmap_mode="r", allow_pickle=False)
        if kind == "shard":
            path, _ = self._shard_record(asset)
            offset = int(asset["shard_offset"])
            length = int(asset["shard_length"])
            if offset < 0 or length <= 0:
                raise ValueError("分片资产定位参数无效")
            itemsize = np.dtype("complex64").itemsize
            with path.open("rb") as stream:
                stream.seek(offset * itemsize)
                payload = stream.read(length * itemsize)
            if len(payload) != length * itemsize or \
                    hashlib.sha256(payload).hexdigest() != asset["sha256"]:
                raise ValueError("资产文件缺失或校验失败")
            return asset, np.frombuffer(payload, dtype=np.complex64).copy()
        raise ValueError(f"暂不支持的存储类型：{kind}")

    def set_label(self, asset_id, label):
        if not isinstance(label, str) or len(label) > 200:
            raise ValueError("备注最多 200 个字符")
        self.get_asset(asset_id)
        with self.connect() as conn:
            conn.execute("UPDATE assets SET label=?, updated_at=? WHERE id=?",
                         (label, utc_now(), asset_id))

    def save_run(self, kind, result, arrays=None, asset_id=None):
        if asset_id is not None:
            self.get_asset(asset_id)
        return super().save_run(kind, {**result, "asset_id": asset_id}, arrays, source_id=asset_id)

    # ------------------------------------------------------------------ 分片
    def create_shard(self, *, name="", created_by=None):
        """创建分片写入器；封存前可继续追加，封存后不可改。"""
        return ShardWriter(self, name=name, created_by=created_by)

    def get_shard(self, shard_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM asset_shards WHERE id=?", (shard_id,)).fetchone()
        if row is None:
            raise ValueError("分片不存在")
        return dict(row)

    def list_shards(self, limit=500):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM asset_shards ORDER BY created_at DESC, id "
                                "LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 生成器摘要 → 目标
    def _register_generation_targets(self, conn, asset_id, summary, sample_count, rate):
        """把生成器摘要登记为 targets / target_versions（幂等：已有目标则跳过）。"""
        from .evaluation import hop_truth, signal_truth

        exists = conn.execute("SELECT 1 FROM targets WHERE asset_id=? LIMIT 1",
                              (asset_id,)).fetchone()
        if exists:
            return
        raw_signals = [entry for entry in (summary.get("signals") or [])
                       if isinstance(entry, dict)]
        truth = signal_truth(summary)
        hopping_hops = hop_truth(summary)
        if not truth:
            return
        for index, entry in enumerate(truth):
            mode = str(entry.get("mode", ""))
            hopping = bool(entry.get("hopping"))
            scope = "session" if (hopping or len(truth) > 1) else "whole_record"
            # AMC 适用标记：有规范调制名的样式置位；跳频/演示样式保持 0。
            target_id = self._insert_target_row(
                conn, asset_id=asset_id, target_key=f"s{index}", scope=scope,
                parent_target_id=None, hop_index=None, for_detection=1,
                for_amc=1 if mode in _MODULATION_NAMES else 0)
            raw = raw_signals[index] if index < len(raw_signals) else {}
            mapped = {key: _json_safe(value) for key, value in raw.items()
                      if key not in ("mode", "offset", "bandwidth", "bandwidth_actual",
                                     "power_dbfs", "power_dbfs_actual", "snr_inband_db")}
            self._insert_target_version_row(
                conn, target_id=target_id, source="generator",
                sample_start=0, sample_end=int(sample_count),
                f_low_hz=entry.get("f_low_hz"), f_high_hz=entry.get("f_high_hz"),
                nominal_center_hz=entry.get("nominal_offset_hz"),
                nominal_bandwidth_hz=entry.get("nominal_bandwidth_hz"),
                signal_type=None, waveform_mode=mode,
                modulation=_MODULATION_NAMES.get(mode),
                symbol_rate_baud=_number(raw.get("symbol_rate"), "符号率", minimum=0.0)
                if raw.get("symbol_rate") is not None else None,
                hop_rate_hz=_number(raw.get("hop_rate"), "跳速", minimum=0.0)
                if raw.get("hop_rate") is not None else None,
                is_hopping=1 if hopping else 0,
                snr_db=entry.get("snr_inband_db"), snr_definition="inband_snr_v1",
                power_dbfs=entry.get("power_dbfs"), params_json=mapped)
            if not hopping:
                continue
            for hop in hopping_hops:
                if hop.get("session_index") != index:
                    continue
                hop_index = int(hop["hop_index"])
                hop_target = self._insert_target_row(
                    conn, asset_id=asset_id, target_key=f"s{index}.h{hop_index}",
                    scope="hop", parent_target_id=target_id, hop_index=hop_index,
                    for_detection=1, for_amc=0)
                self._insert_target_version_row(
                    conn, target_id=hop_target, source="generator",
                    sample_start=int(round(float(hop["t_start_s"]) * rate)),
                    sample_end=int(round(float(hop["t_end_s"]) * rate)),
                    f_low_hz=hop.get("f_low_hz"), f_high_hz=hop.get("f_high_hz"),
                    nominal_center_hz=hop.get("center_hz"),
                    nominal_bandwidth_hz=hop.get("bandwidth_hz"),
                    signal_type=None, waveform_mode=mode, modulation=None,
                    symbol_rate_baud=None, hop_rate_hz=hop.get("hop_rate_hz"),
                    is_hopping=1, snr_db=hop.get("snr_inband_db"),
                    snr_definition="inband_snr_v1", power_dbfs=hop.get("power_dbfs"),
                    params_json={"hop_index": hop_index, "hop_count": hop.get("hop_count")})

    def _register_placeholder_target(self, conn, asset_id, sample_count):
        """无生成摘要的资产：写“未知”占位目标，不猜测任何物理参数。"""
        target_id = self._insert_target_row(
            conn, asset_id=asset_id, target_key="s0", scope="whole_record",
            parent_target_id=None, hop_index=None, for_detection=1, for_amc=1)
        self._insert_target_version_row(
            conn, target_id=target_id, source="import",
            sample_start=0, sample_end=int(sample_count), f_low_hz=None, f_high_hz=None,
            nominal_center_hz=None, nominal_bandwidth_hz=None, signal_type=None,
            waveform_mode=None, modulation=None, symbol_rate_baud=None, hop_rate_hz=None,
            is_hopping=None, snr_db=None, snr_definition=None, power_dbfs=None,
            params_json=None)

    # ------------------------------------------------------------------ 目标
    def _insert_target_row(self, conn, *, asset_id, target_key, scope,
                           parent_target_id, hop_index, for_detection, for_amc):
        target_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO targets (id, asset_id, target_key, scope, parent_target_id, "
            "hop_index, for_detection, for_amc, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (target_id, asset_id, target_key, scope, parent_target_id, hop_index,
             int(bool(for_detection)), int(bool(for_amc)), utc_now()))
        return target_id

    def add_target(self, asset_id, target_key, scope, *, parent_target_id=None,
                   hop_index=None, for_detection=0, for_amc=0):
        asset = self.get_asset(asset_id)
        key = _clean_text(target_key, "目标键", 64)
        scope = _enum(scope, SCOPES, "目标粒度")
        hop_index = _integer(hop_index, "跳序号", minimum=0)
        if scope == "hop":
            if parent_target_id is None:
                raise ValueError("跳目标必须指定父会话目标")
            if hop_index is None:
                raise ValueError("跳目标必须给出 hop_index")
        else:
            hop_index = None
        if parent_target_id is not None:
            parent = self.get_target(parent_target_id)
            if parent["asset_id"] != asset["id"]:
                raise ValueError("父目标必须属于同一资产")
        with self.connect() as conn:
            try:
                target_id = self._insert_target_row(
                    conn, asset_id=asset_id, target_key=key, scope=scope,
                    parent_target_id=parent_target_id, hop_index=hop_index,
                    for_detection=for_detection, for_amc=for_amc)
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("同一资产内目标键必须唯一") from exc
                raise
        return self.get_target(target_id)

    def get_target(self, target_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        if row is None:
            raise ValueError("目标不存在")
        return dict(row)

    def update_target_flags(self, target_id, *, for_detection=None, for_amc=None):
        self.get_target(target_id)
        with self.connect() as conn:
            if for_detection is not None:
                conn.execute("UPDATE targets SET for_detection=? WHERE id=?",
                             (int(bool(for_detection)), target_id))
            if for_amc is not None:
                conn.execute("UPDATE targets SET for_amc=? WHERE id=?",
                             (int(bool(for_amc)), target_id))
        return self.get_target(target_id)

    def list_targets(self, asset_id, *, with_current=True):
        self.get_asset(asset_id)
        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM targets WHERE asset_id=?", (asset_id,))]
            versions = [dict(row) for row in conn.execute(
                "SELECT * FROM target_versions WHERE target_id IN "
                "(SELECT id FROM targets WHERE asset_id=?)", (asset_id,))]
        current = {}
        for version in versions:
            kept = current.get(version["target_id"])
            if kept is None or version["version_no"] > kept["version_no"]:
                current[version["target_id"]] = version
        rows.sort(key=_target_sort_key)
        if with_current:
            for row in rows:
                row["current"] = current.get(row["id"])
        return rows

    def append_target_version(self, target_id, *, source, note=None, sample_start=None,
                              sample_end=None, f_low_hz=None, f_high_hz=None,
                              center_hz=None, bandwidth_hz=None, nominal_center_hz=None,
                              nominal_bandwidth_hz=None, signal_type=None,
                              waveform_mode=None, modulation=None, symbol_rate_baud=None,
                              hop_rate_hz=None, is_hopping=None, snr_db=None,
                              snr_definition=None, power_dbfs=None, params_json=None):
        target = self.get_target(target_id)
        asset = self.get_asset(target["asset_id"])
        source = _enum(source, VERSION_SOURCES, "来源")
        start = _integer(sample_start, "起始采样点", minimum=0)
        end = _integer(sample_end, "结束采样点", minimum=0)
        if (start is None) != (end is None):
            raise ValueError("时间范围必须同时给出起止采样点")
        if start is not None:
            if end <= start:
                raise ValueError("结束采样点必须大于起始采样点")
            if end > int(asset["sample_count"]):
                raise ValueError("时间范围不能超出资产采样点数")
        low = _number(f_low_hz, "频率下限")
        high = _number(f_high_hz, "频率上限")
        if (low is None) != (high is None):
            raise ValueError("频率范围必须同时给出上下限")
        if low is not None and high <= low:
            raise ValueError("频率上限必须大于下限")
        derived_center = derived_bandwidth = None
        if low is not None:
            derived_center = (low + high) / 2.0
            derived_bandwidth = high - low
            given_center = _number(center_hz, "中心频率")
            given_bandwidth = _number(bandwidth_hz, "带宽", minimum=0.0)
            tolerance = max(1e-6 * max(1.0, abs(derived_center)), 1e-9)
            if given_center is not None and abs(given_center - derived_center) > tolerance:
                raise ValueError("中心频率与频率边界不一致（应为边界派生值）")
            if given_bandwidth is not None and \
                    abs(given_bandwidth - derived_bandwidth) > tolerance:
                raise ValueError("带宽与频率边界不一致（应为边界派生值）")
        note = _optional_text(note, "备注", 500)
        nominal_bandwidth_hz = _number(nominal_bandwidth_hz, "名义带宽", minimum=0.0)
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(version_no) FROM target_versions WHERE target_id=?",
                               (target_id,)).fetchone()
            previous = conn.execute(
                "SELECT id FROM target_versions WHERE target_id=? ORDER BY version_no DESC "
                "LIMIT 1", (target_id,)).fetchone()
            return self._insert_target_version_row(
                conn, target_id=target_id, source=source, note=note,
                sample_start=start, sample_end=end, f_low_hz=low, f_high_hz=high,
                nominal_center_hz=_number(nominal_center_hz, "名义中心频率"),
                nominal_bandwidth_hz=nominal_bandwidth_hz, signal_type=signal_type,
                waveform_mode=waveform_mode, modulation=modulation,
                symbol_rate_baud=_number(symbol_rate_baud, "符号率", minimum=0.0),
                hop_rate_hz=_number(hop_rate_hz, "跳速", minimum=0.0),
                is_hopping=None if is_hopping is None else int(bool(is_hopping)),
                snr_db=_number(snr_db, "SNR"), snr_definition=snr_definition,
                power_dbfs=_number(power_dbfs, "功率"), params_json=params_json,
                version_no=int(row[0] or 0) + 1,
                supersedes_id=previous["id"] if previous else None)

    def _insert_target_version_row(self, conn, *, target_id, source, version_no=None,
                                   supersedes_id=None, note=None, sample_start=None,
                                   sample_end=None, f_low_hz=None, f_high_hz=None,
                                   nominal_center_hz=None, nominal_bandwidth_hz=None,
                                   signal_type=None, waveform_mode=None, modulation=None,
                                   symbol_rate_baud=None, hop_rate_hz=None, is_hopping=None,
                                   snr_db=None, snr_definition=None, power_dbfs=None,
                                   params_json=None):
        if version_no is None:
            row = conn.execute("SELECT MAX(version_no) FROM target_versions "
                               "WHERE target_id=?", (target_id,)).fetchone()
            version_no = int(row[0] or 0) + 1
        # 派生值与 evaluation._rounded 同为 6 位小数：保证目标真值与生成器摘要
        # 在逐位口径上可比（避免 (high-low) 的浮点噪声污染比较与报表）
        derived_center = (round((f_low_hz + f_high_hz) / 2.0, 6)
                          if f_low_hz is not None and f_high_hz is not None else None)
        derived_bandwidth = (round(f_high_hz - f_low_hz, 6)
                             if f_low_hz is not None and f_high_hz is not None else None)
        version_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO target_versions (id, target_id, version_no, supersedes_id, source, "
            "created_at, note, sample_start, sample_end, f_low_hz, f_high_hz, center_hz, "
            "bandwidth_hz, nominal_center_hz, nominal_bandwidth_hz, signal_type, "
            "waveform_mode, modulation, symbol_rate_baud, hop_rate_hz, is_hopping, snr_db, "
            "snr_definition, power_dbfs, params_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (version_id, target_id, version_no, supersedes_id, source, utc_now(), note,
             sample_start, sample_end, f_low_hz, f_high_hz, derived_center,
             derived_bandwidth, nominal_center_hz, nominal_bandwidth_hz, signal_type,
             waveform_mode, modulation, symbol_rate_baud, hop_rate_hz, is_hopping,
             snr_db, snr_definition, power_dbfs, _json_text(params_json, "样式参数")))
        row = conn.execute("SELECT * FROM target_versions WHERE id=?",
                           (version_id,)).fetchone()
        return dict(row)

    def get_target_version(self, version_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM target_versions WHERE id=?",
                               (version_id,)).fetchone()
        if row is None:
            raise ValueError("目标参数版本不存在")
        return dict(row)

    def current_target_version(self, target_id):
        self.get_target(target_id)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM target_versions WHERE target_id=? "
                               "ORDER BY version_no DESC LIMIT 1", (target_id,)).fetchone()
        return dict(row) if row else None

    def list_target_versions(self, target_id):
        self.get_target(target_id)
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM target_versions WHERE target_id=? "
                                "ORDER BY version_no", (target_id,)).fetchall()
        return [dict(row) for row in rows]

    def collection_targets(self, collection_id, *, task=None, with_current=True):
        """集合内全部目标（可按任务适用标记过滤），附资产来源与当前参数版本。"""
        self.get_collection(collection_id)
        flag = None
        if task is not None:
            flag = {"detection": "for_detection", "amc": "for_amc"}.get(
                _enum(task, TASKS, "任务"))
        clause = f"AND t.{flag}=1" if flag else ""
        with self.connect() as conn:
            rows = [dict(row) for row in conn.execute(
                "SELECT t.*, a.name AS asset_name, a.source_kind AS asset_source_kind, "
                "a.origin_group_id AS origin_group_id, a.sample_rate AS asset_sample_rate, "
                "a.sample_count AS asset_sample_count, m.position AS collection_position "
                "FROM targets t JOIN collection_members m ON m.asset_id=t.asset_id "
                "JOIN assets a ON a.id=t.asset_id WHERE m.collection_id=? "
                f"{clause}", (collection_id,))]
            versions = [dict(row) for row in conn.execute(
                "SELECT v.* FROM target_versions v JOIN targets t ON t.id=v.target_id "
                "JOIN collection_members m ON m.asset_id=t.asset_id "
                "WHERE m.collection_id=?", (collection_id,))]
        current = {}
        for version in versions:
            kept = current.get(version["target_id"])
            if kept is None or version["version_no"] > kept["version_no"]:
                current[version["target_id"]] = version
        rows.sort(key=lambda row: (row["collection_position"],
                                   _target_sort_key(row), row["id"]))
        if with_current:
            for row in rows:
                row["current"] = current.get(row["id"])
        return rows

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

    # ------------------------------------------------------------------ 字典与任务集
    def default_taxonomy(self, task, conn=None):
        """项目的默认类别字典：检测为 ``emitter``，AMC 为 A09 六类。"""
        task = _enum(task, TASKS, "任务")
        if task == "detection":
            name, version, classes, mapping = "emitter", "1", ["emitter"], None
        else:
            from .ml.amc import AMC_CLASSES
            name, version, classes = "A09", "1", list(AMC_CLASSES)
            mapping = None
        own = conn is None
        context = self.connect() if own else _null_context(conn)
        with context as active:
            row = active.execute("SELECT * FROM taxonomies WHERE task=? AND name=? AND version=?",
                                 (task, name, version)).fetchone()
            if row:
                return dict(row)
            taxonomy_id = uuid.uuid4().hex
            active.execute("INSERT INTO taxonomies VALUES (?,?,?,?,?,?)",
                           (taxonomy_id, task, name, version,
                            json.dumps(classes, ensure_ascii=False),
                            json.dumps(mapping, ensure_ascii=False) if mapping else None))
            row = active.execute("SELECT * FROM taxonomies WHERE id=?",
                                 (taxonomy_id,)).fetchone()
            return dict(row)

    def create_taxonomy(self, task, name, version, classes, *, mapping=None):
        task = _enum(task, TASKS, "任务")
        name = _clean_text(name, "字典名称", 50)
        version = _clean_text(version, "字典版本", 20)
        if not isinstance(classes, (list, tuple)) or not classes or \
                any(not isinstance(item, str) or not item for item in classes):
            raise ValueError("类别表必须为非空文本列表")
        taxonomy_id = uuid.uuid4().hex
        with self.connect() as conn:
            try:
                conn.execute("INSERT INTO taxonomies VALUES (?,?,?,?,?,?)",
                             (taxonomy_id, task, name, version,
                              json.dumps(list(classes), ensure_ascii=False),
                              _json_text(mapping, "类别映射")))
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("同任务下字典名称与版本必须唯一") from exc
                raise
        return self.get_taxonomy(taxonomy_id)

    def get_taxonomy(self, taxonomy_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM taxonomies WHERE id=?",
                               (taxonomy_id,)).fetchone()
        if row is None:
            raise ValueError("类别字典不存在")
        return dict(row)

    def list_taxonomies(self, task=None):
        with self.connect() as conn:
            if task is None:
                rows = conn.execute("SELECT * FROM taxonomies ORDER BY task, name, version").fetchall()
            else:
                rows = conn.execute("SELECT * FROM taxonomies WHERE task=? "
                                    "ORDER BY name, version", (task,)).fetchall()
        return [dict(row) for row in rows]

    def create_task_set(self, collection_id, task, *, name=None, taxonomy_id=None,
                        label_semantics=None, created_by=None):
        self.get_collection(collection_id)
        task = _enum(task, TASKS, "任务")
        name = _clean_text(name or ("检测标注" if task == "detection" else "AMC 标注"),
                           "标注集名称", 100)
        if task == "detection":
            label_semantics = _enum(label_semantics or "session_v1", DETECTION_SEMANTICS,
                                    "标签粒度")
        else:
            if label_semantics is not None:
                raise ValueError("AMC 标注集不使用 label_semantics")
            label_semantics = None
        taxonomy = (self.get_taxonomy(taxonomy_id) if taxonomy_id is not None
                    else self.default_taxonomy(task))
        if taxonomy["task"] != task:
            raise ValueError("类别字典的任务类型与标注集不一致")
        task_set_id = uuid.uuid4().hex
        with self.connect() as conn:
            try:
                conn.execute("INSERT INTO task_sets VALUES (?,?,?,?,?,?,?,?)",
                             (task_set_id, collection_id, task, name, taxonomy["id"],
                              label_semantics, _optional_text(created_by, "创建者", 100),
                              utc_now()))
            except Exception as exc:
                if "UNIQUE" in str(exc):
                    raise ValueError("同一集合内同任务的标注集名称必须唯一") from exc
                raise
        return self.get_task_set(task_set_id)

    def get_task_set(self, task_set_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM task_sets WHERE id=?",
                               (task_set_id,)).fetchone()
        if row is None:
            raise ValueError("任务标注集不存在")
        return dict(row)

    def list_task_sets(self, collection_id=None, task=None):
        clauses, params = [], []
        if collection_id is not None:
            clauses.append("collection_id=?")
            params.append(collection_id)
        if task is not None:
            clauses.append("task=?")
            params.append(_enum(task, TASKS, "任务"))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM task_sets {where} ORDER BY created_at",
                                params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 覆盖
    def set_asset_coverage(self, task_set_id, asset_id, coverage, *, negative_kind=None,
                           source="manual"):
        task_set = self.get_task_set(task_set_id)
        self.get_asset(asset_id)
        coverage = _enum(coverage, COVERAGE_VALUES, "覆盖状态")
        source = _enum(source, VERSION_SOURCES, "来源")
        negative_kind = _optional_text(negative_kind, "负样本类型", 100)
        if negative_kind is not None and coverage != "complete":
            raise ValueError("negative_kind 仅在 coverage=complete 时有意义")
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(revision_no) FROM asset_coverage WHERE "
                               "task_set_id=? AND asset_id=?",
                               (task_set_id, asset_id)).fetchone()
            revision = int(row[0] or 0) + 1
            conn.execute("INSERT INTO asset_coverage VALUES (?,?,?,?,?,?,?,?)",
                         (uuid.uuid4().hex, task_set_id, asset_id, revision, coverage,
                          negative_kind, source, utc_now()))
        return self.get_asset_coverage(task_set_id, asset_id)

    def get_asset_coverage(self, task_set_id, asset_id):
        self.get_task_set(task_set_id)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM asset_coverage WHERE task_set_id=? AND "
                               "asset_id=? ORDER BY revision_no DESC LIMIT 1",
                               (task_set_id, asset_id)).fetchone()
        return dict(row) if row else None

    def list_asset_coverage(self, task_set_id):
        self.get_task_set(task_set_id)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT cov.* FROM asset_coverage cov JOIN ("
                "SELECT asset_id, MAX(revision_no) AS rn FROM asset_coverage "
                "WHERE task_set_id=? GROUP BY asset_id) m ON m.asset_id=cov.asset_id "
                "AND m.rn=cov.revision_no WHERE cov.task_set_id=?",
                (task_set_id, task_set_id)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 标签
    def _resolve_label_semantics(self, task_set, label_semantics):
        if task_set["task"] != "detection":
            raise ValueError("AMC 标注不使用 label_semantics")
        semantics = _enum(label_semantics or task_set["label_semantics"] or "session_v1",
                          DETECTION_SEMANTICS, "标签粒度")
        if task_set["label_semantics"] and semantics != task_set["label_semantics"]:
            raise ValueError("标签粒度与任务标注集声明不一致")
        return semantics

    def append_detection_label(self, task_set_id, target_id, *, source,
                               class_name=None, target_version_id=None, include=True,
                               note=None, label_semantics=None):
        task_set = self.get_task_set(task_set_id)
        if task_set["task"] != "detection":
            raise ValueError("该任务标注集不是检测集")
        semantics = self._resolve_label_semantics(task_set, label_semantics)
        target = self.get_target(target_id)
        version = (self.get_target_version(target_version_id) if target_version_id
                   else self.current_target_version(target_id))
        if version is None:
            raise ValueError("目标缺少参考参数版本，无法标注")
        if version["target_id"] != target_id:
            raise ValueError("参考参数版本与目标不匹配")
        taxonomy = self.get_taxonomy(task_set["taxonomy_id"])
        classes = json.loads(taxonomy["classes_json"])
        include = int(bool(include))
        class_name = _optional_text(class_name, "类别名称", 100)
        if include:
            if class_name is None and len(classes) == 1:
                class_name = classes[0]
            if class_name not in classes:
                raise ValueError(f"检测类别应为字典内名称：{' / '.join(classes)}")
        else:
            class_name = None
        return self._append_label("detection", task_set_id, target_id,
                                  source=source, note=note, class_name=class_name,
                                  target_version_id=version["id"],
                                  extra={"label_semantics": semantics,
                                         "include": include})

    def append_amc_label(self, task_set_id, target_id, *, source, class_state,
                         class_name=None, window_start=None, window_end=None,
                         analysis_center_hz=None, analysis_bandwidth_hz=None,
                         target_version_id=None, note=None):
        task_set = self.get_task_set(task_set_id)
        if task_set["task"] != "amc":
            raise ValueError("该任务标注集不是 AMC 集")
        class_state = _enum(class_state, CLASS_STATES, "类别状态")
        target = self.get_target(target_id)
        version = (self.get_target_version(target_version_id) if target_version_id
                   else self.current_target_version(target_id))
        if version is None:
            raise ValueError("目标缺少参考参数版本，无法标注")
        if version["target_id"] != target_id:
            raise ValueError("参考参数版本与目标不匹配")
        taxonomy = self.get_taxonomy(task_set["taxonomy_id"])
        classes = json.loads(taxonomy["classes_json"])
        class_name = _optional_text(class_name, "类别名称", 100)
        if class_state == "known":
            if class_name not in classes:
                raise ValueError(f"已知类别应为字典内名称：{' / '.join(classes)}")
        elif class_state == "out_of_taxonomy":
            if class_name is None:
                raise ValueError("字典外类别必须保留原始类名")
            if class_name in classes:
                raise ValueError("类名在字典内，应记为 known")
        else:
            if class_name is not None:
                raise ValueError("unknown 状态不携带类别名称")
        start = _integer(window_start, "窗口起点", minimum=0)
        end = _integer(window_end, "窗口终点", minimum=0)
        if (start is None) != (end is None):
            raise ValueError("提取范围必须同时给出起止采样点")
        if start is not None:
            if end <= start:
                raise ValueError("提取窗口终点必须大于起点")
            asset = self.get_asset(target["asset_id"])
            if end > int(asset["sample_count"]):
                raise ValueError("提取范围不能超出资产采样点数")
        return self._append_label("amc", task_set_id, target_id, source=source, note=note,
                                  class_name=class_name, target_version_id=version["id"],
                                  extra={"class_state": class_state,
                                         "window_start": start, "window_end": end,
                                         "analysis_center_hz": _number(
                                             analysis_center_hz, "分析中心频率"),
                                         "analysis_bandwidth_hz": _number(
                                             analysis_bandwidth_hz, "分析带宽", minimum=0.0)})

    def _append_label(self, task, task_set_id, target_id, *, source, note, class_name,
                      target_version_id, extra):
        source = _enum(source, VERSION_SOURCES, "来源")
        note = _optional_text(note, "备注", 500)
        table = LABEL_TABLES[task]
        columns = {"detection": ("target_version_id", "label_semantics", "class_name",
                                 "include", "note"),
                   "amc": ("target_version_id", "class_state", "class_name",
                           "window_start", "window_end", "analysis_center_hz",
                           "analysis_bandwidth_hz", "note")}[task]
        values = {"target_version_id": target_version_id, "class_name": class_name,
                  "note": note, **extra}
        with self.connect() as conn:
            row = conn.execute(f"SELECT MAX(revision_no) FROM {table} WHERE "
                               "task_set_id=? AND target_id=?",
                               (task_set_id, target_id)).fetchone()
            previous = conn.execute(f"SELECT id FROM {table} WHERE task_set_id=? AND "
                                    "target_id=? ORDER BY revision_no DESC LIMIT 1",
                                    (task_set_id, target_id)).fetchone()
            label_id = uuid.uuid4().hex
            placeholders = ",".join("?" for _ in range(7 + len(columns)))
            conn.execute(
                f"INSERT INTO {table} (id, task_set_id, target_id, revision_no, "
                f"supersedes_id, source, created_at, {', '.join(columns)}) "
                f"VALUES ({placeholders})",
                (label_id, task_set_id, target_id, int(row[0] or 0) + 1,
                 previous["id"] if previous else None, source, utc_now(),
                 *[values.get(column) for column in columns]))
        return self.get_label(task, label_id)

    def get_label(self, task, label_id):
        task = _enum(task, TASKS, "任务")
        with self.connect() as conn:
            row = conn.execute(f"SELECT * FROM {LABEL_TABLES[task]} WHERE id=?",
                               (label_id,)).fetchone()
        if row is None:
            raise ValueError("标签不存在")
        return dict(row)

    def current_label(self, task, task_set_id, target_id):
        task = _enum(task, TASKS, "任务")
        table = LABEL_TABLES[task]
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE task_set_id=? AND target_id=? "
                "ORDER BY revision_no DESC LIMIT 1", (task_set_id, target_id)).fetchone()
        return dict(row) if row else None

    def current_labels(self, task, task_set_id, *, include_all=False):
        """当前（最新 revision）标签列表；``include_all=True`` 时含 include=0 的负标注。"""
        task = _enum(task, TASKS, "任务")
        table = LABEL_TABLES[task]
        clause = "" if include_all else (
            " AND l.include=1" if task == "detection" else "")
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT l.* FROM {table} l JOIN (SELECT target_id, MAX(revision_no) AS rn "
                f"FROM {table} WHERE task_set_id=? GROUP BY target_id) m "
                "ON m.target_id=l.target_id AND m.rn=l.revision_no "
                f"WHERE l.task_set_id=?{clause}", (task_set_id, task_set_id)).fetchall()
        return [dict(row) for row in rows]

    def label_revisions(self, task, task_set_id, target_id):
        task = _enum(task, TASKS, "任务")
        table = LABEL_TABLES[task]
        with self.connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE task_set_id=? AND target_id=? "
                "ORDER BY revision_no", (task_set_id, target_id)).fetchall()
        return [dict(row) for row in rows]

    def asset_label_status(self, asset_id):
        """资产内每个目标是否有标签（跨全部任务标注集，任一条即算）。"""
        self.get_asset(asset_id)
        with self.connect() as conn:
            detection = {row[0] for row in conn.execute(
                "SELECT DISTINCT target_id FROM detection_labels WHERE target_id IN "
                "(SELECT id FROM targets WHERE asset_id=?)", (asset_id,))}
            amc = {row[0] for row in conn.execute(
                "SELECT DISTINCT target_id FROM amc_labels WHERE target_id IN "
                "(SELECT id FROM targets WHERE asset_id=?)", (asset_id,))}
        return {"detection": detection, "amc": amc}

    def adopt_label(self, task, task_set_id, target_id, *, source_run_id=None, **fields):
        """采纳算法结论：以 ``source=algorithm`` 追加新版本（不覆盖历史）。"""
        key = "source"
        if key in fields:
            raise ValueError("请勿手工指定来源；采纳固定为 algorithm")
        fields = dict(fields)
        if source_run_id is not None:
            note = fields.get("note")
            fields["note"] = ((note + " · ") if note else "") + f"来源运行 {source_run_id}"
        if task == "detection":
            return self.append_detection_label(task_set_id, target_id, source="algorithm",
                                               **fields)
        return self.append_amc_label(task_set_id, target_id, source="algorithm", **fields)

    # ------------------------------------------------------------------ 进度统计
    def task_set_progress(self, task_set_id):
        """标注完成率：目标总数、已标注、待标注、其中重新标注过的数量，以及覆盖度。"""
        task_set = self.get_task_set(task_set_id)
        task, collection_id = task_set["task"], task_set["collection_id"]
        table, flag = LABEL_TABLES[task], ("for_detection" if task == "detection"
                                           else "for_amc")
        with self.connect() as conn:
            targets = conn.execute(
                f"SELECT t.id FROM targets t JOIN collection_members m "
                f"ON m.asset_id=t.asset_id WHERE m.collection_id=? AND t.{flag}=1",
                (collection_id,)).fetchall()
            target_ids = [row[0] for row in targets]
            labeled, reannotated = set(), set()
            if target_ids:
                placeholders = ",".join("?" for _ in target_ids)
                rows = conn.execute(
                    f"SELECT target_id, MAX(revision_no) FROM {table} WHERE task_set_id=? "
                    f"AND target_id IN ({placeholders}) GROUP BY target_id",
                    (task_set_id, *target_ids)).fetchall()
                for target_id, revision in rows:
                    labeled.add(target_id)
                    if int(revision or 1) > 1:
                        reannotated.add(target_id)
            assets = conn.execute(
                "SELECT COUNT(*) FROM collection_members WHERE collection_id=?",
                (collection_id,)).fetchone()[0]
            coverage_rows = conn.execute(
                "SELECT cov.coverage, COUNT(*) FROM asset_coverage cov JOIN ("
                "SELECT asset_id, MAX(revision_no) AS rn FROM asset_coverage "
                "WHERE task_set_id=? GROUP BY asset_id) m ON m.asset_id=cov.asset_id "
                "AND m.rn=cov.revision_no WHERE cov.task_set_id=? GROUP BY cov.coverage",
                (task_set_id, task_set_id)).fetchall()
        coverage = {"complete": 0, "partial": 0, "unknown": 0}
        for name, count in coverage_rows:
            coverage[str(name)] = int(count)
        coverage["unmarked"] = max(int(assets) - sum(
            coverage[key] for key in ("complete", "partial", "unknown")), 0)
        return {"id": task_set_id, "task": task, "name": task_set["name"],
                "label_semantics": task_set["label_semantics"],
                "targets": len(target_ids), "labeled": len(labeled),
                "pending": len(target_ids) - len(labeled),
                "reannotated": len(reannotated),
                "assets": int(assets), "coverage": coverage}

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

    # ------------------------------------------------------------------ 生成配方
    def save_recipe(self, name, engine, recipe, *, created_by=None):
        from .recipes import validate_recipe

        name = _clean_text(name, "配方名称", 100)
        engine = _clean_text(engine, "生成引擎", 50)
        validate_recipe(recipe)
        payload = json.dumps(_json_safe(recipe), ensure_ascii=False, sort_keys=True,
                             allow_nan=False)
        recipe_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute("INSERT INTO recipes VALUES (?,?,?,?,?,?,?)",
                         (recipe_id, name, engine, payload,
                          hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                          _optional_text(created_by, "创建者", 100), utc_now()))
        return self.get_recipe(recipe_id)

    def ensure_recipe(self, name, engine, recipe, *, created_by=None):
        """按内容哈希取配方；同一份配方（键序无关）重复生成时复用，不堆积副本。"""
        payload = json.dumps(_json_safe(recipe), ensure_ascii=False, sort_keys=True,
                             allow_nan=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM recipes WHERE recipe_sha256=? "
                               "ORDER BY created_at LIMIT 1", (digest,)).fetchone()
        if row is not None:
            return dict(row)
        return self.save_recipe(name, engine, recipe, created_by=created_by)

    def set_collection_recipe(self, collection_id, recipe_id):
        """给还没有配方的集合绑定配方（已绑定则保持原配方，返回 False）。"""
        collection = self.get_collection(collection_id)
        self.get_recipe(recipe_id)
        if collection["recipe_id"]:
            return False
        with self.connect() as conn:
            conn.execute("UPDATE collections SET recipe_id=?, updated_at=? WHERE id=?",
                         (recipe_id, utc_now(), collection_id))
        return True

    def get_recipe(self, recipe_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM recipes WHERE id=?", (recipe_id,)).fetchone()
        if row is None:
            raise ValueError("生成配方不存在")
        return dict(row)

    def list_recipes(self, limit=200):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM recipes ORDER BY created_at DESC, id "
                                "LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def write_coverage_cache(self, collection_id, recipe_id, stats):
        """写入覆盖度缓存（可重建）：``stats`` 为 ``{(axis, bin): count}``。"""
        with self.connect() as conn:
            conn.execute("DELETE FROM coverage_cache WHERE collection_id=? AND recipe_id=?",
                         (collection_id, recipe_id or ""))
            conn.executemany("INSERT INTO coverage_cache VALUES (?,?,?,?,?)",
                             [(collection_id, recipe_id or "", axis, bin_name, int(count))
                              for (axis, bin_name), count in stats.items()])

    def read_coverage_cache(self, collection_id, recipe_id=None):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM coverage_cache WHERE collection_id=? AND "
                                "recipe_id=?", (collection_id, recipe_id or "")).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------ 数据版本 / 实验 / 评估
    def add_dataset_version(self, task_set_id, *, status, manifest_path=None,
                            manifest_sha256=None, sample_count=0, asset_count=0,
                            split=None, preprocessing=None, taxonomy_snapshot=None,
                            card=None):
        self.get_task_set(task_set_id)
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(version_no) FROM dataset_versions WHERE "
                               "task_set_id=?", (task_set_id,)).fetchone()
            version_no = int(row[0] or 0) + 1
            version_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO dataset_versions (id, task_set_id, version_no, manifest_path, "
                "manifest_sha256, status, sample_count, asset_count, split_json, "
                "preprocessing_json, taxonomy_snapshot_json, card_json, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (version_id, task_set_id, version_no, manifest_path, manifest_sha256,
                 status, int(sample_count), int(asset_count), _json_text(split, "划分"),
                 _json_text(preprocessing, "预处理"), _json_text(taxonomy_snapshot, "字典快照"),
                 _json_text(card, "卡片"), utc_now()))
        return self.get_dataset_version(version_id)

    def update_dataset_version(self, dataset_version_id, **fields):
        allowed = {"status", "manifest_path", "manifest_sha256", "sample_count",
                   "asset_count", "split", "preprocessing", "taxonomy_snapshot", "card"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"不支持更新的字段：{' / '.join(sorted(unknown))}")
        mapping = {"split": "split_json", "preprocessing": "preprocessing_json",
                   "taxonomy_snapshot": "taxonomy_snapshot_json", "card": "card_json"}
        self.get_dataset_version(dataset_version_id)
        sets, params = [], []
        for key, value in fields.items():
            sets.append(f"{mapping.get(key, key)}=?")
            params.append(_json_text(value, key) if key in mapping else value)
        params.append(dataset_version_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE dataset_versions SET {', '.join(sets)} WHERE id=?", params)
        return self.get_dataset_version(dataset_version_id)

    def get_dataset_version(self, dataset_version_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM dataset_versions WHERE id=?",
                               (dataset_version_id,)).fetchone()
        if row is None:
            raise ValueError("数据版本不存在")
        return dict(row)

    def list_dataset_versions(self, task_set_id=None, *, status=None, limit=200):
        clauses, params = [], []
        if task_set_id is not None:
            clauses.append("task_set_id=?")
            params.append(task_set_id)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM dataset_versions {where} "
                                "ORDER BY task_set_id, version_no DESC LIMIT ?",
                                [*params, limit]).fetchall()
        return [dict(row) for row in rows]

    def register_experiment(self, task, config, *, dataset_version_id=None, status="created",
                            output_path=None, model_sha256=None, experiment_id=None,
                            finished_at=None, created_at=None):
        task = _clean_text(task, "任务名", 50)
        experiment_id = experiment_id or uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)",
                (experiment_id, task, dataset_version_id, json.dumps(config,
                 ensure_ascii=False, allow_nan=False), status, output_path, model_sha256,
                 created_at or utc_now(), finished_at))
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?",
                               (experiment_id,)).fetchone()
        if row is None:
            raise ValueError("实验记录不存在")
        return dict(row)

    def find_experiment(self, experiment_id):
        """按 ID 查找实验；不存在返回 ``None``（迁移时用于幂等判断）。"""
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?",
                               (experiment_id,)).fetchone()
        return dict(row) if row else None

    def list_experiments(self, task=None, dataset_version_id=None, limit=200):
        clauses, params = [], []
        if task is not None:
            clauses.append("task=?")
            params.append(task)
        if dataset_version_id is not None:
            clauses.append("dataset_version_id=?")
            params.append(dataset_version_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM experiments {where} "
                                "ORDER BY created_at DESC LIMIT ?", [*params, limit]).fetchall()
        return [dict(row) for row in rows]

    def register_evaluation(self, dataset_version_id, contract, config, metrics, *,
                            experiment_id=None, model_sha256=None, result_path=None):
        self.get_dataset_version(dataset_version_id)
        evaluation_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
                (evaluation_id, dataset_version_id, experiment_id, model_sha256, contract,
                 json.dumps(config, ensure_ascii=False, allow_nan=False),
                 json.dumps(metrics, ensure_ascii=False, allow_nan=False),
                 result_path, utc_now()))
        return self.get_evaluation(evaluation_id)

    def get_evaluation(self, evaluation_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM evaluations WHERE id=?",
                               (evaluation_id,)).fetchone()
        if row is None:
            raise ValueError("评估记录不存在")
        return dict(row)

    def list_evaluations(self, dataset_version_id=None, contract=None, limit=200):
        clauses, params = [], []
        if dataset_version_id is not None:
            clauses.append("dataset_version_id=?")
            params.append(dataset_version_id)
        if contract is not None:
            clauses.append("contract=?")
            params.append(contract)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM evaluations {where} "
                                "ORDER BY created_at DESC LIMIT ?", [*params, limit]).fetchall()
        return [dict(row) for row in rows]


class _null_context:
    """把已有连接包装成上下文（不关闭、不提交，由外层负责）。"""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False


class ShardWriter:
    """把多条录制连续写入一个分片文件；封存后不可再写、不可改。

    生成或批量导入的产物：每个资产在 ``assets`` 里仍是一行，物理数据位于
    ``assets/shards/<id>.bin`` 的 ``[offset, offset+length)`` 采样点区间，
    读取时按 complex64 内存映射定位。单条导入仍用独立 NPY（保持现状）。
    """

    def __init__(self, workspace, *, name="", created_by=None):
        self.workspace = workspace
        self.id = uuid.uuid4().hex
        self.name = str(name or "")
        folder = workspace.root / "assets" / "shards"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{self.id}.bin"
        self.relative = self.path.relative_to(workspace.root).as_posix()
        self.stream = self.path.open("wb")
        self.offset = 0
        self.count = 0
        self.sealed = False
        self.created_by = created_by
        with workspace.connect() as conn:
            conn.execute("INSERT INTO asset_shards (id, name, path, sha256, record_count, "
                         "size_bytes, created_at, sealed_at) VALUES (?,?,?,'',0,0,?,NULL)",
                         (self.id, self.name, self.relative, utc_now()))

    def append(self, samples, sample_rate, name, source="generated", *, metadata=None,
               source_kind=None, origin_group_id=None, parent_asset_id=None,
               rf_center_hz=None, capture_started_at=None):
        if self.sealed:
            raise ValueError("分片已封存，不能继续写入")
        data = validate_samples(samples)
        rate = validate_rate(sample_rate)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("名称应为 1～200 个字符")
        inferred_kind, inferred_parent = _classify_source(source)
        kind = _enum(source_kind or inferred_kind, SOURCE_KINDS, "来源分类")
        payload = data.tobytes()
        digest = hashlib.sha256(payload).hexdigest()
        asset_id = uuid.uuid4().hex
        self.stream.write(payload)
        self.stream.flush()  # 资产一经登记就必须可读：不把数据留在写缓冲里
        with self.workspace.connect() as conn:
            self.workspace._insert_asset(
                conn, asset_id=asset_id, name=name, relative=self.relative, digest=digest,
                rate=rate, count=data.size, source=source, source_kind=kind,
                sample_kind="complex", dtype="complex64", storage_kind="shard",
                shard_id=self.id, shard_offset=self.offset, shard_length=data.size,
                origin_group_id=origin_group_id,
                parent_asset_id=parent_asset_id or inferred_parent,
                rf_center_hz=_number(rf_center_hz, "射频中心"),
                capture_started_at=_optional_text(capture_started_at, "采集时间", 64),
                created_by=_optional_text(self.created_by, "创建者", 100))
            if metadata is not None:
                conn.execute("INSERT INTO asset_metadata VALUES (?,?)",
                             (asset_id, json.dumps(metadata, ensure_ascii=False,
                                                   allow_nan=False)))
                generation = metadata.get("generation") if isinstance(metadata, dict) else None
                if isinstance(generation, dict):
                    self.workspace._register_generation_targets(
                        conn, asset_id, generation, int(data.size), float(rate))
        self.offset += int(data.size)
        self.count += 1
        return self.workspace.get_asset(asset_id)

    def seal(self):
        """封存分片：写入总体哈希与统计，之后可读不可写。"""
        if self.sealed:
            return self.workspace.get_shard(self.id)
        self.stream.flush()
        self.stream.close()
        self.sealed = True
        digest = file_digest(self.path)
        size = self.path.stat().st_size
        with self.workspace.connect() as conn:
            conn.execute("UPDATE asset_shards SET sha256=?, record_count=?, size_bytes=?, "
                         "sealed_at=? WHERE id=?",
                         (digest, self.count, size, utc_now(), self.id))
        return self.workspace.get_shard(self.id)

    def abort(self):
        """放弃空分片；已写入资产的分片不允许放弃（先导出或保留）。"""
        if self.sealed:
            raise ValueError("分片已封存，无法放弃")
        self.stream.close()
        self.sealed = True
        if self.count:
            raise ValueError("分片内已有资产记录，不能直接放弃；请保留并封存")
        self.path.unlink(missing_ok=True)
        with self.workspace.connect() as conn:
            conn.execute("DELETE FROM asset_shards WHERE id=?", (self.id,))
