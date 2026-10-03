"""数据分析工作区 schema v3 的常量、列扩展与 DDL。

数据模型权威文档：docs/数据库设计.md。本模块只放静态数据，不导入领域实现；
供 ``signal_analysis.data`` 的 Workspace 与迁移逻辑共用。
"""

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

#: 生成器样式 → 规范调制名；跳频样式没有单一调制名（NULL = 未知）。
_MODULATION_NAMES = {"am": "AM", "fm": "FM", "ssb": "SSB", "ask2": "2ASK",
                     "qpsk": "QPSK", "qam16": "16QAM", "qam64": "64QAM"}

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
