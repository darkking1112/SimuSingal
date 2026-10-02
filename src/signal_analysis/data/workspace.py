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
from ..core_api import validate_rate, validate_samples

from ..storage.schema import (SOURCE_KINDS, SAMPLE_KINDS, STORAGE_KINDS, SCOPES,
                              VERSION_SOURCES, COVERAGE_VALUES, CLASS_STATES,
                              DETECTION_SEMANTICS, TASKS, LABEL_TABLES,
                              _ASSET_COLUMNS_V3, _MODULATION_NAMES, _SCHEMA_V3)
from ..storage.utils import (_TARGET_KEY, _bin_specs, _classify_source, _clean_text,
                             _ensure_columns, _enum, _fmt_edge, _integer, _json_safe,
                             _json_text, _null_context, _number, _optional_text,
                             _target_sort_key)
from ..storage.migrations import MigrationMixin
from .assets import AssetMixin, ShardWriter  # ShardWriter re-export 保持 data.workspace 旧导入面
from .targets import TargetMixin
from .collections import CollectionMixin
from .labels import LabelMixin
from .versions import VersionMixin

#: schema v3 常量与 DDL 已迁移到 ``..storage.schema``（见本文件顶部导入）。


class Workspace(AssetMixin, TargetMixin, CollectionMixin, LabelMixin, VersionMixin,
                MigrationMixin, RunWorkspace):
    project = "signal_analysis"

    def __init__(self, root):
        super().__init__(root)
        (self.root / "assets").mkdir(exist_ok=True)


