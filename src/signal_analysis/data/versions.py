"""生成配方、数据版本、实验与评估（VersionMixin）。

配方按内容哈希复用；数据版本挂在任务标注集下；实验/评估记录训练与验证
链条。由 :class:`signal_analysis.data.workspace.Workspace` 组合使用。
"""

import hashlib
import json
import uuid

from common.storage import utc_now
from ..storage.utils import _clean_text, _json_safe, _json_text, _optional_text


class VersionMixin:
    """生成配方、覆盖度缓存与数据版本/实验/评估记录。"""

    # ------------------------------------------------------------------ 生成配方
    def save_recipe(self, name, engine, recipe, *, created_by=None):
        from ..algorithms.generation.recipes import validate_recipe

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
