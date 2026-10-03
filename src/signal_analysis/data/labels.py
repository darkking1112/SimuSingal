"""类别字典、任务标注集、覆盖与标签（LabelMixin）。

标注追加式版本化：新版本 revision_no 递增并以 supersedes_id 指向旧版本，
旧行不改不删；“当前标签”= 最大 revision_no。由
:class:`signal_analysis.data.workspace.Workspace` 组合使用。
"""

import json
import uuid

from common.storage import utc_now
from ..storage.schema import (CLASS_STATES, COVERAGE_VALUES, DETECTION_SEMANTICS,
                              LABEL_TABLES, TASKS, VERSION_SOURCES)
from ..storage.utils import (_clean_text, _enum, _integer, _json_text, _null_context,
                             _number, _optional_text)


class LabelMixin:
    """字典/任务集、覆盖度、检测与 AMC 标签及标注进度。"""

    # ------------------------------------------------------------------ 字典与任务集
    def default_taxonomy(self, task, conn=None):
        """项目的默认类别字典：检测为 ``emitter``，AMC 为 A09 六类。"""
        task = _enum(task, TASKS, "任务")
        if task == "detection":
            name, version, classes, mapping = "emitter", "1", ["emitter"], None
        else:
            from ..contracts.amc import AMC_CLASSES
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
        target = self.get_target(target_id)
        version = (self.get_target_version(target_version_id) if target_version_id
                   else self.current_target_version(target_id))
        if version is None:
            raise ValueError("目标缺少参考参数版本，无法标注")
        if version["target_id"] != target_id:
            raise ValueError("参考参数版本与目标不匹配")
        extra = self._amc_label_extra(task_set, target, class_state, class_name,
                                      window_start, window_end, analysis_center_hz,
                                      analysis_bandwidth_hz)
        return self._append_label("amc", task_set_id, target_id, source=source, note=note,
                                  class_name=extra.pop("class_name"),
                                  target_version_id=version["id"], extra=extra)

    def append_amc_annotation(self, task_set_id, target_id, *, source, class_state,
                              class_name=None, note=None, label_note=None,
                              window_start=None, window_end=None, analysis_center_hz=None,
                              analysis_bandwidth_hz=None, version_note=None,
                              **version_fields):
        """同一事务写目标参数版本 + AMC 标签；校验失败不留下任何写入。

        「信号标注」子页同时改参数与类别：分两次写会出现「标签校验失败但参数版本已落盘」
        的半成品（第 11 节），因此这里先校验标签字段，再在同一个连接里写版本与标签。
        """
        task_set = self.get_task_set(task_set_id)
        if task_set["task"] != "amc":
            raise ValueError("该任务标注集不是 AMC 集")
        target = self.get_target(target_id)
        extra = self._amc_label_extra(task_set, target, class_state, class_name,
                                      window_start, window_end, analysis_center_hz,
                                      analysis_bandwidth_hz)
        with self.connect() as conn:
            version = self._append_target_version(conn, target_id, source=source,
                                                 note=version_note, **version_fields)
            label = self._insert_label_row(
                conn, "amc", task_set_id, target_id, source=source, note=label_note,
                class_name=extra.pop("class_name"), target_version_id=version["id"],
                extra=extra)
        return version, label

    def _amc_label_extra(self, task_set, target, class_state, class_name, window_start,
                         window_end, analysis_center_hz, analysis_bandwidth_hz):
        """校验并归一化 AMC 标签字段（不写库）；失败即抛，调用方不会留下写入。"""
        class_state = _enum(class_state, CLASS_STATES, "类别状态")
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
        return {"class_name": class_name, "class_state": class_state,
                "window_start": start, "window_end": end,
                "analysis_center_hz": _number(analysis_center_hz, "分析中心频率"),
                "analysis_bandwidth_hz": _number(analysis_bandwidth_hz, "分析带宽",
                                                 minimum=0.0)}

    def _append_label(self, task, task_set_id, target_id, *, source, note, class_name,
                      target_version_id, extra):
        with self.connect() as conn:
            return self._insert_label_row(conn, task, task_set_id, target_id,
                                          source=source, note=note, class_name=class_name,
                                          target_version_id=target_version_id, extra=extra)

    def _insert_label_row(self, conn, task, task_set_id, target_id, *, source, note,
                          class_name, target_version_id, extra):
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
        inserted = conn.execute(f"SELECT * FROM {table} WHERE id=?", (label_id,)).fetchone()
        return dict(inserted)

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
