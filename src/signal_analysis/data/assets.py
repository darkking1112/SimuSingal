"""资产分片写入器：批量生成/导入连续写入分片，封存后不可变。"""

import hashlib
import json
import uuid

import numpy as np

from common.storage import file_digest, utc_now
from ..core_api import validate_rate, validate_samples
from ..storage.schema import SOURCE_KINDS
from ..storage.utils import _classify_source, _enum, _number, _optional_text
from .io import (BINARY_DTYPES, FORMAT_EXTENSIONS, read_samples,
                 resolve_storage_format, write_samples)


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


class AssetMixin:
    """资产（独立文件 / 分片）的登记、检索、读取与备注。"""

    def _insert_asset(self, conn, *, asset_id, name, relative, digest, rate, count,
                      source, source_kind, sample_kind, dtype, storage_kind,
                      storage_format="npy", endian="little",
                      shard_id=None, shard_offset=None, shard_length=None,
                      origin_group_id=None, parent_asset_id=None, rf_center_hz=None,
                      capture_started_at=None, created_by=None):
        now = utc_now()
        conn.execute(
            "INSERT INTO assets (id, name, path, sha256, sample_rate, sample_count, "
            "created_at, source, label, source_kind, sample_kind, dtype, storage_kind, "
            "storage_format, endian, shard_id, shard_offset, shard_length, "
            "origin_group_id, parent_asset_id, rf_center_hz, capture_started_at, "
            "created_by, updated_at, archived_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
            (asset_id, name, relative, digest, rate, count, now, str(source), "",
             source_kind, sample_kind, dtype, storage_kind, storage_format, endian,
             shard_id, shard_offset, shard_length, origin_group_id or asset_id,
             parent_asset_id, rf_center_hz, capture_started_at, created_by, now))

    def _remove_asset_files(self, asset_id):
        """删除本次写入的资产文件，含 SigMF 对文件与原子写残留。"""
        for path in (self.root / "assets").glob(f"{asset_id}.*"):
            path.unlink(missing_ok=True)

    def add_samples(self, samples, sample_rate, name, source="generated", *, metadata=None,
                    source_kind=None, origin_group_id=None, parent_asset_id=None,
                    rf_center_hz=None, capture_started_at=None, created_by=None,
                    storage_format="npy", endian="little"):
        """新增一条独立资产；带生成摘要时同时登记目标与参考参数版本。

        ``storage_format`` 决定资产文件的编码（``npy``/``csv``/``iq16``/``iq32``/``sigmf``），
        ``endian`` 只对交织 IQ 二进制生效；逻辑 dtype 一律按 complex64 登记。
        """
        data = validate_samples(samples)
        rate = validate_rate(sample_rate)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("名称应为 1～200 个字符")
        fmt, order = resolve_storage_format(storage_format, endian)
        inferred_kind, inferred_parent = _classify_source(source)
        kind = _enum(source_kind or inferred_kind, SOURCE_KINDS, "来源分类")
        parent = parent_asset_id or inferred_parent
        if parent is not None:
            with self.connect() as conn:
                exists = conn.execute("SELECT 1 FROM assets WHERE id=?", (parent,)).fetchone()
            if not exists:
                raise ValueError("父资产不存在")
        generation = metadata.get("generation") if isinstance(metadata, dict) else None
        asset_id = uuid.uuid4().hex
        destination = self.root / "assets" / f"{asset_id}{FORMAT_EXTENSIONS[fmt]}"
        try:
            written = write_samples(destination, data, fmt, order, sample_rate=rate,
                                    description=name, generation=generation)
            # SigMF 的载荷在数据文件上，元数据文件由同名后缀推导。
            payload = written if fmt != "sigmf" else written.with_suffix(".sigmf-data")
            relative = payload.relative_to(self.root).as_posix()
            with self.connect() as conn:
                self._insert_asset(
                    conn, asset_id=asset_id, name=name, relative=relative,
                    digest=file_digest(payload), rate=rate, count=data.size,
                    source=source, source_kind=kind, sample_kind="complex",
                    dtype="complex64", storage_kind="file",
                    storage_format=fmt, endian=order,
                    origin_group_id=origin_group_id, parent_asset_id=parent,
                    rf_center_hz=_number(rf_center_hz, "射频中心"),
                    capture_started_at=_optional_text(capture_started_at, "采集时间", 64),
                    created_by=_optional_text(created_by, "创建者", 100))
                if metadata is not None:
                    conn.execute("INSERT INTO asset_metadata VALUES (?,?)",
                                 (asset_id, json.dumps(metadata, ensure_ascii=False,
                                                       allow_nan=False)))
                    if isinstance(generation, dict):
                        self._register_generation_targets(conn, asset_id, generation,
                                                          int(data.size), float(rate))
        except BaseException:
            self._remove_asset_files(asset_id)
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

    #: 侧栏标签过滤器：任务 → 该任务的标签表。判定为「资产有目标但没有任何该任务标签」。
    MISSING_LABEL_FILTERS = {"detection": "detection_labels", "amc": "amc_labels"}

    def _asset_clauses(self, search="", *, collection_id=None, scattered=False,
                       include_archived=False, missing_labels=None):
        """资产筛选条件列表（不含 WHERE 前缀）与参数。

        ``missing_labels`` 取 ``"detection"`` 或 ``"amc"``：只保留「有目标但没有任何该任务
        标签」的资产（标签按任意标注集判定，与侧栏只读状态口径一致）；没有目标的资产
        （纯噪声负样本、已确认无信号）不进入结果，其完整性由覆盖度与完成动作表达。
        """
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
        if missing_labels is not None:
            if missing_labels not in self.MISSING_LABEL_FILTERS:
                raise ValueError("标签过滤只能是 detection 或 amc")
            table = self.MISSING_LABEL_FILTERS[missing_labels]
            clauses.append(
                "EXISTS (SELECT 1 FROM targets t WHERE t.asset_id=assets.id) AND NOT "
                f"EXISTS (SELECT 1 FROM {table} l JOIN targets t2 ON t2.id=l.target_id "
                "WHERE t2.asset_id=assets.id)")
        return clauses, params

    def list_assets(self, search="", limit=100, offset=0, *, collection_id=None,
                    scattered=False, include_archived=False, after=None,
                    missing_labels=None):
        """分页列出资产。

        ``collection_id`` 指定时按集合内 ``position`` 顺序；否则按
        ``created_at DESC, id``（键集分页用 ``after=(created_at, id)`` 取下一页）。
        """
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError("分页参数不合法")
        clauses, params = self._asset_clauses(search, collection_id=collection_id,
                                              scattered=scattered,
                                              include_archived=include_archived,
                                              missing_labels=missing_labels)
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
                     include_archived=False, missing_labels=None):
        clauses, params = self._asset_clauses(search, collection_id=collection_id,
                                              scattered=scattered,
                                              include_archived=include_archived,
                                              missing_labels=missing_labels)
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
        """独立资产的文件路径与校验；SigMF 指向 ``.sigmf-data`` 并核对同一对元数据。

        分片资产必须走 :meth:`load_samples`。
        """
        if asset.get("storage_kind", "file") != "file":
            raise ValueError("分片资产没有独立文件，请通过 load_samples 读取")
        path = (self.root / asset["path"]).resolve()
        if not path.is_relative_to(self.root / "assets"):
            raise ValueError("资产路径越界")
        if not path.is_file():
            raise ValueError("资产文件缺失或校验失败")
        if str(asset.get("storage_format") or "npy") == "sigmf" and \
                not path.with_suffix(".sigmf-meta").is_file():
            raise ValueError("资产文件缺失或校验失败")
        if file_digest(path) != asset["sha256"]:
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
        """按资产登记的存储格式返回 ``(asset, samples)``；NPY 保持内存映射读取。"""
        asset = self.get_asset(asset_id)
        kind = asset.get("storage_kind", "file")
        if kind == "file":
            path = self.resolve_asset(asset)
            fmt = str(asset.get("storage_format") or "npy")
            if fmt == "npy":
                return asset, np.load(path, mmap_mode="r", allow_pickle=False)
            # CSV / 交织 IQ / SigMF 无法内存映射：按登记格式整体读入并校验。
            return asset, read_samples(path, binary_dtype=BINARY_DTYPES.get(fmt),
                                       endian=str(asset.get("endian") or "little"))
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
