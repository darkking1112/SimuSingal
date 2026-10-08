"""Shared run catalog. Business packages own their own workspace and tables.

数据库 schema 采用**按项目、按顺序的迁移机制**：项目通过
:meth:`Workspace.schema_migrations` 声明 ``[(版本号, 说明, 执行函数)]``，
初始化时把 ``PRAGMA user_version`` 从当前值逐级升到最高版本。约定：

* 每个迁移函数都必须**幂等**（``CREATE TABLE IF NOT EXISTS``、按行判断跳过），
  允许在失败后用备份或原目录重试；
* 升级已有数据库前先自动备份到 ``backups/``（无表的空库不备份）；
* 版本高于当前程序支持的数据库直接拒绝打开，避免旧程序改坏新结构。
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import uuid
from . import __version__

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _user_tables(conn):
    return conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchone()[0]


class Workspace:
    project = "common"
    # 每个项目声明自己的数据库文件名；默认沿用历史名字 catalog.sqlite3，
    # 需要同目录并存多个项目数据库的新项目可覆盖它。
    db_filename = "catalog.sqlite3"

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "project.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")).get("project") != self.project:
                raise ValueError("该工作目录属于另一项目，请选择独立目录")
        elif (self.root / self.db_filename).exists():
            raise ValueError("旧版或未知数据库，请创建新目录并重新导入数据；原目录不会被修改")
        else:
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    json.dump({"project": self.project, "schema": 0}, stream)
            except FileExistsError:
                if json.loads(marker.read_text(encoding="utf-8")).get("project") != self.project:
                    raise ValueError("工作目录已被另一项目占用")
        for folder in ("runs", "jobs"):
            (self.root / folder).mkdir(exist_ok=True)
        self.database = self.root / self.db_filename
        self._batch_connection = None   # 由 batch() 设置：批量事务期间的共享连接
        self.schema_version = self._apply_migrations()
        if marker.exists():
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("schema") != self.schema_version:
                payload["schema"] = self.schema_version
                marker.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ 迁移框架
    def schema_migrations(self):
        """按版本升序返回 ``[(版本, 说明, 执行函数)]``；子类先取 ``super()`` 再追加。"""
        return [(1, "运行记录表", self._migration_runs_v1)]

    @staticmethod
    def _migration_runs_v1(conn):
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT,
                created_at TEXT NOT NULL, result_json TEXT NOT NULL);
        """)

    def _apply_migrations(self):
        migrations = sorted(self.schema_migrations(), key=lambda step: step[0])
        versions = [step[0] for step in migrations]
        if versions != sorted(set(versions)):
            raise ValueError("迁移版本号必须唯一且升序声明")
        target = versions[-1] if versions else 0
        existed = self.database.exists()
        with self.connect() as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > target:
                raise ValueError(
                    f"数据库版本 {current} 高于当前程序支持的 {target}，请使用新版本程序打开")
            pending = [step for step in migrations if step[0] > current]
            if pending and existed and _user_tables(conn):
                self._backup_database(current)
            for version, description, apply in pending:
                try:
                    apply(conn)
                    conn.execute(f"PRAGMA user_version = {int(version)}")
                    conn.commit()
                except BaseException as exc:
                    conn.rollback()
                    raise ValueError(
                        f"数据库迁移失败（版本 {version}：{description}）：{exc}；"
                        "可从 backups/ 恢复备份后重试") from exc
        return target

    def _backup_database(self, version):
        folder = self.root / "backups"
        folder.mkdir(exist_ok=True)
        stem = Path(self.db_filename).stem
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        candidate = folder / f"{stem}-v{version}-{stamp}.sqlite3"
        suffix = 1
        while candidate.exists():
            candidate = folder / f"{stem}-v{version}-{stamp}-{suffix}.sqlite3"
            suffix += 1
        shutil.copy2(self.database, candidate)
        return candidate

    # ------------------------------------------------------------------ 连接与运行记录
    @contextmanager
    def connect(self):
        """打开连接；在 :meth:`batch` 作用域内改为复用同一个连接（不各自提交）。"""
        if self._batch_connection is not None:
            yield self._batch_connection
            return
        conn = sqlite3.connect(self.database, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def batch(self):
        """把一段批量写入合并成一个事务（可重入）。

        平时每个数据层调用都会自己开连接并提交；批量导入（例如集合包导入）会产生上千次
        ``fsync``，实测占掉九成以上的时间。在 ``with workspace.batch():`` 里所有
        ``connect()`` 共用一个连接、退出时统一提交，失败则整体回滚——写入因此变成
        "要么全部成功、要么全都不落库"。文件类副作用（资产文件）不参与回滚，调用方需要
        自己清理，见 ``services/transfer.import_collection``。
        """
        if self._batch_connection is not None:
            yield self
            return
        conn = sqlite3.connect(self.database, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        self._batch_connection = conn
        try:
            yield self
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()
        finally:
            self._batch_connection = None
            conn.close()

    def save_run(self, kind, result, arrays=None, source_id=None):
        run_id = uuid.uuid4().hex
        folder = self.root / "runs" / run_id
        folder.mkdir()
        complete = {**result, "run_id": run_id, "kind": kind,
                    "created_at": utc_now(), "source_id": source_id,
                    "environment": {"application": __version__, "python": platform.python_version(),
                                    "platform": platform.platform()}}
        try:
            if arrays is not None:
                import numpy as np
                np.savez_compressed(folder / "plots.npz", **arrays)
                complete["plots_path"] = f"runs/{run_id}/plots.npz"
            payload = json.dumps(complete, ensure_ascii=False, allow_nan=False, indent=2)
            (folder / "result.json").write_text(payload, encoding="utf-8")
            with self.connect() as conn:
                conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                             (run_id, kind, source_id, complete["created_at"], payload))
        except BaseException:
            shutil.rmtree(folder)
            raise
        return complete

    def get_run(self, run_id):
        with self.connect() as conn:
            row = conn.execute("SELECT result_json FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("运行记录不存在")
        return json.loads(row[0])

    def list_runs(self, limit=100):
        with self.connect() as conn:
            rows = conn.execute("SELECT id,kind,created_at,source_id FROM runs "
                                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
