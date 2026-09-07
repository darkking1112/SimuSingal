"""Shared run catalog. Business packages own their own workspace and tables."""
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


class Workspace:
    project = "common"

    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / "project.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")).get("project") != self.project:
                raise ValueError("该工作目录属于另一项目，请选择独立目录")
        elif (self.root / "catalog.sqlite3").exists():
            raise ValueError("旧版或未知数据库，请创建新目录并重新导入数据；原目录不会被修改")
        else:
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    json.dump({"project": self.project, "schema": 1}, stream)
            except FileExistsError:
                if json.loads(marker.read_text(encoding="utf-8")).get("project") != self.project:
                    raise ValueError("工作目录已被另一项目占用")
        for folder in ("runs", "jobs"):
            (self.root / folder).mkdir(exist_ok=True)
        self.database = self.root / "catalog.sqlite3"
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"不支持的数据库版本：{version}")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT,
                    created_at TEXT NOT NULL, result_json TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.database, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
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
