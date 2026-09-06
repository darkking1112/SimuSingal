"""Local SQLite catalog and immutable array assets."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sqlite3
import uuid

import numpy as np

from . import __version__
from .core_api import validate_rate, validate_samples


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        for folder in ("assets", "runs", "jobs"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.database = self.root / "catalog.sqlite3"
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError(f"不支持的数据库版本：{version}")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS assets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL,
                    sha256 TEXT NOT NULL, sample_rate REAL NOT NULL,
                    sample_count INTEGER NOT NULL, created_at TEXT NOT NULL,
                    source TEXT NOT NULL, label TEXT NOT NULL DEFAULT '');
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, asset_id TEXT,
                    created_at TEXT NOT NULL, result_json TEXT NOT NULL,
                    FOREIGN KEY(asset_id) REFERENCES assets(id));
                CREATE INDEX IF NOT EXISTS idx_assets_name ON assets(name);
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

    def add_samples(self, samples, sample_rate, name, source="generated"):
        data = validate_samples(samples)
        rate = validate_rate(sample_rate)
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise ValueError("名称应为 1～200 个字符")
        asset_id = uuid.uuid4().hex
        relative = f"assets/{asset_id}.npy"
        destination = self.root / relative
        temporary = destination.with_suffix(".tmp")
        try:
            with temporary.open("wb") as stream:
                np.save(stream, data, allow_pickle=False)
            temporary.replace(destination)
            with self.connect() as conn:
                conn.execute("INSERT INTO assets VALUES (?,?,?,?,?,?,?,?,?)",
                             (asset_id, name, relative, file_digest(destination), rate,
                              data.size, utc_now(), str(source), ""))
        except BaseException:
            temporary.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return self.get_asset(asset_id)

    def get_asset(self, asset_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            raise ValueError("数据记录不存在")
        return dict(row)

    def list_assets(self, search="", limit=100, offset=0):
        if not 1 <= limit <= 500 or offset < 0:
            raise ValueError("分页参数不合法")
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assets WHERE instr(name,?) > 0 "
                "ORDER BY created_at DESC,id LIMIT ? OFFSET ?", (search, limit, offset)
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_asset(self, asset):
        path = (self.root / asset["path"]).resolve()
        if not path.is_relative_to(self.root / "assets"):
            raise ValueError("资产路径越界")
        if not path.is_file() or file_digest(path) != asset["sha256"]:
            raise ValueError("资产文件缺失或校验失败")
        return path

    def load_samples(self, asset_id):
        asset = self.get_asset(asset_id)
        return asset, np.load(self.resolve_asset(asset), mmap_mode="r", allow_pickle=False)

    def set_label(self, asset_id, label):
        if not isinstance(label, str) or len(label) > 200:
            raise ValueError("备注最多 200 个字符")
        self.get_asset(asset_id)
        with self.connect() as conn:
            conn.execute("UPDATE assets SET label=? WHERE id=?", (label, asset_id))

    def save_run(self, kind, result, arrays=None, asset_id=None):
        run_id = uuid.uuid4().hex
        folder = self.root / "runs" / run_id
        folder.mkdir()
        complete = {**result, "run_id": run_id, "kind": kind,
                    "created_at": utc_now(), "asset_id": asset_id,
                    "environment": {"application": __version__, "python": platform.python_version(),
                                    "numpy": np.__version__, "platform": platform.platform()}}
        try:
            if arrays is not None:
                np.savez_compressed(folder / "plots.npz", **arrays)
                complete["plots_path"] = f"runs/{run_id}/plots.npz"
            payload = json.dumps(complete, ensure_ascii=False, allow_nan=False, indent=2)
            (folder / "result.json").write_text(payload, encoding="utf-8")
            with self.connect() as conn:
                conn.execute("INSERT INTO runs VALUES (?,?,?,?,?)",
                             (run_id, kind, asset_id, complete["created_at"], payload))
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
            rows = conn.execute("SELECT id,kind,created_at,asset_id FROM runs "
                                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
