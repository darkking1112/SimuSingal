"""Signal-only asset tables and arrays, extending shared run storage."""
import uuid
import json
import numpy as np
from common.storage import Workspace as RunWorkspace, file_digest, utc_now
from .core_api import validate_rate, validate_samples

class Workspace(RunWorkspace):
    project = "signal_analysis"

    def __init__(self, root):
        super().__init__(root)
        (self.root / "assets").mkdir(exist_ok=True)
        with self.connect() as conn:
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

    def add_samples(self, samples, sample_rate, name, source="generated", *, metadata=None):
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
                if metadata is not None:
                    conn.execute("INSERT INTO asset_metadata VALUES (?,?)",
                                 (asset_id, json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
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
        if asset_id is not None:
            self.get_asset(asset_id)
        return super().save_run(kind, {**result, "asset_id": asset_id}, arrays, source_id=asset_id)
