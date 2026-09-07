from pathlib import Path

import numpy as np
import pytest

from signal_analysis.dataio import read_samples
from common.reports import export_report
from signal_analysis.storage import Workspace


def test_import_complex_csv_and_npy(tmp_path):
    csv = tmp_path / "中文.csv"
    csv.write_text("1,2\n3,4\n", encoding="utf-8")
    expected = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    np.testing.assert_array_equal(read_samples(csv), expected)
    npy = tmp_path / "array.npy"
    np.save(npy, expected)
    np.testing.assert_array_equal(read_samples(npy), expected)
    np.save(npy, np.array([{"unsafe": 1}], dtype=object))
    with pytest.raises(ValueError):
        read_samples(npy)


def test_asset_persistence_label_and_tampering(tmp_path):
    store = Workspace(tmp_path / "工作目录")
    asset = store.add_samples([1, 2], 100, "a'b")
    store.set_label(asset["id"], "参考备注")
    reopened = Workspace(store.root)
    assert reopened.list_assets("a'b")[0]["label"] == "参考备注"
    _, data = reopened.load_samples(asset["id"])
    np.testing.assert_array_equal(data, [1, 2])
    del data
    (store.root / asset["path"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="校验"):
        reopened.load_samples(asset["id"])


def test_run_roundtrip_and_escaped_report(tmp_path):
    store = Workspace(tmp_path / "data")
    asset = store.add_samples([1], 100, "sample")
    result = store.save_run("analysis", {"summary": {"label": "<script>alert(1)</script>"}},
                            {"points": np.array([1, 2])}, asset["id"])
    assert Workspace(store.root).get_run(result["run_id"]) == result
    assert not Path(result["plots_path"]).is_absolute()
    output = tmp_path / "report.html"
    export_report(result, output)
    assert "<script>" not in output.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        export_report(result, tmp_path / "report.exe")


def test_invalid_import_does_not_create_asset(tmp_path):
    store = Workspace(tmp_path)
    with pytest.raises(ValueError):
        store.add_samples([float("nan")], 100, "invalid")
    assert store.list_assets() == []
    assert list((tmp_path / "assets").iterdir()) == []


def test_failed_run_does_not_leave_success_artifact(tmp_path):
    import sqlite3
    store = Workspace(tmp_path)
    with pytest.raises(ValueError):
        store.save_run("analysis", {"summary": {}}, asset_id="missing")
    assert store.list_runs() == []
    assert list((tmp_path / "runs").iterdir()) == []
