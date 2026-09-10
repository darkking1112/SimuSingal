"""Official-library interoperability, service integration, and failure boundaries."""
import json

import numpy as np
import pytest
import sigmf

from signal_analysis.dataio import read_samples, write_samples
from signal_analysis.sigmf_io import read_sigmf
from signal_analysis.services import execute
from signal_analysis.storage import Workspace


def test_official_roundtrip(tmp_path):
    data = np.array([0.1 + 0.2j, -2 + 3j, 0j], dtype=np.complex64)
    path = write_samples(tmp_path / "中文测试.sigmf-meta", data, "sigmf",
                         sample_rate=48000, description="基带测试", generation={"seed": 7})
    official = sigmf.fromfile(path)
    official.validate()
    assert official.sample_rate == 48000
    assert official.get_global_info()["core:datatype"] == "cf32_le"
    assert "core:frequency" not in official.get_captures()[0]
    np.testing.assert_array_equal(official.read_samples(), data)
    for file in (path, path.with_suffix(".sigmf-data")):
        samples, rate, metadata = read_sigmf(file)
        np.testing.assert_array_equal(samples, data)
        np.testing.assert_array_equal(read_samples(file), data)
        assert rate == 48000
        assert '"seed": 7' in metadata["global"]["core:description"]


@pytest.mark.parametrize("dtype", ["<c8", ">c8", "<c16", ">c16"])
def test_import_official_float_types(tmp_path, dtype):
    data = np.array([.5 + .25j, -.5 - .25j], dtype=dtype)
    recording = sigmf.fromarray(data)
    recording.sample_rate = 96000
    recording.tofile(tmp_path / "external")
    back, rate, _ = read_sigmf(tmp_path / "external.sigmf-data")
    np.testing.assert_array_equal(back, data)
    assert rate == 96000


@pytest.mark.parametrize("endian", ["le", "be"])
def test_import_official_integer_scaling(tmp_path, endian):
    data_path = tmp_path / "integer.sigmf-data"
    np.array([16384, -16384, 0, 8192], dtype="<i2" if endian == "le" else ">i2").tofile(data_path)
    meta = sigmf.SigMFFile(data_file=data_path, global_info={
        "core:datatype": f"ci16_{endian}", "core:sample_rate": 48000})
    meta.tofile(tmp_path / "integer.sigmf-meta")
    back, _, _ = read_sigmf(data_path)
    np.testing.assert_allclose(back, [.5 - .5j, .25j], atol=1 / 32768)


def test_generate_import_service_metadata(tmp_path):
    result = execute({"workspace": str(tmp_path / "workspace"), "action": "generate",
                      "sample_rate": 48000, "duration": .01, "signals": [], "seed": 9,
                      "noise": {"enabled": True, "power_dbfs": -20, "bandwidth": 48000},
                      "export": {"format": "sigmf"}})
    store = Workspace(tmp_path / "workspace")
    assert store.get_metadata(result["id"])["generation"]["seed"] == 9
    imported = execute({"workspace": str(store.root), "action": "import",
                        "path": result["export_path"]})
    assert imported["sample_rate"] == 48000
    assert "sigmf" in store.get_metadata(imported["id"])
    np.testing.assert_array_equal(store.load_samples(imported["id"])[1],
                                  store.load_samples(result["id"])[1])
    with pytest.raises(ValueError, match="采样率"):
        execute({"workspace": str(store.root), "action": "import",
                 "path": result["export_path"], "sample_rate": 100})


@pytest.mark.parametrize("change", ["missing_rate", "multi_channel", "external", "truncated", "missing", "bad_json"])
def test_invalid_pair(tmp_path, change):
    path = write_samples(tmp_path / "bad", np.ones(4, dtype=np.complex64), "sigmf", sample_rate=1000)
    metadata = json.loads(path.read_text())
    if change == "missing_rate":
        del metadata["global"]["core:sample_rate"]
    elif change == "multi_channel":
        metadata["global"]["core:num_channels"] = 2
    elif change == "external":
        metadata["global"]["core:dataset"] = "../outside.sigmf-data"
    elif change == "truncated":
        path.with_suffix(".sigmf-data").write_bytes(b"123")
    elif change == "missing":
        path.with_suffix(".sigmf-data").unlink()
    path.write_text("{" if change == "bad_json" else json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        read_sigmf(path)


def test_no_overwrite_and_no_partial_output(tmp_path):
    data = np.ones(4, dtype=np.complex64)
    path = write_samples(tmp_path / "keep", data, "sigmf", sample_rate=1000)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="已存在"):
        write_samples(path, data * 2, "sigmf", sample_rate=1000)
    assert path.read_bytes() == original
    with pytest.raises((TypeError, ValueError)):
        write_samples(tmp_path / "invalid", data, "sigmf", sample_rate=None)
    assert not (tmp_path / "invalid.sigmf-data").exists()


def test_existing_asset_metadata_default(tmp_path):
    store = Workspace(tmp_path)
    asset = store.add_samples(np.ones(4), 1000, "legacy")
    assert store.get_metadata(asset["id"]) == {}


def test_cli_automatic_rate(tmp_path, capsys):
    from signal_analysis.cli import main
    path = write_samples(tmp_path / "cli", np.ones(4), "sigmf", sample_rate=12345)
    assert main(["--workspace", str(tmp_path / "store"), "import", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["sample_rate"] == 12345
    assert main(["--workspace", str(tmp_path / "store"), "import", str(path),
                 "--sample-rate", "123"]) == 1


def test_checksum_and_invalid_values(tmp_path):
    path = write_samples(tmp_path / "checksum", np.ones(4), "sigmf", sample_rate=1000)
    metadata = json.loads(path.read_text())
    metadata["global"]["core:sha512"] = "0" * 128
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        read_sigmf(path)
    with pytest.raises(ValueError, match="NaN"):
        write_samples(tmp_path / "nan", np.array([complex(float("nan"), 0)]), "sigmf", sample_rate=1000)


def test_failed_pair_publication_cleans_up(tmp_path, monkeypatch):
    import signal_analysis.sigmf_io as adapter
    original = adapter.shutil.copyfileobj
    def fail_metadata(source, destination):
        if str(destination.name).endswith(".sigmf-meta"):
            raise OSError("simulated disk failure")
        original(source, destination)
    monkeypatch.setattr(adapter.shutil, "copyfileobj", fail_metadata)
    with pytest.raises(ValueError, match="disk failure"):
        write_samples(tmp_path / "fail", np.ones(4), "sigmf", sample_rate=1000)
    assert list(tmp_path.iterdir()) == []


def test_filename_with_dots(tmp_path):
    path = write_samples(tmp_path / "capture.v1.sigmf-meta", np.ones(4), "sigmf", sample_rate=1000)
    assert path.name == "capture.v1.sigmf-meta"
    assert read_sigmf(path)[1] == 1000
