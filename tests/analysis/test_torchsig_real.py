"""Real generation and round-trip acceptance; runs in isolated Python processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.torchsig, pytest.mark.skipif(
    importlib.util.find_spec("torchsig") is None, reason="requires .[train] or .[torchsig]")]


def run_script(name, *arguments, tmp_path):
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / "mpl"), OMP_NUM_THREADS="2")
    result = subprocess.run([sys.executable, str(ROOT / "training" / name),
                             *map(str, arguments)], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_real_qpsk_generation_reproducible_and_ingested(tmp_path):
    args = ("--count", 4, "--num-iq-samples", 65536, "--signals-range", "1,1",
            "--signal-generators", "qpsk", "--seed", 7, "--snr-range", "10,20")
    for name in ("first", "repeat"):
        run_script("build_torchsig.py", "--output", tmp_path / name, *args, tmp_path=tmp_path)
    first = tmp_path / "first"
    for path in (first / "iq").glob("*.npy"):
        samples = np.load(path, allow_pickle=False)
        assert samples.shape == (65536,) and np.iscomplexobj(samples)
        assert np.isfinite(samples).all()
        assert np.array_equal(samples, np.load(tmp_path / "repeat/iq" / path.name))
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["torchsig_version"] == "2.2.0"
    run_script("ingest_torchsig.py", "--bundle", first, "--output", tmp_path / "detection",
               "--image-size", 128, "--nfft", 512, tmp_path=tmp_path)
    card = json.loads((tmp_path / "detection/dataset.json").read_text(encoding="utf-8"))
    assert card["ingest"]["accepted"] == 4
    assert card["ingest"]["rejected_total"] == 0
    records = [json.loads(line) for line in (tmp_path / "detection/samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4 and all(len(r["boxes"]) == 1 for r in records)
    from signal_analysis.annotations import validate_boxes
    for record in records:
        validate_boxes(record["boxes"])
        image = np.load(tmp_path / "detection" / record["image"])
        assert image.shape == (128, 128) and np.isfinite(image).all()
    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"qpsk":"qpsk"}')
    run_script("build_iq_dataset.py", "--output", tmp_path / "iq", "--per-class", 3,
               "--samples", 128, "--modes", "qpsk", "--class-set", "custom",
               "--classes", "qpsk", "--torchsig-bundle", first, "--torchsig-map", mapping,
               tmp_path=tmp_path)
    with np.load(tmp_path / "iq/iq_dataset.npz") as data:
        assert sum(data["source"] == "torchsig") == 4
        assert set(data["labels"]) == {"qpsk"}
