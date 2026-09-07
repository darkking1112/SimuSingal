import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest

from signal_analysis.plugins import create_demo_manifest, read_manifest
from signal_analysis.storage import Workspace
from signal_analysis.tasks import JobError, run_job

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def demo_library(tmp_path_factory):
    if not shutil.which("cmake"):
        pytest.skip("CMake unavailable")
    folder = tmp_path_factory.mktemp("native")
    subprocess.run(["cmake", "-S", str(ROOT / "examples/native_plugin"), "-B", str(folder)],
                   check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(folder), "--config", "Release"], check=True, capture_output=True)
    libraries = list(folder.rglob("demo_plugin.dll")) + list(folder.rglob("libdemo_plugin.so"))
    if not libraries:
        pytest.skip("Demo targets Windows and Linux")
    return libraries[0]


def test_worker_analysis_and_persisted_failure(tmp_path):
    request = {"workspace": str(tmp_path), "action": "demo", "count": 256}
    asset = run_job(request)
    result = run_job({**request, "action": "analyze", "asset_id": asset["id"]})
    assert result["summary"]["sample_count"] == 256
    with pytest.raises(JobError, match="不支持"):
        run_job({**request, "action": "missing"})
    states = [json.loads(path.read_text())["state"] for path in (tmp_path / "jobs").glob("*/status.json")]
    assert sorted(states) == ["failed", "success", "success"]


def test_cancelled_job(tmp_path):
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(JobError) as caught:
        run_job({"workspace": str(tmp_path), "action": "demo"}, cancel=cancelled)
    assert caught.value.code == "cancelled"


@pytest.mark.native
def test_native_roundtrip_through_worker(tmp_path, demo_library):
    manifest = demo_library.parent / "plugin.json"
    create_demo_manifest(demo_library, manifest)
    store = Workspace(tmp_path)
    asset = store.add_samples([1 + 2j, -3 + 4j], 100, "reference")
    result = run_job({"workspace": str(tmp_path), "action": "native", "asset_id": asset["id"], "manifest": str(manifest)})
    _, output = store.load_samples(result["derived_asset_id"])
    assert list(output) == [1 + 2j, -3 + 4j]
    assert result["plugin"]["sha256"]


@pytest.mark.native
def test_manifest_rejects_platform_mismatch(tmp_path, demo_library):
    local = tmp_path / demo_library.name
    shutil.copyfile(demo_library, local)
    path = tmp_path / "plugin.json"
    manifest = create_demo_manifest(local, path)
    manifest["bits"] = 32
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="bits"):
        read_manifest(path)


@pytest.mark.native
@pytest.mark.parametrize("behavior,expected", [("raise(SIGSEGV); return 1;", "worker_crashed"),
                                               ("for (;;) {}", "timeout"),
                                               ("return 99;", "task_failed")])
def test_native_failures_are_isolated(tmp_path, behavior, expected):
    if not shutil.which("cc"):
        pytest.skip("Unix C compiler unavailable")
    source = tmp_path / "bad.c"
    source.write_text("#include <stdint.h>\n#include <signal.h>\nuint32_t demo_abi_version(void){" + behavior + "}\n")
    library = tmp_path / "bad.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True)
    manifest = tmp_path / "plugin.json"
    create_demo_manifest(library, manifest)
    store = Workspace(tmp_path / "data")
    asset = store.add_samples([1], 100, "input")
    with pytest.raises(JobError) as caught:
        run_job({"workspace": str(store.root), "action": "native", "asset_id": asset["id"], "manifest": str(manifest)}, timeout=3)
    assert caught.value.code == expected
    assert store.list_runs() == []
    # A fresh job remains usable after the previous worker failed.
    assert run_job({"workspace": str(store.root), "action": "demo", "count": 16})["sample_count"] == 16
