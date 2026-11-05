"""发布编排的纯逻辑测试：核心模块清单、产物命名与 wheel 成员校验。"""

import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("release_script", ROOT / "scripts/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def core_modules():
    return json.loads((ROOT / "packaging/numeric_core.json").read_text(encoding="utf-8"))


def test_runtime_selftest_core_list_matches_build_list():
    from signal_analysis.selftest import CORE_MODULES
    assert list(CORE_MODULES) == core_modules()


def test_artifact_name_is_stable():
    assert release.artifact_name("SignalAnalysis", "0.2.0", "linux-x86_64", "portable") == \
        f"SignalAnalysis-0.2.0-{release.python_tag()}-linux-x86_64"
    assert release.artifact_name("SignalAnalysis", "0.2.0", "linux-x86_64", "system") == \
        f"SignalAnalysis-0.2.0-{release.python_tag()}-linux-x86_64-system"


def test_core_manifest_requires_one_binary_per_module(tmp_path):
    wheel = tmp_path / "signal_analysis-0.2.0-cp312-cp312-linux_x86_64.whl"
    with ZipFile(wheel, "w") as archive:
        for module in core_modules():
            archive.writestr(module.replace(".", "/") + ".cpython-312-x86_64-linux-gnu.so",
                             module.encode("utf-8"))
    records = release.core_manifest(wheel)
    assert [item["module"] for item in records] == core_modules()
    assert all(len(item["sha256"]) == 64 for item in records)

    with ZipFile(wheel, "a") as archive:
        archive.writestr(core_modules()[0].replace(".", "/") + ".pyd", b"duplicate")
    with pytest.raises(release.ReleaseError, match="不唯一"):
        release.core_manifest(wheel)


def test_core_manifest_rejects_missing_binary(tmp_path):
    wheel = tmp_path / "broken.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("signal_analysis/__init__.py", b"")
    with pytest.raises(release.ReleaseError, match="不唯一"):
        release.core_manifest(wheel)
