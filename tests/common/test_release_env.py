"""发布环境检查的纯逻辑测试：版本、require 清单与环境标记。"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "release_env_check", ROOT / "scripts/check_release_env.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_version_tuple_ignores_suffixes():
    assert module.version_tuple("2.5.2") == (2, 5, 2)
    assert module.version_tuple("2.14.0+cpu") == (2, 14, 0)
    assert module.version_tuple("1.8.0rc1") == (1, 8, 0)
    assert module.version_tuple("") == ()


def test_parse_requirement_supports_pins_and_markers():
    assert module.parse_requirement("numpy==2.5.2") == {
        "raw": "numpy==2.5.2", "name": "numpy", "op": "==", "version": "2.5.2", "marker": ""}
    assert module.parse_requirement("pytest-xdist>=3.5")["op"] == ">="
    marker = module.parse_requirement('typing-extensions==4.16.0; python_version < "3.13"')
    assert marker["name"] == "typing-extensions"
    assert marker["marker"] == 'python_version < "3.13"'
    assert module.parse_requirement("# comment") is None
    assert module.parse_requirement("") is None


def test_read_requirements_rejects_unparsed_lines(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("numpy==2.5.2\nPySide6>=6.6,<7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无法解析"):
        module.read_requirements(path)


def test_markers_use_string_platform_and_version_python():
    environment = {"python_version": "3.12", "sys_platform": "linux"}
    assert module.marker_applies('python_version < "3.13"', environment)
    assert not module.marker_applies('python_version >= "3.13"', environment)
    assert module.marker_applies('sys_platform == "linux"', environment)
    assert not module.marker_applies('sys_platform == "win32"', environment)
    assert module.marker_applies("", environment)
    assert module.marker_applies("unknown_marker", environment)


def test_satisfies_handles_pins_and_ranges():
    assert module.satisfies("2.5.2", "==", "2.5.2")
    assert module.satisfies("2.5.2", "==", "2.5.2.0")
    assert not module.satisfies("2.5.1", "==", "2.5.2")
    assert module.satisfies("3.3.0", ">=", "3.0")
    assert module.satisfies("3.3.0", "<", "4")
    assert not module.satisfies("4.0.0", "<", "4")
    assert module.satisfies("3.3.5", "~=", "3.3.0")
    assert not module.satisfies("3.4.0", "~=", "3.3.0")


def test_release_requirements_are_covered_by_validated_lock():
    lock = {item["name"].lower(): item for item in
            module.read_requirements(ROOT / "requirements-linux-py312.lock")}
    for name in ("build-linux-py312.txt", "runtime-linux-py312.txt"):
        for item in module.read_requirements(ROOT / "packaging/requirements" / name):
            assert item["name"].lower() in lock, item["name"]
            if item["op"]:
                assert (item["op"], item["version"]) == (
                    lock[item["name"].lower()]["op"], lock[item["name"].lower()]["version"])


def test_release_config_lists_all_core_modules_and_platforms():
    import json
    config = json.loads((ROOT / "packaging/release.json").read_text(encoding="utf-8"))["analysis"]
    core = json.loads((ROOT / "packaging/numeric_core.json").read_text(encoding="utf-8"))
    assert len(core) == 8
    assert set(config["platforms"]) == {"linux", "win32"}
    assert config["executable"] == "SignalAnalysis"
    for platform in ("linux", "win32"):
        assert (ROOT / config["requirements"]["build"].format(platform=platform)).is_file()
        assert (ROOT / config["requirements"]["runtime"].format(platform=platform)).is_file()
