"""训练环境检查脚本的纯逻辑测试：模拟各种版本与路径，不依赖真实环境。"""

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "training_env_check", ROOT / "scripts/check_training_env.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_release_ignores_local_and_prerelease_suffixes():
    assert module.release("2.14.0+cpu") == (2, 14, 0)
    assert module.release("1.8.0rc1") == (1, 8, 0)
    assert module.release("") == ()
    assert module.release("unknown") == ()


def test_package_states_cover_missing_low_and_ok(monkeypatch):
    versions = {"torch": "1.13.0", "torchvision": "0.29.0", "onnx": "1.22.0",
                "onnxruntime": "1.30.0", "torchsig": "2.2.0"}
    monkeypatch.setattr(module, "installed", versions.get)
    result = module.check_packages()
    assert result["torch"] == ("old", "1.13.0")
    assert result["onnxruntime"] == ("ok", "1.30.0")
    assert result["ultralytics"] == ("missing", "-")
    assert result["torchsig"] == ("ok", "2.2.0")


def test_torchsig_exact_version_is_enforced(monkeypatch):
    monkeypatch.setattr(module, "installed",
                        lambda name: "2.3.0" if name == "torchsig" else None)
    assert module.check_packages()["torchsig"] == ("old", "2.3.0")


def test_rtdetr_directory_marker_is_required(tmp_path):
    args = argparse.Namespace(rtdetr=str(tmp_path), rtdetr_config="")
    assert module.print_rtdetr(args) == ["rtdetr 目录"]
    marker = tmp_path / "src" / "core" / "yaml_config.py"
    marker.parent.mkdir(parents=True)
    marker.write_text("")
    assert module.print_rtdetr(args) == []


def test_rtdetr_config_file_is_required(tmp_path):
    config = tmp_path / "rtdetrv2_r18vd_120e_coco.yml"
    args = argparse.Namespace(rtdetr="", rtdetr_config=str(config))
    assert module.print_rtdetr(args) == ["rtdetr 模型 YAML"]
    config.write_text("")
    assert module.print_rtdetr(args) == []


def test_group_ready_requires_every_member():
    results = {display: ("ok", "1.0") for display, _, _, _ in module.PACKAGES}
    assert module.group_ready(results, "common")
    results["onnxruntime"] = ("missing", "-")
    assert not module.group_ready(results, "common")
    assert module.group_ready(results, "rtdetr")
