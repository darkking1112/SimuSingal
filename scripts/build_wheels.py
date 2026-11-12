"""Build one project's wheel from its manifest and shared source tree."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def _copy_training_support(root, stage):
    """把发布所需的模型实现与检测支持代码装进 signal_analysis.training。"""
    target = stage / "src" / "signal_analysis" / "training"
    target.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "runs", "data")
    for package in ("amc_models", "detectors"):
        source = root / "training" / package
        if source.is_dir():
            shutil.copytree(source, target / package, dirs_exist_ok=True, ignore=ignore)
            # amc_models 的模块按 training_core.json 在构建期编译；保留一份
            # 与清单 source_root 一致的源码路径给 setup.py 使用。
            if package == "amc_models":
                shutil.copytree(source, stage / "training" / package,
                                dirs_exist_ok=True, ignore=ignore)
    custom_source = root / "custom"
    if custom_source.is_dir():
        custom_target = target / "custom"
        custom_target.mkdir(parents=True, exist_ok=True)
        (custom_target / "__init__.py").write_text(
            '"""随训练组件分发的第三方模型实现（按文件路径加载，不改原文件）。"""\n',
            encoding="utf-8")
        for path in sorted(custom_source.glob("*.py")):
            shutil.copyfile(path, custom_target / path.name)
    for name in ("desktop_evaluate.py", "rtdetr_desktop.py"):
        source = root / "training" / name
        if source.is_file():
            shutil.copyfile(source, target / name)
    verifier = root / "training" / "verify_onnx.py"
    if verifier.is_file():
        text = verifier.read_text(encoding="utf-8")
        text = re.sub(
            r"REPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]\n"
            r"if str\(REPO_ROOT / \"src\"\) not in sys\.path:\n"
            r"    sys\.path\.insert\(0, str\(REPO_ROOT / \"src\"\)\)\n",
            "", text)
        (target / "verify_onnx.py").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["analysis", "simulation"])
    parser.add_argument("--compile-core", action="store_true")
    parser.add_argument("--edition", choices=["trial", "full"], default="trial",
                        help="编译身份：trial（默认，启用固定限额）或 full")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.compile_core and args.project != "analysis":
        parser.error("当前仅分析项目的数值核心实现 Cython 编译，仿真模型仍为 Python")
    output = args.output or Path("dist") / args.edition / "wheels"
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "packaging/projects.json").read_text(encoding="utf-8"))[args.project]
    with tempfile.TemporaryDirectory(prefix=f"{args.project}-{args.edition}-build-") as directory:
        stage = Path(directory)
        for package in ("common", config["package"]):
            shutil.copytree(root / "src" / package, stage / "src" / package,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if args.project == "analysis":
            _copy_training_support(root, stage)
        shutil.copyfile(root / "apps" / config["app_dir"] / "pyproject.toml", stage / "pyproject.toml")
        shutil.copyfile(root / "setup.py", stage / "setup.py")
        (stage / "packaging").mkdir()
        shutil.copyfile(root / "packaging/numeric_core.json", stage / "packaging/numeric_core.json")
        if args.project == "analysis":
            shutil.copyfile(root / "packaging/training_core.json",
                            stage / "packaging/training_core.json")
        shutil.copyfile(root / "MANIFEST.in", stage / "MANIFEST.in")
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        env = os.environ.copy()
        env["SIMUSIGNAL_COMPILE_CORE"] = "1" if args.compile_core else "0"
        env["SIMUSIGNAL_BUILD_EDITION"] = args.edition
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(output.resolve())], cwd=stage, env=env, check=True)


if __name__ == "__main__":
    main()
