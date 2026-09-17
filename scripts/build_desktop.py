"""Build a desktop distribution from an already compiled core wheel."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from check_binary import validate_core_members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["analysis", "simulation"])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "packaging/projects.json").read_text(encoding="utf-8"))[args.project]
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        if not any(name.startswith(config["package"] + "/") for name in names):
            parser.error("wheel 与所选项目不匹配")
        if any(name.startswith(package + "/") for name in names for package in config["exclude"]):
            parser.error("wheel 包含另一项目代码")
        if args.project == "analysis":
            try:
                validate_core_members(names)
            except RuntimeError as exc:
                parser.error(f"请先构建全部核心均已编译且不附核心源码的 wheel：{exc}")
    with tempfile.TemporaryDirectory(prefix="simusignal-release-") as folder:
        folder = Path(folder)
        stage = folder / "stage"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
                        "--target", str(stage), str(wheel)], check=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stage)
        exclusions = [arg for package in config["exclude"] for arg in ("--exclude-module", package)]
        collections = [arg for package in config.get("collect_all", []) for arg in ("--collect-all", package)]
        # ``--collect-submodules`` 只收 Python 模块，包内数据文件（如调制识别的内置模型
        # JSON）必须单独声明，否则冻结后运行时才报找不到模型。
        data_files = [arg for package in config.get("collect_data", []) for arg in ("--collect-data", package)]
        subprocess.run([
            sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
            "--name", config["executable"], "--paths", str(stage),
            "--collect-submodules", config["package"], "--collect-submodules", "common",
            "--exclude-module", "matplotlib", "--exclude-module", "scipy",
            "--exclude-module", "PyQt5", "--exclude-module", "PyQt6",
            "--workpath", str(folder / "work"), "--specpath", str(folder),
            "--distpath", str(args.output.resolve()), *exclusions, *collections, *data_files,
            str(root / "apps" / config["app_dir"] / "main.py"),
        ], check=True, env=env)


if __name__ == "__main__":
    main()
