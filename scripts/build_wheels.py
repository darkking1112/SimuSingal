"""Build one project's wheel from its manifest and shared source tree."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["analysis", "simulation"])
    parser.add_argument("--compile-core", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("dist/wheels"))
    args = parser.parse_args()
    if args.compile_core and args.project != "analysis":
        parser.error("当前仅分析项目的数值核心实现 Cython 编译，仿真模型仍为 Python")
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "packaging/projects.json").read_text())[args.project]
    with tempfile.TemporaryDirectory(prefix=f"{args.project}-build-") as directory:
        stage = Path(directory)
        for package in ("common", config["package"]):
            shutil.copytree(root / "src" / package, stage / "src" / package,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(root / "apps" / config["app_dir"] / "pyproject.toml", stage / "pyproject.toml")
        shutil.copyfile(root / "setup.py", stage / "setup.py")
        shutil.copyfile(root / "LICENSE", stage / "LICENSE")
        env = os.environ.copy()
        env["SIMUSIGNAL_COMPILE_CORE"] = "1" if args.compile_core else "0"
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(args.output.resolve())], cwd=stage, env=env, check=True)


if __name__ == "__main__":
    main()
