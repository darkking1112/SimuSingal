"""Inspect a compiled wheel and run the numerical tests against its extension."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="simusignal-binary-test-") as stage:
        with ZipFile(args.wheel) as archive:
            names = archive.namelist()
            core = [name for name in names if name.startswith("simusignal/_numeric.")]
            if len(core) != 1 or not core[0].endswith((".so", ".pyd")):
                raise RuntimeError(f"wheel 中核心产物不符合要求：{core}")
            if any(name.endswith((".c", ".pyx")) for name in names):
                raise RuntimeError("wheel 中包含 Cython 中间源码")
            archive.extractall(stage)
        env = os.environ.copy()
        env["PYTHONPATH"] = stage
        subprocess.run([sys.executable, "-c", "import simusignal._numeric as n; "
                        "assert n.__file__.endswith(('.so','.pyd')), n.__file__; print(n.__file__)"],
                       check=True, env=env, cwd=stage)
        subprocess.run([sys.executable, "-m", "pytest", str(root / "tests/test_numeric.py"),
                        "-q", "-o", "pythonpath="], check=True, env=env, cwd=stage)


if __name__ == "__main__":
    main()
