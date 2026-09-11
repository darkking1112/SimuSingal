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
            core = [name for name in names if name.startswith("signal_analysis/_numeric.")]
            if len(core) != 1 or not core[0].endswith((".so", ".pyd")):
                raise RuntimeError(f"wheel 中核心产物不符合要求：{core}")
            if any(name.endswith((".c", ".pyx")) for name in names):
                raise RuntimeError("wheel 中包含 Cython 中间源码")
            # 调制识别的内置模型是运行时资源而不是源码，漏打包只会在用户机器上暴露，
            # 因此在这里和核心产物一样做硬校验。
            model = [name for name in names if name == "signal_analysis/ml/amc_default.json"]
            if len(model) != 1:
                raise RuntimeError(f"wheel 中缺少调制识别内置模型：{model}")
            archive.extractall(stage)
        env = os.environ.copy()
        env["PYTHONPATH"] = stage
        subprocess.run([sys.executable, "-c", "import signal_analysis._numeric as n; "
                        "assert n.__file__.endswith(('.so','.pyd')), n.__file__; print(n.__file__)"],
                       check=True, env=env, cwd=stage)
        subprocess.run([sys.executable, "-c", "from signal_analysis.ml import amc; "
                        "path = amc.default_model_path(); assert path.is_file(), path; "
                        "model = amc.load_model(path); "
                        "assert model['contract'] == amc.AMC_MODEL_CONTRACT; "
                        "print(model['id'], model['version'])"],
                       check=True, env=env, cwd=stage)
        subprocess.run([sys.executable, "-m", "pytest", str(root / "tests/analysis/test_numeric.py"),
                        str(root / "tests/analysis/test_detect.py"),
                        "-q", "-o", "pythonpath="], check=True, env=env, cwd=stage)


if __name__ == "__main__":
    main()
