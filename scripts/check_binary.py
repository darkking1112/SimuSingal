"""检查全部数值扩展、明文排除和模型资源，并在隔离目录运行真实 wheel 回归。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


def core_module_names():
    """读取与构建共用的核心模块清单。"""
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "packaging/numeric_core.json").read_text(encoding="utf-8"))


def validate_core_members(names):
    """逐模块检查唯一扩展产物，拒绝核心明文与 Cython 中间源码。"""
    for module in core_module_names():
        stem = module.replace(".", "/")
        artifacts = [name for name in names if name.startswith(stem + ".")]
        if len(artifacts) != 1 or not artifacts[0].endswith((".so", ".pyd")):
            raise RuntimeError(f"wheel 中核心产物不符合要求：{module}: {artifacts}")
    if any(name.endswith((".c", ".pyx")) for name in names):
        raise RuntimeError("wheel 中包含 Cython 中间源码")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--baseline", type=Path, help="可选的同环境源码数值基线，逐字节核对 wheel 输出")
    parser.add_argument("--junitxml", type=Path, help="将二进制测试记录保存到指定路径")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="simusignal-binary-test-") as stage:
        with ZipFile(args.wheel) as archive:
            names = archive.namelist()
            validate_core_members(names)
            # 调制识别的内置模型是运行时资源而不是源码，漏打包只会在用户机器上暴露，
            # 因此在这里和核心产物一样做硬校验。
            model = [name for name in names if name == "signal_analysis/ml/amc_default.json"]
            if len(model) != 1:
                raise RuntimeError(f"wheel 中缺少调制识别内置模型：{model}")
            archive.extractall(stage)
        env = os.environ.copy()
        env["PYTHONPATH"] = stage
        env["QT_QPA_PLATFORM"] = "offscreen"
        modules = core_module_names()
        subprocess.run([sys.executable, "-c", "import importlib; "
                        f"modules = [importlib.import_module(name) for name in {modules!r}]; "
                        "assert all(m.__file__.endswith(('.so','.pyd')) for m in modules); "
                        "print([m.__file__ for m in modules])"],
                       check=True, env=env, cwd=stage)
        subprocess.run([sys.executable, "-c", "from signal_analysis.ml import amc; "
                        "path = amc.default_model_path(); assert path.is_file(), path; "
                        "model = amc.load_model(path); "
                        "assert model['contract'] == amc.AMC_MODEL_CONTRACT; "
                        "print(model['id'], model['version'])"],
                       check=True, env=env, cwd=stage)
        # 部分历史测试会插入相对仓库的 src。将测试和训练工具复制到隔离目录，
        # 使此路径不可能指回真实仓库；不复制 src，也不靠预先 import 掩盖污染。
        tests = ["test_numeric.py", "test_numeric_modules.py", "test_iqgen.py",
                 "test_detect.py", "test_detect_hops.py", "test_amc.py", "test_iq.py",
                 "test_ml_detect.py", "test_ml_detect_hops.py", "test_detector_adapters.py"]
        test_dir = Path(stage) / "tests/analysis"
        test_dir.mkdir(parents=True)
        # 沿用标记注册等测试配置；命令行仍清空 pythonpath，且此处不存在源码树。
        shutil.copyfile(root / "pyproject.toml", Path(stage) / "pyproject.toml")
        for name in tests:
            shutil.copyfile(root / "tests/analysis" / name, test_dir / name)
        shutil.copytree(root / "training", Path(stage) / "training",
                        ignore=shutil.ignore_patterns("__pycache__", "data", "runs", "*.pyc"))
        env["SIMUSIGNAL_EXPECT_BINARY"] = "1"
        report_args = []
        if args.junitxml:
            args.junitxml.parent.mkdir(parents=True, exist_ok=True)
            report_args = ["--junitxml", str(args.junitxml.resolve())]
        subprocess.run([sys.executable, "-m", "pytest", str(test_dir),
                        "-q", "-rs", "-p", "no:cacheprovider", "-o", "pythonpath=", *report_args],
                       check=True, env=env, cwd=stage)
        if args.baseline:
            subprocess.run([sys.executable, str(root / "scripts/check_numeric_equivalence.py"),
                            str(args.baseline.resolve()), "--source", stage],
                           check=True, env=env, cwd=stage)


if __name__ == "__main__":
    main()
