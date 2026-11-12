"""检查数值/训练扩展、明文排除和模型资源，并在隔离目录运行真实 wheel 回归。"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def core_module_names():
    """读取与构建共用的数值核心模块清单。"""
    return json.loads((ROOT / "packaging/numeric_core.json").read_text(encoding="utf-8"))


def training_config():
    return json.loads((ROOT / "packaging/training_core.json").read_text(encoding="utf-8"))


def training_runtime_names():
    """无需 torch 即可导入的训练入口（发布检查在构建机执行，不一定装 torch）。"""
    return [item["module"] for item in training_config().get("modules", [])]


def training_module_names():
    """全部训练扩展：入口 + 模型实现（只做产物校验，不在此进程导入）。"""
    config = training_config()
    names = training_runtime_names()
    for package in config.get("packages", []):
        source_root = ROOT / package["source_root"]
        exclude = set(package.get("exclude", []))
        for path in sorted(source_root.glob("*.py")):
            if path.name not in exclude:
                names.append(f"{package['package']}.{path.stem}")
    return names


def validate_core_members(names):
    """逐模块检查唯一扩展产物，拒绝核心明文与 Cython 中间源码。"""
    for module in core_module_names() + training_module_names():
        stem = module.replace(".", "/")
        artifacts = [name for name in names if name.startswith(stem + ".")]
        if len(artifacts) != 1 or not artifacts[0].endswith((".so", ".pyd")):
            raise RuntimeError(f"wheel 中核心产物不符合要求：{module}: {artifacts}")
    forbidden_suffixes = (".py", ".pyc", ".pyx", ".pxd", ".pxi", ".c", ".cpp", ".h")
    for module in training_module_names():
        stem = module.replace(".", "/")
        leaked = [name for name in names
                  if name.startswith(stem + ".") and name.endswith(forbidden_suffixes)]
        if leaked:
            raise RuntimeError(f"wheel 中训练模块包含可修改源码：{module}: {leaked}")
    source_only = training_config().get("source_only", [])
    for module in source_only:
        stem = module.replace(".", "/")
        leaked = [name for name in names if name.startswith(stem + ".")]
        if leaked:
            raise RuntimeError(f"wheel 中混入仅开发用训练源码：{module}: {leaked}")
    if any("/_native_training/" in name for name in names):
        raise RuntimeError("wheel 中包含构建期原生头文件或 include 源码")
    if any(name.endswith(forbidden_suffixes) for name in names
           if name.startswith("signal_analysis/_native_training/")):
        raise RuntimeError("wheel 中包含构建期原生源码")


def _run_pytest(path, env, extra_args=()):
    subprocess.run([sys.executable, "-m", "pytest", str(path),
                    "-q", "-rs", "-p", "no:cacheprovider",
                    "-o", "pythonpath=", "-o", "addopts=", *extra_args],
                   check=True, env=env, cwd=Path(path).parent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--baseline", type=Path, help="可选的同环境源码数值基线，逐字节核对 wheel 输出")
    parser.add_argument("--junitxml", type=Path, help="将二进制测试记录保存到指定路径")
    parser.add_argument("--skip-tests", action="store_true",
                        help="只检查 wheel 结构与核心导入，不运行隔离 pytest（快速发布检查）")
    parser.add_argument("--edition", choices=("trial", "full"), default=None,
                        help="核对 wheel 实际加载到的编译身份")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="simusignal-binary-test-") as stage:
        with ZipFile(args.wheel) as archive:
            names = archive.namelist()
            validate_core_members(names)
            # 调制识别的内置模型是运行时资源而不是源码，漏打包只会在用户机器上暴露，
            # 因此在这里和核心产物一样做硬校验。
            model = [name for name in names if name == "signal_analysis/algorithms/amc/amc_default.json"]
            if len(model) != 1:
                raise RuntimeError(f"wheel 中缺少调制识别内置模型：{model}")
            archive.extractall(stage)
        env = os.environ.copy()
        env["PYTHONPATH"] = stage
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["SIMUSIGNAL_EXPECT_BINARY"] = "1"
        if args.edition:
            env["SIMUSIGNAL_EXPECT_EDITION"] = args.edition
        modules = core_module_names()
        subprocess.run([sys.executable, "-c", "import importlib; "
                        f"modules = [importlib.import_module(name) for name in {modules!r}]; "
                        "assert all(m.__file__.endswith(('.so','.pyd')) for m in modules); "
                        "print([m.__file__ for m in modules])"],
                       check=True, env=env, cwd=stage)
        subprocess.run([sys.executable, "-c", "from signal_analysis.algorithms.amc import feature_model as amc; "
                        "path = amc.default_model_path(); assert path.is_file(), path; "
                        "model = amc.load_model(path); "
                        "assert model['contract'] == amc.AMC_MODEL_CONTRACT; "
                        "print(model['id'], model['version'])"],
                       check=True, env=env, cwd=stage)
        training = training_runtime_names()
        subprocess.run([sys.executable, "-c", "import importlib, json; "
                        f"modules = [importlib.import_module(name) for name in {training!r}]; "
                        "assert all(m.__file__.endswith(('.so','.pyd')) for m in modules); "
                        "from signal_analysis._training_policy import get_build_info; "
                        "print(json.dumps(get_build_info(), ensure_ascii=False)); "
                        "print([m.__file__ for m in modules])"],
                       check=True, env=env, cwd=stage)
        # 部分历史测试会插入相对仓库的 src。将测试和训练工具复制到隔离目录，
        # 使此路径不可能指回真实仓库；不复制 src，也不靠预先 import 掩盖污染。
        if not args.skip_tests:
            # 先在尚未复制 training/ 的安装目录中跑训练组件自包含用例：
            # 这些用例必须能只靠安装的 wheel 工作，证明训练 wheel 已自包含。
            isolated = Path(stage) / "isolated_tests"
            isolated.mkdir(parents=True)
            shutil.copyfile(ROOT / "tests/analysis/test_trial_training.py",
                            isolated / "test_trial_training.py")
            report_args = []
            if args.junitxml:
                args.junitxml.parent.mkdir(parents=True, exist_ok=True)
                report_args = ["--junitxml", str(args.junitxml.resolve())]
            _run_pytest(isolated, env, report_args)
            tests = ["test_numeric.py", "test_numeric_modules.py", "test_iqgen.py",
                     "test_detect.py", "test_detect_hops.py", "test_amc.py", "test_iq.py",
                     "test_ml_detect.py", "test_ml_detect_hops.py", "test_detector_adapters.py"]
            test_dir = Path(stage) / "tests/analysis"
            test_dir.mkdir(parents=True)
            # 沿用标记注册等测试配置；命令行仍清空 pythonpath，且此处不存在源码树。
            shutil.copyfile(ROOT / "pyproject.toml", Path(stage) / "pyproject.toml")
            for name in tests:
                shutil.copyfile(ROOT / "tests/analysis" / name, test_dir / name)
            shutil.copytree(ROOT / "training", Path(stage) / "training",
                            ignore=shutil.ignore_patterns("__pycache__", "data", "runs", "*.pyc"))
            env["SIMUSIGNAL_EXPECT_BINARY"] = "1"
            # 发布校验固定串行：不依赖发布机安装 pytest-xdist，也避免并行测试放大资源占用。
            _run_pytest(test_dir, env, report_args)
        if args.baseline:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_numeric_equivalence.py"),
                            str(args.baseline.resolve()), "--source", stage],
                           check=True, env=env, cwd=stage)


if __name__ == "__main__":
    main()
