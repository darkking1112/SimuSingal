"""Opt-in native core: SIMUSIGNAL_COMPILE_CORE=1 python -m build --wheel.

SIMUSIGNAL_BUILD_EDITION 选择 trial（默认）或 full；它是构建期宏，不是运行时开关。
"""

import json
import os
from pathlib import Path
import re
import shutil
from setuptools import Extension, setup
from setuptools.command.build_py import build_py

compile_core = os.environ.get("SIMUSIGNAL_COMPILE_CORE") == "1"
edition = os.environ.get("SIMUSIGNAL_BUILD_EDITION", "trial")
if edition not in {"trial", "full"}:
    raise RuntimeError("SIMUSIGNAL_BUILD_EDITION 必须为 trial 或 full")

root = Path(__file__).parent
core_modules = json.loads((root / "packaging/numeric_core.json").read_text(encoding="utf-8"))
training_path = root / "packaging/training_core.json"
training_config = (json.loads(training_path.read_text(encoding="utf-8"))
                   if training_path.is_file() else {})
training_modules = list(training_config.get("modules", []))
for package in training_config.get("packages", []):
    source_root = Path(package["source_root"])
    exclude = set(package.get("exclude", []))
    for path in sorted(source_root.glob("*.py")):
        if path.name in exclude:
            continue
        training_modules.append({"module": f"{package['package']}.{path.stem}",
                                 "source": str(path), "package": True})
training_source_only = training_config.get("source_only", [])
compiled_modules = list(core_modules) + [item["module"] for item in training_modules]
removed_modules = compiled_modules + list(training_source_only)


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _copy_training_runtime(build_lib):
    """把未编译的训练支持代码放进 build_lib。

    正常运行由 ``scripts/build_wheels.py`` 先把这些文件复制到临时 ``src``；
    从 sdist 直接构建时它们只存在于 sdist 根目录，因此这里再做一次兜底复制。
    """
    target = Path(build_lib) / "signal_analysis" / "training"
    target.mkdir(parents=True, exist_ok=True)
    staged = root / "src/signal_analysis/training"
    amc_init = _first_existing(staged / "amc_models/__init__.py",
                               root / "training/amc_models/__init__.py")
    if amc_init is not None:
        destination = target / "amc_models"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(amc_init, destination / "__init__.py")
        template = _first_existing(staged / "amc_models/_template.py",
                                   root / "training/amc_models/_template.py")
        if template is not None:
            shutil.copyfile(template, destination / "_template.py")
    detectors = _first_existing(staged / "detectors", root / "training/detectors")
    if detectors is not None:
        shutil.copytree(detectors, target / "detectors", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    custom = _first_existing(staged / "custom", root / "custom")
    if custom is not None:
        destination = target / "custom"
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(custom.glob("*.py")):
            shutil.copyfile(source, destination / source.name)
        (destination / "__init__.py").write_text(
            '"""随训练组件分发的第三方模型实现（按文件路径加载，不改原文件）。"""\n',
            encoding="utf-8")
    for name in ("desktop_evaluate.py", "verify_onnx.py", "rtdetr_desktop.py"):
        source = _first_existing(staged / name, root / "training" / name)
        if source is None:
            continue
        text = source.read_text(encoding="utf-8")
        if name == "verify_onnx.py":
            text = re.sub(
                r"REPO_ROOT = Path\(__file__\)\.resolve\(\)\.parents\[1\]\n"
                r"if str\(REPO_ROOT / \"src\"\) not in sys\.path:\n"
                r"    sys\.path\.insert\(0, str\(REPO_ROOT / \"src\"\)\)\n",
                "", text)
        (target / name).write_text(text, encoding="utf-8")


class BuildPy(build_py):
    def run(self):
        super().run()
        if compile_core:
            # 清理先前源码构建留下的所有核心明文，避免混入二进制 wheel。
            for module in removed_modules:
                stem = Path(self.build_lib) / module.replace(".", "/")
                for suffix in (".py", ".pyc", ".c", ".cpp", ".pyx", ".pxd", ".pxi", ".h"):
                    stem.with_suffix(suffix).unlink(missing_ok=True)
            # 构建期头文件与 include 源码不属于客户包。
            native = Path(self.build_lib) / "signal_analysis/_native_training"
            if native.exists():
                shutil.rmtree(native, ignore_errors=True)
            _copy_training_runtime(self.build_lib)

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if compile_core:
            modules = [item for item in modules if f"{item[0]}.{item[1]}" not in removed_modules]
        return modules


extensions = []
if compile_core:
    from Cython.Build import cythonize

    numeric_extensions = [
        Extension(module, ["src/" + module.replace(".", "/") + ".py"])
        for module in core_modules
    ]
    training_extensions = [
        Extension(
            item["module"],
            [item["source"]],
            include_dirs=["src/signal_analysis/_native_training"],
            define_macros=[("SA_BUILD_TRIAL", "1" if edition == "trial" else "0")],
            depends=[
                "src/signal_analysis/_native_training/trial_limits.h",
                "src/signal_analysis/_native_training/trial_checks.pxi",
            ],
        )
        for item in training_modules
    ]
    extensions = cythonize(
        numeric_extensions + training_extensions,
        build_dir="build/cython",
        compiler_directives={"language_level": 3, "binding": True},
    )

setup(ext_modules=extensions, cmdclass={"build_py": BuildPy})
