"""Opt-in native core: SIMUSIGNAL_COMPILE_CORE=1 python -m build --wheel."""

import json
import os
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_py import build_py

compile_core = os.environ.get("SIMUSIGNAL_COMPILE_CORE") == "1"
core_modules = json.loads((Path(__file__).parent / "packaging/numeric_core.json").read_text(encoding="utf-8"))


class BuildPy(build_py):
    def run(self):
        super().run()
        if compile_core:
            # 清理先前源码构建留下的所有核心明文，避免混入二进制 wheel。
            for module in core_modules:
                (Path(self.build_lib) / (module.replace(".", "/") + ".py")).unlink(missing_ok=True)

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if compile_core:
            modules = [item for item in modules if f"{item[0]}.{item[1]}" not in core_modules]
        return modules


extensions = []
if compile_core:
    from Cython.Build import cythonize
    extensions = cythonize(
        [Extension(module, ["src/" + module.replace(".", "/") + ".py"])
         for module in core_modules],
        build_dir="build/cython",
        compiler_directives={"language_level": 3, "binding": True},
    )

setup(ext_modules=extensions, cmdclass={"build_py": BuildPy})
