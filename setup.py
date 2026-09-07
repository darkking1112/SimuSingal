"""Opt-in native core: SIMUSIGNAL_COMPILE_CORE=1 python -m build --wheel."""

import os
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_py import build_py

compile_core = os.environ.get("SIMUSIGNAL_COMPILE_CORE") == "1"


class BuildPy(build_py):
    def run(self):
        super().run()
        if compile_core:
            # A previous source build may have left this file in build_lib.
            (Path(self.build_lib) / "signal_analysis" / "_numeric.py").unlink(missing_ok=True)

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        if compile_core:
            modules = [item for item in modules if item[1] != "_numeric"]
        return modules


extensions = []
if compile_core:
    from Cython.Build import cythonize
    extensions = cythonize(
        [Extension("signal_analysis._numeric", ["src/signal_analysis/_numeric.py"])],
        build_dir="build/cython",
        compiler_directives={"language_level": 3, "binding": True},
    )

setup(ext_modules=extensions, cmdclass={"build_py": BuildPy})
