"""构建校验必须覆盖每个核心模块，不能只检查旧兼容入口。"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("numeric_binary_check", ROOT / "scripts/check_binary.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def compiled_members():
    return [name.replace(".", "/") + ".cp310-win_amd64.pyd" for name in checker.core_module_names()]


def test_complete_binary_core_is_accepted():
    checker.validate_core_members(compiled_members())


@pytest.mark.parametrize("module", checker.core_module_names())
def test_missing_or_plaintext_core_is_rejected(module):
    stem = module.replace(".", "/")
    members = [name for name in compiled_members() if not name.startswith(stem + ".")]
    with pytest.raises(RuntimeError, match="核心产物"):
        checker.validate_core_members(members)
    with pytest.raises(RuntimeError, match="核心产物"):
        checker.validate_core_members(members + [stem + ".py"])
    with pytest.raises(RuntimeError, match="核心产物"):
        checker.validate_core_members(compiled_members() + [stem + ".py"])


@pytest.mark.parametrize("suffix", [".c", ".pyx"])
def test_cython_intermediates_are_rejected(suffix):
    with pytest.raises(RuntimeError, match="中间源码"):
        checker.validate_core_members(compiled_members() + ["signal_analysis/leaked" + suffix])
