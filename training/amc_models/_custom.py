"""加载 ``custom/`` 下的第三方模型源文件（原文件不改，也不做 sys.path 注入）。

模型实现只在构建时按文件路径导入一次，模块名带 ``amc_custom_`` 前缀，
避免与项目内的同名模块冲突；缺文件时直接报错，不做静默兜底。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#: 第三方模型源码目录（``<repo>/custom``）
CUSTOM_ROOT = Path(__file__).resolve().parents[2] / "custom"


def load_custom_module(name: str):
    """按文件名加载 ``custom/<name>.py``（同一进程内只加载一次）。"""
    path = CUSTOM_ROOT / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少第三方模型文件：{path}")
    module_name = f"amc_custom_{name}"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - 取决于文件系统
        raise ImportError(f"无法加载第三方模型文件：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # 加载失败不留半成品，便于重试
        sys.modules.pop(module_name, None)
        raise
    return module
