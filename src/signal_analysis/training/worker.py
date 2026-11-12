"""源码开发时的训练 worker 回退；trial/full 构建替换为编译扩展。"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(argv=None):
    worker = Path(__file__).resolve().parents[3] / "training/desktop_worker.py"
    if not worker.is_file():
        raise SystemExit(f"训练源码目录缺少 {worker}")
    sys.argv = [str(worker), *([] if argv is None else argv)]
    runpy.run_path(str(worker), run_name="__main__")
