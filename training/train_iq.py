#!/usr/bin/env python3
"""开发入口：转交 ``signal_analysis.training._iq_cli``（wheel 使用编译运行时）。

用法与历史脚本一致；trial/full 构建中真正的训练入口是
``signal_analysis.training.iq_runtime`` 的编译扩展，不随客户分发本文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from signal_analysis.training._iq_cli import (  # noqa: E402,F401
    MANIFEST_NAME,
    TRAIN_EXTRA,
    _channels,
    _dependency_versions,
    _fmt,
    _model_params,
    _parse_args,
    _per_snr,
    _print_confusion,
    _print_metrics,
    _require_torch,
    _split,
    load_dataset,
    main,
    preflight_main,
)

if __name__ == "__main__":
    raise SystemExit(main())
