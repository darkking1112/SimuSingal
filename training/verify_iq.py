#!/usr/bin/env python3
"""开发入口：转交 ``signal_analysis.training.verify``（随训练 wheel 分发）。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", Path(__file__).resolve().parent):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from signal_analysis.training.verify import (  # noqa: E402,F401
    RESULT_KEYS,
    _check,
    _dataset_metrics,
    _input_contract,
    _output_contract,
    _parse_args,
    _scenes,
    _strip_timing,
    main,
)

if __name__ == "__main__":
    raise SystemExit(main())
