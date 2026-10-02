"""信号分析桌面界面包。

公共导入面（CLI 启动与测试使用）：``MainWindow``、``launch``、
``SignalParamsDialog``（生成页“信号与采样”弹框版）以及模块级辅助
（``_mirrored_spectrum`` / ``_asset_exports`` / ``_iq_binary_kind`` /
``IMPORT_COL_*``），均由 :mod:`.main_window` 再导出；子页见 :mod:`.pages`。
"""

from .helpers import _asset_exports, _iq_binary_kind, _mirrored_spectrum
from .main_window import (IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN, IMPORT_COL_FILE,
                          IMPORT_COL_FORMAT, IMPORT_COL_MOD, IMPORT_COL_NOTE,
                          IMPORT_COL_POINTS, IMPORT_COL_RATE, IMPORT_COL_RF_CENTER,
                          IMPORT_COL_STATUS, MainWindow, SignalParamsDialog, launch)

__all__ = ["MainWindow", "SignalParamsDialog", "launch",
           "IMPORT_COL_FILE", "IMPORT_COL_FORMAT", "IMPORT_COL_RATE", "IMPORT_COL_DTYPE",
           "IMPORT_COL_ENDIAN", "IMPORT_COL_POINTS", "IMPORT_COL_RF_CENTER",
           "IMPORT_COL_MOD", "IMPORT_COL_NOTE", "IMPORT_COL_STATUS",
           "_mirrored_spectrum", "_asset_exports", "_iq_binary_kind"]
