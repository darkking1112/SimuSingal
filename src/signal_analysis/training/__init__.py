"""受控训练入口包。

源码开发时本包提供 Python 回退实现；trial/full 构建会把训练入口替换成编译扩展，
并在 wheel 中附带 ``amc_models`` / ``detectors`` 支持代码。本模块保持轻量，不导入 torch。
"""


def _install_aliases():
    """把包内训练支持包注册为历史的顶层模块名（仅编译 wheel 中存在）。

    模型目录与检测适配器仍使用 ``amc_models.*`` / ``detectors.*`` 导入路径；
    发布包把实现放在 ``signal_analysis.training`` 下，通过本函数保持兼容。
    """
    import sys

    try:
        from . import amc_models as amc
    except ImportError:
        return
    try:
        from . import detectors as det
    except ImportError:
        det = None

    sys.modules.setdefault("amc_models", amc)
    if det is None:
        return
    sys.modules.setdefault("detectors", det)
