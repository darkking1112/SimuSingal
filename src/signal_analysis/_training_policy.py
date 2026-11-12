"""源码开发时的构建身份回退；trial/full 构建会替换为编译扩展。

本文件不参与训练执行路径，也不提供任何强制限额。发布构建会从 wheel 中删除它。
"""

from . import __version__ as _app_version

POLICY_VERSION = 1


def is_trial():
    return False


def get_build_info():
    return {
        "edition": "development",
        "policy_version": POLICY_VERSION,
        "app_version": _app_version,
        "max_train_samples": None,
        "max_iq_samples": None,
        "max_model_parameters": None,
        "max_epochs": None,
    }
