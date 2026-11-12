# cython: language_level=3
"""编译后的构建身份与试用版展示限额（不导入 torch）。

源码开发时由同名的 ``_training_policy.py`` 回退；trial/full 构建会用本模块的
``.so/.pyd`` 替换它。限额来自 C 编译期宏，运行时没有覆盖入口。
"""
include "_native_training/trial_checks.pxi"

from . import __version__ as _app_version

POLICY_VERSION = 1


def is_trial():
    return bool(SA_BUILD_TRIAL)


def get_build_info():
    """供界面展示和版本握手使用；训练内核从不把返回字典当作限额来源。"""
    if SA_BUILD_TRIAL:
        return {
            "edition": "trial",
            "policy_version": POLICY_VERSION,
            "app_version": _app_version,
            "max_train_samples": SA_TRIAL_MAX_TRAIN_SAMPLES,
            "max_iq_samples": SA_TRIAL_MAX_IQ_SAMPLES,
            "max_model_parameters": SA_TRIAL_MAX_MODEL_PARAMETERS,
            "max_epochs": SA_TRIAL_MAX_EPOCHS,
        }
    return {
        "edition": "full",
        "policy_version": POLICY_VERSION,
        "app_version": _app_version,
        "max_train_samples": None,
        "max_iq_samples": None,
        "max_model_parameters": None,
        "max_epochs": None,
    }
