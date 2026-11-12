# cython: language_level=3
"""编译后的检测训练限额：样本数、模型参数量、轮数；IQ 窗口限制不适用。"""
include "../_native_training/trial_checks.pxi"


def is_trial():
    return bool(SA_BUILD_TRIAL)


def guard_detection_samples(object count):
    _guard_train_count(count)
    return int(count)


def guard_detection_epochs(object epochs):
    return _checked_epochs(epochs)


def guard_detection_model(object model):
    return _guard_model(model)


def require_rtdetr_allowed(arch=None):
    """试用版不把训练控制交给可编辑的上游 RT-DETR 源码；YOLO 不受此限制。"""
    if SA_BUILD_TRIAL and arch == "rtdetr":
        raise ValueError(
            "TRIAL_UNSUPPORTED: 试用版暂不开放 RT-DETR 训练；"
            "该入口需要受控的检测训练适配层")
    return None
