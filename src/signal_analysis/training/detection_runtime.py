"""源码开发时的检测训练限额回退；trial/full 构建替换为编译扩展。"""

from __future__ import annotations


def is_trial():
    return False


def guard_detection_samples(count):
    if type(count) is not int or count < 1:
        raise ValueError("训练集不能为空")
    return count


def guard_detection_epochs(epochs):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("训练轮数必须为正整数")
    if epochs > 10000:
        raise ValueError("训练轮数不能超过 10000")
    return epochs


def guard_detection_model(model):
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        raise ValueError("模型参数尚未初始化，无法检查规模") from exc


def require_rtdetr_allowed(arch=None):
    return None
