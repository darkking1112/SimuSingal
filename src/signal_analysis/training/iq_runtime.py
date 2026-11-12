"""源码开发时的 IQ 训练运行时回退；trial/full 构建替换为编译扩展。

开发回退只执行项目原有的参数合法性检查，不启用试用版商业限额。
"""

from __future__ import annotations


def check_epochs(epochs):
    if type(epochs) is not int or epochs < 1:
        raise ValueError("训练轮数必须为正整数")
    if epochs > 10000:
        raise ValueError("训练轮数不能超过 10000")
    return epochs


def guard_train_count(count):
    if type(count) is not int or count < 1:
        raise ValueError("训练集不能为空")
    return count


def guard_iq_data(train_x, val_x):
    if train_x.ndim != 3 or val_x.ndim != 3:
        raise ValueError("IQ 输入必须是 (M, 2, N)")
    if train_x.shape[1] != 2 or val_x.shape[1] != 2:
        raise ValueError("IQ 输入必须包含 I、Q 两个通道")
    if train_x.shape[2] != val_x.shape[2]:
        raise ValueError("训练集与验证集窗口长度不一致")
    if val_x.shape[0] < 1:
        raise ValueError("验证集不能为空")
    guard_train_count(int(train_x.shape[0]))


def guard_model(model):
    try:
        return sum(parameter.numel() for parameter in model.parameters())
    except (AttributeError, NotImplementedError, RuntimeError) as exc:
        raise ValueError("模型参数尚未初始化，无法检查规模") from exc


def train_classifier(train_x, train_y, val_x, val_y, **kwargs):
    from ._iq_train_python import train_classifier as _python

    return _python(train_x, train_y, val_x, val_y, **kwargs)


def main(argv=None):
    from ._iq_cli import main as _main

    return _main(argv)


def preflight_main(argv=None):
    from ._iq_cli import preflight_main as _preflight

    return _preflight(argv)
