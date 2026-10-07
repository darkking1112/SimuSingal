"""``mcldnn`` 实现：包装 ``custom/MCLDNN.py``（原文件不改）。

原实现按 ``(B, 2, N)`` 直接计算（它自带的 ``(B, N, 2)`` 兼容分支只对 N=128 生效），
因此不需要布局适配：包装层只负责把目录参数与类别数传进去。
结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.mcldnn``。
"""

from __future__ import annotations

from torch import nn

from ._custom import load_custom_module


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits；``samples`` 不参与结构）。"""
    module = load_custom_module("MCLDNN")
    return module.MCLDNN(num_classes=int(num_classes),
                         dropout_rate=float(params["dropout_rate"]))
