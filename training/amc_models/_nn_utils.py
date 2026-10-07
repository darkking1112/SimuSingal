"""timm 里用到的两个小工具的本地等价实现（避免为两个函数引入 timm 依赖）。

语义与 ``timm.layers.trunc_normal_`` / ``timm.layers.DropPath`` 一致：
截断正态初始化（均值 0、std 给定、范围 ±2σ，边界值重采样）与随机深度
（训练态按 ``drop_prob`` 丢整条残差、按存活概率缩放，``eval`` 恒等）。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """把张量填成截断正态分布（与 timm 的实现同语义，就地修改）。"""
    with torch.no_grad():
        if mean < a - 2 * std or mean > b + 2 * std:
            raise ValueError("mean 必须落在截断区间内")
        if std <= 0:
            raise ValueError("std 必须为正数")

        def norm_cdf(x):
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * low - 1, 2 * up - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0)).add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """随机深度（``drop_prob=0`` 时退化为恒等）。"""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, inputs):
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.dim() - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs.div(keep) * mask

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:g}"
