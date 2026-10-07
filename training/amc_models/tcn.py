"""IQTCN：膨胀因果卷积基线（模型目录 ``tcn`` 的实现）。

结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.tcn``；
本文件只写网络本体，不写默认值、不做布局转换、不加 softmax。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ._common import pooling

#: 默认 TCN 隐层宽度
DEFAULT_TCN_CHANNELS = 64
#: 默认 TCN 膨胀层数与核长
DEFAULT_TCN_LEVELS = 5
DEFAULT_TCN_KERNEL = 3


class _CausalResidualBlock(nn.Module):
    """膨胀因果卷积残差块：左填充 ``(kernel-1) * dilation``，右侧不越界。"""

    def __init__(self, channels, kernel, dilation, dropout=0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, waveform):
        residual = waveform
        out = F.pad(waveform, (self.pad, 0))
        out = self.dropout(F.gelu(self.norm1(self.conv1(out))))
        out = F.pad(out, (self.pad, 0))
        out = self.dropout(F.gelu(self.norm2(self.conv2(out))))
        return out + residual


class IQTCN(nn.Module):
    """膨胀因果卷积（TCN）基线：1×1 stem → 膨胀因果残差块堆叠 → 池化 → 分类头。"""

    def __init__(self, classes, channels=DEFAULT_TCN_CHANNELS, levels=DEFAULT_TCN_LEVELS,
                 kernel=DEFAULT_TCN_KERNEL, dropout=0.1):
        super().__init__()
        if classes < 2:
            raise ValueError("类别数至少为 2")
        if channels < 1 or levels < 1 or kernel < 2:
            raise ValueError("channels/levels 必须为正、kernel 至少为 2")
        self.stem = nn.Conv1d(2, int(channels), 1)
        self.blocks = nn.Sequential(*[
            _CausalResidualBlock(int(channels), int(kernel), 2 ** index, dropout)
            for index in range(int(levels))])
        head = int(channels) * 2
        self.classifier = nn.Sequential(
            nn.Linear(head, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, int(classes)),
        )

    def forward(self, waveform):
        return self.classifier(pooling(self.blocks(F.gelu(self.stem(waveform)))))


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits；``samples`` 不参与结构）。"""
    channels = tuple(params["channels"])
    return IQTCN(int(num_classes), channels=int(channels[0]),
                 kernel=int(params["kernel"]), dropout=float(params["dropout"]))
