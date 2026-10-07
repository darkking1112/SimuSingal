"""IQCNN：步长一维卷积基线（模型目录 ``cnn`` 的实现）。

结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.cnn``；
本文件只写网络本体，不写默认值、不做布局转换、不加 softmax。
"""

from __future__ import annotations

import torch
from torch import nn

from ._common import pooling

#: 默认卷积通道数（1D CNN）
DEFAULT_CHANNELS = (32, 64, 128)
#: 默认一级卷积核长度
DEFAULT_KERNEL = 7


class IQCNN(nn.Module):
    """两级输入的三级一维卷积基线。"""

    def __init__(self, classes, channels=DEFAULT_CHANNELS, kernel=DEFAULT_KERNEL, dropout=0.1):
        super().__init__()
        if classes < 2:
            raise ValueError("类别数至少为 2")
        if len(channels) != 3:
            raise ValueError("channels 需要 3 个宽度（三级卷积）")
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("kernel 需要不小于 3 的奇数")
        widths = [2, *[int(width) for width in channels]]
        blocks = []
        for index in range(3):
            half = kernel // 2
            blocks.append(nn.Sequential(
                nn.Conv1d(widths[index], widths[index + 1], kernel, stride=2, padding=half),
                nn.BatchNorm1d(widths[index + 1]),
                nn.GELU(),
            ))
            kernel = max(3, kernel - 2)
        self.features = nn.Sequential(*blocks)
        head = int(widths[-1]) * 2
        self.classifier = nn.Sequential(
            nn.Linear(head, head), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head, int(classes)),
        )

    def forward(self, waveform):
        return self.classifier(pooling(self.features(waveform)))


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数构建模型（输入 ``(B, 2, N)``、输出 logits；``samples`` 不参与结构）。"""
    return IQCNN(int(num_classes), channels=tuple(params["channels"]),
                 kernel=int(params["kernel"]), dropout=float(params["dropout"]))
