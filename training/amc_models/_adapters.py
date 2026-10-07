"""输入布局适配与导出包装（torch 侧）。

统一契约：**模型目录工厂返回的模型直接接收 ``(B, 2, N)``、输出 ``(B, C)`` logits**。
偏好其它布局的实现（如 ``(B, N, 2)``）在自己的 ``build()`` 里用 :func:`adapt` 包一层，
使训练、验证评分与导出走的是同一个对象；导出时只在最外层追加 softmax。
"""

from __future__ import annotations

import torch
from torch import nn

#: 项目对外契约的布局（``iq_waveform_v1``，见 contracts/iq.py）
CHANNELS_FIRST = "iq_channels_first_v1"
CHANNELS_LAST = "iq_channels_last_v1"


class SoftmaxWrapper(nn.Module):
    """把 softmax 包进 ``forward``，使 ONNX 输出直接就是概率。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return torch.softmax(self.model(waveform), dim=-1)


class ChannelsLastAdapter(nn.Module):
    """把契约的 ``(B, 2, N)`` 转成模型内部偏好的 ``(B, N, 2)``。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return self.model(waveform.transpose(1, 2))


def adapt(model, layout: str):
    """按目录声明的 ``input_layout`` 把实现包成统一契约 ``(B, 2, N)``。"""
    if layout == CHANNELS_FIRST:
        return model
    if layout == CHANNELS_LAST:
        return ChannelsLastAdapter(model)
    raise ValueError(f"不支持的输入布局：{layout!r}")
