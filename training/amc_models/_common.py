"""AMC 训练模型的共享小工具（池化等，torch 侧）。"""

from __future__ import annotations

import torch


def pooling(waveform: torch.Tensor) -> torch.Tensor:
    """时间维全局平均池化与最大池化拼接（对 N 的奇偶不敏感）。"""
    return torch.cat([waveform.mean(dim=-1), waveform.amax(dim=-1)], dim=1)
