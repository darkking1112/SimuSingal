"""``petcgdnn`` 实现：包装 ``custom/PETCGDNN.py``（原文件不改）。

原实现接受 ``(B, 2, N)`` 并内部转成 ``(B, N, 2)``；它的 PET 旋转层把窗口长度写进了
权重形状（``Linear 2N→1``），所以构建时必须拿到数据集的窗口长度并作为 ``frame_length``。
结构与参数约束的唯一声明在 ``signal_analysis.algorithms.amc.ai_model.petcgdnn``。
"""

from __future__ import annotations

from torch import nn

from ._custom import load_custom_module


def build(*, num_classes: int, samples, params: dict) -> nn.Module:
    """按目录参数与窗口长度构建模型（输入 ``(B, 2, N)``、输出 logits）。"""
    if samples is None:
        raise ValueError("模型 petcgdnn 的结构依赖窗口长度：请提供 samples"
                         "（训练链路会从数据集契约传入）")
    module = load_custom_module("PETCGDNN")
    return module.PETCGDNN(num_classes=int(num_classes), frame_length=int(samples),
                           hidden_size=int(params["hidden_size"]))
