"""内置最小无锚框检测头（自研，Apache-2.0）——用于跑通"训练→导出→推理"全流程。

真实项目建议接入 YOLOX / RT-DETR（Apache-2.0）以获得可用精度；本文件的目的
不是刷指标，而是在不引入第三方检测库的前提下产出**契约合规**的 ONNX：
``forward`` 直接返回 ``(B, K, 6)`` 的检测框张量，列定义与推理端
``normalized_boxes_v1`` 完全一致（``[x_center, y_center, width, height,
confidence, class]``，前四列按图像宽高归一化），因此 ``torch.onnx.export``
之后无需任何后处理即可被 :func:`signal_analysis.ml.parse_model_output` 解码。

结构：若干次步长 2 卷积（默认 4 次 → 步长 16 网格）→ 3×3 + 1×1 卷积输出
``1（目标性）+ 4（框）+ C（类别）``；解码在图中用 ``sigmoid / softplus /
topk / gather`` 完成，全部是 ONNX 可导出的算子。

本文件只在安装了 ``[train]`` 额外依赖（torch）时才能导入。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def offsets_from_raw(raw):
    """目标性格心偏移的编码区间：``sigmoid(·) ∈ (0, 1)`` 映射到 ``(-0.25, 1.25)``。"""
    return torch.sigmoid(raw) * 1.5 - 0.25


class TinyDetector(nn.Module):
    """网格式无锚框检测头；输出形状固定为 ``(B, max_boxes, 6)``。"""

    def __init__(self, max_boxes=32, classes=1, width=32, strides=4):
        super().__init__()
        self.max_boxes = int(max_boxes)
        self.classes = int(classes)
        self.strides = int(strides)
        blocks = []
        channels = 1
        for _ in range(self.strides):
            blocks += [nn.Conv2d(channels, width, 3, stride=2, padding=1),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            channels = width
            width = min(width * 2, 256)
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, 1 + 4 + self.classes, 1))

    def forward(self, images):
        features = self.head(self.backbone(images))
        batch, _, grid_h, grid_w = features.shape
        objectness = torch.sigmoid(features[:, 0]).reshape(batch, -1)
        offsets = offsets_from_raw(features[:, 1:3]).reshape(batch, 2, -1)
        sizes = torch.nn.functional.softplus(features[:, 3:5]).reshape(batch, 2, -1)
        classes = features[:, 5:].reshape(batch, self.classes, -1)
        count = min(self.max_boxes, grid_h * grid_w)
        scores, index = torch.topk(objectness, count, dim=1)
        gather_index = index.reshape(batch, 1, count).expand(batch, 2, count)
        offset = torch.gather(offsets, 2, gather_index)
        size = torch.gather(sizes, 2, gather_index)
        column = torch.remainder(index, grid_w).to(features.dtype)
        row = torch.div(index, grid_w, rounding_mode="floor").to(features.dtype)
        center_x = (column + offset[:, 0]) / float(grid_w)
        center_y = (row + offset[:, 1]) / float(grid_h)
        box_width = size[:, 0] / float(grid_w)
        box_height = size[:, 1] / float(grid_h)
        if self.classes > 1:
            gathered = torch.gather(classes, 2, index.reshape(batch, 1, count)
                                    .expand(batch, self.classes, count))
            label = gathered.argmax(dim=1).to(features.dtype)
        else:
            label = torch.zeros_like(center_x)
        return torch.stack([
            center_x.clamp(0.0, 1.0), center_y.clamp(0.0, 1.0),
            box_width.clamp(1e-4, 1.0), box_height.clamp(1e-4, 1.0),
            scores, label,
        ], dim=2)


def grid_size(image_size, strides=4):
    """输入边长 → 特征网格边长（与 :class:`TinyDetector` 的下采样一致）。"""
    size = int(image_size)
    for _ in range(int(strides)):
        size = math.ceil(size / 2)
    return size


def encode_targets(boxes, grid_h, grid_w):
    """标签框 ``(M, 4) = [cx, cy, w, h]``（归一化）→ 训练目标字典。

    每个真值框落到"框心所在的那一格"（该格 ``objectness`` 为正样本，并
    负责预测整框）；这与 :class:`TinyDetector` 的解码严格互逆，因此训练
    目标与推理输出处在同一坐标系里。同一格出现多框时按面积优先取大框。
    """
    objectness = np.zeros((grid_h, grid_w), dtype=np.float32)
    offsets = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    sizes = np.zeros((2, grid_h, grid_w), dtype=np.float32)
    labels = np.zeros((grid_h, grid_w), dtype=np.int64)
    areas = np.zeros((grid_h, grid_w), dtype=np.float64)
    for box in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        center_x, center_y, width, height = (float(value) for value in box)
        column = int(np.clip(np.floor(center_x * grid_w), 0, grid_w - 1))
        row = int(np.clip(np.floor(center_y * grid_h), 0, grid_h - 1))
        area = width * height
        if objectness[row, column] > 0.5 and areas[row, column] >= area:
            continue
        objectness[row, column] = 1.0
        areas[row, column] = area
        offsets[0, row, column] = (center_x * grid_w) - column
        offsets[1, row, column] = (center_y * grid_h) - row
        sizes[0, row, column] = width * grid_w
        sizes[1, row, column] = height * grid_h
        labels[row, column] = 0
    return {"objectness": objectness, "offsets": offsets, "sizes": sizes, "labels": labels}


def detector_loss(features, targets, objectness_weight=1.0, box_weight=5.0, class_weight=1.0):
    """目标性 + 框 + 类别的多任务损失（框回归只作用在正样本格上）。"""
    objectness_logits = features[:, 0]
    device = objectness_logits.device
    total = torch.zeros((), device=device)
    batch = objectness_logits.shape[0]
    for index in range(batch):
        target = targets[index]
        truth = torch.as_tensor(target["objectness"], device=device)
        weight = torch.where(truth > 0.5, torch.full_like(truth, 2.0),
                             torch.full_like(truth, 0.5))
        total = total + objectness_weight * torch.nn.functional.binary_cross_entropy_with_logits(
            objectness_logits[index], truth, weight=weight)
        positive = truth.reshape(-1) > 0.5
        if not bool(positive.any()):
            continue
        offset_target = torch.as_tensor(target["offsets"], device=device).reshape(2, -1)[:, positive]
        size_target = torch.as_tensor(target["sizes"], device=device).reshape(2, -1)[:, positive]
        offset_pred = offsets_from_raw(features[index, 1:3]).reshape(2, -1)[:, positive]
        size_pred = torch.nn.functional.softplus(features[index, 3:5]).reshape(2, -1)[:, positive]
        total = total + box_weight * (
            torch.nn.functional.l1_loss(offset_pred, offset_target, reduction="mean")
            + torch.nn.functional.l1_loss(torch.log1p(size_pred),
                                          torch.log1p(size_target.clamp(min=0.0)),
                                          reduction="mean"))
        label_target = torch.as_tensor(target["labels"], device=device).reshape(-1)[positive]
        logits = features[index, 5:].reshape(features.shape[1] - 5, -1).T[positive]
        total = total + class_weight * torch.nn.functional.cross_entropy(logits, label_target)
    return total / max(batch, 1)


def count_parameters(model):
    """可训练参数量（打印用）。"""
    return int(sum(parameter.numel() for parameter in model.parameters()))
