#!/usr/bin/env python3
"""A09 调制识别的 Transformer 分支（可选，需要 ``pip install .[train]``）。

定位
----
本文件是 :mod:`signal_analysis.ml.amc` 里**同一个特征向量契约**的深度模型分支：
输入仍是 ``extract_features`` 产出的定长特征向量（未标准化），输出仍是六类概率。
因此它与线性基线可以**同口径对比**（同样的特征、同样的数据集、同样的验证划分），
差别只在于判别函数从"线性 + softmax"换成"小型 Transformer 编码器"。

结构（FT-Transformer 风格）
--------------------------
* 每个特征视为一个标量 token，``Linear(1, d_model)`` 嵌入后拼接一个 CLS token；
* ``layers`` 层 pre-norm Transformer 编码层（多头自注意力 + GELU 前馈）；
* 取 CLS 位置经 LayerNorm 后接线性分类头，softmax 输出概率；
* 标准化（``(x - mean) / scale``）以 buffer 形式写进导出图，使 ONNX 文件自身
  就是完整口径——与清单里 ``standardize`` 字段的一致性由 :mod:`amc` 校验。

注意：本分支不会自动下载任何预训练权重，训练数据全部来自本项目生成器合成，
无第三方模型的许可证约束。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_ONNX_OPSET = 17
TOLERANCE = 2e-4


class AMCTransformer(nn.Module):
    """把定长特征向量当作 token 序列的小型 Transformer 编码器。"""

    def __init__(self, feature_count, classes, d_model=64, heads=4, layers=2,
                 dropout=0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model 必须能被 heads 整除")
        if feature_count < 1 or classes < 2:
            raise ValueError("特征数必须为正、类别数至少为 2")
        self.feature_count = int(feature_count)
        self.classes = int(classes)
        self.embedding = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, features):
        tokens = self.embedding(features.unsqueeze(-1))
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(encoded[:, 0]))


class StandardizedClassifier(nn.Module):
    """标准化 + 分类器：ONNX 的输入是**原始特征**，输出是六类概率。"""

    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features):
        normalized = (features - self.mean) / self.scale
        return torch.softmax(self.model(normalized), dim=-1)


def _standardizer(matrix):
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    scale = np.maximum(np.asarray(matrix.std(axis=0), dtype=np.float64), 1e-3)
    return mean, scale


def train_classifier(train_x, train_y, val_x, val_y, *, classes, d_model=64, heads=4,
                     layers=2, epochs=60, batch_size=128, learning_rate=3e-3,
                     seed=0, weight_decay=1e-4, patience=12, verbose=True):
    """训练小型 Transformer，返回 ``{model, standardize, best_accuracy, best_epoch}``。

    只做最朴素的确定性训练循环（AdamW + 交叉熵 + 早停，按验证准确率选最优轮），
    不引入任何外部训练框架或预训练权重。
    """
    train_x = np.asarray(train_x, dtype=np.float32)
    val_x = np.asarray(val_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    val_y = np.asarray(val_y, dtype=np.int64)
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("训练集与验证集特征维度不一致")
    mean, scale = _standardizer(train_x)
    torch.manual_seed(seed)
    model = AMCTransformer(train_x.shape[1], len(classes), d_model=d_model,
                           heads=heads, layers=layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_function = nn.CrossEntropyLoss()
    normalized_train = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32)
    targets = torch.as_tensor(train_y, dtype=torch.long)
    normalized_val = torch.as_tensor((val_x - mean) / scale, dtype=torch.float32)
    val_targets = torch.as_tensor(val_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_accuracy, best_epoch, stale = float("-inf"), 0, 0
    for epoch in range(1, max(epochs, 1) + 1):
        model.train()
        order = torch.randperm(normalized_train.shape[0], generator=generator)
        for start in range(0, order.numel(), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_function(model(normalized_train[index]), targets[index])
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            accuracy = float((model(normalized_val).argmax(dim=1) == val_targets)
                             .to(torch.float64).mean())
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
        if verbose:
            print(f"  epoch {epoch:3d}  验证准确率 {accuracy:.4f}"
                  f"（最佳 {best_accuracy:.4f} @ {best_epoch}）", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "standardize": {"mean": [float(v) for v in mean],
                                            "scale": [float(v) for v in scale]},
            "best_accuracy": best_accuracy, "best_epoch": best_epoch}


def export_onnx(model, path, *, standardize, feature_count, classes, opset=DEFAULT_ONNX_OPSET):
    """导出 ONNX 分类器并用 ``onnxruntime`` 做一次数值一致性校验。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StandardizedClassifier(model, standardize["mean"], standardize["scale"])
    wrapper.eval()
    dummy = torch.zeros(1, int(feature_count), dtype=torch.float32)
    torch.onnx.export(
        wrapper, (dummy,), str(path), input_names=["features"], output_names=["scores"],
        dynamic_axes={"features": {0: "batch"}, "scores": {0: "batch"}}, opset_version=int(opset),
        do_constant_folding=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("ONNX 导出失败：文件为空")

    try:
        import onnxruntime
    except ImportError:  # pragma: no cover - 取决于环境
        return path
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    probe = rng.standard_normal((4, int(feature_count))).astype(np.float32)
    got = np.asarray(session.run(["scores"], {"features": probe})[0], dtype=np.float64)
    with torch.no_grad():
        want = wrapper(torch.as_tensor(probe)).numpy().astype(np.float64)
    if got.shape != (4, len(classes)):
        raise RuntimeError(f"ONNX 输出形状 {got.shape} 与 (4, {len(classes)}) 不符")
    deviation = float(np.max(np.abs(got - want)))
    if not math.isfinite(deviation) or deviation > TOLERANCE:
        raise RuntimeError(f"ONNX 与 PyTorch 输出不一致，最大偏差 {deviation:g}")
    if float(np.max(np.abs(got.sum(axis=1) - 1.0))) > 1e-4:
        raise RuntimeError("ONNX 输出概率之和不为 1，softmax 未写入导出图")
    return path
