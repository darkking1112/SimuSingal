#!/usr/bin/env python3
"""原始 IQ 分类网络（1D CNN / TCN 基线，可选分支，需要 ``pip install .[train]``）。

定位
----
本文件与 :mod:`signal_analysis.ml.iq` 的 ``iq_waveform_v1`` 契约配套：

* **输入**：``(B, 2, N)`` 的**单位 RMS** 复基带窗口，通道 0 = I、通道 1 = Q，
  N 由模型清单的 ``input.samples`` 决定（默认 1024）；
* **输出**：``(B, C)`` 概率，softmax **写进导出图**，与 :mod:`amc_transformer`
  的约定一致——ONNX 文件自身就是完整口径。

与 A09 特征通路的关系是**互补而非替代**：

============================  ==========================  ==========================
                              ``amc_classify``（特征）    ``amc_iq_classify``（IQ）
============================  ==========================  ==========================
输入契约                      ``amc_feature_vector_v1``   ``iq_waveform_v1``
输入内容                      34 维人工特征               ``(2, N)`` 原始 IQ
类别字典                      A09 六类（冻结）            清单声明，可为更宽字典
模型结构                      线性判别 / FT-Transformer   1D CNN / TCN
============================  ==========================  ==========================

因此两者**不共享类别字典也不共享模型文件**：TorchSig 的信号实例/调制类别只有在
显式映射成项目字典后才能进训练集（见 :mod:`build_iq_dataset`）。

结构
----
* :class:`IQCNN`：三级带步长的一维卷积（2→32→64→128）+ 批归一化 + GELU，
  时间维全局平均与最大池化拼接后接两层全连接；
* :class:`IQTCN`：膨胀因果卷积残差块（dilation 1,2,4,8,16），适合捕捉符号级
  周期结构，时间维池化方式相同。

两者都**不含任何归一化层**：窗口归一化由 :func:`signal_analysis.ml.iq.iq_waveform`
在推理前统一完成，训练数据也是同一个函数产出的，不给"训练-推理口径分叉"留口子。
本文件不下载任何预训练权重，训练数据全部来自本项目生成器与显式映射后的 TorchSig
导入结果。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#: 默认卷积通道数（1D CNN）
DEFAULT_CHANNELS = (32, 64, 128)
#: 默认一级卷积核长度
DEFAULT_KERNEL = 7
#: 默认 TCN 隐层宽度
DEFAULT_TCN_CHANNELS = 64
#: 默认 TCN 膨胀层数与核长
DEFAULT_TCN_LEVELS = 5
DEFAULT_TCN_KERNEL = 3
#: ONNX 必填的输入/输出节点名（与 ``signal_analysis.ml.iq`` 的常量一致）
INPUT_NAME = "iq"
OUTPUT_NAME = "scores"
#: PyTorch 与 ONNX 的输出偏差上限
TOLERANCE = 2e-4
#: 架构名称（``train_iq.py --arch`` 的取值）
ARCHITECTURES = ("cnn", "tcn")


def _pooling(waveform):
    """时间维全局平均池化与最大池化拼接（对 N 的奇偶不敏感）。"""
    return torch.cat([waveform.mean(dim=-1), waveform.amax(dim=-1)], dim=1)


class IQCNN(nn.Module):
    """两级输入的三级一维卷积基线（结构见模块文档）。"""

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
        return self.classifier(_pooling(self.features(waveform)))


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
    """膨胀因果卷积（TCN）基线，层数与核长见模块级常量。"""

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
        return self.classifier(_pooling(F.gelu(self.stem(waveform))))


def build_model(arch, classes, *, channels=None, kernel=None, dropout=0.1):
    """按 ``--arch`` 构建模型（未 softmax，输出 logits）。"""
    arch = str(arch).lower()
    if arch == "cnn":
        kwargs = {"channels": tuple(channels) if channels else DEFAULT_CHANNELS,
                  "kernel": int(kernel) if kernel else DEFAULT_KERNEL, "dropout": dropout}
        return IQCNN(classes, **kwargs)
    if arch == "tcn":
        kwargs = {"channels": int(channels[0]) if channels else DEFAULT_TCN_CHANNELS,
                  "kernel": int(kernel) if kernel else DEFAULT_TCN_KERNEL, "dropout": dropout}
        return IQTCN(classes, **kwargs)
    raise ValueError(f"未知架构 {arch!r}，可用：{', '.join(ARCHITECTURES)}")


class SoftmaxClassifier(nn.Module):
    """把 softmax 包进 ``forward``，使 ONNX 输出直接就是概率。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, waveform):
        return torch.softmax(self.model(waveform), dim=-1)


def train_classifier(train_x, train_y, val_x, val_y, *, classes, arch="cnn", channels=None,
                     kernel=None, dropout=0.1, epochs=30, batch_size=64, learning_rate=1e-3,
                     weight_decay=1e-4, patience=8, seed=0, verbose=True,
                     device="cpu", progress=None):
    """确定性训练循环（AdamW + 交叉熵 + 按验证准确率早停）。

    返回 ``{model, arch, best_accuracy, best_epoch, epochs_run, history}``。
    只依赖 torch 自身，不引入任何训练框架。
    """
    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    val_x = np.asarray(val_x, dtype=np.float32)
    val_y = np.asarray(val_y, dtype=np.int64)
    if train_x.ndim != 3 or train_x.shape[1] != 2:
        raise ValueError("训练波形必须是 (M, 2, N)")
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("训练集与验证集波形形状不一致")
    if not len(train_x) or not len(val_x):
        raise ValueError("训练集与验证集都不能为空")

    torch.manual_seed(seed)
    model = build_model(arch, len(classes), channels=channels, kernel=kernel, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    loss_function = nn.CrossEntropyLoss()
    inputs = torch.as_tensor(train_x, dtype=torch.float32)
    targets = torch.as_tensor(train_y, dtype=torch.long)
    val_inputs = torch.as_tensor(val_x, dtype=torch.float32)
    val_targets = torch.as_tensor(val_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_accuracy, best_epoch, stale, history = float("-inf"), 0, 0, []
    for epoch in range(1, max(int(epochs), 1) + 1):
        model.train()
        order = torch.randperm(inputs.shape[0], generator=generator)
        total_loss = 0.0
        for start in range(0, order.numel(), int(batch_size)):
            index = order[start:start + int(batch_size)]
            optimizer.zero_grad()
            loss = loss_function(model(inputs[index].to(device)), targets[index].to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * int(index.numel())
        scheduler.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            for start in range(0, len(val_inputs), int(batch_size)):
                stop = start + int(batch_size)
                correct += int((model(val_inputs[start:stop].to(device)).argmax(dim=1)
                                == val_targets[start:stop].to(device)).sum())
            accuracy = correct / len(val_inputs)
        history.append({"epoch": epoch, "loss": total_loss / order.numel(),
                        "validation_accuracy": accuracy})
        if progress is not None:
            progress(dict(history[-1]))
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break
        if verbose:
            print(f"  epoch {epoch:3d}  损失 {total_loss / order.numel():.4f}"
                  f"  验证准确率 {accuracy:.4f}（最佳 {best_accuracy:.4f} @ {best_epoch}）",
                  flush=True)
    model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    return {"model": model, "arch": arch, "best_accuracy": best_accuracy,
            "best_epoch": best_epoch, "epochs_run": len(history), "history": history}


def export_onnx(model, path, *, classes, samples, opset=17):
    """导出 ONNX（输入 ``iq (1,2,N)``、输出 ``scores (1,C)``）并做数值一致性校验。"""
    from detectors.torch_export import export_torch_module, require_torch

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SoftmaxClassifier(model)
    wrapper.eval()
    dummy = torch.zeros(1, 2, int(samples), dtype=torch.float32)
    export_torch_module(wrapper, dummy, path, opset=int(opset),
                        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], torch=torch)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("ONNX 导出失败：文件为空")

    try:
        import onnxruntime
    except ImportError:  # pragma: no cover - 取决于环境
        return path
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    # 导出图固定 batch=1：推理端 ``iq_scores`` / ``IQModelRunner`` 也只按 (1,2,N) 送数，
    # 因此这里用同一形状探针，保证"训练导出的图"和"运行时实际喂的形状"完全一致。
    probe = rng.standard_normal((1, 2, int(samples))).astype(np.float32)
    got = np.asarray(session.run([OUTPUT_NAME], {INPUT_NAME: probe})[0], dtype=np.float64)
    with torch.no_grad():
        want = wrapper(torch.as_tensor(probe)).numpy().astype(np.float64)
    if got.shape != (1, len(classes)):
        raise RuntimeError(f"ONNX 输出形状 {got.shape} 与 (1, {len(classes)}) 不符")
    deviation = float(np.max(np.abs(got - want)))
    if not math.isfinite(deviation) or deviation > TOLERANCE:
        raise RuntimeError(f"ONNX 与 PyTorch 输出不一致，最大偏差 {deviation:g}")
    if float(np.max(np.abs(got.sum(axis=1) - 1.0))) > 1e-4:
        raise RuntimeError("ONNX 输出概率之和不为 1，softmax 未写入导出图")
    return path
