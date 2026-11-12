"""训练循环与 ONNX 导出（模型目录驱动；供 ``train_iq.py`` 与兼容层使用）。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ._adapters import SoftmaxWrapper

#: ONNX 必填的输入/输出节点名（与 ``signal_analysis.algorithms.amc.iq_model`` 的常量一致）
INPUT_NAME = "iq"
OUTPUT_NAME = "scores"
#: PyTorch 与 ONNX 的输出偏差上限
TOLERANCE = 2e-4


def train_classifier(train_x, train_y, val_x, val_y, *, classes, arch="cnn", params=None,
                     channels=None, kernel=None, dropout=None,
                     epochs=20, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                     patience=8, seed=0, verbose=True, device="cpu", progress=None,
                     lr_scheduler="cosine", monitor="accuracy", checkpoint=None):
    """转交受控训练运行时；发布构建中该入口是编译扩展并强制试用版上限。"""
    from signal_analysis.training.iq_runtime import train_classifier as _runtime

    return _runtime(train_x, train_y, val_x, val_y, classes=classes, arch=arch, params=params,
                    channels=channels, kernel=kernel, dropout=dropout,
                    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    weight_decay=weight_decay, patience=patience, seed=seed,
                    verbose=verbose, device=device, progress=progress,
                    lr_scheduler=lr_scheduler, monitor=monitor, checkpoint=checkpoint)


def export_onnx(model, path, *, classes, samples, opset=17):
    """导出 ONNX（输入 ``iq (1,2,N)``、输出 ``scores (1,C)``）并做数值一致性校验。"""
    from detectors.torch_export import export_torch_module, require_torch

    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = SoftmaxWrapper(model)
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
