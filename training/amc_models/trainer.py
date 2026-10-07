"""训练循环与 ONNX 导出（模型目录驱动；供 ``train_iq.py`` 与兼容层使用）。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from signal_analysis.algorithms.amc.ai_model import merge_param_sources, model_spec

from ._adapters import SoftmaxWrapper

#: ONNX 必填的输入/输出节点名（与 ``signal_analysis.algorithms.amc.iq_model`` 的常量一致）
INPUT_NAME = "iq"
OUTPUT_NAME = "scores"
#: PyTorch 与 ONNX 的输出偏差上限
TOLERANCE = 2e-4


def train_classifier(train_x, train_y, val_x, val_y, *, classes, arch="cnn", params=None,
                     channels=None, kernel=None, dropout=None,
                     epochs=30, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                     patience=8, seed=0, verbose=True, device="cpu", progress=None,
                     lr_scheduler="cosine", monitor="accuracy", checkpoint=None):
    """确定性训练循环（AdamW + 交叉熵；默认余弦退火 + 按验证准确率早停）。

    ``arch`` 与 ``params`` 由模型目录解释（默认值来自目录声明）；``channels`` /
    ``kernel`` / ``dropout`` 是旧调用方式的兼容入口，与 ``params`` 冲突时报错。

    ``lr_scheduler`` / ``monitor`` 是 opt-in（默认值与历史口径一致）：
    ``cosine`` 用 ``--epochs`` 为周期；``plateau`` 用 ``ReduceLROnPlateau(val_loss)``
    （``factor=0.5``、``patience=5``、``min_lr=1e-7``，见 ``custom/训练参数.md``）；
    ``monitor="val_loss"`` 时按**最低验证损失**选最佳轮（否则按最高验证准确率）。

    ``checkpoint`` 给出路径时把最佳权重与身份信息（模型 ID/结构版本/参数/类别/窗口）
    一并存盘，供 :func:`amc_models.checkpoint.restore` 复现加载。
    返回 ``{model, arch, monitor, best_accuracy, best_loss, best_epoch, epochs_run,
    history, checkpoint}``。
    """
    from . import build_model

    if lr_scheduler not in ("cosine", "plateau"):
        raise ValueError("学习率调度器只能是 cosine 或 plateau")
    if monitor not in ("accuracy", "val_loss"):
        raise ValueError("最佳权重判据只能是 accuracy 或 val_loss")
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

    spec = model_spec(arch)
    legacy = {name: value for name, value in
              (("channels", channels), ("kernel", kernel), ("dropout", dropout))
              if value is not None}
    resolved = merge_param_sources(spec, legacy=legacy, explicit=params or {})

    torch.manual_seed(seed)
    model = build_model(arch, classes=len(classes), samples=int(train_x.shape[-1]),
                        params=resolved).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    loss_function = nn.CrossEntropyLoss()
    inputs = torch.as_tensor(train_x, dtype=torch.float32)
    targets = torch.as_tensor(train_y, dtype=torch.long)
    val_inputs = torch.as_tensor(val_x, dtype=torch.float32)
    val_targets = torch.as_tensor(val_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_accuracy, best_loss, best_epoch, stale, history = float("-inf"), float("inf"), 0, 0, []
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
        model.eval()
        validation_loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(val_inputs), int(batch_size)):
                batch_inputs = val_inputs[start:start + int(batch_size)].to(device)
                batch_targets = val_targets[start:start + int(batch_size)].to(device)
                logits = model(batch_inputs)
                validation_loss += float(loss_function(logits, batch_targets)) * batch_inputs.shape[0]
                correct += int((logits.argmax(dim=1) == batch_targets).sum())
            accuracy = correct / len(val_inputs)
        validation_loss /= len(val_inputs)
        # 余弦调度按轮推进（与历史口径相同）；plateau 依赖验证损失，必须等评估完成
        if lr_scheduler == "plateau":
            scheduler.step(validation_loss)
        else:
            scheduler.step()
        history.append({"epoch": epoch, "loss": total_loss / order.numel(),
                        "validation_accuracy": accuracy, "validation_loss": validation_loss,
                        "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if progress is not None:
            progress(dict(history[-1]))
        improved = validation_loss < best_loss if monitor == "val_loss" else accuracy > best_accuracy
        if improved:
            best_accuracy, best_loss, best_epoch, stale = accuracy, validation_loss, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(patience):
                break
        if verbose:
            print(f"  epoch {epoch:3d}  损失 {total_loss / order.numel():.4f}"
                  f"  验证准确率 {accuracy:.4f}  验证损失 {validation_loss:.4f}"
                  f"（最佳 {best_accuracy:.4f} @ {best_epoch}）", flush=True)
    model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    written = None
    if checkpoint:
        from .checkpoint import save_checkpoint

        written = save_checkpoint(Path(checkpoint), model=model, arch=arch, classes=classes,
                                  samples=int(train_x.shape[-1]), params=resolved,
                                  extra={"epochs_run": len(history), "seed": seed,
                                         "monitor": monitor, "lr_scheduler": lr_scheduler,
                                         "best_epoch": best_epoch,
                                         "best_accuracy": best_accuracy,
                                         "best_validation_loss": best_loss})
    return {"model": model, "arch": arch, "monitor": monitor, "lr_scheduler": lr_scheduler,
            "best_accuracy": best_accuracy, "best_loss": best_loss, "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history, "checkpoint": written}


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
