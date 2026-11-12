# cython: language_level=3
"""编译后的 IQ 训练运行时：四项试用版上限在训练循环内部强制执行。

``main`` / ``preflight_main`` 只转交 :mod:`signal_analysis.training._iq_cli`
完成数据装配与导出；真正的训练循环、样本数、窗口长度、模型参数量和轮数检查
全部位于本编译扩展。
"""
include "../_native_training/trial_checks.pxi"

import numpy as np

from signal_analysis._training_policy import POLICY_VERSION


def check_epochs(object epochs):
    return _checked_epochs(epochs)


def guard_train_count(object count):
    _guard_train_count(count)
    return int(count)


def guard_iq_data(object train_x, object val_x):
    _guard_iq_data(train_x, val_x)


def guard_model(object model):
    return _guard_model(model)


def _ensure_aliases():
    from . import _install_aliases

    _install_aliases()


def train_classifier(train_x, train_y, val_x, val_y, *, classes, arch="cnn", params=None,
                     channels=None, kernel=None, dropout=None,
                     epochs=20, batch_size=64, learning_rate=1e-3, weight_decay=1e-4,
                     patience=8, seed=0, verbose=True, device="cpu", progress=None,
                     lr_scheduler="cosine", monitor="accuracy", checkpoint=None):
    """确定性训练循环；商业上限从本模块的 C 编译期常量读取。"""
    cdef Py_ssize_t epoch_count
    cdef Py_ssize_t epoch
    cdef object parameter_count
    cdef object improved
    cdef object total_loss
    cdef object validation_loss
    cdef object correct
    cdef object accuracy
    cdef object order
    cdef object index
    cdef object loss
    cdef object batch_inputs
    cdef object batch_targets
    cdef object logits
    cdef object start
    cdef object stale

    from signal_analysis.algorithms.amc.ai_model import merge_param_sources, model_spec

    _ensure_aliases()
    from amc_models import build_model

    if lr_scheduler not in ("cosine", "plateau"):
        raise ValueError("学习率调度器只能是 cosine 或 plateau")
    if monitor not in ("accuracy", "val_loss"):
        raise ValueError("最佳权重判据只能是 accuracy 或 val_loss")

    epoch_count = _checked_epochs(epochs)
    train_x = np.array(train_x, dtype=np.float32, copy=True)
    train_y = np.array(train_y, dtype=np.int64, copy=True)
    val_x = np.array(val_x, dtype=np.float32, copy=True)
    val_y = np.array(val_y, dtype=np.int64, copy=True)
    _guard_iq_data(train_x, val_x)
    if train_x.ndim != 3 or train_x.shape[1] != 2:
        raise ValueError("训练波形必须是 (M, 2, N)")
    if train_x.shape[1:] != val_x.shape[1:]:
        raise ValueError("训练集与验证集波形形状不一致")
    if len(train_x) != len(train_y):
        raise ValueError("训练标签数量与波形数量不一致")
    if len(val_x) != len(val_y):
        raise ValueError("验证标签数量与波形数量不一致")
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("波形里存在非有限值")
    if not classes:
        raise ValueError("类别字典不能为空")
    if train_y.min() < 0 or train_y.max() >= len(classes):
        raise ValueError("训练标签超出类别字典范围")
    if val_y.min() < 0 or val_y.max() >= len(classes):
        raise ValueError("验证标签超出类别字典范围")

    import torch
    from torch import nn

    spec = model_spec(arch)
    legacy = {name: value for name, value in
              (("channels", channels), ("kernel", kernel), ("dropout", dropout))
              if value is not None}
    resolved = merge_param_sources(spec, legacy=legacy, explicit=params or {})

    torch.manual_seed(seed)
    model = build_model(arch, classes=len(classes), samples=int(train_x.shape[-1]),
                        params=resolved)
    # 先完成规模检查，再搬到 GPU 和创建优化器。
    parameter_count = _guard_model(model)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)
    loss_function = nn.CrossEntropyLoss()
    inputs = torch.as_tensor(train_x, dtype=torch.float32)
    targets = torch.as_tensor(train_y, dtype=torch.long)
    val_inputs = torch.as_tensor(val_x, dtype=torch.float32)
    val_targets = torch.as_tensor(val_y, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_accuracy, best_loss, best_epoch, stale, history = float("-inf"), float("inf"), 0, 0, []
    for epoch in range(1, epoch_count + 1):
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
        from amc_models.checkpoint import save_checkpoint

        written = save_checkpoint(checkpoint, model=model, arch=arch, classes=classes,
                                  samples=int(train_x.shape[-1]), params=resolved,
                                  extra={"epochs_run": len(history), "seed": seed,
                                         "monitor": monitor, "lr_scheduler": lr_scheduler,
                                         "best_epoch": best_epoch,
                                         "best_accuracy": best_accuracy,
                                         "best_validation_loss": best_loss,
                                         "edition": "trial" if SA_BUILD_TRIAL else "full",
                                         "policy_version": POLICY_VERSION,
                                         "model_parameters": int(parameter_count)})
    return {"model": model, "arch": arch, "monitor": monitor, "lr_scheduler": lr_scheduler,
            "best_accuracy": best_accuracy, "best_loss": best_loss, "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history, "checkpoint": written,
            "edition": "trial" if SA_BUILD_TRIAL else "full",
            "policy_version": POLICY_VERSION,
            "train_samples": int(train_x.shape[0]),
            "iq_samples": int(train_x.shape[-1]),
            "model_parameters": int(parameter_count),
            "requested_epochs": int(epoch_count)}


def main(argv=None):
    _ensure_aliases()
    from ._iq_cli import main as _main

    return _main(argv)


def preflight_main(argv=None):
    _ensure_aliases()
    from ._iq_cli import preflight_main as _preflight

    return _preflight(argv)
