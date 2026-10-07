"""训练权重 checkpoint 的读写与兼容性检查（torch 侧）。

checkpoint 里保存的不只是权重，还有**复现所需的身份信息**：模型 ID、结构版本、
目录版本、结构参数、类别顺序与窗口长度。加载时逐项核对，任一不符即拒绝，
避免"拿 A 模型的结构参数去加载 B 模型的权重"这类静默错配。
"""

from __future__ import annotations

from pathlib import Path

import torch

from signal_analysis.algorithms.amc.ai_model import (CATALOG_VERSION, model_spec,
                                                     validate_params)

#: checkpoint 文件格式版本（字段含义变更时递增）
FORMAT_VERSION = 1


def save_checkpoint(path, *, model, arch, classes, samples, params, extra=None):
    """保存最佳权重与身份信息，返回写下的字典（便于记录到 metrics/清单）。"""
    spec = model_spec(arch)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_id": spec.id,
        "model_revision": spec.model_revision,
        "catalog_version": CATALOG_VERSION,
        "params": dict(params),
        "classes": list(classes),
        "samples": int(samples),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "extra": dict(extra or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def read_checkpoint(path):
    """读回 checkpoint（只接受本模块写出的格式）并校验格式版本。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"checkpoint 格式不受支持：{path}（期望 format_version"
                         f"={FORMAT_VERSION}）")
    return payload


def check_compatible(payload, *, arch, classes=None, samples=None, params=None):
    """核对 checkpoint 与当前模型口的口径；不匹配即抛 ``ValueError`` 并说明差异。"""
    spec = model_spec(arch)
    if payload["model_id"] != spec.id:
        raise ValueError(f"checkpoint 是模型 {payload['model_id']} 的权重，当前是 {spec.id}")
    if payload["model_revision"] != spec.model_revision:
        raise ValueError(f"checkpoint 的结构版本是 {payload['model_revision']}，"
                         f"当前 {spec.id} 是 {spec.model_revision}（结构与前向语义已变更，"
                         "请重新训练）")
    if classes is not None and list(payload["classes"]) != list(classes):
        raise ValueError(f"checkpoint 的类别顺序 {payload['classes']} 与当前 {list(classes)} 不符")
    if samples is not None and int(payload["samples"]) != int(samples):
        raise ValueError(f"checkpoint 的窗口长度是 {payload['samples']} 点，当前是 {int(samples)} 点")
    if params is not None:
        wanted = validate_params(spec, params)
        if dict(payload["params"]) != wanted:
            raise ValueError(f"checkpoint 的结构参数是 {payload['params']}，"
                             f"当前是 {wanted}")
    return payload


def load_state(path, *, arch, model, classes=None, samples=None, params=None):
    """把 checkpoint 的权重载入**已构建**的模型（形状不符由 ``load_state_dict`` 报错）。"""
    payload = check_compatible(read_checkpoint(path), arch=arch, classes=classes,
                               samples=samples, params=params)
    model.load_state_dict(payload["state_dict"])
    return payload


def restore(path, *, classes, device="cpu"):
    """按 checkpoint 里的身份信息重建模型并载入权重（窗口长度与结构参数取自文件）。"""
    from . import build_model

    payload = read_checkpoint(path)
    spec = model_spec(payload["model_id"])
    if payload["model_revision"] != spec.model_revision:
        raise ValueError(f"checkpoint 的结构版本是 {payload['model_revision']}，"
                         f"当前 {spec.id} 是 {spec.model_revision}（结构与前向语义已变更，"
                         "请重新训练）")
    if payload["catalog_version"] != CATALOG_VERSION:
        raise ValueError(f"checkpoint 的模型目录版本是 {payload['catalog_version']}，"
                         f"当前是 {CATALOG_VERSION}")
    if list(payload["classes"]) != list(classes):
        raise ValueError(f"checkpoint 的类别顺序 {payload['classes']} 与当前 {list(classes)} 不符")
    model = build_model(payload["model_id"], classes=len(classes),
                        samples=int(payload["samples"]), params=payload["params"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload
