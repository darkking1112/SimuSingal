"""AMC 训练模型实现注册表（torch 侧；**不随 wheel 分发**）。

模型目录（spec）在 ``signal_analysis.algorithms.amc.ai_model``；本包按
``spec.implementation`` 解析实现模块。**本模块自身不导入 torch**，保证
``train_iq.py --help``、界面侧的元数据查询与冻结应用都不需要 torch；
依赖检查（``requires``）发生在导入实现模块之前。
"""

from __future__ import annotations

import importlib
import importlib.util

from signal_analysis.algorithms.amc.ai_model import (CATALOG_VERSION, ModelSpec, SPECS,
                                                     check_samples, model_spec, validate_params)

__all__ = ["CATALOG_VERSION", "SPECS", "available_models", "build_model", "export_onnx",
           "implementation", "missing_requirements", "model_spec", "train_classifier"]


def available_models() -> tuple[str, ...]:
    """已登记模型的 ID（= ``train_iq.py --arch`` 的取值）。"""
    return tuple(SPECS)


def missing_requirements(name: str) -> tuple[str, ...]:
    """训练环境里缺失的依赖（按目录声明的 ``requires`` 检查）。"""
    spec = model_spec(name)
    return tuple(dependency for dependency in spec.requires
                 if importlib.util.find_spec(dependency) is None)


def implementation(name: str):
    """按目录解析并导入实现模块；依赖缺失时给出可执行的安装提示。"""
    spec = model_spec(name)
    missing = missing_requirements(name)
    if missing:
        raise RuntimeError(f"模型 {spec.id} 需要 {'、'.join(missing)}：请在训练环境安装"
                           f"（pip install {' '.join(missing)}）")
    return importlib.import_module(spec.implementation)


def build_model(name: str, *, classes: int, samples: int | None = None,
                params: dict | None = None):
    """按目录构建模型：**返回的模型统一接收 ``(B, 2, N)``、输出 ``(B, C)`` logits**。

    参数缺省取目录声明的默认值；未知参数、非法取值、窗口不匹配与依赖缺失
    都在这里报错（不静默兜底）。训练、验证评分、导出与预检使用同一个对象。
    """
    spec = model_spec(name)
    resolved = validate_params(spec, params)
    if samples is not None:
        check_samples(spec, int(samples))
    return implementation(name).build(num_classes=int(classes), samples=samples, params=resolved)


def export_onnx(model, path, *, classes, samples, opset=17):
    """导出 ONNX（输入 ``iq (1,2,N)``、输出 ``scores (1,C)``，softmax 写进图）并自检。"""
    from .trainer import export_onnx as _export_onnx

    return _export_onnx(model, path, classes=classes, samples=samples, opset=opset)


def train_classifier(train_x, train_y, val_x, val_y, **kwargs):
    """训练循环入口（延迟导入 ``torch`` 侧实现）。"""
    from .trainer import train_classifier as _train_classifier

    return _train_classifier(train_x, train_y, val_x, val_y, **kwargs)
