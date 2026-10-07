"""AMC 训练模型目录（纯元数据，**不导入 torch**）。

GUI、服务层与 CLI 共用这一份目录：``SPECS`` 显式聚合各模型的 :class:`ModelSpec`
（不做运行时目录扫描），模型 ID 等于实现模块名（``amc_models.<id>``），
由 ``tests/analysis/test_amc_model_catalog.py`` 强制一一对应。

新模型 = 新增两个文件：本目录下的 ``<id>.py``（声明）与
``training/amc_models/<id>.py``（torch 实现），再在这里登记一次。
"""

from __future__ import annotations

from .base import (CATALOG_VERSION, KINDS, ModelSpec, ParamSpec, check_model_samples,
                   check_samples, describe, describe_model, merge_param_sources, param_hint,
                   parse_samples_constraint, samples_constraint, spec_json, validate_params,
                   validate_value)
from .base import catalog_json as _catalog_json
from .cnn import SPEC as _CNN_SPEC
from .tcn import SPEC as _TCN_SPEC

#: 模型目录（显式聚合，顺序稳定；ID 唯一性由测试保证）
SPECS: dict[str, ModelSpec] = {spec.id: spec for spec in (_CNN_SPEC, _TCN_SPEC)}

__all__ = ["CATALOG_VERSION", "KINDS", "ModelSpec", "ParamSpec", "SPECS", "available_models",
           "catalog_json", "check_model_samples", "check_samples", "describe", "describe_model",
           "merge_param_sources", "model_spec", "param_hint", "parse_samples_constraint",
           "samples_constraint", "spec_json", "validate_params", "validate_value"]


def available_models() -> tuple[str, ...]:
    """已登记模型的 ID（= ``train_iq.py --arch`` 的取值）。"""
    return tuple(SPECS)


def model_spec(name: str) -> ModelSpec:
    """按 ID 取目录条目；未知 ID 直接报错并列出可用模型。"""
    text = str(name)
    if text not in SPECS:
        raise ValueError(f"未知架构 {text!r}，可用：{', '.join(available_models())}")
    return SPECS[text]


def catalog_json() -> str:
    """整份目录的 JSON（应用与训练源码目录握手/展示用）。"""
    return _catalog_json(SPECS)
