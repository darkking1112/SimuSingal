"""兼容转发层：模型定义与训练/导出已迁移到 :mod:`amc_models`（模型目录驱动）。

保留旧名字与旧签名，供历史脚本、测试与文档引用继续使用；新代码请直接用
``amc_models``（实现注册表）与 ``signal_analysis.algorithms.amc.ai_model``
（模型目录：结构、参数约束与版本）。

本文件与迁移前一样需要 torch（``.[train]``）；纯 Python 的目录查询见
``signal_analysis.algorithms.amc.ai_model``（GUI 与 ``--help`` 用它，不装 torch 也能跑）。
"""

from __future__ import annotations

from amc_models import available_models
from amc_models import build_model as _build_model
from amc_models import export_onnx, train_classifier  # noqa: F401  （兼容导出）
from amc_models._adapters import SoftmaxWrapper as SoftmaxClassifier  # noqa: F401
from amc_models.cnn import DEFAULT_CHANNELS, DEFAULT_KERNEL, IQCNN  # noqa: F401
from amc_models.tcn import (DEFAULT_TCN_CHANNELS, DEFAULT_TCN_KERNEL,  # noqa: F401
                            DEFAULT_TCN_LEVELS, IQTCN)
from amc_models.trainer import INPUT_NAME, OUTPUT_NAME, TOLERANCE  # noqa: F401

from signal_analysis.algorithms.amc.ai_model import SPECS

#: 架构名称（``train_iq.py --arch`` 的取值）
ARCHITECTURES = available_models()
#: 架构版本：结构或前向语义变更时递增，随实验记录写入（历史实验按此区分口径）
ARCH_REVISIONS = {spec.id: spec.model_revision for spec in SPECS.values()}


def build_model(arch, classes, *, channels=None, kernel=None, dropout=0.1):
    """旧签名：按 ``arch`` 构建模型（未 softmax，输出 logits）。"""
    params = {"dropout": dropout} if dropout is not None else {}
    if channels is not None:
        params["channels"] = tuple(channels)
    if kernel is not None:
        params["kernel"] = int(kernel)
    return _build_model(arch, classes=classes, params=params)
