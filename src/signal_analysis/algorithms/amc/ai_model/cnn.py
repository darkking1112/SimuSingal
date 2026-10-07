"""``cnn`` 模型目录条目（纯元数据；实现在 ``training/amc_models/cnn.py``）。"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="cnn",
    title="IQCNN（步长卷积）",
    summary="三级步长一维卷积 + 时间维平均/最大池化拼接 + 两层全连接",
    implementation="amc_models.cnn",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS",
        "Conv1d 2→32 · 核 7 · 步长 2 · BatchNorm · GELU",
        "Conv1d 32→64 · 核 5 · 步长 2 · BatchNorm · GELU",
        "Conv1d 64→128 · 核 3 · 步长 2 · BatchNorm · GELU",
        "时间维平均池化 ⊕ 最大池化 → 256 维",
        "Linear 256→256 · GELU · Dropout",
        "Linear 256→C（导出时追加图内 softmax）",
    ),
    params=(
        ParamSpec("channels", "int-list", (32, 64, 128), label="卷积通道",
                  help="三级卷积的通道数", length=3, element_minimum=1),
        ParamSpec("kernel", "int", 7, label="卷积核长",
                  help="首级卷积核长，后续每级减 2", minimum=3, odd=True),
        ParamSpec("dropout", "float", 0.1, label="Dropout",
                  help="分类头 dropout", minimum=0.0, maximum=1.0, exclusive_maximum=True),
    ),
    notes="感受野由层数与步长决定；长窗口上比 TCN 轻，是默认基线。",
)
