"""``tcn`` 模型目录条目（纯元数据；实现在 ``training/amc_models/tcn.py``）。"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="tcn",
    title="IQTCN（膨胀因果卷积）",
    summary="1×1 stem + 5 级膨胀因果残差块 + 时间维池化 + 两层全连接",
    implementation="amc_models.tcn",
    model_revision=2,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS",
        "stem Conv1d 2→64 · 核 1 · GELU",
        "5 级膨胀因果残差块（dilation 1/2/4/8/16 · 核 3 · 双卷积 + BatchNorm + Dropout）",
        "时间维平均池化 ⊕ 最大池化 → 128 维",
        "Linear 128→128 · GELU · Dropout",
        "Linear 128→C（导出时追加图内 softmax）",
    ),
    params=(
        ParamSpec("channels", "int-list", (64,), label="卷积通道",
                  help="TCN 隐层宽度（只取第 1 个值）",
                  minimum_length=1, maximum_length=3, element_minimum=1),
        ParamSpec("kernel", "int", 3, label="卷积核长", help="残差块内的卷积核长",
                  minimum=2),
        ParamSpec("dropout", "float", 0.1, label="Dropout", help="残差块与分类头的 dropout",
                  minimum=0.0, maximum=1.0, exclusive_maximum=True),
    ),
    notes="膨胀因果卷积，感受野指数增长，适合长窗口；结构版本 2 起残差块真正参与前向。",
)
