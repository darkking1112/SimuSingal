"""``mcldnn`` 模型目录条目（纯元数据；实现在 ``training/amc_models/mcldnn.py``）。

来源：``custom/MCLDNN.py``（原文件不改，由实现侧的包装层接入项目契约）。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="mcldnn",
    title="MCLDNN（多通道 LSTM-DNN）",
    summary="I/Q 图像分支 + 双因果卷积分支 → 二维卷积融合 → 双层 LSTM → 全连接",
    implementation="amc_models.mcldnn",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS（原实现直接吃该布局）",
        "分支一：Conv2d 1→50 · 核 (2,8) · same · ReLU（I/Q 视作图像）",
        "分支二/三：I、Q 各自过因果 Conv1d 1→50 · 核 8 · ReLU",
        "两支特征堆叠为 2 行 → Conv2d 50→50 · 核 (1,8) · same · ReLU",
        "拼接 → Conv2d 100→100 · 核 (2,5) · valid（高度 2→1，时间 N→N-4）",
        "LSTM 100→128 → LSTM 128→128（取末步）",
        "Linear 128→128 · SELU · Dropout → Linear 128→128 · SELU · Dropout",
        "Linear 128→C（导出时追加图内 softmax）",
    ),
    params=(
        ParamSpec("dropout_rate", "float", 0.5, label="Dropout",
                  help="两层全连接之间的丢弃率", minimum=0.0, maximum=1.0,
                  exclusive_maximum=True),
    ),
    samples="min:16",
    notes="原文献按 11 类训练，这里用数据集的类别数与窗口重新训练；LSTM 沿时间步展开，"
          "窗口越长越慢（1024 点实测约 0.9 s/训练步，批 8）。",
)
