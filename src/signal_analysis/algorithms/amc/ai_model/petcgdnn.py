"""``petcgdnn`` 模型目录条目（纯元数据；实现在 ``training/amc_models/petcgdnn.py``）。

来源：``custom/PETCGDNN.py``（原文件不改，由实现侧的包装层接入项目契约）。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="petcgdnn",
    title="PETCGDNN（相位旋转 + CNN + GRU）",
    summary="PET 相位旋转预处理 → 两级二维卷积 → GRU → 全连接",
    implementation="amc_models.petcgdnn",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS（原实现内部转成 (B, N, 2)）",
        "PET 相位旋转：Linear 2N→1 学一个旋转角 θ，用 sin/cos 混 I/Q 两路",
        "Conv2d 1→75 · 核 (8,2) · valid · ReLU（时间 N→N-7）",
        "Conv2d 75→25 · 核 (5,1) · valid · ReLU（时间 N→N-11）",
        "GRU 25→隐藏宽度（取末步）",
        "Linear 隐藏宽度→C（导出时追加图内 softmax）",
    ),
    params=(
        ParamSpec("hidden_size", "int", 128, label="GRU 隐藏宽度",
                  help="GRU 的隐层宽度", minimum=8, maximum=512),
    ),
    samples="min:16",
    notes="PET 的旋转层把窗口长度写进了权重形状（Linear 2N→1），因此构建时必须给出"
          "数据集窗口长度：包装层用 samples 作为 frame_length，不提供独立可调参数。",
)
