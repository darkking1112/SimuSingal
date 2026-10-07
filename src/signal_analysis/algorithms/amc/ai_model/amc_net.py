"""``amc_net`` 模型目录条目（纯元数据；实现在 ``training/amc_models/amc_net.py``）。

来源：``custom/AMC_Net.py``（原文件不改，训练侧是维护副本：复数算子实数化）。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="amc_net",
    title="AMC-Net（自适应相关 + 多尺度 + 特征融合）",
    summary="频域自适应相关（AdaCorr）→ 多尺度卷积 → 卷积主干 → 通道注意力融合 → 分类头",
    implementation="amc_models.amc_net",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS（实现内转成原要求的分 I/Q 形式）",
        "AdaCorr：DFT → 两组 TinyMLP 分别修正实部/虚部 → IDFT 取实部 + 残差",
        "L2 归一化 → 多尺度卷积（核 3/5/7 拼接，宽度减半）",
        "卷积主干：4 级 (1,3) 卷积 + BN + ReLU（宽度不变）",
        "通道注意力融合：Q/K/V 线性 → 多头注意力（在通道维上）→ GAP",
        "Linear 256→256 · Dropout · PReLU → Linear → C",
    ),
    params=(
        ParamSpec("extend_channel", "int", 36, label="多尺度通道",
                  help="多尺度分支的总通道数（三段各 1/3），同时是主干的输入通道",
                  minimum=3, maximum=256),
        ParamSpec("conv_chan_list", "int-list", (36, 64, 128, 256), label="主干通道",
                  help="主干各级通道数，第 1 个必须等于多尺度通道数", length=4,
                  element_minimum=4),
        ParamSpec("num_heads", "int", 2, label="注意力头数",
                  help="融合模块的注意力头数；窗口长度必须能被它整除", minimum=1,
                  maximum=8),
    ),
    samples="min:64",
    notes="来源 custom/AMC_Net.py（原文献按 RML2016 128 点、11 类训练）；"
          "sig_len 取数据集窗口长度（DFT 表与注意力维度与该长度绑定），"
          "分类头的 512 维由 num_heads × conv_chan_list[-1] 推出，不是独立参数。",
)
