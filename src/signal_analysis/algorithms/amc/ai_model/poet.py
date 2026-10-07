"""``poet`` 模型目录条目（纯元数据；实现在 ``training/amc_models/poet.py``）。

来源：``custom/POET.py``（原文件不改，训练侧是维护副本，只替换 timm 工具导入）。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="poet",
    title="POET（多尺度复值 Transformer）",
    summary="训练期物理增广 + 逐样本 AGC + 多尺度嵌入 + 复值 Transformer + 判别分类头",
    implementation="amc_models.poet",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS（实现内转成原要求的分 I/Q 形式）",
        "训练期物理增广：随机相位/幅度/噪声（eval 恒等，导出不受影响）",
        "InstanceAGC：逐样本功率归一化（确定性计算，训练与推理一致）",
        "多尺度嵌入：scales 个卷积尺度 + 跨尺度注意力 → N/2 个 token",
        "类 token ⊕ 位置编码 → 复值 Transformer 层（EfficientCMHSA + 复值前馈）",
        "最终 LayerNorm → 判别分类头（类 token 与 token 统计量拼接）→ C",
    ),
    params=(
        ParamSpec("d_model", "int", 80, label="模型维度",
                  help="token 与注意力宽度", minimum=8, maximum=512),
        ParamSpec("n_head", "int", 4, label="注意力头数",
                  help="必须整除模型维度", minimum=1, maximum=16, divides="d_model"),
        ParamSpec("d_ff", "int", 320, label="前馈隐层",
                  help="复值前馈的隐层宽度", minimum=8, maximum=2048),
        ParamSpec("layer_num", "int", 3, label="Transformer 层数",
                  help="复值 Transformer 层数", minimum=1, maximum=12),
        ParamSpec("dropout", "float", 0.15, label="Dropout",
                  help="注意力与前馈的 dropout", minimum=0.0, maximum=1.0,
                  exclusive_maximum=True),
        ParamSpec("drop_path", "float", 0.15, label="随机深度",
                  help="逐层 drop path 的上限（0 = 关闭）", minimum=0.0, maximum=1.0,
                  exclusive_maximum=True),
        ParamSpec("use_rpe", "bool", True, label="相对位置编码",
                  help="是否启用 RPE"),
        ParamSpec("scales", "int-list", (1, 2, 4, 8), label="多尺度",
                  help="多尺度卷积的尺度列表", minimum_length=1, maximum_length=6,
                  element_minimum=1),
    ),
    samples="min:32",
    notes="来源 custom/POET.py（原文献按 11/24 类训练）；seq_length 取数据集窗口长度"
          "（token 数与位置编码表与该长度绑定）。",
)
