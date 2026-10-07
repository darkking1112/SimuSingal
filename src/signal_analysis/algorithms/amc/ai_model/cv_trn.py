"""``cv_trn`` 模型目录条目（纯元数据；实现在 ``training/amc_models/cv_trn.py``）。

来源：``custom/CV_TRN.py``（原文件不改，训练侧是维护副本，只替换 timm 工具导入与设备处理）。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="cv_trn",
    title="CV-TRN（复值 Transformer）",
    summary="I/Q 共享参数的复值多头自注意力 + 相对位置编码 + DB-GLU 前馈",
    implementation="amc_models.cv_trn",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS（实现内转成原要求的分帧形式）",
        "分帧嵌入：帧长 L、步长 R → (N-L)/R+1 个 token · 共享给 I 与 Q",
        "类 token ⊕ 帧 token → 4 层复值 Transformer（RMHSA + RPE + DB-GLU）",
        "复值注意力：Q/K/V 由 I、Q 共享投影，实部 QᵢKᵢᵀ+Q_qK_qᵀ、虚部 Q_qKᵢᵀ-QᵢK_qᵀ",
        "Talking-heads 头间混合 + 相对位置偏置 + 输出复乘",
        "取 I/Q 两条类 token 拼接（2·d_model）→ Linear → C",
        "训练期可选 RPO 随机相位增广（eval 恒等，导出不受影响）",
    ),
    params=(
        ParamSpec("frame_length", "int", 32, label="帧长 L",
                  help="分帧卷积的窗口长度，必须不大于窗口长度", minimum=4, maximum=256),
        ParamSpec("step_size", "int", 16, label="帧步长 R",
                  help="分帧步长（过采样比）", minimum=1, maximum=128),
        ParamSpec("d_model", "int", 64, label="token 维度 dt",
                  help="帧嵌入与注意力的模型维度", minimum=8, maximum=512),
        ParamSpec("d_mid", "int", 128, label="FFN 隐层 df",
                  help="DB-GLU 前馈的隐层宽度", minimum=8, maximum=2048),
        ParamSpec("n_head", "int", 4, label="注意力头数 h",
                  help="头数（各头维度独立，不要求整除 dt）", minimum=1, maximum=16),
        ParamSpec("layer_num", "int", 4, label="Transformer 层数 M",
                  help="复值 Transformer 层数", minimum=1, maximum=12),
        ParamSpec("dropout", "float", 0.0, label="Dropout",
                  help="注意力与 FFN 的 dropout", minimum=0.0, maximum=1.0,
                  exclusive_maximum=True),
        ParamSpec("use_rpe", "bool", True, label="相对位置编码",
                  help="是否启用 RPE（关掉会减少参数）"),
        ParamSpec("use_rpo", "bool", True, label="训练期随机相位增广",
                  help="只在训练态生效；导出与验收不受影响"),
    ),
    samples="min:32",
    notes="来源 custom/CV_TRN.py（原文献按 RML2016/2018 的 5/24 类训练）；"
          "seq_length 取数据集窗口长度（帧数与 RPE 表与该长度绑定）。",
)
