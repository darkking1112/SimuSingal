"""模型**目录条目**模板：未登记、不会被加载（复制成 ``src/.../ai_model/<id>.py``）。

本目录是**纯元数据、不许 import torch**（桌面应用冻结包不含 torch，由
``tests/analysis/test_amc_model_catalog.py`` 强制）；真正的网络实现在
``training/amc_models/<id>.py``（见那边的 ``_template.py``）。登记位置：同包
``__init__.py`` 的 ``SPECS`` 元组；模型增删或参数 schema 变更时 ``CATALOG_VERSION`` +1
（应用与训练源码目录按它握手）。字段与规则的完整说明见
``docs/电磁信号分析和识别/algorithms/调制识别_custom模型接入指南.md``。
"""

from __future__ import annotations

from .base import ModelSpec, ParamSpec

SPEC = ModelSpec(
    id="example",                              # 模型 ID：文件名 = 实现模块名 = --arch 取值
    title="示例模型（论文简称）",
    summary="一句话结构摘要（界面模型列表显示）",
    implementation="amc_models.example",       # 必须等于 f"amc_models.{id}"
    model_revision=1,                          # 结构或前向语义变更时 +1
    layers=(                                   # 「模型架构」只读区逐行显示
        "输入 iq (B, 2, N) · 单位 RMS",
        "……（每行一层/一段，写清核大小、通道数等可观测量）",
        "Linear → C（导出时追加图内 softmax）",
    ),
    params=(
        # 默认值是唯一事实源；kind 取 int / float / int-list / bool / choice
        ParamSpec("hidden_size", "int", 128, label="隐层宽度",
                  help="界面与 CLI 帮助共用的解释", minimum=8, maximum=512),
        ParamSpec("channels", "int-list", (32, 64), label="通道数",
                  help="逐层通道宽度", element_minimum=1, length=2),  # 元素须为奇数时加 element_odd
        ParamSpec("dropout", "float", 0.5, label="Dropout",
                  minimum=0.0, maximum=1.0, exclusive_maximum=True),
        ParamSpec("norm", "choice", "bn", label="归一化", choices=("bn", "ln")),
        ParamSpec("bias", "bool", True, label="使用偏置"),
        # 参数间整除关系写在被约束的一方：ParamSpec("heads", "int", 4, divides="width")
    ),
    samples="min:16",                          # any / exact:<N> / min:<N>（数据窗口约束）
    input_layout="iq_channels_first_v1",       # 或 iq_channels_last_v1（由 adapt() 转置）
    requires=("torch",),                       # 训练环境缺依赖时在导入实现前报错
    exportable=True,                           # 未通过 ONNX 导出与验收前必须标 False
    notes="来源（custom/xxx.py）、与原文的差异、窗口/耗时提醒等",
    # 参数间的复杂约束用 cross_validate=lambda params: ...（失败 raise ValueError）
)
