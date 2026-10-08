"""新模型**实现模块**模板：未登记、不会被加载（复制成 ``training/amc_models/<id>.py``）。

模型 ID 有硬约束：**目录条目 id = 实现模块名 = 实现文件名**（``spec.implementation``
写成 ``amc_models.<id>``），由 ``tests/analysis/test_amc_model_catalog.py`` 强制。
接入规则、判定路径与排查表见
``docs/电磁信号分析和识别/algorithms/调制识别_custom模型接入指南.md``。

三条硬性约定：

1. :func:`build` 返回的模型**直接接收契约布局 ``(B, 2, N)``**（单位 RMS 的 I/Q 窗口）、
   输出 ``(B, C)`` **logits**；softmax 由导出层追加，不要写进模型；
2. 参数的默认值与取值范围只在目录条目（``ai_model/<id>.py`` 的 ``ParamSpec``）里声明，
   这里只读 ``params``，不得再写一套默认值；
3. 结构依赖窗口长度时必须显式要求 ``samples``（缺失即报错），不要静默兜底。
"""

from __future__ import annotations

from torch import nn

#: 路径 A（直接包装 ``custom/`` 原文件）用的两个工具；走路径 B（维护副本）时删掉 _custom
from ._adapters import CHANNELS_LAST, adapt
from ._custom import load_custom_module


def build(*, num_classes: int, samples: int | None, params: dict) -> nn.Module:
    """按目录参数构建模型：输入 ``(B, 2, N)``、输出 ``(B, C)`` logits。

    路径 A —— 原文件"能直接用"：只做"类名 + 参数映射"，原文件保持不改。示例把
    ``ExampleNet`` 换成 ``custom/`` 下的真实文件名与类名，参数名与目录条目一一对应。

    路径 B —— 原文件需要改写（timm 导入、设备写死、复数算子、结构参数写死…）：删掉
    下面几行，把改写后的模型类放进**本文件**，在这里构造，并在模块 docstring 里
    逐条列出"相对原文件的差异"（差异可比对、数值差异要量化）。
    """
    # 结构依赖窗口长度时保留这段（MCLDNN 这类与窗口无关的模型整段删掉）：
    # if samples is None:
    #     raise ValueError("模型 <id> 的结构依赖窗口长度：请提供 samples"
    #                      "（训练链路会从数据集契约传入）")

    module = load_custom_module("ExampleNet")
    model = module.ExampleNet(num_classes=int(num_classes),
                              hidden_size=int(params["hidden_size"]))
    # 原实现吃 (B, N, 2)、或目录 input_layout 声明为 iq_channels_last_v1 时才需要这层：
    return adapt(model, CHANNELS_LAST)
