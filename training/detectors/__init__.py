"""检测器接入框架（``training/detectors/``，不随 wheel 分发）。

本包解决的问题只有一个：**把任意第三方检测器的原始输出，压成项目冻结契约**
``normalized_boxes_v1``，并且把预处理写进导出图，让"训练-推理口径"不可能分叉。

只做一个方向
------------

* **训练**仍由各框架自己的训练器负责（``tiny`` 除外，它自带训练循环）；
* 本包负责 **数据集格式转换 → 加载权重 → 包装 → 导出 ONNX → 生成清单**，
  每一步都可单独调用、单独测试。

三层结构
--------

``dataset`` / ``labels``
    数据集读取与**原生标注格式**导出（Ultralytics YOLO、YOLOX）。标签一律由
    :func:`signal_analysis.ml.tensor.band_to_box` 生成，绝不另写一份 y 公式。
``contract`` / ``torch_export`` / ``onnx_contract``
    布局说明表（无依赖）→ torch 侧包装（把缩放/归一化/通道复制写进图）→
    ONNX 图改写（给只能用自己的导出器的框架用）。
``registry`` / ``base`` / ``yolox`` / ``rtdetr`` / ``ultralytics`` / ``tiny``
    可插拔的 ``--arch``；每个适配器自带许可证与"是否可分发"的声明。

依赖策略
--------

``torch`` 与 ``onnx`` 都是**惰性导入**：只查注册表、只导原生标注格式不需要
安装它们；真正导出时才要求安装，并给出可执行的安装命令。
"""

from __future__ import annotations

from .registry import AGPL, APACHE, AdapterInfo, describe_all, lookup, names, register

# 导入适配器模块即完成注册（这些模块本身不导入 torch / onnx）
from . import rtdetr, tiny, ultralytics, yolox  # noqa: E402,F401

__all__ = [
    "AGPL",
    "APACHE",
    "AdapterInfo",
    "describe_all",
    "lookup",
    "names",
    "register",
]
