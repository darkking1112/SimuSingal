"""检测器原始输出的布局说明表（**无第三方依赖**）。

本模块只回答一个问题：**原始张量每一列是什么**。它被
:mod:`detectors.torch_export`（torch 侧包装）与 :mod:`detectors.onnx_contract`
（ONNX 图改写）共用，因此刻意不导入 ``torch`` / ``onnx``。

目标契约（与 :mod:`signal_analysis.ml.decode` 完全一致，冻结为
``normalized_boxes_v1``）：

* 形状 ``(1, N, 6)``，每行 ``[x_center, y_center, width, height, confidence, class]``；
* 前四列按图像宽高归一化到 ``[0, 1]``，坐标是**边框**而不是像素中心；
* ``x`` 自左向右；``y`` **自上向下**，而图像行 0 对应 ``+fs/2``。

坐标为什么要单独强调
--------------------

第三方检测器（YOLO 系、RT-DETR、YOLOX）统一使用"左上角为原点、``y`` 向下"
的图像坐标，而本项目的时频图行 0 是 ``+fs/2``（
:func:`signal_analysis.ml.tensor.detection_image` 的排布）。两者**恰好同向**，
因此**不需要任何翻转**：只要标签来自
:func:`signal_analysis.ml.tensor.band_to_box`，同一套 ``(x, y, w, h)`` 就可以
在训练侧与推理侧直接互通。一旦有人"顺手"写了 ``y = 1 - y``，框会整体上下翻转，
而形状校验查不出来——这就是 README §7 把"绝不另写一份 y 坐标公式"列为第一条的原因。
"""

from __future__ import annotations

#: 契约列数（等于 ``signal_analysis.ml.decode.BOX_COLUMNS``）
CONTRACT_COLUMNS = 6

#: 契约名称
CONTRACT_NAME = "normalized_boxes_v1"

#: 输出布局表。每项必须给出：列数、几何表示、坐标量纲、是否有类别列。
LAYOUTS = {
    "normalized_cxcywh": {
        "columns": 6,
        "geometry": "cxcywh",
        "scale": "normalized",
        "has_class": True,
        "columns_desc": "x_center, y_center, width, height, confidence, class",
        "description": "本项目契约本身；已经是归一化边框，只做截断与限幅",
        "examples": "自研检测头、已经写过契约转换的导出图",
    },
    "pixel_xyxy": {
        "columns": 6,
        "geometry": "xyxy",
        "scale": "pixel",
        "has_class": True,
        "columns_desc": "x1, y1, x2, y2, confidence, class",
        "description": "像素坐标的两角点表示；需要除以图像边长并转成中心式",
        "examples": "Ultralytics YOLO26 端到端头（export(..., nms=False) 的 (N, 300, 6)）、"
                    "YOLOX 已在图内完成解码的导出",
    },
    "pixel_cxcywh": {
        "columns": 6,
        "geometry": "cxcywh",
        "scale": "pixel",
        "has_class": True,
        "columns_desc": "x_center, y_center, width, height, confidence, class",
        "description": "像素坐标的中心式表示；只需要除以图像边长",
        "examples": "RT-DETR 解码后输出、Ultralytics export(..., xywh=True)",
    },
    "pixel_xyxy_objectness": {
        "columns": 5,
        "geometry": "xyxy",
        "scale": "pixel",
        "has_class": False,
        "columns_desc": "x1, y1, x2, y2, objectness",
        "description": "单类、无类别列；置信度取 objectness，类别恒为 0",
        "examples": "单类检测器的原始 head 输出",
    },
}


def layout_spec(name):
    """取布局说明；未知布局时列出全部可用值（供 CLI 与报错复用）。"""
    key = str(name).strip().lower()
    if key not in LAYOUTS:
        available = "、".join(layout_names())
        raise ValueError(f"未知的输出布局 {name!r}；可用取值：{available}")
    spec = dict(LAYOUTS[key])
    spec["name"] = key
    return spec


def layout_names():
    """全部布局名（排序，保证报错与文档稳定）。"""
    return tuple(sorted(LAYOUTS))


def describe_layouts():
    """布局表的可打印描述（``--list-layouts`` 与文档生成共用）。"""
    return [
        {
            "name": key,
            "columns": spec["columns"],
            "columns_desc": spec["columns_desc"],
            "geometry": spec["geometry"],
            "scale": spec["scale"],
            "has_class": spec["has_class"],
            "description": spec["description"],
            "examples": spec["examples"],
        }
        for key, spec in sorted(LAYOUTS.items())
    ]
