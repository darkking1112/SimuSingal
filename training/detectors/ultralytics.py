"""Ultralytics YOLO11 / YOLO26 适配器（**AGPL-3.0，默认拒绝导出**）。

先说结论，免得踩坑（数据来自 Ultralytics 官方文档，2026-09-11 核对）：

* **YOLO11 与 YOLO26 都是 Ultralytics 出品、都是 AGPL-3.0**，
  它们**不是** YOLOX，也不是 RT-DETR；用它们就绕不开 AGPL 的传染性；
* 若确实要用 Ultralytics，**选 YOLO26s 而不是 YOLO11s**：
  - 一对一检测头（``nms=False``）端到端输出 ``(N, 300, 6)``，本身就是
    ``xyxy + score + class``，与 :data:`normalized_boxes_v1` 只差一次几何转换，
    而 YOLO11 只有一对多头 ``(1, 4+nc, 8400)``，需要图内 argmax 才能落成单类置信度；
  - YOLO26 去掉了 DFL，导出图更短；官方实测 CPU ONNX 推理最多快 43%；
  - 同等开销下 mAP50-95 48.6 vs 47.0。

**输入预处理（已对源码核对）**：``engine/predictor.py::BasePredictor.preprocess``
的顺序是 BHWC→BCHW →（3 通道时）``im.flip(1)`` 把 BGR 翻成 RGB → ``.div_(255)``。
**没有** ImageNet 均值方差。对应 :attr:`input_scale` = 255.0、
:attr:`channel_repeat` = 3、``mean``/``std`` = None。
我们的输入是单通道灰度复制成 3 通道，BGR/RGB 翻转对结果是恒等变换，
方形时频图下 letterbox 也是恒等变换，所以两项都不用额外处理。

因此本适配器**默认只支持端到端头**，并强制 ``--allow-copyleft`` 才会真正导出。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .base import DetectorAdapter
from .registry import AGPL, AdapterInfo, register

#: YOLO26 一对一检测头固定输出的候选框数
END2END_MAX_BOXES = 300


@register
class UltralyticsAdapter(DetectorAdapter):
    """Ultralytics YOLO11 / YOLO26（AGPL-3.0）。"""

    info = AdapterInfo(
        name="ultralytics",
        title="Ultralytics YOLO11 / YOLO26",
        license=AGPL,
        upstream="https://github.com/ultralytics/ultralytics",
        notes="AGPL-3.0：仅供内部基线评测，权重与代码都不要带进发行包；"
              "若要用 Ultralytics，选 YOLO26s 并导出 nms=False 的端到端头")
    #: 端到端头输出 (N, 300, 6) = xyxy + score + class，像素坐标
    layout = "pixel_xyxy"
    #: Ultralytics 内部把输入当 BGR 三通道，``im.flip(1)`` 再转 RGB
    channel_repeat = 3
    #: ``BasePredictor.preprocess`` 全流程只有 ``.div_(255)`` + BGR→RGB，
    #: **不做** ImageNet 均值方差（早期版本曾用，现已去掉）
    input_scale = 255.0
    mean = None
    std = None
    dataset_format = "yolo"
    weights_hint = "yolo26s.pt（端到端头；不要用 yolo11*.pt 的一对多头）"
    #: 端到端头的候选框上限，供 CLI 校验
    native_max_boxes = END2END_MAX_BOXES

    def is_available(self):
        return importlib.util.find_spec("ultralytics") is not None

    def install_hint(self):
        return (".venv/bin/python -m pip install ultralytics\n"
                "  （AGPL-3.0；装之前请确认内部评测用法，并准备 --allow-copyleft）")

    def load_model(self, weights, *, device="cpu", image_size=1024, max_boxes=32):
        raise SystemExit(
            "Ultralytics 走的是「官方 exporter 产出 nms=False 端到端 ONNX」的路径："
            "直接传 .pt 给本适配器即可（会自动调用 model.export），"
            "或先用它导出再把 ONNX 交给 --onnx 模式。\n  " + self.install_hint())

    def export_native_onnx(self, weights, *, image_size, output, opset=17):
        """用 Ultralytics 自己的导出器产出 ``nms=False`` 的端到端 ONNX。"""
        from ultralytics import YOLO

        model = YOLO(str(weights))
        path = Path(output) / "native.onnx"
        exported = model.export(format="onnx", imgsz=int(image_size), opset=int(opset),
                                nms=False, half=False, dynamic=False)
        exported = Path(exported)
        if exported != path:
            path.write_bytes(exported.read_bytes())
        return path


def describe_versions():
    """返回官方给出的 YOLO11s / YOLO26s 参数对比（文档与报错提示共用）。"""
    return {
        "yolo11s": {"mAP50-95": 47.0, "params_m": 9.4, "gflops": 21.6, "license": AGPL,
                    "e2e_head": False},
        "yolo26s": {"mAP50-95": 48.6, "params_m": 9.5, "gflops": 20.9, "license": AGPL,
                    "e2e_head": True},
    }
