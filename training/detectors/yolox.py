"""YOLOX（Megvii，Apache-2.0）适配器 —— **本项目推荐的落地方案之一**。

为什么推荐 YOLOX：许可证是 Apache-2.0，与 README §8 的"权重优先选
Apache-2.0 框架"一致，可以随发行包一起走；YOLOX 的导出图在
``decode_in_inference=True`` 下已经完成网格解码，输出为
``(1, N, 5 + nc)`` 的 ``cx, cy, w, h, obj, cls...``（**像素坐标、置信度已过
sigmoid**），因此只要把几何量除以 ``imgsz`` 就落进契约布局。

**输入预处理（已对源码核对，别凭记忆写）**：

* 官方 ``yolox/data/data_augment.py::preproc``：pad 值 114 → 等比 resize →
  HWC→CHW → ``dtype=np.float32``，**数值仍是 0-255 的 BGR，不除 255、
  不减均值**（``/255`` + ImageNet 归一化只存在于 ``ValTransform(legacy=True)``
  这条旧分支，官方 ``demo/onnx_inference.py`` 用的是 ``legacy=False``）；
* 所以 :attr:`channel_repeat` = 3、:attr:`input_scale` = 1.0、``mean``/``std`` = None。
  我们的输入是单通道灰度复制成 3 通道，BGR 与 RGB 等价，无需翻转。
* 方形时频图（``imgsz`` 等于图像边长）下 letterbox 是恒等变换，无需补边。

接入路径（README §7 的顺序）：

.. code-block:: text

    git clone https://github.com/Megvii-BaseDetection/YOLOX
    .venv/bin/python -m pip install -e YOLOX          # 只在本机开发环境用
    # 1) 用 YOLOX 自带 trainer 在原生数据集上训练（先做数据集导出）
    .venv/bin/python training/export_contract.py --framework yolox \
        --data workspace_data/analysis/... --dataset-only --dataset-output /tmp/yolox-data
    # 2) 用 YOLOX 自带 tools/export_onnx.py 导出（保持 decode_in_inference=True）
    # 3) 做契约改写
    .venv/bin/python training/export_contract.py --framework yolox \
        --onnx runs/yolox.onnx --layout pixel_cxcywh --imgsz 1024 --max-boxes 32 --data <数据集>

**单类假设**：契约的第 4 列是"置信度"。YOLOX 单类（``labels`` 只有 1 个）时
导出图的第 4 列是 ``obj`` 而第 5 列是 ``cls``，本适配器取 ``obj`` 作为置信度，
略偏乐观（真值是 ``obj × cls``）。多类时请在导出前把 ``obj × cls`` 乘好，
否则请改用 torch 侧 wrapper（``train_yolox.py --arch yolox``）。
"""

from __future__ import annotations

import importlib.util

from .base import DetectorAdapter
from .registry import APACHE, AdapterInfo, register


@register
class YoloxAdapter(DetectorAdapter):
    """Megvii YOLOX（Apache-2.0，可随发行包分发）。"""

    info = AdapterInfo(
        name="yolox",
        title="YOLOX（Megvii）",
        license=APACHE,
        upstream="https://github.com/Megvii-BaseDetection/YOLOX",
        notes="Apache-2.0，可随产品分发；导出图 decode_in_inference=True 时输出已解码，"
              "单类下第 4 列取 obj 作为置信度（真值为 obj×cls，略偏乐观）")
    #: 导出图输出 cx, cy, w, h, obj, cls（像素坐标）
    layout = "pixel_cxcywh"
    #: YOLOX 的 ``preproc`` 输出是 **3 通道 BGR**（灰度复制成 3 通道后 BGR/RGB 等价）
    channel_repeat = 3
    #: YOLOX 官方 demo/onnx_inference 用 ``ValTransform(legacy=False)``：
    #: 只做 pad(114) + resize + HWC→CHW，**不做 /255、不做 ImageNet 归一化**
    input_scale = 1.0
    mean = None
    std = None
    dataset_format = "coco"
    weights_hint = "YOLOX tools/export_onnx.py 导出的 .onnx（decode_in_inference=True）"

    def is_available(self):
        return importlib.util.find_spec("yolox") is not None

    def install_hint(self):
        return ("git clone https://github.com/Megvii-BaseDetection/YOLOX\n"
                "  .venv/bin/python -m pip install -e YOLOX\n"
                "  训练后用它自带的 tools/export_onnx.py 导出，再用 --onnx 模式做契约改写"
                "（完整步骤见 training/README.md §7）")

    def load_model(self, weights, *, device="cpu", image_size=1024, max_boxes=32):
        raise SystemExit(
            "YOLOX 建议用它自带的 exporter 产出 ONNX 后走 --onnx 模式，"
            "这样权重与上游完全一致、也不需要把 AGPL/第三方依赖装进产品环境。\n  "
            + self.install_hint())
