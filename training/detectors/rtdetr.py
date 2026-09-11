"""RT-DETR（Apache-2.0）适配器 —— **本项目推荐的首选方案**。

为什么首选 RT-DETR：

* 许可证 Apache-2.0，可随发行包一起走（README §8）；
* 它是 DETR 系**端到端**检测器，本来就不需要 NMS，天然贴合"图内解码"的要求；
* 它的后处理输出就是 ``cx, cy, w, h`` 的**归一化**坐标 + ``score`` + ``label``，
  与 :data:`normalized_boxes_v1` 逐列同构 —— 契约改写只需要补输入预处理和 TopK，
  连几何转换都不用做，是四个适配器里"距离契约最近"的一个。

接入路径：

.. code-block:: text

    git clone https://github.com/lyuwenyu/RT-DETR      # Apache-2.0
    .venv/bin/python training/export_contract.py --framework rtdetr \
        --data <数据集> --dataset-only --dataset-output /tmp/rtdetr-data --dataset-format coco
    # 用 RT-DETR 自带 trainer 训练，再用 tools/export_onnx.py 导出
    # 注意：导出时必须把坐标解码 + concat(boxes, scores, labels) 包含进图，
    #      让它成为单输出 (1, 300, 6) —— 这是唯一需要你确认的地方
    .venv/bin/python training/export_contract.py --framework rtdetr \
        --onnx runs/rtdetr.onnx --layout normalized_cxcywh --imgsz 1024 --max-boxes 32 \
        --input-scale 255          # 具体取值见下面的分支表

**输入预处理必须显式声明（:attr:`prep_required`）**：RT-DETR 有三个常用实现，
官方仓库里它们的 dataloader 配置**口径不一致**，猜错不会报错、只会静默错到底，
所以本适配器不给默认值，缺 ``--input-scale`` 直接退出并打印这张表：

.. list-table::
   :header-rows: 1

   * - 分支
     - 预处理
     - 参数
   * - ``rtdetr_paddle``（PaddleDetection）
     - ``NormalizeImage(mean=[.485,.456,.406], std=[.229,.224,.225], is_scale=True)``，
       先 ``/255`` 再减均值除方差
     - ``--input-scale 255 --input-mean .485 .456 .406 --input-std .229 .224 .225``
   * - ``rtdetr_pytorch``（v1）
     - ``ToImageTensor`` + ``ConvertDtype``：只转 float32，**数值仍是 0-255**
     - ``--input-scale 1``
   * - ``rtdetrv2_pytorch``（v2）
     - ``ConvertPILImage(dtype='float32', scale=True)``：缩放成 0-1
     - ``--input-scale 255``

三个分支都是 3 通道、ImageNet 预训练主干，所以 :attr:`channel_repeat` = 3 是通用的；
灰度复制成 3 通道后 BGR/RGB 等价，无需翻转；方形时频图下 resize 是恒等变换。
"""

from __future__ import annotations

import importlib.util

from .base import DetectorAdapter
from .registry import APACHE, AdapterInfo, register

#: RT-DETR 端到端后处理的候选框上限
DEFAULT_QUERIES = 300


@register
class RtdetrAdapter(DetectorAdapter):
    """RT-DETR（Apache-2.0，可随发行包分发）。"""

    info = AdapterInfo(
        name="rtdetr",
        title="RT-DETR（lyuwenyu / PaddleDetection）",
        license=APACHE,
        upstream="https://github.com/lyuwenyu/RT-DETR",
        notes="Apache-2.0；端到端无需 NMS；后处理输出归一化 cxcywh + score + label，"
              "与 normalized_boxes_v1 同构")
    #: 后处理输出归一化 cxcywh + score + label
    layout = "normalized_cxcywh"
    #: RT-DETR 各分支都是 3 通道（ImageNet 预训练主干）
    channel_repeat = 3
    input_scale = 1.0
    #: 三个分支的输入口径不一致，**不允许猜**：缺 ``--input-scale`` 直接报错
    prep_required = True
    prep_hint = (
        "  * rtdetr_paddle（PaddleDetection）：NormalizeImage(mean=[.485,.456,.406], "
        "std=[.229,.224,.225], is_scale=True) → --input-scale 255 --input-mean .485 .456 .406 "
        "--input-std .229 .224 .225\n"
        "  * rtdetr_pytorch（v1）：ToImageTensor + ConvertDtype → float32 仍是 0-255，"
        "不给 mean/std → --input-scale 1\n"
        "  * rtdetrv2_pytorch（v2）：ConvertPILImage(dtype=float32, scale=True) → 0-1，"
        "不给 mean/std → --input-scale 255")
    dataset_format = "coco"
    weights_hint = "RT-DETR 导出的单输出 (1, 300, 6) ONNX"
    native_max_boxes = DEFAULT_QUERIES

    def is_available(self):
        return importlib.util.find_spec("rtdetr") is not None

    def install_hint(self):
        return ("git clone https://github.com/lyuwenyu/RT-DETR\n"
                "  .venv/bin/python -m pip install -e RT-DETR\n"
                "  导出时请把坐标解码与 concat(boxes, scores, labels) 包进图，"
                "使其成为单输出 (1, 300, 6)；否则请先用 torch 侧 wrapper 合并三个输出"
                "（完整步骤见 training/README.md §7）")

    def load_model(self, weights, *, device="cpu", image_size=1024, max_boxes=32):
        raise SystemExit(
            "RT-DETR 请用它自带的 tools/export_onnx.py 导出单输出 ONNX 后走 --onnx 模式。\n  "
            + self.install_hint())
