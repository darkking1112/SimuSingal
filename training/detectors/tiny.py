"""自研最小检测头（``training/tiny_detector.py``）的适配器。

``tiny`` 的价值不是精度，而是**一次契约合规的导出**：它把
"数据集 → 训练 → 导出 → 清单 → 推理/评测"整条链路先跑通，
给后续接入 YOLOX / RT-DETR 提供一个可对照的基线（README §7 的建议顺序）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .base import DetectorAdapter
from .registry import APACHE, AdapterInfo, register

_TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))


@register
class TinyAdapter(DetectorAdapter):
    """内置最小无锚框检测头；输出本身就是契约布局。"""

    info = AdapterInfo(
        name="tiny",
        title="内置最小检测头（training/tiny_detector.py）",
        license=APACHE,
        upstream="本仓库 training/tiny_detector.py",
        notes="冒烟基线与链路验收用；真实精度请接入 RT-DETR / YOLOX")
    layout = "normalized_cxcywh"
    channel_repeat = 1
    input_scale = 1.0
    dataset_format = "yolo"
    weights_hint = "train_yolox.py --save-state 产生的 state_dict（可省略，表示随机初始化）"
    #: 与 train_yolox.py 的默认值保持一致
    width = 32
    strides = 4

    def is_available(self):
        return importlib.util.find_spec("torch") is not None

    def install_hint(self):
        return '.venv/bin/python -m pip install -e ".[train]"'

    def configure(self, **options):
        self.width = int(options.get("width") or self.width)
        self.strides = int(options.get("strides") or self.strides)
        return self

    def load_model(self, weights, *, device="cpu", image_size=1024, max_boxes=32):
        import torch

        import tiny_detector  # type: ignore[import-not-found]

        model = tiny_detector.TinyDetector(max_boxes=int(max_boxes), classes=1,
                                          width=self.width, strides=self.strides)
        if weights:
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
        return model.to(device).eval()
