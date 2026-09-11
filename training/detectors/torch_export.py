"""torch 侧契约导出：把**预处理与输出几何转换写进导出图**。

这是整套接入框架的核心。README §7 有两条硬要求：

* **归一化必须写进导出图**（作为网络第一层），不要在 Python 侧偷偷改，
  否则 ``detection_image`` 的数值语义就分叉了；
* **原生输出必须用 wrapper 在 ``forward`` 里切/拼成 6 列**，
  不要在 Python 侧解码。

本模块的 :func:`build_contract_head` 就是那个 wrapper：

.. code-block:: text

    images (1, 1, H, W) float32 [0, 1]
        │
        ├─ repeat  (可选) 单通道 → 3 通道，喂给 ImageNet 预训练主干
        ├─ * input_scale   (可选) [0,1] → [0,255]，YOLO 系在 Python 侧就是这么干的
        ├─ (x - mean) / std (可选) ImageNet 标准化，作为图内第一层
        │
        └─ detector(...)  →  原始输出（布局由 --layout 声明）
                │
                ├─ 几何转换：xyxy → cxcywh、像素 → 归一化
                ├─ 限幅：cx, cy ∈ [0,1]，w, h ∈ [1e-4, 1]
                └─ 按置信度 TopK 截断到 max_boxes
                │
        detections (1, max_boxes, 6) float32

输出列序与 :data:`signal_analysis.ml.decode.BOX_COLUMNS` 完全一致，
所以推理端只需要 :func:`signal_analysis.ml.decode.parse_model_output`，
不需要任何模型相关的解码分支。``y`` 方向不做翻转（见 :mod:`.contract` 的说明）。

``torch`` 始终惰性导入：本模块可以被 import，但只有真正构建/导出时才要求安装。
"""

from __future__ import annotations

from pathlib import Path

from .contract import CONTRACT_COLUMNS, layout_spec

#: 宽度/高度下限（与 ``signal_analysis.ml.tensor.band_to_box`` 同一量级）
MIN_EXTENT = 1e-4

#: ImageNet 标准化常量（YOLOX / RT-DETR 的默认预处理）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def require_torch():
    """惰性导入 torch，缺失时给出可执行的安装命令。"""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 取决于本地环境
        raise SystemExit(
            '导出需要 torch，请先安装：.venv/bin/python -m pip install -e ".[train]"') from exc
    return torch


def _as_tensor(torch, value, name):
    if value is None:
        return None
    data = [float(item) for item in value]
    if not data or not all(abs(item) > 0 for item in data):
        raise SystemExit(f"{name} 不能含有 0（会除零）；如需关闭标准化请不要传该参数")
    return torch.tensor(data, dtype=torch.float32)


def build_contract_head(detector, *, layout="pixel_xyxy", image_size, max_boxes=32,
                        channel_repeat=1, input_scale=1.0, mean=None, std=None, torch=None):
    """把任意检测器包成"契约合规"的 ``nn.Module``。

    ``detector`` 必须是一个 ``nn.Module``，接受 ``(B, C, H, W)`` 并返回
    **单个张量** ``(B, N, C_cols)``（列含义由 ``layout`` 声明）。
    返回的模块输入是契约输入 ``(B, 1, H, W)``，输出是契约输出
    ``(B, max_boxes, 6)``。
    """
    torch = torch or require_torch()
    spec = layout_spec(layout)
    size = float(image_size)
    cap = int(max_boxes)
    if cap < 1 or cap > 256:
        raise SystemExit("--max-boxes 应在 1～256 之间（推理端 max_detections 上限为 256）")
    channels = int(channel_repeat)
    if channels not in (1, 3):
        raise SystemExit("--channel-repeat 只支持 1（单通道主干）或 3（ImageNet 预训练主干）")
    scale = float(input_scale)
    mean_tensor = _as_tensor(torch, mean, "--input-mean")
    std_tensor = _as_tensor(torch, std, "--input-std")
    if mean_tensor is not None and std_tensor is None:
        raise SystemExit("给了 --input-mean 就必须同时给 --input-std")
    if mean_tensor is not None and mean_tensor.numel() not in (1, channels):
        raise SystemExit(f"--input-mean 应为 1 个或 {channels} 个数值（按通道）")

    geometry = spec["geometry"]
    pixel_scale = spec["scale"] == "pixel"
    has_class = bool(spec["has_class"])
    columns = int(spec["columns"])

    class ContractHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detector = detector
            self.channel_repeat = channels
            self.input_scale = scale
            self.normalise = mean_tensor is not None
            if self.normalise:
                self.register_buffer("input_mean", mean_tensor.clone())
                self.register_buffer("input_std", std_tensor.clone())

        def _prepare(self, images):
            out = images
            if self.channel_repeat > 1:
                out = out.repeat(1, self.channel_repeat, 1, 1)
            if self.input_scale != 1.0:
                out = out * self.input_scale
            if self.normalise:
                shape = (1, out.shape[1], 1, 1)
                out = (out - self.input_mean.reshape(shape)) / self.input_std.reshape(shape)
            return out

        def _to_contract(self, raw):
            if len(raw.shape) == 2:
                raw = raw.unsqueeze(0)
            if len(raw.shape) != 3 or raw.shape[-1] != columns:
                raise SystemExit(
                    f"检测器输出应为 (1, N, {columns})（布局 {spec['name']}），"
                    f"实际为 {tuple(raw.shape)}；请用 --probe 查看真实形状后改 --layout")
            first, second = raw[:, :, 0], raw[:, :, 1]
            if geometry == "xyxy":
                third, fourth = raw[:, :, 2], raw[:, :, 3]
                cx = (first + third) * 0.5
                cy = (second + fourth) * 0.5
                width = third - first
                height = fourth - second
            else:
                cx, cy, width, height = first, second, raw[:, :, 2], raw[:, :, 3]
            if pixel_scale:
                cx = cx / size
                cy = cy / size
                width = width / size
                height = height / size
            if has_class:
                scores = raw[:, :, 4]
                labels = raw[:, :, 5]
            else:
                scores = raw[:, :, 4]
                labels = torch.zeros_like(scores)
            cx = cx.clamp(0.0, 1.0)
            cy = cy.clamp(0.0, 1.0)
            width = width.clamp(MIN_EXTENT, 1.0)
            height = height.clamp(MIN_EXTENT, 1.0)
            stacked = torch.stack([cx, cy, width, height, scores, labels], dim=2)
            if torch.jit.is_tracing():
                # 追踪时形状是常量，直接取 cap；候选框数不足的情况在 eager 下提前拦住
                count = cap
            else:
                available = int(stacked.shape[1])
                if available < cap:
                    raise SystemExit(
                        f"检测器只给出 {available} 个候选框，小于 --max-boxes {cap}；"
                        f"请把 --max-boxes 降到 {available} 以内")
                # eager 下也截到 cap，保证输出形状与导出的 ONNX 完全一致
                count = cap
            order = torch.topk(scores, count, dim=1, largest=True, sorted=True).indices
            return stacked.gather(1, order.unsqueeze(-1).expand(-1, -1, CONTRACT_COLUMNS))

        def forward(self, images):
            return self._to_contract(self.detector(self._prepare(images)))

    return ContractHead()


def probe_detector(detector, *, image_size, torch=None, channel_repeat=1,
                   input_scale=1.0, mean=None, std=None):
    """跑一遍**原始**检测器，返回输出描述（用于确定 ``--layout`` 与 ``--max-boxes``）。"""
    torch = torch or require_torch()
    prepared = _prepare_probe(torch, image_size, channel_repeat, input_scale, mean, std)
    detector.eval()
    with torch.no_grad():
        raw = detector(prepared)
    if isinstance(raw, (list, tuple)):
        shapes = [tuple(item.shape) for item in raw]
        return {"kind": "sequence", "shapes": shapes,
                "hint": "检测器返回多级特征图，请用它自带的解码器合成单一张量后再接入"}
    return {"kind": "tensor", "shape": tuple(raw.shape), "dtype": str(raw.dtype),
            "hint": "按最后一维的列数选择 --layout（见 --list-layouts）"}


def _prepare_probe(torch, image_size, channel_repeat, input_scale, mean, std):
    data = torch.zeros(1, 1, int(image_size), int(image_size), dtype=torch.float32)
    if int(channel_repeat) > 1:
        data = data.repeat(1, int(channel_repeat), 1, 1)
    if float(input_scale) != 1.0:
        data = data * float(input_scale)
    if mean is not None:
        mean_tensor = _as_tensor(torch, mean, "--input-mean")
        std_tensor = _as_tensor(torch, std, "--input-std")
        shape = (1, data.shape[1], 1, 1)
        data = (data - mean_tensor.reshape(shape)) / std_tensor.reshape(shape)
    return data


def validate_head(head, *, image_size, max_boxes, torch=None):
    """空跑一次：确认输出形状、有限性与置信度单调性，避免导出"形状对但语义错"的图。"""
    torch = torch or require_torch()
    head.eval()
    dummy = torch.zeros(1, 1, int(image_size), int(image_size), dtype=torch.float32)
    with torch.no_grad():
        output = head(dummy)
    shape = tuple(output.shape)
    expected = (1, int(max_boxes), CONTRACT_COLUMNS)
    if shape != expected:
        raise SystemExit(
            f"包装后的输出形状为 {shape}，期望 {expected}；"
            f"通常是 --max-boxes 超过了检测器的候选框数（YOLO26 端到端头固定 300）")
    array = output.detach().cpu().numpy()
    if not all(abs(float(value)) < 1e6 for value in array.reshape(-1)):
        raise SystemExit("包装后的输出包含异常数值，请检查 --layout 与 --input-scale 是否匹配")
    return {"shape": shape, "finite": True}


def export_torch_module(module, dummy, path, *, opset=17, input_names=("images",),
                        output_names=("detections",), torch=None):
    """跨 torch 版本稳定地导出 ONNX。

    torch ≥ 2.9 起 ``torch.onnx.export`` 默认改用 dynamo 导出器，而它依赖可选的
    ``onnxscript``——本仓库不希望为一个只影响导出的包拖住整条链路（README §7 的
    "核心链路不引入硬依赖"），因此优先用等价的 TorchScript 导出器；
    只有它被真正移除后才回退到 dynamo。
    """
    torch = torch or require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common = {"opset_version": int(opset), "input_names": list(input_names),
              "output_names": list(output_names), "do_constant_folding": True}
    try:
        torch.onnx.export(module, (dummy,), str(path), dynamo=False, **common)
        return path
    except (TypeError, ValueError) as exc:  # pragma: no cover - 取决于 torch 版本
        legacy = f"{type(exc).__name__}: {exc}"
    try:
        torch.onnx.export(module, (dummy,), str(path), **common)
    except Exception as exc:  # pragma: no cover - 取决于 torch 版本
        raise SystemExit(
            f"ONNX 导出失败：{legacy}；改用 dynamo 导出器也失败：{exc}\n"
            '  可尝试：.venv/bin/python -m pip install onnxscript') from exc
    return path


def export_contract_onnx(head, path, *, image_size, opset=17, torch=None,
                         input_name="images", output_name="detections"):
    """导出 ONNX：输入 ``images (1,1,H,W)``，输出 ``detections (1,max_boxes,6)``。"""
    torch = torch or require_torch()
    head.eval()
    dummy = torch.zeros(1, 1, int(image_size), int(image_size), dtype=torch.float32)
    return export_torch_module(head, dummy, path, opset=opset, input_names=[input_name],
                               output_names=[output_name], torch=torch)


def describe_onnx(path):
    """读取导出图的输入/输出形状（``onnx`` 缺失时返回空描述，不阻断导出）。"""
    try:
        import onnx
    except ImportError:  # pragma: no cover - onnx 属于 [train] 额外依赖
        return {}
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    def shapes(values):
        result = []
        for value in values:
            dims = [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                    for dim in value.type.tensor_type.shape.dim]
            result.append({"name": value.name, "shape": dims})
        return result

    return {"inputs": shapes(model.graph.input), "outputs": shapes(model.graph.output)}
