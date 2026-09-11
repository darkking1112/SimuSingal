"""ONNX 图改写：给**别人的导出图**补上契约输入与契约输出。

为什么需要这一步？因为 Ultralytics 之类框架的导出图与我们的契约差两件事：

1. **输入**：它们的图接受 ``[0,255]`` 三通道（ImageNet 标准化在 Python 侧做），
   而我们要求 ``[0,1]`` 单通道灰度时频图 —— README §7 明确要求把预处理
   写进网络第一层，不能在 Python 侧补；
2. **输出**：它们的图吐 ``(1, C, 8400)`` 这种"通道在前、未归一化、含全部候选框"
   的张量，而契约要求 ``(1, max_boxes, 6)`` 的 ``normalized_cxcywh_score_label``。

本模块用纯 ONNX 算子重写图，**不引入任何框架依赖**，因此：

* 与训练框架解耦（哪个框架都行，只要它导得出 ONNX）；
* 不改权重、不重训，只在头部/尾部插入几十个算子；
* 可离线审计——插入的算子都是标准 ONNX op，没有自定义算子。

插入的算子：``Tile``（通道复制）、``Mul`` / ``Sub`` / ``Div``（标准化）、
``Gather``（取列）、``Add`` / ``Sub`` / ``Mul`` / ``Div``（几何转换）、
``Clip``（限幅）、``Concat`` / ``Expand`` / ``GatherElements``（组装）、
``TopK``（按置信度截断）。全部要求 opset ≥ 13，若原图更低则自动提升。

几何转换的 ``y`` 方向**不做翻转**：第三方框架的图像行 0 在顶部，
而 ``detection_image`` 的行 0 也被约定为 ``+fs/2``（见 :mod:`.contract`），
两者方向一致，任何"贴心地翻一下"都会把频段上下颠倒。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import CONTRACT_COLUMNS, layout_spec

MIN_OPSET = 13
PROBE_PREFIX = "contract"
DEFAULT_MAX_BOXES = 32
MIN_EXTENT = 1e-4


def require_onnx():
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - onnx 属于 [train] 额外依赖
        raise SystemExit(
            'ONNX 图改写需要 onnx，请先安装：.venv/bin/python -m pip install -e ".[train]"'
        ) from exc
    return onnx


def _helper():
    from onnx import helper

    return helper


def _make_initializer(name, values, dtype):
    from onnx import numpy_helper

    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name=name)


def _used_names(graph):
    names = {value.name for value in graph.input}
    names.update(value.name for value in graph.output)
    names.update(value.name for value in graph.value_info)
    names.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        names.update(node.input)
        names.update(node.output)
    return names


def _namer(graph):
    used = _used_names(graph)

    def make(label):
        candidate = f"{PROBE_PREFIX}/{label}"
        index = 0
        while candidate in used:
            index += 1
            candidate = f"{PROBE_PREFIX}/{label}_{index}"
        used.add(candidate)
        return candidate

    return make


def infer_shapes(model):
    """形状推断；失败时返回原模型（有些图在缺少自定义算子时推不出来）。"""
    from onnx import shape_inference

    try:
        return shape_inference.infer_shapes(model)
    except Exception:  # pragma: no cover - 取决于第三方导出图的整洁程度
        return model


def _dims(value):
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in value.type.tensor_type.shape.dim]


def _output_shape(model):
    """原图第一个输出的静态/符号形状（用于自动判断是否需要转置）。"""
    for value in infer_shapes(model).graph.output:
        shape = _dims(value)
        if all(isinstance(dim, int) and dim > 0 for dim in shape):
            return shape
    return []


def _resolve_transpose(model, columns, transpose):
    """返回 ``(是否需要转置, 候选框数或 0)``。

    约定原生输出是 ``(N, B, cols)``；Ultralytics 的检测头是 ``(1, cols, B)``，
    此时最后一维不是 ``cols`` 而是候选框数，需要先转置。
    """
    if transpose is not None:
        return bool(transpose), 0
    shape = _output_shape(model)
    if len(shape) != 3:
        return False, 0
    if shape[-1] == columns:
        return False, shape[-2]
    if shape[1] == columns:
        return True, shape[-1]
    return False, 0


# --------------------------------------------------------------------------- 输入


def prepend_input_prep(model, *, channel_repeat=1, input_scale=1.0, mean=None, std=None):
    """把预处理写进图内第一层；返回 ``(模型, 是否真的插入了算子)``。

    插入后图输入变成单通道 ``[0,1]``，原来那个三通道输入名由新链路的末端提供，
    因此下游节点一个都不用改。
    """
    o = require_onnx()
    helper = _helper()
    graph = model.graph
    if not graph.input:
        raise SystemExit("ONNX 图没有输入，无法插入预处理")
    original = graph.input[0]
    channels = int(channel_repeat)
    if channels not in (1, 3):
        raise SystemExit("--channel-repeat 只支持 1 或 3")
    scale = float(input_scale)
    mean_values = None if mean is None else np.asarray(list(mean), dtype=np.float32)
    std_values = None if std is None else np.asarray(list(std), dtype=np.float32)
    if mean_values is not None and std_values is None:
        raise SystemExit("给了均值就必须同时给标准差")
    if mean_values is not None and mean_values.size not in (1, channels):
        raise SystemExit(f"均值的元素个数应为 1 或 {channels}")

    wants = channels > 1 or scale != 1.0 or mean_values is not None
    if not wants:
        return model, False

    make = _namer(graph)
    original_name = original.name
    new_input = make("images")
    graph.input[0].name = new_input

    nodes = []
    initializers = []
    current = new_input
    if channels > 1:
        repeats = make("repeats")
        initializers.append(_make_initializer(repeats, [1, channels, 1, 1], np.int64))
        tiled = make("tiled")
        nodes.append(helper.make_node("Tile", [current, repeats], [tiled], name=make("Tile")))
        current = tiled
    if scale != 1.0:
        factor = make("scale")
        initializers.append(_make_initializer(factor, scale, np.float32))
        scaled = make("scaled")
        nodes.append(helper.make_node("Mul", [current, factor], [scaled], name=make("Mul")))
        current = scaled
    if mean_values is not None:
        shape = (1, int(mean_values.size), 1, 1)
        mean_name = make("mean")
        std_name = make("std")
        initializers.append(_make_initializer(mean_name, mean_values.reshape(shape), np.float32))
        initializers.append(_make_initializer(std_name, std_values.reshape(shape), np.float32))
        centred = make("centred")
        nodes.append(helper.make_node("Sub", [current, mean_name], [centred], name=make("Sub")))
        # 链条末端直接写回原输入名，因此下游节点一个都不用改
        nodes.append(helper.make_node("Div", [centred, std_name], [original_name],
                                      name=make("Div")))
    else:
        # 最后一个算子直接产出原输入名
        tail = nodes[-1]
        nodes[-1] = helper.make_node(tail.op_type, list(tail.input), [original_name],
                                     name=tail.name)

    # 插到最前面，保持节点列表的拓扑序（onnxruntime 会重新排序，但可读性更好）
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.node.extend(existing)
    graph.initializer.extend(initializers)

    # 新输入的静态形状：契约输入固定为 (1, 1, H, W)
    height, width = _input_hw(original)
    dims = original.type.tensor_type.shape.dim
    dims[0].dim_value = 1
    dims[1].dim_value = 1
    if isinstance(height, int):
        dims[2].dim_value = int(height)
        dims[3].dim_value = int(width)
    original.type.tensor_type.elem_type = o.TensorProto.FLOAT
    return _bump_opset(model), True


def _input_hw(value):
    dims = _dims(value)
    if len(dims) == 4 and all(isinstance(dim, int) and dim > 0 for dim in dims[1:]):
        return dims[2], dims[3]
    return "height", "width"


def _bump_opset(model, minimum=MIN_OPSET):
    from onnx import helper

    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx") and entry.version < minimum:
            entry.version = minimum
    if not any(entry.domain in ("", "ai.onnx") for entry in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", minimum))
    return model


# --------------------------------------------------------------------------- 输出


def append_contract_head(model, *, layout="pixel_xyxy", image_size=0, max_boxes=DEFAULT_MAX_BOXES,
                         transpose=None, output_name="detections"):
    """在原图输出后面接上几何转换 + TopK，产出 ``(1, max_boxes, 6)``。"""
    o = require_onnx()
    helper = _helper()
    graph = model.graph
    spec = layout_spec(layout)
    columns = int(spec["columns"])
    cap = int(max_boxes)
    if cap < 1 or cap > 256:
        raise SystemExit("--max-boxes 应在 1～256 之间")
    size = float(image_size)
    if spec["scale"] == "pixel" and size <= 0:
        raise SystemExit(f"布局 {spec['name']} 是像素坐标，必须给 --imgsz")

    if not graph.output:
        raise SystemExit("ONNX 图没有输出，无法追加契约头")
    native_output = graph.output[0].name
    # 必须在摘掉旧输出之前判断列序，否则形状推断拿不到结果
    flip, anchors = _resolve_transpose(model, columns, transpose)
    if anchors and anchors < cap:
        raise SystemExit(
            f"原图候选框数为 {anchors}，小于 --max-boxes {cap}；请把 --max-boxes 降到 {anchors} 以内")
    del graph.output[0]

    make = _namer(graph)
    nodes = []
    initializers = []

    def const(label, values, dtype=np.float32):
        name = make(label)
        initializers.append(_make_initializer(name, values, dtype))
        return name

    current = native_output
    if flip:
        transposed = make("raw")
        nodes.append(helper.make_node("Transpose", [current], [transposed],
                                      name=make("Transpose"), perm=[0, 2, 1]))
        current = transposed

    def column(index, label):
        name = make(label)
        nodes.append(helper.make_node("Gather", [current, const(f"{label}_idx", [index], np.int64)],
                                      [name], name=make(f"Gather_{label}"), axis=2))
        return name

    first = column(0, "x1")
    second = column(1, "y1")
    third = column(2, "x2")
    fourth = column(3, "y2")
    scores = column(4, "score")
    labels = column(5, "label") if columns >= 6 else None

    half = const("half", 0.5)

    def binary(op, left, right, label):
        name = make(label)
        nodes.append(helper.make_node(op, [left, right], [name], name=make(label)))
        return name

    if spec["geometry"] == "xyxy":
        cx = binary("Mul", binary("Add", first, third, "x_sum"), half, "cx")
        cy = binary("Mul", binary("Add", second, fourth, "y_sum"), half, "cy")
        width = binary("Sub", third, first, "w")
        height = binary("Sub", fourth, second, "h")
    else:
        cx, cy, width, height = first, second, third, fourth

    if spec["scale"] == "pixel":
        divisor = const("size", size)
        cx = binary("Div", cx, divisor, "cx_norm")
        cy = binary("Div", cy, divisor, "cy_norm")
        width = binary("Div", width, divisor, "w_norm")
        height = binary("Div", height, divisor, "h_norm")

    one = const("one", 1.0)
    extent = const("min_extent", MIN_EXTENT)
    zero = const("zero", 0.0)

    def clip(value, minimum, label):
        name = make(label)
        nodes.append(helper.make_node("Clip", [value, minimum, one], [name],
                                      name=make(f"Clip_{label}")))
        return name

    cx = clip(cx, zero, "cx_clip")
    cy = clip(cy, zero, "cy_clip")
    width = clip(width, extent, "w_clip")
    height = clip(height, extent, "h_clip")

    if labels is None:
        # 单类检测器补齐第 6 列：乘 0 保持与 score 同形状（Concat 不做广播）
        labels = binary("Mul", scores, zero, "label_zero")

    boxes = make("boxes")
    nodes.append(helper.make_node("Concat", [cx, cy, width, height, scores, labels], [boxes],
                                  name=make("Concat"), axis=2))

    topk_values = make("topk_values")
    topk_indices = make("topk_indices")
    k = const("k", [cap], np.int64)
    nodes.append(helper.make_node("TopK", [scores, k], [topk_values, topk_indices],
                                  name=make("TopK"), axis=1, largest=1, sorted=1))
    expanded = make("indices")
    expand_shape = const("expand_shape", [1, cap, CONTRACT_COLUMNS], np.int64)
    nodes.append(helper.make_node("Expand", [topk_indices, expand_shape], [expanded],
                                  name=make("Expand")))
    nodes.append(helper.make_node("GatherElements", [boxes, expanded], [output_name],
                                  name=make("GatherElements"), axis=1))

    graph.node.extend(nodes)
    graph.initializer.extend(initializers)
    graph.output.append(helper.make_tensor_value_info(
        output_name, o.TensorProto.FLOAT, [1, cap, CONTRACT_COLUMNS]))
    return _bump_opset(model)


# --------------------------------------------------------------------------- 总入口


def rewrite_model(model, *, layout="pixel_xyxy", image_size=0, max_boxes=DEFAULT_MAX_BOXES,
                  channel_repeat=1, input_scale=1.0, mean=None, std=None, transpose=None):
    """先补输入预处理，再补输出契约头，最后做一次 checker 校验。"""
    o = require_onnx()
    model = prepend_input_prep(model, channel_repeat=channel_repeat, input_scale=input_scale,
                               mean=mean, std=std)[0]
    model = append_contract_head(model, layout=layout, image_size=image_size,
                                 max_boxes=max_boxes, transpose=transpose)
    model = _bump_opset(model)
    o.checker.check_model(model)
    return infer_shapes(model)


def rewrite(path, output=None, **options):
    """读入第三方导出图，写出契约合规图；返回 ``(输出路径, 图结构摘要)``。"""
    o = require_onnx()
    path = Path(path)
    if output is None:
        output = path.with_name(path.stem + "_contract" + path.suffix)
    output = Path(output)
    model = o.load(str(path))
    rewritten = rewrite_model(model, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    o.save(rewritten, str(output))
    return output, describe(output)


def describe(path):
    """返回 ``{"inputs": [...], "outputs": [...]}``。"""
    o = require_onnx()
    model = infer_shapes(o.load(str(path)))
    return {"inputs": [{"name": value.name, "shape": _dims(value)} for value in model.graph.input],
            "outputs": [{"name": value.name, "shape": _dims(value)} for value in model.graph.output],
            "opset": [{"domain": entry.domain or "ai.onnx", "version": entry.version}
                      for entry in model.opset_import],
            "nodes": len(model.graph.node)}
