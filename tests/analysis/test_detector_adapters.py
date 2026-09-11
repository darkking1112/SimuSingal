"""第三方检测器接入框架回归（``training/detectors/`` + ``training/export_contract.py``）。

覆盖三件事：

1. **适配器注册表与许可证门禁**：``--arch`` 取值、AGPL 传染性许可证必须显式放行、
   「只改写已导出的 ONNX」不需要框架的 Python 包；
2. **契约层**：布局表自洽、原生标注导出与 ``band_to_box`` 完全一致（不做 y 翻转）、
   PNG 量化误差在 1/255 以内；
3. **两条导出路径的数值正确性**：
   * torch 侧 ``ContractHead``（需要 torch）；
   * ONNX 侧 ``onnx_contract`` 图改写（需要 onnx/onnxruntime）。

这些用例**在本机没有安装任何第三方检测器**的前提下也必须通过：框架本身不进入
运行时依赖，接入只在 ``training/`` 里发生。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
for extra in (REPO_ROOT / "src", TRAINING):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from detectors import contract, labels, registry  # noqa: E402
from signal_analysis.ml import band_to_box  # noqa: E402

ADAPTER_NAMES = ("rtdetr", "tiny", "ultralytics", "yolox")


def _load(name):
    """按路径加载 ``training/`` 下的脚本（它们不在包里，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("build_dataset")


@pytest.fixture(scope="module")
def exporter():
    return _load("export_contract")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, builder):
    """极小数据集：64×64、4 个样本、无纯噪声场景（保证每个样本都有框）。"""
    root = tmp_path_factory.mktemp("detector_data")
    code = builder.main(["--output", str(root), "--count", "4", "--seed", "11",
                         "--image-size", "64", "--nfft", "64",
                         "--duration-range", "0.1,0.1", "--max-signals", "2",
                         "--noise-only-ratio", "0.0", "--snr-range", "20,20"])
    assert code == 0
    card = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return root, card, records


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch")


# --------------------------------------------------------------------- 注册表 / 许可证

def test_registry_names_are_stable():
    assert registry.names() == ADAPTER_NAMES


def test_describe_all_covers_every_adapter():
    described = registry.describe_all()
    assert [item["name"] for item in described] == list(ADAPTER_NAMES)
    for item in described:
        assert item["license"]
        assert item["upstream"]
        assert item["layout"] in contract.layout_names()
        assert isinstance(item["available"], bool)
    for item in described:
        if item["name"] != "tiny":  # tiny 是本仓库自带，不是上游项目
            assert item["upstream"].startswith("http")


def test_unknown_arch_lists_choices_and_points_to_readme():
    with pytest.raises(SystemExit) as error:
        registry.lookup("yolov9")
    message = str(error.value)
    for name in ADAPTER_NAMES:
        assert name in message
    assert "README" in message


def test_only_ultralytics_is_copyleft():
    distributable = {item["name"]: item["distributable"] for item in registry.describe_all()}
    assert distributable == {"rtdetr": True, "tiny": True, "ultralytics": False, "yolox": True}


def test_copyleft_gate_requires_explicit_flag():
    adapter = registry.lookup("ultralytics")
    with pytest.raises(SystemExit) as error:
        registry.require_available(adapter, runtime=False)
    message = str(error.value)
    assert "AGPL" in message and "--allow-copyleft" in message and "README" in message


def test_missing_runtime_dependency_reports_install_hint():
    adapter = registry.lookup("rtdetr")
    with pytest.raises(SystemExit) as error:
        registry.require_available(adapter)
    message = str(error.value)
    assert "未检测到" in message
    assert adapter.install_hint() in message
    assert "README" in message


def test_onnx_rewrite_needs_no_framework_package(monkeypatch):
    """``--onnx`` 模式不导入框架：允许通过，许可证门禁仍然生效。"""
    adapter = registry.lookup("yolox")
    monkeypatch.setattr(adapter, "is_available", lambda: False)
    assert registry.require_available(adapter, runtime=False) is adapter
    with pytest.raises(SystemExit):
        registry.require_available(adapter, runtime=True)


def test_declared_preprocess_matches_upstream_source():
    """把三家上游源码里核对过的输入口径钉住，防止再凭记忆改错。

    核对依据（均已查源码）：

    * Ultralytics ``engine/predictor.py::BasePredictor.preprocess``：
      BGR→RGB 翻转 + ``.div_(255)``，**没有** ImageNet 均值方差；
    * YOLOX ``data/data_augment.py::preproc``：pad 114 + resize + HWC→CHW，
      float32 且仍是 0-255 的 BGR（官方 demo 用 ``ValTransform(legacy=False)``）；
    * RT-DETR 各分支口径不一致，因此不给默认值、强制显式声明。
    """
    expected = {
        "ultralytics": {"channel_repeat": 3, "input_scale": 255.0, "mean": None, "std": None},
        "yolox": {"channel_repeat": 3, "input_scale": 1.0, "mean": None, "std": None},
        "tiny": {"channel_repeat": 1, "input_scale": 1.0, "mean": None, "std": None},
        "rtdetr": {"channel_repeat": 3, "mean": None, "std": None},
    }
    for name, want in expected.items():
        described = registry.lookup(name).describe()
        for key, value in want.items():
            assert described[key] == value, f"{name}.{key} 与上游源码口径不一致"
    assert registry.lookup("rtdetr").describe()["prep_required"] is True
    assert not registry.lookup("ultralytics").describe()["prep_required"]


def test_rtdetr_refuses_to_guess_input_preprocess():
    """RT-DETR 三个分支口径不同：不给 ``--input-scale`` 必须直接报错。"""
    adapter = registry.lookup("rtdetr")
    with pytest.raises(SystemExit) as error:
        adapter._resolve_prep()
    message = str(error.value)
    assert "--input-scale" in message
    for branch in ("rtdetr_paddle", "rtdetr_pytorch", "rtdetrv2_pytorch"):
        assert branch in message           # 报错里必须列出"选哪个"
    assert "README" in message
    # 显式声明后能拿到合并结果，且入参覆盖适配器默认值
    resolved = adapter._resolve_prep(channel_repeat=3, input_scale=255.0,
                                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert resolved["input_scale"] == 255.0
    assert resolved["mean"] == (0.485, 0.456, 0.406)
    # 不要求显式声明的适配器照旧套默认值
    assert registry.lookup("yolox")._resolve_prep()["channel_repeat"] == 3


def test_tiny_adapter_is_available_with_torch():
    adapter = registry.lookup("tiny")
    spec = importlib.util.find_spec("torch")
    assert adapter.is_available() is (spec is not None)


def test_summary_table_mentions_every_framework():
    from detectors.base import summary_table

    table = summary_table(registry.describe_all())
    for name in ADAPTER_NAMES:
        assert registry.lookup(name).info.title in table
    assert "AGPL-3.0" in table and "Apache-2.0" in table


# --------------------------------------------------------------------- 契约布局表

def test_layout_table_is_self_consistent():
    specs = {item["name"]: item for item in contract.describe_layouts()}
    assert set(specs) == set(contract.layout_names())
    for name, spec in specs.items():
        assert spec["columns"] == (6 if spec["has_class"] else 5), name
        assert spec["geometry"] in ("cxcywh", "xyxy")
        assert spec["scale"] in ("normalized", "pixel")
        assert spec["examples"]
    assert specs["pixel_xyxy_objectness"]["columns"] == 5
    assert specs["pixel_xyxy_objectness"]["has_class"] is False


def test_layout_spec_returns_a_copy():
    spec = contract.layout_spec("pixel_xyxy")
    spec["columns"] = 999
    assert contract.layout_spec("pixel_xyxy")["columns"] == 6


def test_layout_spec_rejects_unknown_name():
    with pytest.raises(ValueError):
        contract.layout_spec("yolo_v8_head")


# --------------------------------------------------------------------- 标注导出

def test_dataset_loader_accepts_builder_output(dataset):
    from detectors.dataset import contract_of, load_dataset, split_records

    root, card, records = dataset
    loaded_card, loaded_records = load_dataset(root)
    assert loaded_card["sample_count"] == len(loaded_records) == len(records)
    assert contract_of(loaded_card)["output_layout"] == "normalized_boxes_v1"
    train, val = split_records(loaded_records, "train"), split_records(loaded_records, "val")
    assert len(train) + len(val) == len(loaded_records)
    assert all(record["split"] == "train" for record in train)


@pytest.mark.parametrize("fmt,relative", [
    ("yolo", "labels/train"),
    ("coco", "annotations/instances_train.json"),
])
def test_export_dataset_writes_labels_from_band_to_box(dataset, tmp_path, fmt, relative):
    """标签只能来自 ``record['boxes']``，且不做任何 y 翻转。"""
    root, card, records = dataset
    target = tmp_path / fmt
    summary = labels.export_dataset(root, records, card, target, fmt)
    assert summary["images"] == len(records)
    assert (target / relative).exists()

    record = next(item for item in records if item["boxes"])
    image_size = int(card["contract"]["image_size"])
    stem = Path(record["image"]).stem
    if fmt == "yolo":
        rows = [[float(value) for value in line.split()] for line in
                (target / "labels" / "train" / f"{stem}.txt").read_text().splitlines()
                if line.strip()]
        assert len(rows) == len(record["boxes"])
        for row, box in zip(rows, record["boxes"]):
            assert row[0] == 0.0
            assert row[1:5] == pytest.approx(box[:4], abs=1e-9)
    else:
        annotations = json.loads((target / "annotations" / "instances_train.json")
                                .read_text(encoding="utf-8"))
        image = next(item for item in annotations["images"]
                     if item["file_name"].endswith(f"{stem}.png"))
        assert (image["width"], image["height"]) == (image_size, image_size)
        truth = np.array([box[:4] for box in record["boxes"]])
        for item in annotations["annotations"]:
            if item["image_id"] != image["id"]:
                continue
            x, y, width, height = item["bbox"]
            back = np.array([(x + width / 2) / image_size, (y + height / 2) / image_size,
                             width / image_size, height / image_size])
            assert np.abs(truth - back).sum(axis=1).min() < 1e-5


def test_export_dataset_rejects_unknown_format(dataset, tmp_path):
    root, card, records = dataset
    with pytest.raises(SystemExit):
        labels.export_dataset(root, records, card, tmp_path / "x", "voc")


def test_png_round_trip_error_is_below_one_level(dataset, tmp_path):
    """8 位 PNG 量化误差必须 ≤ 1/255，否则时频图口径会漂。"""
    root, card, records = dataset
    record = records[0]
    image = np.load(root / record["image"]).astype(np.float64)
    assert image.min() >= 0.0 and image.max() <= 1.0

    path = tmp_path / "one.png"
    labels.write_gray_png(path, image)

    import struct
    import zlib

    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert raw[25] == 0  # 灰度
    assert raw[24] == 8  # 位深

    # 解回像素做逐点比对（纯 stdlib，避免依赖 Pillow）
    width = struct.unpack(">I", raw[16:20])[0]
    height = struct.unpack(">I", raw[20:24])[0]
    assert (height, width) == image.shape
    idat = b""
    offset = 8
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"IDAT":
            idat += raw[offset + 8:offset + 8 + length]
        offset += 12 + length
    scan = zlib.decompress(idat)
    rows = np.frombuffer(scan, dtype=np.uint8).reshape(height, width + 1)[:, 1:]
    assert np.abs(rows.astype(np.float64) / 255.0 - image).max() <= labels.PNG_QUANTIZATION + 1e-12


def test_quantization_note_documented():
    assert "0.118" in labels.QUANTIZATION_NOTE or "量化" in labels.QUANTIZATION_NOTE


# --------------------------------------------------------------------- torch 契约头

def _fake_detector(torch, rows, dtype=None):
    tensor = torch.tensor(rows, dtype=dtype or torch.float32).reshape(1, -1, 6)

    class Fake(torch.nn.Module):
        def forward(self, images):  # noqa: ARG002
            return tensor

    return Fake()


def test_contract_head_converts_pixel_xyxy_to_normalized_cxcywh(torch_module):
    from detectors.torch_export import build_contract_head

    torch = torch_module
    rows = [[0, 0, 128, 128, 0.95, 0],
            [10, 10, 30, 30, 0.90, 0],
            [20, 20, 50, 50, 0.70, 0],
            [30, 30, 60, 60, 0.50, 0]]
    head = build_contract_head(_fake_detector(torch, rows), layout="pixel_xyxy",
                               image_size=128, max_boxes=2, torch=torch)
    out = head(torch.zeros(1, 1, 128, 128)).detach().numpy()[0]
    assert out.shape == (2, 6)
    assert out[:, 0] == pytest.approx([0.5, 0.15625], abs=1e-6)
    assert out[:, 1] == pytest.approx([0.5, 0.15625], abs=1e-6)
    assert out[:, 2] == pytest.approx([1.0, 0.15625], abs=1e-6)
    assert out[:, 3] == pytest.approx([1.0, 0.15625], abs=1e-6)
    assert out[:, 4] == pytest.approx([0.95, 0.90], abs=1e-6)
    assert out[:, 5] == pytest.approx([0.0, 0.0], abs=1e-6)


def test_contract_head_keeps_normalized_cxcywh_untouched(torch_module):
    from detectors.torch_export import build_contract_head

    torch = torch_module
    rows = [[0.25, 0.5, 0.2, 0.4, 0.8, 0.0]]
    head = build_contract_head(_fake_detector(torch, rows), layout="normalized_cxcywh",
                               image_size=64, max_boxes=1, torch=torch)
    out = head(torch.zeros(1, 1, 64, 64)).detach().numpy()[0]
    assert out[0, :4] == pytest.approx(rows[0][:4], abs=1e-6)
    assert out[0, 4] == pytest.approx(0.8, abs=1e-6)


def test_contract_head_sorts_by_score_and_caps_boxes(torch_module):
    from detectors.torch_export import build_contract_head

    torch = torch_module
    rows = [[0, 0, 10, 10, 0.1, 0], [0, 0, 20, 20, 0.9, 0], [0, 0, 30, 30, 0.5, 0]]
    head = build_contract_head(_fake_detector(torch, rows), layout="pixel_xyxy",
                               image_size=32, max_boxes=3, torch=torch)
    scores = head(torch.zeros(1, 1, 32, 32)).detach().numpy()[0][:, 4]
    assert scores == pytest.approx([0.9, 0.5, 0.1], abs=1e-6)


def test_contract_head_rejects_max_boxes_above_candidates(torch_module):
    from detectors.torch_export import build_contract_head

    torch = torch_module
    head = build_contract_head(_fake_detector(torch, [[0, 0, 8, 8, 0.5, 0]]),
                               layout="pixel_xyxy", image_size=64, max_boxes=8,
                               torch=torch)
    with pytest.raises(SystemExit) as error:
        head(torch.zeros(1, 1, 64, 64))
    assert "--max-boxes" in str(error.value)


def test_contract_head_clamps_zero_extent(torch_module):
    from detectors.torch_export import MIN_EXTENT, build_contract_head

    torch = torch_module
    head = build_contract_head(_fake_detector(torch, [[4, 4, 4, 4, 0.5, 0]]),
                               layout="pixel_xyxy", image_size=8, max_boxes=1,
                               torch=torch)
    out = head(torch.zeros(1, 1, 8, 8)).detach().numpy()[0]
    assert out[0, 2] >= MIN_EXTENT and out[0, 3] >= MIN_EXTENT


def test_contract_head_bakes_input_preprocess(torch_module):
    """``channel_repeat`` / ``input_scale`` / ``mean`` / ``std`` 必须写进包装层。"""
    from detectors.torch_export import build_contract_head

    torch = torch_module
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, images):
            seen["shape"] = tuple(images.shape)
            seen["mean"] = float(images.mean())
            return torch.zeros(1, 2, 6, dtype=images.dtype)

    prep = {"channel_repeat": 3, "input_scale": 255.0,
            "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
    head = build_contract_head(Spy(), layout="pixel_xyxy", image_size=8, max_boxes=2,
                               torch=torch, **prep)
    head(torch.ones(1, 1, 8, 8))

    assert seen["shape"] == (1, 3, 8, 8)
    expected = float(np.mean([(255.0 - m) / s for m, s in zip(prep["mean"], prep["std"])]))
    assert seen["mean"] == pytest.approx(expected, rel=1e-5)


def test_probe_detector_reports_shape_and_channel(torch_module):
    from detectors.torch_export import probe_detector

    torch = torch_module
    probe = probe_detector(_fake_detector(torch, [[0, 0, 8, 8, 0.5, 0]]), image_size=32,
                           torch=torch, channel_repeat=3)
    assert probe["kind"] == "tensor"
    assert probe["shape"] == (1, 1, 6)
    assert "layout" in probe["hint"]


def test_validate_head_reports_shape_and_finiteness(torch_module):
    from detectors.torch_export import build_contract_head, validate_head

    torch = torch_module
    rows = [[0, 0, 4, 4, 0.9, 0], [0, 0, 8, 8, 0.7, 0],
            [0, 0, 12, 12, 0.5, 0], [0, 0, 16, 16, 0.3, 0]]
    head = build_contract_head(_fake_detector(torch, rows), layout="pixel_xyxy",
                               image_size=32, max_boxes=4, torch=torch)
    result = validate_head(head, image_size=32, max_boxes=4, torch=torch)
    assert result["shape"] == (1, 4, 6)
    assert result["finite"] is True


def test_validate_head_rejects_too_few_candidates(torch_module):
    from detectors.torch_export import build_contract_head, validate_head

    torch = torch_module
    head = build_contract_head(_fake_detector(torch, [[0, 0, 4, 4, 0.5, 0]]),
                               layout="pixel_xyxy", image_size=32, max_boxes=4,
                               torch=torch)
    with pytest.raises(SystemExit) as error:
        validate_head(head, image_size=32, max_boxes=4, torch=torch)
    assert "--max-boxes" in str(error.value)


# --------------------------------------------------------------------- ONNX 图改写

def _synthetic_native_model(tmp_path, rows):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    array = np.asarray(rows, dtype=np.float32).reshape(1, len(rows), 6)
    inputs = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])
    outputs = helper.make_tensor_value_info("output0", TensorProto.FLOAT,
                                            [1, array.shape[1], 6])
    nodes = [
        helper.make_node("Constant", [], ["rows"],
                         value=numpy_helper.from_array(array, "const_rows")),
        helper.make_node("ReduceMean", ["images"], ["offset"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Add", ["rows", "offset"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "native", [inputs], [outputs])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    path = tmp_path / "native.onnx"
    onnx.save(model, str(path))
    return path, array[0]


def _run_onnx(path, image):
    session = pytest.importorskip("onnxruntime").InferenceSession(
        str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return session.run(None, {name: image})[0]


def test_onnx_rewrite_matches_hand_computed_contract(tmp_path):
    rows = [[0, 0, 64, 64, 0.95, 0],
            [8, 16, 24, 48, 0.20, 0],
            [16, 16, 48, 48, 0.75, 0],
            [1, 1, 3, 3, 0.40, 0]]
    native_path, _ = _synthetic_native_model(tmp_path, rows)

    from detectors.onnx_contract import rewrite

    output_path, graph = rewrite(native_path, tmp_path / "contract.onnx",
                                 layout="pixel_xyxy", image_size=64, max_boxes=3,
                                 channel_repeat=3, input_scale=255.0,
                                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    assert graph["outputs"][0]["shape"] == [1, 3, 6]
    assert graph["inputs"][0]["shape"] == [1, 1, 64, 64]
    versions = [entry["version"] for entry in graph["opset"]]
    assert max(versions) >= 13, versions

    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
    image = np.full((1, 1, 64, 64), 0.25, np.float32)
    inner = (np.repeat(image, 3, axis=1) * 255.0 - mean) / std
    native = _run_onnx(native_path, inner.astype(np.float32))[0]
    rewritten = _run_onnx(output_path, image)[0]

    x1, y1, x2, y2, score, _label = native.T
    expected = np.stack([np.clip((x1 + x2) / 2 / 64, 0, 1), np.clip((y1 + y2) / 2 / 64, 0, 1),
                         np.clip((x2 - x1) / 64, 1e-4, 1), np.clip((y2 - y1) / 64, 1e-4, 1),
                         score, np.zeros_like(score)], axis=1)
    expected = expected[np.argsort(-score)[:3]]

    assert rewritten.shape == (3, 6)
    # 第 5 列（类别）在本合成图里被 ReduceMean 污染，故只对几何 + 分数列做对拍
    assert np.abs(expected[:, :5] - rewritten[:, :5]).max() <= 1e-4
    assert rewritten[0, 4] == pytest.approx(float(np.max(score)), abs=1e-4)
    assert rewritten[:, 4] == pytest.approx(np.sort(score)[::-1][:3], abs=1e-4)


def test_onnx_rewrite_detects_transposed_head(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rows = np.array([[16.0, 16.0, 6.4, 6.4, 0.9, 0.0],
                     [3.2, 3.2, 3.2, 3.2, 0.4, 0.0]],
                    np.float32).T[None]  # (1, 6, 2) —— 列优先（非 (N, 6)）
    inputs = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 1, 32, 32])
    outputs = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 6, 2])
    nodes = [helper.make_node("Constant", [], ["output0"],
                              value=numpy_helper.from_array(rows, "rows"))]
    model = helper.make_model(helper.make_graph(nodes, "t", [inputs], [outputs]),
                              opset_imports=[helper.make_opsetid("", 13)])
    path = tmp_path / "checker.onnx"
    onnx.save(model, str(path))

    from detectors.onnx_contract import rewrite

    output_path, graph = rewrite(path, tmp_path / "fixed.onnx", layout="pixel_cxcywh",
                                 image_size=32, max_boxes=2)
    assert graph["outputs"][0]["shape"] == [1, 2, 6]
    result = _run_onnx(output_path, np.zeros((1, 1, 32, 32), np.float32))[0]
    assert result[0, :5] == pytest.approx([0.5, 0.5, 0.2, 0.2, 0.9], abs=1e-6)


def test_onnx_rewrite_rejects_unknown_layout(tmp_path):
    rows = [[0, 0, 8, 8, 0.5, 0]]
    native_path, _ = _synthetic_native_model(tmp_path, rows)

    from detectors.onnx_contract import rewrite

    with pytest.raises(ValueError):
        rewrite(native_path, tmp_path / "bad.onnx", layout="detr_v2", image_size=64,
                max_boxes=1)


# --------------------------------------------------------------------- CLI

def test_cli_lists_layouts_and_frameworks(exporter, capsys):
    assert exporter.main(["--list-layouts"]) == 0
    layouts = capsys.readouterr().out
    for name in contract.layout_names():
        assert name in layouts

    assert exporter.main(["--list-frameworks"]) == 0
    frameworks = capsys.readouterr().out
    for name in ADAPTER_NAMES:
        assert name in frameworks
    assert "AGPL-3.0" in frameworks


def test_cli_requires_framework(exporter):
    with pytest.raises(SystemExit) as error:
        exporter.main(["--data", "x", "--output", "y"])
    assert "--framework" in str(error.value)


def test_cli_dataset_only_needs_no_framework_package(exporter, dataset, tmp_path):
    root, _card, _records = dataset
    target = tmp_path / "native"
    code = exporter.main(["--framework", "yolox", "--data", str(root), "--dataset-only",
                          "--dataset-output", str(target), "--dataset-format", "yolo"])
    assert code == 0
    assert (target / "data.yaml").exists()
    assert (target / "labels" / "train").is_dir()


def test_cli_onnx_mode_gates_copyleft(exporter, tmp_path):
    rows = [[0, 0, 8, 8, 0.5, 0]]
    native_path, _ = _synthetic_native_model(tmp_path, rows)
    with pytest.raises(SystemExit) as error:
        exporter.main(["--framework", "ultralytics", "--onnx", str(native_path),
                       "--layout", "pixel_xyxy", "--imgsz", "64", "--max-boxes", "1",
                       "--output", str(tmp_path / "run")])
    assert "AGPL" in str(error.value)


def test_trainer_arch_choices_come_from_registry():
    trainer = _load("train_yolox")
    assert tuple(trainer._arch_choices()) == ADAPTER_NAMES
    assert "tiny" in trainer._arch_choices()


def test_trainer_rejects_unknown_arch():
    """``--arch`` 的取值由注册表给出，未知名字在 argparse 阶段就被拦住。"""
    trainer = _load("train_yolox")
    with pytest.raises(SystemExit):
        trainer.main(["--data", "nowhere", "--output", "nowhere", "--arch", "detr"])


def test_trainer_foreign_arch_explains_two_step_flow():
    trainer = _load("train_yolox")
    with pytest.raises(SystemExit) as error:
        trainer.main(["--data", "nowhere", "--output", "nowhere", "--arch", "rtdetr"])
    message = str(error.value)
    assert "README" in message
    assert "export_contract.py" in message and "--dataset-only" in message


def test_export_contract_module_help_runs():
    """入口脚本必须能独立启动（``--help`` 退出码 0）。"""
    result = subprocess.run([sys.executable, str(TRAINING / "export_contract.py"), "--help"],
                            capture_output=True, text=True, check=False,
                            cwd=str(REPO_ROOT))
    assert result.returncode == 0, result.stderr
    assert "--framework" in result.stdout and "--onnx" in result.stdout
