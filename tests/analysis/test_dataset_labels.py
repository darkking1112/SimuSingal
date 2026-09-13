"""逐跳标签语义的数据集回归（``build_dataset.py --labels hop`` ↔ ``detectors/dataset.py``）。

逐跳模型（``ml-detect-hops``）与数据集之间只有一条约定：**清单/卡片声明的
``label_semantics`` 必须是 ``per_hop_v1``**。声明错方向的代价是静默而非报错——
会话级标签训出的模型在逐跳通路里会把一段 8 跳的传输报成"1 跳"，数字看上去正常
但对不上真值。因此这里覆盖三件事：

* 生成端：``--labels hop`` 时标签真的按 ``hop_truth`` 逐跳展开，且**只替换
  ``fh*`` 信号**（连续信号的粒度本来就是整段传输，不能被整体切换）；
* 卡片：``label_semantics`` 与标签几何统计（像素尺度）都写进 ``dataset.json``；
* 读取端：取值校验、缺字段按会话级兼容、``contract_of`` 补全语义字段；
* 训练端（``train_yolox.py``）：网格步长量化守卫、按语义选评测真值、把语义写进清单。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

from signal_analysis.ml import band_to_box, read_model_manifest  # noqa: E402


def _load(name, path):
    """按路径加载 training/ 下的脚本（它们不在包内，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("build_dataset", TRAINING / "build_dataset.py")


@pytest.fixture(scope="module")
def reader():
    return _load("detectors_dataset", TRAINING / "detectors" / "dataset.py")


@pytest.fixture(scope="module")
def trainer():
    return _load("train_yolox", TRAINING / "train_yolox.py")


def _build(builder, root, *extra):
    code = builder.main(["--output", str(root), "--count", "4", "--seed", "7",
                         "--image-size", "128", "--nfft", "128",
                         "--duration-range", "0.1,0.1", "--max-signals", "3",
                         "--noise-only-ratio", "0.0", "--snr-range", "20,20",
                         "--hop-rate-range", "40,40", *extra])
    assert code == 0
    card = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    return card, records


@pytest.fixture(scope="module")
def hop_dataset(tmp_path_factory, builder):
    """纯跳频、逐跳标签的数据集。"""
    root = tmp_path_factory.mktemp("hopds")
    return _build(builder, root, "--modes", "fh_rc", "--labels", "hop")


@pytest.fixture(scope="module")
def session_dataset(tmp_path_factory, builder):
    """同一批参数、会话级标签的数据集（对照）。"""
    root = tmp_path_factory.mktemp("sessds")
    return _build(builder, root, "--modes", "fh_rc")


def test_hop_card_declares_per_hop_semantics(hop_dataset):
    card, records = hop_dataset
    contract = card["contract"]
    assert contract["label_semantics"] == "per_hop_v1"
    # 图像/标签契约本身不变：只有标签粒度不同
    assert contract["output_layout"] == "normalized_boxes_v1"
    assert contract["labels"] == ["emitter"]
    statistics = card["statistics"]["labels"]
    assert statistics["semantics"] == "per_hop_v1"
    assert statistics["total"] == sum(len(record["boxes"]) for record in records)
    assert statistics["total"] == card["statistics"]["targets"]
    # 一跳一个框：单场景标签数必然多于"一段传输一个框"
    assert statistics["boxes_per_sample"]["max"] >= 4
    assert statistics["boxes_per_sample"]["min"] >= 2


def test_session_card_stays_the_default(session_dataset):
    card, records = session_dataset
    assert card["contract"]["label_semantics"] == "session_v1"
    assert card["statistics"]["labels"]["semantics"] == "session_v1"
    # 会话级：每个信号一个框，与历史行为一致
    for record in records:
        assert len(record["boxes"]) == len(record["scene"]["signals"])


def test_hop_labels_follow_hop_truth(hop_dataset, reader):
    """逐跳标签的每个框都对应一条真值跳，且几何量与真值完全一致。"""
    card, records = hop_dataset
    semantics_total = 0
    for record in records:
        hops = record["hops"]
        assert hops, "跳频数据集必须带逐跳真值"
        assert len(record["boxes"]) == len(hops)
        meta = reader.scene_meta(record, card)
        for box, hop in zip(record["boxes"], hops):
            expected = band_to_box(meta, hop["f_low_hz"], hop["f_high_hz"],
                                   hop["t_start_s"], hop["t_end_s"])
            assert box[:4] == pytest.approx(list(expected), abs=1e-6)
            assert box[4] == 1.0 and box[5] == 0.0
        semantics_total += len(hops)
    assert semantics_total == card["statistics"]["labels"]["total"]


def test_hop_labels_only_replace_hopping_sessions(builder, tmp_path):
    """混合场景：``fh*`` 会话按跳展开，连续信号仍是一个整段框。

    这条性质决定了同一个数据集里两类标签能共存，而不是"打开逐跳开关就整体切换"。
    """
    card, records = _build(builder, tmp_path / "mixed",
                           "--modes", "fh_rc,qpsk", "--labels", "hop")
    assert card["contract"]["label_semantics"] == "per_hop_v1"
    seen_hopping = seen_continuous = False
    for record in records:
        groups = {}
        for hop in record["hops"]:
            groups.setdefault(hop["session_index"], []).append(hop)
        sessions = len(record["scene"]["signals"])
        expected = sessions - len(groups) + sum(len(items) for items in groups.values())
        assert len(record["boxes"]) == expected
        seen_hopping = seen_hopping or bool(groups)
        seen_continuous = seen_continuous or len(groups) < sessions
    assert seen_hopping and seen_continuous, "本用例需要场景里同时出现跳频与连续信号"


def test_hop_labels_need_a_hopping_mode(builder, tmp_path):
    with pytest.raises(SystemExit) as error:
        _build(builder, tmp_path / "bad", "--modes", "qpsk", "--labels", "hop")
    assert "跳频" in str(error.value)


def test_label_geometry_statistics_are_measurable(hop_dataset, session_dataset):
    """标签像素尺度写进卡片：训练脚本据此拒绝"网格比标签还粗"的配置。"""
    hop_card, _ = hop_dataset
    labels = hop_card["statistics"]["labels"]
    size = hop_card["contract"]["image_size"]
    assert 0 < labels["min_width_px"] <= labels["median_width_px"] <= size
    assert 0 < labels["min_height_px"] <= labels["median_height_px"] <= size
    session_labels = session_dataset[0]["statistics"]["labels"]
    # 一跳只占整段传输带宽的一小部分，逐跳标签必然更"小"
    assert labels["min_height_px"] < session_labels["min_height_px"]
    assert labels["min_width_px"] < session_labels["min_width_px"]


def test_reader_accepts_both_semantics(reader, hop_dataset, session_dataset):
    """卡片里的语义字段能原样读出，并经 ``contract_of`` 传给训练脚本。"""
    for card, expected in ((hop_dataset[0], "per_hop_v1"),
                           (session_dataset[0], "session_v1")):
        assert reader.label_semantics(card) == expected
        assert reader.contract_of(card)["label_semantics"] == expected


def test_reader_defaults_to_session_semantics_for_legacy_cards(reader):
    """历史数据集没有 ``label_semantics``：按会话级解释（向后兼容）。"""
    legacy = {"contract": {"input_contract": "tf_image_v1",
                           "layout": "time_frequency_grayscale_v1",
                           "output_layout": "normalized_boxes_v1",
                           "image_size": 64, "spectrogram_nfft": 64,
                           "dynamic_range_db": 60.0, "labels": ["emitter"]}}
    assert reader.label_semantics(legacy) == reader.DEFAULT_LABEL_SEMANTICS == "session_v1"
    assert reader.contract_of(legacy)["label_semantics"] == "session_v1"


def test_reader_rejects_unknown_semantics(reader, session_dataset, tmp_path):
    card = json.loads(json.dumps(session_dataset[0]))
    card["contract"]["label_semantics"] = "hop_v2"
    with pytest.raises(SystemExit) as error:
        reader.label_semantics(card)
    assert "label_semantics" in str(error.value) or "hop_v2" in str(error.value)
    # 读取时就校验：不能等训练完才发现语义对不上
    root = tmp_path / "ds"
    root.mkdir()
    (root / "dataset.json").write_text(json.dumps(card), encoding="utf-8")
    (root / "samples.jsonl").write_text(json.dumps({"index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        reader.load_dataset(root)


def _card(image_size, min_width_px, min_height_px, max_boxes_per_sample=24):
    """只保留守卫用到的字段的最小卡片（守卫是 card + args 的纯函数）。"""
    return {
        "contract": {"image_size": image_size},
        "statistics": {"labels": {"min_width_px": min_width_px,
                                  "min_height_px": min_height_px,
                                  "boxes_per_sample": {"max": max_boxes_per_sample}}},
    }


def test_grid_guard_blocks_a_grid_coarser_than_the_labels(trainer):
    """量化守卫：网格步长一超过标签尺寸就要报错，并指出该用哪个 --strides。"""
    import tiny_detector

    card = _card(1024, min_width_px=12.0, min_height_px=6.4)
    coarse = tiny_detector.grid_size(1024, 4)
    assert 1024 / coarse == 16.0
    with pytest.raises(SystemExit) as error:
        trainer._grid_guard(card, SimpleNamespace(image_size=1024, strides=4,
                                                  max_boxes=128), coarse)
    message = str(error.value)
    assert "6.4" in message and "16" in message and "--strides 2" in message
    # 步长降到 2（网格步长 4 像素 ≤ 6.4 像素）后放行，并报出供打印的几何量
    fine = tiny_detector.grid_size(1024, 2)
    guard = trainer._grid_guard(card, SimpleNamespace(image_size=1024, strides=2,
                                                      max_boxes=128), fine)
    assert guard["cell_px"] == pytest.approx(4.0)
    assert guard["min_label_px"] == pytest.approx(6.4)
    assert guard["max_boxes_per_sample"] == 24


def test_grid_guard_flags_a_sub_pixel_hop_dataset(trainer, hop_dataset):
    """本用例里的 128 像素 / 0.1 s 数据集本身就表达不了逐跳标签。

    一跳带宽只占采样率的千分之几，投影到 128 像素图上不到 1 像素。守卫因此
    连 ``--strides 1`` 也不建议，而是直接要求提高图像尺寸——这正是它存在的意义：
    这种数据集上“训练能不能跑完”和“结果对不对”完全无关，真实逐跳数据集要用
    1024 像素图（见 training/README.md）。
    """
    import tiny_detector

    card, _ = hop_dataset
    labels = card["statistics"]["labels"]
    assert min(labels["min_width_px"], labels["min_height_px"]) < 1.0
    with pytest.raises(SystemExit) as error:
        trainer._grid_guard(card, SimpleNamespace(image_size=128, strides=2,
                                                  max_boxes=128),
                            tiny_detector.grid_size(128, 2))
    assert "提高 --image-size" in str(error.value)


def test_grid_guard_blocks_too_few_output_boxes(trainer):
    """单样本标签数超过 --max-boxes 时多出来的框无法被输出表示。"""
    import tiny_detector

    card = _card(1024, min_width_px=40.0, min_height_px=32.0, max_boxes_per_sample=24)
    with pytest.raises(SystemExit) as error:
        trainer._grid_guard(card, SimpleNamespace(image_size=1024, strides=2,
                                                  max_boxes=16),
                            tiny_detector.grid_size(1024, 2))
    assert "--max-boxes" in str(error.value) and "24" in str(error.value)


def test_trainer_declares_the_dataset_semantics(trainer, hop_dataset, session_dataset,
                                                tmp_path):
    """导出清单里的标签语义与数据集一致：推理端据此选解码通路。"""
    import tiny_detector
    import torch

    for index, (card, records) in enumerate((hop_dataset, session_dataset)):
        root = tmp_path / f"run{index}"
        root.mkdir()
        model = tiny_detector.TinyDetector(max_boxes=8, classes=1, width=8, strides=2)
        args = SimpleNamespace(
            image_size=card["contract"]["image_size"], opset=17, arch="tiny",
            model_id="hop-labels", version="9.9.9", license="Apache-2.0",
            data=str(tmp_path), epochs=1, batch=2, lr=1e-3, seed=0)
        written = trainer._export(model, torch, args, card, None,
                                  root / "detector.onnx", root / "detector.json")
        expected = card["contract"]["label_semantics"]
        assert written["training"]["label_semantics"] == expected
        manifest, library = read_model_manifest(root / "detector.json")
        assert manifest["label_semantics"] == expected
        assert library.name == "detector.onnx"


def test_validation_scores_against_the_matching_granularity(trainer, hop_dataset):
    """逐跳数据集用 ``hop_truth`` 评分：真值个数是跳数而不是会话数。"""
    import tiny_detector
    import torch

    card, records = hop_dataset
    model = tiny_detector.TinyDetector(max_boxes=8, classes=1, width=8, strides=2)
    args = SimpleNamespace(model_id="hop-labels", version="9.9.9", val_samples=1)
    pooled, per_scene = trainer._validate(model, torch, card, records, args)
    assert len(per_scene) == 1
    val_record = [record for record in records if record["split"] == "val"][0]
    assert pooled["true"] == len(val_record["hops"]) > 1
    assert set(per_scene[0]) >= {"true", "detected", "matched", "missed", "false_alarm"}
