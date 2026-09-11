"""训练工具链回归（training/ 目录）：数据集构建、契约校验与验收脚本的自检。

``training/`` 不打包进 wheel（``[tool.setuptools.packages.find] where = ["src"]``），
但它是"训练-推理口径一致"的关键环节，所以这里用轻量方式覆盖：

* ``build_dataset.py``：契约字段、标签与真值一致、**样本可逐字节复现**
  （训练脚本与验收脚本都依赖这一点重建波形）；
* ``train_yolox.py``：数据集契约校验、指标池化、缺少 torch / YOLOX 接入位的提示；
* ``verify_onnx.py``：ONNX 输入输出形状检查（用假会话，不需要 onnxruntime）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.evaluation import signal_truth  # noqa: E402
from signal_analysis.ml import (  # noqa: E402
    BOX_COLUMNS,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    OUTPUT_LAYOUT,
    band_to_box,
    detection_image,
    spectral_context,
)


def _load(name):
    """按路径加载 training/ 下的脚本（它们不在包内，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("build_dataset")


@pytest.fixture(scope="module")
def trainer():
    return _load("train_yolox")


@pytest.fixture(scope="module")
def verifier():
    return _load("verify_onnx")


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory, builder):
    """极小数据集（64×64、3 个样本、无纯噪声场景）。"""
    root = tmp_path_factory.mktemp("dataset")
    code = builder.main(["--output", str(root), "--count", "3", "--seed", "11",
                         "--image-size", "64", "--nfft", "64", "--duration-range", "0.1,0.1",
                         "--max-signals", "2", "--noise-only-ratio", "0.0",
                         "--snr-range", "20,20"])
    assert code == 0
    card = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return root, card, records


def test_dataset_card_matches_inference_contract(tiny_dataset):
    _, card, records = tiny_dataset
    contract = card["contract"]
    assert contract["input_contract"] == INPUT_CONTRACT
    assert contract["layout"] == IMAGE_LAYOUT
    assert contract["output_layout"] == OUTPUT_LAYOUT
    assert contract["image_size"] == 64
    assert contract["spectrogram_nfft"] == 64
    assert contract["labels"] == ["emitter"]
    assert len(contract["box_columns"]) == BOX_COLUMNS
    assert contract["box_columns"][-2:] == ["confidence", "class"]
    assert card["sample_count"] == len(records) == 3
    assert card["splits"]["train"] == 2 and card["splits"]["val"] == 1
    assert card["statistics"]["targets"] == sum(len(record["boxes"]) for record in records)


def test_samples_are_exactly_reproducible(tiny_dataset):
    """每个样本都能用记录里的 scene 逐字节重建图像与标签。

    训练脚本的端到端验证、验收脚本的场景复现都依赖这条性质。
    """
    root, card, records = tiny_dataset
    contract = card["contract"]
    for record in records:
        scene = record["scene"]
        samples, generation = generate_iq(scene["rate_hz"], scene["duration_s"],
                                          scene["signals"], noise=scene["noise"],
                                          seed=scene["seed"])
        summary, arrays = spectral_context(samples, scene["rate_hz"],
                                           {"nfft": contract["spectrogram_nfft"]})
        image, meta = detection_image(arrays, summary, contract["image_size"],
                                      contract["dynamic_range_db"])
        stored = np.load(root / record["image"])
        assert np.array_equal(stored, image)
        expected = [band_to_box(meta, entry["f_low_hz"], entry["f_high_hz"],
                                entry["t_start_s"], entry["t_end_s"])
                    for entry in signal_truth(generation)]
        assert len(expected) == len(record["boxes"])
        for box, entry in zip(record["boxes"], expected):
            assert box[:4] == pytest.approx(list(entry), abs=1e-6)
            assert box[4] == 1.0 and box[5] == 0.0


def test_labels_stay_inside_image_and_do_not_overlap(tiny_dataset):
    root, card, records = tiny_dataset
    for record in records:
        boxes = np.asarray(record["boxes"], dtype=np.float64).reshape(-1, 6)
        if not len(boxes):
            continue
        assert np.all(boxes[:, 0] - boxes[:, 2] / 2 >= -1e-9)
        assert np.all(boxes[:, 0] + boxes[:, 2] / 2 <= 1 + 1e-9)
        assert np.all(boxes[:, 1] - boxes[:, 3] / 2 >= -1e-9)
        assert np.all(boxes[:, 1] + boxes[:, 3] / 2 <= 1 + 1e-9)
        assert np.all(boxes[:, 2] > 0) and np.all(boxes[:, 3] > 0)
        image = np.load(root / record["image"])
        assert image.shape == (card["contract"]["image_size"],) * 2
        assert image.dtype == np.float32
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
        # 标签框在频率轴上互不重叠（数据集用互斥频段槽生成场景）
        spans = sorted((box[1] - box[3] / 2, box[1] + box[3] / 2) for box in boxes)
        for (_, previous_high), (next_low, _) in zip(spans, spans[1:]):
            assert next_low >= previous_high - 1e-6


def test_dataset_builder_rejects_unknown_mode(builder, tmp_path):
    with pytest.raises(SystemExit) as error:
        builder.main(["--output", str(tmp_path / "bad"), "--count", "1", "--modes", "qpsk,ofdm"])
    assert "ofdm" in str(error.value)


def test_dataset_builder_rejects_bad_range(builder, tmp_path):
    with pytest.raises(SystemExit):
        builder.main(["--output", str(tmp_path / "bad"), "--count", "1", "--train-fraction", "1.5"])


def test_trainer_loads_dataset_and_checks_contract(trainer, tiny_dataset):
    root, card, records = tiny_dataset
    loaded_card, loaded_records = trainer._load_dataset(root)
    assert loaded_card["sample_count"] == card["sample_count"]
    assert len(loaded_records) == len(records)


def test_trainer_rejects_incompatible_contract(trainer, tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "dataset.json").write_text(json.dumps({
        "sample_count": 1,
        "contract": {"input_contract": "tf_image_v1", "layout": "time_frequency_grayscale_v1",
                     "output_layout": "raw_boxes_v9"},
    }), encoding="utf-8")
    (root / "samples.jsonl").write_text(json.dumps({"index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        trainer._load_dataset(root)
    assert "normalized_boxes_v1" in str(error.value)


def test_trainer_requires_torch_with_install_hint(trainer, tiny_dataset, monkeypatch):
    """缺少 torch 时必须给出可执行的安装提示（而不是裸 ImportError）。"""
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("本环境已安装 torch")
    root, _, _ = tiny_dataset
    with pytest.raises(SystemExit) as error:
        trainer.main(["--data", str(root), "--output", str(root / "run"), "--epochs", "1"])
    assert "train" in str(error.value)


def test_trainer_yolox_arch_points_to_readme(trainer, tmp_path):
    with pytest.raises(SystemExit) as error:
        trainer.main(["--data", str(tmp_path), "--output", str(tmp_path / "run"), "--arch", "yolox"])
    assert "README" in str(error.value)


def test_trainer_pooled_metrics(trainer):
    scenes = [
        {"true": 2, "detected": 2, "matched": 2, "missed": 0, "false_alarm": 0,
         "center_mae_hz": 100.0, "bandwidth_mape": 0.1},
        {"true": 2, "detected": 3, "matched": 1, "missed": 1, "false_alarm": 2,
         "center_mae_hz": 300.0, "bandwidth_mape": 0.3},
    ]
    pooled = trainer._pooled(scenes)
    assert pooled["true"] == 4 and pooled["detected"] == 5 and pooled["matched"] == 3
    assert pooled["false_alarm"] == 2 and pooled["missed"] == 1
    assert pooled["precision"] == pytest.approx(0.6)
    assert pooled["recall"] == pytest.approx(0.75)
    assert pooled["center_mae_hz"] == pytest.approx(200.0)
    assert pooled["bandwidth_mape"] == pytest.approx(0.2)


def test_trainer_pooled_handles_empty_scene(trainer):
    pooled = trainer._pooled([{"true": 0, "detected": 0, "matched": 0,
                               "missed": 0, "false_alarm": 0}])
    assert pooled["precision"] is None and pooled["recall"] is None and pooled["f1"] is None


def _fake_session(inputs, outputs):
    return SimpleNamespace(get_inputs=lambda: list(inputs), get_outputs=lambda: list(outputs))


def test_verifier_accepts_contract_conforming_model(verifier):
    session = _fake_session([SimpleNamespace(name="images", shape=[1, 1, 512, 512], type="tensor(float)")],
                            [SimpleNamespace(name="detections", shape=[1, 32, BOX_COLUMNS])])
    manifest = {"input": {"name": "images", "image_size": 512}}
    ok, detail = verifier._input_contract(session, manifest)
    assert ok, detail
    ok, detail = verifier._output_contract(session)
    assert ok, detail


def test_verifier_rejects_wrong_image_size(verifier):
    session = _fake_session([SimpleNamespace(name="images", shape=[1, 1, 256, 256], type="tensor(float)")],
                            [SimpleNamespace(name="detections", shape=[1, 32, BOX_COLUMNS])])
    ok, detail = verifier._input_contract(session, {"input": {"name": "images", "image_size": 512}})
    assert not ok and "image_size" in detail


def test_verifier_rejects_wrong_output_columns(verifier):
    session = _fake_session([SimpleNamespace(name="images", shape=[1, 1, 512, 512], type="tensor(float)")],
                            [SimpleNamespace(name="detections", shape=[1, 32, 4])])
    ok, detail = verifier._output_contract(session)
    assert not ok and str(BOX_COLUMNS) in detail


def test_verifier_allows_dynamic_shapes(verifier):
    session = _fake_session([SimpleNamespace(name="images", shape=["batch", 1, "h", "w"],
                                             type="tensor(float)")],
                            [SimpleNamespace(name="boxes", shape=[None, -1, BOX_COLUMNS])])
    ok, _ = verifier._input_contract(session, {"input": {"name": "images", "image_size": 512}})
    assert ok
    ok, _ = verifier._output_contract(session)
    assert ok
