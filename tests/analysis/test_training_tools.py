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


# --------------------------------------------------------- 场景扩充（标签安全、可关闭）
# 这些选项只为"扩充训练场景与扰动"服务，每一条的默认值都必须与历史行为逐位一致，
# 否则既有数据集、既有验收结论都会在无人察觉的情况下改变。


def _args(builder, *argv):
    """构建器参数对象（``--output`` 必填，这里不落盘）。"""
    return builder._parse_args(["--output", "/tmp/unused-dataset", *argv])


def test_slot_padding_off_reproduces_the_legacy_layout(builder):
    """``--cfo-ratio 0``（默认）：槽位划分与历史实现逐位一致，随机数消耗也一致。"""
    args = _args(builder, "--min-bandwidth-ratio", "0.02", "--max-bandwidth-ratio", "0.08")
    rate = 1_000_000.0
    rng = np.random.default_rng(3)
    slots = builder._plan_slots(rng, args, 4, rate, padding=0.0)
    legacy_rng = np.random.default_rng(3)
    cursor = -rate / 2.0 + args.guard_hz
    high_limit = rate / 2.0 - args.guard_hz
    legacy = []
    for _ in range(4):
        width = legacy_rng.uniform(args.min_bandwidth_ratio, args.max_bandwidth_ratio) * rate
        room = high_limit - cursor
        if room < width + args.min_gap_hz:
            break
        start = cursor + legacy_rng.uniform(0.0, room - width)
        low, high = float(start), float(start + width)
        legacy.append((low, high, high - low))
        cursor = high + args.min_gap_hz
    assert len(slots) == len(legacy) == 4
    assert slots == legacy
    assert rng.random() == legacy_rng.random()
    for low, high, width in slots:
        assert high - low == width          # 关闭抖动时槽宽就是抽到的带宽
        assert args.min_bandwidth_ratio * rate <= width <= args.max_bandwidth_ratio * rate


def test_slot_padding_does_not_amplify_the_target_bandwidth(builder):
    """``--cfo-ratio`` 只买到槽内余量，不能把 ``--max-bandwidth-ratio`` 的标定放大。"""
    rate = 1_000_000.0
    plain = _args(builder, "--min-bandwidth-ratio", "0.05", "--max-bandwidth-ratio", "0.2")
    padded = _args(builder, "--min-bandwidth-ratio", "0.05", "--max-bandwidth-ratio", "0.2",
                   "--cfo-ratio", "0.6")
    slots_plain = builder._plan_slots(np.random.default_rng(5), plain, 4, rate, padding=0.0)
    slots_padded = builder._plan_slots(np.random.default_rng(5), padded, 4, rate, padding=0.6)
    assert [slot[2] for slot in slots_padded] == [slot[2] for slot in slots_plain]
    for (_, _, target), (low, high, _) in zip(slots_plain, slots_padded):
        assert target <= padded.max_bandwidth_ratio * rate
        assert high - low == pytest.approx(target * (1.0 + 2.0 * 0.6))
        assert high - low >= target


def test_signal_spec_keeps_the_band_inside_the_slot(builder):
    """抖动/槽内余量之下，占用带始终落在槽内（所以标签框不会互相重叠）。"""
    args = _args(builder, "--cfo-ratio", "0.6", "--min-bandwidth-ratio", "0.05",
                 "--max-bandwidth-ratio", "0.2")
    rate = 1_000_000.0
    slots = builder._plan_slots(np.random.default_rng(9), args, 5, rate, padding=0.6)
    assert slots
    for low, high, target in slots:
        assert target <= args.max_bandwidth_ratio * rate
        assert high - low > target          # 槽比占用带宽宽 → _signal_spec 会做窄化
        for mode in ("qpsk", "fm", "fh_rc"):
            spec = builder._signal_spec(np.random.default_rng(2), mode, low, high,
                                        target, -3.0, args)
            assert spec["bandwidth"] == pytest.approx(target)
            band_low = spec["offset"] - spec["bandwidth"] / 2.0
            band_high = spec["offset"] + spec["bandwidth"] / 2.0
            assert low - 1e-9 <= band_low and band_high <= high + 1e-9


def test_scene_rate_without_range_consumes_no_random_numbers(builder):
    """默认（不给 ``--rate-range``）时不抽采样率：历史数据集才逐位可复现。"""
    args = _args(builder)
    rng = np.random.default_rng(2)
    assert builder._scene_rate(rng, args) == args.rate
    assert rng.random() == np.random.default_rng(2).random()


def test_scene_rate_range_is_sampled_per_scene(builder):
    args = _args(builder, "--rate-range", "800000,1200000")
    rng = np.random.default_rng(4)
    values = [builder._scene_rate(rng, args) for _ in range(50)]
    assert all(800_000.0 <= value <= 1_200_000.0 for value in values)
    assert len(set(values)) == 50


def test_snr_bias_off_keeps_the_uniform_draw(builder):
    """``--snr-bias 0``（默认）等价于原来的 ``rng.uniform(*snr_range)``。"""
    args = _args(builder)
    rng = np.random.default_rng(9)
    other = np.random.default_rng(9)
    drawn = [builder._sample_snr(rng, args) for _ in range(20)]
    expected = [float(other.uniform(*args.snr_range)) for _ in range(20)]
    assert drawn == expected


def test_snr_bias_skews_towards_low_snr_monotonically(builder):
    """偏置越大，低信噪比占比在任何阈值上都单调增加（三角分布做不到这一点）。"""
    previous_mean, previous_tail = None, None
    for bias in (0.2, 0.5, 1.0):
        args = _args(builder, "--snr-bias", str(bias))
        rng = np.random.default_rng(1)
        values = np.array([builder._sample_snr(rng, args) for _ in range(4000)])
        assert values.min() >= args.snr_range[0]
        assert values.max() <= args.snr_range[1]
        mean, tail = float(values.mean()), float((values < 0.0).mean())
        if previous_mean is not None:
            assert mean < previous_mean
            assert tail > previous_tail
        previous_mean, previous_tail = mean, tail
    assert previous_tail > 0.3             # bias=1.0 时 0 dB 以下约占 38%（均匀分布 14%）


def test_dataset_builder_rejects_augmentation_options_out_of_range(builder, tmp_path):
    for argv, hint in ((["--cfo-ratio", "1.5"], "cfo-ratio"),
                       (["--snr-bias", "-0.1"], "snr-bias"),
                       (["--rate", "10000"], "采样率")):
        with pytest.raises(SystemExit) as error:
            builder.main(["--output", str(tmp_path / "bad"), "--count", "1", *argv])
        assert hint in str(error.value)


def test_augmented_dataset_is_still_label_safe(builder, tmp_path):
    """打开全部扩充旋钮后：标签仍在图内、两轴都不碰撞、带宽比例不超标定区间。"""
    root = tmp_path / "augmented"
    code = builder.main(["--output", str(root), "--count", "8", "--seed", "3",
                         "--image-size", "64", "--nfft", "64", "--duration-range", "0.1,0.2",
                         "--max-signals", "3", "--noise-only-ratio", "0.1",
                         "--rate-range", "800000,1200000", "--snr-bias", "0.7",
                         "--cfo-ratio", "0.6", "--min-bandwidth-ratio", "0.02",
                         "--max-bandwidth-ratio", "0.33"])
    assert code == 0
    card = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert card["scene"]["rate_hz_range"] == [800_000.0, 1_200_000.0]
    assert card["scene"]["snr_bias"] == 0.7 and card["scene"]["cfo_ratio"] == 0.6
    assert card["scene"]["bandwidth_ratio_range"] == [0.02, 0.33]
    records = [json.loads(line) for line in
               (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rates, ratios = set(), []
    for record in records:
        rate = record["scene"]["rate_hz"]
        rates.add(rate)
        assert 800_000.0 <= rate <= 1_200_000.0
        boxes = np.asarray(record["boxes"], dtype=np.float64).reshape(-1, 6)
        if len(boxes):
            assert np.all(boxes[:, 0] - boxes[:, 2] / 2 >= -1e-9)
            assert np.all(boxes[:, 0] + boxes[:, 2] / 2 <= 1 + 1e-9)
            assert np.all(boxes[:, 1] - boxes[:, 3] / 2 >= -1e-9)
            assert np.all(boxes[:, 1] + boxes[:, 3] / 2 <= 1 + 1e-9)
            spans = sorted((box[1] - box[3] / 2, box[1] + box[3] / 2) for box in boxes)
            for (_, previous_high), (next_low, _) in zip(spans, spans[1:]):
                assert next_low >= previous_high - 1e-6
            centers = np.unique(np.round(boxes[:, 1] * rate, 6))
            assert len(centers) == len(boxes)     # 频率槽互斥，中心不会重合
        for entry in record["truth"]:
            ratios.append(entry["bandwidth_hz"] / rate)
    assert len(rates) > 1                        # 每个场景独立抽采样率
    assert ratios and max(ratios) <= 0.33 + 0.01  # FM 实测占用略超声明带宽
    statistics = card["statistics"]["sources"]["generator"]
    assert statistics["collision_pairs"] == 0
    assert statistics["bandwidth_ratio"]["count"] == len(ratios)
    assert statistics["bandwidth_ratio"]["max"] <= 0.33 + 0.01


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


@pytest.mark.parametrize("name", [
    "build_dataset", "build_amc_dataset", "build_iq_dataset", "build_torchsig",
    "ingest_torchsig", "train_iq", "train_amc", "train_yolox",
    "verify_iq", "verify_amc", "verify_onnx",
])
def test_every_training_script_prints_help(name, capsys):
    """``--help`` 必须能跑：argparse 对 help 文本做 ``%`` 格式化，正文里出现裸
    ``%``（例如“0 dB 以下占比从 14% 升到 28%”）会让 ``--help`` 直接抛
    ``ValueError``。这条用例把整目录的脚本都釘住。"""
    module = _load(name)
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out.lstrip().startswith("usage:")
