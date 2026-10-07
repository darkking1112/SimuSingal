"""原始 IQ 训练工具链回归（``training/build_iq_dataset.py`` / ``train_iq.py`` / ``iq_cnn.py``）。

``training/`` 不打包进 wheel，但它是"训练-推理口径一致"的关键环节，所以这里覆盖：

* ``build_iq_dataset.py``：卡片契约与推理端 ``iq_waveform_v1`` 一致、**样本可逐字节
  复现**、类别字典（A09 / 更宽字典）、TorchSig 混入的显式映射与跳过计数；
* ``train_iq.py``：``load_dataset`` 的契约校验（形状/通道/类别/字段缺失一律报错）、
  分信噪比统计语义、**不导入 torch 也能 ``--help``**；
* ``iq_cnn.py``（兼容转发层）与 ``training/amc_models``：结构形状、训练可跑通、导出 ONNX 后与
  PyTorch 数值一致且输入形状为 ``(1, 2, N)``、TCN 残差块确实参与前向与反传。

torch 相关用例统一 ``importorskip("torch")``：本仓库的常驻测试不依赖 torch，
训练依赖放在 ``[train]`` extra 里。
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
for _extra in (REPO_ROOT / "src", TRAINING):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from signal_analysis.core_api import generate_iq  # noqa: E402
from signal_analysis.algorithms.amc.iq_model import iq_waveform  # noqa: E402
from signal_analysis.contracts.amc import AMC_CLASSES  # noqa: E402
from signal_analysis.contracts.iq import (  # noqa: E402
    CLASS_SET_A09,
    CLASS_SET_CUSTOM,
    IQ_INPUT_CHANNELS,
    IQ_WAVEFORM_CONTRACT,
)

RATE = 200_000.0
SAMPLES = 512
MODES = list(AMC_CLASSES)


def _load(name, path=None):
    path = path or (TRAINING / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load("build_iq_dataset")


@pytest.fixture(scope="module")
def trainer():
    return _load("train_iq")


@pytest.fixture(scope="module")
def cnn():
    return _load("iq_cnn")


def _read(root):
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    with np.load(root / "iq_dataset.npz") as store:
        arrays = {name: store[name] for name in store.files}
    return SimpleNamespace(root=Path(root), card=card, arrays=arrays)


@pytest.fixture(scope="module")
def small_iq_dataset(tmp_path_factory, builder):
    """极小数据集：每类 2 个样本、窗口 512 点、无 TorchSig。"""
    root = tmp_path_factory.mktemp("iq_dataset")
    assert builder.main(["--output", str(root), "--per-class", "2", "--seed", "5",
                         "--samples", str(SAMPLES), "--duration-range", "0.3,0.4",
                         "--snr-range", "15,25", "--train-fraction", "0.5"]) == 0
    return _read(root)


# --------------------------------------------------------------------------- 数据集卡片


def test_iq_dataset_card_matches_inference_contract(small_iq_dataset):
    card, arrays = small_iq_dataset.card, small_iq_dataset.arrays
    contract = card["contract"]
    assert contract["task"] == "amc_iq"
    assert contract["input_contract"] == IQ_WAVEFORM_CONTRACT
    assert contract["layout"] == "iq_channels_first_v1"
    assert contract["channels"] == IQ_INPUT_CHANNELS == 2
    assert contract["normalization"] == "unit_rms"
    assert contract["samples"] == SAMPLES
    assert contract["class_set"] == CLASS_SET_A09
    assert contract["classes"] == list(AMC_CLASSES)
    assert contract["class_count"] == len(AMC_CLASSES)
    # 波形形状 = iq_waveform 的 (2, N)，且落盘为 float32
    waveforms = arrays["waveforms"]
    assert waveforms.shape == (len(MODES) * 2, IQ_INPUT_CHANNELS, SAMPLES)
    assert waveforms.dtype == np.float32 and np.isfinite(waveforms).all()
    # 训练与推理同源：每个窗口都是单位 RMS（前段与归一化只有一份实现）
    power = np.mean(waveforms[:, 0].astype(np.float64) ** 2
                    + waveforms[:, 1].astype(np.float64) ** 2, axis=1)
    assert np.allclose(power, 1.0, rtol=1e-5)
    assert card["sample_count"] == len(waveforms) == len(arrays["labels"])
    assert set(arrays["source"]) == {"generator"}
    assert set(arrays["split"]) == {"train", "val"}
    # 分层：每类都同时出现在训练与验证集（否则指标会完全失真）
    assert card["splits"]["strategy"] == "stratified_per_class"
    for mode in MODES:
        assert card["splits"]["per_class"][mode] == {"train": 1, "val": 1}
    assert card["statistics"]["labels"] == {mode: 2 for mode in MODES}
    assert card["statistics"]["sources"] == {"generator": 2 * len(MODES)}
    json.dumps(card, ensure_ascii=False, allow_nan=False)


def test_iq_dataset_window_reproduces_from_the_recorded_scene(builder):
    """样本必须能由记录里的场景参数重建（训练/验收复盘都依赖这一点）。"""
    args = builder._parse_args(["--output", "unused", "--per-class", "1",
                                "--samples", str(SAMPLES), "--snr-range", "20,20"])
    rng = np.random.default_rng(0)
    record = None
    for _ in range(builder.MAX_ATTEMPTS):
        record = builder._generator_sample(rng, args, "qpsk")
        if record is not None:
            break
    assert record is not None, "生成器场景抽样连续失败"
    scene = record["scene"]
    # 记录里的分析几何按 3 位小数取整（落盘/卡片要 JSON 友好）。取整引入的相位误差
    # 上限约 2π·5e-4·duration ≈ 1e-3 rad，因此只能用容差比较，不能逐位比较。
    assert record["analysis"]["offset_hz"] == round(record["analysis"]["offset_hz"], 3)
    assert record["analysis"]["bandwidth_hz"] == round(record["analysis"]["bandwidth_hz"], 3)
    samples, summary = generate_iq(scene["rate_hz"], scene["duration_s"], scene["signals"],
                                  scene["noise"], scene["seed"])
    tensor, meta = iq_waveform(samples, scene["rate_hz"], record["analysis"]["offset_hz"],
                              record["analysis"]["bandwidth_hz"], window_samples=SAMPLES)
    assert tensor.shape == record["waveform"].shape == (IQ_INPUT_CHANNELS, SAMPLES)
    assert np.allclose(tensor, record["waveform"], atol=5e-3)
    assert meta["samples"] == SAMPLES and meta["normalization"] == "unit_rms"
    assert meta["window_start"] == (meta["analysis_samples"] - SAMPLES) // 2
    # 训练/推理同源：记录里的波形信息就是 iq_waveform 的原样返回（只有被取整的
    # 两个几何字段需要用容差比）
    info = record["waveform_info"]
    for key in ("offset_hz", "bandwidth_hz"):
        assert info[key] == pytest.approx(meta[key], abs=1e-3)
    assert {k: v for k, v in info.items() if k not in ("offset_hz", "bandwidth_hz")} == {
        k: v for k, v in meta.items() if k not in ("offset_hz", "bandwidth_hz")}
    # 真值来自生成器：中心频率/带宽/功率/带内信噪比都能对回生成摘要
    signal = summary["signals"][0]
    truth = record["truth"]
    assert truth["snr_inband_db"] == signal["snr_inband_db"]
    assert truth["power_dbfs"] == signal["power_dbfs_actual"]
    assert truth["center_hz"] == pytest.approx(signal["offset"], abs=1e-3)
    assert truth["bandwidth_hz"] == pytest.approx(signal["bandwidth_actual"], abs=1e-3)


def test_iq_dataset_is_byte_reproducible(tmp_path, builder):
    first, second = tmp_path / "a", tmp_path / "b"
    arguments = ["--per-class", "1", "--seed", "9", "--samples", str(SAMPLES),
                 "--duration-range", "0.3,0.4"]
    assert builder.main(["--output", str(first), *arguments]) == 0
    assert builder.main(["--output", str(second), *arguments]) == 0
    assert (first / "iq_dataset.npz").read_bytes() == (second / "iq_dataset.npz").read_bytes()
    left = json.loads((first / "iq_dataset.json").read_text(encoding="utf-8"))
    right = json.loads((second / "iq_dataset.json").read_text(encoding="utf-8"))
    # 卡片里只有创建时间会变：其余字段（含分段统计）必须一致
    left.pop("created"), right.pop("created")
    assert left == right


def test_iq_dataset_supports_a_wider_class_dictionary(tmp_path, builder):
    root = tmp_path / "wide"
    assert builder.main(["--output", str(root), "--per-class", "1", "--seed", "3",
                         "--samples", str(SAMPLES), "--duration-range", "0.3,0.4",
                         "--class-set", "custom", "--classes", "bpsk,qpsk",
                         "--modes", "qpsk"]) == 0
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    assert card["contract"]["class_set"] == CLASS_SET_CUSTOM
    assert card["contract"]["classes"] == ["bpsk", "qpsk"]
    # 冻结的 A09 字典没有被改动
    assert list(AMC_CLASSES)[:2] != ["bpsk", "qpsk"]


def test_iq_dataset_rejects_invalid_arguments(builder, tmp_path):
    root = str(tmp_path / "bad")
    base = ["--output", root, "--per-class", "1", "--samples", str(SAMPLES)]
    with pytest.raises(SystemExit) as error:
        builder.main([*base, "--class-set", "a09", "--classes", "bpsk"])
    assert "A09 六类" in str(error.value) and "覆盖" in str(error.value)
    with pytest.raises(SystemExit, match="custom"):
        builder.main([*base, "--class-set", "custom"])
    with pytest.raises(SystemExit, match="未知样式"):
        builder.main([*base, "--modes", "qpsk,not-a-mode"])
    # custom 字典里不含样式映射出的类别：直接报错，不静默丢弃这些样本
    with pytest.raises(SystemExit, match="不在类别字典内"):
        builder.main([*base, "--class-set", "custom", "--classes", "bpsk", "--modes", "qpsk"])
    with pytest.raises(SystemExit, match="--samples"):
        builder.main(["--output", root, "--per-class", "1", "--samples", "32"])
    with pytest.raises(SystemExit, match="每类样本数"):
        builder.main(["--output", root, "--per-class", "0"])
    with pytest.raises(SystemExit, match="训练集比例"):
        builder.main(["--output", root, "--per-class", "1", "--train-fraction", "1.5"])
    with pytest.raises(SystemExit, match="占用带宽比例"):
        builder.main(["--output", root, "--per-class", "1", "--min-bandwidth-ratio", "0.5",
                      "--max-bandwidth-ratio", "0.1"])


def test_iq_dataset_requires_an_explicit_torchsig_mapping(builder, tmp_path):
    root = str(tmp_path / "mix")
    base = ["--output", root, "--per-class", "1", "--samples", str(SAMPLES),
            "--duration-range", "0.3,0.4"]
    with pytest.raises(SystemExit) as error:
        builder.main([*base, "--torchsig-bundle", str(tmp_path / "bundle")])
    assert "--torchsig-map" in str(error.value) and "不会替你猜映射" in str(error.value)
    with pytest.raises(SystemExit, match="没有 --torchsig-bundle"):
        builder.main([*base, "--torchsig-map", str(tmp_path / "map.json")])
    with pytest.raises(SystemExit, match="不存在"):
        builder.main([*base, "--torchsig-bundle", str(tmp_path / "bundle"),
                      "--torchsig-map", str(tmp_path / "missing.json")])
    # 映射目标不在类别字典内：拒绝而不是悄悄改字典
    bundle = _write_synthetic_bundle(tmp_path / "bundle", class_names=("QPSK",))
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"QPSK": "8psk"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="映射目标不在类别字典内"):
        builder.main([*base, "--torchsig-bundle", str(bundle), "--torchsig-map", str(mapping)])
    # 映射文件本身非法（空对象 / 非字符串目标）：同样报错
    mapping.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(SystemExit, match="映射文件为空"):
        builder.main([*base, "--torchsig-bundle", str(bundle), "--torchsig-map", str(mapping)])
    mapping.write_text(json.dumps({"QPSK": 3}), encoding="utf-8")
    with pytest.raises(SystemExit, match="非空字符串"):
        builder.main([*base, "--torchsig-bundle", str(bundle), "--torchsig-map", str(mapping)])
    mapping.write_text(json.dumps(["qpsk"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="JSON 对象"):
        builder.main([*base, "--torchsig-bundle", str(bundle), "--torchsig-map", str(mapping)])


def _write_synthetic_bundle(root, *, class_names=("QPSK",), seed=13):
    """造一个 ``torchsig_bundle_v1``（用项目生成器的波形，测的是读端口径）。"""
    tools = _load("torchsig_bundle")
    rate = RATE
    entries = []
    for index, name in enumerate(class_names):
        offset = -40_000.0 + 20_000.0 * index
        signals = [{"mode": "qpsk", "offset": offset, "bandwidth": 25_000.0,
                    "power_dbfs": -8.0}]
        samples, _ = generate_iq(rate, 0.3, signals, {"enabled": True, "snr_db": 18.0},
                                seed=seed + index)
        component = tools.component(
            class_name=name, center_freq=offset, bandwidth=25_000.0,
            lower_freq=offset - 12_500.0, upper_freq=offset + 12_500.0,
            start_in_samples=0, stop_in_samples=60_000, duration_in_samples=60_000,
            sample_rate=rate, snr_db=18.0)
        entries.append((tools.record(index=index, iq=np.asarray(samples, dtype=np.complex64),
                                     components=[component], sample_rate_hz=rate),
                        np.asarray(samples, dtype=np.complex64)))
    return tools.write_bundle(root, entries, sample_rate_hz=rate, torchsig_version="2.2.0")


# --------------------------------------------------------------------------- TorchSig 混入


def test_iq_dataset_mixes_torchsig_records_and_records_skips(tmp_path, builder):
    bundle = _write_synthetic_bundle(tmp_path / "bundle", class_names=("QPSK", "8PSK", "OFDM"))
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"QPSK": "qpsk", "8PSK": "8psk"}), encoding="utf-8")
    root = tmp_path / "mixed"
    assert builder.main(["--output", str(root), "--per-class", "2", "--seed", "4",
                         "--samples", str(SAMPLES), "--duration-range", "0.3,0.4",
                         "--class-set", "custom", "--classes", "qpsk,8psk",
                         "--modes", "qpsk",
                         "--torchsig-bundle", str(bundle),
                         "--torchsig-map", str(mapping)]) == 0
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    sources = card["sources"]
    assert sources["torchsig"]["mapping"] == {"8PSK": "8psk", "QPSK": "qpsk"}
    assert sources["torchsig"]["skipped"] == {"unmapped_class": 1}
    # 未映射的类名原样记录、不做猜测（"OFDM" 无映射 → 该条整条跳过并计数）
    assert sources["unmapped_classes"] == {"OFDM": 1}
    assert sources["unmapped_note"].startswith("TorchSig 类名原样记录")
    assert card["statistics"]["sources"] == {"generator": 2, "torchsig": 2}
    with np.load(root / "iq_dataset.npz") as store:
        assert set(store["source"]) == {"generator", "torchsig"}
        assert set(store["labels"]) == {"qpsk", "8psk"}
        # 标签来自映射目标，而不是 bundle 自己的类名（bundle 里没有 "qpsk" 这个类名）
        assert set(store["labels"][store["source"] == "torchsig"]) <= {"qpsk", "8psk"}
    # TorchSig 侧按类内位置划分：1 条记录按 0.8 比例落在训练集
    assert card["splits"]["per_class"]["8psk"] == {"train": 1, "val": 0}


def test_iq_dataset_mixing_rejects_layout_conflicts(tmp_path, builder):
    """bundle 的 class_name 属于它自己的体系：没有映射就连不上，必须报错。"""
    bundle = _write_synthetic_bundle(tmp_path / "bundle", class_names=("16QAM",))
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"16QAM": "qpsk"}), encoding="utf-8")
    root = tmp_path / "mixed"
    assert builder.main(["--output", str(root), "--per-class", "1", "--seed", "2",
                         "--samples", str(SAMPLES), "--duration-range", "0.3,0.4",
                         "--torchsig-bundle", str(bundle),
                         "--torchsig-map", str(mapping),
                         "--torchsig-per-class", "1"]) == 0
    card = json.loads((root / "iq_dataset.json").read_text(encoding="utf-8"))
    assert card["sources"]["torchsig"]["per_class_limit"] == 1
    assert card["sources"]["torchsig"]["record_count"] == 1
    assert card["statistics"]["sources"]["torchsig"] == 1  # 上限生效


def test_shipped_example_mapping_is_valid_and_only_targets_frozen_classes(builder):
    """``training/iq_map.example.json`` 是"占位版"：左边待填，右边必须落在 A09 内。"""
    path = TRAINING / "iq_map.example.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and payload
    mapping = builder._read_map(str(path))
    assert mapping == payload
    assert set(mapping.values()) <= set(AMC_CLASSES)
    # 占位键不是任何真实类名：照原样跑一次只会"未映射跳过并记录"，不会崩
    assert all(key.startswith("<") and key.endswith(">") for key in mapping)
    assert "iq_map.example.json" in (TRAINING / "build_iq_dataset.py").read_text(
        encoding="utf-8")


# --------------------------------------------------------------------------- 训练脚本


def test_train_iq_load_dataset_validates_the_contract(small_iq_dataset, trainer, tmp_path):
    card, waveforms, labels, splits, snrs, sources = trainer.load_dataset(small_iq_dataset.root)
    assert card["contract"]["classes"] == list(AMC_CLASSES)
    assert waveforms.shape == (len(MODES) * 2, 2, SAMPLES)
    assert len(labels) == len(splits) == len(snrs) == len(sources) == len(waveforms)
    assert set(splits) == {"train", "val"}

    def copy_to(name):
        target = tmp_path / name
        target.mkdir(parents=True, exist_ok=True)
        (target / "iq_dataset.json").write_text(
            (small_iq_dataset.root / "iq_dataset.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        with np.load(small_iq_dataset.root / "iq_dataset.npz") as store:
            np.savez(target / "iq_dataset.npz", **{key: store[key] for key in store.files})
        return target

    with pytest.raises(SystemExit, match="数据集不完整"):
        trainer.load_dataset(tmp_path / "absent")

    broken = copy_to("wrong_contract")
    payload = json.loads((broken / "iq_dataset.json").read_text(encoding="utf-8"))
    payload["contract"]["input_contract"] = "tf_image_v1"
    (broken / "iq_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="不一致"):
        trainer.load_dataset(broken)

    broken = copy_to("wrong_samples")
    payload = json.loads((broken / "iq_dataset.json").read_text(encoding="utf-8"))
    payload["contract"]["samples"] = SAMPLES * 2
    (broken / "iq_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="波形形状"):
        trainer.load_dataset(broken)

    broken = copy_to("wrong_channels")
    payload = json.loads((broken / "iq_dataset.json").read_text(encoding="utf-8"))
    payload["contract"]["channels"] = 1
    (broken / "iq_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="通道数"):
        trainer.load_dataset(broken)

    broken = copy_to("outside_labels")
    payload = json.loads((broken / "iq_dataset.json").read_text(encoding="utf-8"))
    payload["contract"]["classes"] = ["qpsk"]
    (broken / "iq_dataset.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="类别字典之外"):
        trainer.load_dataset(broken)

    broken = copy_to("missing_field")
    with np.load(small_iq_dataset.root / "iq_dataset.npz") as store:
        arrays = {key: store[key] for key in store.files if key != "snr_db"}
    np.savez(broken / "iq_dataset.npz", **arrays)
    with pytest.raises(SystemExit, match="缺少字段"):
        trainer.load_dataset(broken)


def test_train_iq_split_requires_both_sides(small_iq_dataset, trainer):
    card, waveforms, labels, splits, snrs, _ = trainer.load_dataset(small_iq_dataset.root)
    only_train = np.where(splits == "val", "train", splits)
    with pytest.raises(SystemExit, match="没有验证划分"):
        trainer._split(waveforms, labels, only_train, snrs)
    with pytest.raises(SystemExit, match="没有训练划分"):
        trainer._split(waveforms, labels, np.where(splits == "train", "val", splits), snrs)
    train_x, train_labels, train_snr, val_x, val_labels, val_snr = trainer._split(
        waveforms, labels, splits, snrs)
    assert len(train_x) == len(train_labels) == len(train_snr) == len(MODES)
    assert len(val_x) == len(val_labels) == len(val_snr) == len(MODES)


def test_train_iq_per_snr_buckets_are_explicit_about_empty_bands(trainer):
    waveforms = np.zeros((3, 2, 16), dtype=np.float32)
    labels = np.asarray(["a", "b", "a"])
    snrs = np.asarray([1.0, 2.0, 25.0])
    rows = trainer._per_snr(waveforms, labels, snrs, lambda batch: ["a"] * len(batch))
    assert [(row["low_db"], row["high_db"]) for row in rows] == [
        (-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, None)]
    assert [row["count"] for row in rows] == [0, 2, 0, 0, 1]
    assert rows[0]["accuracy"] is None  # 空档位显式"不适用"，不编 0
    assert rows[1]["accuracy"] == 0.5 and rows[4]["accuracy"] == 1.0
    assert rows[2]["accuracy"] is None and rows[3]["accuracy"] is None
    assert rows[4]["high_db"] is None


def test_training_scripts_do_not_import_torch_or_torchsig_at_module_level(trainer):
    """``--help``、纯 NumPy 的数据集构建/验收与模型目录查询都不依赖 torch / torchsig。

    网络定义在 ``training/amc_models``（由 ``train_iq`` 在真正训练时才导入），
    ``iq_cnn.py`` 只是兼容转发层；``amc_models/__init__.py`` 与 ``src`` 侧的模型目录
    必须保持零 torch，GUI 与 ``--help`` 才不用装 ``[train]``。
    """
    pattern = re.compile(r"^(?:import|from)\s+(torch|torchsig)(?:\.|\s|$)")
    for name in ("train_iq.py", "build_iq_dataset.py", "verify_iq.py",
                 "amc_models/__init__.py"):
        text = (TRAINING / name).read_text(encoding="utf-8")
        assert not [line for line in text.splitlines() if pattern.match(line)], name
    for name in ("amc_models/cnn.py", "amc_models/tcn.py", "amc_models/mcldnn.py",
                 "amc_models/petcgdnn.py", "amc_models/trainer.py", "amc_models/checkpoint.py"):
        text = (TRAINING / name).read_text(encoding="utf-8")
        assert [line for line in text.splitlines() if pattern.match(line)], name
    assert "torchsig" not in sys.modules


def test_train_iq_help_runs_without_torch(trainer, capsys):
    with pytest.raises(SystemExit) as error:
        trainer.main(["--help"])
    assert error.value.code == 0
    assert "--torchsig" not in capsys.readouterr().out


# --------------------------------------------------------------------------- 高级训练参数


def test_iq_plan_passes_advanced_tuning_args(tmp_path):
    """iq_plan 把高级超参转成 CLI 参数；缺省时不出现这些参数；非法值在起任务前拒绝。"""
    from signal_analysis.services.training_jobs import iq_plan
    repo = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    data.mkdir()
    (data / "iq_dataset.json").write_text("{}", encoding="utf-8")
    (data / "iq_dataset.npz").write_bytes(b"placeholder")
    base = {"repository": str(repo), "python": sys.executable, "arch": "cnn",
            "epochs": 5, "batch": 4, "lr": 1e-3, "seed": 7, "device": "cpu",
            "data": str(data)}

    tuned = {**base, "channels": "64,128,256", "kernel": 9, "dropout": 0.2,
             "weight_decay": 1e-3, "patience": 4}
    argv = iq_plan(tuned, tmp_path / "run")[1]["argv"]
    for flag, value in (("--channels", "64,128,256"), ("--kernel", "9"),
                        ("--dropout", "0.2"), ("--weight-decay", "0.001"),
                        ("--patience", "4")):
        assert flag in argv and argv[argv.index(flag) + 1] == value, flag

    argv = iq_plan(base, tmp_path / "run2")[1]["argv"]
    for flag in ("--channels", "--kernel", "--dropout", "--weight-decay", "--patience"):
        assert flag not in argv, flag  # 缺省 = 沿用模型目录声明的默认值

    # 新参数入口：没有旧 flag 的键走 --model-params JSON
    argv = iq_plan({**base, "model_params": {"channels": [64, 128, 256]}},
                   tmp_path / "run3")[1]["argv"]
    assert "--model-params" in argv
    assert json.loads(argv[argv.index("--model-params") + 1]) == {"channels": [64, 128, 256]}

    with pytest.raises(ValueError, match="3 个正整数"):
        iq_plan({**tuned, "channels": "64,128"}, tmp_path / "run4")
    with pytest.raises(ValueError, match="奇数"):
        iq_plan({**tuned, "kernel": 4}, tmp_path / "run5")
    with pytest.raises(ValueError, match="[Dd]ropout"):
        iq_plan({**tuned, "dropout": 1.0}, tmp_path / "run6")
    with pytest.raises(ValueError, match="早停轮数"):
        iq_plan({**tuned, "patience": 0}, tmp_path / "run7")
    with pytest.raises(ValueError, match="未知架构"):
        iq_plan({**tuned, "arch": "conv"}, tmp_path / "run8")
    with pytest.raises(ValueError, match="不小于 2"):
        iq_plan({**base, "arch": "tcn", "kernel": 1}, tmp_path / "run9")
    with pytest.raises(ValueError, match="不认识参数"):
        iq_plan({**base, "model_params": {"nope": 1}}, tmp_path / "run10")
    with pytest.raises(ValueError, match="不能同时"):
        iq_plan({**tuned, "model_params": {"kernel": 9}}, tmp_path / "run11")


def test_iq_plan_training_strategy_flags(tmp_path):
    """训练策略（调度器/判据/checkpoint）走同一份校验，并落到 CLI 与运行目录路径。"""
    from signal_analysis.services.training_jobs import iq_plan, iq_tuning_args
    repo = Path(__file__).resolve().parents[2]
    data = tmp_path / "data"
    data.mkdir()
    (data / "iq_dataset.json").write_text("{}", encoding="utf-8")
    (data / "iq_dataset.npz").write_bytes(b"placeholder")
    base = {"repository": str(repo), "python": sys.executable, "arch": "cnn",
            "epochs": 5, "batch": 4, "lr": 1e-3, "seed": 7, "device": "cpu",
            "data": str(data)}
    strategy = {**base, "scheduler": "plateau", "monitor": "val_loss", "save_checkpoint": True}
    args = iq_tuning_args(strategy, "cnn")
    assert args[args.index("--lr-scheduler") + 1] == "plateau"
    assert args[args.index("--monitor") + 1] == "val_loss"

    run = tmp_path / "run"
    argv = iq_plan(strategy, run)[1]["argv"]
    path = argv[argv.index("--save-checkpoint") + 1]
    assert path == str((run / "model" / "iq_checkpoint.pt").resolve())
    assert iq_plan(base, tmp_path / "run2")[1]["argv"].count("--save-checkpoint") == 0

    with pytest.raises(ValueError, match="cosine"):
        iq_tuning_args({**base, "scheduler": "step"}, "cnn")
    with pytest.raises(ValueError, match="accuracy"):
        iq_tuning_args({**base, "monitor": "loss"}, "cnn")
    with pytest.raises(ValueError, match="布尔值"):
        iq_tuning_args({**base, "save_checkpoint": "model.pt"}, "cnn")



    """preflight_iq 在快照之前补齐窗口长度并拒绝未知模型（纯 Python，无 torch）。"""
    from signal_analysis.services.training_jobs import preflight_iq
    from signal_analysis.algorithms.amc.ai_model import CATALOG_VERSION
    from signal_analysis.contracts.iq import DEFAULT_IQ_SAMPLES
    config = {"arch": "cnn", "epochs": 1, "batch": 2, "lr": 1e-3}
    assert preflight_iq(config)["samples"] == DEFAULT_IQ_SAMPLES
    assert preflight_iq({**config, "samples": 512})["samples"] == 512
    assert preflight_iq({**config, "catalog_version": CATALOG_VERSION})["samples"] == 1024
    with pytest.raises(ValueError, match="未知架构"):
        preflight_iq({**config, "arch": "nope"})
    # 界面声明的目录版本与训练源码不一致：先刷新模型列表，避免按旧参数起任务
    with pytest.raises(ValueError, match="刷新模型列表"):
        preflight_iq({**config, "catalog_version": CATALOG_VERSION + 1})


def test_query_catalog_reads_the_catalog_from_a_training_environment(tmp_path):
    """query_catalog 在训练环境子进程里读目录，并补上每个模型的依赖可用性。"""
    from signal_analysis.services.training_jobs import query_catalog
    from signal_analysis.algorithms.amc.ai_model import CATALOG_VERSION, available_models
    repo = Path(__file__).resolve().parents[2]
    payload = query_catalog(sys.executable, str(repo))
    assert payload["catalog_version"] == CATALOG_VERSION
    assert [model["id"] for model in payload["models"]] == list(available_models())
    cnn = payload["models"][0]
    assert cnn["layers"] and cnn["params"] and cnn["missing"] == []
    assert all("missing" in model for model in payload["models"])

    with pytest.raises(ValueError, match="缺少 src"):
        query_catalog(sys.executable, str(tmp_path))
    # 解释器路径非法时给出可读报错，而不是崩溃
    with pytest.raises(ValueError, match="无法在训练环境执行"):
        query_catalog(str(tmp_path / "python"), str(repo))


# --------------------------------------------------------------------------- IQCNN / TCN


def test_build_model_shapes_and_architecture_guard(cnn):
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="架构"):
        cnn.build_model("rnn", 3)
    for arch in cnn.ARCHITECTURES:
        model = cnn.build_model(arch, 3, samples=1024)
        with torch.no_grad():
            output = model(torch.zeros(1, 2, 1024))
        assert output.shape == (1, 3)
        assert torch.isfinite(output).all()
    packed = cnn.SoftmaxClassifier(cnn.build_model("cnn", 3))
    with torch.no_grad():
        probabilities = packed(torch.randn(1, 2, 256))
    assert probabilities.shape == (1, 3)
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------ 第三方模型（MCLDNN / PETCGDNN）


def test_wrapped_custom_models_train_export_and_keep_the_contract(cnn, tmp_path):
    """``custom/`` 包装层：目录里的每个模型都能按 1024 点窗口前向/反传并导出。

    契约与 cnn/tcn 一致：输入 ``(B, 2, N)``、输出 ``(B, C)`` logits；导出走
    ``amc_models.export_onnx``（图内 softmax + batch=1 探针 + 2e-4 数值容差）。
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    import amc_models
    from signal_analysis.algorithms.amc.ai_model import SPECS

    for name in SPECS:
        model = amc_models.build_model(name, classes=6, samples=1024)
        model.train()
        logits = model(torch.randn(2, 2, 1024))
        assert logits.shape == (2, 6), name
        assert torch.isfinite(logits).all(), name
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        idle = [label for label, parameter in model.named_parameters()
                if parameter.grad is None]
        assert not idle, f"{name} 未参与前向的参数：{idle}"
        amc_models.export_onnx(model.eval(), tmp_path / f"{name}.onnx",
                               classes=list("abcdef"), samples=1024, opset=17)


def test_petcgdnn_takes_its_window_from_the_dataset(cnn):
    """PETCGDNN 的旋转层把窗口长度写进权重形状：构建时必须给 samples，且不暴露该参数。"""
    pytest.importorskip("torch")
    import amc_models
    with pytest.raises(ValueError, match="窗口长度"):
        amc_models.build_model("petcgdnn", classes=6)
    with pytest.raises(ValueError, match="不少于 16"):
        amc_models.build_model("petcgdnn", classes=6, samples=8)
    with pytest.raises(ValueError, match="不认识参数"):
        amc_models.build_model("petcgdnn", classes=6, samples=1024,
                               params={"frame_length": 512})


@pytest.mark.parametrize("arch", ["cv_trn", "poet"])
def test_transformer_models_are_deterministic_in_eval(cnn, arch):
    """第三方 Transformer 的随机部分（RPO/物理增广/drop path）必须只在 train 生效。"""
    torch = pytest.importorskip("torch")
    import amc_models
    model = amc_models.build_model(arch, classes=4, samples=512)
    waveform = torch.randn(2, 2, 512)
    model.train()
    outputs = [model(waveform) for _ in range(3)]
    assert any(not torch.allclose(outputs[0], other) for other in outputs[1:]), arch

    model.eval()
    with torch.no_grad():
        first, second = model(waveform), model(waveform)
    assert torch.equal(first, second), arch          # eval 无随机行为（导出/验收可复现）
    assert first.shape == (2, 4)


def test_third_party_transformer_copies_do_not_need_timm():
    """CV_TRN/POET 的训练副本用本地工具替代 timm；custom/ 原文件保持原样（只读来源）。"""
    import importlib.util
    timm_import = re.compile(r"^(?:import|from)\s+timm(?:\.|\s|$)")
    for name in ("cv_trn", "poet"):
        text = (TRAINING / "amc_models" / f"{name}.py").read_text(encoding="utf-8")
        assert not [line for line in text.splitlines() if timm_import.match(line)], name
        assert "._nn_utils import" in text, name
    # 原文件仍带 timm 导入：它是我们的来源，副本只改导入与设备处理
    for name in ("CV_TRN.py", "POET.py"):
        assert "from timm" in (REPO_ROOT / "custom" / name).read_text(encoding="utf-8"), name
    if importlib.util.find_spec("torch") is None:
        pytest.skip("没有 torch 时不检查依赖可用性")
    import amc_models
    for name in ("cv_trn", "poet"):
        assert "timm" not in amc_models.missing_requirements(name), name


def test_local_timm_replacements_keep_the_original_semantics():
    """`_nn_utils` 的 `trunc_normal_` / `DropPath` 与 timm 同语义。

    注意 timm 的 `trunc_normal_(t, std=0.02)` 语义是"标准正态截断在 ±2（绝对单位）后乘以 std"，
    因此小 std 下几乎不产生截断——副本刻意保留这一行为，避免"更正确"的实现改变初始化口径。
    """
    torch = pytest.importorskip("torch")
    from amc_models import _nn_utils
    torch.manual_seed(0)
    tensor = torch.empty(200_000)
    _nn_utils.trunc_normal_(tensor, std=0.02)
    assert abs(float(tensor.mean())) < 5e-4
    assert abs(float(tensor.std()) - 0.02) < 5e-4
    assert float(tensor.abs().max()) <= 2.0                    # 绝对值上限（timm 约定）

    torch.manual_seed(0)
    bounded = torch.empty(200_000)
    _nn_utils.trunc_normal_(bounded, std=1.0)                  # std=1 时 ±2 截断真正生效
    assert float(bounded.abs().max()) <= 2.0
    assert 0.85 < float(bounded.std()) < 1.0

    layer = _nn_utils.DropPath(0.5)
    inputs = torch.ones(20_000, 4)
    layer.eval()
    assert torch.equal(layer(inputs), inputs)               # eval 恒等
    layer.train()
    dropped = layer(inputs)
    zero_ratio = float((dropped == 0).float().mean())
    assert 0.45 < zero_ratio < 0.55                         # 按 0.5 丢整条
    assert float(dropped.mean()) == pytest.approx(1.0, abs=0.02)   # 除以 keep 保均值
    assert torch.equal(_nn_utils.DropPath(0.0)(inputs), inputs)


@pytest.mark.parametrize("arch", ["mcldnn", "petcgdnn"])
def test_train_iq_records_the_resolved_params_of_wrapped_models(trainer, small_iq_dataset,
                                                               tmp_path, arch):
    """训练记录只写目录参数（不再写死 cnn/tcn 的 channels/kernel/dropout）。"""
    pytest.importorskip("torch")
    assert trainer.main(["--data", str(small_iq_dataset.root), "--arch", arch, "--epochs", "1",
                         "--batch-size", "4", "--onnx-dir", str(tmp_path)]) == 0
    training = json.loads((tmp_path / "iq_manifest.json").read_text(encoding="utf-8"))["training"]
    assert training["arch"] == arch
    assert training["model_revision"] >= 1
    assert training["num_params"] > 0
    assert training["model_params"]  # 目录参数（默认值也记录）
    for legacy in ("channels", "kernel", "dropout"):
        assert legacy not in training, legacy
    card = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert card["arch"] == arch and card["model_params"] == training["model_params"]
    assert training["dataset_contract"]["samples"] == 512


def test_train_iq_records_training_strategy_and_checkpoint(trainer, small_iq_dataset, tmp_path):
    """`--lr-scheduler/--monitor/--save-checkpoint` opt-in：记录最佳验证损失与 checkpoint 元信息。"""
    pytest.importorskip("torch")
    checkpoint = tmp_path / "iq_checkpoint.pt"
    assert trainer.main(["--data", str(small_iq_dataset.root), "--arch", "cnn", "--epochs", "3",
                         "--batch-size", "4", "--lr-scheduler", "plateau", "--monitor", "val_loss",
                         "--patience", "10", "--save-checkpoint", str(checkpoint),
                         "--onnx-dir", str(tmp_path)]) == 0
    training = json.loads((tmp_path / "iq_manifest.json").read_text(encoding="utf-8"))["training"]
    assert training["monitor"] == "val_loss" and training["lr_scheduler"] == "plateau"
    assert training["best_validation_loss"] is not None
    assert training["best_validation_accuracy"] is not None   # 旧口径字段保留
    assert training["checkpoint"] == str(checkpoint)

    from amc_models.checkpoint import read_checkpoint
    payload = read_checkpoint(checkpoint)
    assert payload["model_id"] == "cnn" and payload["samples"] == 512
    assert payload["extra"]["monitor"] == "val_loss"
    assert payload["classes"] == training["dataset_contract"]["classes"]

    card = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert card["best_validation_loss"] == training["best_validation_loss"]
    assert card["history"][0]["validation_loss"] is not None
    # 默认路径不写 checkpoint，且仍是余弦退火 + 按验证准确率存优
    assert trainer.main(["--data", str(small_iq_dataset.root), "--arch", "cnn", "--epochs", "1",
                         "--batch-size", "4", "--onnx-dir", str(tmp_path / "plain")]) == 0
    plain = json.loads((tmp_path / "plain" / "iq_manifest.json").read_text(encoding="utf-8"))
    assert plain["training"]["monitor"] == "accuracy"
    assert plain["training"]["lr_scheduler"] == "cosine"
    assert plain["training"]["checkpoint"] is None


def test_tcn_forward_uses_the_residual_block_stack(cnn):
    """TCN 的残差块必须真的参与前向与反传（防止"参数在 state_dict 里但前向不用"）。

    模型默认训练态时 ``classifier`` 的 Dropout 会让两次前向天然不同，直接比较输出会误通过，
    因此扰动参数前后都在 ``eval()`` + ``no_grad()`` 下取输出。
    """
    torch = pytest.importorskip("torch")
    model = cnn.build_model("tcn", 3).eval()
    waveform = torch.randn(2, 2, 256)
    with torch.no_grad():
        before = model(waveform)
        next(model.blocks.parameters()).data.add_(1.0)
        assert not torch.allclose(before, model(waveform))

    model.train()
    model(waveform).sum().backward()
    gradients = [parameter.grad for parameter in model.blocks.parameters()]
    assert gradients
    assert all(gradient is not None and bool(gradient.abs().sum() > 0)
               for gradient in gradients)


def test_train_classifier_learns_and_exports_consistently(cnn, tmp_path):
    torch = pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")
    rng = np.random.default_rng(0)
    classes = ["tone", "noise"]
    # 两个"类别"用明显的波形差异：谱线 vs 白噪声，2 类 24 样本足够学到
    tones = np.stack([np.cos(2 * np.pi * 0.05 * np.arange(256)) for _ in range(24)])
    noise = rng.standard_normal((24, 2, 256))
    waveforms = np.concatenate([np.stack([tones, np.zeros_like(tones)], axis=1), noise])
    # 训练标签是类别下标（train_iq 用契约里的 classes 顺序换算），不是类别名
    labels = np.asarray([0, 1] * 24, dtype=np.int64)
    trained = cnn.train_classifier(waveforms[:32], labels[:32], waveforms[32:], labels[32:],
                                   classes=classes, arch="cnn", epochs=4, batch_size=8,
                                   seed=1, verbose=False)
    assert trained["arch"] == "cnn" and trained["best_accuracy"] is not None
    assert len(trained["history"]) == trained["epochs_run"] > 0
    model = trained["model"]
    model.eval()
    path = tmp_path / "iq.onnx"
    cnn.export_onnx(model, path, classes=classes, samples=256, opset=17)
    assert path.is_file()
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == [cnn.INPUT_NAME]
    assert list(session.get_inputs()[0].shape) == [1, 2, 256]
    assert [item.name for item in session.get_outputs()] == [cnn.OUTPUT_NAME]
    # 导出图与 PyTorch 数值一致（含 softmax 已在图内）
    probe = rng.standard_normal((1, 2, 256)).astype(np.float32)
    got = np.asarray(session.run([cnn.OUTPUT_NAME], {cnn.INPUT_NAME: probe})[0],
                     dtype=np.float64)
    with torch.no_grad():
        want = cnn.SoftmaxClassifier(model)(torch.as_tensor(probe)).numpy().astype(np.float64)
    assert got.shape == (1, 2)
    assert float(np.max(np.abs(got - want))) < cnn.TOLERANCE
    assert float(got.sum()) == pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------- 训练策略（阶段 4：可选调度与判据）


def _toy_split(seed=0, samples=256):
    """tones（类别 0）/ noise（类别 1）各 24 条，按奇偶交替划分，两侧类别均衡。"""
    rng = np.random.default_rng(seed)
    tones = np.cos(2 * np.pi * 0.05 * np.arange(samples))[None, :].repeat(24, axis=0)
    waveforms = np.concatenate([
        np.stack([tones, tones], axis=1),                     # (24, 2, N) 类别 0
        rng.standard_normal((24, 2, samples))]).astype(np.float32)  # 类别 1
    labels = np.array([0] * 24 + [1] * 24, dtype=np.int64)
    order = np.arange(len(labels))
    return (waveforms[order[0::2]], labels[order[0::2]],
            waveforms[order[1::2]], labels[order[1::2]])


def test_train_classifier_default_strategy_is_unchanged(cnn):
    """不传新参数时仍是"余弦退火 + 按验证准确率存优"，历史里只多出记录字段。"""
    pytest.importorskip("torch")
    train_x, train_y, val_x, val_y = _toy_split()
    outcome = cnn.train_classifier(train_x, train_y, val_x, val_y, classes=["a", "b"],
                                   arch="cnn", epochs=4, batch_size=8, seed=1, verbose=False)
    assert outcome["monitor"] == "accuracy" and outcome["lr_scheduler"] == "cosine"
    assert outcome["checkpoint"] is None
    history = outcome["history"]
    assert [entry["epoch"] for entry in history] == list(range(1, len(history) + 1))
    assert all(set(entry) == {"epoch", "loss", "validation_accuracy", "validation_loss",
                              "learning_rate"} for entry in history)
    best = max(history, key=lambda entry: (entry["validation_accuracy"], -entry["epoch"]))
    assert outcome["best_epoch"] == best["epoch"]
    assert outcome["best_accuracy"] == pytest.approx(best["validation_accuracy"])
    assert outcome["best_loss"] == pytest.approx(best["validation_loss"])
    assert history[0]["learning_rate"] > history[-1]["learning_rate"]  # 余弦退火在衰减


def test_train_classifier_can_monitor_validation_loss_and_plateau(cnn):
    """``monitor="val_loss"`` 按最低验证损失存优；plateau 调度按验证损失、每 5 轮降半。"""
    pytest.importorskip("torch")
    train_x, train_y, val_x, val_y = _toy_split()
    outcome = cnn.train_classifier(train_x, train_y, val_x, val_y, classes=["a", "b"],
                                   arch="cnn", epochs=8, batch_size=8, seed=1, verbose=False,
                                   lr_scheduler="plateau", monitor="val_loss", patience=100)
    history = outcome["history"]
    best = min(history, key=lambda entry: (entry["validation_loss"], entry["epoch"]))
    assert outcome["best_epoch"] == best["epoch"]
    assert outcome["best_loss"] == pytest.approx(best["validation_loss"])
    assert outcome["best_accuracy"] == pytest.approx(best["validation_accuracy"])
    assert outcome["monitor"] == "val_loss" and outcome["lr_scheduler"] == "plateau"
    # ReduceLROnPlateau(val_loss, factor=0.5, patience=5)：前 5 轮不可能降，
    # 之后只按 1e-3 × 0.5^k 的台阶下降，且停在 min_lr 之上
    assert all(entry["learning_rate"] == pytest.approx(1e-3) for entry in history[:5])
    assert {entry["learning_rate"] for entry in history} <= {1e-3, 5e-4, 2.5e-4, 1.25e-4}
    with pytest.raises(ValueError, match="cosine 或 plateau"):
        cnn.train_classifier(train_x, train_y, val_x, val_y, classes=["a", "b"], arch="cnn",
                             epochs=1, verbose=False, lr_scheduler="step")
    with pytest.raises(ValueError, match="accuracy 或 val_loss"):
        cnn.train_classifier(train_x, train_y, val_x, val_y, classes=["a", "b"], arch="cnn",
                             epochs=1, verbose=False, monitor="f1")


def test_checkpoint_round_trip_and_compatibility_checks(cnn, tmp_path):
    """checkpoint 带身份信息；模型 ID/结构版本/结构参数/类别/窗口任一不符即拒绝加载。"""
    torch = pytest.importorskip("torch")
    from amc_models import checkpoint as ckpt
    train_x, train_y, val_x, val_y = _toy_split()
    path = tmp_path / "iq_checkpoint.pt"
    outcome = cnn.train_classifier(train_x, train_y, val_x, val_y, classes=["a", "b"],
                                   arch="cnn", epochs=2, batch_size=8, seed=1, verbose=False,
                                   checkpoint=path)
    assert path.is_file()
    payload = ckpt.read_checkpoint(path)
    assert payload["model_id"] == "cnn" and payload["samples"] == 256
    assert payload["classes"] == ["a", "b"]
    assert payload["params"] == {"channels": [32, 64, 128], "kernel": 7, "dropout": 0.1}
    assert payload["extra"]["monitor"] == "accuracy" and payload["extra"]["epochs_run"] >= 1
    assert outcome["checkpoint"]["model_id"] == "cnn"

    restored, _ = ckpt.restore(path, classes=["a", "b"])
    trained = outcome["model"]
    for key, value in trained.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key

    with pytest.raises(ValueError, match="模型 cnn"):
        ckpt.load_state(path, arch="tcn", model=cnn.build_model("tcn", 2, samples=256))
    with pytest.raises(ValueError, match="类别顺序"):
        ckpt.restore(path, classes=["b", "a"])
    with pytest.raises(ValueError, match="窗口长度"):
        ckpt.load_state(path, arch="cnn", model=cnn.build_model("cnn", 2, samples=256),
                        samples=512)
    with pytest.raises(ValueError, match="结构参数"):
        ckpt.load_state(path, arch="cnn", model=trained, params={"kernel": 9})
    torch.save(dict(payload, model_revision=payload["model_revision"] + 1),
               tmp_path / "older.pt")
    with pytest.raises(ValueError, match="结构版本"):
        ckpt.restore(tmp_path / "older.pt", classes=["a", "b"])
