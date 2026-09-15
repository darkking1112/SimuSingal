"""TorchSig 数据适配的离线回归（``torchsig_bundle_v1`` → ``tf_image_v1`` 数据集）。

为什么这个测试能在没有 torchsig 的机器上跑
------------------------------------------

生成和导入被拆成
两半：``training/build_torchsig.py``（真实生成见 test_torchsig_real.py）与
``training/ingest_torchsig.py``（纯 NumPy，本文件覆盖）。测试里用项目自己的
:func:`generate_iq` 造 IQ、用 ``torchsig_bundle.py`` 写 bundle，全程不碰 torchsig，
并且显式断言 ``torchsig`` 从未进入 ``sys.modules``。

覆盖的核心不比"函数被调用过"更弱——它钉住三条容易静默出错的约定
--------------------------------------------------------------

1. **单位**：TorchSig 2.2 的 ``start``/``stop``/``duration`` 是"占记录长度的比例"，
   而 bundle 用秒制。写端会同时写出采样点字段，读端必须**优先用采样点**；否则
   信号会被整体压到记录头部，而框仍在图内、形状校验查不出来。
2. **标签来源**：框只能由 :func:`band_to_box` 从频段/时间真值算出（本项目唯一的
   "真值 → 框"公式）。TorchSig 的 ``yolo_label`` 是另一套 ``y`` 轴约定，一旦顺手
   消费，框会整体翻转。这里断言读端只有 ``component_band`` 这一条来源。
3. **拒绝而不是容忍**：数据集有一条被测试锁定的不变量——标签框在频率轴上互不
   重叠。TorchSig 允许同频叠加，因此每条记录都要过几何守卫；这里逐个守卫都构造
   一份坏 bundle，断言它**硬失败**（``SystemExit``）而不是悄悄丢框（丢框会让图像里
   的信号变成训练时的假阴性）。
"""

from __future__ import annotations

import argparse
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
from signal_analysis.ml import (  # noqa: E402
    BOX_COLUMNS,
    IMAGE_LAYOUT,
    INPUT_CONTRACT,
    OUTPUT_LAYOUT,
    box_to_band,
    measure_band,
    spectral_context,
)

#: 场景参数（与训练脚本默认一致的量级：1 MSps / 32.8 ms / nfft 512 / 1024² 图像）
RATE = 1_000_000.0
RECORD_SAMPLES = 32768
NFFT = 512
IMAGE_SIZE = 1024
TORCHSIG_VERSION = "2.2.0"


def _load(name, path):
    """按路径加载 training/ 下的脚本（它们不在包内，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(f"training_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: 模块级（顶格）的 torchsig 导入。注意不能简单按前缀匹配：``from torchsig_bundle
#: import ...``（本项目自己的格式模块）也以 ``from torchsig`` 开头。
_TOP_LEVEL_TORCHSIG = re.compile(r"^(?:import|from)\s+torchsig(?:\.|\s|$)")


def _top_level_torchsig_imports(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines()
            if _TOP_LEVEL_TORCHSIG.match(line)]


@pytest.fixture(scope="module")
def bundle_tools():
    return _load("torchsig_bundle", TRAINING / "torchsig_bundle.py")


@pytest.fixture(scope="module")
def ingester():
    return _load("ingest_torchsig", TRAINING / "ingest_torchsig.py")


@pytest.fixture(scope="module")
def builder():
    return _load("build_torchsig", TRAINING / "build_torchsig.py")


@pytest.fixture(scope="module")
def reader():
    return _load("detectors_dataset", TRAINING / "detectors" / "dataset.py")


# --------------------------------------------------------------------------- 构造工具


def _iq(signals=None, *, seed=11, samples=RECORD_SAMPLES, rate=RATE):
    """用项目生成器造一条真实 IQ（TorchSig 的波形对读端不重要，能测出能量即可）。"""
    if signals is None:
        signals = [{"mode": "qpsk", "offset": -100_000.0, "bandwidth": 25_000.0,
                    "power_dbfs": -10.0}]
    data, _summary = generate_iq(rate, samples / rate, signals,
                                 {"enabled": True, "snr_db": 18.0}, seed=seed)
    return np.asarray(data, dtype=np.complex64)


def _component(bundle_tools, *, class_name="qpsk", center_hz=-100_000.0, bandwidth_hz=25_000.0,
               start_in_samples=0, duration_in_samples=RECORD_SAMPLES, snr_db=18.0, rate=RATE,
               **extra):
    """写端（``build_torchsig._component_from_signal``）产出的实例条目。"""
    return bundle_tools.component(
        class_name=class_name,
        center_freq=center_hz,
        bandwidth=bandwidth_hz,
        lower_freq=center_hz - bandwidth_hz / 2.0,
        upper_freq=center_hz + bandwidth_hz / 2.0,
        start_in_samples=start_in_samples,
        stop_in_samples=start_in_samples + duration_in_samples,
        duration_in_samples=duration_in_samples,
        sample_rate=rate,
        snr_db=snr_db,
        **extra,
    )


def _write_bundle(bundle_tools, root, entries, *, rate=RATE):
    """``entries`` 是 ``(components, iq)`` 序列；返回 bundle 根目录。"""
    collected = []
    for entry in entries:
        components, iq = entry
        collected.append((bundle_tools.record(index=len(collected), iq=iq,
                                              components=components, sample_rate_hz=rate), iq))
    return bundle_tools.write_bundle(
        root, collected, sample_rate_hz=rate, torchsig_version=TORCHSIG_VERSION,
        metadata_overrides={"cochannel_overlap_probability": 0.0})


def _read_dataset(root):
    card = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (root / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return SimpleNamespace(card=card, records=records, root=Path(root))


def _ingest(ingester, bundle, output, *extra):
    code = ingester.main(["--bundle", str(bundle), "--output", str(output),
                          "--nfft", str(NFFT), "--image-size", str(IMAGE_SIZE),
                          "--train-fraction", "0.5", *extra])
    assert code == 0
    return _read_dataset(output)


# --------------------------------------------------------------------------- 夹具


@pytest.fixture(scope="module")
def primary(tmp_path_factory, bundle_tools, ingester):
    """一个真实 bundle：第 0 条两个信号（qpsk + fm），第 1 条一个（am）。"""
    bundle = tmp_path_factory.mktemp("ts_bundle")
    _write_bundle(bundle_tools, bundle, [
        ([_component(bundle_tools, class_name="qpsk", center_hz=-100_000.0,
                     bandwidth_hz=25_000.0, snr_db=18.0),
          _component(bundle_tools, class_name="fm", center_hz=120_000.0,
                     bandwidth_hz=40_000.0, snr_db=15.0)],
         _iq([{"mode": "qpsk", "offset": -100_000.0, "bandwidth": 25_000.0,
               "power_dbfs": -10.0},
              {"mode": "fm", "offset": 120_000.0, "bandwidth": 40_000.0,
               "power_dbfs": -12.0}], seed=11)),
        ([_component(bundle_tools, class_name="am", center_hz=200_000.0,
                     bandwidth_hz=30_000.0, snr_db=21.0)],
         _iq([{"mode": "am", "offset": 200_000.0, "bandwidth": 30_000.0,
               "power_dbfs": -10.0}], seed=12)),
    ])
    output = tmp_path_factory.mktemp("ts_dataset")
    dataset = _ingest(ingester, bundle, output)
    dataset.bundle = Path(bundle)
    return dataset


# --------------------------------------------------------------------------- 卡片契约


def test_card_matches_the_frozen_inference_contract(primary):
    """卡片必须与推理端契约逐字对齐，否则训练出的模型推理时会静默错位。"""
    card = primary.card
    contract = card["contract"]
    assert card["generator_script"] == "training/ingest_torchsig.py"
    assert contract["input_contract"] == INPUT_CONTRACT
    assert contract["layout"] == IMAGE_LAYOUT
    assert contract["output_layout"] == OUTPUT_LAYOUT
    assert contract["image_size"] == IMAGE_SIZE
    assert contract["channels"] == 1
    assert contract["spectrogram_nfft"] == NFFT
    assert contract["dynamic_range_db"] == 60.0
    assert contract["labels"] == ["emitter"]
    assert len(contract["box_columns"]) == BOX_COLUMNS
    assert contract["box_columns"][-2:] == ["confidence", "class"]
    assert contract["box_column_count"] == BOX_COLUMNS
    # 行 0 = +fs/2：TorchSig 的 center_freq 同样是"相对频率原点的有符号偏移"
    assert contract["frequency_reference"] == "baseband_offset"
    assert "行 0 = +fs/2" in contract["note"]


def test_label_semantics_is_locked_to_session(primary):
    """TorchSig 没有跳频族，语义只能是会话级；写错会让逐跳通路报"1 跳"。"""
    card = primary.card
    assert card["contract"]["label_semantics"] == "session_v1"
    assert card["statistics"]["labels"]["semantics"] == "session_v1"
    assert card["ingest"]["label_source"] == "component_band"
    for record in primary.records:
        assert record["hops"] == []
        assert record["source"] == "torchsig"
        for entry in record["truth"]:
            assert entry["hopping"] is False


def test_ingest_block_records_provenance(primary, bundle_tools):
    """溯源段必须能回答"这批数据从哪来、怎么标的、用什么口径算 SNR"。"""
    block = primary.card["ingest"]
    assert set(block) == {"source", "torchsig_version", "bundle", "bundle_version",
                          "metadata_overrides", "label_source", "snr_source",
                          "record_count", "accepted", "rejected", "rejected_total",
                          "image_source", "note"}
    assert block["source"] == "torchsig"
    assert block["torchsig_version"] == TORCHSIG_VERSION
    assert block["bundle_version"] == bundle_tools.BUNDLE_VERSION == "torchsig_bundle_v1"
    assert block["metadata_overrides"] == {"cochannel_overlap_probability": 0.0}
    # 图像与标签都由本项目同一份公式生成，训练/推理口径才不会分叉
    assert "本项目" in block["image_source"]
    assert "measure_band" in block["snr_source"]
    assert block["record_count"] == block["accepted"] == len(primary.records)
    assert block["rejected"] == {} and block["rejected_total"] == 0
    assert "yolo_label" in block["note"] or "同一份公式" in block["note"]


def test_scene_and_statistics_are_consistent(primary):
    card = primary.card
    scene = card["scene"]
    assert scene["sample_rate_hz"] == RATE
    assert scene["class_names"] == ["am", "fm", "qpsk"]
    assert scene["max_signals"] == 2
    assert scene["max_boxes"] == 32 and scene["min_box_px"] == 2.0
    assert scene["duration_range_s"] == [RECORD_SAMPLES / RATE] * 2
    statistics = card["statistics"]
    assert statistics["targets"] == sum(len(r["boxes"]) for r in primary.records) == 3
    assert statistics["labels"]["total"] == statistics["targets"]
    assert statistics["targets_per_sample"] == {"1": 1, "2": 1}
    source = statistics["sources"]["torchsig"]
    assert set(source) == {"records", "instances", "bandwidth_ratio",
                          "snr_inband_db", "snr_nominal_db", "collision_pairs"}
    assert source["records"] == card["ingest"]["accepted"]
    assert source["instances"] == statistics["targets"]
    assert source["bandwidth_ratio"]["count"] == 3
    assert source["snr_nominal_db"]["count"] == 3
    # 第 0 条记录的两个信号都覆盖整条记录、频率不同：会话级标签下这是时间上
    # 重叠但**合法**的（两个频段并存），不能计作碰撞
    assert source["collision_pairs"] == 0


def _instance_record(rate, boxes, *, bandwidth_hz=100_000.0, snr_inband_db=-3.0,
                     snr_nominal_db=None):
    entry = {"bandwidth_hz": bandwidth_hz, "snr_inband_db": snr_inband_db}
    if snr_nominal_db is not None:
        entry["snr_nominal_db"] = snr_nominal_db
    return {"scene": {"rate_hz": rate}, "boxes": [list(box) for box in boxes], "truth": [entry]}


def test_instance_statistics_uses_each_record_rate(reader):
    """``bandwidth_ratio`` 按**每个样本自己的采样率**算，不假设全局单一采样率。

    ``--rate-range`` 打开后项目生成器每个场景的采样率都不同，若用卡片里的
    标称值当分母，带宽比例会被整段算错，而这是混训前唯一用来判断"两个来源的
    分布有没有重叠"的量。
    """
    box = (0.5, 0.5, 1.0, 0.05, 1.0, 0.0)
    summary = reader.instance_statistics([
        _instance_record(RATE, [box]),
        _instance_record(2.0 * RATE, [box]),
    ])
    assert summary["records"] == 2 and summary["instances"] == 2
    assert summary["bandwidth_ratio"] == {"min": 0.05, "max": 0.1,
                                         "mean": 0.075, "count": 2}
    assert summary["snr_inband_db"] == {"min": -3.0, "max": -3.0, "mean": -3.0, "count": 2}
    # 生成器侧不写逐实例标称 SNR（它在 scene.noise.snr_db 上），此时不能
    # 留下一个 count=0 的键，否则跟 TorchSig 侧对比时会误判成"统计坏了"
    assert "snr_nominal_db" not in summary
    assert reader.instance_statistics([
        _instance_record(RATE, [box], snr_nominal_db=12.0)])["snr_nominal_db"] == {
        "min": 12.0, "max": 12.0, "mean": 12.0, "count": 1}


def test_instance_statistics_counts_only_real_collisions(reader):
    """碰撞对 = 两轴都真正重叠的框；合法的同频异时、以及只相接触的框都不算。"""
    same_band_same_time = [(0.5, 0.30, 0.20, 0.05, 1.0, 0.0),
                           (0.5, 0.32, 0.20, 0.05, 1.0, 0.0)]
    same_band_other_time = [(0.2, 0.30, 0.20, 0.05, 1.0, 0.0),
                            (0.8, 0.30, 0.20, 0.05, 1.0, 0.0)]
    touching = [(0.3, 0.30, 0.20, 0.05, 1.0, 0.0),
                (0.5, 0.30, 0.20, 0.05, 1.0, 0.0)]
    other_band = [(0.5, 0.10, 0.20, 0.05, 1.0, 0.0),
                  (0.5, 0.90, 0.20, 0.05, 1.0, 0.0)]
    assert reader.instance_statistics(
        [_instance_record(RATE, same_band_same_time)])["collision_pairs"] == 1
    for legal in (same_band_other_time, touching, other_band):
        assert reader.instance_statistics(
            [_instance_record(RATE, legal)])["collision_pairs"] == 0


def test_reader_accepts_the_dataset(primary, reader):
    """读取端（训练脚本共用）必须认这份卡片，语义也在读取时就校验。"""
    card, records = reader.load_dataset(primary.root)
    assert card["sample_count"] == len(records) == len(primary.records)
    assert reader.label_semantics(card) == "session_v1"
    assert reader.contract_of(card)["label_semantics"] == "session_v1"
    assert len(reader.split_records(records, "train")) == 1
    assert len(reader.split_records(records, "val")) == 1
    meta = reader.scene_meta(records[0], card)
    assert meta["size"] == IMAGE_SIZE
    assert meta["sample_rate_hz"] == RATE
    assert meta["t_start_s"] == 0.0 and meta["t_end_s"] == RECORD_SAMPLES / RATE


# --------------------------------------------------------------------------- 几何


def test_boxes_round_trip_through_band_to_box(primary, reader):
    """框必须能反解回真值频段/时间——这正是"标签由 band_to_box 生成"的可验证形式。"""
    for record in primary.records:
        meta = reader.scene_meta(record, primary.card)
        assert len(record["boxes"]) == len(record["truth"])
        for box, truth in zip(record["boxes"], record["truth"]):
            assert box[4] == 1.0 and box[5] == 0.0
            band = box_to_band(meta, *box[:4])
            assert band["f_low_hz"] == pytest.approx(truth["f_low_hz"], abs=2.0)
            assert band["f_high_hz"] == pytest.approx(truth["f_high_hz"], abs=2.0)
            assert band["t_start_s"] == pytest.approx(truth["t_start_s"], abs=1e-6)
            assert band["t_end_s"] == pytest.approx(truth["t_end_s"], abs=1e-6)
            assert truth["nominal_offset_hz"] == pytest.approx(
                0.5 * (truth["f_low_hz"] + truth["f_high_hz"]), abs=1e-6)
            assert truth["nominal_bandwidth_hz"] == pytest.approx(
                truth["f_high_hz"] - truth["f_low_hz"], abs=1e-6)


def test_labels_stay_inside_image_and_do_not_overlap(primary):
    """数据集不变量的逐字复刻（与 ``test_training_tools`` 同一判据与容差）。"""
    for record in primary.records:
        boxes = np.asarray(record["boxes"], dtype=np.float64).reshape(-1, 6)
        assert np.all(boxes[:, 0] - boxes[:, 2] / 2 >= -1e-9)
        assert np.all(boxes[:, 0] + boxes[:, 2] / 2 <= 1 + 1e-9)
        assert np.all(boxes[:, 1] - boxes[:, 3] / 2 >= -1e-9)
        assert np.all(boxes[:, 1] + boxes[:, 3] / 2 <= 1 + 1e-9)
        assert np.all(boxes[:, 2] > 0) and np.all(boxes[:, 3] > 0)
        spans = sorted((box[1] - box[3] / 2, box[1] + box[3] / 2) for box in boxes)
        for (_, previous_high), (next_low, _) in zip(spans, spans[1:]):
            assert next_low >= previous_high - 1e-6


def test_images_match_the_image_contract_and_are_reproducible(primary, ingester, tmp_path):
    """图像必须是 float32 单通道 [0,1]，且同一 bundle 两次转写逐字节相同。"""
    for record in primary.records:
        image = np.load(primary.root / record["image"])
        assert image.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert image.dtype == np.float32
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0

    again = tmp_path / "again"
    repeated = _ingest(ingester, primary.bundle, again)
    for before, after in zip(primary.records, repeated.records):
        assert before["image"] == after["image"]
        assert np.array_equal(np.load(primary.root / before["image"]),
                             np.load(again / after["image"]))
    # samples.jsonl 与卡片必须逐字节相同（唯一的非确定项是 created 时间戳）
    assert (primary.root / "samples.jsonl").read_bytes() == (again / "samples.jsonl").read_bytes()
    card, repeat = dict(primary.card), dict(repeated.card)
    card.pop("created"), repeat.pop("created")
    assert card == repeat


# --------------------------------------------------------------------------- 单位陷阱


def test_sample_indices_win_over_normalized_ratios(bundle_tools, ingester, tmp_path):
    """TorchSig 2.2 的 ``start``/``stop`` 是比例；读端必须优先用采样点字段。

    这里故意让两套字段互相矛盾（采样点 8192 + 16384 ≈ 0.0082 s／0.0246 s，
    比例写 0.25／0.75）：若读端照搬比例，信号会被压到记录头部，而框仍在图内。
    """
    bundle = tmp_path / "bundle"
    conflicted = _component(bundle_tools, start_in_samples=8192,
                            duration_in_samples=16384)
    conflicted.update({"start": 0.25, "stop": 0.75})
    _write_bundle(bundle_tools, bundle, [([conflicted], _iq())])
    dataset = _ingest(ingester, bundle, tmp_path / "dataset")
    truth = dataset.records[0]["truth"][0]
    assert truth["t_start_s"] == pytest.approx(8192 / RATE, abs=1e-6)
    assert truth["t_end_s"] == pytest.approx((8192 + 16384) / RATE, abs=1e-6)
    assert dataset.records[0]["boxes"][0][2] == pytest.approx(16384 / RECORD_SAMPLES, abs=1e-6)


def test_seconds_are_used_when_sample_fields_are_absent(bundle_tools, ingester, tmp_path):
    """退路：只有秒制 ``start``/``stop`` 时按秒解释（bundle 格式即秒制）。"""
    bundle = tmp_path / "bundle"
    without_samples = {"class_name": "qpsk", "center_freq": -100_000.0, "bandwidth": 25_000.0,
                       "lower_freq": -112_500.0, "upper_freq": -87_500.0,
                       "start": 0.005, "stop": 0.015, "snr_db": 18.0}
    _write_bundle(bundle_tools, bundle, [([without_samples], _iq())])
    dataset = _ingest(ingester, bundle, tmp_path / "dataset")
    truth = dataset.records[0]["truth"][0]
    assert truth["t_start_s"] == pytest.approx(0.005, abs=1e-6)
    assert truth["t_end_s"] == pytest.approx(0.015, abs=1e-6)


def test_missing_time_fields_span_the_whole_record(bundle_tools, ingester, tmp_path):
    """时间信息全缺时按整条记录处理（覆盖式生成器的常见形态）。"""
    bundle = tmp_path / "bundle"
    full = {"class_name": "qpsk", "center_freq": -100_000.0, "bandwidth": 25_000.0}
    _write_bundle(bundle_tools, bundle, [([full], _iq())])
    dataset = _ingest(ingester, bundle, tmp_path / "dataset")
    truth = dataset.records[0]["truth"][0]
    assert truth["t_start_s"] == 0.0
    assert truth["t_end_s"] == pytest.approx(RECORD_SAMPLES / RATE, abs=1e-6)
    assert dataset.records[0]["boxes"][0][2] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- SNR 口径


def test_inband_snr_is_remeasured_with_the_project_convention(primary):
    """``snr_inband_db`` 必须是 ``measure_band`` 的口径，标称值只作溯源。"""
    contract = primary.card["contract"]
    for record in primary.records:
        iq = np.load(primary.bundle / record["scene"]["bundle"]).astype(np.complex64)
        summary, arrays = spectral_context(iq, record["scene"]["rate_hz"],
                                           {"nfft": contract["spectrogram_nfft"]})
        for truth in record["truth"]:
            measured = measure_band(arrays, summary, truth)
            assert truth["snr_inband_db"] == pytest.approx(float(measured["snr_db"]), abs=1e-6)
            assert truth["power_dbfs"] == pytest.approx(float(measured["power_dbfs"]), abs=1e-6)
            # 注入标称值原样留存，但绝不当作项目口径使用
            assert truth["snr_nominal_db"] == pytest.approx(_nominal_of(truth["mode"]), abs=1e-6)


def _nominal_of(mode):
    return {"qpsk": 18.0, "fm": 15.0, "am": 21.0}[mode]


# --------------------------------------------------------------------------- 守卫


def _bad_bundle(bundle_tools, root, components, *, iq=None):
    return _write_bundle(bundle_tools, root, [(components, iq if iq is not None else _iq())])


def test_hop_labels_are_refused(ingester, primary, tmp_path):
    """TorchSig 没有 fh* 族；``--labels hop`` 必须报错而不是退化成会话级。"""
    with pytest.raises(SystemExit) as error:
        ingester.main(["--bundle", str(primary.bundle), "--output", str(tmp_path / "hop"),
                       "--labels", "hop"])
    message = str(error.value)
    assert "per_hop_v1" in message and "build_dataset.py" in message
    assert "fh_rc" in message


def test_overlapping_frequency_boxes_are_refused(bundle_tools, ingester, tmp_path):
    """同频不同时（TorchSig 允许的叠加）也必须拒绝：网格化标签会静默丢框。"""
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle", [
        _component(bundle_tools, center_hz=-100_000.0, bandwidth_hz=40_000.0,
                   start_in_samples=0, duration_in_samples=8192),
        _component(bundle_tools, class_name="fm", center_hz=-100_000.0, bandwidth_hz=40_000.0,
                   start_in_samples=16384, duration_in_samples=8192),
    ])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "dataset")
    message = str(error.value)
    assert "overlap_frequency" in message
    assert "但未" in message          # 频率重叠、时间不重叠——正是最隐蔽的那种
    assert "假阴性" in message


def test_sub_pixel_boxes_are_refused(bundle_tools, ingester, tmp_path):
    """比网格还细的框在训练时会被丢掉，必须在构建时就失败。"""
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle",
                         [_component(bundle_tools, bandwidth_hz=100.0)])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "dataset")
    assert "sub_pixel" in str(error.value)


def test_missing_geometry_is_refused(bundle_tools, ingester, tmp_path):
    """既没有 lower/upper_freq 也没有 center_freq+bandwidth → miss 几何信息。

    写端的 :meth:`component` 会挡住这种情况，所以这里直接构造裸条目（bundle 的
    元数据是外部生成的，读端不能假定它一定合法）。
    """
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle",
                         [{"class_name": "qpsk", "snr_db": 12.0}])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "dataset")
    assert "missing_geometry" in str(error.value)


def test_out_of_band_and_degenerate_bands_are_refused(bundle_tools, ingester, tmp_path):
    """完全落在奈奎斯特之外、以及带宽归零的实例都要拒绝。"""
    outside = _bad_bundle(bundle_tools, tmp_path / "outside",
                          [_component(bundle_tools, center_hz=4.0e6)])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, outside, tmp_path / "ds_outside")
    assert "out_of_band" in str(error.value)

    degenerate = {"class_name": "ook", "lower_freq": 0.0, "upper_freq": 0.0}
    zero = _bad_bundle(bundle_tools, tmp_path / "zero", [degenerate])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, zero, tmp_path / "ds_zero")
    assert "too_narrow" in str(error.value)


def test_too_many_boxes_is_refused(bundle_tools, ingester, tmp_path):
    """框数超限即拒绝（训练脚本 ``--max-boxes`` 之外的实例会被丢掉）。"""
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle", [
        _component(bundle_tools, center_hz=-100_000.0, bandwidth_hz=25_000.0),
        _component(bundle_tools, class_name="fm", center_hz=200_000.0, bandwidth_hz=25_000.0),
    ])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "dataset", "--max-boxes", "1")
    assert "too_many_boxes" in str(error.value)


def test_malformed_component_is_refused(bundle_tools, ingester, tmp_path):
    """``components`` 里混进非字典条目时按格式错误拒绝，而不是崩在别处。

    写端的 ``record()`` 会 ``dict(item)``，构造不出这种条目——但 bundle 的元数据
    是外部生成的，读端不能假定它一定合法，所以这里直接改 ``meta/*.json``。
    """
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle", [_component(bundle_tools)])
    meta_path = bundle / "meta" / "000000.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["components"] = [payload["components"][0], "qpsk"]
    meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "dataset")
    assert "malformed_component" in str(error.value)


def test_skip_rejected_keeps_running_and_counts(bundle_tools, ingester, tmp_path):
    """``--skip-rejected`` 只把坏记录计数跳过；保留的记录序号仍是 bundle 序号。"""
    bundle = _write_bundle(bundle_tools, tmp_path / "bundle", [
        ([_component(bundle_tools, bandwidth_hz=100.0)], _iq()),          # 亚像素 → 拒绝
        ([_component(bundle_tools, center_hz=200_000.0, bandwidth_hz=30_000.0)], _iq(seed=12)),
    ])
    with pytest.raises(SystemExit) as error:
        _ingest(ingester, bundle, tmp_path / "strict")
    assert "第 0 条记录" in str(error.value)

    dataset = _ingest(ingester, bundle, tmp_path / "lenient", "--skip-rejected")
    block = dataset.card["ingest"]
    assert block["record_count"] == 2
    assert block["accepted"] == 1
    assert block["rejected"] == {"sub_pixel": 1}
    assert block["rejected_total"] == 1
    # 序号 = bundle 内的位置（被跳过的记录不重新编号），便于回溯到 iq/000001.npy
    assert dataset.records[0]["index"] == 1
    assert dataset.records[0]["image"] == "images/000001.npy"
    assert dataset.card["statistics"]["sources"]["torchsig"]["records"] == 1
    assert dataset.card["scene"]["class_names"] == ["qpsk"]


def test_ingest_rejects_invalid_command_line(ingester, primary, tmp_path):
    for extra in (["--max-boxes", "0"], ["--train-fraction", "0"], ["--train-fraction", "1.5"]):
        with pytest.raises(SystemExit):
            ingester.main(["--bundle", str(primary.bundle), "--output", str(tmp_path / "bad"),
                           *extra])


# --------------------------------------------------------------------------- 不依赖 torchsig


def test_reading_side_never_imports_torchsig(bundle_tools, ingester, builder):
    """读端与格式定义都只能依赖 NumPy；torchsig 只在生成脚本里、且是惰性导入。"""
    assert "torchsig" not in sys.modules
    for name in ("torchsig_bundle.py", "ingest_torchsig.py"):
        found = _top_level_torchsig_imports(TRAINING / name)
        assert found == [], f"{name} 在模块级导入了 torchsig：{found}"


def test_build_script_only_imports_torchsig_lazily(builder):
    """``build_torchsig.py`` 允许用 torchsig，但必须在函数内导入（否则测试环境直接崩）。"""
    found = _top_level_torchsig_imports(TRAINING / "build_torchsig.py")
    assert found == [], f"build_torchsig.py 在模块级导入了 torchsig：{found}"


def test_fail_fast_when_torchsig_is_missing(builder):
    """未安装时给出可执行的安装命令与前置条件，而不是 ImportError 堆栈。"""
    if importlib.util.find_spec("torchsig") is not None:  # pragma: no cover - 取决于本机
        pytest.skip("本机已安装 torchsig，跳过缺失路径")
    with pytest.raises(SystemExit) as error:
        builder._import_torchsig()
    message = str(error.value)
    assert "pip install torchsig" in message
    assert "ingest_torchsig.py" in message
    for requirement in builder.PREREQUISITES:
        assert requirement in message


def test_bundle_writer_rejects_unknown_component_fields(bundle_tools, tmp_path):
    """bundle 的字段表是白名单：``yolo_label`` 这类"另一套坐标约定"绝不能带进来。"""
    with pytest.raises(ValueError) as error:
        bundle_tools.component(class_name="qpsk", center_freq=0.0, bandwidth=1.0,
                               yolo_label=[0.5, 0.5, 0.1, 0.1])
    assert "yolo_label" in str(error.value)
    assert "center_freq" in str(error.value)
    with pytest.raises(ValueError):
        bundle_tools.component(class_name="qpsk")  # 没有任何几何信息


def test_bundle_loader_rejects_foreign_formats(bundle_tools, tmp_path):
    with pytest.raises(SystemExit) as error:
        bundle_tools.load_bundle(tmp_path)
    assert "manifest.json" in str(error.value)

    root = tmp_path / "bundle"
    _write_bundle(bundle_tools, root, [([_component(bundle_tools)], _iq())])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["bundle_version"] = "torchsig_bundle_v2"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        bundle_tools.load_bundle(root)
    assert "torchsig_bundle_v2" in str(error.value)


# --------------------------------------------------------------------------- 生成侧纯函数
#
# ``build_torchsig.py`` 只能在本机装 torchsig 时整体运行，但它与上游 API 打交道的
# 那几个纯函数（字段搬运、单位换算、元数据覆盖）可以在没有 torchsig 的环境里用替身
# 覆盖——单位换算恰恰是本次接入最容易错的地方。


class _FakeSignal:
    """``torchsig`` 的 ``Signal`` 替身：属性优先，``metadata`` 字典作后备。

    TorchSig 把 ``lower_freq`` / ``upper_freq`` / ``stop_in_samples`` 实现成**派生
    属性**（metadata 字典里没有），所以取字段必须走属性；这里复刻这一行为。
    """

    def __init__(self, attributes=None, metadata=None):
        self.attributes = dict(attributes or {})
        self.metadata = dict(metadata or {})

    def __getattr__(self, name):
        try:
            return self.attributes[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def test_read_field_prefers_attributes_and_drops_unusable_values(builder):
    derived = _FakeSignal({"lower_freq": -120_000.0})
    assert builder._read_field(derived, "lower_freq") == -120_000.0
    assert builder._read_field(derived, "upper_freq") is None

    from_metadata = _FakeSignal(metadata={"snr_db": 12.5})
    assert builder._read_field(from_metadata, "snr_db") == 12.5

    assert builder._read_field(_FakeSignal({"snr_db": float("nan")}), "snr_db") is None
    assert builder._read_field(_FakeSignal({"snr_db": True}), "snr_db") is None
    assert builder._read_field(_FakeSignal({"class_name": "  qpsk "}), "class_name") == "qpsk"
    assert builder._read_field(_FakeSignal({"class_name": "   "}), "class_name") is None


def test_component_from_signal_emits_samples_and_seconds(builder, bundle_tools):
    """写端必须同时写出采样点与秒制时间，采样点字段是读端的第一优先。"""
    signal = _FakeSignal({"class_name": "qpsk", "center_freq": -100_000.0,
                          "bandwidth": 40_000.0, "lower_freq": -120_000.0,
                          "upper_freq": -80_000.0, "start_in_samples": 1000,
                          "duration_in_samples": 4000, "stop_in_samples": 5000,
                          "snr_db": 12.5, "class_index": 3.0, "oversampling_rate": 1.0})
    entry = builder._component_from_signal(signal, RATE, 0, 0)
    assert entry["class_name"] == "qpsk"
    assert entry["start_in_samples"] == 1000 and isinstance(entry["start_in_samples"], int)
    assert entry["duration_in_samples"] == 4000
    assert entry["class_index"] == 3
    assert entry["start"] == pytest.approx(0.001)
    assert entry["stop"] == pytest.approx(0.005)
    assert entry["sample_rate"] == RATE
    assert set(entry) <= set(bundle_tools.COMPONENT_FIELDS)


def test_component_from_signal_requires_sample_fields(builder):
    """缺 ``start_in_samples``／``duration_in_samples`` 时定位到具体实例并早失败。"""
    signal = _FakeSignal({"class_name": "ook", "center_freq": 0.0, "bandwidth": 1000.0},
                         metadata={"class_name": "ook", "snr_db": 10.0})
    with pytest.raises(SystemExit) as error:
        builder._component_from_signal(signal, RATE, 2, 1)
    message = str(error.value)
    assert "第 2 条记录的第 1 个信号实例" in message
    assert "start_in_samples" in message
    assert "snr_db" in message  # 列出可用字段，便于定位上游行为变更


def test_builder_components_feed_the_reader_with_consistent_units(bundle_tools, ingester,
                                                                  tmp_path):
    """生成侧 → 读端的单位闭环：秒制时间必须落在采样点对应的位置上。"""
    builder_entry = {"class_name": "qpsk", "center_freq": -100_000.0, "bandwidth": 25_000.0,
                     "lower_freq": -112_500.0, "upper_freq": -87_500.0,
                     "start_in_samples": 8192, "stop_in_samples": 24576,
                     "duration_in_samples": 16384, "sample_rate": RATE, "snr_db": 18.0,
                     # 故意混入归一化比例的字段名（TorchSig 2.2 的原生含义）
                     "start": 8192 / RATE, "stop": 24576 / RATE}
    bundle = _bad_bundle(bundle_tools, tmp_path / "bundle", [builder_entry])
    dataset = _ingest(ingester, bundle, tmp_path / "dataset")
    truth = dataset.records[0]["truth"][0]
    assert truth["t_start_s"] == pytest.approx(8192 / RATE, abs=1e-6)
    assert truth["t_end_s"] == pytest.approx(24576 / RATE, abs=1e-6)
    assert dataset.records[0]["boxes"][0][2] == pytest.approx(0.5, abs=1e-6)


def test_pair_parsers_validate_ranges(builder):
    assert builder._pair("-5,30") == (-5.0, 30.0)
    assert builder._pair_int("0,3") == (0, 3)
    for text in ("1", "a,2", "30,-5", "nan,1", "0,inf"):
        with pytest.raises(argparse.ArgumentTypeError):
            builder._pair(text)


def test_apply_overrides_supports_both_upstream_shapes(builder):
    """TorchSig 2.2 的 dataset metadata 是 dict；保留 model_copy／setattr 两条退路。"""
    plain = {"sample_rate": 10_000_000}
    assert builder._apply_overrides(plain, {"sample_rate": RATE, "extra": None}) is plain
    assert plain == {"sample_rate": RATE}  # None 不写入

    class Pydanticish:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_copy(self, update=None):
            clone = Pydanticish(**self.__dict__)
            clone.__dict__.update(update or {})
            return clone

    original = Pydanticish(sample_rate=10_000_000)
    updated = builder._apply_overrides(original, {"sample_rate": RATE})
    assert updated is not original and updated.sample_rate == RATE
    assert original.sample_rate == 10_000_000

    class Bare:
        pass

    bare = builder._apply_overrides(Bare(), {"sample_rate": RATE})
    assert bare.sample_rate == RATE


def test_summarize_reports_the_distribution(builder, bundle_tools):
    """生成后的分布摘要（类名分布、带宽占比、标称 SNR）——用来先看数对不对。"""
    iq = _iq()
    collected = [
        (bundle_tools.record(index=0, iq=iq, sample_rate_hz=RATE, components=[
            {"class_name": "qpsk", "bandwidth": 20_000.0, "snr_db": 10.0},
            {"class_name": "fm", "bandwidth": 50_000.0, "snr_db": 20.0},
        ]), iq),
        (bundle_tools.record(index=1, iq=iq, sample_rate_hz=RATE, components=[
            {"class_name": "qpsk", "bandwidth": 30_000.0},          # 无标称 SNR
        ]), iq),
    ]
    summary = builder._summarize(collected)
    assert summary["records"] == 2
    assert summary["instances"] == 3
    assert summary["classes"] == {"fm": 1, "qpsk": 2}
    assert summary["bandwidth_ratio"] == {"min": 0.02, "max": 0.05, "mean": 0.033333}
    assert summary["snr_nominal_db"] == {"min": 10.0, "max": 20.0, "mean": 15.0}


def test_summarize_handles_noise_only_records(builder):
    assert builder._summarize([])["instances"] == 0
    assert builder._summarize([])["classes"] == {}


def test_parser_defaults_match_the_documented_distribution(builder):
    """CLI 默认值就是"向项目分布靠拢"的那组参数，改动等于改变数据集口径。"""
    args = builder._parse_args(["--output", "/tmp/x"])
    assert (args.count, args.seed, args.sample_rate) == (256, 7, RATE)
    assert (args.num_iq_samples, args.nfft, args.fft_stride) == (262144, NFFT, 0)
    assert args.signals_range == (0, 3)
    assert args.snr_range == (-5.0, 30.0)
    assert args.bandwidth_ratio_range == (0.02, 0.12)
    assert args.duration_ratio_range == (0.8, 1.0)
    assert args.frequency_span_ratio == 0.8
    assert args.cochannel_overlap == 0.0      # 本项目要求标签框互不重叠
    assert args.impairment_level is None      # 默认不加扰动：IQ 与元数据严格对应
