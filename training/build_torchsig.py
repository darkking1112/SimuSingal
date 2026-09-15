#!/usr/bin/env python3
"""用 TorchSig 生成本项目的补充 IQ 数据，落成 ``torchsig_bundle_v1``。

这是全仓库**唯一** import ``torchsig`` 的地方
--------------------------------------------

TorchSig 已包含在项目的 ``.[train]`` 依赖中，也可单独安装 ``.[torchsig]``。
生成规模决定资源用量，小样本验证不需要 TB 级磁盘。项目仍把生成和导入分开：

* 本脚本：唯一依赖 torchsig 的地方，负责生成 IQ 与信号实例元数据，写出
  ``torchsig_bundle_v1``（格式定义见 ``training/torchsig_bundle.py``）；
* ``training/ingest_torchsig.py``：只依赖 NumPy，把 bundle 转成本项目的
  ``tf_image_v1`` 数据集，**没有 torchsig 也能跑，并被单元测试覆盖**。

导入端测试不依赖 TorchSig；真实生成集成测试在安装后执行。缺失时给出安装提示，
不会悄悄生成空数据集。

为什么要覆盖 TorchSig 的默认元数据
----------------------------------

TorchSig 的默认值是为"宽带多信号"设计的，与本项目的场景分布差得很远：

===================  ======================  =================  ================
键                    TorchSig 默认            本脚本默认          理由
===================  ======================  =================  ================
``sample_rate``      10 MHz                  1 MHz              与
                                                                 ``build_dataset
                                                                 .py --rate``
                                                                 一致
``num_signals_min``  1 / 1                   0 / 3              与 ``--max-signals
``/``_max``                                                      `` 一致；0 提供纯
                                                                 噪声样本
``cochannel_overlap 0.2                     0.0                数据集不变量要求标
_probability``                                                   签框在频率轴互不
                                                                 重叠
``snr_db_min``/      0 / 50 dB               −5 / 30 dB         与 ``--snr-range``
``_max``                                                         一致
``bandwidth_min``/   2.5M / 3.33M（≈33%）    2% / 12% × Fs      与 ``--min/max-
``_max``                                                         bandwidth-ratio``
                                                                 一致
===================  ======================  =================  ================

不消费 ``yolo_label``
---------------------

TorchSig 的 ``target_labels`` 支持 ``yolo_label``，那是**另一套 y 轴约定**。本项目
只用它的 IQ 与信号实例元数据（中心频率/带宽/起止点/标称信噪比/类名），标签一律由
``ingest_torchsig.py`` 用 :func:`band_to_box` 重新生成。因此这里 ``target_labels=
None``（只取 ``Signal`` 对象），绝不请求 ``yolo_label``。

时间轴的单位差（一处容易踩的坑）
--------------------------------

TorchSig 2.2 里 ``SignalMetadataObject.start`` / ``stop`` / ``duration`` 是**占记录
长度的比例（0–1）**，不是秒。本项目 bundle 里 ``start``/``stop`` 统一是**秒**。为
避免同名字段不同单位，本脚本：

* 始终写出无歧义的采样点字段 ``start_in_samples`` / ``stop_in_samples`` /
  ``duration_in_samples``（``ingest_torchsig.py`` 优先使用它们）；
* 另存换算后的秒制 ``start``/``stop`` 供人读与排错。

产物::

    <bundle>/manifest.json    # 采样率、torchsig 版本、元数据覆盖、记录清单
    <bundle>/iq/000000.npy    # complex64 一维 IQ
    <bundle>/meta/000000.json # 信号实例元数据（components[]）

用法::

    # 1) 生成（需要 torchsig；慢，先小批量试跑）
    .venv/bin/python training/build_torchsig.py --output /data/ts_bundle --count 256

    # 2) 转成本项目数据集（不需要 torchsig，可在任意环境跑）
    .venv/bin/python training/ingest_torchsig.py --bundle /data/ts_bundle \\
        --output training/data/torchsig --nfft 512 --image-size 1024
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from torchsig_bundle import component, record, write_bundle  # noqa: E402

#: 本项目真实生成的验证平台；磁盘需求由 count × num_iq_samples 决定。
PREREQUISITES = ("Linux", "Python ≥ 3.10", "磁盘空间按生成规模准备")

#: 从 ``SignalMetadataObject`` 上平移过来的字段（bundle 里同名同义）
COMPONENT_ATTRS = ("center_freq", "bandwidth", "lower_freq", "upper_freq",
                   "start_in_samples", "stop_in_samples", "duration_in_samples",
                   "snr_db", "oversampling_rate", "class_index")

#: 需要写成整数的字段（其余按浮点写）
INTEGER_FIELDS = ("start_in_samples", "stop_in_samples", "duration_in_samples",
                  "class_index")


def _import_torchsig():
    """导入 torchsig；缺失时给出安装命令与前置条件（fail fast，不静默降级）。"""
    try:
        import torchsig
    except ImportError as exc:  # pragma: no cover - 取决于本地环境
        requirements = "、".join(PREREQUISITES)
        raise SystemExit(
            "生成 TorchSig 数据需要 torchsig，请先安装：\n"
            "  .venv/bin/python -m pip install -e '.[train]'\n"
            "或只安装数据源：pip install torchsig==2.2.0\n"
            f"本项目验证环境：{requirements}。\n"
            "本仓库的测试不依赖它——读端 training/ingest_torchsig.py 只需要 "
            "NumPy，可在没有 torchsig 的环境里把已生成的 bundle 转成数据集。") from exc
    return torchsig


def _pair(text, cast=float):
    """``"最小值,最大值"`` → 二元组（区间必须有限且有序）。"""
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("请给出形如 最小值,最大值 的区间")
    try:
        low, high = cast(parts[0]), cast(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"区间元素无法解析：{text}") from exc
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise argparse.ArgumentTypeError("区间必须有限且满足 最小值 ≤ 最大值")
    return low, high


def _pair_int(text):
    return _pair(text, cast=int)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="用 TorchSig 生成 IQ 补充数据（torchsig_bundle_v1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", required=True, help="bundle 输出目录")
    parser.add_argument("--count", type=int, default=256, help="生成的记录数")
    parser.add_argument("--seed", type=int, default=7, help="随机种子（决定场景与元数据）")
    parser.add_argument("--sample-rate", type=float, default=1_000_000.0,
                        help="采样率（Hz），需与 ingest 时的场景一致")
    parser.add_argument("--num-iq-samples", type=int, default=262144,
                        help="单条记录的 IQ 长度（采样点）")
    parser.add_argument("--nfft", type=int, default=512,
                        help="TorchSig 内部 FFT 点数（与 ingest --nfft 保持一致）")
    parser.add_argument("--fft-stride", type=int, default=0,
                        help="TorchSig 内部 FFT 步长（0 = 与 --nfft 相同）")
    parser.add_argument("--signals-range", type=_pair_int, default=(0, 3),
                        help="单条记录信号数区间（0 表示包含纯噪声样本）")
    parser.add_argument("--snr-range", type=_pair, default=(-5.0, 30.0),
                        help="标称信噪比区间（dB；TorchSig 的 snr_db 是注入标称值）")
    parser.add_argument("--bandwidth-ratio-range", type=_pair, default=(0.02, 0.12),
                        help="占用带宽区间（占采样率的比例，与 build_dataset 一致）")
    parser.add_argument("--duration-ratio-range", type=_pair, default=(0.8, 1.0),
                        help="信号时长区间（占单条记录长度的比例）")
    parser.add_argument("--frequency-span-ratio", type=float, default=0.8,
                        help="信号中心频率的可放范围（占 ±Fs/2 的比例，留出混叠余量）")
    parser.add_argument("--cochannel-overlap", type=float, default=0.0,
                        help="同频叠加概率；本项目要求标签框互不重叠，默认 0")
    parser.add_argument("--signal-generators", default="all",
                        help="信号族：all 或逗号分隔的类名（如 ook,bpsk,qpsk）")
    parser.add_argument("--impairment-level", type=int, choices=(0, 1, 2), default=None,
                        help="TorchSig 扰动档位：0=理想 1=有线 2=无线；"
                             "不指定则不加任何变换（IQ 与元数据严格对应）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印生效后的元数据与首条样本的实例摘要，不写盘")
    parser.add_argument("--overwrite", action="store_true",
                        help="输出目录已有 manifest.json 时允许覆盖")
    return parser.parse_args(argv)


def _read_field(signal, name):
    """从 TorchSig 的 ``Signal`` 上取一个字段：属性优先，退回 metadata 字典。

    ``SignalMetadataObject`` 把 ``lower_freq`` / ``upper_freq`` /
    ``stop_in_samples`` 实现成**派生属性**（由 ``center_freq`` / ``bandwidth`` /
    ``start_in_samples`` 算出），metadata 字典里只有存下来的那几个键。因此属性
    访问才是"全量"的；取不到时返回 ``None``，由调用方决定是拒绝还是跳过。
    """
    try:
        value = getattr(signal, name)
    except Exception:  # noqa: BLE001 - 上游用 __getattr__ 抛 AttributeError/KeyError
        value = None
    if value is None:
        metadata = getattr(signal, "metadata", None)
        value = metadata.get(name) if isinstance(metadata, dict) else None
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    return text or None


def _component_from_signal(signal, rate, position, order):
    """一个 TorchSig 信号实例 → bundle 的 components[] 条目。

    元数据不完整时直接 ``SystemExit``：留在 bundle 里也只会被
    ``ingest_torchsig.py`` 以 ``missing_geometry`` 拒绝，早失败能定位到生成侧。
    """
    fields = {}
    for name in COMPONENT_ATTRS:
        value = _read_field(signal, name)
        if value is None:
            continue
        fields[name] = int(round(value)) if name in INTEGER_FIELDS else float(value)
    start = fields.get("start_in_samples")
    duration = fields.get("duration_in_samples")
    if start is None or duration is None:
        raise SystemExit(
            f"第 {position} 条记录的第 {order} 个信号实例缺少 start_in_samples / "
            f"duration_in_samples（TorchSig 的 _insert_component_signal 会写这两个"
            f"字段，缺失说明上游行为变了）；现有字段："
            f"{sorted(getattr(signal, 'metadata', {}) or {})}")
    # 秒制时间随采样点字段一起写出：读端优先用采样点，秒制只便于人读与排错
    fields["start"] = start / rate
    fields["stop"] = (start + duration) / rate
    fields["sample_rate"] = float(rate)
    class_name = _read_field(signal, "class_name") or "unknown"
    try:
        return component(class_name=class_name, **fields)
    except ValueError as exc:
        raise SystemExit(
            f"第 {position} 条记录的第 {order} 个信号实例元数据不完整：{exc}") from exc


def _apply_overrides(metadata, overrides):
    """把覆盖写进 dataset metadata；兼容 dict 与 pydantic 模型两种上游写法。

    TorchSig 2.2 的 ``TorchSigDefaults().default_dataset_metadata`` 返回普通 dict
    （``.update`` 即可）。这里保留 ``model_copy`` 分支是为了上游改成 pydantic 模型
    时不再炸，而不是为了"猜"——三条件都失败就直接失败。
    """
    applied = {key: value for key, value in overrides.items() if value is not None}
    if hasattr(metadata, "model_copy"):
        try:
            return metadata.model_copy(update=applied)
        except Exception:  # noqa: BLE001 - 上游实现差异，继续尝试下一种
            pass
    if hasattr(metadata, "update"):
        try:
            metadata.update(applied)
            return metadata
        except Exception:  # noqa: BLE001 - 同上
            pass
    for key, value in applied.items():
        setattr(metadata, key, value)
    return metadata


def _dataset_metadata(args):
    """TorchSig 默认元数据 + 本项目分布覆盖；返回 ``(metadata, overrides)``。"""
    from torchsig.utils.defaults import TorchSigDefaults

    rate = float(args.sample_rate)
    samples = int(args.num_iq_samples)
    span = rate / 2.0 * float(args.frequency_span_ratio)
    stride = int(args.fft_stride) or int(args.nfft)
    signals_min, signals_max = (int(value) for value in args.signals_range)
    bandwidth_min, bandwidth_max = (float(value) for value in args.bandwidth_ratio_range)
    duration_min, duration_max = (float(value) for value in args.duration_ratio_range)
    snr_min, snr_max = (float(value) for value in args.snr_range)
    overrides = {
        "sample_rate": rate,
        "num_iq_samples_dataset": samples,
        "fft_size": int(args.nfft),
        "fft_stride": stride,
        "num_signals_min": signals_min,
        "num_signals_max": signals_max,
        # 数据集不变量：标签框在频率轴上互不重叠 → 不允许同频叠加
        "cochannel_overlap_probability": float(args.cochannel_overlap),
        "snr_db_min": snr_min,
        "snr_db_max": snr_max,
        "bandwidth_min": bandwidth_min * rate,
        "bandwidth_max": bandwidth_max * rate,
        "signal_center_freq_min": -span,
        "signal_center_freq_max": span - 1.0,
        "frequency_min": -span,
        "frequency_max": span - 1.0,
        "signal_duration_in_samples_min": int(duration_min * samples),
        "signal_duration_in_samples_max": int(duration_max * samples),
    }
    return _apply_overrides(TorchSigDefaults().default_dataset_metadata, overrides), \
        overrides


def _make_dataset(args, metadata):
    """构造一维迭代数据集；返回 ``(dataset, component_transforms, description)``。

    ``--impairment-level`` 走 TorchSig 自己的 :class:`Impairments`：它的变换会同步
    维护 ``center_freq`` / ``start_in_samples`` 等元数据（如 ``ChannelSwap`` 会把
    ``center_freq`` 取反），所以本脚本从返回的元数据推标签仍然自洽。
    """
    from torchsig.datasets.datasets import TorchSigIterableDataset
    from torchsig.transforms.impairments import Impairments

    component_transforms = []
    transforms = []
    if args.impairment_level is not None:
        impairments = Impairments(int(args.impairment_level))
        component_transforms = [impairments.signal_transforms]
        transforms = [impairments.dataset_transforms]
    generators = args.signal_generators
    if str(generators).strip().lower() != "all":
        generators = [name.strip() for name in str(generators).split(",") if name.strip()]
        if not generators:
            raise SystemExit("--signal-generators 为空；请给出 all 或类名列表")
    dataset = TorchSigIterableDataset(
        metadata=metadata,
        signal_generators=generators,
        transforms=transforms,
        component_transforms=component_transforms,
        # 关键：只取 Signal 对象（IQ + 元数据），绝不请求 yolo_label
        target_labels=None,
        seed=int(args.seed),
    )
    description = f"generators={generators} impairment={args.impairment_level}"
    return dataset, component_transforms, description


def _class_names(dataset):
    """数据集已注册的信号类名（TorchSig 在 ``init_signal_generator`` 时写入）。"""
    try:
        names = list(dataset.class_names)
    except Exception:  # noqa: BLE001 - 取不到就不写进溯源信息，不影响生成
        return []
    return [str(name) for name in names]


def _collect(dataset, args, rate):
    """迭代数据集，产出 ``write_bundle`` 需要的 ``(meta, iq)`` 序列。"""
    count = int(args.count)
    if count <= 0:
        raise SystemExit("--count 必须为正整数")
    samples = int(args.num_iq_samples)
    step = max(1, count // 10)
    iterator = iter(dataset)
    collected = []
    for index in range(count):
        sample = next(iterator)
        iq = np.asarray(getattr(sample, "data", None))
        if iq.ndim != 1 or iq.size < 2:
            raise SystemExit(
                f"第 {index} 条记录的 IQ 形状为 {iq.shape}，应为单通道一维序列")
        if iq.size != samples:
            raise SystemExit(
                f"第 {index} 条记录的 IQ 长度为 {iq.size}，与 --num-iq-samples "
                f"{samples} 不一致（TorchSig 的元数据覆盖没有生效？）")
        components = [_component_from_signal(signal, rate, index, order)
                      for order, signal in enumerate(sample.component_signals)]
        meta = record(index=index, iq=iq, components=components,
                      duration_s=float(iq.size) / rate, sample_rate_hz=rate)
        collected.append((meta, iq))
        if (index + 1) % step == 0 or index + 1 == count:
            print(f"  已生成 {index + 1}/{count} 条记录", flush=True)
    return collected


def _summarize(collected):
    """实例数、类名分布、带宽占比与标称信噪比——生成后一眼看出分布对不对。"""
    instances = 0
    by_class = {}
    bandwidth_ratios = []
    snrs = []
    for meta, iq in collected:
        duration = len(iq) / float(meta["sample_rate_hz"])
        for entry in meta["components"]:
            instances += 1
            name = str(entry.get("class_name") or "unknown")
            by_class[name] = by_class.get(name, 0) + 1
            bandwidth = entry.get("bandwidth")
            if bandwidth is not None and duration > 0:
                bandwidth_ratios.append(float(bandwidth) / float(meta["sample_rate_hz"]))
            if entry.get("snr_db") is not None:
                snrs.append(float(entry["snr_db"]))

    def _triple(values):
        if not values:
            return None
        return {"min": round(min(values), 6), "max": round(max(values), 6),
                "mean": round(float(np.mean(values)), 6)}

    return {
        "records": len(collected),
        "instances": instances,
        "classes": dict(sorted(by_class.items())),
        "bandwidth_ratio": _triple(bandwidth_ratios),
        "snr_nominal_db": _triple(snrs),
    }


def main(argv=None):
    args = _parse_args(argv)
    output = Path(args.output)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"{output} 已有 manifest.json；确认要覆盖请加 --overwrite")

    torchsig = _import_torchsig()
    version = getattr(torchsig, "__version__", "unknown")
    metadata, overrides = _dataset_metadata(args)
    dataset, component_transforms, description = _make_dataset(args, metadata)
    print(f"TorchSig {version}：数据集已构造（{description}）", flush=True)
    print(f"  覆盖后的元数据：{json.dumps(overrides, ensure_ascii=False, sort_keys=True)}",
          flush=True)
    if component_transforms:
        print("  注意：已启用 TorchSig 扰动，元数据由上游变换同步维护；"
              "若下游发现标签与图像不符，请先用 --impairment-level 0 复现。", flush=True)

    rate = float(args.sample_rate)
    collected = _collect(dataset, args, rate)
    summary = _summarize(collected)
    names = _class_names(dataset)
    print(f"  信号实例 {summary['instances']} 个，类名分布 "
          f"{json.dumps(summary['classes'], ensure_ascii=False)}", flush=True)
    print(f"  占用带宽比 {summary['bandwidth_ratio']}，标称信噪比 "
          f"{summary['snr_nominal_db']}", flush=True)

    if args.dry_run:
        print("  --dry-run：未写盘。第一条记录的实例摘要：")
        print(json.dumps(collected[0][0]["components"], ensure_ascii=False, indent=2))
        return 0

    note = (f"TorchSig {version} 生成：count={int(args.count)} seed={int(args.seed)} "
            f"{description} num_iq_samples={int(args.num_iq_samples)} "
            f"classes={'/'.join(names) or 'unknown'}")
    write_bundle(output, collected, sample_rate_hz=rate, torchsig_version=version,
                 metadata_overrides=overrides, note=note)
    print(f"bundle 已写出：{output}（iq/ 与 meta/ 各 {len(collected)} 个文件）",
          flush=True)
    print("下一步：.venv/bin/python training/ingest_torchsig.py --bundle "
          f"{output} --output <数据集目录> --nfft {int(args.nfft)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
