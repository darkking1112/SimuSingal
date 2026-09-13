#!/usr/bin/env python3
"""``torchsig_bundle_v1``：TorchSig 与本项目之间的中间格式（**纯 NumPy**，不 import torchsig）。

为什么要有中间层
----------------

* **CI 里不能有 torchsig**：它要求 ≥1 TB 磁盘、多核、Ubuntu ≥22.04，装进开发环境
  只会让测试变脆。因此"读 bundle 转数据集"（``training/ingest_torchsig.py``）与
  "用 torchsig 生成 bundle"（``training/build_torchsig.py``）必须分开：
  前者只依赖 NumPy，能在任何环境里跑并被测试覆盖；后者是唯一 import torchsig 的地方。
* **格式定义只有一份**：读端与写端共用本模块，避免"生成脚本写 A、转换脚本读 B"
  的分叉——这正是本仓库在数据集卡片上吃过亏的地方（见 ``detectors/dataset.py``）。

目录结构::

    <bundle>/manifest.json      # bundle 级：采样率、torchsig 版本、元数据覆盖、记录清单
    <bundle>/iq/000000.npy      # complex64 一维序列（也接受 (N, 2) 实部/虚部）
    <bundle>/meta/000000.json   # 记录级：duration_s + components[]（TorchSig 字段名）

``components`` 里的字段名与 TorchSig 的 ``SignalMetadataObject`` 保持一致
（``center_freq / bandwidth / lower_freq / upper_freq / start / stop / snr_db /
class_name / start_in_samples / duration_in_samples``），因此从 TorchSig 元数据
平移过来时不需要改名，读端也不必猜。

**单位**（同名字段在两边含义不同，这里以 bundle 为准）::

    频率   center_freq / bandwidth / lower_freq / upper_freq       Hz
    时间   start / stop                                          秒（记录内绝对时间）
           start_in_samples / stop_in_samples / duration_in_samples 采样点
    其它   snr_db 为注入标称值；真实带内信噪比由读端 measure_band 重测

注意 TorchSig 2.2 的 ``SignalMetadataObject.start``/``stop``/``duration`` 是
**占记录长度的比例（0–1）**，与这里的秒制不同名同义；写端（
``training/build_torchsig.py``）会换算成秒并同时写出采样点字段，读端优先用采样
点字段，因此两边不会错位。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: bundle 的格式版本：清单里必须声明，防止读错格式还一路算下去
BUNDLE_VERSION = "torchsig_bundle_v1"

#: 记录元数据里已知的信号实例字段（写入时只保留这些键，避免把上游的
#: ``yolo_label`` 之类的"另一套坐标约定"带进来——它绝不能被下游消费）
COMPONENT_FIELDS = ("center_freq", "bandwidth", "lower_freq", "upper_freq",
                    "start", "stop", "start_in_samples", "stop_in_samples",
                    "duration_in_samples", "sample_rate", "snr_db",
                    "oversampling_rate", "class_name", "class_index")


def component(*, class_name="unknown", **fields):
    """构造一条信号实例元数据（只保留 :data:`COMPONENT_FIELDS` 里的字段）。

    至少要有 ``center_freq + bandwidth``，或 ``lower_freq + upper_freq``，
    否则读端会以 ``missing_geometry`` 拒绝该记录（这是有意的：宁可早失败）。
    """
    entry = {"class_name": str(class_name)}
    for key, value in fields.items():
        if key not in COMPONENT_FIELDS:
            raise ValueError(f"未知的信号实例字段 {key!r}；可用字段：{COMPONENT_FIELDS}")
        if value is None:
            continue
        entry[key] = value
    if "center_freq" not in entry and "lower_freq" not in entry:
        raise ValueError("信号实例至少需要 center_freq（或 lower_freq/upper_freq）")
    return entry


def record(*, index, iq, components, duration_s=None, sample_rate_hz=None, note=None):
    """一条记录的元数据（IQ 数组单独存放，这里只描述它）。"""
    entry = {
        "index": int(index),
        "duration_s": None if duration_s is None else float(duration_s),
        "sample_rate_hz": None if sample_rate_hz is None else float(sample_rate_hz),
        "components": [dict(item) for item in components],
    }
    if note:
        entry["note"] = str(note)
    return entry


def write_bundle(root, records, *, sample_rate_hz, torchsig_version=None,
                 metadata_overrides=None, note=None):
    """把 ``(记录元数据, IQ 数组)`` 列表写成 ``torchsig_bundle_v1`` 目录。

    ``records`` 是 ``(meta, iq)`` 二元组序列：``meta`` 由 :func:`record` 构造，
    ``iq`` 是 ``complex64`` 一维数组（写成 ``.npy``）。
    """
    root = Path(root)
    (root / "iq").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    manifest_records = []
    for position, (meta, iq) in enumerate(records):
        samples = np.asarray(iq)
        if samples.ndim != 1 or samples.size < 2:
            raise ValueError(f"第 {position} 条记录的 IQ 应为单通道序列，实际形状 {samples.shape}")
        if not np.iscomplexobj(samples):
            samples = samples.astype(np.float64).astype(np.complex64)
        samples = np.ascontiguousarray(samples, dtype=np.complex64)
        iq_path = f"iq/{position:06d}.npy"
        meta_path = f"meta/{position:06d}.json"
        np.save(root / iq_path, samples)
        payload = dict(meta)
        payload["index"] = position
        payload.setdefault("sample_rate_hz", float(sample_rate_hz))
        payload.setdefault("duration_s", float(samples.size) / float(sample_rate_hz))
        (root / meta_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        manifest_records.append({"index": position, "iq": iq_path, "meta": meta_path,
                                 "duration_s": payload["duration_s"]})
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "source": "torchsig",
        "torchsig_version": None if torchsig_version is None else str(torchsig_version),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample_rate_hz": float(sample_rate_hz),
        "sample_count": len(manifest_records),
        "metadata_overrides": dict(metadata_overrides or {}),
        "note": note or "TorchSig IQ + 信号实例元数据；图像与标签由本项目自行生成",
        "records": manifest_records,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return root


def load_bundle(root):
    """读取 ``torchsig_bundle_v1`` 清单并做最低限度的格式校验（返回清单字典）。"""
    root = Path(root)
    path = root / "manifest.json"
    if not path.is_file():
        raise SystemExit(f"{root} 不是有效的 bundle（缺少 manifest.json）；"
                         "请先用 training/build_torchsig.py 生成")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    version = manifest.get("bundle_version")
    if version != BUNDLE_VERSION:
        raise SystemExit(f"bundle 格式版本为 {version!r}，本脚本只认 {BUNDLE_VERSION}")
    rate = manifest.get("sample_rate_hz")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not rate > 0:
        raise SystemExit("bundle 清单缺少有效的 sample_rate_hz")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise SystemExit("bundle 清单里没有 records")
    for position, entry in enumerate(records):
        if not isinstance(entry, dict) or not entry.get("iq"):
            raise SystemExit(f"bundle 清单第 {position} 条记录缺少 iq 字段")
    return manifest
