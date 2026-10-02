"""原始 IQ 分类分支（``iq_waveform_v1``）：复基带窗口 + 1D CNN／TCN 判别。

与 A09 特征通路（:mod:`signal_analysis.contracts.amc`）的分工：

* **输入不同**：A09 通路把频带内的 34 维确定性特征交给线性／ONNX 判别器
  （``amc_feature_vector_v1``）；本分支把**原始复基带采样点**直接交给网络
  （``iq_waveform_v1``，``(1, 2, N)``），由网络自行学习调制特征。
* **前段完全相同**：两条通路都走 ``algorithms.dsp.preprocess`` 的混频 → 65 抽头低通 → 按
  ``SAMPLES_PER_BAND = 8.0`` 抽取，因此"分析率"与"每符号点数"这两个最容易
  分叉的量逐位一致。清单若声明了不同的前段参数会被直接拒绝
  （见 :func:`_require_front_end`），而不是静默换一种预处理。
* **结果契约独立**：``amc_iq_classify_v1``。标签集合由模型清单声明：A09 六类
  时为 ``class_set = "a09"``；更宽的独立标签字典（例如第三方数据集的调制族）
  仍走同一契约、标 ``class_set = "custom"``。**冻结的
  :data:`~signal_analysis.contracts.amc.AMC_CLASSES` 从不被扩展或修改**。

``iq_waveform_v1`` 的唯一定义（训练与推理共用 :func:`iq_waveform`）：

1. 把占用 ``[offset ± 带宽/2]`` 的信号搬到零频、低通、按带宽抽取到
   ``分析率 = 8 × 带宽``（与 A09 特征通路同一实现）；
2. 取**居中**的 ``N`` 个采样点（``N`` 由模型清单固定；不足直接报错，不补零——
   补零等于凭空造样本）；
3. 按窗口 RMS 归一化为单位平均功率（``normalization = "unit_rms"``）：判别只看
   波形形状，绝对功率在 :func:`iq_waveform` 里另行测量并写入
   ``waveform.power_dbfs``，不因归一化而丢失；
4. 通道排布 ``iq_channels_first_v1``：``channel 0 = I``（实部）、
   ``channel 1 = Q``（虚部）。

ONNX 清单的输入节点为 ``(1, 2, N)`` float32、输出节点为 ``(1, C)`` 分数向量；
标准化（单位 RMS）必须发生在**进入图之前**的网络外，因为它是本项目的数据口径
而不是模型的自由选择——清单里记录该口径供核对。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from ...contracts.iq import (
    DEFAULT_IQ_SAMPLES,
    IQ_INPUT_CHANNELS,
    IQ_LAYOUT,
    IQ_NORMALIZATION,
    IQ_RESULT_CONTRACT,
    IQ_WAVEFORM_CONTRACT,
    IQModelError,
    MAX_IQ_SAMPLES,
    MIN_IQ_SAMPLES,
    _require_front_end,
    class_labels,
    class_set_name,
    read_iq_manifest,
)

from ..dsp.base import (
    _rounded,
)

from ..dsp.preprocess import (
    LOWPASS_TAPS,
    SAMPLES_PER_BAND,
    _inband_snr,
    _mix_and_decimate,
    _validate,
)

from .feature_model import (
    LOW_SNR_WARNING_DB,
    _softmax,
)

_CONFIG_KEYS = ("offset_hz", "bandwidth_hz")
_RESULT_KEYS = ("contract", "algorithm", "classes", "class_set", "labels", "waveform",
                "snr_estimate_db", "prediction", "model", "timing", "pending")


def _resolve_config(config):
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("IQ 分类配置应为字典")
    unknown = sorted(set(config) - set(_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"IQ 分类配置不支持以下字段：{'、'.join(unknown)}")
    return {key: config[key] for key in config if config[key] is not None}


def _number(value, label, *, minimum=None, maximum=None, default=None):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{label}应为有限数值")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label}不应小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label}不应大于 {maximum}")
    return number


def _classes(manifest):
    classes = list((manifest.get("output") or {}).get("classes") or [])
    return classes


# ---------------------------------------------------------------------------
# 输入张量
# ---------------------------------------------------------------------------


def iq_waveform(samples, sample_rate, offset_hz=0.0, bandwidth_hz=None, *,
                window_samples=DEFAULT_IQ_SAMPLES):
    """产出 ``iq_waveform_v1`` 张量，返回 ``(tensor, meta)``。

    ``tensor`` 形状 ``(2, N)`` float32（I、Q 两通道，单位 RMS）；``meta`` 记录
    全部换算参数与测量量，供结果留档与复核。``window_samples`` 是窗口长度
    ``N``（对应清单的 ``input.samples``），只能按关键字给，以免与 IQ 数据本身
    的 ``samples`` 混滑。
    """
    if isinstance(sample_rate, bool) or not np.isfinite(float(sample_rate)) \
            or float(sample_rate) <= 0:
        raise ValueError("采样率必须为正的有限数值")
    rate = float(sample_rate)
    view = _validate(samples)
    center = _number(offset_hz or 0.0, "中心频率", default=0.0)
    if abs(center) > rate:
        raise ValueError("中心频率必须为有限数值且不超过采样率")
    if bandwidth_hz is None:
        band = rate
        center = 0.0
    else:
        band = _number(bandwidth_hz, "占用带宽")
        if not 0.0 < band <= rate:
            raise ValueError("占用带宽必须大于 0 且不超过采样率")
    length = int(window_samples if window_samples is not None else DEFAULT_IQ_SAMPLES)
    if not MIN_IQ_SAMPLES <= length <= MAX_IQ_SAMPLES:
        raise ValueError(f"窗口采样点数应为 {MIN_IQ_SAMPLES}～{MAX_IQ_SAMPLES} 之间的整数")

    work, work_rate, factor = _mix_and_decimate(view, rate, center, band)
    if work.size < length:
        raise ValueError(
            f"频带内可用样本 {work.size} 少于模型要求的 {length}：请加长记录、放宽带宽，"
            "或改用窗口更短的模型清单（此处不补零，以免伪造样本）")
    start = (work.size - length) // 2
    window = work[start:start + length]
    power = float(np.mean(np.abs(window) ** 2))
    if not power > 0.0:
        raise ValueError("频带内功率为零，无法构造 IQ 窗口")
    rms = math.sqrt(power)
    peak = float(np.max(np.abs(window)))
    normalized = window / rms
    tensor = np.stack([normalized.real, normalized.imag]).astype(np.float32)
    assert tensor.shape == (IQ_INPUT_CHANNELS, length)
    snr_estimate = _inband_snr(view, rate, center, band)
    meta = {
        "contract": IQ_WAVEFORM_CONTRACT,
        "layout": IQ_LAYOUT,
        "channels": IQ_INPUT_CHANNELS,
        "normalization": IQ_NORMALIZATION,
        "samples": int(length),
        "sample_rate_hz": rate,
        "offset_hz": center,
        "bandwidth_hz": band,
        "analysis_rate_hz": float(work_rate),
        "decimation": int(factor),
        "samples_per_band": float(SAMPLES_PER_BAND),
        "lowpass_taps": int(LOWPASS_TAPS),
        "source_samples": int(view.size),
        "analysis_samples": int(work.size),
        "window_start": int(start),
        "power_dbfs": _rounded(10.0 * math.log10(power)),
        "rms": _rounded(rms),
        "peak": _rounded(peak),
        "crest_factor": _rounded(peak / rms),
        "snr_estimate_db": _rounded(snr_estimate, 2),
    }
    json.dumps(meta, ensure_ascii=False, allow_nan=False)
    return tensor, meta


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------


def iq_scores(manifest_path, waveform, threads=None):
    """按清单跑一次 ONNX 分类器，返回 ``(scores, manifest, probabilities)``。"""
    from ...inference.runtime import check_version, runtime_module

    manifest, library = read_iq_manifest(manifest_path)
    expected = int(manifest["input"]["samples"])
    array = np.asarray(waveform, dtype=np.float32)
    if array.shape != (IQ_INPUT_CHANNELS, expected):
        raise IQModelError(f"输入应为 ({IQ_INPUT_CHANNELS}, {expected}) 的 IQ 张量，"
                           f"实际为 {array.shape}")
    runtime = runtime_module()
    check_version(manifest["runtime_min_version"], runtime)
    options = runtime.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
    session = runtime.InferenceSession(str(library), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    output = session.run([manifest["output"]["name"]],
                         {manifest["input"]["name"]: array.reshape(1, *array.shape)})[0]
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    classes = manifest["output"]["classes"]
    if values.size != len(classes):
        raise IQModelError(f"分类器输出维度 {values.size} 与类别数 {len(classes)} 不符")
    probabilities = _probabilities(values)
    return ({label: _rounded(float(probabilities[position]))
             for position, label in enumerate(classes)}, manifest, probabilities)


def _probabilities(values):
    """把网络输出解释成概率：已是概率分布则原样使用，否则做一次 softmax 并在结果里标注。"""
    values = np.asarray(values, dtype=np.float64)
    if float(np.min(values)) >= 0.0 and abs(float(np.sum(values)) - 1.0) < 1e-3:
        return values
    return _softmax(values)


def _model_info(runner, manifest):
    return {
        "id": manifest.get("id", getattr(runner, "model_name", "injected")),
        "version": manifest.get("version", ""),
        "manifest_path": manifest.get("manifest_path"),
        "sha256": manifest.get("sha256"),
        "library": manifest.get("library"),
        "class_set": manifest.get("class_set") or class_set_name(_classes(manifest)),
        "training": manifest.get("training") or {},
        "runtime_version": getattr(runner, "runtime_version", None),
    }


def amc_iq_classify(samples, sample_rate, config=None, model=None, runner=None, threads=None):
    """原始 IQ 分类入口，返回 ``amc_iq_classify_v1`` 结果字典。

    * ``model``：IQ 清单路径（``contract = iq_waveform_v1``）；
    * ``runner``：已加载的会话（可注入，便于测试或复用已加载模型）；
    * ``config``：只接受 ``offset_hz`` / ``bandwidth_hz``；窗口长度、通道排布、
      归一化与低通全部由清单固定，不给调用方静默改口径的机会。
    """
    settings = _resolve_config(config)
    manifest = {}
    if runner is None:
        if not model:
            raise ValueError("请提供 IQ 分类器清单路径（model=...）或已加载的推理会话")
        from ...inference.runtime import load_iq_runner

        runner, manifest, _ = load_iq_runner(model, threads=threads)
        manifest = dict(manifest)
    else:
        manifest = dict(getattr(runner, "manifest", {}) or {})
    classes = _classes(manifest)
    if not classes:
        raise IQModelError("分类器清单缺少类别定义")
    incoming = manifest.get("input") if isinstance(manifest.get("input"), dict) else {}
    preprocess = manifest.get("preprocess") if isinstance(manifest.get("preprocess"), dict) else {}
    front_end = _require_front_end(preprocess)
    length = int(incoming.get("samples", DEFAULT_IQ_SAMPLES))

    declared_offset = front_end.get("default_offset_hz")
    declared_bandwidth = front_end.get("default_bandwidth_hz")
    center = settings.get("offset_hz", declared_offset)
    band = settings.get("bandwidth_hz", declared_bandwidth)
    started = time.perf_counter()
    tensor, meta = iq_waveform(samples, sample_rate, center, band, window_samples=length)
    preprocess_ms = (time.perf_counter() - started) * 1000.0
    inference_started = time.perf_counter()
    output = runner.run(tensor)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    if values.size != len(classes):
        raise IQModelError(f"分类器输出维度 {values.size} 与类别数 {len(classes)} 不符")
    probabilities = _probabilities(values)
    order = np.argsort(-probabilities)
    best = int(order[0])
    runner_up = float(probabilities[order[1]]) if order.size > 1 else 0.0
    confidence = _rounded(float(probabilities[best]))
    label = classes[best]
    snr_db = meta.get("snr_estimate_db")
    if snr_db is None:
        reliable = True
        reason = "带内信噪比无法估计（占用带几乎覆盖采样带宽），未做低信噪比标注"
    elif snr_db < LOW_SNR_WARNING_DB:
        reliable = False
        reason = (f"粗略估计的带内信噪比 {snr_db:.1f} dB 低于 {LOW_SNR_WARNING_DB:.0f} dB，"
                  "识别结果仅供参考")
    elif confidence is not None and confidence < 0.5:
        reliable = False
        reason = f"最高类概率仅 {confidence:.2f}，类别区分度不足"
    else:
        reliable = True
        reason = None
    labels = class_labels(classes)
    result = {
        "contract": IQ_RESULT_CONTRACT,
        "algorithm": f"amc_iq_onnx_v1:{manifest.get('id', getattr(runner, 'model_name', 'injected'))}",
        "classes": list(classes),
        "class_set": manifest.get("class_set") or class_set_name(classes),
        "labels": labels,
        "waveform": meta,
        "snr_estimate_db": _rounded(snr_db, 2),
        "prediction": {
            "label": label,
            "label_text": labels.get(label, label),
            "confidence": confidence,
            "margin": _rounded(float(probabilities[best]) - runner_up),
            "scores": {name: _rounded(float(probabilities[position]))
                       for position, name in enumerate(classes)},
            "reliable": reliable,
            "reason": reason,
            "snr_note": ("带内信噪比按占用带内/外平均功率谱密度之比粗估，"
                         "仅用于可信度提示，不是验收口径"),
        },
        "model": _model_info(runner, manifest),
        "timing": {"preprocess_ms": round(preprocess_ms, 3),
                   "inference_ms": round(inference_ms, 3),
                   "total_ms": round(preprocess_ms + inference_ms, 3)},
        "pending": [
            "原始 IQ 通路的识别准确率合格门限尚未确认（技术方案待确认项）",
            "IQ 模型仅在本项目合成数据与转写数据上训练过，尚未用独立实采数据验证泛化",
            "同一分析频带内的多信号重叠会让 IQ 窗口混叠，需先由检测切分",
        ],
    }
    assert set(result) == set(_RESULT_KEYS)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
