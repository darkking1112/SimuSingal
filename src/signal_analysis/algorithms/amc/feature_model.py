"""A09 六类调制识别（AMC）：线性判别模型、ONNX 评分与推理编排。

契约（类别字典、特征顺序、清单）位于 ``contracts.amc``；确定性特征位于
``algorithms.amc.features``，启发式判定位于 ``algorithms.amc.heuristic``；
训练与推理继续使用同一份实现。模型管理与推理为纯 Python，不参与核心编译。

类别字典**按技术方案 A09 原文主体**固定为六类：

    FM、SSB、2ASK、QPSK、16QAM、64QAM

生成器里的 ``am`` 与两种跳频样式不在该六类之内：跳频是"会话级"检测对象，
AM 只作为原文数据库示例出现，因此不并入类别字典（``mode_to_class`` 返回 ``None``）。
评测时只统计类别字典覆盖的波形，未覆盖样本必须计入"不适用"而不是丢弃。

契约
----
* ``AMC_FEATURE_CONTRACT = "amc_feature_vector_v1"``：定长特征向量，顺序由
  :data:`AMC_FEATURES` 固定，全部为有限浮点数（见 :func:`extract_features`）。
* ``AMC_MODEL_CONTRACT = "amc_model_v1"``：JSON 线性模型，判别函数

      z = (x - mean) / scale
      p = softmax(temperature * (z @ W + b))

  ``mean``/``scale`` 与 ``W``/``b`` 之外不再有隐藏状态，便于复核与跨版本复现。
* ``AMC_ONNX_CONTRACT = "amc_feature_vector_v1"``：ONNX 分类器输入节点
  ``features``（``(1, F)`` float32，**未标准化**的原始特征），输出节点 ``scores``
  （``(1, C)``）。标准化必须写在导出图内（用训练脚本的 wrapper 模块完成），
  使"模型文件"自身就是完整口径；清单里的 ``standardize`` 仅用于核对与留档。

定位说明：这里的特征是**可解释的确定性基线**（A09 的双路方案之一），
不是"已验收的识别性能"。准确率门限仍列在技术方案的待确认项中。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from common.storage import file_digest
from ...contracts.amc import (
    AMC_CLASSES,
    AMC_FEATURE_CONTRACT,
    AMC_FEATURES,
    AMC_MODEL_CONTRACT,
    AMC_ONNX_CONTRACT,
    AMC_RESULT_CONTRACT,
    CLASS_LABELS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_NAME,
    ModelError,
    read_amc_manifest,
    write_amc_manifest,
)

from ..dsp.base import (
    _EPS,
    _rounded,
)

from .features import (
    extract_features,
    feature_vector,
)

from .heuristic import (
    classify_modulation,
)


#: 标准化尺度下限：直方图空桶等“近乎常量”的特征不放大噪声
STANDARDIZE_FLOOR = 1e-2


DEFAULT_L2 = 1e-3
LOW_SNR_WARNING_DB = 5.0
_TEMPERATURE_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)


def mode_to_class(mode):
    """生成器 ``mode`` → A09 类别名；六类之外的样式返回 ``None``。"""
    text = str(mode)
    return text if text in AMC_CLASSES else None


# ---------------------------------------------------------------------------
# 线性判别模型
# ---------------------------------------------------------------------------


def _softmax(logits):
    shifted = logits - float(np.max(logits))
    exponent = np.exp(shifted)
    return exponent / max(float(np.sum(exponent)), _EPS)


def fit_model(records, *, l2=DEFAULT_L2, provenance=None):
    """用特征记录训练线性判别模型。

    ``records`` 是 ``{"features": <特征>, "label": <类别>, "snr_db": <可选>}``
    的序列；至少覆盖两个类别。返回可直接 JSON 化的 ``AMC_MODEL_CONTRACT`` 模型。
    温度参数在训练集上按对数损失网格校准（**属于训练集内校准，不能当作独立指标**）。
    """
    rows = []
    labels = []
    snrs = []
    for record in records:
        label = str(record.get("label", ""))
        if label not in AMC_CLASSES:
            raise ValueError(f"标签不在 A09 六类字典内：{label!r}")
        rows.append(feature_vector(record["features"]))
        labels.append(label)
        snrs.append(record.get("snr_db"))
    if not rows:
        raise ValueError("训练记录为空")
    matrix = np.asarray(rows, dtype=np.float64)
    index_of = {label: position for position, label in enumerate(AMC_CLASSES)}
    targets = np.zeros((len(labels), len(AMC_CLASSES)), dtype=np.float64)
    for row, label in enumerate(labels):
        targets[row, index_of[label]] = 1.0
    mean = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), STANDARDIZE_FLOOR)
    normalized = (matrix - mean) / scale
    features = normalized.shape[1]
    gram = normalized.T @ normalized + float(l2) * np.eye(features)
    weights = np.linalg.solve(gram, normalized.T @ targets)
    bias = (targets - normalized @ weights).mean(axis=0)
    logits = normalized @ weights + bias
    temperature = _calibrate_temperature(logits, labels)
    scaled = logits * temperature
    probabilities = np.vstack([_softmax(row) for row in scaled])
    predictions = [AMC_CLASSES[int(np.argmax(row))] for row in probabilities]
    confusion = [[0] * len(AMC_CLASSES) for _ in AMC_CLASSES]
    for truth, predicted in zip(labels, predictions):
        confusion[index_of[truth]][index_of[predicted]] += 1
    accuracy = sum(truth == predicted for truth, predicted in zip(labels, predictions)) / len(labels)
    centroids = {}
    for position, label in enumerate(AMC_CLASSES):
        members = normalized[targets[:, position] > 0.5]
        centroids[label] = ([_rounded(value) for value in members.mean(axis=0)]
                            if members.size else [_rounded(value) for value in np.zeros(features)])
    model = {
        "contract": AMC_MODEL_CONTRACT,
        "feature_contract": AMC_FEATURE_CONTRACT,
        "schema_version": 1,
        "classes": list(AMC_CLASSES),
        "labels": {label: CLASS_LABELS[label] for label in AMC_CLASSES},
        "features": list(AMC_FEATURES),
        "l2": float(l2),
        "temperature": float(temperature),
        "standardize": {"mean": [_rounded(value) for value in mean],
                        "scale": [_rounded(value) for value in scale]},
        "weights": [[_rounded(value) for value in row] for row in weights],
        "bias": [_rounded(value) for value in bias],
        "centroids": centroids,
        "training": {
            "samples": len(labels),
            "support": {label: labels.count(label) for label in AMC_CLASSES},
            "accuracy_in_sample": _rounded(accuracy),
            "confusion_in_sample": confusion,
            "snr_db": {"min": _rounded(np.min([s for s in snrs if s is not None]), 3)
                       if any(s is not None for s in snrs) else None,
                       "max": _rounded(np.max([s for s in snrs if s is not None]), 3)
                       if any(s is not None for s in snrs) else None},
            "note": "in-sample 指标仅用于自检；正式指标须用独立验证集（evaluate_model）",
            **(provenance or {}),
        },
    }
    return model


def _calibrate_temperature(logits, labels):
    index_of = {label: position for position, label in enumerate(AMC_CLASSES)}
    truth = np.asarray([index_of[label] for label in labels])
    best = _TEMPERATURE_GRID[0]
    best_loss = float("inf")
    for temperature in _TEMPERATURE_GRID:
        probabilities = np.vstack([_softmax(row * temperature) for row in logits])
        picked = probabilities[np.arange(truth.size), truth]
        loss = float(-np.mean(np.log(np.maximum(picked, 1e-12))))
        if loss < best_loss - 1e-12:
            best, best_loss = temperature, loss
    return float(best)


def _validate_model(model):
    if not isinstance(model, dict):
        raise ModelError("模型应为 JSON 对象")
    if model.get("contract") != AMC_MODEL_CONTRACT:
        raise ModelError(f"不支持的模型契约：{model.get('contract')!r}")
    if list(model.get("classes") or []) != list(AMC_CLASSES):
        raise ModelError("模型类别字典与当前 A09 六类不一致")
    if list(model.get("features") or []) != list(AMC_FEATURES):
        raise ModelError("模型特征顺序与当前 AMC_FEATURES 不一致")
    features = len(AMC_FEATURES)
    classes = len(AMC_CLASSES)
    standardize = model.get("standardize") or {}
    try:
        mean = np.asarray([float(value) for value in standardize["mean"]], dtype=np.float64)
        scale = np.asarray([float(value) for value in standardize["scale"]], dtype=np.float64)
        weights = np.asarray(model["weights"], dtype=np.float64)
        bias = np.asarray([float(value) for value in model["bias"]], dtype=np.float64)
        temperature = float(model.get("temperature", 1.0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelError(f"模型参数不完整：{exc}") from exc
    if mean.shape != (features,) or scale.shape != (features,):
        raise ModelError("模型标准化参数长度与特征数不符")
    if weights.shape != (features, classes) or bias.shape != (classes,):
        raise ModelError("模型权重形状与特征数/类别数不符")
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()
            and np.isfinite(weights).all() and np.isfinite(bias).all()):
        raise ModelError("模型参数包含 NaN 或 Inf")
    if not np.all(scale > 0) or not math.isfinite(temperature) or temperature <= 0:
        raise ModelError("模型标准化尺度或温度参数非法")
    return mean, scale, weights, bias, temperature


def predict(model, features):
    """用线性模型预测六类概率。返回 ``label/confidence/scores/logits/nearest``。"""
    mean, scale, weights, bias, temperature = _validate_model(model)
    vector = np.asarray(feature_vector(features), dtype=np.float64)
    normalized = (vector - mean) / scale
    logits = normalized @ weights + bias
    probabilities = _softmax(logits * temperature)
    order = np.argsort(-probabilities)
    best = int(order[0])
    runner_up = float(probabilities[order[1]]) if order.size > 1 else 0.0
    centroids = model.get("centroids") or {}
    nearest = None
    if centroids:
        distances = {}
        for label in AMC_CLASSES:
            centroid = centroids.get(label)
            if centroid is None:
                continue
            distances[label] = float(np.sum((normalized - np.asarray(centroid, dtype=np.float64)) ** 2))
        if distances:
            nearest = min(distances, key=distances.get)
    return {
        "label": AMC_CLASSES[best],
        "label_text": CLASS_LABELS[AMC_CLASSES[best]],
        "confidence": _rounded(float(probabilities[best])),
        "margin": _rounded(float(probabilities[best]) - runner_up),
        "scores": {label: _rounded(float(probabilities[position]))
                   for position, label in enumerate(AMC_CLASSES)},
        "logits": {label: _rounded(float(logits[position]))
                   for position, label in enumerate(AMC_CLASSES)},
        "nearest_centroid": nearest,
    }


def save_model(model, path):
    """写出 ``AMC_MODEL_CONTRACT`` JSON（禁止 NaN/Inf，落盘前先自校验）。"""
    _validate_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def load_model(path):
    """读取并校验 ``AMC_MODEL_CONTRACT`` JSON。"""
    path = Path(path)
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ModelError(f"模型文件不可读：{exc}") from exc
    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"模型文件不是合法 JSON：{exc}") from exc
    _validate_model(model)
    model = dict(model)
    model["model_path"] = str(path)
    return model


def default_model_path():
    """内置模型路径（随包分发的合成数据训练基线，可能不存在）。"""
    return Path(__file__).resolve().parent / DEFAULT_MODEL_NAME


def load_default_model():
    """加载内置线性基线；缺失时给出明确的自训练指引。"""
    path = default_model_path()
    if not path.is_file():
        raise ModelError(
            f"未找到内置 AMC 模型：{path}\n"
            "请先用训练脚本重新生成（可用内置合成数据，约 1 分钟）：\n"
            "  .venv/bin/python training/build_amc_dataset.py --output training/data/amc --seed 11\n"
            "  .venv/bin/python training/train_amc.py --data training/data/amc\n"
            "或指定自己的模型：signal-analysis amc-classify <asset_id> --model <model.json|manifest.json>")
    model = load_model(path)
    model["source"] = "builtin"
    return model


def evaluate_model(model, records, labels=AMC_CLASSES):
    """在给定记录上评测模型：混淆矩阵、每类指标、总体准确率与分信噪比准确率。"""
    from ...evaluation import classification_metrics

    truth = []
    predicted = []
    snr_truth = {}
    snr_predicted = {}
    for record in records:
        label = str(record.get("label", ""))
        outcome = predict(model, record["features"])
        truth.append(label)
        predicted.append(outcome["label"])
        snr = record.get("snr_db")
        if snr is not None:
            bucket = int(math.floor(float(snr) / 5.0) * 5)
            snr_truth.setdefault(bucket, []).append(label)
            snr_predicted.setdefault(bucket, []).append(outcome["label"])
    metrics = classification_metrics(truth, predicted, labels=list(labels))
    metrics["per_snr"] = {
        f"{bucket:+d}~{bucket + 5:+d} dB": _rounded(
            sum(truth_value == predicted_value
                for truth_value, predicted_value in zip(snr_truth[bucket], snr_predicted[bucket]))
            / len(snr_truth[bucket]))
        for bucket in sorted(snr_truth)
    }
    metrics["model_id"] = model.get("id", DEFAULT_MODEL_ID)
    return metrics


# ---------------------------------------------------------------------------
# ONNX 分类器推理（可选路径，需要 onnxruntime）
# ---------------------------------------------------------------------------


def onnx_scores(manifest_path, features, threads=None):
    """按清单跑 ONNX 分类器，返回 ``(scores, manifest)``（概率之和为 1）。"""
    from ...inference.runtime import check_version, runtime_module

    manifest, library = read_amc_manifest(manifest_path)
    runtime = runtime_module()
    check_version(manifest["runtime_min_version"], runtime)
    vector = np.asarray([feature_vector(features)], dtype=np.float32)
    options = runtime.SessionOptions()
    if threads:
        options.intra_op_num_threads = int(threads)
    session = runtime.InferenceSession(str(library), sess_options=options,
                                       providers=["CPUExecutionProvider"])
    output = session.run([manifest["output"]["name"]],
                         {manifest["input"]["name"]: vector})[0]
    values = np.asarray(output, dtype=np.float64).reshape(-1)
    classes = manifest["output"]["classes"]
    if values.size != len(classes):
        raise ModelError(f"分类器输出维度 {values.size} 与类别数 {len(classes)} 不符")
    if float(np.min(values)) >= 0.0 and abs(float(np.sum(values)) - 1.0) < 1e-3:
        probabilities = values
    else:
        probabilities = _softmax(values)
    return {label: _rounded(float(probabilities[position]))
            for position, label in enumerate(classes)}, manifest, probabilities


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


_CONFIG_KEYS = ("offset_hz", "bandwidth_hz")
_RESULT_KEYS = ("contract", "algorithm", "classes", "labels", "features",
                "feature_contract", "band", "snr_estimate_db", "prediction",
                "baseline", "model", "pending")


def _resolve_config(config):
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("AMC 配置应为字典")
    unknown = sorted(set(config) - set(_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"AMC 配置不支持以下字段：{'、'.join(unknown)}")
    return {key: config[key] for key in config if config[key] is not None}


def amc_classify(samples, sample_rate, config=None, model=None, threads=None):
    """A09 六类调制识别：确定性特征 + 线性模型（或 ONNX 分类器）。

    返回结果遵守 ``AMC_RESULT_CONTRACT``，其中 ``prediction`` 为六类概率，
    ``baseline`` 是既有启发式"数字/模拟"判定（作为传统对照，始终计算）。
    低信噪比时 ``prediction.reliable`` 为 ``False``，但结果仍照常给出。
    """
    settings = _resolve_config(config)
    center = float(settings.get("offset_hz") or 0.0)
    bandwidth = settings.get("bandwidth_hz")
    features, info = extract_features(samples, sample_rate, center, bandwidth)
    try:
        baseline_label, clusters = classify_modulation(samples)
    except ValueError as exc:
        baseline_label, clusters = None, None
        baseline_note = str(exc)
    else:
        baseline_note = ""
    # 带内信噪比与特征中的 ``snr_estimate_db`` 同源（见 extract_features）
    snr_db = info.get("snr_estimate_db")

    runner = None
    if model is None:
        loaded = load_default_model()
        source = "builtin"
    elif isinstance(model, dict):
        loaded = model
        source = model.get("source", "inline")
    else:
        path = Path(model)
        if not path.is_file():
            raise ModelError(f"AMC 模型文件不存在：{path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelError(f"AMC 模型文件不是合法 JSON：{exc}") from exc
        if isinstance(payload, dict) and payload.get("contract") == AMC_ONNX_CONTRACT:
            source = "onnx"
            model_meta = {"id": payload.get("id", DEFAULT_MODEL_ID),
                          "version": payload.get("version", "0.1.0"),
                          "path": str(path)}
            prediction_scores, manifest, probabilities = onnx_scores(path, features, threads)
            runner = manifest
            order = sorted(range(len(manifest["output"]["classes"])),
                           key=lambda index: -probabilities[index])
            classes = manifest["output"]["classes"]
            prediction = {
                "label": classes[order[0]],
                "label_text": CLASS_LABELS.get(classes[order[0]], classes[order[0]]),
                "confidence": _rounded(float(probabilities[order[0]])),
                "margin": _rounded(float(probabilities[order[0]] - probabilities[order[1]]))
                if len(order) > 1 else None,
                "scores": prediction_scores,
                "nearest_centroid": None,
            }
            model_meta.update({"sha256": manifest["sha256"], "source": "onnx"})
        else:
            loaded = load_model(path)
            source = "file"
    if runner is None:
        prediction = predict(loaded, features)
        model_meta = {
            "id": loaded.get("id", DEFAULT_MODEL_ID),
            "version": loaded.get("version", "0.1.0"),
            "source": source,
            "model_path": loaded.get("model_path"),
            "sha256": (file_digest(loaded["model_path"])
                       if loaded.get("model_path") else None),
        }
        trained = (loaded.get("training") or {})
        if trained.get("accuracy_in_sample") is not None:
            model_meta["accuracy_in_sample"] = trained["accuracy_in_sample"]

    confidence = prediction["confidence"]
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
    prediction["reliable"] = reliable
    prediction["reason"] = reason
    prediction["snr_note"] = ("带内信噪比按占用带内/外平均功率谱密度之比粗估，"
                              "仅用于可信度提示，不是验收口径")

    result = {
        "contract": AMC_RESULT_CONTRACT,
        "algorithm": "amc_linear_v1" if runner is None else "amc_onnx_v1",
        "classes": list(AMC_CLASSES),
        "labels": dict(CLASS_LABELS),
        "features": features,
        "feature_contract": AMC_FEATURE_CONTRACT,
        "band": info,
        "snr_estimate_db": _rounded(snr_db, 2),
        "prediction": prediction,
        "baseline": {
            "algorithm": "heuristic_digital_analog_v1",
            "classification": baseline_label,
            "cluster_estimate": clusters,
            "note": baseline_note or "传统启发式只给出数字/模拟与簇数估计，不区分类别",
        },
        "model": model_meta,
        "pending": ["识别准确率的合格门限尚未确认（技术方案待确认项）",
                    "AMC 使用单信号频带特征，多信号重叠场景需先由检测切分"],
    }
    assert set(result) == set(_RESULT_KEYS)
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result
