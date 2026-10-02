"""A09 六类调制识别契约：类别字典、特征顺序与 ONNX 分类器清单。

类别字典**按技术方案 A09 原文主体**固定为六类：FM、SSB、2ASK、QPSK、16QAM、
64QAM。生成器里的 ``am`` 与两种跳频样式不在该六类之内（``mode_to_class``
返回 ``None``）。特征顺序由 :data:`AMC_FEATURES` 冻结，训练与推理必须一致。

本模块只依赖标准库与 NumPy；特征提取实现见
:mod:`signal_analysis.algorithms.amc.features`，模型训练与推理见
:mod:`signal_analysis.algorithms.amc.feature_model`。
"""

import json
from pathlib import Path

import numpy as np

from common.storage import file_digest


AMC_CLASSES = ("fm", "ssb", "ask2", "qpsk", "qam16", "qam64")
CLASS_LABELS = {
    "fm": "FM 调频",
    "ssb": "SSB 单边带",
    "ask2": "2ASK 幅度键控",
    "qpsk": "QPSK 四相键控",
    "qam16": "16QAM",
    "qam64": "64QAM",
}
AMC_MODEL_CONTRACT = "amc_model_v1"
AMC_ONNX_CONTRACT = "amc_feature_vector_v1"
AMC_RESULT_CONTRACT = "amc_classify_v1"
AMC_ONNX_INPUT = "features"
AMC_ONNX_OUTPUT = "scores"
AMC_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MODEL_NAME = "amc_default.json"
DEFAULT_MODEL_ID = "amc-linear-default"
MIN_RUNTIME_VERSION = "1.17"


AMC_FEATURES = (
    "env_cv",              # 归一化包络变异系数：AM/ASK/SSB 高，FM/PSK 低
    "env_kurtosis",        # 包络峰度（超额）：ASK 双电平最低
    "env_gamma_max_db",    # 归一化包络谱峰值（相对均值，dB）：数字调制有符号率线谱
    "spec_flatness",       # 占用带内功率谱平坦度（几何/算术均值比）：SSB 矩形谱最高
    "spec_edge_ratio",     # 带外沿 10% 与中间 50% 的平均功率谱密度之比：SSB 接近 1
    "psd_peak_ratio",      # 带内峰均功率谱密度：含载波线（AM/2ASK）最高
    "inst_freq_std",       # 归一化瞬时频率标准差（以占用带宽为单位）
    "inst_freq_kurtosis",  # 瞬时频率峰度：PSK/QAM 相位跳变呈冲击性
    "phase_diff_std",      # 相邻相位差标准差（/π）：恒模 PSK 有离散台阶
    "m20_mag",             # |M20|（单位功率归一化）：实信号（AM/2ASK）接近 1
    "c42_mag",             # |C42|：2ASK 高，QPSK≈1，16/64QAM≈0.6
    "c63_mag",             # 六阶累积量：|C63| 在 QPSK≈4，16QAM≈2.1，64QAM≈1.8。
    "amp_spread",          # (p90-p10)/中位数幅度：多电平幅度键控大
    "amp_clusters",        # 全样本幅度直方图峰数（1～8）
    "peak_cv",             # 峰值采样（近似符号时刻）幅度变异系数
    "peak_c63",            # 峰值采样 |C63|：区分 16QAM 与 64QAM 的主特征
    "peak_clusters",       # 峰值采样幅度直方图峰数（1～8）
    # 峰值幅度归一化直方图模板（16 桶，和为 1）：16QAM 只有 3 个幅度环
    # （多重度 4:8:4），64QAM 约 10 个环，低 SNR 下形状比单个高阶累积量稳。
    "peak_hist_01", "peak_hist_02", "peak_hist_03", "peak_hist_04",
    "peak_hist_05", "peak_hist_06", "peak_hist_07", "peak_hist_08",
    "peak_hist_09", "peak_hist_10", "peak_hist_11", "peak_hist_12",
    "peak_hist_13", "peak_hist_14", "peak_hist_15", "peak_hist_16",
    "snr_estimate_db",     # 带内信噪比粗估（dB）：线性模型据此补偿各阶量的 SNR 漂移
)


AMC_FEATURE_CONTRACT = "amc_feature_vector_v1"


class ModelError(ValueError):
    """AMC 模型文件缺失、格式不符或校验失败。"""


def _require_digest(value):
    digest = str(value or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ModelError("模型摘要应为 64 位十六进制字符串")
    return digest


def write_amc_manifest(output, model, *, classes=None, features=None, standardize=None,
                       identifier=DEFAULT_MODEL_ID, version="0.1.0", opset=17,
                       input_name=AMC_ONNX_INPUT, output_name=AMC_ONNX_OUTPUT,
                       training=None, notes=""):
    """为 ONNX 分类器生成清单（``AMC_ONNX_CONTRACT``）。

    模型必须位于清单目录内并使用相对路径；``standardize`` 用于留档与核对，
    真正的标准化必须已写入导出图。
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = Path(model).resolve(strict=True)
    if model.suffix.lower() != ".onnx":
        raise ModelError("分类器模型应为 .onnx")
    if not model.is_relative_to(output.resolve().parent):
        raise ModelError("模型文件必须位于清单目录内")
    classes = list(classes or AMC_CLASSES)
    features = list(features or AMC_FEATURES)
    if len(classes) != len(AMC_CLASSES) or set(classes) != set(AMC_CLASSES):
        raise ModelError("分类器清单的类别必须与 A09 六类字典一致")
    if features != list(AMC_FEATURES):
        raise ModelError("分类器清单的特征顺序必须与 AMC_FEATURES 一致")
    payload = {
        "schema_version": AMC_MANIFEST_SCHEMA_VERSION,
        "task": "amc",
        "id": str(identifier).strip() or DEFAULT_MODEL_ID,
        "version": str(version).strip() or "0.1.0",
        "runtime": "onnxruntime",
        "runtime_min_version": MIN_RUNTIME_VERSION,
        "contract": AMC_ONNX_CONTRACT,
        "library": str(model.relative_to(output.resolve().parent)),
        "sha256": file_digest(model),
        "opset": int(opset or 0),
        "input": {"name": input_name, "size": len(features)},
        "output": {"name": output_name, "classes": classes},
        "features": features,
        "standardize": {
            "mean": [float(value) for value in (standardize or {}).get("mean", [0.0] * len(features))],
            "scale": [float(value) for value in (standardize or {}).get("scale", [1.0] * len(features))],
            "note": "标准化应已写入导出图；此处仅留档核对",
        },
        "training": dict(training or {}),
        "notes": str(notes),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                      encoding="utf-8")
    manifest, library = read_amc_manifest(output)
    return manifest, library


def read_amc_manifest(path):
    """校验分类器清单，返回 ``(manifest, 模型绝对路径)``。"""
    path = Path(path)
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ModelError(f"分类器清单不可读：{exc}") from exc
    if path.stat().st_size > 64 * 1024:
        raise ModelError("分类器清单超过 64 KiB")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelError(f"分类器清单不是合法 JSON：{exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelError("分类器清单应为 JSON 对象")
    if manifest.get("schema_version") != AMC_MANIFEST_SCHEMA_VERSION:
        raise ModelError(f"不支持的清单版本：{manifest.get('schema_version')!r}")
    if manifest.get("contract") != AMC_ONNX_CONTRACT:
        raise ModelError(f"不支持的分类器契约：{manifest.get('contract')!r}")
    if manifest.get("runtime", "onnxruntime") != "onnxruntime":
        raise ModelError(f"不支持的推理运行时：{manifest.get('runtime')!r}")
    digest = _require_digest(manifest.get("sha256"))
    features = list(manifest.get("features") or [])
    if features != list(AMC_FEATURES):
        raise ModelError("分类器清单的特征顺序与 AMC_FEATURES 不一致")
    incoming = manifest.get("input") or {}
    if int(incoming.get("size", 0) or 0) != len(AMC_FEATURES):
        raise ModelError("分类器清单的特征维数与 AMC_FEATURES 不一致")
    classes = list((manifest.get("output") or {}).get("classes") or manifest.get("classes") or [])
    if len(classes) != len(AMC_CLASSES) or set(classes) != set(AMC_CLASSES):
        raise ModelError("分类器清单的类别与 A09 六类字典不一致")
    standardize = manifest.get("standardize") or {}
    for key, default in (("mean", 0.0), ("scale", 1.0)):
        values = standardize.get(key, [default] * len(features))
        if len(values) != len(features) or not np.isfinite([float(v) for v in values]).all():
            raise ModelError(f"分类器清单的 standardize.{key} 非法")
    relative = Path(str(manifest.get("library", "")))
    if not str(manifest.get("library") or "").strip() or relative.is_absolute():
        raise ModelError("分类器模型必须使用清单目录内的相对路径")
    try:
        library = (path.parent / relative).resolve(strict=True)
    except OSError as exc:
        raise ModelError(f"分类器模型文件不可读：{exc}") from exc
    if not library.is_relative_to(path.parent) or not library.is_file() \
            or library.stat().st_size == 0:
        raise ModelError("分类器模型文件缺失、为空或越过清单目录")
    if file_digest(library) != digest:
        raise ModelError("分类器模型摘要与清单不符，请重新生成清单")
    resolved = {
        "id": str(manifest.get("id") or DEFAULT_MODEL_ID),
        "version": str(manifest.get("version") or "0.1.0"),
        "contract": AMC_ONNX_CONTRACT,
        "sha256": digest,
        "opset": int(manifest.get("opset", 0) or 0),
        "input": {"name": str(incoming.get("name") or AMC_ONNX_INPUT),
                  "size": len(features)},
        "output": {"name": str((manifest.get("output") or {}).get("name") or AMC_ONNX_OUTPUT),
                   "classes": classes},
        "features": features,
        "standardize": standardize,
        "runtime_min_version": str(manifest.get("runtime_min_version", MIN_RUNTIME_VERSION)),
        "training": manifest.get("training") if isinstance(manifest.get("training"), dict) else {},
        "notes": str(manifest.get("notes") or ""),
        "manifest_path": str(path),
        "library": str(library),
    }
    return resolved, library
