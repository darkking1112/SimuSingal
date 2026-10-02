"""原始 IQ 分类契约（``iq_waveform_v1``）：常量、类别工具与分类器清单。

本模块只依赖标准库与 NumPy；窗口提取实现见
:mod:`signal_analysis.algorithms.amc.iq_model`（训练脚本共用同一份
``iq_waveform``），推理会话见 :mod:`signal_analysis.inference.runtime`。

清单校验是硬门禁：清单声明的归一化、抽取比与低通抽头数必须与本实现
（:mod:`signal_analysis.contracts.preprocess`）逐项一致，否则直接拒绝而不
静默换一种预处理。
"""

import json
from pathlib import Path

import numpy as np

from common.storage import file_digest

from .amc import AMC_CLASSES, CLASS_LABELS
from .preprocess import LOWPASS_TAPS, SAMPLES_PER_BAND


#: 输入张量契约（``(1, 2, N)`` float32 复基带窗口）
IQ_WAVEFORM_CONTRACT = "iq_waveform_v1"
#: 通道排布：0 = I，1 = Q
IQ_LAYOUT = "iq_channels_first_v1"
#: 归一化口径：窗口 RMS 归一化为单位平均功率
IQ_NORMALIZATION = "unit_rms"
#: 分类器清单契约与结果契约
IQ_ONNX_CONTRACT = IQ_WAVEFORM_CONTRACT
IQ_RESULT_CONTRACT = "amc_iq_classify_v1"
IQ_MANIFEST_SCHEMA_VERSION = 1
IQ_TASK = "amc_iq"
IQ_ONNX_INPUT = "iq"
IQ_ONNX_OUTPUT = "scores"
IQ_INPUT_CHANNELS = 2
DEFAULT_IQ_SAMPLES = 1024
MIN_IQ_SAMPLES = 64
MAX_IQ_SAMPLES = 65536
#: 标签集合标识：与 A09 六类完全一致 / 更宽的独立字典
CLASS_SET_A09 = "a09"
CLASS_SET_CUSTOM = "custom"
MAX_CLASSES = 64
MAX_CLASS_NAME = 64
MIN_RUNTIME_VERSION = "1.17"
IQ_DEFAULT_MODEL_NAME = "iq.onnx"
IQ_DEFAULT_MODEL_ID = "iq-cnn-default"


class IQModelError(ValueError):
    """IQ 分类器清单缺失、格式不符或校验失败。"""


def class_set_name(classes):
    """标签集合标识：与 A09 六类一致时 ``"a09"``，否则 ``"custom"``。"""
    return CLASS_SET_A09 if list(classes) == list(AMC_CLASSES) else CLASS_SET_CUSTOM


def class_labels(classes):
    """类别显示名：A09 六类用中文全称，其余类别回落到原始标签。"""
    return {name: CLASS_LABELS.get(name, name) for name in classes}


def _require_digest(manifest):
    digest = str(manifest.get("sha256") or "").lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise IQModelError("分类器清单的 sha256 应为 64 位十六进制字符串")
    return digest


def _require_channels(manifest):
    incoming = manifest.get("input")
    if not isinstance(incoming, dict):
        raise IQModelError("分类器清单缺少 input 段")
    layout = incoming.get("layout", IQ_LAYOUT)
    if layout != IQ_LAYOUT:
        raise IQModelError(f"不支持的 IQ 通道排布：{layout!r}（应为 {IQ_LAYOUT}）")
    if int(incoming.get("channels", IQ_INPUT_CHANNELS) or 0) != IQ_INPUT_CHANNELS:
        raise IQModelError("IQ 输入契约固定为 2 通道（I、Q）")
    size = incoming.get("samples", DEFAULT_IQ_SAMPLES)
    if isinstance(size, bool) or not isinstance(size, int) or not MIN_IQ_SAMPLES <= size <= MAX_IQ_SAMPLES:
        raise IQModelError(f"输入窗口采样点数应为 {MIN_IQ_SAMPLES}～{MAX_IQ_SAMPLES} 之间的整数")
    name = incoming.get("name") or IQ_ONNX_INPUT
    if not isinstance(name, str) or not name.strip():
        raise IQModelError("模型输入名称不合法")
    return {"name": name.strip(), "samples": int(size), "channels": IQ_INPUT_CHANNELS,
            "layout": IQ_LAYOUT}


def _require_class_names(manifest):
    incoming = manifest.get("output") or {}
    if not isinstance(incoming, dict):
        raise IQModelError("分类器清单的 output 段应为对象")
    raw = incoming.get("classes") or manifest.get("classes")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise IQModelError("分类器清单缺少 output.classes")
    if len(raw) > MAX_CLASSES:
        raise IQModelError(f"类别数不应超过 {MAX_CLASSES}")
    cleaned = []
    for label in raw:
        if not isinstance(label, str) or not label.strip() or len(label) > MAX_CLASS_NAME:
            raise IQModelError("output.classes 中存在非法类别名")
        cleaned.append(label.strip())
    if len(set(cleaned)) != len(cleaned):
        raise IQModelError("output.classes 中存在重复类别名")
    name = incoming.get("name") or IQ_ONNX_OUTPUT
    if not isinstance(name, str) or not name.strip():
        raise IQModelError("模型输出名称不合法")
    return {"name": name.strip(), "classes": cleaned}


def _require_front_end(preprocess):
    """拒绝清单里声明的、与本实现不一致的前段参数。

    这三个量决定"每个采样点代表多少带宽"，一旦与训练时不同，模型看到的输入
    分布就变了（而且不会有任何报错）。因此这里做硬门禁：清单可以少写，但写了
    就必须等于本实现的口径。
    """
    if not isinstance(preprocess, dict) or not preprocess:
        return {"normalization": IQ_NORMALIZATION, "samples_per_band": SAMPLES_PER_BAND,
                "lowpass_taps": LOWPASS_TAPS, "declared": False}
    normalization = preprocess.get("normalization", IQ_NORMALIZATION)
    if normalization != IQ_NORMALIZATION:
        raise IQModelError(f"清单声明的归一化方式为 {normalization!r}，本实现只支持 "
                           f"{IQ_NORMALIZATION!r}；请用同一份 training/build_iq_dataset.py 重训")
    samples_per_band = preprocess.get("samples_per_band", SAMPLES_PER_BAND)
    if abs(float(samples_per_band) - float(SAMPLES_PER_BAND)) > 1e-9:
        raise IQModelError(f"清单声明的每带宽采样点数为 {samples_per_band}，本实现为 "
                           f"{SAMPLES_PER_BAND}；前段不一致会让输入分布漂移，已拒绝")
    lowpass_taps = preprocess.get("lowpass_taps", LOWPASS_TAPS)
    if int(lowpass_taps) != int(LOWPASS_TAPS):
        raise IQModelError(f"清单声明的低通抽头数为 {lowpass_taps}，本实现为 {LOWPASS_TAPS}")
    return {"normalization": IQ_NORMALIZATION, "samples_per_band": float(SAMPLES_PER_BAND),
            "lowpass_taps": int(LOWPASS_TAPS), "declared": True,
            "default_offset_hz": preprocess.get("default_offset_hz"),
            "default_bandwidth_hz": preprocess.get("default_bandwidth_hz")}


def read_iq_manifest(path):
    """校验 IQ 分类器清单，返回 ``(manifest, 模型绝对路径)``。

    这是与 :func:`signal_analysis.contracts.manifest.read_model_manifest`
    **并列**的读取器：后者继续只接受 ``tf_image_v1`` 时频图清单，不会因为
    新增 IQ 分支而放宽（用错清单类型会在两个方向上都报错，而不是静默按错的
    契约推理）。
    """
    path = Path(path)
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise IQModelError(f"IQ 分类器清单不可读：{exc}") from exc
    if path.stat().st_size > 64 * 1024:
        raise IQModelError("IQ 分类器清单超过 64 KiB")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IQModelError(f"IQ 分类器清单不是合法 JSON：{exc}") from exc
    if not isinstance(manifest, dict):
        raise IQModelError("IQ 分类器清单应为 JSON 对象")
    if manifest.get("schema_version") != IQ_MANIFEST_SCHEMA_VERSION:
        raise IQModelError(f"不支持的清单版本：{manifest.get('schema_version')!r}")
    if manifest.get("contract") != IQ_ONNX_CONTRACT:
        raise IQModelError(
            f"不支持的分类器契约：{manifest.get('contract')!r}"
            f"（IQ 分支需要 {IQ_ONNX_CONTRACT}；时频图检测模型请用 ml-detect）")
    if manifest.get("runtime", "onnxruntime") != "onnxruntime":
        raise IQModelError(f"不支持的推理运行时：{manifest.get('runtime')!r}")
    digest = _require_digest(manifest)
    incoming = _require_channels(manifest)
    outgoing = _require_class_names(manifest)
    front_end = _require_front_end(manifest.get("preprocess") or {})
    identifier = str(manifest.get("id") or IQ_DEFAULT_MODEL_ID).strip()
    if not identifier or len(identifier) > 200:
        raise IQModelError("分类器清单的 id 非法")
    relative = Path(str(manifest.get("library") or ""))
    if not str(manifest.get("library") or "").strip() or relative.is_absolute():
        raise IQModelError("分类器模型必须使用清单目录内的相对路径")
    try:
        library = (path.parent / relative).resolve(strict=True)
    except OSError as exc:
        raise IQModelError(f"分类器模型文件不可读：{exc}") from exc
    if not library.is_relative_to(path.parent) or not library.is_file() \
            or library.stat().st_size == 0:
        raise IQModelError("分类器模型文件缺失、为空或越过清单目录")
    if file_digest(library) != digest:
        raise IQModelError("分类器模型摘要与清单不符，请重新生成清单")
    resolved = {
        "id": identifier,
        "version": str(manifest.get("version") or "0.1.0"),
        "contract": IQ_ONNX_CONTRACT,
        "sha256": digest,
        "opset": int(manifest.get("opset", 0) or 0),
        "input": incoming,
        "output": outgoing,
        "classes": list(outgoing["classes"]),
        "class_set": class_set_name(outgoing["classes"]),
        "labels": class_labels(outgoing["classes"]),
        "preprocess": front_end,
        "runtime_min_version": str(manifest.get("runtime_min_version", MIN_RUNTIME_VERSION)),
        "training": manifest.get("training") if isinstance(manifest.get("training"), dict) else {},
        "notes": str(manifest.get("notes") or ""),
        "manifest_path": str(path),
        "library": str(library),
    }
    return resolved, library


def write_iq_manifest(output, model, *, identifier=IQ_DEFAULT_MODEL_ID, version="0.1.0",
                      classes=None, samples=DEFAULT_IQ_SAMPLES, opset=17,
                      input_name=IQ_ONNX_INPUT, output_name=IQ_ONNX_OUTPUT,
                      default_offset_hz=None, default_bandwidth_hz=None,
                      training=None, notes=""):
    """为已有 ``.onnx`` 分类器写出 ``iq_waveform_v1`` 清单。

    ``classes`` 是模型输出向量的类别顺序（必须与训练时的标签顺序一致）；
    A09 六类只是一种取值，更宽的标签字典同样合法——本函数**不会**修改冻结的
    :data:`~signal_analysis.contracts.amc.AMC_CLASSES`。
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = Path(model).resolve(strict=True)
    if model.suffix.lower() != ".onnx":
        raise IQModelError("分类器模型应为 .onnx")
    if not model.is_relative_to(output.resolve().parent):
        raise IQModelError("模型文件必须位于清单目录内，请先移动或复制模型")
    labels = [str(name).strip() for name in (classes or AMC_CLASSES)]
    payload = {
        "schema_version": IQ_MANIFEST_SCHEMA_VERSION,
        "task": IQ_TASK,
        "id": str(identifier).strip() or IQ_DEFAULT_MODEL_ID,
        "version": str(version).strip() or "0.1.0",
        "runtime": "onnxruntime",
        "runtime_min_version": MIN_RUNTIME_VERSION,
        "contract": IQ_ONNX_CONTRACT,
        "library": str(model.relative_to(output.resolve().parent)),
        "sha256": file_digest(model),
        "opset": int(opset or 0),
        "input": {"name": input_name, "samples": int(samples), "channels": IQ_INPUT_CHANNELS,
                  "layout": IQ_LAYOUT},
        "output": {"name": output_name, "classes": labels},
        "preprocess": {
            "normalization": IQ_NORMALIZATION,
            "samples_per_band": float(SAMPLES_PER_BAND),
            "lowpass_taps": int(LOWPASS_TAPS),
            "default_offset_hz": default_offset_hz,
            "default_bandwidth_hz": default_bandwidth_hz,
            "note": "前段与 A09 特征通路共享实现；清单声明的取值必须与本实现一致",
        },
        "training": dict(training or {}),
        "notes": str(notes or ""),
    }
    # 自校验：写出的清单必须能被自己的读取器接受，且类别顺序与标签集合标识一致
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(output)
    manifest, library = read_iq_manifest(output)
    return manifest, library
