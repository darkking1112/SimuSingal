"""AI 检测模型清单：格式定义、校验与生成。

清单把"模型文件 + 输入输出契约 + 训练出处"固定下来，使推理侧不必猜测模型
的输入格式，也能在加载前发现文件被替换或跨目录引用。

清单字段（``schema_version = 1``）：

```json
{
  "schema_version": 1,
  "id": "spectrogram-detector",
  "version": "0.1.0",
  "runtime": "onnxruntime",
  "runtime_min_version": "1.17",
  "contract": "tf_image_v1",
  "library": "detector.onnx",
  "sha256": "<64 位十六进制>",
  "opset": 17,
  "input": {"name": "images", "image_size": 1024, "channels": 1},
  "output": {"name": "detections", "layout": "normalized_boxes_v1"},
  "labels": ["emitter"],
  "training": {"framework": "yolox", "license": "Apache-2.0", "dataset": "自建 1024² 时频图"},
  "notes": "可选的人工说明"
}
```

约定：

* ``contract`` 固定为 :data:`INPUT_CONTRACT`，指时频图张量契约（见 ``tensor.py``）。
* ``output.layout`` 固定为 :data:`OUTPUT_LAYOUT`：``(1, N, 6)`` 浮点数组，
  每行 ``[x_center, y_center, width, height, confidence, class]``，前四项按
  图像宽高归一化到 ``[0, 1]``（边框坐标，不是像素中心）。
* 模型文件必须使用清单目录内的**相对路径**，并带 SHA-256 摘要。
"""

from __future__ import annotations

import json
import platform
import struct
from pathlib import Path

from common.storage import file_digest

MANIFEST_SCHEMA_VERSION = 1
INPUT_CONTRACT = "tf_image_v1"
IMAGE_LAYOUT = "time_frequency_grayscale_v1"
OUTPUT_LAYOUT = "normalized_boxes_v1"
RUNTIME = "onnxruntime"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_MODEL_NAME = "detector.onnx"
DEFAULT_NFFT = 512
DEFAULT_DYNAMIC_RANGE_DB = 60.0
MIN_NFFT = 16
MAX_NFFT = 4096
MIN_DYNAMIC_RANGE_DB = 10.0
MAX_DYNAMIC_RANGE_DB = 120.0
MAX_MANIFEST_BYTES = 64 * 1024
ALLOWED_IMAGE_SIZES = (64, 128, 256, 512, 1024, 2048)
MIN_RUNTIME_VERSION = "1.17"


class ManifestError(ValueError):
    """清单缺失、格式不符或模型文件校验失败。"""


def _require_text(manifest, field, limit=200):
    value = manifest.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"模型清单缺少字段：{field}")
    if len(value) > limit:
        raise ManifestError(f"模型清单字段过长：{field}")
    return value.strip()


def _require_digest(manifest):
    digest = _require_text(manifest, "sha256", limit=64).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ManifestError("模型摘要应为 64 位十六进制字符串")
    return digest


def _require_input(manifest):
    settings = manifest.get("input")
    if not isinstance(settings, dict):
        raise ManifestError("模型清单缺少 input 段")
    size = settings.get("image_size", DEFAULT_IMAGE_SIZE)
    if not isinstance(size, int) or isinstance(size, bool) or size not in ALLOWED_IMAGE_SIZES:
        raise ManifestError(f"模型输入尺寸应为 {ALLOWED_IMAGE_SIZES} 之一")
    if settings.get("channels", 1) != 1:
        raise ManifestError("模型输入契约固定为单通道灰度时频图")
    nfft = settings.get("spectrogram_nfft", DEFAULT_NFFT)
    if not isinstance(nfft, int) or isinstance(nfft, bool) or not MIN_NFFT <= nfft <= MAX_NFFT:
        raise ManifestError(f"时频图 nfft 应为 {MIN_NFFT}～{MAX_NFFT} 之间的整数")
    dynamic = settings.get("dynamic_range_db", DEFAULT_DYNAMIC_RANGE_DB)
    if not isinstance(dynamic, (int, float)) or isinstance(dynamic, bool) \
            or not MIN_DYNAMIC_RANGE_DB <= float(dynamic) <= MAX_DYNAMIC_RANGE_DB:
        raise ManifestError("时频图动态范围应为 10～120 dB")
    layout = settings.get("layout", IMAGE_LAYOUT)
    if layout != IMAGE_LAYOUT:
        raise ManifestError(f"暂不支持的时频图排布：{layout}")
    name = settings.get("name") or "images"
    if not isinstance(name, str) or not name.strip():
        raise ManifestError("模型输入名称不合法")
    return {"name": name.strip(), "image_size": size, "channels": 1,
            "spectrogram_nfft": int(nfft), "dynamic_range_db": float(dynamic),
            "layout": IMAGE_LAYOUT}


def _require_output(manifest, labels):
    settings = manifest.get("output") or {}
    if not isinstance(settings, dict):
        raise ManifestError("模型清单的 output 段应为对象")
    layout = settings.get("layout", OUTPUT_LAYOUT)
    if layout != OUTPUT_LAYOUT:
        raise ManifestError(f"暂不支持的解码方式：{layout}")
    name = settings.get("name") or "detections"
    if not isinstance(name, str) or not name.strip():
        raise ManifestError("模型输出名称不合法")
    return {"name": name.strip(), "layout": OUTPUT_LAYOUT, "labels": list(labels)}


def _require_labels(manifest):
    labels = manifest.get("labels")
    if labels is None:
        labels = ["emitter"]
    if not isinstance(labels, (list, tuple)) or not labels:
        raise ManifestError("labels 应为非空字符串列表")
    cleaned = []
    for label in labels:
        if not isinstance(label, str) or not label.strip() or len(label) > 100:
            raise ManifestError("labels 中存在非法类别名")
        cleaned.append(label.strip())
    return cleaned


def read_model_manifest(path):
    """校验清单并返回 ``(manifest, 模型文件绝对路径)``。"""
    path = Path(path)
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"模型清单不可读：{exc}") from exc
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError("模型清单超过 64 KiB")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"模型清单不是合法 JSON：{exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("模型清单应为 JSON 对象")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(f"不支持的清单版本：{manifest.get('schema_version')!r}")
    if manifest.get("contract", INPUT_CONTRACT) != INPUT_CONTRACT:
        raise ManifestError(f"不支持的输入契约：{manifest.get('contract')!r}")
    if manifest.get("runtime", RUNTIME) != RUNTIME:
        raise ManifestError(f"不支持的推理运行时：{manifest.get('runtime')!r}")
    identifier = _require_text(manifest, "id")
    version = _require_text(manifest, "version", limit=40)
    digest = _require_digest(manifest)
    labels = _require_labels(manifest)
    input_contract = _require_input(manifest)
    output_contract = _require_output(manifest, labels)
    relative = Path(_require_text(manifest, "library"))
    if relative.is_absolute():
        raise ManifestError("模型文件必须使用清单目录内的相对路径")
    try:
        library = (path.parent / relative).resolve(strict=True)
    except OSError as exc:
        raise ManifestError(f"模型文件不可读：{exc}") from exc
    if not library.is_relative_to(path.parent):
        raise ManifestError("模型文件路径越过清单目录")
    if not library.is_file() or library.stat().st_size == 0:
        raise ManifestError("模型文件缺失或为空")
    if file_digest(library) != digest:
        raise ManifestError("模型摘要与清单不符，请重新生成清单")
    resolved = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": identifier,
        "version": version,
        "runtime": RUNTIME,
        "runtime_min_version": str(manifest.get("runtime_min_version", MIN_RUNTIME_VERSION)),
        "contract": INPUT_CONTRACT,
        "library": str(library.relative_to(path.parent)),
        "sha256": digest,
        "opset": int(manifest.get("opset", 0) or 0),
        "input": input_contract,
        "output": output_contract,
        "labels": labels,
        "training": manifest.get("training") if isinstance(manifest.get("training"), dict) else {},
        "notes": manifest.get("notes") if isinstance(manifest.get("notes"), str) else "",
        "manifest_path": str(path),
    }
    return resolved, library


def write_model_manifest(output, model, *, identifier, version,
                         image_size=DEFAULT_IMAGE_SIZE, opset=0, labels=None,
                         training=None, notes="", spectrogram_nfft=DEFAULT_NFFT,
                         dynamic_range_db=DEFAULT_DYNAMIC_RANGE_DB, extra=None):
    """为已有 ``.onnx`` 文件生成清单（训练脚本与命令行共用）。"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = Path(model).resolve(strict=True)
    if not model.is_relative_to(output.resolve().parent):
        raise ManifestError("模型文件必须位于清单目录内，请先移动或复制模型")
    if model.suffix.lower() != ".onnx":
        raise ManifestError("模型文件应为 .onnx")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ManifestError("模型标识不能为空")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "id": identifier.strip(),
        "version": str(version).strip() or "0.1.0",
        "runtime": RUNTIME,
        "runtime_min_version": MIN_RUNTIME_VERSION,
        "contract": INPUT_CONTRACT,
        "library": str(model.relative_to(output.resolve().parent)),
        "sha256": file_digest(model),
        "opset": int(opset or 0),
        "input": {"name": "images", "image_size": int(image_size), "channels": 1,
                  "spectrogram_nfft": int(spectrogram_nfft),
                  "dynamic_range_db": float(dynamic_range_db), "layout": IMAGE_LAYOUT},
        "output": {"name": "detections", "layout": OUTPUT_LAYOUT},
        "labels": list(labels or ["emitter"]),
        "training": dict(training or {}),
        "notes": str(notes or ""),
        "generated": {"python": platform.python_version(), "platform": platform.platform(),
                      "machine": platform.machine(), "bits": struct.calcsize("P") * 8},
    }
    if extra:
        manifest.update(extra)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    # 自校验：新写的清单必须能被自己的读取器接受
    read_model_manifest(output)
    return manifest
