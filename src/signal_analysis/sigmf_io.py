"""Application limits and metadata mapping around the official SigMF library."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import sigmf

from .core_api import MAX_SAMPLES, validate_rate, validate_samples

SIGMF_EXTENSIONS = (".sigmf-meta", ".sigmf-data")
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_DATA_BYTES = 512 * 1024 * 1024
SUPPORTED_TYPES = {"cf32_le": 8, "cf32_be": 8, "cf64_le": 16, "cf64_be": 16,
                   "ci16_le": 4, "ci16_be": 4}


def read_sigmf(path):
    """Return validated IQ, metadata sample rate, and original metadata.

    Only contiguous, single-channel, same-name file pairs are accepted.
    Integer scaling and binary decoding are performed by sigmf itself.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in SIGMF_EXTENSIONS:
        raise ValueError("请选择 .sigmf-meta 或 .sigmf-data 双文件之一")
    meta_path = path.with_suffix(".sigmf-meta")
    data_path = path.with_suffix(".sigmf-data")
    if not meta_path.is_file() or not 0 < meta_path.stat().st_size <= MAX_METADATA_BYTES:
        raise ValueError("SigMF 元数据文件缺失、为空或超过 4 MiB")
    if not data_path.is_file() or not 0 < data_path.stat().st_size <= MAX_DATA_BYTES:
        raise ValueError("SigMF 数据文件缺失、为空或超过 512 MiB")
    try:
        # Preflight before the library opens or maps any referenced dataset.
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        info = metadata["global"]
        datatype = info.get("core:datatype")
        if datatype not in SUPPORTED_TYPES:
            raise ValueError("SigMF 当前仅支持 cf32/cf64/ci16 的 little/big endian 复数 IQ")
        if info.get("core:num_channels", 1) != 1:
            raise ValueError("SigMF 当前仅支持单通道")
        if info.get("core:dataset", data_path.name) != data_path.name:
            raise ValueError("SigMF 当前仅支持同名双文件，不支持外部 dataset 引用")
        if any(info.get(key, 0) for key in ("core:offset", "core:trailing_bytes", "core:metadata_only")):
            raise ValueError("SigMF 当前仅支持无文件头和尾部附加数据的连续 IQ")
        if any(capture.get("core:header_bytes", 0) for capture in metadata.get("captures", [])):
            raise ValueError("SigMF 不支持带分段文件头的数据")
        if any(ext.get("required", False) for ext in info.get("core:extensions", [])):
            raise ValueError("SigMF 包含尚未支持的必需扩展")
        size = data_path.stat().st_size
        width = SUPPORTED_TYPES[datatype]
        if size % width or not 1 <= size // width <= MAX_SAMPLES:
            raise ValueError("SigMF 数据被截断或超过采样点数限制")
        rate = validate_rate(info.get("core:sample_rate"))
        recording = sigmf.fromfile(meta_path, autoscale=True)
        recording.validate()
        samples = validate_samples(recording.read_samples())
        return samples, rate, metadata
    except Exception as exc:
        raise ValueError(f"SigMF 导入失败：{exc}") from exc


def write_sigmf(path, samples, sample_rate, description="", generation=None):
    """Write cf32_le via sigmf.fromarray/tofile; publish metadata last.

    Existing pairs are never overwritten. Temporary files are kept on the
    destination filesystem and removed on failure. The final pair is copied
    exclusively, metadata last; this is not a two-file atomic transaction.
    Return the metadata path.
    """
    data = validate_samples(samples).astype("<c8")
    if sample_rate is None:
        raise ValueError("SigMF 导出必须提供采样率")
    rate = validate_rate(sample_rate)
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() in SIGMF_EXTENSIONS:
        base = path.with_suffix("")
    elif path.suffix:
        raise ValueError("SigMF 输出路径请使用无后缀名称或 .sigmf-meta/.sigmf-data")
    else:
        base = path
    meta_path = base.with_name(base.name + ".sigmf-meta")
    data_path = base.with_name(base.name + ".sigmf-data")
    if meta_path.exists() or data_path.exists():
        raise ValueError("SigMF 输出文件已存在，请使用新的文件名")
    base.parent.mkdir(parents=True, exist_ok=True)
    recording = sigmf.fromarray(data)
    recording.sample_rate = rate
    recording.description = str(description)
    if generation is not None:
        # A standard free-text field preserves the recipe without inventing core keys.
        recording.description += "\nGeneration recipe (JSON):\n" + json.dumps(
            generation, ensure_ascii=True, allow_nan=False, sort_keys=True)
    recording.add_capture(start_index=0, metadata={})
    published = []
    try:
        with TemporaryDirectory(prefix=".sigmf-", dir=base.parent) as directory:
            staged = Path(directory) / base.name
            recording.tofile(staged)
            # Exclusive creation works on removable filesystems too. Metadata
            # is copied last; the pair is only successful after both close.
            for destination in (data_path, meta_path):
                source = staged.with_name(staged.name + destination.suffix)
                with destination.open("xb") as output:
                    published.append(destination)
                    with source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
    except Exception as exc:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise ValueError(f"SigMF 导出失败：{exc}") from exc
    return meta_path
