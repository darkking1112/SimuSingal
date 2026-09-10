"""Explicit offline file formats. No format guessing for unlabelled binaries."""

from pathlib import Path
import numpy as np

from .core_api import validate_samples
from .sigmf_io import SIGMF_EXTENSIONS, read_sigmf, write_sigmf

MAX_FILE_BYTES = 512 * 1024 * 1024

IQ_EXTENSIONS = (".bin", ".raw", ".iq")
IQ_DTYPES = ("int16", "float32")
IQ_ENDIANS = ("little", "big")

_EXTENSIONS = {"npy": ".npy", "csv": ".csv", "iq16": ".bin", "iq32": ".bin",
               "sigmf": ".sigmf-meta"}


def _check_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("文件不存在、为空或超过基础版 512 MiB 限制")
    return path


def read_samples(path, *, binary_dtype=None, endian="little"):
    """Read NPY, CSV, explicitly-typed IQ binaries, or a SigMF pair.

    Interleaved IQ binaries (``.bin`` / ``.raw`` / ``.iq``) are never guessed
    from the extension: the caller must pass ``binary_dtype`` (int16 or
    float32) and optionally ``endian`` (little or big).
    """
    path = _check_file(path)
    suffix = path.suffix.lower()
    if suffix in SIGMF_EXTENSIONS:
        return read_sigmf(path)[0]
    if suffix == ".npy":
        samples = np.load(path, allow_pickle=False, mmap_mode="r")
    elif suffix == ".csv":
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape[1] not in (1, 2):
            raise ValueError("CSV 应为无表头的一列实数或两列 I,Q 数据")
        samples = values[:, 0]
        if values.shape[1] == 2:
            samples = samples + 1j * values[:, 1]
    elif suffix in IQ_EXTENSIONS:
        samples = read_iq_binary(path, binary_dtype, endian)
    else:
        raise ValueError("支持 .npy、无表头 .csv、交织 IQ（.bin/.raw/.iq）及 SigMF 双文件")
    return validate_samples(samples)


def read_iq_binary(path, dtype, endian="little"):
    """Read interleaved I/Q binary (int16 scaled by 1/32768, or float32)."""
    path = _check_file(path)
    if dtype not in IQ_DTYPES:
        raise ValueError(f"二进制 IQ 文件必须通过 binary_dtype 显式指定数据类型：{'、'.join(IQ_DTYPES)}")
    if endian not in IQ_ENDIANS:
        raise ValueError(f"字节序必须显式指定：{'、'.join(IQ_ENDIANS)}")
    raw_dtype = np.dtype(dtype).newbyteorder(endian)
    bytes_per_sample = raw_dtype.itemsize * 2
    if path.stat().st_size % bytes_per_sample:
        raise ValueError(f"文件字节数与 {dtype} 交织 I/Q 长度不匹配，文件可能被截断")
    raw = np.fromfile(path, dtype=raw_dtype)
    with np.errstate(invalid="ignore"):
        real = raw[0::2].astype(np.float64)
        imag = raw[1::2].astype(np.float64)
    samples = real + 1j * imag
    if dtype == "int16":
        samples = samples / 32768.0
    return samples


def write_samples(path, samples, fmt, endian="little", *, sample_rate=None,
                  description="", generation=None):
    """Write IQ as NPY / CSV / binary / official-library SigMF pairs.

    Single-file writes are atomically replaced; SigMF uses its pair adapter. int16
    output scales 1.0 to 32767 and clips out-of-range samples.
    """
    if fmt == "sigmf":
        return write_sigmf(path, samples, sample_rate, description, generation)
    data = validate_samples(samples)
    if fmt not in _EXTENSIONS:
        raise ValueError(f"不支持的文件格式：{fmt}，可选：{', '.join(_EXTENSIONS)}")
    path = Path(path).expanduser()
    if not path.suffix:
        path = path.with_suffix(_EXTENSIONS[fmt])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        if fmt == "npy":
            with temporary.open("wb") as stream:
                np.save(stream, data, allow_pickle=False)
        elif fmt == "csv":
            np.savetxt(temporary, np.column_stack((data.real, data.imag)),
                       delimiter=",", fmt="%.9g", header="", comments="")
        else:
            if endian not in IQ_ENDIANS:
                raise ValueError(f"字节序必须为：{'、'.join(IQ_ENDIANS)}")
            item_dtype = np.dtype("int16" if fmt == "iq16" else "float32").newbyteorder(endian)
            interleaved = np.empty(2 * data.size, dtype=item_dtype)
            if fmt == "iq16":
                scaled = np.clip(np.rint(np.stack((data.real, data.imag), axis=-1) * 32768.0),
                                 -32767, 32767).astype(np.int16)
                interleaved[0::2] = scaled[:, 0]
                interleaved[1::2] = scaled[:, 1]
            else:
                interleaved[0::2] = data.real.astype(np.float32)
                interleaved[1::2] = data.imag.astype(np.float32)
            interleaved.tofile(temporary)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path
