"""Explicit offline file formats. No format guessing for unlabelled binaries."""

from pathlib import Path
import numpy as np

from .core_api import validate_samples

MAX_FILE_BYTES = 64 * 1024 * 1024


def read_samples(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("文件不存在、为空或超过基础版 64 MiB 限制")
    if path.suffix.lower() == ".npy":
        samples = np.load(path, allow_pickle=False, mmap_mode="r")
    elif path.suffix.lower() == ".csv":
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape[1] not in (1, 2):
            raise ValueError("CSV 应为无表头的一列实数或两列 I,Q 数据")
        samples = values[:, 0]
        if values.shape[1] == 2:
            samples = samples + 1j * values[:, 1]
    else:
        raise ValueError("基础版只支持 .npy 和无表头 .csv 文件")
    return validate_samples(samples)
