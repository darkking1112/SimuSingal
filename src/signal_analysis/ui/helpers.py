"""界面格式化辅助：频率/时长/指标格式化、资产存储格式描述、频谱镜像判断。"""

from pathlib import Path

import numpy as np


def _fmt_hz(value):
    """Format a Hz value with a readable kHz / MHz / GHz unit."""
    value = float(value)
    for unit, factor in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(value) >= factor:
            return f"{value / factor:g} {unit}"
    return f"{value:g} Hz"


def _fmt_baud(value):
    """Format a symbol rate with a readable kBd / MBd unit."""
    value = float(value)
    for unit, factor in (("MBd", 1e6), ("kBd", 1e3)):
        if abs(value) >= factor:
            return f"{value / factor:g} {unit}"
    return f"{value:g} Bd"


def _fmt_span(seconds):
    """Format a duration in ms / s."""
    seconds = float(seconds)
    return f"{seconds * 1000:g} ms" if seconds < 1.0 else f"{seconds:g} s"


def _fmt_metric(value, spec):
    """Format an optional metric; missing values (无真值/未定义) show as “--”。"""
    return "--" if value is None else format(float(value), spec)


# 导入时 assets.source 记录的是原始绝对路径，这里按后缀还原可读的格式名。
_SOURCE_FORMAT_TEXT = {".npy": "NPY", ".csv": "CSV（无表头 I,Q）",
                       ".bin": "交织 IQ 二进制", ".raw": "交织 IQ 二进制",
                       ".iq": "交织 IQ 二进制",
                       ".sigmf-meta": "SigMF 双文件", ".sigmf-data": "SigMF 双文件"}
#: 资产存储格式 → 状态栏与生成结果区的文案；键与 ``data.io.FORMAT_EXTENSIONS`` 一致。
ASSET_FORMAT_TEXT = {"npy": "NPY（complex64 复基带 IQ）", "csv": "CSV（两列 I,Q）",
                     "iq16": "交织 IQ · int16", "iq32": "交织 IQ · float32",
                     "sigmf": "SigMF 双文件"}


def _asset_format(asset):
    """资产本体格式与原始来源：内置生成的数据没有外部源文件，导入的带原路径后缀。"""
    fmt = str(asset.get("storage_format") or "npy")
    text = ASSET_FORMAT_TEXT.get(fmt) or f"{fmt} 文件"
    if fmt in ("iq16", "iq32"):
        text += "（大端）" if str(asset.get("endian")) == "big" else "（小端）"
    source = str(asset.get("source") or "")
    if source.startswith("generated:"):
        return f"工作区 {text} ← 内置生成 {source.split(':', 1)[1]}"
    suffix = Path(source).suffix.lower()
    origin = _SOURCE_FORMAT_TEXT.get(suffix) or (f"{suffix.lstrip('.')} 文件" if suffix else "未知来源")
    return f"工作区 {text} ← 导入 {origin}"


def _mirrored_spectrum(values):
    """实数记录的 PSD 严格关于 0 Hz 镜像，可据此判断负半轴是否只是重复。

    fftshift 之后索引 0 是奈奎斯特频点、索引 n/2 是直流，两者各自配对；
    其余频点成对镜像（``values[n/2 + d] == values[n/2 - d]``），因此只比较
    这两段即可，不需要再读一遍原始样本。频点为奇数或过少时无法判断，按
    双边处理。
    """
    data = np.asarray(values, dtype=np.float64).ravel()
    if data.size < 8 or data.size % 2:
        return False
    half = data.size // 2
    low = data[1:half]
    high = data[half + 1:][::-1]
    if low.size == 0 or low.size != high.size:
        return False
    # 浮点 FFT 的 k 与 N-k 走不同蝶形，镜像会有 ~1e-15 dB 量级的差异
    return bool(np.allclose(low, high, rtol=0.0, atol=1e-6))


def _comparison_line(left_label, left, right_label, right):
    """One-line side-by-side detection metrics (AI vs traditional baseline)."""
    fields = (("匹配", "matched", "g"), ("漏警", "missed", "g"), ("虚警", "false_alarm", "g"),
              ("中心 MAE", "center_mae_hz", ".1f"), ("带宽相对误差", "bandwidth_mape", ".3f"),
              ("信噪比 MAE", "snr_mae_db", ".2f"))
    parts = [f"{name} {_fmt_metric(left.get(key), spec)} / {_fmt_metric(right.get(key), spec)}"
             for name, key, spec in fields]
    return (f"并排对比（{left_label} / {right_label}）：" + " · ".join(parts))


_AMC_SOURCE_TEXT = {"builtin": "内置基线", "file": "指定模型文件", "onnx": "ONNX 分类器",
                   "inline": "内存模型"}
