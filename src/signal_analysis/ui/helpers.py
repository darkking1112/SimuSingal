"""界面格式化与导出辅助：频率/时长/指标格式化、资产格式与导出物描述、频谱镜像判断。"""

from pathlib import Path

import numpy as np


def _fmt_hz(value):
    """Format a Hz value with a readable kHz / MHz / GHz unit."""
    value = float(value)
    for unit, factor in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(value) >= factor:
            return f"{value / factor:g} {unit}"
    return f"{value:g} Hz"


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
_STORAGE_FORMAT_TEXT = "工作区 NPY（complex64 复基带 IQ）"
# 导出物后缀到导出格式名的还原，键与 dataio.write_samples 的 fmt 一一对应。
_EXPORT_FORMAT_TEXT = {".sigmf-meta": "SigMF 双文件", ".sigmf-data": "SigMF 双文件",
                       ".npy": "NPY", ".csv": "CSV（无表头 I,Q）",
                       ".bin": "交织 IQ 二进制"}


def _asset_format(asset):
    """资产本体格式与原始来源：内置生成的数据没有外部源文件，导入的带原路径后缀。"""
    source = str(asset.get("source") or "")
    if source.startswith("generated:"):
        return f"{_STORAGE_FORMAT_TEXT} ← 内置生成 {source.split(':', 1)[1]}"
    suffix = Path(source).suffix.lower()
    origin = _SOURCE_FORMAT_TEXT.get(suffix) or (f"{suffix.lstrip('.')} 文件" if suffix else "未知来源")
    return f"{_STORAGE_FORMAT_TEXT} ← 导入 {origin}"


def _iq_binary_kind(size, sample_count):
    """反推交织 IQ 的量化类型：int16 每复采样 4 B、float32 为 8 B。"""
    if sample_count and size == 4 * sample_count:
        return "int16"
    if sample_count and size == 8 * sample_count:
        return "float32"
    return "类型未知"


def _asset_exports(exports_root, asset):
    """资产在 ``exports/`` 下的导出物，已格式化为“格式（exports/文件名）”。

    导出物按 ``<asset_id>.<ext>`` 命名，所以按前缀匹配即可，不递归、不依赖内存
    状态（重启后仍能显示）。SigMF 是一对文件，按一次导出计，只报元数据那一个，
    免得同一份导出在状态栏里出现两条。目录不存在或读不动时返回空列表，不影响
    状态栏其余字段。
    """
    prefix = f"{asset['id']}."
    try:
        entries = [entry for entry in Path(exports_root).iterdir()
                   if entry.is_file() and entry.name.startswith(prefix)]
    except OSError:
        return []
    names = {entry.name for entry in entries}
    sample_count = int(asset["sample_count"])
    found = []
    for entry in sorted(entries, key=lambda item: item.name):
        suffix = entry.suffix.lower()
        if suffix == ".sigmf-data":
            paired = entry.name.removesuffix(".sigmf-data") + ".sigmf-meta"
            if paired in names:
                continue  # 与同名元数据成对，按一次导出计
        label = _EXPORT_FORMAT_TEXT.get(suffix) or f"{suffix.lstrip('.') or '无后缀'} 文件"
        if suffix == ".bin":
            try:
                label += f" · {_iq_binary_kind(entry.stat().st_size, sample_count)}"
            except OSError:
                label += " · 类型未知"
        found.append(f"{label}（exports/{entry.name}）")
    return found


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
