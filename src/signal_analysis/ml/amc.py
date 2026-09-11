"""A09 六类调制识别（AMC）：确定性 NumPy 特征 + 线性判别模型。

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
from ..core_api import classify_modulation

AMC_CLASSES = ("fm", "ssb", "ask2", "qpsk", "qam16", "qam64")
CLASS_LABELS = {
    "fm": "FM 调频",
    "ssb": "SSB 单边带",
    "ask2": "2ASK 幅度键控",
    "qpsk": "QPSK 四相键控",
    "qam16": "16QAM",
    "qam64": "64QAM",
}
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
    "c63_mag",             # |C63|：QPSK≈4，16QAM≈2.1，64QAM≈1.8
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
AMC_TEMPLATE_BINS = 16
AMC_TEMPLATE_SCALE = 1.2
#: 标准化尺度下限：直方图空桶等“近乎常量”的特征不放大噪声
STANDARDIZE_FLOOR = 1e-2
AMC_FEATURE_CONTRACT = "amc_feature_vector_v1"
AMC_MODEL_CONTRACT = "amc_model_v1"
AMC_ONNX_CONTRACT = "amc_feature_vector_v1"
AMC_RESULT_CONTRACT = "amc_classify_v1"
AMC_ONNX_INPUT = "features"
AMC_ONNX_OUTPUT = "scores"
AMC_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MODEL_NAME = "amc_default.json"
DEFAULT_MODEL_ID = "amc-linear-default"
MIN_RUNTIME_VERSION = "1.17"

#: 每单位占用带宽保留的采样点数（决定"每符号点数"的量级，须与训练一致）
SAMPLES_PER_BAND = 8.0
MIN_WORK_SAMPLES = 256
MAX_ANALYSIS_SAMPLES = 1 << 20
LOWPASS_TAPS = 65
DEFAULT_L2 = 1e-3
LOW_SNR_WARNING_DB = 5.0
_TEMPERATURE_GRID = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
_EPS = 1e-30


class ModelError(ValueError):
    """AMC 模型文件缺失、格式不符或校验失败。"""


def mode_to_class(mode):
    """生成器 ``mode`` → A09 类别名；六类之外的样式返回 ``None``。"""
    text = str(mode)
    return text if text in AMC_CLASSES else None


def _rounded(value, digits=6):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------


def _validate(samples):
    data = np.asarray(samples)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("需要一维非空数组")
    if data.dtype.kind not in "iufc":
        raise ValueError("只支持实数或复数数值数组")
    view = np.asarray(data, dtype=np.complex128)
    if view.size > MAX_ANALYSIS_SAMPLES:
        start = (view.size - MAX_ANALYSIS_SAMPLES) // 2
        view = view[start:start + MAX_ANALYSIS_SAMPLES]
    if not (np.isfinite(view.real).all() and np.isfinite(view.imag).all()):
        raise ValueError("数据包含 NaN 或 Inf")
    if float(np.mean(np.abs(view) ** 2)) <= 0.0:
        raise ValueError("数据功率为零，无法提取调制特征")
    return view


def _lowpass_taps(ratio, count):
    """汉明窗 sinc 低通（``ratio = 截止频率 / 采样率``，已归一化直流增益）。"""
    cutoff = float(min(max(ratio, 1.0 / count), 0.5))
    center = (count - 1) / 2.0
    offsets = np.arange(count, dtype=np.float64) - center
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
    taps *= np.hamming(count)
    total = float(taps.sum())
    if abs(total) < _EPS:
        raise ValueError("低通滤波器设计失败")
    return taps / total


def _mix_and_decimate(view, sample_rate, center_hz, bandwidth_hz):
    """把占据 ``[center±bandwidth/2]`` 的信号搬到零频，低通后按带宽抽取。"""
    count = view.size
    # ``np.convolve(mode="same")`` 的输出长度取两输入的最大值：记录比滤波器还短时
    # 会被"补齐"成滤波器长度，等于凭空造出样本，因此这里必须先拒绝。
    if count < LOWPASS_TAPS:
        raise ValueError(f"有效样本过少，无法提取调制特征（至少需要 {LOWPASS_TAPS} 个采样点）")
    step = float(center_hz) / sample_rate
    frac = np.mod(step * np.arange(count, dtype=np.float64), 1.0)
    baseband = view * np.exp(-2j * np.pi * frac)
    taps = _lowpass_taps(bandwidth_hz / (2.0 * sample_rate), LOWPASS_TAPS)
    filtered = np.convolve(baseband, taps, mode="same")
    factor = max(1, int(math.floor(sample_rate / (SAMPLES_PER_BAND * bandwidth_hz))))
    factor = max(1, min(factor, max(1, filtered.size // MIN_WORK_SAMPLES)))
    work = filtered[::factor]
    work_rate = sample_rate / factor
    edge = min((taps.size // 2) // factor + 1, work.size // 10)
    if edge and work.size - 2 * edge >= 32:
        work = work[edge:work.size - edge]
    return work, work_rate, factor


def _spectral_shape(work, work_rate, band):
    """Welch 平均功率谱的形状特征：(平坦度, 边沿比, 峰均比)。

    只统计占用带宽内部的谱线：低通阻带会下溢到 0，若计入会把几何平均拉垮。
    单边带信号在"以占用频带中心为参考"的表示下是矩形谱，因此由平坦度与
    边沿比体现（这里不使用载波相位信息，见模块说明）。
    """
    count = work.size
    nfft = 64
    while nfft * 4 <= count and nfft < 2048:
        nfft *= 2
    nfft = max(64, min(nfft, 2048, count))
    hop = max(1, nfft // 2)
    starts = list(range(0, max(1, count - nfft + 1), hop))
    if len(starts) > 64:
        starts = [starts[int(round(position * (len(starts) - 1) / 63.0))] for position in range(64)]
    window = np.hanning(nfft)
    scale = work_rate * float(np.sum(window ** 2))
    total = np.zeros(nfft, dtype=np.float64)
    for start in starts:
        segment = work[start:start + nfft]
        if segment.size < nfft:
            segment = np.pad(segment, (0, nfft - segment.size))
        # 复数输入在某些 NumPy 构建上不能直接 rfft，这里取完整频谱。
        total += np.abs(np.fft.fft(segment * window)) ** 2
    psd = total / (max(len(starts), 1) * max(scale, _EPS))
    frequencies = np.fft.fftfreq(nfft, 1.0 / work_rate)
    inside = np.flatnonzero(np.abs(frequencies) <= 0.5 * band)
    if inside.size < 4:
        inside = np.arange(nfft)
    in_band = np.maximum(psd[inside], _EPS)
    flatness = float(np.exp(np.mean(np.log(in_band))) / np.mean(in_band))
    peak_ratio = float(np.max(in_band) / max(float(np.mean(in_band)), _EPS))
    ranked = inside[np.argsort(np.abs(frequencies[inside]))]
    size = ranked.size
    outer = ranked[int(0.9 * size):]
    middle = ranked[int(0.25 * size):int(0.75 * size)]
    if outer.size == 0:
        outer = ranked[-1:]
    if middle.size == 0:
        middle = ranked
    edge_ratio = float(np.mean(psd[outer]) / max(float(np.mean(psd[middle])), _EPS))
    return flatness, edge_ratio, peak_ratio


def _peak_samples(magnitude, min_distance=5):
    """近似符号时刻的局部极大值位置（对幅度做 3 点平滑后取峰）。"""
    if magnitude.size < 5:
        return np.zeros(0, dtype=np.int64)
    smooth = np.convolve(magnitude, np.ones(3) / 3.0, mode="same")
    interior = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])
    locations = np.flatnonzero(interior) + 1
    if locations.size == 0:
        return locations.astype(np.int64)
    kept = [int(locations[0])]
    for index in locations[1:]:
        if index - kept[-1] >= min_distance:
            kept.append(int(index))
    return np.asarray(kept, dtype=np.int64)


def _amplitude_template(amplitudes, bins=AMC_TEMPLATE_BINS, extent=AMC_TEMPLATE_SCALE):
    """峰值幅度归一化直方图模板（桶内归一化密度，和为 1）。

    参照点用 99 分位幅度（对噪声尖峰不敏感），超出区间的值截断到边界；
    样本太少时返回均匀分布（对模型近似中立，不会变成虚假证据）。
    """
    values = np.asarray(amplitudes, dtype=np.float64)
    if values.size < 8:
        return np.full(bins, 1.0 / bins)
    reference = float(np.percentile(values, 99.0))
    if not reference > 0.0:
        return np.full(bins, 1.0 / bins)
    histogram, _ = np.histogram(np.clip(values / reference, 0.0, extent),
                               bins=bins, range=(0.0, extent))
    total = float(histogram.sum())
    if not total > 0.0:
        return np.full(bins, 1.0 / bins)
    return histogram / total


def _peak_features(work, magnitude):
    """峰值采样（近似符号时刻）幅度统计：变异系数、|C63|、直方图峰数、模板。"""
    locations = _peak_samples(magnitude)
    if locations.size < 16:
        return 0.0, 0.0, 1.0, int(locations.size), _amplitude_template(magnitude)
    picked = work[locations]
    amplitudes = np.abs(picked)
    mean_amplitude = float(np.mean(amplitudes))
    peak_cv = float(np.std(amplitudes)) / max(mean_amplitude, 1e-12)
    centered = (picked - np.mean(picked)) / math.sqrt(float(np.mean(np.abs(picked) ** 2)))
    m21 = float(np.mean(np.abs(centered) ** 2))
    m42 = float(np.mean(np.abs(centered) ** 4))
    m63 = float(np.mean(np.abs(centered) ** 6))
    peak_c63 = abs(m63 - 9.0 * m42 * m21 + 12.0 * m21 ** 3)
    peak_clusters = _histogram_modes(amplitudes)
    template = _amplitude_template(amplitudes)
    return peak_cv, peak_c63, peak_clusters, int(locations.size), template


def _inband_snr(view, sample_rate, center_hz, bandwidth_hz):
    """带内信噪比粗估（dB）：占用带内平均功率谱密度对带外噪声底。

    这是工程粗估：假定占用带外只有噪声（噪声带宽覆盖采样带宽），若噪声本身
    也局限在带内则会被高估，结果中必须按“粗估”引用（见 :func:`amc_classify`）。
    """
    count = int(min(view.size, 1 << 16))
    segment = view[:count]
    if count < 64:
        return None
    window = np.hanning(count)
    spectrum = np.abs(np.fft.fft(segment * window)) ** 2
    frequencies = np.fft.fftfreq(count, 1.0 / float(sample_rate))
    # center_hz 可能落在采样带宽之外，用环形距离衡量
    shift = float(center_hz) - float(sample_rate) / 2.0
    wrapped = np.mod(frequencies - shift, float(sample_rate)) + shift
    distance = np.abs(wrapped - float(center_hz))
    inside = distance <= float(bandwidth_hz) / 2.0
    if int(inside.sum()) < 4 or int((~inside).sum()) < 4:
        return None
    in_band = float(np.mean(spectrum[inside]))
    out_band = float(np.mean(spectrum[~inside]))
    if not out_band > 0.0 or not in_band > out_band:
        return 0.0
    return 10.0 * math.log10((in_band - out_band) / out_band)


def _histogram_modes(values, bins=32):
    """幅度直方图峰数（1～8），用于区分恒模与多电平幅度键控。"""
    peak = float(np.max(values))
    if not peak > 0:
        return 1.0
    histogram, _ = np.histogram(values / peak, bins=bins, range=(0.0, 1.0))
    smoothed = np.convolve(histogram.astype(np.float64), np.ones(3) / 3.0, mode="same")
    top = float(np.max(smoothed))
    if top <= 0:
        return 1.0
    threshold = 0.2 * top
    modes = []
    for index in range(1, bins - 1):
        if smoothed[index] >= threshold and smoothed[index] >= smoothed[index - 1] \
                and smoothed[index] > smoothed[index + 1]:
            if not modes or index - modes[-1] >= 2:
                modes.append(index)
    return float(min(max(len(modes), 1), 8))


def extract_features(samples, sample_rate, offset_hz=0.0, bandwidth_hz=None):
    """提取 A09 六类识别特征。

    返回 ``(features, info)``：``features`` 为 :data:`AMC_FEATURES` 顺序的
    ``{名称: 有限浮点数}``；``info`` 给出频带、抽取比、多项式量，用于复核。

    ``offset_hz`` 是信号占用的**中心频率**（相对基带的频率偏移），
    ``bandwidth_hz`` 是占用带宽；省略带宽时按整个采样带宽处理（不做抽取）。
    """
    if isinstance(sample_rate, bool) or not np.isfinite(float(sample_rate)) \
            or float(sample_rate) <= 0:
        raise ValueError("采样率必须为正的有限数值")
    sample_rate = float(sample_rate)
    view = _validate(samples)
    center = float(offset_hz) if offset_hz else 0.0
    if not np.isfinite(center) or abs(center) > sample_rate:
        raise ValueError("中心频率必须为有限数值且不超过采样率")
    if bandwidth_hz is None:
        band = sample_rate
        center = 0.0
    else:
        if not np.isfinite(float(bandwidth_hz)):
            raise ValueError("占用带宽必须为有限数值")
        band = float(bandwidth_hz)
        if not 0.0 < band <= sample_rate:
            raise ValueError("占用带宽必须大于 0 且不超过采样率")
    work, work_rate, factor = _mix_and_decimate(view, sample_rate, center, band)
    if work.size < 32:
        raise ValueError("有效样本过少，无法提取调制特征")

    mean_power = float(np.mean(np.abs(work) ** 2))
    if not mean_power > 0:
        raise ValueError("频带内功率为零，无法提取调制特征")
    root = math.sqrt(mean_power)
    magnitude = np.abs(work) / root
    centered = (work - np.mean(work)) / root

    mean_magnitude = float(np.mean(magnitude))
    env_cv = float(np.std(magnitude)) / max(mean_magnitude, 1e-12)
    envelope = magnitude - mean_magnitude
    second = float(np.mean(envelope ** 2))
    fourth = float(np.mean(envelope ** 4))
    env_kurtosis = fourth / max(second * second, 1e-24) - 3.0
    window = np.hanning(envelope.size)
    envelope_spectrum = np.abs(np.fft.rfft(envelope * window)) ** 2
    envelope_spectrum = envelope_spectrum[1:] if envelope_spectrum.size > 1 else envelope_spectrum
    envelope_spectrum = np.maximum(envelope_spectrum, _EPS)
    env_gamma_max_db = 10.0 * math.log10(float(np.max(envelope_spectrum))
                                         / float(np.mean(envelope_spectrum)))

    phase = np.angle(work)
    delta = np.angle(np.exp(1j * np.diff(phase))) if work.size > 1 else np.zeros(1)
    phase_diff_std = float(np.std(delta)) / math.pi
    inst = delta * (work_rate / (2.0 * math.pi * band))
    inst_mean = float(np.mean(inst))
    inst_second = float(np.mean((inst - inst_mean) ** 2))
    inst_fourth = float(np.mean((inst - inst_mean) ** 4))
    inst_freq_std = math.sqrt(inst_second)
    inst_freq_kurtosis = inst_fourth / max(inst_second * inst_second, 1e-24) - 3.0

    m20 = complex(np.mean(centered ** 2))
    m21 = float(np.mean(np.abs(centered) ** 2))
    m42 = float(np.mean(np.abs(centered) ** 4))
    m63 = float(np.mean(np.abs(centered) ** 6))
    m20_mag = abs(m20)
    c42_mag = abs(m42 - abs(m20) ** 2 - 2.0 * m21 * m21)
    c63_mag = abs(m63 - 9.0 * m42 * m21 + 12.0 * m21 ** 3)

    low, middle, high = (float(value) for value in
                         np.percentile(magnitude, [10.0, 50.0, 90.0]))
    amp_spread = (high - low) / max(middle, 1e-12)
    amp_clusters = _histogram_modes(magnitude)
    flatness, edge_ratio, peak_ratio = _spectral_shape(work, work_rate, band)
    peak_cv, peak_c63, peak_clusters, peak_count, template = _peak_features(work, magnitude)
    snr_estimate = _inband_snr(view, sample_rate, center, band)
    # 占用带几乎覆盖整个采样带宽时没有噪声参考区，特征值退化为“很高”的哨兵值，
    # 而 info 里保留 None 表示“本次无法估计”。
    snr_feature = 60.0 if snr_estimate is None else snr_estimate

    values = (env_cv, env_kurtosis, env_gamma_max_db, flatness, edge_ratio, peak_ratio,
              inst_freq_std, inst_freq_kurtosis, phase_diff_std, m20_mag, c42_mag,
              c63_mag, amp_spread, amp_clusters, peak_cv, peak_c63, peak_clusters,
              *[float(value) for value in template], snr_feature)
    features = {name: _rounded(value) for name, value in zip(AMC_FEATURES, values)}
    for name, value in features.items():
        if value is None:
            raise ValueError(f"特征 {name} 计算失败（非有限值）")
    info = {
        "center_hz": center,
        "bandwidth_hz": band,
        "sample_rate_hz": sample_rate,
        "analysis_rate_hz": work_rate,
        "decimation": factor,
        "sample_count": int(view.size),
        "analysis_samples": int(work.size),
        "peak_samples": peak_count,
        "power_dbfs": _rounded(10.0 * math.log10(mean_power)),
        "snr_estimate_db": _rounded(snr_estimate, 2),
    }
    return features, info


def feature_vector(features):
    """按 :data:`AMC_FEATURES` 顺序取特征值（dict 或序列均可）。"""
    if isinstance(features, dict):
        try:
            values = [float(features[name]) for name in AMC_FEATURES]
        except KeyError as exc:
            raise ValueError(f"特征缺少字段：{exc.args[0]}") from exc
    else:
        values = [float(value) for value in features]
        if len(values) != len(AMC_FEATURES):
            raise ValueError(f"特征向量长度应为 {len(AMC_FEATURES)}")
    if not np.isfinite(values).all():
        raise ValueError("特征向量包含 NaN 或 Inf")
    return values


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
    from ..evaluation import classification_metrics

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
# ONNX 分类器契约（可选路径，需要 onnxruntime）
# ---------------------------------------------------------------------------


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


def onnx_scores(manifest_path, features, threads=None):
    """按清单跑 ONNX 分类器，返回 ``(scores, manifest)``（概率之和为 1）。"""
    from .runtime import check_version, runtime_module

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
