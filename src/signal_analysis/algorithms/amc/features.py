"""传统调制识别特征：34 维确定性特征与固定顺序向量。

仅依赖标准库、NumPy 与其他底层数值模块，可随数值核心进行 Cython 编译。
"""

import math

import numpy as np

from ..dsp.base import (
    _EPS,
    _rounded,
)

from ...contracts.amc import (
    AMC_FEATURE_CONTRACT,
    AMC_FEATURES,
)

from ..dsp.preprocess import (
    _inband_snr,
    _mix_and_decimate,
    _validate,
)


AMC_TEMPLATE_BINS = 16


AMC_TEMPLATE_SCALE = 1.2


def _spectral_shape(work, work_rate, band):
    """从 Welch 平均谱提取带内形状特征。

    参数：work 为前处理 IQ；work_rate 为 Hz；band 为占用带宽 Hz。

    返回：(flatness, edge_ratio, peak_ratio)，三个无量纲 float。

    算法与边界：Hanning 分帧、50% 重叠，最多选 64 帧，FFT 不超过 2048 点；仅统计占用带内。平坦度为几何／算术均值比，边沿比为按绝对频率排序的最外
    10% 与中间 25%～75% 比值，峰均比为最大／均值。带内少于 4 点则取全谱，以 _EPS 防止除零。
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
    """寻找近似符号时刻的幅度局部峰位置。

    参数：magnitude 为一维幅度数组；min_distance 为最小峰间距（采样点），默认 5。

    返回：一维 int64 索引数组。

    算法与边界：3 点均值平滑后检出左侧严格上升、右侧不增的内部峰，按时间顺序保留满足间距的峰；不足 5 点或没有峰时返回空数组。
    """
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
    """生成按总计数归一化的峰值幅度直方图模板。

    参数：amplitudes 为幅度序列；bins 为桶数，默认 16；extent 为归一化幅度上限，默认 1.2。

    返回：长度 bins 的实数概率质量数组，总和为 1。

    算法与边界：用 99 分位幅度作参照，将幅度比裁到 [0,extent]，直方图除以总计数（不除以桶宽）。不足 8 样本、参照非正或计数为零时返回均匀概率
    1/bins；这是中立回退，不代表观察到均匀分布。
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
    """计算近似符号时刻的幅度与累积量特征。

    参数：work 为前处理 IQ；magnitude 为对应归一化幅度，长度相同。

    返回：(peak_cv, peak_c63, peak_clusters, peak_count, template)，最后一项为 16 桶模板。

    算法与边界：按局部幅度峰取 IQ，计算幅度变异系数、去均值单位功率后的 |C63|、幅度峰数及模板。峰少于 16 个时返回 0、0、1、实际峰数，并对全部
    magnitude 生成模板。
    """
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


def _histogram_modes(values, bins=32):
    """估计归一化幅度直方图中的峰数。

    参数：values 为非空非负幅度数组；bins 为桶数，默认 32。

    返回：范围 1～8 的 float 峰数。

    算法与边界：按最大幅度归一化后统计 [0,1] 的直方图并 3 点平滑，候选峰达到最高值的 20% 且间隔至少 2 桶；零幅度或无有效峰返回 1。
    """
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
    """提取固定顺序的 34 维传统调制特征。

    参数：samples 为 IQ；sample_rate 为 Hz；offset_hz 为占用频带中心偏移 Hz；bandwidth_hz 为占用带宽 Hz，None
    表示整段采样带宽并将中心置零。

    返回：(features, info)：按 AMC_FEATURES 顺序的有限浮点字典，以及频带、抽取率、样本数、功率和 SNR 粗估信息。

    算法与边界：中心裁剪、混频低通抽取后，提取包络、谱形、相位／瞬时频率、高阶累积量、峰值及直方图特征，按 6 位小数舍入。无法粗估 SNR 时特征使用 60 dB 哨兵而
    info 保留 None；无效参数、有效样本不足或非正功率报错。amc_feature_vector_v1 的 34 维顺序冻结；见传统特征与启发式判定设计。
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
    """将特征字典或序列转换为契约顺序的数值向量。

    参数：features 为包含 AMC_FEATURES 全部字段的字典，或长度恰为 34 的序列。

    返回：按 AMC_FEATURES 排列的 Python float 列表。

    算法与边界：字典按固定名称索引，序列保留原序；缺字段、序列长度错误或非有限值抛出 ValueError。保留字典额外字段被忽略的现有行为，不在此标准化。
    """
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
