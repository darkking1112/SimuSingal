"""数字／模拟调制启发式判定（``heuristic_digital_analog_v1``）。

直接处理原始 IQ 的显示用规则树，不消费 34 维特征向量；只依赖标准库与
NumPy，可随数值核心进行 Cython 编译。
"""

import numpy as np


def _marginal_modes(values):
    """估计 I 或 Q 边缘分布的独立峰数。

    参数：values 为一维实数样本，通常为已抽样 IQ 的实部或虚部。

    返回：非负整数峰数。

    算法与边界：在 ±(3std+1e-12) 范围做 64 桶直方图并 5 点平滑；候选峰需达到最高峰的 12%，且与邻近更高峰之间有不高于自身 75%
    的谷。无有效峰返回 0。
    """
    span = 3.0 * float(np.std(values)) + 1e-12
    hist, _ = np.histogram(values, bins=64, range=(-span, span))
    hist = hist.astype(np.float64)
    smoothed = np.convolve(hist, np.ones(5) / 5.0, mode="same")
    peak_max = float(smoothed.max())
    if peak_max <= 0:
        return 0
    idx = [i for i in range(1, 63)
           if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
           and smoothed[i] >= 0.12 * peak_max]
    order = sorted(idx, key=lambda i: -smoothed[i])
    keep = []
    for i in order:
        value = float(smoothed[i])
        left = max((j for j in keep if j < i), default=None)
        right = min((j for j in keep if j > i), default=None)
        if left is not None and right is not None:
            low = min(float(smoothed[left + 1:i].min()), float(smoothed[i + 1:right].min()))
        elif left is not None:
            low = min(float(smoothed[left + 1:i].min()), float(smoothed[i + 1:64].min()))
        elif right is not None:
            low = min(float(smoothed[0:i].min()), float(smoothed[i + 1:right].min()))
        else:
            low = min(float(smoothed[0:i].min()), float(smoothed[i + 1:64].min()))
        # 峰须与两侧更高峰之间存在低于 75% 峰高的谷，否则视为肩部抖动
        if low <= 0.75 * value:
            keep.append(i)
    return len(keep)


def classify_modulation(samples, max_points=20000):
    """按原始 I/Q 统计作数字／模拟启发式判定。

    参数：samples 为一维非空数值数组；max_points 为抽样预算，默认 20000，由调用方保证为正。

    返回：(classification, cluster_estimate)：analog/digital 标签及星座规模粗估；模拟返回 0。

    算法与边界：等步长抽样后校验有限性；常量或包络变异系数<0.25 判模拟；否则 I/Q 最小峰度<2.45 或两侧均多峰判数字，峰数乘积限于
    2～256。其余判模拟。这是显示用规则树，不是六类模型，也不区分 16QAM/64QAM；无效输入抛出 ValueError。见传统特征与启发式判定设计 §2.9。
    """
    data = np.asarray(samples)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("需要一维非空数组")
    if data.dtype.kind not in "iufc":
        raise ValueError("只支持实数或复数数值数组")
    step = max(1, int(np.ceil(data.size / max_points)))
    view = np.asarray(data[::step], dtype=np.complex128)
    real, imag = view.real, view.imag
    if not (np.isfinite(real).all() and np.isfinite(imag).all()):
        raise ValueError("数据包含 NaN 或 Inf")
    if float(np.std(real)) == 0.0 and float(np.std(imag)) == 0.0:
        return "analog", 0
    magnitude = np.abs(view)
    ring_ratio = float(np.std(magnitude) / max(np.mean(magnitude), 1e-12))
    if ring_ratio < 0.25:
        return "analog", 0

    def kurtosis(values):
        """计算启发式判定使用的中心四阶标准化矩。

        参数：values 为一维实数边缘样本。

        返回：float 峰度，正态分布参考值为 3，未减去 3。

        算法与边界：先减均值，计算 mean(v⁴)/max(mean(v²)²,1e-24)，以分母下限避免常量输入除零。
        """
        values = values - values.mean()
        return float(np.mean(values ** 4) / max(float(np.mean(values ** 2)) ** 2, 1e-24))

    ki = kurtosis(real) if float(np.std(real)) > 1e-12 else 99.0
    kq = kurtosis(imag) if float(np.std(imag)) > 1e-12 else 99.0
    modes_i = _marginal_modes(real) if float(np.std(real)) > 1e-12 else 1
    modes_q = _marginal_modes(imag) if float(np.std(imag)) > 1e-12 else 1
    estimate = int(max(2, min(modes_i * modes_q, 256)))
    if min(ki, kq) < 2.45:
        return "digital", estimate
    if modes_i >= 2 and modes_q >= 2:
        return "digital", estimate
    return "analog", 0
