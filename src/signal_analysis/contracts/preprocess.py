"""前段共享口径：抽取比与低通抽头数（"每符号点数"的数值契约）。

训练与推理必须逐位一致；模型清单会声明这些取值并由
:mod:`signal_analysis.contracts.iq` 逐项核对。实现见
:mod:`signal_analysis.algorithms.dsp.preprocess`。
"""

#: 每单位占用带宽保留的采样点数（决定"每符号点数"的量级，须与训练一致）
SAMPLES_PER_BAND = 8.0


MIN_WORK_SAMPLES = 256


MAX_ANALYSIS_SAMPLES = 1 << 20


LOWPASS_TAPS = 65
