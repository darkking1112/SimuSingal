"""``ascs`` 模型目录条目（纯元数据；实现在 ``training/amc_models/ascs.py``）。

来源：``custom/ASCS.py``（原文件不改，训练侧是维护副本：STFT 实数化 + 融合宽度按窗口计算）。
"""

from __future__ import annotations

from .base import ModelSpec

SPEC = ModelSpec(
    id="ascs",
    title="ASCS（ASSE + 残差卷积 + 注意力融合）",
    summary="时频变换 → 自适应频谱增强（ASSE）→ 池化/残差卷积 → 注意力融合分类",
    implementation="amc_models.ascs",
    model_revision=1,
    layers=(
        "输入 iq (B, 2, N) · 单位 RMS",
        "STFT（n_fft 16、hop 1、汉明窗、reflect 补零）→ (B, 2, 16, N+1) 实数/虚部堆叠",
        "ASSE ×2：通道注意力 + 空间注意力的频谱增强（2→16→16）",
        "Same 池化 → Inception 残差块 → Same 池化",
        "残差卷积：16→32→96→128（含 ASSE 增强与池化）",
        "展平宽度（= N/2 + 8，随窗口变）→ 注意力融合（2 头）→ 分类头 → C",
    ),
    samples="min:64",
    notes="来源 custom/ASCS.py（原实现把融合输入宽度写死 72，只对 128 点窗口成立）；"
          "训练副本按 datasets 窗口长度在构造期探测该宽度，因此不同窗口都能训练与导出。",
)
