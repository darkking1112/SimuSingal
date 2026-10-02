"""源码与二进制构建共用的稳定接口；按职责导入数值实现，不引入 GUI。"""

from .algorithms.dsp.spectrum import (
    analyze,
    spectrum_row,
)

from .algorithms.dsp.base import (
    MAX_SAMPLES,
    validate_rate,
    validate_samples,
)

from .algorithms.detection.energy import (
    detect_signals,
)

from .algorithms.detection.hops import (
    detect_hops,
)

from .algorithms.generation.iqgen import (
    MODE_NAMES,
    generate_iq,
    make_demo,
    occupied_interval,
    plan_signal,
)

from .algorithms.amc.heuristic import (
    classify_modulation,
)

__all__ = ["MAX_SAMPLES", "MODE_NAMES", "analyze", "classify_modulation", "detect_hops",
           "detect_signals", "generate_iq", "make_demo", "occupied_interval", "plan_signal",
           "spectrum_row", "validate_rate", "validate_samples"]
