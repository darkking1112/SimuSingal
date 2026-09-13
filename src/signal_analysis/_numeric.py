"""旧数值入口的兼容层。

算法按职责位于 ``_numeric_*`` 模块；此处显式保留原函数与常量的导入路径，
不复制算法实现。对外稳定接口仍由 ``core_api`` 提供。
"""

import numpy as np  # 保留旧模块的 NumPy 属性，供已有调用方兼容。

from ._numeric_analysis import (
    analyze,
    spectrum_row,
)

from ._numeric_common import (
    DETECT_SNR_DEFINITION,
    MAX_SAMPLES,
    _OCCUPIED_RATIO,
    _SESSION_GAP_RATIO,
    _SNR_FLOOR_DB,
    _binary_close,
    _binary_dilate,
    _finite,
    _noise_floor_db,
    _occupied_span,
    _stft_psd,
    _true_runs,
    validate_rate,
    validate_samples,
)

from ._numeric_energy import (
    DETECT_ALGORITHM,
    _DETECT_DEFAULTS,
    _SESSION_OVERLAP_RATIO,
    _band_regions,
    _combine_sessions,
    _detect_config,
    _merge_sessions,
    _overlap_seconds,
    _same_session,
    detect_signals,
)

from ._numeric_hops import (
    HOP_ALGORITHM,
    HOP_CONTRACT,
    _HOP_DEFAULTS,
    _HOP_MIN_DWELL_FRAMES,
    _HOP_REFINE_NFFT,
    _HOP_SESSION_OVERLAP_TOL_FRAMES,
    _HOP_SESSION_SPLIT_RATIO,
    _best_track,
    _cluster_centres,
    _finalise_hops,
    _frame_runs,
    _group_hop_sessions,
    _hop_config,
    _link_tracks,
    _overlaps_any,
    _refine_hop_bands,
    _smooth_psd,
    _split_session,
    _track_band,
    detect_hops,
)

from ._numeric_iqgen import (
    MAX_SIGNALS,
    MODES,
    MODE_NAMES,
    _CONSTELLATIONS,
    _amplitude_modulation,
    _band_noise,
    _band_noise_at,
    _check_band,
    _fh_hop_boundaries,
    _fh_remote_control,
    _fh_video_link,
    _frequency_modulation,
    _hop_points,
    _linear_digital,
    _rrc_taps,
    _scale_power,
    _shape_pulses,
    _shift_to_offset,
    _signal_meta,
    _single_sideband,
    _synthesize,
    generate_iq,
    make_demo,
    occupied_interval,
    plan_signal,
)

from ._numeric_modulation import (
    _marginal_modes,
    classify_modulation,
)
