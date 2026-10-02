"""Scoring shared by every detection/recognition algorithm in the project.

Pure NumPy with no GUI, storage or workspace imports, so the CLI, the GUI
comparison page, the offline benchmarks and the optional training scripts
all score results with exactly the same code. This package also owns the
**frozen detection result contract** used by the traditional detector
(:func:`signal_analysis.algorithms.detection.energy.detect_signals`) and by the optional ONNX
detector in :mod:`signal_analysis.algorithms.detection.ai`.

Contract (``detect_result_v1``), written into every ``detect`` run::

    {"algorithm": "energy_detect_v1",
     "config": {"nfft": 512, "threshold_db": 6.0, "min_bandwidth_hz": 5859.4},
     "noise_floor_dbfs_per_hz": -71.3,
     "snr_definition": "inband_snr_v1",
     "detections": [{"id": 1, "method": "energy",
                     "center_hz": 100000.0, "bandwidth_hz": 190000.0,
                     "f_low_hz": 5000.0, "f_high_hz": 195000.0,
                     "t_start_s": 0.0, "t_end_s": 0.2,
                     "power_dbfs": -10.1, "snr_db": 18.4,
                     "session_id": 0, "confidence": 0.92}],
     "truth": [...], "metrics": {...}}

Naming convention (unchanged from the rest of the repository): IQ data is
complex baseband, so ``center_hz`` is a **baseband frequency offset**, never
an RF carrier. For SSB the occupied band is one-sided, therefore the
reported ``center_hz`` is the centre of the occupied band and
``nominal_offset_hz`` (present in truth entries) keeps the generator's
nominal point.

Truth entries come straight from the IQ generator summary, which is stored
in ``asset_metadata['generation']``; the generator already reports in-band
SNR with the same ``inband_snr_v1`` convention, so detector output and
truth are directly comparable.

实现分三块，包根保持 ``signal_analysis.evaluation.*`` 的旧导入面：

* :mod:`.truth` — 生成器摘要 → 会话级 / 逐跳真值；
* :mod:`.metrics` — 匹配与检测/分类指标（``detect_result_v1``）；
* :mod:`.hops` — 逐跳精确评估（``per_hop_eval_v1``）。
"""

from .truth import _finite_or_none, _rounded, hop_truth, signal_truth
from .metrics import (CONTRACT_VERSION, DETECT_ALGORITHM, SNR_DEFINITION,
                      classification_metrics, evaluate_detections, match_detections)
from .hops import (DEFAULT_HOP_GROUPS, HOP_ALGORITHM, HOP_CONTRACT, PER_HOP_CONTRACT,
                   evaluate_hop_tracks, match_hop_tracks)

__all__ = [
    "CONTRACT_VERSION", "DETECT_ALGORITHM", "SNR_DEFINITION",
    "HOP_CONTRACT", "HOP_ALGORITHM", "PER_HOP_CONTRACT", "DEFAULT_HOP_GROUPS",
    "signal_truth", "hop_truth", "match_detections", "evaluate_detections",
    "classification_metrics", "match_hop_tracks", "evaluate_hop_tracks",
    "_finite_or_none", "_rounded",
]
