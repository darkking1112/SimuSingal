"""验证数值核心的模块归属、共用实现及二进制实际加载位置。"""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from signal_analysis import core_api
from signal_analysis.algorithms.amc import feature_model, features, heuristic, iq_model
from signal_analysis.algorithms.detection import ai, energy, hops, postprocess
from signal_analysis.algorithms.dsp import base as common
from signal_analysis.algorithms.dsp import preprocess, spectrum
from signal_analysis.algorithms.generation import iqgen

#: 编译清单（``packaging/numeric_core.json``）对应的八个职责模块。
MODULES = ("signal_analysis.algorithms.dsp.base",
           "signal_analysis.algorithms.dsp.spectrum",
           "signal_analysis.algorithms.dsp.preprocess",
           "signal_analysis.algorithms.generation.iqgen",
           "signal_analysis.algorithms.amc.features",
           "signal_analysis.algorithms.amc.heuristic",
           "signal_analysis.algorithms.detection.energy",
           "signal_analysis.algorithms.detection.hops")

#: core_api 的稳定导出 → 实现模块：稳定层不得退化为搬运。
_STABLE_SOURCES = {
    "MAX_SAMPLES": common, "MODE_NAMES": iqgen, "analyze": spectrum,
    "classify_modulation": heuristic, "detect_hops": hops, "detect_signals": energy,
    "generate_iq": iqgen, "make_demo": iqgen, "occupied_interval": iqgen,
    "plan_signal": iqgen, "spectrum_row": spectrum, "validate_rate": common,
    "validate_samples": common,
}


@pytest.mark.parametrize("name", core_api.__all__)
def test_stable_api_points_to_shared_implementation(name):
    assert getattr(core_api, name) is getattr(_STABLE_SOURCES[name], name)


def test_features_and_iq_share_the_same_preprocessing():
    for name in ("_validate", "_mix_and_decimate", "_inband_snr"):
        assert getattr(iq_model, name) is getattr(preprocess, name)
        assert getattr(features, name) is getattr(preprocess, name)
    for name in ("extract_features", "feature_vector", "AMC_FEATURES", "AMC_FEATURE_CONTRACT"):
        assert getattr(feature_model, name) is getattr(features, name)
    assert len(feature_model.AMC_FEATURES) == 34
    assert heuristic.classify_modulation is core_api.classify_modulation


def test_ai_and_traditional_detection_share_measurements():
    for name in ("_finalise_hops", "_refine_hop_bands", "_group_hop_sessions", "_hop_config"):
        assert getattr(ai, name) is getattr(hops, name)
    assert ai._merge_sessions is postprocess._merge_sessions
    assert energy._merge_sessions is postprocess._merge_sessions
    assert energy._stft_psd is hops._stft_psd is common._stft_psd
    assert ai._occupied_span is common._occupied_span


@pytest.mark.parametrize("name", MODULES)
def test_modules_load_from_expected_distribution(name):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    package_root = Path(core_api.__file__).resolve().parent
    assert package_root in path.parents, path
    assert (package_root / "algorithms") in path.parents, path
    if os.environ.get("SIMUSIGNAL_EXPECT_BINARY") == "1":
        assert path.suffix in (".pyd", ".so"), path
    else:
        assert path.suffix == ".py", path


def test_numeric_import_does_not_load_model_or_gui_layers():
    code = (
        "import importlib, sys; "
        f"[importlib.import_module(name) for name in {MODULES!r}]; "
        "forbidden = ('torch', 'onnxruntime', 'PySide6', 'communication_sim', "
        "'signal_analysis.inference', 'signal_analysis.ui', 'signal_analysis.services'); "
        "assert not [name for name in sys.modules if any(name == p or name.startswith(p + '.') "
        "for p in forbidden)]"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
