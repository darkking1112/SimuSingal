"""验证模块拆分后的兼容接口、共用实现及二进制实际加载位置。"""

import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from signal_analysis import _numeric, core_api
from signal_analysis import _numeric_common as common
from signal_analysis import _numeric_energy as energy
from signal_analysis import _numeric_hops as hops
from signal_analysis import _numeric_modulation as modulation
from signal_analysis import _numeric_preprocess as preprocess
from signal_analysis.ml import amc, detector, iq

MODULES = ("_numeric", "_numeric_common", "_numeric_analysis", "_numeric_iqgen",
           "_numeric_preprocess", "_numeric_modulation", "_numeric_energy", "_numeric_hops")


@pytest.mark.parametrize("name", core_api.__all__)
def test_stable_api_is_same_object_as_legacy_export(name):
    assert getattr(core_api, name) is getattr(_numeric, name)


def test_features_and_iq_share_the_same_preprocessing():
    for name in ("_validate", "_mix_and_decimate", "_inband_snr"):
        assert getattr(amc, name) is getattr(preprocess, name)
        assert getattr(iq, name) is getattr(preprocess, name)
        assert getattr(modulation, name) is getattr(preprocess, name)
    for name in ("extract_features", "feature_vector", "AMC_FEATURES", "AMC_FEATURE_CONTRACT"):
        assert getattr(amc, name) is getattr(modulation, name)
    assert len(amc.AMC_FEATURES) == 34
    assert modulation.classify_modulation is core_api.classify_modulation


def test_ai_and_traditional_detection_share_measurements():
    for name in ("_finalise_hops", "_refine_hop_bands", "_group_hop_sessions", "_hop_config"):
        assert getattr(detector, name) is getattr(hops, name)
        assert getattr(_numeric, name) is getattr(hops, name)
    assert detector._merge_sessions is energy._merge_sessions
    assert energy._stft_psd is hops._stft_psd is common._stft_psd
    assert detector._occupied_span is common._occupied_span


@pytest.mark.parametrize("name", MODULES)
def test_modules_load_from_expected_distribution(name):
    module = importlib.import_module("signal_analysis." + name)
    path = Path(module.__file__).resolve()
    assert path.parent == Path(core_api.__file__).resolve().parent
    if os.environ.get("SIMUSIGNAL_EXPECT_BINARY") == "1":
        assert path.suffix in (".pyd", ".so"), path
    else:
        assert path.suffix == ".py", path


def test_numeric_import_does_not_load_model_or_gui_layers():
    code = (
        "import importlib, sys; "
        f"[importlib.import_module('signal_analysis.' + name) for name in {MODULES!r}]; "
        "forbidden = ('signal_analysis.ml', 'torch', 'onnxruntime', 'PySide6', 'communication_sim'); "
        "assert not [name for name in sys.modules if any(name == p or name.startswith(p + '.') "
        "for p in forbidden)]"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)
