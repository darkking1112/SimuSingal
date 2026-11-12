"""试用版编译训练内核的四项边界与身份握手。

本文件在 ``scripts/check_binary.py`` 中于“只有安装 wheel、没有仓库 training/”
的隔离目录运行；源码开发环境会跳过（edition=development）。
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from signal_analysis._training_policy import get_build_info
from signal_analysis.training import detection_runtime, iq_runtime, worker  # noqa: F401


class _FakeParameter:
    def __init__(self, count):
        self.count = int(count)

    def numel(self):
        return self.count


class _FakeModel:
    def __init__(self, count):
        self.count = int(count)

    def parameters(self):
        return [_FakeParameter(self.count)]


INFO = get_build_info()
TRIAL = INFO["edition"] == "trial"


@pytest.fixture(scope="module", autouse=True)
def _require_release_build():
    if INFO["edition"] == "development":
        pytest.skip("源码开发构建不启用试用限额")
    expected = os.environ.get("SIMUSIGNAL_EXPECT_EDITION")
    if expected and INFO["edition"] != expected:
        pytest.fail(f"wheel 实际身份 {INFO['edition']} 与请求 {expected} 不一致")
    assert INFO["policy_version"] >= 1


def _compiled(module):
    assert module.__file__.endswith((".so", ".pyd")), module.__file__


def test_policy_and_training_modules_are_compiled_without_torch():
    _compiled(sys.modules["signal_analysis._training_policy"])
    _compiled(iq_runtime)
    _compiled(detection_runtime)
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; import signal_analysis._training_policy as p; "
         "assert 'torch' not in sys.modules; print(p.get_build_info()['edition'])"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == INFO["edition"]


def test_epoch_boundaries():
    limit = INFO.get("max_epochs") or 10000
    assert iq_runtime.check_epochs(1) == 1
    assert iq_runtime.check_epochs(limit) == limit
    if TRIAL:
        with pytest.raises(ValueError, match="TRIAL_EPOCH_LIMIT"):
            iq_runtime.check_epochs(limit + 1)
    else:
        assert iq_runtime.check_epochs(10000) == 10000
        with pytest.raises(ValueError):
            iq_runtime.check_epochs(10001)
    for invalid in (0, -1, True, 1.5, "5", None):
        with pytest.raises(ValueError):
            iq_runtime.check_epochs(invalid)


def test_iq_sample_and_window_boundaries():
    train = np.zeros((2000, 2, 1024), dtype=np.float32)
    val = np.zeros((500, 2, 1024), dtype=np.float32)
    iq_runtime.guard_iq_data(train, val)  # 2000 训练 + 500 验证允许
    if TRIAL:
        with pytest.raises(ValueError, match="TRIAL_DATA_LIMIT"):
            iq_runtime.guard_iq_data(np.zeros((2001, 2, 1024), dtype=np.float32), val)
        with pytest.raises(ValueError, match="TRIAL_IQ_LENGTH_LIMIT"):
            iq_runtime.guard_iq_data(np.zeros((10, 2, 1025), dtype=np.float32),
                                     np.zeros((5, 2, 1025), dtype=np.float32))
    else:
        iq_runtime.guard_iq_data(np.zeros((2001, 2, 1024), dtype=np.float32), val)
        iq_runtime.guard_iq_data(np.zeros((10, 2, 1025), dtype=np.float32),
                                 np.zeros((5, 2, 1025), dtype=np.float32))
    with pytest.raises(ValueError):
        iq_runtime.guard_iq_data(np.zeros((10, 2, 1024)), np.zeros((5, 2, 512)))


def test_model_parameter_boundary_counts_frozen_parameters():
    limit = INFO.get("max_model_parameters")
    if limit is None:
        assert iq_runtime.guard_model(_FakeModel(10_000_000)) == 10_000_000
        return
    assert iq_runtime.guard_model(_FakeModel(limit)) == limit
    with pytest.raises(ValueError, match="TRIAL_MODEL_LIMIT"):
        iq_runtime.guard_model(_FakeModel(limit + 1))


def test_display_dictionary_is_not_the_limit_source():
    info = get_build_info()
    original_epochs = info["max_epochs"]
    try:
        info["max_epochs"] = None
        info["max_train_samples"] = None
        if TRIAL:
            with pytest.raises(ValueError, match="TRIAL_EPOCH_LIMIT"):
                iq_runtime.check_epochs(21)
            with pytest.raises(ValueError, match="TRIAL_DATA_LIMIT"):
                iq_runtime.guard_train_count(2001)
        else:
            iq_runtime.check_epochs(21)
            iq_runtime.guard_train_count(2001)
    finally:
        info["max_epochs"] = original_epochs


def test_runtime_env_cannot_switch_edition(monkeypatch):
    monkeypatch.setenv("SIMUSIGNAL_BUILD_EDITION", "full")
    if INFO["edition"] == "trial":
        with pytest.raises(ValueError, match="TRIAL_EPOCH_LIMIT"):
            iq_runtime.check_epochs(21)


def test_direct_training_api_is_guarded():
    train = np.zeros((2, 2, 1024), dtype=np.float32)
    val = np.zeros((1, 2, 1024), dtype=np.float32)
    bad_epochs = 21 if TRIAL else 10001
    with pytest.raises(ValueError):
        iq_runtime.train_classifier(train, np.array([0, 1]), val, np.array([0]),
                                    classes=["a", "b"], epochs=bad_epochs, verbose=False)
    if TRIAL:
        with pytest.raises(ValueError, match="TRIAL_IQ_LENGTH_LIMIT"):
            iq_runtime.train_classifier(
                np.zeros((2, 2, 1025), dtype=np.float32), np.array([0, 1]),
                np.zeros((1, 2, 1025), dtype=np.float32), np.array([0]),
                classes=["a", "b"], epochs=1, verbose=False)


def test_detection_limits_apply_to_samples_parameters_and_epochs():
    detection_runtime.guard_detection_samples(2000)
    limit = INFO.get("max_epochs") or 10000
    parameter_limit = INFO.get("max_model_parameters")
    if TRIAL:
        with pytest.raises(ValueError, match="TRIAL_DATA_LIMIT"):
            detection_runtime.guard_detection_samples(2001)
        with pytest.raises(ValueError, match="TRIAL_EPOCH_LIMIT"):
            detection_runtime.guard_detection_epochs(limit + 1)
        if parameter_limit is not None:
            with pytest.raises(ValueError, match="TRIAL_MODEL_LIMIT"):
                detection_runtime.guard_detection_model(_FakeModel(parameter_limit + 1))
        with pytest.raises(ValueError, match="TRIAL_UNSUPPORTED"):
            detection_runtime.require_rtdetr_allowed("rtdetr")
        # YOLO 不经过 RT-DETR 的“暂不开放”分支，只受其余三项限额约束。
        detection_runtime.require_rtdetr_allowed("yolo26s")
    else:
        detection_runtime.guard_detection_samples(2001)
        detection_runtime.guard_detection_epochs(limit)
        detection_runtime.guard_detection_model(_FakeModel(10_000_000))
        detection_runtime.require_rtdetr_allowed("rtdetr")
        detection_runtime.require_rtdetr_allowed("yolo26s")
