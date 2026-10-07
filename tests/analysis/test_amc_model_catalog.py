"""AMC 模型目录（``src``）与训练侧实现注册表的一致性回归。

覆盖：目录零 torch（源码级 + 子进程运行期）、spec ↔ 实现一一对应、参数规则
（未知键/长度/奇数/开闭区间/整除/模型级交叉校验）、窗口约束与目录 JSON 查询接口。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING = REPO_ROOT / "training"
CATALOG_DIR = REPO_ROOT / "src" / "signal_analysis" / "algorithms" / "amc" / "ai_model"
for _extra in (REPO_ROOT / "src", TRAINING):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from signal_analysis.algorithms.amc.ai_model import (  # noqa: E402
    CATALOG_VERSION,
    ModelSpec,
    ParamSpec,
    SPECS,
    catalog_json,
    check_samples,
    describe_model,
    merge_param_sources,
    model_spec,
    spec_json,
    validate_params,
)

TORCH_PATTERN = re.compile(r"^(?:import|from)\s+(torch|torchsig)(?:\.|\s|$)")


def test_catalog_source_and_runtime_are_torch_free():
    """目录既不能在源码里 import torch，也不能在导入时加载 torch（GUI 进程无 torch）。"""
    for path in sorted(CATALOG_DIR.glob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert not [line for line in lines if TORCH_PATTERN.match(line)], path.name
    code = ("import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
            "from signal_analysis.algorithms.amc.ai_model import catalog_json; "
            "payload = catalog_json(); assert payload; "
            "assert not [name for name in sys.modules "
            "if name == 'torch' or name.startswith('torch.') or name == 'onnxruntime']")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_specs_and_implementations_are_one_to_one():
    """模型 ID = 实现模块名 = 实现文件名；目录与实现目录互不缺失。"""
    assert CATALOG_VERSION >= 1
    assert set(SPECS) == {spec.id for spec in SPECS.values()}
    # 实现目录里除私有模块、训练器（trainer.py）与 checkpoint 读写（checkpoint.py）外，
    # 每个模块都应对应一个模型
    modules = {path.stem for path in (TRAINING / "amc_models").glob("*.py")
               if not path.name.startswith("_") and path.stem not in ("trainer", "checkpoint")}
    assert modules == set(SPECS)
    for name, spec in SPECS.items():
        assert spec.implementation == f"amc_models.{name}"
        assert (TRAINING / "amc_models" / f"{name}.py").is_file()
        assert spec.model_revision >= 1
        assert spec.layers and spec.requires
        assert spec.input_layout in ("iq_channels_first_v1", "iq_channels_last_v1")
        assert spec.samples == "any" or re.fullmatch(r"(exact|min):\d+", spec.samples)


def test_default_params_satisfy_schema():
    """目录默认值必须能通过自己的校验，且重复校验结果稳定（幂等）。"""
    for spec in SPECS.values():
        resolved = validate_params(spec, {})
        assert set(resolved) == {param.name for param in spec.params}
        assert validate_params(spec, resolved) == resolved


def test_unknown_and_invalid_params_are_rejected():
    cnn = model_spec("cnn")
    with pytest.raises(ValueError, match="不认识参数"):
        validate_params(cnn, {"nope": 1})
    with pytest.raises(ValueError, match="3 个正整数"):
        validate_params(cnn, {"channels": [64, 128]})
    with pytest.raises(ValueError, match="每个元素"):
        validate_params(cnn, {"channels": [64, 0, 128]})
    with pytest.raises(ValueError, match="奇数"):
        validate_params(cnn, {"kernel": 4})
    with pytest.raises(ValueError, match="小于 1"):
        validate_params(cnn, {"dropout": 1.0})
    tcn = model_spec("tcn")
    with pytest.raises(ValueError, match="不小于 2"):
        validate_params(tcn, {"kernel": 1})
    with pytest.raises(ValueError, match="最多 3 个"):
        validate_params(tcn, {"channels": [8, 8, 8, 8]})


def test_merge_rejects_conflicting_sources():
    cnn = model_spec("cnn")
    with pytest.raises(ValueError, match="不能同时"):
        merge_param_sources(cnn, legacy={"kernel": 9}, explicit={"kernel": 7})
    with pytest.raises(ValueError, match="不认识参数"):
        merge_param_sources(cnn, explicit={"nope": 1})


def test_divides_and_cross_validation_support():
    """整除规则与模型级交叉校验是 schema 的扩展点（现有模型暂未使用）。"""

    def cross_validate(params):
        if params["heads"] > 8:
            raise ValueError("dummy：多头数不能超过 8")

    spec = ModelSpec(
        id="dummy", title="dummy", summary="dummy", implementation="amc_models.cnn",
        model_revision=1, layers=("x",),
        params=(ParamSpec("heads", "int", 4, minimum=1, divides="width"),
                ParamSpec("width", "int", 64, minimum=1)),
        cross_validate=cross_validate)
    assert validate_params(spec, {"heads": 8, "width": 64}) == {"heads": 8, "width": 64}
    with pytest.raises(ValueError, match="必须整除"):
        validate_params(spec, {"heads": 7, "width": 64})
    with pytest.raises(ValueError, match="dummy"):
        validate_params(spec, {"heads": 16, "width": 64})  # 整除通过，交叉校验拦截


def test_sample_constraints_are_enforced():
    cnn = model_spec("cnn")
    check_samples(cnn, 1024)  # any：任意长度都通过
    custom = replace(cnn, samples="exact:128")
    with pytest.raises(ValueError, match="固定要求 128"):
        check_samples(custom, 1024)
    custom = replace(cnn, samples="min:256")
    with pytest.raises(ValueError, match="不少于 256"):
        check_samples(custom, 128)


def test_catalog_json_lists_models_and_params():
    payload = json.loads(catalog_json())
    assert payload["catalog_version"] == CATALOG_VERSION
    assert {model["id"] for model in payload["models"]} == set(SPECS)
    cnn = next(model for model in payload["models"] if model["id"] == "cnn")
    assert {param["name"] for param in cnn["params"]} == {"channels", "kernel", "dropout"}
    assert cnn["model_revision"] == SPECS["cnn"].model_revision
    assert cnn["layers"]


def test_spec_json_carries_the_constraints_needed_to_build_controls():
    """GUI 只凭 spec_json 就要能造控件：类型、默认值、区间/长度/奇数约束与提示都在。"""
    entry = spec_json(model_spec("cnn"))
    params = {param["name"]: param for param in entry["params"]}
    channels = params["channels"]
    assert channels["kind"] == "int-list" and channels["default"] == [32, 64, 128]
    assert channels["length"] == 3 and channels["element_minimum"] == 1
    kernel = params["kernel"]
    assert kernel["kind"] == "int" and kernel["odd"] is True and kernel["minimum"] >= 3
    assert params["dropout"]["kind"] == "float" and params["dropout"]["exclusive_maximum"]
    assert all(param["label"] and param["hint"] for param in entry["params"])

    text = describe_model(entry)
    assert "IQCNN" in text and "结构：" in text and "参数：" in text
    assert "窗口：" in text and "依赖：torch" in text
