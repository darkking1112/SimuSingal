"""生成配方校验与参数预览测试（方案文档 §6.3）。"""
import pytest

from signal_analysis.algorithms.generation.recipes import preview_recipe, sample_seed, validate_recipe

RECIPE = {
    "contract": "gen_recipe_v1",
    "engine": "project",
    "base_seed": 7,
    "count": 120,
    "record": {
        "sample_rate_hz": {"choice": [200000, 500000]},
        "duration_s": {"uniform": [0.2, 0.5]},
    },
    "signals": {
        "mode": {"balanced": ["fm", "qpsk"]},
        "snr_db": {"uniform": [-5, 30]},
        "bandwidth_ratio": {"loguniform": [0.02, 0.3]},
        "power_dbfs": {"fixed": -6.0},
    },
    "stratify": {"axes": ["mode", "snr_db:5"], "min_per_cell": 10},
    "holdout": [{"axis": "snr_db", "range": [25, 30], "split": "test"}],
    "labels": {"detection": "per_hop_v1", "amc": True},
}


def test_validate_recipe_ok_and_paths():
    info = validate_recipe(RECIPE)
    assert info["count"] == 120 and info["engine"] == "project"
    paths = {tuple(item["path"]) for item in info["parameters"]}
    assert ("signals", "mode") in paths
    assert ("record", "duration_s") in paths


def test_validate_recipe_rejects_bad_inputs():
    with pytest.raises(ValueError, match="契约"):
        validate_recipe({**RECIPE, "contract": "gen_recipe_v2"})
    with pytest.raises(ValueError, match="引擎"):
        validate_recipe({**RECIPE, "engine": "magic"})
    with pytest.raises(ValueError, match="base_seed"):
        validate_recipe({**RECIPE, "base_seed": -1})
    with pytest.raises(ValueError, match="count"):
        validate_recipe({**RECIPE, "count": 0})
    broken = {**RECIPE, "signals": {**RECIPE["signals"], "snr_db": {"gauss": [0, 1]}}}
    with pytest.raises(ValueError, match="分布节点"):
        validate_recipe(broken)
    broken = {**RECIPE, "signals": {**RECIPE["signals"],
                                    "bandwidth_ratio": {"loguniform": [0, 0.3]}}}
    with pytest.raises(ValueError, match="大于 0"):
        validate_recipe(broken)
    broken = {**RECIPE, "signals": {**RECIPE["signals"],
                                    "mode": {"choice": ["fm"], "weights": [1, 2]}}}
    with pytest.raises(ValueError, match="weights"):
        validate_recipe(broken)
    broken = {**RECIPE, "stratify": {"axes": ["不存在"], "min_per_cell": 1}}
    with pytest.raises(ValueError, match="不存在"):
        validate_recipe(broken)
    broken = {**RECIPE, "holdout": [{"axis": "snr_db", "range": [1], "split": "test"}]}
    with pytest.raises(ValueError, match="holdout"):
        validate_recipe(broken)
    broken = {**RECIPE, "labels": {"detection": "bad"}}
    with pytest.raises(ValueError, match="detection"):
        validate_recipe(broken)


def test_sample_seed_is_deterministic_and_index_dependent():
    assert sample_seed(7, 0) == sample_seed(7, 0)
    assert sample_seed(7, 1) != sample_seed(7, 0)
    assert sample_seed(8, 0) != sample_seed(7, 0)
    assert all(0 <= sample_seed(7, index) <= 2 ** 32 - 1 for index in range(50))


def test_preview_balanced_uniform_and_preview_rows():
    result = preview_recipe(RECIPE, samples=40)
    assert result["count"] == 40
    assert len(result["preview"]) == 5
    assert "signals.mode" in result["preview"][0]
    # balanced 轮转：两种样式各 20 条
    mode = next(item for item in result["axes"] if item["axis"] == "mode")
    counts = {entry["label"]: entry["count"] for entry in mode["values"]}
    assert counts == {"fm": 20, "qpsk": 20}
    # uniform 落在区间内且分箱计数总和等于样本数
    snr = next(item for item in result["axes"] if item["axis"] == "snr_db")
    assert sum(entry["count"] for entry in snr["values"]) == 40
    rows = preview_recipe(RECIPE, samples=40, preview_rows=40)["preview"]
    assert all(-5 <= row["signals.snr_db"] <= 30 for row in rows)
    assert all(0.02 <= row["signals.bandwidth_ratio"] <= 0.3 for row in rows)
    assert all(row["signals.power_dbfs"] == -6.0 for row in rows)
    # 分层格子统计：40 个样本、min_per_cell=10 时必然有格子不足
    assert result["cells"]["min_per_cell"] == 10
    assert result["cells"]["total"] > 0
    assert result["cells"]["below_min_per_cell"] >= 1
    # 同样本数、同种子 → 完全一致（预览可复现）
    again = preview_recipe(RECIPE, samples=40)
    assert again["preview"] == result["preview"]


def test_preview_matches_declared_count_by_default():
    small = {**RECIPE, "count": 6, "stratify": {"axes": ["mode"], "min_per_cell": 2}}
    result = preview_recipe(small)
    assert result["count"] == 6 and len(result["preview"]) == 5
