"""运行时自检命令：源码环境通过，并能在二进制环境要求 .so/.pyd。"""

from signal_analysis.selftest import run_selftest


def test_selftest_reports_core_model_and_numeric_contracts():
    result = run_selftest()
    assert result["ok"], result["errors"]
    assert result["schema"] == "simusignal_selftest_v1"
    assert len(result["core_modules"]) == 8
    assert result["model"]["id"]
    assert result["numeric"]["checks"]["statistics_contract"]
    assert result["numeric"]["checks"]["detection_contract"]
    assert result["numeric"]["checksum"]


def test_source_tree_fails_explicit_binary_expectation():
    result = run_selftest(expect_binary=True)
    assert result["binary"] is False
    assert result["ok"] is False
    assert any(".so/.pyd" in error for error in result["errors"])
