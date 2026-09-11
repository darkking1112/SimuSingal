from pathlib import Path

import numpy as np
import pytest

from signal_analysis.dataio import read_samples
from common.reports import export_report
from signal_analysis.storage import Workspace


def test_import_complex_csv_and_npy(tmp_path):
    csv = tmp_path / "中文.csv"
    csv.write_text("1,2\n3,4\n", encoding="utf-8")
    expected = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    np.testing.assert_array_equal(read_samples(csv), expected)
    npy = tmp_path / "array.npy"
    np.save(npy, expected)
    np.testing.assert_array_equal(read_samples(npy), expected)
    np.save(npy, np.array([{"unsafe": 1}], dtype=object))
    with pytest.raises(ValueError):
        read_samples(npy)


def test_asset_persistence_label_and_tampering(tmp_path):
    store = Workspace(tmp_path / "工作目录")
    asset = store.add_samples([1, 2], 100, "a'b")
    store.set_label(asset["id"], "参考备注")
    reopened = Workspace(store.root)
    assert reopened.list_assets("a'b")[0]["label"] == "参考备注"
    _, data = reopened.load_samples(asset["id"])
    np.testing.assert_array_equal(data, [1, 2])
    del data
    (store.root / asset["path"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="校验"):
        reopened.load_samples(asset["id"])


def test_run_roundtrip_and_escaped_report(tmp_path):
    store = Workspace(tmp_path / "data")
    asset = store.add_samples([1], 100, "sample")
    result = store.save_run("analysis", {"summary": {"label": "<script>alert(1)</script>"}},
                            {"points": np.array([1, 2])}, asset["id"])
    assert Workspace(store.root).get_run(result["run_id"]) == result
    assert not Path(result["plots_path"]).is_absolute()
    output = tmp_path / "report.html"
    export_report(result, output)
    assert "<script>" not in output.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in output.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        export_report(result, tmp_path / "report.exe")


def test_invalid_import_does_not_create_asset(tmp_path):
    store = Workspace(tmp_path)
    with pytest.raises(ValueError):
        store.add_samples([float("nan")], 100, "invalid")
    assert store.list_assets() == []
    assert list((tmp_path / "assets").iterdir()) == []


def _fake_metrics(precision, matched):
    return {"true": 3, "matched": matched, "missed": 3 - matched, "false_alarm": 1,
            "precision": precision, "recall": 0.5, "f1": 0.6, "center_mae_hz": 1200.0,
            "bandwidth_mape": 0.04, "snr_mae_db": 0.5}


def test_html_report_has_metric_tables(tmp_path):
    baseline = _fake_metrics(0.5678, 1)
    del baseline["snr_mae_db"]  # 传统基线没有该项：表格必须显示 “--”，不能写成 0
    detection = tmp_path / "detect.html"
    export_report({"kind": "ml_detect", "run_id": "r1", "asset_name": "合成数据",
                   "algorithm": "yolox_detect_v1", "contract": "detect_result_v1",
                   "metrics": _fake_metrics(0.9123, 2), "baseline_metrics": baseline},
                  detection)
    page = detection.read_text(encoding="utf-8")
    # 检测结果与传统基线并排成表：两列数值都要出现
    assert "<th>检测结果</th>" in page and "<th>传统基线</th>" in page
    assert "<td>0.9123</td>" in page and "<td>0.5678</td>" in page
    assert "中心频率 MAE / Hz" in page and "<td>1200.0</td>" in page
    assert "<td>--</td>" in page
    # 无真值时不编造指标，只保留原始 JSON
    plain = tmp_path / "detect-none.html"
    export_report({"kind": "detect", "run_id": "r2", "metrics": None}, plain)
    assert "检测评测" not in plain.read_text(encoding="utf-8")


def test_html_report_amc_section_counts_pending(tmp_path):
    report = tmp_path / "amc.html"
    export_report({"kind": "amc_classify", "run_id": "r3", "asset_name": "QPSK 数据",
                   "prediction": {"label": "qpsk", "name": "QPSK 四相键控", "confidence": 0.99,
                                  "margin": 0.8, "reliable": True, "reason": "分数明确"},
                   "truth": {"available": False, "reason": "数据没有生成器真值"}, "truth_hit": None,
                   "pending": ["识别准确率的合格门限尚未确认（技术方案待确认项）"]},
                  report)
    page = report.read_text(encoding="utf-8")
    assert "QPSK 四相键控" in page and "<td>0.9900</td>" in page
    assert "按“不适用”计数，不丢弃样本" in page
    assert "尚未确认项" in page and "合格门限尚未确认" in page


def test_failed_run_does_not_leave_success_artifact(tmp_path):
    import sqlite3
    store = Workspace(tmp_path)
    with pytest.raises(ValueError):
        store.save_run("analysis", {"summary": {}}, asset_id="missing")
    assert store.list_runs() == []
    assert list((tmp_path / "runs").iterdir()) == []
