"""独立“信号导入”页（方案 §7.1，2026-10 文件清单改版）。

覆盖：自动识别与缺参数标红、批量设置、逐文件覆盖（备注/调制/目标行）、
CSV 标注清单匹配与未匹配报告、失败项保留、分片自动阈值与三态覆盖、
服务层 import_inspect / import_manifest / import_files 的逐文件语义。
"""
import csv
import os
import time
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets
from signal_analysis.data.io import write_samples
from signal_analysis.ui import (IMPORT_COL_MOD, IMPORT_COL_NOTE,
                                IMPORT_COL_RF_CENTER, IMPORT_COL_STATUS, MainWindow)
from signal_analysis.services import execute
from signal_analysis.data import Workspace


def wait_job(app, window):
    deadline = time.monotonic() + 15
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, "GUI task did not finish"
    assert "未完成" not in window.status.text(), window.status.text()


def _make_npy(tmp_path, name, count=64):
    return write_samples(tmp_path / "incoming" / name,
                         np.ones(count, dtype=np.complex64), "npy")


def _make_bin(tmp_path, name, samples=64, dtype="int16"):
    path = tmp_path / "incoming" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.arange(samples * 2, dtype=dtype).tofile(path)
    return path


def _select_rows(table, rows):
    table.clearSelection()
    model = table.selectionModel()
    for row in rows:
        model.select(table.model().index(row, 0),
                     QtCore.QItemSelectionModel.SelectionFlag.Select
                     | QtCore.QItemSelectionModel.SelectionFlag.Rows)


def _status_text(window, row):
    return window.import_table.item(row, IMPORT_COL_STATUS).text()


@pytest.mark.gui
def test_file_list_batch_settings_and_import(tmp_path, monkeypatch):
    """文件清单：自动识别 → 缺参数标红 → 批量设置 → 导入；成功行移出清单。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source_npy = _make_npy(tmp_path, "record_a.npy")
    source_bin = _make_bin(tmp_path, "record_b.bin")
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window._page_index("信号导入") == 0
        assert not window.import_start_button.isEnabled()
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(source_npy), str(source_bin)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        assert window.import_table.rowCount() == 2

        rows = {window._import_row_info(row)["name"]: row
                for row in range(window.import_table.rowCount())}
        npy_row, bin_row = rows["record_a.npy"], rows["record_b.bin"]
        assert "缺采样率" in _status_text(window, npy_row)
        assert "缺采样率、类型、字节序" in _status_text(window, bin_row)
        assert "64 点" in window.import_table.item(npy_row, 5).text()
        # 未就绪时不能开始导入
        assert not window.import_start_button.isEnabled()

        _select_rows(window.import_table, [npy_row, bin_row])
        window._apply_batch_settings([npy_row, bin_row], sample_rate="48000",
                                     binary_dtype="int16", endian="little")
        assert _status_text(window, npy_row) == "就绪"
        assert _status_text(window, bin_row) == "就绪"
        assert "0.0013" in window.import_table.item(npy_row, 5).text()  # 64/48000 s
        assert window.import_start_button.isEnabled()

        # 备注列写资产备注；目标集合新建
        window.import_table.item(npy_row, IMPORT_COL_NOTE).setText("批次备注")
        window.import_collection.setCurrentIndex(
            window.import_collection.findData("__new__"))
        window.import_collection_name.setText("清单批次集合")
        window.start_import_batch()
        wait_job(app, window)
        assert window.import_table.rowCount() == 0
        assert "导入完成" in window.import_status.text()
        assert not window.import_manage_button.isHidden()
        # 新建集合后左侧自动切到该集合，便于直接核对成员
        created = next(item for item in window.workspace.list_collections()
                       if item["name"] == "清单批次集合")
        assert window.collection_combo.currentData() == created["id"]
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert set(assets) == {"record_a.npy", "record_b.bin"}
    assert assets["record_a.npy"]["label"] == "批次备注"
    assert assets["record_b.bin"]["label"] == ""
    # 2 个文件 < 自动阈值 20 → 独立 NPY；未给目标行 → 未知待标注（不建占位目标）
    assert all(item["storage_kind"] == "file" for item in assets.values())
    assert workspace.list_shards() == []
    assert all(workspace.list_targets(item["id"]) == [] for item in assets.values())
    collection = next(item for item in workspace.list_collections()
                      if item["name"] == "清单批次集合")
    assert len(workspace.collection_asset_ids(collection["id"])) == 2


@pytest.mark.gui
def test_rf_center_per_file_override_and_batch_default(tmp_path, monkeypatch):
    """射频中心：行内优先、留空回落本批默认；非法值标红阻断该行。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = _make_npy(tmp_path, "center_a.npy")
    second = _make_npy(tmp_path, "center_b.npy")
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(first), str(second)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        rows = [window._import_row_by_path(str(first)),
                window._import_row_by_path(str(second))]
        window._apply_batch_settings(rows, sample_rate="48000")
        window.import_rf_center.setText("100e6")  # 本批默认 100 MHz
        # 行内非法值：该行标红，不被隐式忽略
        window.import_table.item(rows[0], IMPORT_COL_RF_CENTER).setText("abc")
        assert _status_text(window, rows[0]) == "射频中心应为数值"
        # 行内覆盖本批默认；第二行留空 → 用默认
        window.import_table.item(rows[0], IMPORT_COL_RF_CENTER).setText("433.5e6")
        assert _status_text(window, rows[0]) == "就绪"
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert assets["center_a.npy"]["rf_center_hz"] == pytest.approx(433.5e6)
    assert assets["center_b.npy"]["rf_center_hz"] == pytest.approx(100e6)


@pytest.mark.gui
def test_capture_time_from_manifest_or_import_time(tmp_path, monkeypatch):
    """采集时间不手填：清单 CSV 逐文件声明（同行须一致），缺省写导入时间。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = _make_npy(tmp_path, "capture_a.npy")
    second = _make_npy(tmp_path, "capture_b.npy")
    third = _make_npy(tmp_path, "capture_c.npy")
    window = MainWindow(tmp_path)
    assert not hasattr(window, "import_capture")  # 页面不提供手填入口
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(first), str(second), str(third)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        manifest = tmp_path / "capture.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["文件", "粒度", "起止单位", "起始", "结束", "采集时间"])
            writer.writerow(["capture_a.npy", "whole_record", "采样点", "0", "64",
                             "2026-09-01T08:30:00Z"])
            writer.writerow(["capture_b.npy", "whole_record", "采样点", "0", "64",
                             "2026-09-02T09:00:00Z"])
            writer.writerow(["capture_b.npy", "whole_record", "采样点", "0", "64",
                             "2026-09-03T09:00:00Z"])  # 与上一行冲突
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(manifest), "CSV"))
        window.import_csv_manifest()
        wait_job(app, window)
        report = window.import_csv_report.text()
        assert "2 个文件带采集时间" in report
        assert "未匹配/非法 1 行" in report
        assert "采集时间与同一文件的其他行不一致" in report
        rows = [window._import_row_by_path(str(item)) for item in (first, second, third)]
        window._apply_batch_settings(rows, sample_rate="48000")
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert assets["capture_a.npy"]["capture_started_at"] == "2026-09-01T08:30:00Z"
    assert assets["capture_b.npy"]["capture_started_at"] == "2026-09-02T09:00:00Z"
    # 清单未声明的文件：缺省导入时间（可解析的 ISO 时间戳）
    assert datetime.fromisoformat(assets["capture_c.npy"]["capture_started_at"])


@pytest.mark.gui
def test_csv_manifest_attaches_targets_with_flags(tmp_path, monkeypatch):
    """CSV 标注清单：按文件名匹配、毫秒换算、未匹配与非法行逐条列出。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = _make_npy(tmp_path, "record_c.npy")
    second = _make_npy(tmp_path, "record_d.npy")
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(first), str(second)], "数据"))
        window.choose_import_files()
        wait_job(app, window)

        manifest = tmp_path / "manifest.csv"
        with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["文件", "粒度", "起止单位", "起始", "结束", "频率下限Hz",
                             "频率上限Hz", "调制", "SNR", "备注"])
            writer.writerow(["record_c.npy", "whole_record", "毫秒", "0", "1",
                             "-12000", "12000", "QPSK", "15", "毫秒换算"])
            writer.writerow(["record_c.npy", "session", "采样点", "", "", "", "",
                             "AM", "", "只给调制"])
            writer.writerow(["missing.npy", "whole_record", "采样点", "0", "", "", "",
                             "", "", "文件不在清单"])
            writer.writerow(["record_d.npy", "whole_record", "采样点", "10", "5", "", "",
                             "", "", "终点不大于起点"])
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(manifest), "CSV"))
        window.import_csv_manifest()
        wait_job(app, window)
        report = window.import_csv_report.text()
        assert "已挂接 2 条目标" in report
        assert "未匹配/非法 2 行" in report
        assert "当前文件清单中没有该文件" in report and "终点必须大于起点" in report
        row = window._import_row_by_path(str(first))
        assert "目标 2 条" in _status_text(window, row)
        # 毫秒换算需要采样率；另一行也补上采样率后整批导入
        rows = [window._import_row_by_path(str(first)), window._import_row_by_path(str(second))]
        window._apply_batch_settings(rows, sample_rate="48000")
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    asset = next(item for item in workspace.list_assets() if item["name"] == "record_c.npy")
    targets = {item["scope"]: item for item in
               workspace.list_targets(asset["id"], with_current=True)}
    assert set(targets) == {"whole_record", "session"}
    whole = targets["whole_record"]
    assert (whole["for_detection"], whole["for_amc"]) == (1, 1)
    assert whole["current"]["source"] == "import"
    assert whole["current"]["sample_start"] == 0
    assert whole["current"]["sample_end"] == 48   # 1 ms × 48000 / 1000
    assert whole["current"]["f_low_hz"] == pytest.approx(-12000)
    assert whole["current"]["modulation"] == "QPSK"
    assert whole["current"]["snr_db"] == pytest.approx(15)
    session = targets["session"]
    # 只给调制：不置检测适用标记，AMC 适用标记置位（字典外规范名同样保留）
    assert (session["for_detection"], session["for_amc"]) == (0, 1)
    assert session["current"]["center_hz"] is None
    untouched = next(item for item in workspace.list_assets() if item["name"] == "record_d.npy")
    assert workspace.list_targets(untouched["id"]) == []


@pytest.mark.gui
def test_failed_rows_stay_in_list(tmp_path, monkeypatch):
    """失败项保留在清单里（带原因）；参数完备的行成功移出。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    good = _make_npy(tmp_path, "record_ok.npy")
    bad = tmp_path / "incoming" / "record_bad.csv"
    bad.write_text("I,Q\n1,abc\n", encoding="utf-8")
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(good), str(bad)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        rows = [window._import_row_by_path(str(good)), window._import_row_by_path(str(bad))]
        window._apply_batch_settings(rows, sample_rate="48000")
        assert window.import_start_button.isEnabled()
        window.start_import_batch()
        wait_job(app, window)
        assert window.import_table.rowCount() == 1
        assert window._import_row_info(0)["name"] == "record_bad.csv"
        assert _status_text(window, 0).startswith("解析失败：")
        assert not window.import_start_button.isEnabled()  # 失败行修正前不可再导入
    finally:
        window.close()
        app.processEvents()
    workspace = Workspace(tmp_path)
    assert [item["name"] for item in workspace.list_assets()] == ["record_ok.npy"]


@pytest.mark.gui
def test_shard_threshold_and_forced_modes(tmp_path):
    """分片：自动阈值（默认 20）+ 强制独立 / 强制分片三态。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    try:
        assert window._resolve_batch_shard(19) is False
        assert window._resolve_batch_shard(20) is True
        window.import_shard_mode.setCurrentIndex(
            window.import_shard_mode.findData("file"))
        assert window._resolve_batch_shard(100) is False
        window.import_shard_mode.setCurrentIndex(
            window.import_shard_mode.findData("shard"))
        assert window._resolve_batch_shard(1) is True
        window.import_shard_mode.setCurrentIndex(
            window.import_shard_mode.findData("auto"))
        window.import_shard_threshold.setValue(5)
        assert window._resolve_batch_shard(5) is True
        assert window._resolve_batch_shard(4) is False
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_single_signal_form_and_free_text_modulation(tmp_path, monkeypatch):
    """③ 信号参数：整条为单一信号表单；调制可自由输入（字典外原名保留）。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = _make_npy(tmp_path, "form_a.npy", 64)
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(source)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        window._apply_batch_settings([0], sample_rate="48000")
        _select_rows(window.import_table, [0])
        window.import_params_toggle.setChecked(True)
        assert not window.import_params_body.isHidden()
        window.import_single_modulation.setCurrentText("2FSK")  # 字典外自由文本
        window.import_single_f_low.setText("-1500")
        window.import_single_f_high.setText("1500")
        window.import_single_snr.setText("20")
        window.apply_single_signal_targets()
        assert "目标 1 条" in _status_text(window, 0)
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    asset = workspace.list_assets()[0]
    target = workspace.list_targets(asset["id"], with_current=True)[0]
    assert target["scope"] == "whole_record"
    assert (target["for_detection"], target["for_amc"]) == (1, 1)
    current = target["current"]
    assert current["modulation"] == "2FSK"  # 字典外原名保留，类别映射留给 AMC 侧
    assert current["f_low_hz"] == pytest.approx(-1500)
    assert current["snr_db"] == pytest.approx(20)
    assert (current["sample_start"], current["sample_end"]) == (0, 64)


@pytest.mark.gui
def test_new_collection_requires_a_name(tmp_path, monkeypatch):
    """选“新建集合…”但没填名称：就地提示，不启动导入任务。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = _make_npy(tmp_path, "name_a.npy")
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(source)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        window._apply_batch_settings([0], sample_rate="48000")
        window.import_collection.setCurrentIndex(
            window.import_collection.findData("__new__"))
        window.start_import_batch()
        assert window.active_job is None
        assert "请填写集合名称" in window.status.text()
    finally:
        window.close()
        app.processEvents()


def test_import_inspect_reads_headers(tmp_path):
    """服务层 import_inspect 只读头部：NPY 形状、SigMF 元数据、二进制字节数、缺失文件。"""
    npy = _make_npy(tmp_path, "inspect_a.npy", 32)
    binary = _make_bin(tmp_path, "inspect_b.bin", 64)
    sigmf = write_samples(tmp_path / "incoming" / "inspect_c",
                          np.ones(16, dtype=np.complex64), "sigmf", sample_rate=12345)
    result = execute({"workspace": str(tmp_path / "ws"), "action": "import_inspect",
                      "paths": [str(npy), str(binary), str(sigmf),
                                str(tmp_path / "incoming" / "missing.npy")]})
    files = {item["name"]: item for item in result["files"]}
    assert files["inspect_a.npy"]["format"] == "npy"
    assert files["inspect_a.npy"]["sample_count"] == 32
    assert files["inspect_b.bin"]["format"] == "binary"
    assert files["inspect_b.bin"]["sample_count"] is None
    assert files["inspect_b.bin"]["size_bytes"] == 64 * 2 * 2
    assert files["inspect_c.sigmf-meta"]["format"] == "sigmf"
    assert files["inspect_c.sigmf-meta"]["sample_rate"] == 12345
    assert files["inspect_c.sigmf-meta"]["sample_count"] == 16
    assert "文件不存在" in files["missing.npy"]["error"]


def test_import_files_per_file_payload_and_legacy(tmp_path):
    """import_files：逐文件覆盖采样率/备注/目标行（分片），旧 paths 请求仍可用。"""
    first = _make_npy(tmp_path, "payload_a.npy", 32)
    second = _make_npy(tmp_path, "payload_b.npy", 48)
    result = execute({"workspace": str(tmp_path), "action": "import_files",
                      "batch_shard": True, "rf_center_hz": 100e6, "files": [
                          {"path": str(first), "sample_rate": 1000.0, "label": "甲",
                           "rf_center_hz": 433e6,
                           "capture_started_at": "2026-09-01T04:05:06Z",
                           "targets": [{"scope": "whole_record", "start": 0, "end": 32,
                                        "start_unit": "samples", "f_low_hz": -100.0,
                                        "f_high_hz": 100.0, "modulation": "QPSK"}]},
                          {"path": str(second), "sample_rate": 2000.0}]})
    assert result["created"] == 2 and result["failed"] == 0
    assert result["targets_total"] == 1 and result["shard_id"]
    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert assets["payload_a.npy"]["sample_rate"] == 1000.0
    assert assets["payload_a.npy"]["label"] == "甲"
    assert assets["payload_b.npy"]["sample_rate"] == 2000.0
    # 逐文件优先；未给出回落请求级（本批默认）
    assert assets["payload_a.npy"]["rf_center_hz"] == pytest.approx(433e6)
    assert assets["payload_b.npy"]["rf_center_hz"] == pytest.approx(100e6)
    assert assets["payload_a.npy"]["capture_started_at"] == "2026-09-01T04:05:06Z"
    assert assets["payload_b.npy"]["capture_started_at"]  # 缺省记本批导入时间
    assert all(item["storage_kind"] == "shard" for item in assets.values())
    target = workspace.list_targets(assets["payload_a.npy"]["id"], with_current=True)[0]
    assert (target["for_detection"], target["for_amc"]) == (1, 1)
    assert workspace.list_targets(assets["payload_b.npy"]["id"]) == []

    third = _make_npy(tmp_path, "payload_c.npy", 16)
    legacy = execute({"workspace": str(tmp_path), "action": "import_files",
                      "paths": [str(third)], "sample_rate": 3000.0,
                      "rf_center_hz": 50e6, "capture_started_at": "2026-09-02T00:00:00Z",
                      "targets": [{"scope": "segment", "f_low_hz": -50.0,
                                   "f_high_hz": 50.0}]})
    assert legacy["created"] == 1
    workspace = Workspace(tmp_path)
    third_asset = next(item for item in workspace.list_assets()
                       if item["name"] == "payload_c.npy")
    assert third_asset["rf_center_hz"] == pytest.approx(50e6)
    assert third_asset["capture_started_at"] == "2026-09-02T00:00:00Z"
    legacy_target = workspace.list_targets(third_asset["id"], with_current=True)[0]
    assert legacy_target["scope"] == "segment"
    assert legacy_target["for_detection"] == 1 and legacy_target["for_amc"] == 0


def test_hop_targets_are_rejected_with_guidance(tmp_path):
    """导入路径不建逐跳目标：给出替代入口而不是静默失败。"""
    source = _make_npy(tmp_path, "hop_a.npy", 32)
    result = execute({"workspace": str(tmp_path), "action": "import_files",
                      "files": [{"path": str(source), "sample_rate": 1000.0,
                                 "targets": [{"scope": "hop", "start": 0, "end": 8}]}]})
    assert result["created"] == 0 and result["failed"] == 1
    message = result["results"][0]["error"]
    assert "逐跳目标" in message and "采纳为参数标注" in message
    assert Workspace(tmp_path).list_assets() == []
