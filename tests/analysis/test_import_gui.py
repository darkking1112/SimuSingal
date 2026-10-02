"""独立“信号导入”页（方案 §7.1，2026-10 只读表格 + 信号参数编辑面板改版）。

覆盖：自动识别与缺参数标红、表格只读、面板多选编辑（信号名称/采样率/调制/
类型/字节序）、混合格式禁用类型与字节序、信号名称自动命名三态、CSV 标注清单
（信号名称列、旧检测列已忽略、同名文件一致性）、仅 AMC 目标（for_detection=0）、
失败项保留、分片自动阈值与三态覆盖、服务层 import_inspect / import_manifest /
import_files 的逐文件语义与取消编排。
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
from signal_analysis.ui import (IMPORT_COL_MOD, IMPORT_COL_NAME, IMPORT_COL_POINTS,
                                IMPORT_COL_SNR, IMPORT_COL_STATUS, MainWindow)
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


def _cell_text(window, row, column):
    return window.import_table.item(row, column).text()


def _apply_params(window, rows, **fields):
    """经“信号参数编辑”面板把非空字段应用到给定行（其余字段保持“不修改”）。"""
    _select_rows(window.import_table, rows)
    window.import_name.setText(fields.get("name", ""))
    window.import_rate.setText(fields.get("rate", ""))
    window.import_modulation.setCurrentText(fields.get("modulation", ""))
    window.import_snr.setText(fields.get("snr", ""))
    window.import_dtype.setCurrentIndex(
        window.import_dtype.findData(fields.get("dtype")))
    window.import_endian.setCurrentIndex(
        window.import_endian.findData(fields.get("endian")))
    window.apply_signal_params()


@pytest.mark.gui
def test_file_list_edit_and_import_via_panel(tmp_path, monkeypatch):
    """文件清单：自动识别 → 缺参数标红 → 面板编辑 → 导入；成功行移出清单。"""
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
        assert "64 点" in _cell_text(window, npy_row, IMPORT_COL_POINTS)
        assert not window.import_start_button.isEnabled()

        # 表格只读：无编辑触发器、单元格不可编辑、没有内嵌下拉控件
        assert (window.import_table.editTriggers()
                == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        assert not (window.import_table.item(npy_row, IMPORT_COL_NAME).flags()
                    & QtCore.Qt.ItemFlag.ItemIsEditable)
        assert window.import_table.cellWidget(bin_row, IMPORT_COL_MOD) is None

        # 混合格式：类型/字节序禁用；采样率仍可应用到两行
        _select_rows(window.import_table, [npy_row, bin_row])
        assert not window.import_dtype.isEnabled()
        assert not window.import_endian.isEnabled()
        _apply_params(window, [npy_row, bin_row], rate="48000")
        assert _status_text(window, npy_row) == "就绪"
        assert "缺类型、字节序" in _status_text(window, bin_row)
        assert "0.0013" in _cell_text(window, npy_row, IMPORT_COL_POINTS)

        # 全二进制选择：类型/字节序可编辑并生效
        _apply_params(window, [bin_row], dtype="int16", endian="little")
        assert window.import_dtype.isEnabled()
        assert _status_text(window, bin_row) == "就绪"

        # 面板填写信号名称、调制与 SNR（自由文本之外的字典内名称）
        _apply_params(window, [npy_row], name="批次信号", modulation="QPSK", snr="15")
        assert _cell_text(window, npy_row, IMPORT_COL_NAME) == "批次信号"
        assert _cell_text(window, npy_row, IMPORT_COL_MOD) == "QPSK"
        assert _cell_text(window, npy_row, IMPORT_COL_SNR) == "15"
        assert "目标 1 条" in _status_text(window, npy_row)
        assert window.import_start_button.isEnabled()

        window.import_collection.setCurrentIndex(
            window.import_collection.findData("__new__"))
        window.import_collection_name.setText("清单批次集合")
        window.start_import_batch()
        wait_job(app, window)
        assert window.import_table.rowCount() == 0
        assert "导入完成" in window.status.text()
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
    assert set(assets) == {"批次信号", "record_b.bin"}
    assert assets["批次信号"]["sample_rate"] == 48000.0
    assert assets["record_b.bin"]["sample_rate"] == 48000.0
    # 2 个文件 < 自动阈值 20 → 独立 NPY；无调制的行不建目标
    assert all(item["storage_kind"] == "file" for item in assets.values())
    assert workspace.list_shards() == []
    target = workspace.list_targets(assets["批次信号"]["id"], with_current=True)[0]
    assert (target["for_detection"], target["for_amc"]) == (0, 1)
    assert target["current"]["modulation"] == "QPSK"
    assert target["current"]["snr_db"] == pytest.approx(15.0)
    assert target["current"]["snr_definition"] == "declared"
    assert workspace.list_targets(assets["record_b.bin"]["id"]) == []
    collection = next(item for item in workspace.list_collections()
                      if item["name"] == "清单批次集合")
    assert len(workspace.collection_asset_ids(collection["id"])) == 2


@pytest.mark.gui
def test_signal_name_defaults_and_auto_naming(tmp_path, monkeypatch):
    """信号名称：默认文件名；有调制时自动 “调制 · 文件名前 5 字符”；显式填写优先。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    first = _make_npy(tmp_path, "record_a.npy")
    second = _make_npy(tmp_path, "record_b.npy")
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(first), str(second)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        rows = [window._import_row_by_path(str(first)),
                window._import_row_by_path(str(second))]
        _apply_params(window, rows, rate="48000")

        # 默认名 = 磁盘文件名（含扩展名）
        assert _cell_text(window, rows[0], IMPORT_COL_NAME) == "record_a.npy"
        assert _cell_text(window, rows[1], IMPORT_COL_NAME) == "record_b.npy"

        # 应用调制 → 自动命名；显式名称优先并保留调制
        _apply_params(window, [rows[0]], modulation="QPSK")
        assert _cell_text(window, rows[0], IMPORT_COL_NAME) == "QPSK · recor"
        _apply_params(window, [rows[0]], name="甲信号")
        assert _cell_text(window, rows[0], IMPORT_COL_NAME) == "甲信号"
        assert _cell_text(window, rows[0], IMPORT_COL_MOD) == "QPSK"

        # 信号名称超长：就地提示且不生效
        _apply_params(window, [rows[0]], name="x" * 201)
        assert "最多 200" in window.status.text()
        assert _cell_text(window, rows[0], IMPORT_COL_NAME) == "甲信号"

        # 调制“未知”= 清除目标；无显式名称时回退文件名
        _apply_params(window, [rows[1]], modulation="AM")
        assert _cell_text(window, rows[1], IMPORT_COL_NAME) == "AM · recor"
        _apply_params(window, [rows[1]], modulation="未知")
        assert _cell_text(window, rows[1], IMPORT_COL_NAME) == "record_b.npy"
        assert "目标" not in _status_text(window, rows[1])
        _apply_params(window, [rows[1]], modulation="AM")
        assert "目标 1 条" in _status_text(window, rows[1])

        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert set(assets) == {"甲信号", "AM · recor"}
    for asset in assets.values():
        target = workspace.list_targets(asset["id"], with_current=True)[0]
        assert (target["for_detection"], target["for_amc"]) == (0, 1)


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
            writer.writerow(["文件", "采集时间"])
            writer.writerow(["capture_a.npy", "2026-09-01T08:30:00Z"])
            writer.writerow(["capture_b.npy", "2026-09-02T09:00:00Z"])
            writer.writerow(["capture_b.npy", "2026-09-03T09:00:00Z"])  # 与上一行冲突
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(manifest), "CSV"))
        window.import_csv_manifest()
        wait_job(app, window)
        report = window.import_csv_report.text()
        assert "2 个文件带采集时间" in report
        assert "未匹配/非法 1 行" in report
        assert "采集时间与同一文件的其他行不一致" in report
        rows = [window._import_row_by_path(str(item)) for item in (first, second, third)]
        _apply_params(window, rows, rate="48000")
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
def test_csv_manifest_attaches_amc_targets_and_names(tmp_path, monkeypatch):
    """CSV 标注清单：信号名称列、旧检测列已忽略报告、仅 AMC 目标、未匹配/非法逐条列出。"""
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
            writer.writerow(["文件", "信号名称", "粒度", "起止单位", "起始", "结束",
                             "调制", "频率下限Hz", "频率上限Hz", "SNR", "备注"])
            writer.writerow(["record_c.npy", "C 信号", "whole_record", "毫秒", "0", "1",
                             "QPSK", "-12000", "12000", "15", "旧检测列已忽略"])
            writer.writerow(["record_c.npy", "", "session", "采样点", "", "",
                             "AM", "", "", "", "只给调制"])
            writer.writerow(["missing.npy", "", "whole_record", "采样点", "0", "",
                             "QPSK", "", "", "", "文件不在清单"])
            writer.writerow(["record_d.npy", "", "whole_record", "采样点", "10", "5",
                             "QPSK", "", "", "", "终点不大于起点"])
            writer.writerow(["record_d.npy", "", "whole_record", "采样点", "0", "",
                             "", "-12000", "12000", "", "缺调制"])
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(manifest), "CSV"))
        window.import_csv_manifest()
        wait_job(app, window)
        report = window.import_csv_report.text()
        assert "已挂接 2 条目标" in report
        assert "未匹配/非法 3 行" in report
        assert "当前文件清单中没有该文件" in report and "终点必须大于起点" in report
        assert "目标行必须给出“调制”" in report
        assert "已忽略列" in report and "频率下限Hz" in report and "备注" in report
        assert "1 条目标带 SNR" in report
        row = window._import_row_by_path(str(first))
        assert "目标 2 条" in _status_text(window, row)
        assert _cell_text(window, row, IMPORT_COL_NAME) == "C 信号"
        # 毫秒换算需要采样率
        rows = [row, window._import_row_by_path(str(second))]
        _apply_params(window, rows, rate="48000")
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    asset = next(item for item in workspace.list_assets() if item["name"] == "C 信号")
    targets = {item["scope"]: item for item in
               workspace.list_targets(asset["id"], with_current=True)}
    assert set(targets) == {"whole_record", "session"}
    whole = targets["whole_record"]
    assert (whole["for_detection"], whole["for_amc"]) == (0, 1)
    assert whole["current"]["source"] == "import"
    assert whole["current"]["sample_start"] == 0
    assert whole["current"]["sample_end"] == 48   # 1 ms × 48000 / 1000
    assert whole["current"]["modulation"] == "QPSK"
    # 频率列仍被忽略；SNR 列写入参考参数（口径 declared）
    assert whole["current"]["f_low_hz"] is None
    assert whole["current"]["snr_db"] == pytest.approx(15.0)
    assert whole["current"]["snr_definition"] == "declared"
    session = targets["session"]
    assert (session["for_detection"], session["for_amc"]) == (0, 1)
    assert session["current"]["modulation"] == "AM"
    assert session["current"]["snr_db"] is None
    untouched = next(item for item in workspace.list_assets()
                     if item["name"] == "record_d.npy")
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
        _apply_params(window, rows, rate="48000")
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
def test_panel_free_text_modulation_and_amc_only(tmp_path, monkeypatch):
    """面板：调制可自由输入（字典外原名保留）；导入只建 AMC 目标。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = _make_npy(tmp_path, "form_a.npy", 64)
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(source)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        _apply_params(window, [0], rate="48000")
        _apply_params(window, [0], modulation="2FSK")  # 字典外自由文本
        assert "目标 1 条" in _status_text(window, 0)
        assert _cell_text(window, 0, IMPORT_COL_MOD) == "2FSK"
        assert _cell_text(window, 0, IMPORT_COL_NAME) == "2FSK · form_"
        window.start_import_batch()
        wait_job(app, window)
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    asset = workspace.list_assets()[0]
    assert asset["name"] == "2FSK · form_"
    target = workspace.list_targets(asset["id"], with_current=True)[0]
    assert target["scope"] == "whole_record"
    assert (target["for_detection"], target["for_amc"]) == (0, 1)
    current = target["current"]
    assert current["modulation"] == "2FSK"  # 字典外原名保留，类别映射留给 AMC 侧
    assert (current["sample_start"], current["sample_end"]) == (0, 64)


@pytest.mark.gui
def test_panel_snr_requires_target_and_updates_existing(tmp_path, monkeypatch):
    """面板 SNR：无目标时不应用并提示；有目标时新建/更新目标的参考参数；清目标同步清空。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = _make_npy(tmp_path, "snr_a.npy", 64)
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(source)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        _apply_params(window, [0], rate="48000")

        # 尚无目标：SNR 不落库，提示先申报调制
        _apply_params(window, [0], snr="12")
        assert "SNR 未应用" in window.status.text()
        assert "目标" not in _status_text(window, 0)
        assert _cell_text(window, 0, IMPORT_COL_SNR) == ""

        # 调制 + SNR 一次应用：目标带上参考参数，清单列显示数值
        _apply_params(window, [0], modulation="QPSK", snr="12")
        assert _cell_text(window, 0, IMPORT_COL_SNR) == "12"
        # 单独改 SNR：面板载入公共值，应用后更新已有目标
        _select_rows(window.import_table, [0])
        assert window.import_snr.text() == "12"
        _apply_params(window, [0], snr="18.5")
        assert _cell_text(window, 0, IMPORT_COL_SNR) == "18.5"
        # 调制“未知”= 清除目标，SNR 列随之清空
        _apply_params(window, [0], modulation="未知")
        assert _cell_text(window, 0, IMPORT_COL_SNR) == ""
        # 非法 SNR：就地提示且不生效
        _apply_params(window, [0], modulation="QPSK", snr="abc")
        assert "SNR 应为数值" in window.status.text()
        assert _cell_text(window, 0, IMPORT_COL_SNR) == ""
    finally:
        window.close()
        app.processEvents()


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
        _apply_params(window, [0], rate="48000")
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


@pytest.mark.gui
def test_sigmf_pair_only_registers_metadata_row(tmp_path, monkeypatch):
    """SigMF 双文件按一条记录计：数据文件归一为元数据行；先后添加/补配不产生重复。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    meta = write_samples(tmp_path / "incoming" / "pair_a",
                         np.ones(32, dtype=np.complex64), "sigmf", sample_rate=48000)
    data = meta.with_suffix(".sigmf-data")
    meta_text = meta.read_text(encoding="utf-8")
    meta.unlink()  # 先制造“只有数据文件”的场景
    window = MainWindow(tmp_path)
    window.show()
    try:
        # 元数据缺失：数据文件保持原样并明确报错，不静默
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(data)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        assert window.import_table.rowCount() == 1
        assert window._import_row_info(0)["path"] == str(data)
        assert "SigMF 元数据文件缺失" in _status_text(window, 0)

        # 补回元数据后再添加：只保留元数据行，数据行被覆盖移除
        meta.write_text(meta_text, encoding="utf-8")
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(meta)], "数据"))
        window.choose_import_files()
        wait_job(app, window)
        assert window.import_table.rowCount() == 1
        assert window._import_row_info(0)["path"] == str(meta)

        # 两个文件一起选（或只选数据）：均视为已有，不新增
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames",
                            lambda *args: ([str(meta), str(data)], "数据"))
        window.choose_import_files()
        assert window.import_table.rowCount() == 1
        assert "没有新增文件" in window.status.text()

        assert _status_text(window, 0) == "就绪"
        window.start_import_batch()
        wait_job(app, window)
        assert window.import_table.rowCount() == 0
    finally:
        window.close()
        app.processEvents()

    workspace = Workspace(tmp_path)
    assets = workspace.list_assets()
    assert len(assets) == 1
    assert assets[0]["sample_count"] == 32


def test_import_files_per_file_payload_and_legacy(tmp_path):
    """import_files：逐文件名称/采样率/AMC 目标（分片），旧 paths 请求与旧检测字段兼容规则。"""
    first = _make_npy(tmp_path, "payload_a.npy", 32)
    second = _make_npy(tmp_path, "payload_b.npy", 48)
    result = execute({"workspace": str(tmp_path), "action": "import_files",
                      "batch_shard": True, "rf_center_hz": 100e6, "files": [
                          {"path": str(first), "sample_rate": 1000.0, "label": "甲",
                           "rf_center_hz": 433e6, "name": "甲信号",
                           "capture_started_at": "2026-09-01T04:05:06Z",
                           "targets": [{"scope": "whole_record", "start": 0, "end": 32,
                                        "start_unit": "samples", "modulation": "QPSK",
                                        "snr_db": 12.5}]},
                          {"path": str(second), "sample_rate": 2000.0,
                           "targets": [{"scope": "whole_record", "start": 0, "end": 48,
                                        "start_unit": "samples", "modulation": "AM",
                                        "f_low_hz": -100.0, "f_high_hz": 100.0}]}]})
    assert result["created"] == 2 and result["failed"] == 0
    assert result["targets_total"] == 2 and result["shard_id"]
    by_path = {item["path"]: item for item in result["results"]}
    # 旧检测字段（频率）不再写入，逐文件回报“已忽略”
    assert by_path[str(second)]["ignored"] == ["频率下限", "频率上限"]
    workspace = Workspace(tmp_path)
    assets = {item["name"]: item for item in workspace.list_assets()}
    assert set(assets) == {"甲信号", "AM · paylo"}  # 后者自动命名：AM + payload_b 前 5 字符
    assert assets["甲信号"]["sample_rate"] == 1000.0
    assert assets["甲信号"]["label"] == "甲"
    assert assets["AM · paylo"]["sample_rate"] == 2000.0
    # 逐文件优先；未给出回落请求级（本批默认）
    assert assets["甲信号"]["rf_center_hz"] == pytest.approx(433e6)
    assert assets["AM · paylo"]["rf_center_hz"] == pytest.approx(100e6)
    assert assets["甲信号"]["capture_started_at"] == "2026-09-01T04:05:06Z"
    assert assets["AM · paylo"]["capture_started_at"]  # 缺省记本批导入时间
    assert all(item["storage_kind"] == "shard" for item in assets.values())
    target = workspace.list_targets(assets["甲信号"]["id"], with_current=True)[0]
    assert (target["for_detection"], target["for_amc"]) == (0, 1)
    assert target["current"]["modulation"] == "QPSK"
    assert target["current"]["snr_db"] == pytest.approx(12.5)
    assert target["current"]["snr_definition"] == "declared"
    second_target = workspace.list_targets(assets["AM · paylo"]["id"],
                                           with_current=True)[0]
    assert second_target["current"]["modulation"] == "AM"
    assert second_target["current"]["f_low_hz"] is None  # 旧频带不再落库

    third = _make_npy(tmp_path, "payload_c.npy", 16)
    legacy = execute({"workspace": str(tmp_path), "action": "import_files",
                      "paths": [str(third)], "sample_rate": 3000.0,
                      "rf_center_hz": 50e6, "capture_started_at": "2026-09-02T00:00:00Z",
                      "targets": [{"scope": "segment", "modulation": "64QAM"}]})
    assert legacy["created"] == 1
    workspace = Workspace(tmp_path)
    third_asset = next(item for item in workspace.list_assets()
                       if item["name"] == "64QAM · paylo")
    assert third_asset["rf_center_hz"] == pytest.approx(50e6)
    assert third_asset["capture_started_at"] == "2026-09-02T00:00:00Z"
    legacy_target = workspace.list_targets(third_asset["id"], with_current=True)[0]
    assert legacy_target["scope"] == "segment"
    assert (legacy_target["for_detection"], legacy_target["for_amc"]) == (0, 1)

    # 只有旧检测字段、没有调制的目标行：拒绝并给出指引，资产不入库
    fourth = _make_npy(tmp_path, "payload_d.npy", 16)
    rejected = execute({"workspace": str(tmp_path), "action": "import_files",
                        "files": [{"path": str(fourth), "sample_rate": 1000.0,
                                   "targets": [{"scope": "segment", "f_low_hz": -50.0,
                                                "f_high_hz": 50.0}]}]})
    assert rejected["created"] == 0 and rejected["failed"] == 1
    assert "仅支持 AMC" in rejected["results"][0]["error"]
    assert all(item["name"] != "payload_d.npy"
               for item in Workspace(tmp_path).list_assets())


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


def _import_candidates(tmp_path, count, prefix="cancel"):
    files = []
    for index in range(count):
        path = _make_npy(tmp_path, f"{prefix}_{index}.npy", 32)
        files.append({"path": str(path), "sample_rate": 1000.0})
    return files


def test_import_files_cooperative_cancel_before_start(tmp_path):
    """预置 cancel.flag：导入在文件边界直接停止，返回成功/失败/未处理计数。"""
    import json

    files = _import_candidates(tmp_path, 5)
    job_dir = tmp_path / "job_pre"
    job_dir.mkdir()
    (job_dir / "cancel.flag").write_text("1", encoding="utf-8")
    result = execute({"workspace": str(tmp_path), "action": "import_files",
                      "batch_shard": True, "files": files, "job_dir": str(job_dir)})
    assert result["cancelled"] is True
    assert (result["created"], result["failed"], result["unprocessed"]) == (0, 0, 5)
    assert result["results"] == [] and result["shard_id"] is None
    progress = json.loads((job_dir / "progress.json").read_text(encoding="utf-8"))
    assert (progress["done"], progress["total"]) == (0, 5)
    assert progress["message"] == "已取消"
    assert Workspace(tmp_path).list_assets() == []


def test_import_files_cancel_stops_at_file_boundary(tmp_path, monkeypatch):
    """块内取消：已完成文件保留（分片封存），未处理计数正确。"""
    from signal_analysis.services import imports as imports_service

    files = _import_candidates(tmp_path, 4)
    state = {"checks": 0}

    class _Reporter:
        def __init__(self, job_dir=None):
            self.directory = None

        def emit(self, *args, **kwargs):
            pass

        def cancelled(self):
            state["checks"] += 1
            return state["checks"] > 1

    monkeypatch.setattr(imports_service, "Reporter", _Reporter)
    result = execute({"workspace": str(tmp_path), "action": "import_files",
                      "batch_shard": True, "files": files})
    assert result["cancelled"] is True
    assert result["created"] == 1
    assert result["unprocessed"] == 3
    assert result["shard_id"]  # 已写数据封存为分片而不是丢弃
    assert [item["name"] for item in Workspace(tmp_path).list_assets()] == ["cancel_0.npy"]


def test_import_inspect_cooperative_cancel(tmp_path):
    """识别链同样在文件边界响应取消，并回报未处理数量。"""
    first = _make_npy(tmp_path, "inspect_x.npy", 16)
    second = _make_npy(tmp_path, "inspect_y.npy", 16)
    job_dir = tmp_path / "job_inspect"
    job_dir.mkdir()
    (job_dir / "cancel.flag").write_text("1", encoding="utf-8")
    result = execute({"workspace": str(tmp_path), "action": "import_inspect",
                      "paths": [str(first), str(second)], "job_dir": str(job_dir)})
    assert result["cancelled"] is True
    assert result["files"] == []
    assert result["unprocessed"] == 2


def test_import_files_chunk_orchestration(monkeypatch):
    """900 个文件拆成 500+400 两块：进度换算整批口径、分片带块号、结果合并。"""
    from signal_analysis.ui import runner as runner_module

    calls, names, reports = [], [], []

    def fake_run_job(request, **kwargs):
        calls.append(len(request["files"]))
        names.append(request.get("shard_name"))
        progress = kwargs.get("progress")
        if progress is not None:
            progress({"done": 0, "total": len(request["files"]),
                      "message": "正在导入 x.npy"})
        results = [{"path": item["path"], "ok": True} for item in request["files"]]
        return {"kind": "import_files", "results": results, "created": len(results),
                "failed": 0, "collection_id": "col-1", "collection_name": "批次集合",
                "shard_id": f"shard-{len(calls)}", "targets_total": 0,
                "initial_labels": 0}

    monkeypatch.setattr(runner_module, "run_job", fake_run_job)
    files = [{"path": f"/x/{index}.npy", "sample_rate": 1000.0} for index in range(900)]
    merged = runner_module._run_task({"action": "import_files", "batch_shard": True,
                                      "files": files}, progress=reports.append)
    assert calls == [500, 400]
    assert names == ["导入批次 1/2（500 个文件）", "导入批次 2/2（400 个文件）"]
    assert reports == [{"done": 0, "total": 900, "message": "第 1/2 块 · 正在导入 x.npy"},
                       {"done": 500, "total": 900, "message": "第 2/2 块 · 正在导入 x.npy"}]
    assert merged["created"] == 900 and merged["unprocessed"] == 0
    assert merged["cancelled"] is False
    assert merged["shard_ids"] == ["shard-1", "shard-2"]
    assert merged["shard_id"] == "shard-1"
    assert merged["collection_name"] == "批次集合" and len(merged["results"]) == 900


def test_import_files_chunk_cancel_between_blocks(monkeypatch):
    """块边界发现取消：后续块不再启动，未处理按整批口径汇总。"""
    import threading

    from signal_analysis.ui import runner as runner_module

    cancel = threading.Event()
    calls = []

    def fake_run_job(request, **kwargs):
        calls.append(len(request["files"]))
        kwargs["cancel"].set()  # 模拟体在第一块执行期间收到取消
        results = [{"path": item["path"], "ok": True} for item in request["files"]]
        return {"kind": "import_files", "results": results, "created": len(results),
                "failed": 0, "collection_id": None, "collection_name": None,
                "shard_id": f"shard-{len(calls)}", "targets_total": 0,
                "initial_labels": 0, "cancelled": False}

    monkeypatch.setattr(runner_module, "run_job", fake_run_job)
    files = [{"path": f"/x/{index}.npy"} for index in range(900)]
    merged = runner_module._run_task({"action": "import_files", "files": files},
                                     cancel=cancel)
    assert calls == [500]
    assert merged["cancelled"] is True
    assert merged["created"] == 500 and merged["unprocessed"] == 400


def test_import_files_cancel_before_first_block_raises(monkeypatch):
    """启动即取消（第一块未执行）：以取消而不是空结果收场。"""
    import threading

    from signal_analysis.ui import runner as runner_module

    cancel = threading.Event()
    cancel.set()
    calls = []
    monkeypatch.setattr(runner_module, "run_job",
                        lambda request, **kwargs: calls.append(request) or {})
    with pytest.raises(RuntimeError, match="任务已取消"):
        runner_module._run_task({"action": "import_files",
                                 "files": [{"path": f"/x/{i}"} for i in range(600)]},
                                cancel=cancel)
    assert calls == []


@pytest.mark.gui
def test_import_cancel_status_text(tmp_path):
    """取消后的状态文案：成功 / 失败 / 未处理 一目了然。"""
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    try:
        text = window.result_status({"kind": "import_files", "cancelled": True,
                                     "created": 3, "failed": 1, "unprocessed": 96,
                                     "collection_name": "大集合"})
        assert "导入已取消" in text
        assert "成功 3" in text and "失败 1" in text and "未处理 96" in text
        assert "大集合" in text
    finally:
        window.close()
