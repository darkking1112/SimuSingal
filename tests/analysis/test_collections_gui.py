"""信号集合侧栏与数据管理子页的离屏 GUI 测试（方案文档 §7.3、§7.5）。"""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.core_api import generate_iq
from signal_analysis.ui import MainWindow


def wait_job(app, window, timeout=30.0):
    deadline = time.monotonic() + timeout
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, "GUI 任务未在预期时间内结束"


def add_generated(window, name, seed=1, mode="fm"):
    samples, summary = generate_iq(
        200_000.0, 0.02,
        [{"mode": mode, "offset": 0.0, "bandwidth": 40_000.0, "power_dbfs": -6.0}],
        {"enabled": True, "snr_db": 12.0}, seed)
    return window.workspace.add_samples(samples, 200_000.0, name,
                                        f"generated:iq_{mode}_v1",
                                        metadata={"generation": summary})


@pytest.mark.gui
def test_sidebar_collection_filter_and_detail(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        collection = window.workspace.create_collection("集合甲")
        assets = [add_generated(window, f"样本{index}", seed=index) for index in range(5)]
        for asset in assets[:3]:
            window.workspace.add_collection_member(collection["id"], asset["id"])
        window.refresh_collections()
        window.refresh_assets()
        assert window.asset_total.text() == "共 5 条"
        assert window.assets.count() == 5
        # 集合下拉：全部 / 零散 / 集合甲（含资产数）
        assert window.collection_combo.itemText(0) == "全部资产"
        assert window.collection_combo.itemText(1) == "零散资产"
        index = window.collection_combo.findData(collection["id"])
        assert index >= 0 and "集合甲（3）" in window.collection_combo.itemText(index)
        # 按集合筛选
        window.collection_combo.setCurrentIndex(index)
        assert window.assets.count() == 3 and window.asset_total.text() == "共 3 条"
        # 零散资产
        scattered = window.collection_combo.findData("__scattered__")
        window.collection_combo.setCurrentIndex(scattered)
        assert window.assets.count() == 2
        # 搜索与集合组合
        window.search.setText("样本4")
        assert window.assets.count() == 1 and window.asset_total.text() == "共 1 条"
        window.search.clear()
        window.collection_combo.setCurrentIndex(0)
        # 目标详情：生成资产带会话目标（只读列表）
        item = window.assets.item(0)
        window.assets.setCurrentItem(item)
        assert window.target_list.count() == 1
        text = window.target_list.item(0).text()
        assert "s0" in text and "整条记录" in text
        assert "检测✓" in text and "AMC✓" in text
        assert "待标注" in text
    finally:
        window.close()


@pytest.mark.gui
def test_sidebar_collection_scope_drives_generation(tmp_path):
    """生成页不再自带目标集合：产物按左侧选中项归属，零散时不写初始标注。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        collection = window.workspace.create_collection("生成目标集")
        window.refresh_collections()
        window.add_iq_signal({"mode": "fm", "offset": 0.0, "power_dbfs": -6.0,
                              "bandwidth": 40_000.0})
        window.gen_duration.setValue(.01)
        # 生成页不再自带目标集合下拉与导出格式：目标集合只能用左侧选中项
        assert not hasattr(window, "gen_collection")
        assert not hasattr(window, "gen_export_format")

        # 侧栏选“全部资产/零散资产”：不入集合，初始标注不可用
        window.collection_combo.setCurrentIndex(0)
        assert not window.gen_initial_labels.isEnabled()
        window.gen_initial_labels.setChecked(True)  # 置灰前的勾选会被清掉
        window.generate_iq_clicked()
        wait_job(app, window)
        scattered = window.last_result
        assert scattered["collection_id"] is None and scattered["initial_labels"] == 0
        assert window.workspace.list_collections()[0]["asset_count"] == 0

        # 侧栏选中集合：产物入集合，初始标注生效
        index = window.collection_combo.findData(collection["id"])
        window.collection_combo.setCurrentIndex(index)
        assert window.gen_initial_labels.isEnabled()
        window.gen_initial_labels.setChecked(True)
        window.generate_iq_clicked()
        wait_job(app, window)
        joined = window.last_result
        assert joined["collection_id"] == collection["id"]
        assert joined["collection_name"] == "生成目标集"
        assert joined["initial_labels"] > 0
        assert window.workspace.list_collections()[0]["asset_count"] == 1
        assert "已加入集合" in window.gen_result.text()
    finally:
        window.close()


@pytest.mark.gui
def test_sidebar_paging_controls(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        for index in range(105):
            window.workspace.add_samples(np.ones(64, dtype=np.complex64), 48_000.0,
                                         f"批量{index:03d}")
        window.refresh_assets()
        assert window.asset_total.text() == "共 105 条"
        assert window.page_label.text() == "第 1/2 页"
        assert window.assets.count() == 100
        assert not window.prev_page.isEnabled() and window.next_page.isEnabled()
        window.next_page.click()
        assert window.page_label.text() == "第 2/2 页"
        assert window.assets.count() == 5
        assert window.prev_page.isEnabled() and not window.next_page.isEnabled()
        window.prev_page.click()
        assert window.page_label.text() == "第 1/2 页"
        # 搜索把页码夹回有效范围
        window.next_page.click()
        window.search.setText("批量10")
        assert window.asset_total.text() == "共 5 条"
        assert window.page_label.text() == "第 1/1 页"
        window.search.clear()
        # 每页 500：单页显示全部
        window.page_size.setCurrentText("500")
        assert window.page_label.text() == "第 1/1 页"
        assert window.assets.count() == 105
    finally:
        window.close()


@pytest.mark.gui
def test_sidebar_target_label_status(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        asset = add_generated(window, "待标注资产", seed=3)
        collection = window.workspace.create_collection("标注集")
        window.workspace.add_collection_member(collection["id"], asset["id"])
        task_set = window.workspace.create_task_set(collection["id"], "detection")
        window.refresh_assets()
        window.assets.setCurrentRow(0)
        assert "待标注" in window.target_list.item(0).text()
        target = window.workspace.list_targets(asset["id"])[0]
        window.workspace.append_detection_label(task_set["id"], target["id"], source="manual")
        window.asset_changed()
        assert "检测已标注" in window.target_list.item(0).text()
        assert "待标注" not in window.target_list.item(0).text()
    finally:
        window.close()


@pytest.mark.gui
def test_collections_panel_add_remove_archive(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        monkeypatch.setattr(QtWidgets.QInputDialog, "getText",
                            staticmethod(lambda *args, **kwargs: ("面板集合", True)))
        window.new_collection()
        assert window.collection_panel.count() == 1
        collection_id = window.collection_panel.item(0).data(0x0100)
        assert "面板集合" in window.collection_detail.toPlainText()
        # 加入所选资产
        add_generated(window, "面板资产", seed=2)
        window.refresh_assets()
        window.assets.setCurrentRow(0)
        window.collection_panel.setCurrentRow(0)
        window.add_selected_asset_to_collection()
        assert window.workspace.count_assets(collection_id=collection_id) == 1
        combo_index = window.collection_combo.findData(collection_id)
        assert "面板集合（1）" in window.collection_combo.itemText(combo_index)
        # 标注集建立后，面板显示按任务的进度与覆盖度
        window.workspace.create_task_set(collection_id, "detection")
        window.refresh_collections_panel()
        detail = window.collection_detail.toPlainText()
        assert "资产 1 · 目标 1" in detail and "已标注 0/1" in detail
        assert "覆盖度：完整 0" in detail
        # 移除只删成员关系，资产仍在
        window.remove_selected_asset_from_collection()
        assert window.workspace.count_assets(collection_id=collection_id) == 0
        assert window.workspace.count_assets() == 1
        # 归档需要确认；确认后集合不再出现在面板与下拉中
        window.add_selected_asset_to_collection()
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "question",
            staticmethod(lambda *args, **kwargs:
                         QtWidgets.QMessageBox.StandardButton.Yes))
        window.archive_selected_collection()
        assert window.collection_panel.count() == 0
        assert window.workspace.list_collections() == []
        assert window.collection_combo.findData(collection_id) == -1
        assert "已归档" in window.status.text()
    finally:
        window.close()


@pytest.mark.gui
def test_migrate_legacy_button(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        legacy = window.workspace.root / "datasets" / "old"
        (legacy / "images").mkdir(parents=True)
        card = {"sample_count": 0, "splits": {}, "contract": {
            "input_contract": "tf_image_v1", "layout": "time_frequency_grayscale_v1",
            "output_layout": "normalized_boxes_v1", "image_size": 128, "channels": 1,
            "spectrogram_nfft": 256, "dynamic_range_db": 60.0, "labels": ["emitter"],
            "label_semantics": "session_v1"}}
        (legacy / "dataset.json").write_text(json.dumps(card, ensure_ascii=False),
                                             encoding="utf-8")
        (legacy / "samples.jsonl").write_text("", encoding="utf-8")
        window.migrate_button.click()
        wait_job(app, window)
        assert "历史数据登记完成" in window.status.text()
        assert window.collection_panel.count() == 1
        assert "旧标注" in window.collection_detail.toPlainText()
        # 幂等：再次点击不新增集合
        window.migrate_button.click()
        wait_job(app, window)
        assert window.collection_panel.count() == 1
    finally:
        window.close()
