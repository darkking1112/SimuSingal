"""“采纳为参数标注”按钮的离屏 GUI 测试（方案 §4.2、§7.4）。"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.ui import MainWindow


def wait_job(app, window, timeout=60.0):
    deadline = time.monotonic() + timeout
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, window.status.text()


@pytest.mark.gui
def test_adopt_buttons_enable_and_adopt_result(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        # 初始没有结果：三个采纳按钮都禁用，点击只提示
        assert not window.adopt_detect_button.isEnabled()
        assert not window.adopt_hops_button.isEnabled()
        assert not window.adopt_amc_button.isEnabled()
        window.adopt_page_result("信号检测")
        assert "还没有可采纳" in window.status.text()
        # 生成一段 qpsk 资产并跑检测（采样率保持默认 1 MHz：噪声带宽与页内默认一致）
        window.gen_duration.setValue(0.08)
        window.add_iq_signal({"mode": "qpsk", "offset": 40_000.0, "power_dbfs": -8.0,
                              "bandwidth": 60_000.0})
        window.generate_button.click()
        wait_job(app, window)
        window.refresh_assets()
        window.assets.setCurrentRow(0)
        window.detect_button.click()
        wait_job(app, window)
        assert window.last_result["kind"] == "detect"
        assert window.adopt_detect_button.isEnabled()
        assert window.adopt_run_ids["信号检测"] == window.last_result["run_id"]
        asset_id = window.selected_asset()["id"]
        before = len(window.workspace.list_target_versions(
            window.workspace.list_targets(asset_id)[0]["id"]))
        window.adopt_detect_button.click()
        wait_job(app, window)
        assert "已采纳为参数标注" in window.status.text()
        # 目标参考参数追加 source=algorithm 版本；侧栏详情同步显示“来源 算法”
        target = window.workspace.list_targets(asset_id)[0]
        versions = window.workspace.list_target_versions(target["id"])
        assert len(versions) == before + 1
        assert versions[-1]["source"] == "algorithm"
        assert "来源 算法" in window.target_list.item(0).text()
        # 调制识别结果同样可采纳
        window.amc_button.click()
        wait_job(app, window)
        assert window.adopt_amc_button.isEnabled()
        window.adopt_amc_button.click()
        wait_job(app, window)
        assert "已采纳为参数标注" in window.status.text()
        # 采纳跟随识别结论（弱模型可能误判）：modulation 与预测标签一致
        mapping = {"fm": "FM", "ssb": "SSB", "ask2": "2ASK", "qpsk": "QPSK",
                   "qam16": "16QAM", "qam64": "64QAM"}
        label = window.last_result["prediction"]["label"]
        current = window.workspace.current_target_version(target["id"])
        assert current["source"] == "algorithm"
        assert current["modulation"] == mapping.get(label, label.upper())
        # 当前页没有结果时（切换后的新结果缺失）按钮状态随之更新
        assert not window.adopt_hops_button.isEnabled()
    finally:
        window.close()
