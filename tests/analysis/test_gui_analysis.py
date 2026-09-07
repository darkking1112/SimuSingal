import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets
from signal_analysis.gui import MainWindow


def wait_job(app, window):
    deadline = time.monotonic() + 15
    while window.active_job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert window.active_job is None, "GUI task did not finish"
    assert "未完成" not in window.status.text(), window.status.text()
    assert "失败" not in window.status.text(), window.status.text()


@pytest.mark.gui
def test_analysis_gui_workflow(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window.tabs.count() == 2
        assert not hasattr(window, "sim_button")
        window.demo_button.click()
        wait_job(app, window)
        assert window.assets.count() == 1
        window.analyze_button.click()
        wait_job(app, window)
        assert window.last_result["kind"] == "analysis"
        assert window.tf_image.image.ndim == 2
        assert len(window.wave.listDataItems()) == 2
        window.label.setText("GUI 参考备注")
        window.save_label()
        assert window.selected_asset()["label"] == "GUI 参考备注"
        assert window.history.count() == 1
        window.tabs.setCurrentIndex(0)
        output = tmp_path / "analysis.json"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(output), "JSON"))
        window.export_current()
        import json
        assert json.loads(output.read_text(encoding="utf-8"))["kind"] == "analysis"
    finally:
        window.close()
        app.processEvents()
