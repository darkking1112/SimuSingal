import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets
from communication_sim.gui import MainWindow


@pytest.mark.gui
def test_simulation_gui_is_independent(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window.tabs.count() == 2
        assert not hasattr(window, "assets")
        assert not hasattr(window, "native_button")
        window.sim_button.click()
        deadline = time.monotonic() + 15
        while window.active_job and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.02)
        app.processEvents()
        assert window.active_job is None
        assert "失败" not in window.status.text(), window.status.text()
        assert window.last_result["kind"] == "simulation"
        window.replay.setValue(0)
        assert window.event_table.rowCount() == 0
        window.replay.setValue(len(window.events))
        assert window.event_table.rowCount() == len(window.events)
        assert window.history.count() == 1
    finally:
        window.close()
        app.processEvents()
