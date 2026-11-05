"""Offscreen GUI tests for the manual signal-drill page."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.drill.catalog import KEYS
from signal_analysis.ui import MainWindow


pytestmark = pytest.mark.gui


@pytest.fixture
def drill_window(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path / "ws")
    window.resize(1440, 950)
    window.show()
    app.processEvents()
    page = window.signal_drill_page
    # The page generates lazily; opening its tab is the user-facing trigger.
    window.tabs.setCurrentIndex(window._page_index("信号判读训练"))
    app.processEvents()
    assert page.round is not None
    yield app, window, page
    page.timer.stop()
    window.close()
    app.processEvents()


def _force_key(page, key):
    page.session._bag = [key]
    page.session._last_key = None
    page._on_restart()


def test_page_is_lazy_and_has_all_controls(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path / "ws")
    window.show()
    app.processEvents()
    try:
        assert window.tabs.count() == 12
        labels = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert labels[-1] == "信号判读训练"
        page = window.signal_drill_page
        assert page.round is None  # hidden tab does not pay for a waveform
        assert len(page.type_buttons) == len(KEYS) == 20
        assert page.aux_tabs.count() == 3
        assert page.time_window.currentText() == "5 ms"
        assert page.time_seek.maximum() == 5000
        assert not hasattr(page, "time_playhead")
        assert not hasattr(page, "tf_playhead")
        assert page.restart_button.text() == "重新开始"
        assert not hasattr(page, "select_button")
        assert page.answer_button.text() == "显示答案"
        window.tabs.setCurrentIndex(window._page_index("信号判读训练"))
        app.processEvents()
        assert page.round is not None
        assert page.play_button.text() in ("▶ 播放", "▶ 继续")
    finally:
        window.close()
        app.processEvents()


def test_time_domain_uses_raw_samples_and_seek_slider(drill_window):
    app, window, page = drill_window
    page.time_window.setCurrentIndex(page.time_window.findData(0.005))
    page.session.seek(0.0)
    page._refresh_position()
    app.processEvents()

    round_data = page.round
    fs = round_data.fs_hz
    expected = round_data.waveform.samples[: int(round(0.005 * fs))]
    _, actual = page.time_trace.getData()
    np.testing.assert_allclose(actual, expected.astype(np.float64))
    assert not hasattr(page, "time_fill")

    page._on_time_seek(2500)
    app.processEvents()
    assert page.session.position_s == pytest.approx(2.5)
    assert page.time_seek.value() == 2500
    start_s = 2.5 - 0.0025
    first = int(round(start_s * fs))
    last = int(round((start_s + 0.005) * fs))
    expected = round_data.waveform.samples[first:last]
    _, actual = page.time_trace.getData()
    np.testing.assert_allclose(actual, expected.astype(np.float64))


def test_cumulative_spectrum_grows_with_playhead(drill_window):
    app, window, page = drill_window
    _force_key(page, "fh_fsk2")
    app.processEvents()
    spectrum = page.round.spectrum

    page.session.seek(0.0)
    page._refresh_position()
    first = np.asarray(page.cum_max.getData()[1]).copy()
    np.testing.assert_allclose(first, spectrum.progressive_max_db[0], atol=1e-4)
    assert not np.allclose(first, spectrum.cumulative_max_db)

    page.session.seek(2.5)
    page._refresh_position()
    middle = np.asarray(page.cum_max.getData()[1]).copy()
    index = spectrum.frame_index(2.5)
    np.testing.assert_allclose(middle, spectrum.progressive_max_db[index],
                               atol=1e-4)
    assert not np.allclose(middle, first)

    page.session.seek(5.0)
    page._refresh_position()
    np.testing.assert_allclose(page.cum_max.getData()[1],
                               spectrum.cumulative_max_db, atol=1e-4)
    assert "5.00" in page.cum_subtitle.text()

    # Revealing the answer deliberately exposes the full-record evidence.
    page.session.seek(1.0)
    page._refresh_position()
    page.session.select(page.round.spec.key)
    page._on_answer()
    app.processEvents()
    np.testing.assert_allclose(page.cum_max.getData()[1],
                               spectrum.cumulative_max_db, atol=1e-4)
    assert "完整累积" in page.cum_subtitle.text()


@pytest.mark.parametrize("key, tab_index", [
    ("qam16", 0), ("qpsk", 0), ("fh_fsk2", 2), ("am", 2), ("fsk8", 2)])
def test_auxiliary_tab_follows_question_type(drill_window, key, tab_index):
    app, window, page = drill_window
    _force_key(page, key)
    app.processEvents()
    assert page.round.spec.key == key
    assert page.aux_tabs.currentIndex() == tab_index
    # PSK/QAM and burst/hopping types open the auxiliary card by default.
    if key in ("qam16", "qpsk", "fh_fsk2", "fsk8"):
        assert page.aux_tabs.isVisible()
    else:
        assert not page.aux_tabs.isVisible()


def test_play_pause_resume_end_and_replay(drill_window):
    app, window, page = drill_window
    page.play_button.click()
    assert page.session.phase == "playing"
    assert page.play_button.text() == "⏸ 暂停"
    page.play_button.click()
    assert page.session.phase == "paused"
    assert page.play_button.text() == "▶ 继续"
    page.play_button.click()
    assert page.session.phase == "playing"
    # Fast-forward without waiting five real seconds.
    page.session.advance(10.0)
    page.timer.stop()
    page._refresh_position()
    page._update_play_label()
    assert page.session.phase == "ended"
    assert page.play_button.text() == "↻ 重播"
    page.play_button.click()
    assert page.session.phase == "playing"
    assert page.session.position_s == 0.0
    page.play_button.click()
    assert page.session.phase == "paused"


def test_select_and_reveal_wrong_answer(drill_window):
    app, window, page = drill_window
    truth = page.round.spec.key
    wrong = next(key for key in KEYS if key != truth)
    page.type_buttons[wrong].setChecked(True)
    app.processEvents()
    assert page.session.selected_key == wrong
    assert "当前选择" in page.selection_label.text()
    page.answer_button.click()
    app.processEvents()
    assert page.session.revealed
    assert page.answer_browser.toPlainText().startswith("✗ 错误")
    assert page.type_buttons[truth].text().startswith("✓")
    assert page.type_buttons[wrong].text().startswith("✗")
    assert not page.answer_button.isEnabled()
    assert not page.type_buttons[truth].isEnabled()
    # Answering again is a no-op.
    page._on_answer()
    assert page.session.revealed


def test_correct_answer_and_restart(drill_window):
    app, window, page = drill_window
    truth = page.round.spec.key
    page.type_buttons[truth].setChecked(True)
    app.processEvents()
    page.answer_button.click()
    app.processEvents()
    assert page.session.correct_count == 1
    assert page.answer_browser.toPlainText().startswith("✓ 正确")
    old_round = page.round
    page.restart_button.click()
    app.processEvents()
    assert page.round is not old_round
    assert not page.session.revealed
    assert page.session.selected_key is None
    assert page.session.correct_count == 1  # score survives a new question
    assert page.selection_label.text() == "未选择"
    assert page.answer_button.isEnabled()


def test_constellation_ideal_overlay_after_answer(drill_window):
    app, window, page = drill_window
    _force_key(page, "qam16")
    app.processEvents()
    page.type_buttons["qam16"].setChecked(True)
    page.answer_button.click()
    app.processEvents()
    # The constellation tab is active and has both received points and an
    # ideal overlay; the exact item count is not part of the public contract.
    assert page.aux_tabs.currentIndex() == 0
    assert page.round.waveform.meta["constellation_size"] == 16
    items = page.const_plot.getPlotItem().listDataItems()
    assert len(items) >= 2


def test_every_catalog_class_renders_without_error(drill_window):
    app, window, page = drill_window
    # The per-class render sweep does not need the full five seconds; the
    # normal flow already covers the 5 s UI elsewhere.
    page.session.duration_s = 1.0
    for key in KEYS:
        _force_key(page, key)
        app.processEvents()
        assert page.round.spec.key == key
        assert len(page.cum_max.getData()[0]) > 0
        assert len(page.instant_curve.getData()[0]) > 0
        assert page.tf_image.image is not None
        if page.round.info.constellation_size:
            assert len(page.const_plot.getPlotItem().listDataItems()) >= 1
        else:
            # The non-applicable tab shows a text hint instead of a scatter.
            assert page.round.info.constellation_size == 0
