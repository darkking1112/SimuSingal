import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets
from signal_analysis.gui import MainWindow, SignalParamsDialog


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
def test_sigmf_generate_import_gui(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    try:
        from signal_analysis.dataio import write_samples
        path = write_samples(tmp_path / "reference", np.ones(32), "sigmf", sample_rate=12345)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(path), ""))
        window.sample_rate.setValue(48000)
        window.import_file()
        wait_job(app, window)
        assert window.selected_asset()["sample_rate"] == 12345
        index = window.gen_export_format.findData("sigmf")
        assert index >= 0
        window.gen_export_format.setCurrentIndex(index)
        assert not window.gen_endian.isEnabled()
        window.gen_duration.setValue(.01)
        window.generate_iq_clicked()
        wait_job(app, window)
        assert len(list((tmp_path / "exports").glob("*.sigmf-meta"))) == 1
        assert len(list((tmp_path / "exports").glob("*.sigmf-data"))) == 1
        assert ".sigmf-data" in window.gen_result.text()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_analysis_gui_workflow(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window.tabs.count() == 6
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


@pytest.mark.gui
def test_compare_page_pairs_ai_with_baseline(tmp_path):
    """对比页在 AI 检测结果上并排列出传统基线（这里用契约化的 payload 直接驱动渲染）。"""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        def metrics(matched, precision, snr_mae):
            return {"true": 2, "matched": matched, "missed": 2 - matched, "false_alarm": 0,
                    "precision": precision, "recall": matched / 2, "f1": precision,
                    "center_mae_hz": 500.0, "bandwidth_mape": 0.02, "snr_mae_db": snr_mae}
        window.tab_results[2] = {
            "kind": "ml_detect", "asset_id": "a1", "asset_name": "合成数据",
            "algorithm": "onnx_yolox_v1", "contract": "detect_result_v1",
            "model": {"id": "dut", "version": "1.0.0"},
            "summary": {"detections": [{}, {}], "threshold_dbfs_per_hz": 3.0,
                        "model": {"id": "dut", "version": "1.0.0"}},
            "metrics": metrics(2, 1.0, 0.4), "baseline_metrics": metrics(1, 0.5, 0.9)}
        window.compare_from_detect()
        assert window.tabs.currentIndex() == 4
        table = window.compare_table
        assert table.rowCount() == 20  # 10 项指标 × 两条路径
        assert [table.horizontalHeaderItem(i).text() for i in range(4)] == ["环节", "对象", "指标", "取值"]
        objects = {table.item(row, 1).text() for row in range(table.rowCount())}
        assert any(text.startswith("AI 检测 · dut@1.0.0") for text in objects)
        assert "传统基线（能量检测）" in objects
        text = window.compare_summary.toPlainText()
        assert "并排对比" in text and "传统基线" in text
        # 没有识别结果时不编造内容，只提示需要先跑一次
        window.tab_results.pop(3, None)
        window._render_compare()
        assert "识别模型来源" not in window.compare_summary.toPlainText()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_fh_edit_dialog_roundtrip():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rate = 1_000_000.0
    # 自动均布保存的 spec（含 hop_count 与显式频偏/符号速率）重新打开不得崩溃
    dialog = SignalParamsDialog("fh_rc", rate,
                                {"hop_count": 8, "deviation": 20000.0, "symbol_rate": 40000.0})
    try:
        assert dialog.hop_auto.isChecked()
        assert not dialog.hop_points_edit.isEnabled()
        out = dialog.spec()
        assert out["hop_count"] == 8
        assert "hop_points" not in out
        assert "hop_span" not in out
        assert "hop_bandwidth" not in out
        assert out["deviation"] == 20000.0
        assert out["symbol_rate"] == 40000.0
    finally:
        dialog.close()
    # 手动三参数（中心跨度/单跳带宽）保存后重新打开，自动勾选框关闭且回读一致
    dialog = SignalParamsDialog("fh_rc", rate,
                                {"hop_count": 8, "hop_span": 80000.0, "hop_bandwidth": 10000.0})
    try:
        assert dialog.hop_span.isEnabled()
        assert dialog.hop_bandwidth.isEnabled()
        out = dialog.spec()
        assert out["hop_span"] == 80000.0
        assert out["hop_bandwidth"] == 10000.0
        assert "中心跨度" in dialog.auto_label.text()
    finally:
        dialog.close()
    # 显式频点列表：自动勾选框关闭、列表可用且回读一致
    dialog = SignalParamsDialog("fh_rc", rate, {"hop_points": [50000.0, 120000.0, 190000.0]})
    try:
        assert not dialog.hop_auto.isChecked()
        assert dialog.hop_points_edit.isEnabled()
        assert dialog.spec()["hop_points"] == [50000.0, 120000.0, 190000.0]
    finally:
        dialog.close()
    # 取消自动但未填频点：切换只显示红色错误提示，不抛出未捕获异常
    dialog = SignalParamsDialog("fh_rc", rate, {})
    try:
        dialog.hop_auto.setChecked(False)
        assert dialog.hop_points_edit.isEnabled()
        assert "参数无效" in dialog.auto_label.text()
    finally:
        dialog.close()
    app.processEvents()


@pytest.mark.gui
def test_iq_generation_gui_workflow(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window.tabs.count() == 6
        window.tabs.setCurrentIndex(1)
        assert window.gen_signals.rowCount() == 0
        window.add_iq_signal({"mode": "qpsk", "offset": 100_000.0, "power_dbfs": -10.0,
                              "bandwidth": 100_000.0})
        assert window.gen_signals.rowCount() == 1
        assert "QPSK" in window.gen_name.text()
        # 二进制导出走完整 write_samples 路径。
        window.gen_export_format.setCurrentIndex(3)  # iq16
        assert window.gen_endian.isEnabled()
        window.generate_button.click()
        wait_job(app, window)
        assert window.last_result["kind"] == "generate"
        assert window.assets.count() == 1
        assert window.assets.currentItem().data(0x0100)["id"] == window.last_result["id"]
        assert "已生成资产" in window.gen_result.text()
        assert window.last_result["export_format"] == "iq16"
        export = tmp_path / "exports" / f"{window.last_result['id']}.bin"
        assert export.exists()
        assert export.stat().st_size > 0
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_detection_tab_workflow(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        window.tabs.setCurrentIndex(1)
        window.gen_duration.setValue(0.5)
        window.gen_rate.setValue(1_000_000.0)
        window.gen_snr.setValue(20.0)
        window.add_iq_signal({"mode": "qpsk", "offset": 120_000.0, "power_dbfs": -10.0,
                              "bandwidth": 100_000.0})
        window.generate_button.click()
        wait_job(app, window)
        # 带宽门限默认跟随检测门限的一半，可切换为手动并受上界约束
        assert window.detect_band_auto.isChecked()
        assert not window.detect_band_threshold.isEnabled()
        window.detect_threshold.setValue(0.5)
        assert 0 < window.detect_band_threshold.value() <= 0.5
        window.detect_threshold.setValue(3.0)
        assert window.detect_band_threshold.value() == pytest.approx(1.5)
        window.detect_band_auto.setChecked(False)
        assert window.detect_band_threshold.isEnabled()
        window.detect_band_threshold.setValue(2.0)
        window.detect_threshold.setValue(1.0)
        assert window.detect_band_threshold.value() <= 1.0
        window.detect_threshold.setValue(3.0)
        window.detect_band_auto.setChecked(True)
        window.tabs.setCurrentIndex(2)
        window.detect_button.click()
        wait_job(app, window)
        result = window.last_result
        assert result["kind"] == "detect"
        assert result["contract"] == "detect_result_v1"
        assert result["algorithm"] == "energy_detect_v1"
        assert len(result["summary"]["detections"]) == 1
        assert window.detect_table.rowCount() == 1
        assert len(window._detect_items) == 1
        assert window.detect_tf_image.image.ndim == 2
        assert window.detect_spectrum.listDataItems()
        assert result["metrics"]["matched"] == 1
        assert 0 < result["metrics"]["center_mae_hz"] < 2000.0
        text = window.detect_summary.toPlainText()
        assert "噪声本底" in text and "真值" in text and "中心频率" in text
        # “算法对比”页随检测结果同步填充；传统检测路径没有对照基线，会明确说明
        assert window.compare_table.rowCount() == 10
        assert "没有同步运行传统基线" in window.compare_summary.toPlainText()
        # 参数控件确实透传到后端配置
        window.detect_nfft.setCurrentText("1024")
        window.detect_merge.setCurrentText("8")
        window.detect_max.setValue(4)
        window.detect_button.click()
        wait_job(app, window)
        config = window.last_result["summary"]["config"]
        assert window.last_result["summary"]["nfft"] == 1024
        assert config["merge_bins"] == 8 and config["max_detections"] == 4
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_constellation_and_playback(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        window.tabs.setCurrentIndex(1)
        window.gen_duration.setValue(1.0)
        window.add_iq_signal({"mode": "qpsk", "offset": 100_000.0, "power_dbfs": -10.0,
                              "bandwidth": 100_000.0})
        window.generate_button.click()
        wait_job(app, window)
        assert window.assets.count() == 1
        window.assets.setCurrentItem(window.assets.item(0))
        window.analyze_mode.setCurrentText("概览分析")
        window.analyze_button.click()
        wait_job(app, window)
        assert window.last_result["summary"]["classification"] == "digital"
        assert window.tf_stack.currentWidget() is window.constellation
        assert window.const_scatter.data.size > 0
        assert window.waterfall_image.image.ndim == 2
        # 手动判定模拟 → 切回时频图
        window.class_combo.setCurrentText("模拟")
        assert window.tf_stack.currentWidget() is window.time_frequency
        assert window.tf_image.image.ndim == 2
        window.class_combo.setCurrentText("自动")
        # 波形与频谱为固定范围，可在界面上调整
        def ranges():
            return (window.wave.viewRange(), window.spectrum.viewRange())

        wave_y = window.wave.viewRange()[1]
        assert wave_y == pytest.approx((-window.amp_max.value(), window.amp_max.value()))
        assert window.wave.viewRange()[0] == pytest.approx((0.0, 1.0))
        spec_x = window.spectrum.viewRange()[0]
        assert spec_x == pytest.approx((-500_000.0, 500_000.0))
        assert window.spec_span.value() == 0.0
        # 手动调整幅度与时窗 → 转为固定范围并立即生效
        window.amp_max.setValue(0.25)
        assert not window.range_follow.isChecked()
        assert window.wave.viewRange()[1] == pytest.approx((-0.25, 0.25))
        window.spec_span.setValue(150_000.0)
        assert window.spectrum.viewRange()[0] == pytest.approx((-150_000.0, 150_000.0))
        window.spec_db_span.setValue(40.0)
        low, high = window.spectrum.viewRange()[1]
        assert high - low == pytest.approx(40.0)
        assert window.wave.viewRange()[0] == pytest.approx((0.0, 1.0))
        window.wave_span.setCurrentText("100 ms")
        wave_x = window.wave.viewRange()[0]
        assert wave_x[1] - wave_x[0] == pytest.approx(0.1)
        # 勾选「范围随数据」重新按数据定标幅度与频宽，时窗与动态范围保持用户设置
        window.wave_span.setCurrentText("整个记录")
        window.range_follow.setChecked(True)
        assert window.spec_span.value() == 0.0
        with np.load(window.workspace.root / window.last_result["plots_path"],
                     allow_pickle=False) as arrays:
            peak = max(np.abs(arrays["wave_i"]).max(), np.abs(arrays["wave_q"]).max())
        assert peak <= window.amp_max.value() <= 3.0 * peak
        assert window.wave.viewRange()[0] == pytest.approx((0.0, 1.0))
        assert window.wave.viewRange()[1] == pytest.approx(
            (-window.amp_max.value(), window.amp_max.value()))
        assert window.spectrum.viewRange()[0] == pytest.approx((-500_000.0, 500_000.0))
        assert window.spec_db_span.value() == pytest.approx(40.0)
        # 「仅正频率」时瀑布图必须只取正半轴，与平均功率谱密度共用同一频率范围
        bins = window._nfft_value()
        window.freq_view.setCurrentText("仅正频率")
        assert window.spectrum.viewRange()[0] == pytest.approx((0.0, 500_000.0))
        assert window.waterfall.viewRange()[0] == pytest.approx((0.0, 500_000.0))
        assert window.waterfall_image.image.shape[1] == bins // 2
        window.freq_view.setCurrentText("双边")
        assert window.waterfall.viewRange()[0] == pytest.approx((-500_000.0, 500_000.0))
        assert window.waterfall_image.image.shape[1] == bins
        window.freq_view.setCurrentText("自动")
        # 调整动态范围即转为手动固定范围，播放时沿用该数值
        window.spec_db_span.setValue(80.0)
        assert not window.range_follow.isChecked()
        # 实时播放：进度推进、暂停冻结、停止复位
        window.analyze_mode.setCurrentText("实时播放")
        assert window.play_bar.isVisible()
        assert window.play_window.currentText() == "200 ms"
        window.play_window.setCurrentText("10 ms")
        # 放慢到 0.25×，使 1 s 记录播放 4 s，测试期间不会播完
        window.play_speed.setCurrentText("0.25×")
        window.analyze_button.click()
        assert window._play_data is not None
        assert "滚动时间窗" in window.waterfall.getPlotItem().titleLabel.text
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and window.play_progress.value() == 0:
            app.processEvents()
            time.sleep(.02)
        assert window.play_progress.value() > 0
        assert window.waterfall_image.image.ndim == 2
        # 播放中瀑布图与频谱的频率范围一致，切换「仅正频率」立即同步
        window.freq_view.setCurrentText("仅正频率")
        app.processEvents()
        assert window.spectrum.viewRange()[0] == pytest.approx((0.0, 500_000.0))
        assert window.waterfall.viewRange()[0] == pytest.approx(window.spectrum.viewRange()[0])
        assert window.waterfall_image.image.shape[1] == window._nfft_value() // 2
        window.freq_view.setCurrentText("自动")
        app.processEvents()
        assert window.waterfall.viewRange()[0] == pytest.approx(window.spectrum.viewRange()[0])
        # 播放中波形与频谱范围保持不变；波形横轴为最近一窗，随时间平移
        def selected_span():
            value, unit = window.play_window.currentText().split()
            return float(value) * (0.001 if unit == "ms" else 1.0)

        play_wave = window.wave.viewRange()
        play_spec = window.spectrum.viewRange()
        assert play_wave[1] == pytest.approx((-window.amp_max.value(), window.amp_max.value()))
        assert play_wave[0][1] - play_wave[0][0] == pytest.approx(selected_span())
        assert play_spec[0] == pytest.approx((-500_000.0, 500_000.0))
        assert play_spec[1][1] - play_spec[1][0] == pytest.approx(80.0)
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.02)
        assert window.wave.viewRange()[1] == pytest.approx(play_wave[1])
        assert window.spectrum.viewRange()[1] == pytest.approx(play_spec[1])
        assert window.spectrum.viewRange()[0] == pytest.approx(play_spec[0])
        assert window.wave.viewRange()[0][1] > play_wave[0][1]
        assert (window.wave.viewRange()[0][1] - window.wave.viewRange()[0][0]
                == pytest.approx(selected_span()))
        # 瀑布图纵轴锁定固定时长窗口，且播放时随时间滚动
        def yspan():
            low, high = window.waterfall.viewRange()[1]
            return high - low, high

        # 10 ms 时间窗：轴跨度等于窗口，且窗内仍有多帧，不随刷新周期退化为一两行
        span, top = yspan()
        assert span == pytest.approx(0.01)
        assert window.waterfall_image.image.shape[0] > 2
        window.play_window.setCurrentText("50 ms")
        app.processEvents()
        assert yspan()[0] == pytest.approx(0.05)
        window.play_window.setCurrentText("5 s")
        app.processEvents()
        assert yspan()[0] == pytest.approx(5.0)
        top = yspan()[1]
        deadline = time.monotonic() + .6
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.02)
        assert yspan()[1] > top
        assert yspan()[0] == pytest.approx(5.0)
        window.play_window.setCurrentText("2 s")
        app.processEvents()
        assert yspan()[0] == pytest.approx(2.0)
        window.play_window.setCurrentText("5 s")
        app.processEvents()
        frozen = yspan()[1]
        window.pause_button.click()
        app.processEvents()
        time.sleep(.1)
        assert window._play_paused
        assert window.play_progress.value() > 0
        assert yspan()[1] == frozen
        window.stop_button.click()
        assert window._play_data is None
        assert "离线历史" in window.waterfall.getPlotItem().titleLabel.text
        # 切换资产时自动停止播放
        window.analyze_button.click()
        assert window._play_data is not None
        window.asset_changed()
        assert window._play_data is None
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_ml_controls_follow_runtime_availability(tmp_path, monkeypatch):
    """推理运行时缺失时禁用 AI 入口并给出安装提示，传统检测路径不受影响。"""
    from signal_analysis.ml import runtime as ml_runtime

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    try:
        widgets = (window.ml_button, window.ml_choose, window.ml_score, window.ml_iou,
                   window.ml_compare)
        monkeypatch.setattr(ml_runtime, "runtime_version", lambda: None)
        window.update_ml_controls()
        assert not any(widget.isEnabled() for widget in widgets)
        status = window.ml_status.text()
        assert "onnxruntime" in status and "[ml]" in status
        assert window.detect_button.isEnabled()
        monkeypatch.setattr(ml_runtime, "runtime_version", lambda: "1.17.0")
        window.update_ml_controls()
        assert all(widget.isEnabled() for widget in widgets)
        assert window.ml_status.text() == "onnxruntime 1.17.0"
    finally:
        window.close()
        app.processEvents()
