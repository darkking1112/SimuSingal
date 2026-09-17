import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore, QtWidgets
from signal_analysis.gui import MainWindow, SignalParamsDialog, _mirrored_spectrum


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
        assert window.tabs.count() == 9
        assert not hasattr(window, "sim_button")
        # 演示入口属于“IQ 信号生成”页，不在数据分析页（与侧栅同一父级）。
        generator = window._page_index("IQ 信号生成")
        analysis = window._page_index("数据分析")
        assert window.tabs.widget(generator).isAncestorOf(window.demo_button)
        assert not window.tabs.widget(analysis).isAncestorOf(window.demo_button)
        window.demo_button.click()
        wait_job(app, window)
        assert window.assets.count() == 1
        # 演示点数跟随生成页“采样率 × 持续时间”，不再固定 8192。
        assert window.selected_asset()["sample_count"] == int(round(
            window.gen_rate.value() * window.gen_duration.value()))
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
        # 演示点数超出 1～16,000,000 时只提示，不启动作业
        window.gen_rate.setValue(1e9)
        window.gen_duration.setValue(10.0)
        window.demo_button.click()
        assert window.active_job is None
        assert "超出" in window.status.text()
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
        window.tab_results["信号检测"] = {
            "kind": "ml_detect", "asset_id": "a1", "asset_name": "合成数据",
            "algorithm": "onnx_yolox_v1", "contract": "detect_result_v1",
            "model": {"id": "dut", "version": "1.0.0"},
            "summary": {"detections": [{}, {}], "threshold_dbfs_per_hz": 3.0,
                        "model": {"id": "dut", "version": "1.0.0"}},
            "metrics": metrics(2, 1.0, 0.4), "baseline_metrics": metrics(1, 0.5, 0.9)}
        window.compare_from_detect()
        assert window.tabs.currentIndex() == window._page_index("算法对比")
        table = window.compare_table
        assert table.rowCount() == 20  # 10 项指标 × 两条路径
        assert [table.horizontalHeaderItem(i).text() for i in range(4)] == ["环节", "对象", "指标", "取值"]
        objects = {table.item(row, 1).text() for row in range(table.rowCount())}
        assert any(text.startswith("AI 检测 · dut@1.0.0") for text in objects)
        assert "传统基线（能量检测）" in objects
        text = window.compare_summary.toPlainText()
        assert "并排对比" in text and "传统基线" in text
        # 没有识别结果时不编造内容，只提示需要先跑一次
        window.tab_results.pop("调制识别", None)
        window._render_compare()
        assert "调制识别" not in window.compare_summary.toPlainText()
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
        assert window.tabs.count() == 9
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
        # 时频图必须按「行 = 时间、列 = 频率」绘制：数组亮斑的坐标要与检测中心
        # 频率/时刻一致。若把矩阵转置后交给 row-major 的 ImageItem，时频图会横过来，
        # 亮斑的横坐标会变成时间（负半轴凭空出现能量），这里的断言会立即失败。
        with np.load(tmp_path / result["plots_path"], allow_pickle=False) as arrays:
            matrix = arrays["spectrogram_db"]
            frequency = arrays["frequency"]
            frame_time = arrays["frame_time"]
        assert window.detect_tf_image.image.shape == matrix.shape
        row, col = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        cell = window.detect_tf_image.mapToScene(QtCore.QPointF(col + 0.5, row + 0.5))
        point = window.detect_tf.getPlotItem().getViewBox().mapSceneToView(cell)
        step_hz = abs(float(frequency[1] - frequency[0]))
        step_s = float(result["summary"]["hop_samples"]) / float(
            result["summary"]["sample_rate_hz"])
        assert point.x() == pytest.approx(float(frequency[col]), abs=step_hz / 2)
        assert point.y() == pytest.approx(float(frame_time[row]), abs=step_s / 2)
        assert frequency[col] == pytest.approx(
            result["summary"]["detections"][0]["center_hz"], abs=3 * step_hz)
        # 复数 IQ 的双边谱都可能有信号，「自动」保留双边；切到「仅正频率」时
        # 平均功率谱、时频图与检测框共用同一非负频率范围
        assert window.detect_freq_view.currentText() == "自动"
        assert window.detect_spectrum.viewRange()[0] == pytest.approx((-500_000.0, 500_000.0))
        window.detect_freq_view.setCurrentText("仅正频率")
        assert window.detect_spectrum.viewRange()[0] == pytest.approx((0.0, 500_000.0))
        assert window.detect_tf.viewRange()[0] == pytest.approx((0.0, 500_000.0))
        assert window.detect_tf_image.image.shape[1] == matrix.shape[1] // 2
        assert "仅正频率" in window.detect_tf.getPlotItem().titleLabel.text
        window.detect_freq_view.setCurrentText("双边")
        assert window.detect_tf_image.image.shape == matrix.shape
        assert window.detect_spectrum.viewRange()[0] == pytest.approx((-500_000.0, 500_000.0))
        window.detect_freq_view.setCurrentText("自动")
        assert window.detect_tf_image.image.shape == matrix.shape
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
def test_detect_page_frequency_display_for_real_record(tmp_path):
    """实数记录的「自动」只显示非负频率；契约里没有实/复标记，判据只能是 PSD 镜像。

    实数记录的负半轴只是正半轴的复共轭，画出来是重复的信息；复数 IQ 的双边谱
    都可能有信号，必须保留。这里同时锁定时频图不再被转置绘制。
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rate = 200_000.0
    length = int(rate * 0.04)
    time_axis = np.arange(length) / rate
    rng = np.random.default_rng(11)
    samples = (np.cos(2 * np.pi * 15_000.0 * time_axis)
               + 0.02 * rng.standard_normal(length)).astype(np.float32)
    # 镜像判据本身：0 Hz 与奈奎斯特频点自配对，逐点比较必须排除它们
    assert _mirrored_spectrum([0.0, 1.0, 2.0, 3.0, 9.0, 3.0, 2.0, 1.0]) is True
    assert _mirrored_spectrum([0.0, 1.0, 2.0, 3.0, 9.0, 4.0, 5.0, 6.0]) is False
    assert _mirrored_spectrum(np.zeros(7)) is False
    window = MainWindow(tmp_path)
    window.show()
    try:
        window.workspace.add_samples(samples, rate, "实采记录", "imported:iq16")
        window.refresh_assets()
        window.assets.setCurrentItem(window.assets.item(0))
        window.tabs.setCurrentIndex(2)
        window.detect_nfft.setCurrentText("256")
        window.detect_button.click()
        wait_job(app, window)
        result = window.last_result
        assert result["kind"] == "detect"
        with np.load(tmp_path / result["plots_path"], allow_pickle=False) as arrays:
            bins = arrays["frequency"].size
            matrix = arrays["spectrogram_db"]
            assert _mirrored_spectrum(arrays["spectrum_db"]) is True
        # 自动 → 折叠到非负频率，平均功率谱、时频图与检测框共用同一范围
        assert window.detect_freq_view.currentText() == "自动"
        assert window.detect_spectrum.viewRange()[0] == pytest.approx((0.0, rate / 2))
        assert window.detect_tf.viewRange()[0] == pytest.approx((0.0, rate / 2))
        assert window.detect_tf_image.image.shape == (matrix.shape[0], bins // 2)
        assert "仅正频率" in window.detect_tf.getPlotItem().titleLabel.text
        # 手动切到双边时负半轴依然可看（保留既有行为）
        window.detect_freq_view.setCurrentText("双边")
        assert window.detect_spectrum.viewRange()[0] == pytest.approx((-rate / 2, rate / 2))
        assert window.detect_tf_image.image.shape == matrix.shape
        assert "双边" in window.detect_tf.getPlotItem().titleLabel.text
    finally:
        window.close()
        app.processEvents()


def _ml_render_result(tmp_path, baseline_boxes):
    """构造最小 ``ml_detect`` 结果与 plots.npz，只覆盖渲染路径所需字段。"""
    rate = 1_000_000.0
    nfft = 64
    frames = 8
    frequency = np.linspace(-rate / 2, rate / 2, nfft, endpoint=False)
    frame_time = np.linspace(0.0, 0.01, frames)
    spectrogram = np.full((frames, nfft), -100.0, dtype=np.float64)
    spectrogram[:, nfft // 2 - 2: nfft // 2 + 2] = -60.0
    arrays = {
        "frequency": frequency, "frame_time": frame_time,
        "spectrogram_db": spectrogram,
        "spectrum_db": spectrogram.mean(axis=0),
        "spectrum_median_db": spectrogram.mean(axis=0),
        "threshold_db": np.array([-97.0]), "noise_floor_db": np.array([-100.0]),
        "detection_boxes": np.asarray([[0.0, 100_000.0, 0.001, 0.009],
                                       [-200_000.0, -100_000.0, 0.001, 0.009]]),
        "baseline_detection_boxes": np.asarray(baseline_boxes, dtype=np.float64).reshape(-1, 4),
    }
    relative = "runs/fake-ml/plots.npz"
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **arrays)
    detection = {"method": "ml", "centroid_hz": 50_000.0, "bandwidth_hz": 100_000.0,
                 "t_start_s": 0.001, "t_end_s": 0.009, "power_dbfs": -10.0,
                 "snr_db": 20.0, "session_id": None, "hopping": False, "sub_bands": 1,
                 "bin_count": 4, "label": "emitter", "model": "fake@1.0"}
    detections = [dict(detection, id=1, center_hz=50_000.0,
                       f_low_hz=0.0, f_high_hz=100_000.0, confidence=0.5,
                       occupied_f_low_hz=10_000.0, occupied_f_high_hz=90_000.0),
                  dict(detection, id=2, center_hz=-150_000.0,
                       f_low_hz=-200_000.0, f_high_hz=-100_000.0, confidence=0.4,
                       occupied_f_low_hz=-190_000.0, occupied_f_high_hz=-110_000.0)]
    summary = {
        "algorithm": "ml_detect:fake@1.0", "contract": "detect_result_v1",
        "snr_definition": "inband_snr_v1", "frequency_reference": "baseband_offset",
        "sample_count": 10_000, "sample_rate_hz": rate, "duration_s": 0.01, "nfft": nfft,
        "hop_samples": 64, "frame_count": frames, "freq_resolution_hz": rate / nfft,
        "noise_floor_dbfs_per_hz": -100.0, "threshold_dbfs_per_hz": -97.0,
        "config": {"nfft": nfft, "threshold_db": 3.0, "band_threshold_db": 1.5,
                   "min_bandwidth_hz": 100.0, "min_duration_s": 0.0,
                   "max_detections": 32, "merge_bins": 2, "score_threshold": 0.05,
                   "iou_threshold": 0.5, "image_size": 64, "dynamic_range_db": 60.0},
        "model": {"id": "fake", "version": "1.0", "sha256": "0" * 64,
                  "manifest_path": "model.json", "library": "detector.onnx",
                  "labels": ["emitter"], "training": {}, "runtime_version": "test"},
        "image": {"layout": "time_frequency_grayscale_v1", "size": 64,
                  "db_floor": -80.0, "db_ceiling": -20.0},
        "raw_boxes": {"output_shape": [1, 8, 6], "rows": 8, "candidates": 2,
                      "score_threshold": 0.05},
        "timing": {"context_ms": 1.0, "inference_ms": 1.0, "total_ms": 2.0},
        "detections": detections,
    }
    return {"kind": "ml_detect", "asset_id": "a" * 32, "asset_name": "渲染回归",
            "plots_path": relative, "summary": summary}


@pytest.mark.gui
@pytest.mark.parametrize("baseline", [[], [[-200_000.0, -100_000.0, 0.001, 0.009]]])
def test_ml_detection_render_fills_table(tmp_path, baseline):
    """AI 检测渲染必须填充明细表格与摘要。

    ``baseline_boxes`` 曾是 numpy 数组却用 ``if baseline_boxes`` 判空——布尔上下文
    直接 ValueError：图上的框画完后渲染中断，表格与摘要整块空白（用户可见症状）。
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    try:
        window.display_result(_ml_render_result(tmp_path, baseline))
        app.processEvents()
        assert window.detect_table.rowCount() == 2
        assert window.detect_table.item(0, 1).text() != ""
        assert window.detect_table.item(1, 6).text().endswith("dB")
        assert "检出目标 2 个" in window.detect_summary.toPlainText()
        title = window.detect_tf.getPlotItem().titleLabel.text
        assert "时频图与检测框" in title
        if baseline:
            assert "灰虚线传统基线" in title
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


@pytest.mark.gui
def test_data_management_tab_scan_and_cleanup_gating(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert window.tabs.tabText(window._page_index("数据管理")) == "数据管理"
        window.tabs.setCurrentIndex(window._page_index("数据管理"))
        assert not window.storage_cleanup_button.isEnabled()
        window.storage_scan_button.click()
        wait_job(app, window)
        assert "工作区" in window.storage_overview.text()
        assert len(window.storage_chart.listDataItems()) >= 1
        # 空工作区各表格给出一行占位而不是空白
        assert window.storage_asset_table.item(0, 0).text() == "（无）"
        assert window.storage_issue_table.item(0, 0).text() == "无"
        assert "没有可清理项" in window.storage_cleanup_hint.text()
        # 普通扫描不解锁删除，必须先“预览清理”
        assert not window.storage_cleanup_button.isEnabled()
        window.storage_retention.setValue(0)
        assert not window.storage_cleanup_button.isEnabled()
        window.storage_preview_button.click()
        wait_job(app, window)
        assert window.storage_cleanup_table.rowCount() >= 1
        assert "运行中" in window.storage_cleanup_hint.text()
        # 预览后仍需勾选才能删除
        assert not window.storage_cleanup_button.isEnabled()
        window.storage_cleanup_table.item(0, 0).setCheckState(QtCore.Qt.CheckState.Checked)
        assert window.storage_cleanup_button.isEnabled()
        # 改动保留天数后权限立即失效，避免用旧清单删除
        window._retention_changed(7)
        assert not window.storage_cleanup_button.isEnabled()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_data_management_scan_and_export_never_write_runs(tmp_path, monkeypatch):
    import json

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        window.storage_scan_button.click()
        wait_job(app, window)
        assert window.history.count() == 0  # 只读盘点不写运行记录
        assert "未写入运行记录" in window.status.text()
        output = tmp_path / "storage_report.json"
        monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                            lambda *args: (str(output), "JSON (*.json)"))
        window.export_storage_report()
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["kind"] == "storage_report"
        assert payload["root"] == str(tmp_path)
        assert window.history.count() == 0
        assert "已导出" in window.status.text()
        # 未扫描时导出给出提示而不是报错
        window._storage_report = None
        window.export_storage_report()
        assert "请先扫描" in window.status.text()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_asset_selection_reports_file_in_status_bar(tmp_path):
    """选中数据资产时状态栏第二行给出文件名/位置/大小/资产/导出；文件缺失时明示而不报错。"""
    from signal_analysis.maintenance import format_bytes

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        assert not window.status_detail.isVisible()  # 未选数据时不占位
        window.demo_button.click()
        wait_job(app, window)
        window.assets.setCurrentRow(0)
        window.asset_changed()
        asset = window.selected_asset()
        path = tmp_path / asset["path"]
        text = window.status_detail.text()
        assert text.startswith(f"文件名 {asset['name']}")
        assert f"位置 {asset['path']}" in text  # 相对工作目录，不重复写工作目录本身
        assert str(tmp_path) not in text
        assert f"大小 {format_bytes(path.stat().st_size)}" in text
        assert f"{asset['sample_count']:,} 复采样" in text
        assert f"{asset['sample_rate']:g} Hz" in text
        assert "资产：" in text and "complex64" in text
        assert "内置生成 tones_v1" in text  # 内置生成的数据没有外部源文件
        # 演示动作不导出（不读导出格式下拉框），所以导出字段如实写“无”
        assert "导出：无" in text
        assert window.status_detail.isVisible()
        # 完整文本保留在 text()/tooltip，界面只显示省略后的字符串
        assert window.status_detail.toolTip() == text
        # 任务状态写在第一行，不会覆盖资产信息（两者是分开的控件）
        window.status.setText("任务运行中……")
        assert window.status_detail.text() == text
        # 文件被外部删除时提示“文件缺失”，而不是抛异常或静默空白
        path.unlink()
        window.asset_changed()
        assert "文件缺失" in window.status_detail.text()
        # 清空选择后第二行隐藏，回到不占位状态
        window.assets.setCurrentRow(-1)
        window.asset_changed()
        assert window.status_detail.text() == ""
        assert not window.status_detail.isVisible()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.gui
def test_asset_status_reports_imported_source_format(tmp_path, monkeypatch):
    """导入的资产要把原始来源格式一并写进状态栏（source 记录的是原始绝对路径）。"""
    from signal_analysis.dataio import write_samples

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        samples = np.exp(2j * np.pi * 0.01 * np.arange(4096)).astype(np.complex64)
        source = write_samples(tmp_path / "reference", samples, "npy", sample_rate=250_000)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(source), ""))
        window.import_file()
        wait_job(app, window)
        window.assets.setCurrentRow(0)
        window.asset_changed()
        asset = window.selected_asset()
        assert asset["source"] == str(source)
        text = window.status_detail.text()
        assert f"位置 {asset['path']}" in text
        assert "complex64" in text and "导入 NPY" in text
        assert asset["name"] in text
    finally:
        window.close()
        app.processEvents()


def _export_label(text):
    """从状态栏第二行取出“导出：”字段（到行尾）。"""
    return text.split("导出：", 1)[1]


@pytest.mark.gui
def test_asset_status_reports_export_files(tmp_path, monkeypatch):
    """导出字段从 exports/ 下按 <asset_id>.* 现扫现算：SigMF 成对文件只计一条，

    没有导出物写“无”，且重启后（无内存状态）依然能显示。
    """
    from signal_analysis.dataio import write_samples

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(tmp_path)
    window.show()
    try:
        # 先造一个没有导出物的资产
        samples = np.exp(2j * np.pi * 0.01 * np.arange(4096)).astype(np.complex64)
        source = write_samples(tmp_path / "reference", samples, "npy", sample_rate=250_000)
        monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                            lambda *args: (str(source), ""))
        window.import_file()
        wait_job(app, window)
        imported = window.selected_asset()
        assert _export_label(window.status_detail.text()) == "无"

        # SigMF 双文件是按“一次导出”计的：只报元数据那条，不重复报 .sigmf-data
        window.gen_export_format.setCurrentIndex(window.gen_export_format.findData("sigmf"))
        window.gen_duration.setValue(.01)
        window.generate_iq_clicked()
        wait_job(app, window)
        generated = window.selected_asset()
        assert generated["id"] != imported["id"]
        exports = _export_label(window.status_detail.text())
        assert exports.count("SigMF 双文件") == 1
        assert f"exports/{generated['id']}.sigmf-meta" in exports
        assert ".sigmf-data" not in exports

        # 换回没有导出物的那个资产，字段要跟着变（不是缓存的一次性快照）
        for row in range(window.assets.count()):
            item = window.assets.item(row)
            if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == imported["id"]:
                window.assets.setCurrentItem(item)
                break
        assert _export_label(window.status_detail.text()) == "无"

        # int16 交织二进制无法从后缀判断量化类型，按字节数反推
        window.gen_export_format.setCurrentIndex(window.gen_export_format.findData("iq16"))
        window.generate_iq_clicked()
        wait_job(app, window)
        exports = _export_label(window.status_detail.text())
        assert "交织 IQ 二进制 · int16" in exports
        assert "exports/" in exports and ".bin" in exports
    finally:
        window.close()
        app.processEvents()


def test_asset_exports_helper_reads_disk_state(tmp_path):
    """_asset_exports 只看磁盘：目录缺失、无匹配、成对 SigMF、无宿主名字段都安全。"""
    from signal_analysis.gui import _asset_exports, _iq_binary_kind

    root = tmp_path / "exports"
    asset = {"id": "a" * 32, "sample_count": 100}
    assert _asset_exports(root, asset) == []  # 目录还不存在
    root.mkdir()
    assert _asset_exports(root, asset) == []  # 空目录
    # 别人资产的导出物不能被串到本资产名下
    (root / f"{'b' * 32}.csv").write_bytes(b"0,0\n")
    assert _asset_exports(root, asset) == []
    # SigMF 成对：只报元数据；只有数据文件（半成品）时如实报出来
    (root / f"{asset['id']}.sigmf-meta").write_text("{}")
    (root / f"{asset['id']}.sigmf-data").write_bytes(b"\0" * 800)
    assert _asset_exports(root, asset) == [f"SigMF 双文件（exports/{asset['id']}.sigmf-meta）"]
    (root / f"{asset['id']}.sigmf-meta").unlink()
    assert _asset_exports(root, asset) == [f"SigMF 双文件（exports/{asset['id']}.sigmf-data）"]
    # 未知后缀不隐藏，照原样列出来（多个导出物按文件名排序，.dat 在 .sigmf-data 之前）
    (root / f"{asset['id']}.dat").write_bytes(b"\0")
    assert _asset_exports(root, asset) == [
        f"dat 文件（exports/{asset['id']}.dat）",
        f"SigMF 双文件（exports/{asset['id']}.sigmf-data）",
    ]
    # 每复采样 4 B = int16、8 B = float32，对不上就说“类型未知”
    assert _iq_binary_kind(400, 100) == "int16"
    assert _iq_binary_kind(800, 100) == "float32"
    assert _iq_binary_kind(123, 100) == "类型未知"
    assert _iq_binary_kind(800, 0) == "类型未知"  # 零采样不猜
