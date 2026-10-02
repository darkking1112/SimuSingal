"""态势显示页（mixin）：统计/时频渲染、频率与幅度显示范围、实时播放与滚动瀑布图。"""
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ...data import Workspace
from ...tasks import run_job
from ..constants import (EXPORT_FORMATS, IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NAME, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..dialogs import SignalParamsDialog
from ..helpers import (_AMC_SOURCE_TEXT, _asset_exports, _asset_format, _comparison_line,
                       _fmt_hz, _fmt_metric, _fmt_span, _iq_binary_kind, _mirrored_spectrum)
from ..runner import _run_task
from ..widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class AnalysisPageMixin:
    def build_analysis(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("通用统计与时频展示 · 调制识别见“调制识别”标签页"), 1)
        bar.addWidget(QtWidgets.QLabel("信号判定"))
        self.class_combo = QtWidgets.QComboBox()
        self.class_combo.addItems(["自动", "数字", "模拟"])
        self.class_combo.currentIndexChanged.connect(lambda *_: self._apply_display_mode())
        bar.addWidget(self.class_combo)
        bar.addWidget(QtWidgets.QLabel("频率显示"))
        self.freq_view = QtWidgets.QComboBox()
        self.freq_view.addItems(["自动", "双边", "仅正频率"])
        self.freq_view.currentIndexChanged.connect(lambda *_: self._apply_display_mode())
        bar.addWidget(self.freq_view)
        bar.addWidget(QtWidgets.QLabel("FFT 点数"))
        self.nfft = QtWidgets.QComboBox()
        self.nfft.addItems(["128", "256", "512", "1024", "2048"])
        self.nfft.setCurrentText("256")
        self.nfft.currentIndexChanged.connect(self._on_playback_nfft)
        bar.addWidget(self.nfft)
        self.analyze_mode = QtWidgets.QComboBox()
        self.analyze_mode.addItems(["概览分析", "实时播放"])
        self.analyze_mode.currentIndexChanged.connect(self._on_analyze_mode)
        bar.addWidget(self.analyze_mode)
        self.analyze_button = QtWidgets.QPushButton("分析所选数据")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.clicked.connect(self.analyze_selected)
        bar.addWidget(self.analyze_button)
        layout.addLayout(bar)
        from common.gui import TaskBanner
        self.analysis_banner = TaskBanner("态势显示", "取消本次分析")
        self.register_task_banner("态势显示", self.analysis_banner)
        layout.addWidget(self.analysis_banner)
        span_bar = QtWidgets.QHBoxLayout()
        span_bar.addWidget(QtWidgets.QLabel("显示范围"))
        self.range_follow = QtWidgets.QCheckBox("范围随数据")
        self.range_follow.setChecked(True)
        self.range_follow.setToolTip("勾选时按当前数据定标波形幅度与频谱频宽；手动改动任一数值即切换为固定范围。"
                                     "无论哪种方式，播放过程中范围都保持不变。")
        self.range_follow.stateChanged.connect(self._on_range_follow)
        span_bar.addWidget(self.range_follow)
        span_bar.addWidget(QtWidgets.QLabel("波形幅度 ±"))
        self.amp_max = QtWidgets.QDoubleSpinBox()
        self.amp_max.setRange(1e-5, 1e6)
        self.amp_max.setDecimals(4)
        self.amp_max.setSingleStep(0.05)
        self.amp_max.setValue(1.0)
        self.amp_max.setToolTip("波形纵轴固定为 ±该值（任意单位）")
        self.amp_max.valueChanged.connect(self._on_range_edited)
        span_bar.addWidget(self.amp_max)
        span_bar.addWidget(QtWidgets.QLabel("波形时窗"))
        self.wave_span = QtWidgets.QComboBox()
        self.wave_span.addItems(["整个记录", "10 ms", "20 ms", "50 ms", "100 ms", "200 ms",
                                 "500 ms", "1 s", "2 s", "5 s"])
        self.wave_span.setToolTip("概览显示记录起点起该时长；播放时显示最近该时长的数据，"
                                 "「整个记录」在播放时跟随瀑布时间窗")
        self.wave_span.currentIndexChanged.connect(self._on_range_edited)
        span_bar.addWidget(self.wave_span)
        span_bar.addWidget(QtWidgets.QLabel("频谱频宽 ±"))
        spec_row, self.spec_span = _freq_spin(0.0, 5e8, 0.0, 2)
        self.spec_span.setToolTip("频谱横轴固定为 ±该频宽；0 表示整个基带（±采样率/2）")
        self.spec_span.valueChanged.connect(self._on_range_edited)
        span_bar.addWidget(spec_row)
        span_bar.addWidget(QtWidgets.QLabel("频谱动态范围"))
        db_row, self.spec_db_span = _plain_spin(10.0, 200.0, 80.0, 0, "dB")
        self.spec_db_span.setToolTip("频谱纵轴与瀑布色标固定为 [峰值 − 该值, 峰值]")
        self.spec_db_span.valueChanged.connect(self._on_range_edited)
        span_bar.addWidget(db_row)
        span_bar.addStretch(1)
        layout.addLayout(span_bar)
        self.play_bar = QtWidgets.QWidget()
        self.play_bar.setVisible(False)
        play_layout = QtWidgets.QHBoxLayout(self.play_bar)
        play_layout.setContentsMargins(0, 0, 0, 0)
        self.play_button = QtWidgets.QPushButton("▶ 播放")
        self.play_button.clicked.connect(self._start_playback)
        play_layout.addWidget(self.play_button)
        self.pause_button = QtWidgets.QPushButton("暂停")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._toggle_pause)
        play_layout.addWidget(self.pause_button)
        self.stop_button = QtWidgets.QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self._stop_playback())
        play_layout.addWidget(self.stop_button)
        play_layout.addWidget(QtWidgets.QLabel("速度"))
        self.play_speed = QtWidgets.QComboBox()
        self.play_speed.addItems(["0.1×", "0.25×", "0.5×", "1×", "2×", "4×", "8×", "10×"])
        self.play_speed.setCurrentText("1×")
        play_layout.addWidget(self.play_speed)
        play_layout.addWidget(QtWidgets.QLabel("时间窗"))
        self.play_window = QtWidgets.QComboBox()
        self.play_window.addItems(["10 ms", "20 ms", "50 ms", "100 ms", "200 ms", "500 ms",
                                   "1 s", "2 s", "5 s", "10 s", "20 s"])
        self.play_window.setCurrentText("200 ms")
        self.play_window.currentIndexChanged.connect(self._on_playback_window)
        play_layout.addWidget(self.play_window)
        self.play_progress = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.play_progress.setRange(0, 1000)
        self.play_progress.sliderReleased.connect(self._seek_playback)
        play_layout.addWidget(self.play_progress, 1)
        layout.addWidget(self.play_bar)
        grid = QtWidgets.QGridLayout()
        self.wave = pg.PlotWidget(title="I / Q 波形（预览）")
        self.wave.setLabel("bottom", "时间", units="s")
        self.wave.setLabel("left", "幅度（任意单位）")
        self.wave.addLegend()
        self.spectrum = pg.PlotWidget(title="平均功率谱密度")
        self.spectrum.setLabel("bottom", "基带频率偏移", units="Hz")
        self.spectrum.setLabel("left", "PSD（dB，参考 1 任意单位²/Hz）")
        self.time_frequency = pg.PlotWidget(title="时频图")
        self.time_frequency.setLabel("bottom", "时间", units="s")
        self.time_frequency.setLabel("left", "基带频率偏移", units="Hz")
        self.constellation = pg.PlotWidget(title="星座图（数字信号判定）")
        self.constellation.setLabel("bottom", "同相分量 I")
        self.constellation.setLabel("left", "正交分量 Q")
        self.const_scatter = pg.ScatterPlotItem(size=2, pen=None,
                                                brush=pg.mkBrush(35, 101, 179, 120))
        self.constellation.addItem(self.const_scatter)
        self.constellation.hide()
        self.tf_stack = QtWidgets.QStackedWidget()
        self.tf_stack.addWidget(self.time_frequency)
        self.tf_stack.addWidget(self.constellation)
        self.waterfall = pg.PlotWidget(title="瀑布图（离线历史）")
        self.waterfall.setLabel("bottom", "基带频率偏移", units="Hz")
        self.waterfall.setLabel("left", "时间", units="s")
        self.tf_image = pg.ImageItem(axisOrder="row-major")
        self.waterfall_image = pg.ImageItem(axisOrder="row-major")
        self.time_frequency.addItem(self.tf_image)
        self.waterfall.addItem(self.waterfall_image)
        color_map = pg.colormap.get("viridis")
        for item in (self.tf_image, self.waterfall_image):
            item.setLookupTable(color_map.getLookupTable())
        for index, plot in enumerate((self.wave, self.spectrum, self.tf_stack, self.waterfall)):
            grid.addWidget(plot, index // 2, index % 2)
        layout.addLayout(grid, 1)
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMaximumHeight(110)
        self.summary.setPlaceholderText("分析后显示采样数、均值、RMS、峰值及显示范围。")
        layout.addWidget(self.summary)
        actions = QtWidgets.QHBoxLayout()
        self.native_button = QtWidgets.QPushButton("加载原生复制插件清单…")
        self.native_button.clicked.connect(self.native_selected)
        actions.addWidget(self.native_button)
        export = QtWidgets.QPushButton("导出当前报告…")
        export.clicked.connect(self.export_current)
        actions.addWidget(export)
        actions.addStretch()
        layout.addLayout(actions)
        return box


    def analyze_selected(self):
        if self.analyze_mode.currentText() == "实时播放":
            self._start_playback()
            return
        asset = self.selected_asset()
        if asset:
            self.start_job("analyze", owner="态势显示", label=f"态势显示 · {asset['name']}",
                           cancel_text="取消本次分析",
                           asset_id=asset["id"], nfft=int(self.nfft.currentText()))
        else:
            self.status.setText("请先导入并选择数据")


    def native_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先选择数据")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择并运行原生复制插件", "", "插件清单 (*.json)")
        if path:
            self.start_job("native", owner="态势显示",
                           label=f"原生插件复制 · {asset['name']}",
                           cancel_text="取消本次复制",
                           asset_id=asset["id"], manifest=path)


    def _effective_classification(self, summary):
        choice = self.class_combo.currentText()
        if choice == "数字":
            return "digital", True
        if choice == "模拟":
            return "analog", True
        return summary.get("classification", "analog"), False

    def _positive_half_mask(self, frequencies, real_valued):
        choice = self.freq_view.currentText()
        if choice == "双边":
            return slice(None)
        if choice == "仅正频率" or (choice == "自动" and real_valued):
            return frequencies >= 0
        return slice(None)

    def _on_range_follow(self):
        if not self._range_syncing:
            self._apply_display_mode()

    def _on_range_edited(self):
        """手动改动任一范围数值即切换为固定范围，避免下次分析被自动定标覆盖。"""
        if self._range_syncing:
            return
        self._range_syncing = True
        self.range_follow.setChecked(False)
        self._range_syncing = False
        self._apply_display_mode()

    def _wave_span_seconds(self):
        text = self.wave_span.currentText()
        if text == "整个记录":
            return None
        value, _, unit = text.partition(" ")
        seconds = float(value)
        return seconds / 1000.0 if unit.strip() == "ms" else seconds

    def _spec_half_span(self, rate):
        span = float(self.spec_span.value())
        return min(span, rate / 2.0) if span > 0 else rate / 2.0

    def _set_range_fields(self, amplitude, half_span):
        """自动定标时把推导值写回控件，使界面显示的就是实际使用的范围。"""
        self._range_syncing = True
        self.amp_max.setValue(amplitude)
        self.spec_span.setValue(half_span)
        self._range_syncing = False

    def _apply_wave_range(self, t_end, span):
        amplitude = float(self.amp_max.value())
        self.wave.setXRange(t_end - span, t_end, padding=0.0)
        self.wave.setYRange(-amplitude, amplitude, padding=0.0)
        self.wave.setTitle(f"I / Q 波形 · ±{amplitude:g}（任意单位）· 时窗 {_fmt_span(span)}")

    def _apply_spectrum_range(self, rate, positive_only, hi_db):
        half = self._spec_half_span(rate)
        low = 0.0 if positive_only else -half
        span_db = float(self.spec_db_span.value())
        self.spectrum.setXRange(low, half, padding=0.0)
        self.spectrum.setYRange(hi_db - span_db, hi_db, padding=0.0)
        side = "仅正频率" if positive_only else "双边"
        self.spectrum.setTitle(f"平均功率谱密度 · {side} {_fmt_hz(low)}～{_fmt_hz(half)}"
                               f" · 动态范围 {span_db:g} dB")
        return low, half, span_db

    @staticmethod
    def _nice_ceiling(value):
        """1 / 2 / 5 × 10ⁿ 中不小于 value 的最小值，用作幅度定标上限。"""
        if not value > 0:
            return 1.0
        magnitude = 10.0 ** np.floor(np.log10(value))
        for step in (1.0, 2.0, 5.0, 10.0):
            candidate = step * magnitude
            if candidate >= value:
                return float(candidate)
        return float(10.0 * magnitude)

    def _render_analysis(self, result, arrays):
        summary = result["summary"]
        f = arrays["frequency"]
        t = arrays["frame_time"]
        rate = float(summary["sample_rate_hz"])
        mask = self._positive_half_mask(f, bool(summary.get("real_valued", False)))
        fv = f[mask]
        positive_only = isinstance(mask, np.ndarray)
        spectrum_db = arrays["spectrum_db"][mask]
        matrix = arrays["spectrogram_db"][:, mask]
        high = float(matrix.max()) if matrix.size else -120.0
        if spectrum_db.size:
            high = max(high, float(spectrum_db.max()))
        if self.range_follow.isChecked():
            peak = max(float(np.abs(arrays["wave_i"]).max()) if arrays["wave_i"].size else 0.0,
                       float(np.abs(arrays["wave_q"]).max()) if arrays["wave_q"].size else 0.0)
            self._set_range_fields(self._nice_ceiling(peak), 0.0)
        levels = [high - float(self.spec_db_span.value()), high]
        classification, manual = self._effective_classification(summary)
        self.wave.clear()
        self.wave.plot(arrays["wave_time"], arrays["wave_i"], pen="#2365b3", name="I")
        self.wave.plot(arrays["wave_time"], arrays["wave_q"], pen="#e39b35", name="Q")
        duration = float(summary["duration_s"])
        wave_span = self._wave_span_seconds()
        wave_span = duration if wave_span is None else min(wave_span, duration)
        self._apply_wave_range(wave_span, wave_span)
        self.spectrum.clear()
        self.spectrum.plot(fv, spectrum_db, pen="#2365b3")
        low_f, high_f, span_db = self._apply_spectrum_range(rate, positive_only, high)
        df = f[1] - f[0]
        dt = summary["hop_samples"] / summary["sample_rate_hz"]
        if classification == "digital":
            self.const_scatter.setData(x=arrays["const_i"], y=arrays["const_q"])
            self.constellation.setTitle(
                f"星座图（数字信号判定 · 簇数估计 {summary.get('cluster_estimate', 0)}"
                f"{' · 手动判定' if manual else ''}）")
            self.tf_stack.setCurrentWidget(self.constellation)
        else:
            self.tf_image.setImage(matrix.T, levels=levels, autoLevels=False)
            self.tf_image.setRect(QtCore.QRectF(t[0] - dt / 2, fv[0] - df / 2,
                                                dt * len(t), df * len(fv)))
            self.tf_stack.setCurrentWidget(self.time_frequency)
            self.time_frequency.autoRange()
        self.waterfall_image.setImage(matrix, levels=levels, autoLevels=False)
        self.waterfall_image.setRect(QtCore.QRectF(fv[0] - df / 2, t[0] - dt / 2,
                                                   df * len(fv), dt * len(t)))
        # 瀑布图与平均功率谱密度共用同一频率范围（含「仅正频率」选择），
        # 纵轴固定为整段记录时长，不随刷新变动。
        self.waterfall.setXRange(low_f, high_f, padding=0.0)
        self.waterfall.setYRange(float(t[0] - dt / 2), float(t[-1] + dt / 2), padding=0.0)
        s = summary
        label = "数字" if classification == "digital" else "模拟"
        data_kind = "实数（默认仅显示正频率）" if s.get("real_valued") else "复数 IQ（默认双边频率）"
        self.summary.setPlainText(
            f"数据：{result.get('asset_name', result['asset_id'])}  |  采样数 {s['sample_count']:,}  |  时长 {s['duration_s']:.6f} s\n"
            f"均值 I={s['mean_i']:.6g}, Q={s['mean_q']:.6g}  |  RMS={s['rms']:.6g}  |  峰值={s['peak']:.6g}\n"
            f"信号判定：{label}（{'手动选择' if manual else '自动启发式'}，簇数估计 {s.get('cluster_estimate', 0)}）"
            f"  |  数据：{data_kind}\n"
            f"显示范围（固定，可在工具栏调整）：波形 ±{self.amp_max.value():g}（时窗 {_fmt_span(wave_span)}）；"
            f"频谱 {_fmt_hz(low_f)}～{_fmt_hz(high_f)}，动态范围 {span_db:g} dB"
            f"（{levels[0]:.1f}～{levels[1]:.1f} dB，含瀑布色标）；"
            f"波形抽点预览：{'是' if s['preview_decimated'] else '否'}；"
            f"瀑布时间范围 = 记录时长（与 FFT 点数无关）")

    def _apply_display_mode(self):
        if self._play_data is not None:
            self._play_render()
            return
        result = self.last_result
        if result is None or result.get("kind") != "analysis":
            return
        with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
            self._render_analysis(result, arrays)

    def _rerender_tab(self, title, render):
        """按最近一次结果重画某个标签页（切换频率显示等显示选项时用）。"""
        result = self.tab_results.get(title)
        if not result:
            return
        with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
            render(result, arrays)

    def _apply_detect_display_mode(self):
        self._rerender_tab("信号检测", self._render_detect)

    def _apply_hops_display_mode(self):
        self._rerender_tab("跳频参数", self._render_hops)

    def _on_analyze_mode(self):
        playing_mode = self.analyze_mode.currentText() == "实时播放"
        self.play_bar.setVisible(playing_mode)
        self.analyze_button.setText("播放所选数据" if playing_mode else "分析所选数据")
        if not playing_mode and self._play_data is not None:
            self._stop_playback()

    def _nfft_value(self):
        return int(self.nfft.currentText())

    def _on_playback_nfft(self):
        if self._play_data is not None:
            self._rebuild_playback_window()

    def _start_playback(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        if self._play_data is not None:
            self._stop_playback()
        try:
            meta, data = self.workspace.load_samples(asset["id"])
        except Exception as exc:
            self.status.setText(f"读取数据失败：{exc}")
            return
        self._play_data = data
        self._play_rate = float(meta["sample_rate"])
        self._play_real = bool(np.all(data.imag == 0))
        if self.range_follow.isChecked():
            probe = np.asarray(data[:min(data.size, 200_000)])
            peak = float(max(np.abs(probe.real).max(), np.abs(probe.imag).max())) if probe.size else 1.0
            self._set_range_fields(self._nice_ceiling(peak), 0.0)
        self._play_pos = 0
        self._play_paused = False
        self._play_buffer = None
        self._play_times = []
        self._play_level_hi = None
        self._play_last = time.monotonic()
        self.play_timer.start()
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.waterfall.setTitle("瀑布图（滚动时间窗）")
        self.status.setText(f"实时播放中：{asset['name']}（速度 {self.play_speed.currentText()}，"
                            f"时间窗 {self.play_window.currentText()}）")

    def _on_playback_tick(self):
        if self._play_data is None:
            return
        if self._play_paused:
            self._play_last = time.monotonic()
            return
        now = time.monotonic()
        elapsed = now - self._play_last
        self._play_last = now
        advance = int(self._play_rate * self._play_speed() * elapsed)
        if advance < 1:
            return
        size = self._play_data.size
        target = min(self._play_pos + advance, size)
        nfft = self._nfft_value()
        hop = self._play_hop(advance)
        # 时间窗可能短于一次刷新的间隔，此时需要在一次刷新内补齐多帧，
        # 否则窗口里只剩一两行、远不足以表现滚动。
        if self._play_buffer is None:
            self._play_buffer = []
            self._play_times = []
        positions = []
        pos = self._play_pos + hop
        while pos <= target:
            positions.append(pos)
            pos += hop
        if not positions:
            positions.append(target)
        for pos in positions:
            _, row = spectrum_row(self._play_data, pos, nfft, self._play_rate)
            self._play_buffer.append(row)
            self._play_times.append(pos / self._play_rate)
        self._play_pos = target
        keep = self._play_window_rows()
        if len(self._play_buffer) > keep:
            self._play_buffer = self._play_buffer[-keep:]
            self._play_times = self._play_times[-keep:]
        self._play_render()
        self.play_progress.setValue(int(1000 * self._play_pos / size))
        if self._play_pos >= size:
            self._stop_playback(final=True)

    def _window_seconds(self):
        value, _, unit = self.play_window.currentText().partition(" ")
        seconds = float(value)
        return seconds / 1000.0 if unit.strip() == "ms" else seconds

    def _play_speed(self):
        return float(self.play_speed.currentText().rstrip("×"))

    def _play_hop(self, advance):
        """相邻两帧相隔的信号样点数。

        基准为 FFT 窗长的一半（标准 STFT 交叠），再受两个上限约束：
        窗口内保留的行数不超过 `PLAY_MAX_ROWS`，单次刷新计算的行数不超过
        `PLAY_MAX_ROWS_PER_TICK`。窗口很短时靠前一约束保证行数够用，
        快放或长窗口时靠后一约束限制每次刷新的计算量。
        """
        hop = max(1, self._nfft_value() // 2)
        window_samples = self._window_seconds() * self._play_rate
        if window_samples > 0:
            hop = max(hop, int(np.ceil(window_samples / PLAY_MAX_ROWS)))
        if advance > 0:
            hop = max(hop, int(np.ceil(advance / PLAY_MAX_ROWS_PER_TICK)))
        return max(1, hop)

    def _play_window_rows(self):
        if len(self._play_times) > 1:
            dt = float(np.median(np.diff(self._play_times)))
        else:
            dt = self._play_hop(0) / self._play_rate
        if not dt > 0:
            dt = max(1, self._nfft_value() // 2) / self._play_rate
        return max(2, int(np.ceil(self._window_seconds() / dt)) + 1)

    def _on_playback_window(self):
        if self._play_data is not None:
            self._rebuild_playback_window()

    def _rebuild_playback_window(self):
        """按当前时间窗与 FFT 点数重铺最近一窗数据的帧。

        时间窗或 FFT 点数变化后，帧间距随之改变，旧缓冲行的疏密不再匹配，
        因此丢弃旧缓冲并从当前播放位置向前回溯一窗重建，使瀑布图立刻以新
        分辨率显示最近一段历史，而不是留一格空白或沿用旧疏密。
        """
        hop = self._play_hop(0)
        span = int(self._window_seconds() * self._play_rate)
        start = max(0, self._play_pos - span)
        positions = list(range(start + hop, self._play_pos + 1, hop))
        if not positions or positions[-1] != self._play_pos:
            positions.append(self._play_pos)
        nfft = self._nfft_value()
        self._play_buffer = []
        self._play_times = []
        for pos in positions[-(PLAY_MAX_ROWS + 1):]:
            _, row = spectrum_row(self._play_data, pos, nfft, self._play_rate)
            self._play_buffer.append(row)
            self._play_times.append(pos / self._play_rate)
        self._play_render()

    def _play_render(self):
        if not self._play_buffer:
            return
        matrix = np.vstack(self._play_buffer)
        f_full, _ = spectrum_row(self._play_data, self._play_pos, self._nfft_value(), self._play_rate)
        mask = self._positive_half_mask(f_full, self._play_real)
        positive_only = isinstance(mask, np.ndarray)
        fv = f_full[mask]
        # 瀑布图与频谱共用同一频率子集，否则「仅正频率」时瀑布图仍按双边铺满，
        # 与上方曲线错位（横轴取正半轴、图像却仍是整段双边数据）。
        matrix = matrix[:, mask]
        df = fv[1] - fv[0] if fv.size > 1 else self._play_rate / self._nfft_value()
        # 色标与频谱纵轴在整个播放过程中保持不变：仅首次刷新按数据定标，
        # 否则每帧重算会让颜色与曲线随数据跳动。
        if self._play_level_hi is None:
            self._play_level_hi = float(matrix.max()) if matrix.size else -120.0
        high = self._play_level_hi
        span_db = float(self.spec_db_span.value())
        levels = [high - span_db, high]
        self.spectrum.clear()
        self.spectrum.plot(fv, matrix[-1], pen="#2365b3")
        low_f, high_f, _ = self._apply_spectrum_range(self._play_rate, positive_only, high)
        window_s = self._window_seconds()
        wave_span = self._wave_span_seconds()
        wave_span = window_s if wave_span is None else wave_span
        count = int(wave_span * self._play_rate)
        start = max(0, self._play_pos - count)
        step = max(1, int(np.ceil((self._play_pos - start) / PLAY_WAVE_POINTS)))
        chunk = self._play_data[start:self._play_pos:step]
        tt = (start + np.arange(chunk.size) * step) / self._play_rate
        self.wave.clear()
        self.wave.plot(tt, chunk.real, pen="#2365b3", name="I")
        self.wave.plot(tt, chunk.imag, pen="#e39b35", name="Q")
        self._apply_wave_range(self._play_times[-1], wave_span)
        # 瀑布图为固定时长的滚动窗口：纵轴始终锁定最近「时间窗」秒，
        # 新数据自顶端进入并随时间向上流动；窗口时长可调，行高随之重新标定。
        rows_needed = self._play_window_rows()
        view = matrix[-rows_needed:]
        t_top = self._play_times[-1] if self._play_times else window_s
        height = view.shape[0] * (window_s / rows_needed)
        self.waterfall_image.setImage(view, levels=levels, autoLevels=False)
        self.waterfall_image.setRect(QtCore.QRectF(fv[0] - df / 2, t_top - height,
                                                   df * len(fv), height))
        # 横轴与上方平均功率谱密度完全一致，便于两图按频率对齐比较。
        self.waterfall.setXRange(low_f, high_f, padding=0.0)
        self.waterfall.setYRange(t_top - window_s, t_top, padding=0.0)

    def _toggle_pause(self):
        if self._play_data is None:
            return
        self._play_paused = not self._play_paused
        self.pause_button.setText("继续" if self._play_paused else "暂停")
        self._play_last = time.monotonic()
        self.status.setText("播放已暂停" if self._play_paused else "继续播放")

    def _stop_playback(self, final=False):
        self.play_timer.stop()
        self._play_data = None
        self._play_buffer = None
        self._play_times = []
        self._play_pos = 0
        self._play_paused = False
        self._play_level_hi = None
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("暂停")
        self.waterfall.setTitle("瀑布图（离线历史）")
        self.status.setText("播放完成" if final else "播放已停止")

    def _seek_playback(self):
        if self._play_data is None:
            return
        frac = self.play_progress.value() / 1000.0
        self._play_pos = int(frac * self._play_data.size)
        self._play_buffer = None
        self._play_times = []
        self._play_last = time.monotonic()
        self._rebuild_playback_window()
