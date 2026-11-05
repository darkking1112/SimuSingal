"""Manual signal-analysis drill page.

The page is a self-contained teaching view: three primary plots on the left,
a grouped type picker and an auxiliary evidence card on the right, and the three
actions required by the training flow.  It deliberately does not read or write
the workspace database.
"""

import html
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...drill import (AUX_INDEX, CLASS_BY_KEY, DrillSession, grouped_classes)
from ...drill.features import hop_track_khz


PLOT_BG = "#061826"
TRACE = "#39ff88"
AVERAGE = "#4dd0e1"
AMBER = "#ffd54f"
AXIS_PEN = "#4b6b7a"
AXIS_TEXT = "#cfe3f5"


def _style_plot(plot, bottom_label, left_label):
    plot.setBackground(PLOT_BG)
    item = plot.getPlotItem()
    item.showGrid(x=True, y=True, alpha=0.18)
    item.setLabel("bottom", bottom_label, color="#9fb3c8")
    item.setLabel("left", left_label, color="#9fb3c8")
    for name in ("left", "bottom"):
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(AXIS_PEN))
        axis.setTextPen(pg.mkPen(AXIS_TEXT))
    item.getViewBox().setMouseEnabled(x=True, y=False)
    plot.setMenuEnabled(False)


class SignalDrillPage(QtWidgets.QWidget):
    """Single-level manual signal drill page."""

    OWNER = "信号判读训练"

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.session = DrillSession()
        self.round = None
        self._busy = False
        self._last_tick = time.monotonic()
        self._build_ui()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._on_tick)
        self._render_placeholders()
        self._update_question_label()
        # Generate lazily when the user actually opens this tab; constructing
        # the other pages/tests must not pay for a 5 s waveform + STFT.
        self.window.tabs.currentChanged.connect(self._on_tab_changed)

    # ------------------------------------------------------------- UI building
    def _build_ui(self):
        self.setStyleSheet("""
            QFrame#drillCard {
                background:#ffffff; border:1px solid #d8e1ec; border-radius:8px;
            }
            QLabel#drillTitle {font-size:20px; font-weight:600; color:#183c65;}
            QLabel#drillSub {color:#5b7387;}
            QLabel#drillPlotTitle {color:#183c65; font-weight:600;}
            QLabel#drillPlotSub {color:#6b7f92; font-size:11px;}
            QFrame#drillAnswer {
                background:#e8f1fb; border:1px solid #cfe0f3; border-radius:8px;
            }
            QFrame#drillPicker {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 #0d47a1, stop:1 #061f4a);
                border-radius:10px;
            }
            QFrame#drillPicker QLabel, QFrame#drillPicker QRadioButton,
            QFrame#drillPicker QScrollArea, QFrame#drillPicker QToolButton {
                background:transparent;
            }
            QLabel#pickerTitle {color:#ffd54f; font-size:20px; font-weight:700;}
            QLabel#pickerGroup {color:#90caf9; font-weight:700; padding-top:5px;}
            QLabel#pickerSelection {color:#ffd54f;}
            QLabel#pickerHint {color:#cfe3f5; font-size:11px;}
            QRadioButton#pickerItem {
                color:#ffffff; spacing:6px; padding:0px 2px; font-size:11px;
            }
            QRadioButton#pickerItem:checked {color:#ffd54f; font-weight:700;}
            QRadioButton#pickerItem:disabled {color:#90a4ae;}
            QRadioButton#pickerItem::indicator {
                width:13px; height:13px; border:1px solid #cfe3f5;
                border-radius:7px; background:transparent;
            }
            QRadioButton#pickerItem::indicator:checked {
                background:#ffd54f; border:2px solid #ffd54f;
            }
            QRadioButton#pickerItem::indicator:disabled {border-color:#607d8b;}
            QFrame#drillAux {
                background:rgba(255,255,255,0.06);
                border:1px solid rgba(255,255,255,0.16);
                border-radius:8px;
            }
            QFrame#drillAux QLabel {color:#e3f2fd;}
            QTabWidget::pane {border:0; background:#061826;}
            QTabBar::tab {background:#12344a; color:#cfe3f5; padding:4px 7px;}
            QTabBar::tab:selected {background:#2365b3; color:white;}
            QPushButton#drillAnswer {background:#b86f00; color:white;}
            QLabel#drillStatus {color:#4a5b6e;}
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("信号判读训练")
        title.setObjectName("drillTitle")
        header.addWidget(title)
        subtitle = QtWidgets.QLabel(
            "根据时域波形、瞬时频谱、累积频谱判断信号类型")
        subtitle.setObjectName("drillSub")
        header.addWidget(subtitle)
        header.addStretch(1)
        self.question_label = QtWidgets.QLabel()
        self.question_label.setObjectName("drillSub")
        header.addWidget(self.question_label)
        layout.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self._build_left(left_layout)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._build_right(right_layout)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([920, 400])
        layout.addWidget(splitter, 1)

        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("drillStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def _build_left(self, layout):
        self.time_window = QtWidgets.QComboBox()
        for label, seconds in (
            ("1 ms", 0.001), ("5 ms", 0.005), ("10 ms", 0.010),
            ("20 ms", 0.020), ("50 ms", 0.050), ("100 ms", 0.100),
        ):
            self.time_window.addItem(label, seconds)
        self.time_window.setCurrentIndex(1)  # 5 ms: readable IF carrier detail
        self.time_window.currentIndexChanged.connect(self._refresh_time_plot)
        time_card, self.time_plot = self._make_card(
            "时域波形", "播放位置附近原始采样 · f_IF=45.0 kHz",
            [self.time_window])
        self.time_seek = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.time_seek.setRange(0, 5000)
        self.time_seek.setToolTip("拖动定位 5 s 记录中的任意时刻")
        self.time_seek.valueChanged.connect(self._on_time_seek)
        time_card.layout().addWidget(self.time_seek)
        self.time_subtitle = time_card.findChild(
            QtWidgets.QLabel, "drillPlotSub")
        layout.addWidget(time_card, 1)

        instant_card, self.instant_plot = self._make_card(
            "瞬时频谱", "20 ms 窗 · t = 0.00 s")
        self.instant_subtitle = instant_card.findChild(
            QtWidgets.QLabel, "drillPlotSub")
        layout.addWidget(instant_card, 1)

        cumulative_card, self.cumulative_plot = self._make_card(
            "累积频谱", "0.00 / 5.00 s · 当前累积")
        self.cum_subtitle = cumulative_card.findChild(
            QtWidgets.QLabel, "drillPlotSub")
        layout.addWidget(cumulative_card, 1)

        self._init_time_plot()
        self._init_spectrum_plots()

        answer_frame = QtWidgets.QFrame()
        answer_frame.setObjectName("drillAnswer")
        answer_layout = QtWidgets.QVBoxLayout(answer_frame)
        answer_layout.setContentsMargins(10, 7, 10, 7)
        answer_layout.setSpacing(2)
        self.answer_browser = QtWidgets.QTextBrowser()
        self.answer_browser.setOpenExternalLinks(False)
        self.answer_browser.setMinimumHeight(100)
        self.answer_browser.setMaximumHeight(150)
        self.answer_browser.setStyleSheet(
            "QTextBrowser {background:transparent; border:0; color:#23374d;}")
        answer_layout.addWidget(self.answer_browser)
        layout.addWidget(answer_frame, 0)

    def _make_card(self, title, subtitle, header_widgets=()):
        card = QtWidgets.QFrame()
        card.setObjectName("drillCard")
        card.setMinimumHeight(140)
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(8, 6, 8, 6)
        card_layout.setSpacing(3)
        head = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("drillPlotTitle")
        head.addWidget(title_label)
        sub = QtWidgets.QLabel(subtitle)
        sub.setObjectName("drillPlotSub")
        head.addWidget(sub)
        head.addStretch(1)
        for widget in header_widgets:
            head.addWidget(widget)
        card_layout.addLayout(head)
        plot = pg.PlotWidget()
        plot.setMinimumHeight(110)
        card_layout.addWidget(plot, 1)
        return card, plot

    def _init_time_plot(self):
        _style_plot(self.time_plot, "时间 (ms)", "幅度")
        self.time_trace = pg.PlotDataItem(pen=pg.mkPen(TRACE, width=1.0))
        self.time_plot.addItem(self.time_trace)
        self.time_plot.setXRange(0.0, 5.0, padding=0)
        self.time_plot.setToolTip(
            "显示播放位置附近的原始中频采样，不做 min/max 包络压缩。"
            "拖动下方滑块可查看 5 s 记录中的任意位置。")

    def _init_spectrum_plots(self):
        _style_plot(self.instant_plot, "频率 (kHz)", "相对幅度 (dB)")
        self.instant_curve = pg.PlotDataItem(
            pen=pg.mkPen(TRACE, width=1.2), fillLevel=-80,
            brush=pg.mkBrush(57, 255, 136, 42))
        self.instant_peak = pg.ScatterPlotItem(
            size=9, brush=pg.mkBrush(AMBER), pen=pg.mkPen(AMBER))
        self.instant_fif = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#7fa6b5", width=1.0,
                         style=QtCore.Qt.PenStyle.DotLine))
        self.instant_plot.addItem(self.instant_curve)
        self.instant_plot.addItem(self.instant_peak)
        self.instant_plot.addItem(self.instant_fif)
        self.instant_plot.setXRange(0.0, 100.0, padding=0)
        self.instant_plot.setYRange(-80.0, 3.0, padding=0)

        _style_plot(self.cumulative_plot, "频率 (kHz)", "相对幅度 (dB)")
        self.cum_max = pg.PlotDataItem(
            pen=pg.mkPen(TRACE, width=1.2), fillLevel=-80,
            brush=pg.mkBrush(57, 255, 136, 40))
        self.cum_avg = pg.PlotDataItem(
            pen=pg.mkPen(AVERAGE, width=1.0,
                         style=QtCore.Qt.PenStyle.DashLine))
        self.cum_peaks = pg.ScatterPlotItem(
            size=9, symbol="t", brush=pg.mkBrush(AMBER),
            pen=pg.mkPen(AMBER))
        self.cum_fif = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen("#7fa6b5", width=1.0,
                         style=QtCore.Qt.PenStyle.DotLine))
        self.cumulative_plot.addItem(self.cum_max)
        self.cumulative_plot.addItem(self.cum_avg)
        self.cumulative_plot.addItem(self.cum_peaks)
        self.cumulative_plot.addItem(self.cum_fif)
        self.cumulative_plot.setXRange(0.0, 100.0, padding=0)
        self.cumulative_plot.setYRange(-80.0, 3.0, padding=0)

    def _build_right(self, layout):
        picker = QtWidgets.QFrame()
        picker.setObjectName("drillPicker")
        picker.setMinimumWidth(370)
        picker.setMaximumWidth(470)
        picker_layout = QtWidgets.QVBoxLayout(picker)
        picker_layout.setContentsMargins(14, 12, 14, 12)
        picker_layout.setSpacing(6)

        title = QtWidgets.QLabel("请 选 择")
        title.setObjectName("pickerTitle")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        picker_layout.addWidget(title)
        self.selection_label = QtWidgets.QLabel("未选择")
        self.selection_label.setObjectName("pickerSelection")
        self.selection_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        picker_layout.addWidget(self.selection_label)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea {background:transparent; border:0;}"
            "QScrollArea > QWidget > QWidget {background:transparent;}")
        content = QtWidgets.QWidget()
        content.setAutoFillBackground(False)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        self.type_group = QtWidgets.QButtonGroup(self)
        self.type_group.setExclusive(True)
        self.type_buttons = {}
        for group, items in grouped_classes():
            group_label = QtWidgets.QLabel(group)
            group_label.setObjectName("pickerGroup")
            content_layout.addWidget(group_label)
            grid = QtWidgets.QGridLayout()
            grid.setContentsMargins(2, 0, 2, 0)
            grid.setHorizontalSpacing(5)
            grid.setVerticalSpacing(0)
            for index, item in enumerate(items):
                button = QtWidgets.QRadioButton(item.short)
                button.setObjectName("pickerItem")
                button.setToolTip(f"{item.label}：{'；'.join(item.cues)}")
                button.setProperty("drill_key", item.key)
                self.type_group.addButton(button)
                self.type_buttons[item.key] = button
                button.toggled.connect(
                    lambda checked, key=item.key:
                    self._on_type_toggled(key, checked))
                grid.addWidget(button, index // 3, index % 3)
            content_layout.addLayout(grid)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        picker_layout.addWidget(scroll, 1)

        self.aux_frame = QtWidgets.QFrame()
        self.aux_frame.setObjectName("drillAux")
        self.aux_frame.setFixedHeight(220)
        self.aux_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed)
        aux_layout = QtWidgets.QVBoxLayout(self.aux_frame)
        aux_layout.setContentsMargins(7, 5, 7, 5)
        aux_layout.setSpacing(2)
        aux_head = QtWidgets.QHBoxLayout()
        aux_title = QtWidgets.QLabel("辅助视图")
        aux_title.setObjectName("drillPlotTitle")
        aux_head.addWidget(aux_title)
        aux_head.addStretch(1)
        self.aux_toggle = QtWidgets.QToolButton()
        self.aux_toggle.setText("收起")
        self.aux_toggle.setCheckable(True)
        self.aux_toggle.setStyleSheet("QToolButton {color:#cfe3f5;border:0;}")
        self.aux_toggle.toggled.connect(self._toggle_aux)
        aux_head.addWidget(self.aux_toggle)
        aux_layout.addLayout(aux_head)

        self.aux_tabs = QtWidgets.QTabWidget()
        self.aux_tabs.setMinimumHeight(178)
        self.aux_tabs.setMaximumHeight(185)
        self.const_plot = pg.PlotWidget()
        _style_plot(self.const_plot, "I", "Q")
        self.const_plot.setAspectLocked(True)
        self.const_plot.setXRange(-2.2, 2.2, padding=0)
        self.const_plot.setYRange(-2.2, 2.2, padding=0)
        self.aux_tabs.addTab(self.const_plot, "星座图")

        self.iq_plot = pg.PlotWidget()
        _style_plot(self.iq_plot, "时间 (ms)", "幅度")
        self.iq_i = pg.PlotDataItem(pen=pg.mkPen(TRACE, width=1.0))
        self.iq_q = pg.PlotDataItem(pen=pg.mkPen(AMBER, width=1.0))
        self.iq_plot.addItem(self.iq_i)
        self.iq_plot.addItem(self.iq_q)
        self.aux_tabs.addTab(self.iq_plot, "IQ 相位")

        self.tf_plot = pg.PlotWidget()
        _style_plot(self.tf_plot, "时间 (s)", "频率 (kHz)")
        self.tf_image = pg.ImageItem(axisOrder="row-major")
        self.tf_image.setColorMap(pg.colormap.get("viridis"))
        self.tf_plot.addItem(self.tf_image)
        self.tf_track = pg.PlotDataItem(
            pen=pg.mkPen(AVERAGE, width=1.2))
        self.tf_plot.addItem(self.tf_track)
        self.tf_plot.setXRange(0.0, 5.0, padding=0)
        self.tf_plot.setYRange(0.0, 100.0, padding=0)
        self.aux_tabs.addTab(self.tf_plot, "时频 / 跳频")
        self.aux_tabs.currentChanged.connect(self._on_aux_tab_changed)
        aux_layout.addWidget(self.aux_tabs)
        picker_layout.addWidget(self.aux_frame)

        hint = QtWidgets.QLabel(
            "右侧单选即选中；点击“显示答案”判卷。")
        hint.setObjectName("pickerHint")
        hint.setWordWrap(True)
        picker_layout.addWidget(hint)

        self.restart_button = QtWidgets.QPushButton("重新开始")
        self.restart_button.clicked.connect(self._on_restart)
        picker_layout.addWidget(self.restart_button)
        self.play_button = QtWidgets.QPushButton("▶ 播放")
        self.play_button.setObjectName("primary")
        self.play_button.clicked.connect(self._on_play)
        picker_layout.addWidget(self.play_button)
        self.answer_button = QtWidgets.QPushButton("显示答案")
        self.answer_button.setObjectName("drillAnswer")
        self.answer_button.clicked.connect(self._on_answer)
        picker_layout.addWidget(self.answer_button)
        layout.addWidget(picker, 1)

    # --------------------------------------------------------------- lifecycle
    def ensure_round(self):
        if self.round is None:
            self._generate_round()
        return self.round

    def _on_tab_changed(self, index):
        if self.window.tabs.widget(index) is self:
            self.ensure_round()

    def showEvent(self, event):
        super().showEvent(event)
        # The tab hook normally covers this; showEvent also handles a page
        # embedded in a plain window by tests or a future standalone shell.
        if self.isVisible() and self.window.tabs.currentWidget() is self:
            self.ensure_round()

    def hideEvent(self, event):
        self.timer.stop()
        if self.session.phase == "playing":
            self.session.pause()
            self._update_play_label()
        super().hideEvent(event)

    # ------------------------------------------------------------- generation
    def _generate_round(self):
        if self._busy:
            return
        self._busy = True
        self.timer.stop()
        QtWidgets.QApplication.setOverrideCursor(
            QtCore.Qt.CursorShape.WaitCursor)
        try:
            round_data = self.session.new_round()
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._set_status(f"生成信号失败：{exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._busy = False
        self.round = round_data
        self._render_round(round_data)
        self._set_status(
            f"新信号已生成（seed={round_data.spec.seed}）：点击播放观察，"
            f"或直接根据三幅图判读。")

    def _render_placeholders(self):
        empty = np.asarray([], dtype=np.float64)
        self.time_trace.setData(empty, empty)
        self.time_seek.setValue(0)
        self.instant_curve.setData(empty, empty)
        self.instant_peak.setData([], [])
        self.cum_max.setData(empty, empty)
        self.cum_avg.setData(empty, empty)
        self.cum_peaks.setData([], [])
        self.cum_subtitle.setText("0.00 / 5.00 s · 当前累积")
        self.answer_browser.setHtml(
            "<p><b>判读提示</b></p><p>打开本页后会自动生成一道 5 s 中频信号。"
            "先观察原始时域波形、瞬时谱峰和累积谱带，再在右侧选择信号类型。</p>")

    def _render_round(self, round_data):
        self.round = round_data
        self._update_question_label()
        self.selection_label.setText("未选择")
        self._reset_type_buttons()
        self.answer_button.setEnabled(True)
        self.play_button.setEnabled(True)
        self.cum_peaks.setData([], [])
        self.tf_track.setData([], [])
        self._show_constellation(show_ideal=False)
        self._render_tf()
        self._render_iq(0.0)
        self.aux_tabs.setCurrentIndex(AUX_INDEX[round_data.info.aux_tab])
        expanded = round_data.spec.key in {
            "bpsk", "qpsk", "psk8", "qam16", "qam64",
            "fsk4", "fsk8", "fsk2_burst", "fh_fsk2",
            "link11_single", "audio_multichannel",
        }
        self.aux_toggle.blockSignals(True)
        self.aux_toggle.setChecked(not expanded)
        self.aux_toggle.blockSignals(False)
        self._apply_aux_visibility(expanded)
        self._render_hint(round_data)
        peak = float(np.max(np.abs(round_data.waveform.samples)))
        self.time_plot.setYRange(-1.05 * peak, 1.05 * peak, padding=0)
        self.time_seek.blockSignals(True)
        self.time_seek.setRange(0, int(round(round_data.duration_s * 1000.0)))
        self.time_seek.setValue(0)
        self.time_seek.blockSignals(False)
        self._refresh_time_plot()
        self.instant_plot.setXRange(
            0.0, round_data.fs_hz / 2.0 / 1e3, padding=0)
        self.cumulative_plot.setXRange(
            0.0, round_data.fs_hz / 2.0 / 1e3, padding=0)
        self.instant_fif.setPos(round_data.spec.f_if_hz / 1e3)
        self.cum_fif.setPos(round_data.spec.f_if_hz / 1e3)
        self._refresh_position()
        self._update_play_label()

    def _reset_type_buttons(self):
        self.type_group.setExclusive(False)
        for key, button in self.type_buttons.items():
            item = CLASS_BY_KEY[key]
            button.setEnabled(True)
            button.setChecked(False)
            button.setText(item.short)
            button.setStyleSheet("")
        self.type_group.setExclusive(True)

    def _render_hint(self, round_data):
        if round_data.info.aux_tab == "constellation":
            extra = "本题已自动打开星座图；可切换 IQ 相位观察相位/幅度跳变。"
        elif round_data.spec.key in ("fsk2_burst", "fh_fsk2", "link11_single",
                                     "audio_multichannel", "fsk4", "fsk8"):
            extra = "本题已自动打开时频/跳频图；注意门控、频率台阶或音频子带。"
        else:
            extra = "先看波形幅度是否恒定或突发，再看谱峰数量和对称性。"
        self.answer_browser.setHtml(
            "<p><b>判读提示</b></p>"
            "<p>时域看原始波形：幅度恒定、两电平、突发还是多电平；瞬时谱看峰数、"
            "对称性和当前跳频信道；累积谱看真实占用的频带、跳频带或音频子带。</p>"
            f"<p>{extra}</p>")

    # --------------------------------------------------------------- rendering
    def _refresh_time_plot(self, *_):
        if self.round is None:
            return
        round_data = self.round
        duration = round_data.duration_s
        window = min(float(self.time_window.currentData() or duration), duration)
        position = self.session.position_s
        if window >= duration - 1e-9:
            start = 0.0
            first = 0
            last = round_data.waveform.samples.size
        else:
            start = min(max(position - window / 2.0, 0.0),
                        max(0.0, duration - window))
            first = int(round(start * round_data.fs_hz))
            last = min(round_data.waveform.samples.size,
                       int(round((start + window) * round_data.fs_hz)))
        t_ms = (np.arange(first, last, dtype=np.float64)
                / round_data.fs_hz * 1000.0)
        self.time_trace.setData(
            t_ms, round_data.waveform.samples[first:last].astype(np.float64))
        self.time_plot.setXRange(start * 1000.0,
                                 (start + window) * 1000.0, padding=0)
        self.time_subtitle.setText(
            f"原始采样 · {window * 1000.0:.0f} ms 时窗 · "
            f"t = {position:.3f} s / {duration:.2f} s")

    def _refresh_position(self):
        if self.round is None:
            return
        position = self.session.position_s
        self.time_seek.blockSignals(True)
        self.time_seek.setValue(int(round(position * 1000.0)))
        self.time_seek.blockSignals(False)
        self._refresh_time_plot()
        index = self.round.frame_index(position)
        frame = self.round.spectrum.frame_db[index]
        x = self.round.freqs_khz
        self.instant_curve.setData(x, frame)
        peak_index = int(np.argmax(frame))
        self.instant_peak.setData([x[peak_index]], [frame[peak_index]])
        self.instant_subtitle.setText(f"20 ms 窗 · t = {position:.2f} s")
        self._refresh_cumulative(position)
        if self.aux_tabs.currentIndex() == AUX_INDEX["iq"]:
            self._render_iq(position)

    def _refresh_cumulative(self, position_s):
        if self.round is None:
            return
        spectrum = self.round.spectrum
        x = self.round.freqs_khz
        if self.session.revealed:
            max_values = spectrum.cumulative_max_db
            avg_values = spectrum.cumulative_avg_db
            self.cum_subtitle.setText("5.00 s 完整累积（答案）")
        else:
            index = spectrum.frame_index(position_s)
            max_values = spectrum.progressive_max_db[index]
            avg_values = spectrum.progressive_avg_db[index]
            self.cum_subtitle.setText(
                f"{position_s:.2f} / {self.round.duration_s:.2f} s · 当前累积")
        self.cum_max.setData(x, max_values)
        self.cum_avg.setData(x, avg_values)

    def _on_time_seek(self, value_ms):
        if self.round is None:
            return
        self.session.seek(value_ms / 1000.0)
        self._refresh_position()
        self._update_play_label()

    def _render_iq(self, position):
        if self.round is None:
            return
        t, i_values, q_values = self.session.iq_window(0.05)
        if t is None:
            return
        self.iq_i.setData(t * 1e3, i_values)
        self.iq_q.setData(t * 1e3, q_values)
        self.iq_plot.setXRange(0.0, max(1.0, float(t[-1] * 1e3)), padding=0)
        level = max(0.2, float(np.max(np.abs(i_values))) if i_values.size else 0.2,
                    float(np.max(np.abs(q_values))) if q_values.size else 0.2)
        self.iq_plot.setYRange(-1.1 * level, 1.1 * level, padding=0)

    def _show_constellation(self, show_ideal=False):
        self.const_plot.clear()
        if self.round is None:
            return
        points = self.round.waveform.meta.get("constellation_points")
        if points is None or self.round.info.constellation_size <= 0:
            text = pg.TextItem(
                "该题型无星座\n请切换到时频 / 跳频",
                color="#9fb3c8", anchor=(0.5, 0.5))
            self.const_plot.addItem(text)
            text.setPos(0.0, 0.0)
            self.const_plot.setXRange(-1.0, 1.0, padding=0)
            self.const_plot.setYRange(-1.0, 1.0, padding=0)
            return
        points = np.asarray(points, dtype=np.complex128)
        scale = float(np.sqrt(np.mean(np.abs(points) ** 2)))
        scale = scale if scale > 1e-12 else 1.0
        points = points / scale
        scatter = pg.ScatterPlotItem(
            size=4, pen=None, brush=pg.mkBrush(57, 255, 136, 150))
        scatter.setData(points.real, points.imag)
        self.const_plot.addItem(scatter)
        if self.round.info.constellation_size in (2, 4, 8):
            theta = np.linspace(0, 2 * np.pi, 256)
            circle = pg.PlotDataItem(
                np.cos(theta), np.sin(theta),
                pen=pg.mkPen("#4dd0e1", width=1.0,
                             style=QtCore.Qt.PenStyle.DotLine))
            self.const_plot.addItem(circle)
        if show_ideal:
            ideal = self.round.waveform.meta.get("constellation_ideal")
            if ideal is not None:
                ideal = np.asarray(ideal, dtype=np.complex128) / scale
                overlay = pg.ScatterPlotItem(
                    size=9, symbol="x",
                    pen=pg.mkPen(AMBER, width=1.6))
                overlay.setData(ideal.real, ideal.imag)
                self.const_plot.addItem(overlay)
        self.const_plot.setXRange(-2.2, 2.2, padding=0)
        self.const_plot.setYRange(-2.2, 2.2, padding=0)

    def _render_tf(self):
        if self.round is None:
            return
        image = self.round.spectrum.frame_db
        self.tf_image.setImage(image, autoLevels=False, levels=(-70.0, 0.0))
        self.tf_image.setRect(QtCore.QRectF(
            0.0, 0.0, self.round.duration_s, self.round.fs_hz / 2.0 / 1e3))
        self.tf_plot.setXRange(0.0, self.round.duration_s, padding=0)
        self.tf_plot.setYRange(0.0, self.round.fs_hz / 2.0 / 1e3, padding=0)
        track = hop_track_khz(self.round.waveform.meta,
                              self.round.spectrum.frame_times_s)
        if track is not None:
            self.tf_track.setData(self.round.spectrum.frame_times_s, track)
        else:
            self.tf_track.setData([], [])

    def _on_aux_tab_changed(self, index):
        if self.round is None:
            return
        if index == AUX_INDEX["constellation"]:
            self._show_constellation(show_ideal=self.session.revealed)
        elif index == AUX_INDEX["iq"]:
            self._render_iq(self.session.position_s)
        elif index == AUX_INDEX["timefreq"]:
            self._render_tf()

    def _toggle_aux(self, checked):
        self._apply_aux_visibility(not checked)

    def _apply_aux_visibility(self, expanded):
        self.aux_tabs.setVisible(bool(expanded))
        self.aux_frame.setFixedHeight(220 if expanded else 36)
        self.aux_toggle.setText("收起" if expanded else "展开")

    # --------------------------------------------------------------- actions
    def _on_type_toggled(self, key, checked):
        if not checked or self.round is None or self.session.revealed:
            return
        self.session.select(key)
        item = CLASS_BY_KEY[key]
        self.selection_label.setText(f"当前选择：{item.label}")
        self._set_status(f"已选中 {item.label}；可继续观察或点击“显示答案”。")

    def _on_answer(self):
        if self.round is None:
            self._generate_round()
        if self.round is None or self.session.revealed:
            return
        if self.session.selected_key is None:
            choice = QtWidgets.QMessageBox.question(
                self, "显示答案",
                "尚未选择信号类型，仍要显示答案吗？",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
            if choice != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        result = self.session.reveal()
        self.timer.stop()
        self._render_answer(result)
        self._update_play_label()
        self._set_status(
            "回答正确。" if result.correct else
            ("未作答，已显示标准答案。" if not result.answered
             else "回答错误，请对照判读依据。"))

    def _render_answer(self, result):
        self._update_question_label()
        color = "#2e7d32" if result.correct else (
            "#c62828" if result.answered else "#b26a00")
        title = "✓ 正确" if result.correct else (
            "✗ 错误" if result.answered else "未作答")
        body = html.escape(result.explanation).replace("\n", "<br>")
        self.answer_browser.setHtml(
            f"<h3 style='color:{color}; margin:0 0 5px 0'>{title}</h3>"
            f"<p style='margin:0'>{body}</p>")
        for key, button in self.type_buttons.items():
            button.setEnabled(False)
            if key == result.truth_key:
                button.setStyleSheet(
                    "color:#66bb6a; font-weight:700;")
                button.setText("✓ " + CLASS_BY_KEY[key].short)
            elif key == result.selected_key and not result.correct:
                button.setStyleSheet(
                    "color:#ef5350; font-weight:700;")
                button.setText("✗ " + CLASS_BY_KEY[key].short)
        self.answer_button.setEnabled(False)
        self._show_constellation(show_ideal=True)
        self._refresh_cumulative(self.session.position_s)
        self._highlight_cumulative_peaks(result)
        self._render_tf()

    def _highlight_cumulative_peaks(self, result):
        self.cum_peaks.setData([], [])
        if result.truth_key not in ("fsk2", "fsk4", "fsk8", "multitone"):
            return
        frequencies = np.asarray(
            self.round.features.get("peak_frequencies_hz", ()),
            dtype=np.float64) / 1e3
        if frequencies.size == 0:
            return
        axis = self.round.spectrum.frequencies_hz
        values = self.round.spectrum.cumulative_max_db
        levels = [float(values[int(np.argmin(np.abs(axis - freq * 1e3)))])
                  for freq in frequencies]
        self.cum_peaks.setData(frequencies, levels)

    def _on_restart(self):
        if self._busy:
            return
        self._generate_round()

    def _on_play(self):
        if self.round is None:
            self._generate_round()
        if self.round is None:
            return
        if self.session.phase == "playing":
            self.session.pause()
            self.timer.stop()
            self._set_status("已暂停。")
        else:
            self.session.play()
            self._last_tick = time.monotonic()
            self.timer.start()
            self._set_status(
                "播放中：观察瞬时频谱、累积频谱与播放位置；可随时暂停或显示答案。")
        self._update_play_label()
        self._refresh_position()

    def _on_tick(self):
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self._last_tick))
        self._last_tick = now
        self.session.advance(dt)
        if self.session.phase == "ended":
            self.timer.stop()
            self._update_play_label()
            self._set_status("播放结束；可点击“重播”再看一遍，或显示答案。")
        self._refresh_position()

    def _update_play_label(self):
        labels = {
            "ready": "▶ 播放",
            "playing": "⏸ 暂停",
            "paused": "▶ 继续",
            "ended": "↻ 重播",
        }
        self.play_button.setText(labels.get(self.session.phase, "▶ 播放"))

    def _update_question_label(self):
        self.question_label.setText(
            f"第 {self.session.question_number} 题   ·   "
            f"正确 {self.session.correct_count}/{self.session.answered_count}"
            f"   ·   连续 {self.session.streak}")

    def _set_status(self, text):
        self.status_label.setText(text)
        if hasattr(self.window, "status"):
            self.window.status.setText(text)
