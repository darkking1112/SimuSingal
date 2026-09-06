"""PySide6 workbench. Numerical/native operations run outside the GUI process."""

import json
from pathlib import Path
import threading

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .reports import export_report
from .storage import Workspace
from .tasks import run_job


class JobSignals(QtCore.QObject):
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class JobRunner(QtCore.QRunnable):
    def __init__(self, request):
        super().__init__()
        self.request = request
        self.signals = JobSignals()
        self.cancelled = threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.completed.emit(run_job(self.request, cancel=self.cancelled))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = Workspace(workspace)
        self.active_job = None
        self.last_result = None
        self.tab_results = {}
        self.events = []
        self.asset_limit = 100
        self.setWindowTitle("SimuSignal · 离线实验工作台")
        self.resize(1440, 950)
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        pg.setConfigOptions(background="#ffffff", foreground="#34465a", antialias=True)
        self.setStyleSheet("""
            QMainWindow,QWidget {background:#f3f6fa;color:#23374d;font-size:13px;}
            QLineEdit,QDoubleSpinBox,QSpinBox,QListWidget,QPlainTextEdit,QTableWidget {
                background:white;border:1px solid #d8e1ec;border-radius:5px;padding:5px;}
            QPushButton {background:#e4edf8;border:0;border-radius:5px;padding:9px 14px;}
            QPushButton:hover {background:#cddff3;} QPushButton:disabled {color:#97a6b7;}
            QPushButton#primary {background:#2365b3;color:white;}
            QTabBar::tab {padding:11px 24px;} QTabBar::tab:selected {background:white;color:#2365b3;}
            QLabel#title {font-size:25px;font-weight:600;color:#183c65;}
        """)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(24, 18, 24, 16)
        title = QtWidgets.QLabel("SimuSignal  /  离线实验工作台")
        title.setObjectName("title")
        layout.addWidget(title)
        layout.addWidget(QtWidgets.QLabel("数据分析 · 原生模块验证 · 通用事件仿真"))
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.build_sidebar())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.build_analysis(), "数据分析")
        self.tabs.addTab(self.build_simulation(), "事件仿真")
        self.tabs.addTab(self.build_history(), "运行记录")
        splitter.addWidget(self.tabs)
        splitter.setSizes([290, 1100])
        layout.addWidget(splitter, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("就绪 · 导入文件或生成演示数据")
        bottom.addWidget(self.status, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        self.cancel_button = QtWidgets.QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_job)
        bottom.addWidget(self.cancel_button)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.refresh_assets()
        self.refresh_history()

    def build_sidebar(self):
        box = QtWidgets.QWidget()
        box.setMinimumWidth(250)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("数据资产"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("按文件名查找")
        self.search.textChanged.connect(self.refresh_assets)
        layout.addWidget(self.search)
        self.assets = QtWidgets.QListWidget()
        self.assets.currentItemChanged.connect(self.asset_changed)
        layout.addWidget(self.assets, 1)
        more = QtWidgets.QPushButton("显示更多（最多 500 条）")
        more.clicked.connect(self.more_assets)
        layout.addWidget(more)
        self.asset_info = QtWidgets.QLabel("尚未选择数据")
        self.asset_info.setWordWrap(True)
        layout.addWidget(self.asset_info)
        self.label = QtWidgets.QLineEdit()
        self.label.setPlaceholderText("数据备注，最多 200 字")
        self.label.setMaxLength(200)
        layout.addWidget(self.label)
        save = QtWidgets.QPushButton("保存备注")
        save.clicked.connect(self.save_label)
        layout.addWidget(save)
        layout.addWidget(QtWidgets.QLabel("导入 / 演示采样率（Hz）"))
        self.sample_rate = QtWidgets.QDoubleSpinBox()
        self.sample_rate.setRange(1, 1e9)
        self.sample_rate.setValue(48000)
        layout.addWidget(self.sample_rate)
        self.import_button = QtWidgets.QPushButton("导入 NPY / CSV")
        self.import_button.clicked.connect(self.import_file)
        layout.addWidget(self.import_button)
        self.demo_button = QtWidgets.QPushButton("生成数学双音演示")
        self.demo_button.clicked.connect(lambda: self.start_job("demo", sample_rate=self.sample_rate.value()))
        layout.addWidget(self.demo_button)
        workspace_label = QtWidgets.QLabel(f"工作目录\n{self.workspace.root}")
        workspace_label.setWordWrap(True)
        layout.addWidget(workspace_label)
        return box

    def build_analysis(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("通用统计与时频展示 · 识别模型尚未配置"), 1)
        bar.addWidget(QtWidgets.QLabel("FFT 点数"))
        self.nfft = QtWidgets.QComboBox()
        self.nfft.addItems(["128", "256", "512", "1024", "2048"])
        self.nfft.setCurrentText("256")
        bar.addWidget(self.nfft)
        self.analyze_button = QtWidgets.QPushButton("分析所选数据")
        self.analyze_button.setObjectName("primary")
        self.analyze_button.clicked.connect(self.analyze_selected)
        bar.addWidget(self.analyze_button)
        layout.addLayout(bar)
        grid = QtWidgets.QGridLayout()
        self.wave = pg.PlotWidget(title="I / Q 波形（预览）")
        self.wave.setLabel("bottom", "时间", units="s")
        self.wave.setLabel("left", "幅度（任意单位）")
        self.wave.addLegend()
        self.spectrum = pg.PlotWidget(title="双边平均功率谱密度")
        self.spectrum.setLabel("bottom", "基带频率偏移", units="Hz")
        self.spectrum.setLabel("left", "PSD（dB，参考 1 任意单位²/Hz）")
        self.time_frequency = pg.PlotWidget(title="时频图")
        self.time_frequency.setLabel("bottom", "时间", units="s")
        self.time_frequency.setLabel("left", "基带频率偏移", units="Hz")
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
        for index, plot in enumerate((self.wave, self.spectrum, self.time_frequency, self.waterfall)):
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

    def build_simulation(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("终端 A → 中继队列 → 终端 B  /  通用事件演示，未接入星地协议模型"))
        form = QtWidgets.QHBoxLayout()
        self.message_count = QtWidgets.QSpinBox()
        self.message_count.setRange(1, 1000)
        self.message_count.setValue(12)
        self.sim_fields = {}
        form.addWidget(QtWidgets.QLabel("消息数"))
        form.addWidget(self.message_count)
        for key, label, value in (("interval_s", "生成间隔", .1), ("transit_s", "单段传递时间", .02),
                                  ("service_s", "处理时间", .15), ("duration_s", "仿真时长", 3.0)):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(.001, 3600)
            spin.setDecimals(3)
            spin.setValue(value)
            spin.setSuffix(" s")
            form.addWidget(QtWidgets.QLabel(label))
            form.addWidget(spin)
            self.sim_fields[key] = spin
        self.sim_button = QtWidgets.QPushButton("运行实验")
        self.sim_button.setObjectName("primary")
        self.sim_button.clicked.connect(self.run_simulation)
        form.addWidget(self.sim_button)
        layout.addLayout(form)
        self.timeline = pg.PlotWidget(title="消息事件时间轴")
        self.timeline.setLabel("bottom", "模型时间", units="s")
        self.timeline.getAxis("left").setTicks([[(0, "终端 A"), (1, "中继"), (2, "终端 B")]])
        self.timeline.setYRange(-.5, 2.5)
        layout.addWidget(self.timeline, 1)
        self.replay = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.replay.setRange(0, 0)
        self.replay.valueChanged.connect(self.show_events)
        layout.addWidget(self.replay)
        self.sim_summary = QtWidgets.QLabel("运行后可拖动时间轴回放；尚未完成的消息单独计数。")
        layout.addWidget(self.sim_summary)
        self.event_table = QtWidgets.QTableWidget(0, 4)
        self.event_table.setHorizontalHeaderLabels(["时间 / s", "消息 ID", "节点", "阶段"])
        self.event_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.event_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.event_table, 1)
        export = QtWidgets.QPushButton("导出当前实验报告…")
        export.clicked.connect(self.export_current)
        layout.addWidget(export)
        return box

    def build_history(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("最近 100 次成功运行 · 双击查看或回放"))
        self.history = QtWidgets.QListWidget()
        self.history.itemDoubleClicked.connect(self.open_history)
        layout.addWidget(self.history)
        return box

    def selected_asset(self):
        item = self.assets.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def refresh_assets(self, *_):
        selected = self.selected_asset()
        self.assets.clear()
        for asset in self.workspace.list_assets(self.search.text(), self.asset_limit):
            item = QtWidgets.QListWidgetItem(asset["name"])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, asset)
            self.assets.addItem(item)
            if selected and selected["id"] == asset["id"]:
                self.assets.setCurrentItem(item)
        if self.assets.currentItem() is None and self.assets.count():
            self.assets.setCurrentRow(0)

    def more_assets(self):
        self.asset_limit = min(500, self.asset_limit + 100)
        self.refresh_assets()

    def asset_changed(self, *_):
        asset = self.selected_asset()
        if asset:
            self.asset_info.setText(f"{asset['sample_count']:,} 个复采样\n{asset['sample_rate']:g} Hz")
            self.label.setText(asset["label"])
        else:
            self.asset_info.setText("尚未选择数据")
            self.label.clear()

    def save_label(self):
        asset = self.selected_asset()
        if asset:
            self.workspace.set_label(asset["id"], self.label.text())
            self.refresh_assets()
            self.status.setText("数据备注已保存")

    def import_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择离线数据", "", "数据 (*.npy *.csv)")
        if path:
            self.start_job("import", path=path, sample_rate=self.sample_rate.value())

    def analyze_selected(self):
        asset = self.selected_asset()
        if asset:
            self.start_job("analyze", asset_id=asset["id"], nfft=int(self.nfft.currentText()))
        else:
            self.status.setText("请先导入并选择数据")

    def native_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先选择数据")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择并运行原生复制插件", "", "插件清单 (*.json)")
        if path:
            self.start_job("native", asset_id=asset["id"], manifest=path)

    def run_simulation(self):
        scenario = {key: spin.value() for key, spin in self.sim_fields.items()}
        scenario["messages"] = self.message_count.value()
        self.start_job("simulate", scenario=scenario)

    def start_job(self, action, **kwargs):
        if self.active_job is not None:
            return
        job = JobRunner({"workspace": str(self.workspace.root), "action": action, **kwargs})
        self.active_job = job
        job.signals.completed.connect(self.job_completed)
        job.signals.failed.connect(self.job_failed)
        self.set_busy(True)
        self.status.setText("任务运行中，可取消…")
        self.pool.start(job)

    def set_busy(self, busy):
        for button in (self.demo_button, self.import_button, self.analyze_button, self.native_button, self.sim_button):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(1)

    @QtCore.Slot(object)
    def job_completed(self, result):
        self.active_job = None
        self.set_busy(False)
        self.refresh_assets()
        self.refresh_history()
        try:
            if "kind" in result:
                self.display_result(result)
            else:
                for row in range(self.assets.count()):
                    item = self.assets.item(row)
                    if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == result["id"]:
                        self.assets.setCurrentItem(item)
            self.status.setText("任务完成 · 结果已保存到工作目录")
        except Exception as exc:
            self.status.setText(f"结果已保存，但显示失败：{exc}")

    @QtCore.Slot(str)
    def job_failed(self, message):
        self.active_job = None
        self.set_busy(False)
        self.status.setText(f"任务未完成：{message}")

    def cancel_job(self):
        if self.active_job:
            self.active_job.cancelled.set()

    def display_result(self, result):
        self.last_result = result
        if result["kind"] == "analysis":
            self.tab_results[0] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self.wave.clear()
                self.wave.plot(arrays["wave_time"], arrays["wave_i"], pen="#2365b3", name="I")
                self.wave.plot(arrays["wave_time"], arrays["wave_q"], pen="#e39b35", name="Q")
                self.spectrum.clear()
                f, t = arrays["frequency"], arrays["frame_time"]
                self.spectrum.plot(f, arrays["spectrum_db"], pen="#2365b3")
                matrix = arrays["spectrogram_db"]
                high = float(matrix.max())
                levels = [high - 80, high]
                self.tf_image.setImage(matrix.T, levels=levels, autoLevels=False)
                self.waterfall_image.setImage(matrix, levels=levels, autoLevels=False)
                df = f[1] - f[0]
                dt = result["summary"]["hop_samples"] / result["summary"]["sample_rate_hz"]
                self.tf_image.setRect(QtCore.QRectF(t[0] - dt / 2, f[0] - df / 2, dt * len(t), df * len(f)))
                self.waterfall_image.setRect(QtCore.QRectF(f[0] - df / 2, t[0] - dt / 2, df * len(f), dt * len(t)))
                self.time_frequency.autoRange()
                self.waterfall.autoRange()
            s = result["summary"]
            self.summary.setPlainText(f"数据：{result.get('asset_name', result['asset_id'])}  |  采样数 {s['sample_count']:,}  |  时长 {s['duration_s']:.6f} s\n"
                                      f"均值 I={s['mean_i']:.6g}, Q={s['mean_q']:.6g}  |  RMS={s['rms']:.6g}  |  峰值={s['peak']:.6g}\n"
                                      f"幅度：任意单位；颜色：PSD，{levels[0]:.1f}～{levels[1]:.1f} dB；"
                                      f"波形抽点预览：{'是' if s['preview_decimated'] else '否'}")
            self.tabs.setCurrentIndex(0)
        elif result["kind"] == "simulation":
            self.tab_results[1] = result
            self.events = result["events"]
            self.replay.setRange(0, len(self.events))
            self.replay.setValue(len(self.events))
            self.show_events(len(self.events))
            s = result["summary"]
            self.sim_summary.setText(f"计划 {s['planned']}  |  已发送 {s['sent']}  |  已接收 {s['received']}  |  "
                                     f"在途/排队 {s['pending']}  |  未启动 {s['not_started']}  |  "
                                     f"模型时间 {s['simulated_duration_s']:g} s / 实际耗时 {s['wall_duration_s']:.4f} s")
            self.tabs.setCurrentIndex(1)
        elif result["kind"] == "native":
            self.tab_results[0] = result
            self.wave.clear()
            self.spectrum.clear()
            self.tf_image.clear()
            self.waterfall_image.clear()
            self.summary.setPlainText(f"原生复制完成：{result['plugin']['id']}\n"
                                      f"输出资产：{result['derived_asset_id']}\n请选择输出资产进行分析。")
            self.tabs.setCurrentIndex(0)

    def show_events(self, count):
        events = self.events[:count]
        self.timeline.clear()
        positions = {"A": 0, "relay": 1, "B": 2}
        for node, y in positions.items():
            times = [event["time_s"] for event in events if event["node"] == node]
            self.timeline.plot(times, [y] * len(times), pen=None, symbol="o", symbolSize=7,
                               symbolBrush={"A": "#2365b3", "relay": "#e39b35", "B": "#29947d"}[node])
        visible = events[-200:]
        self.event_table.setRowCount(len(visible))
        for row, event in enumerate(visible):
            for column, value in enumerate((f"{event['time_s']:.6f}", event["message_id"], event["node"], event["stage"])):
                self.event_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))

    def refresh_history(self):
        self.history.clear()
        for result in self.workspace.list_runs():
            item = QtWidgets.QListWidgetItem(f"{result['created_at'][:19]}  ·  {result['kind']}  ·  {result['id'][:8]}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result["id"])
            self.history.addItem(item)

    def open_history(self, item):
        try:
            self.display_result(self.workspace.get_run(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            self.status.setText("已读取历史结果")
        except (ValueError, OSError) as exc:
            self.status.setText(f"无法读取历史结果：{exc}")

    def export_current(self):
        result = self.tab_results.get(self.tabs.currentIndex(), self.last_result if self.tabs.currentIndex() == 2 else None)
        if result is None:
            self.status.setText("请先完成一次分析或仿真")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出报告", "report.html", "HTML (*.html);;JSON (*.json)")
        if path:
            try:
                export_report(result, path)
                self.status.setText(f"报告已导出：{path}")
            except (ValueError, OSError) as exc:
                self.status.setText(f"导出失败：{exc}")

    def closeEvent(self, event):
        self.cancel_job()
        self.pool.waitForDone(5000)
        if self.pool.activeThreadCount():
            self.status.setText("正在结束后台任务，请稍后关闭")
            event.ignore()
        else:
            event.accept()


def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("SimuSignal")
    window = MainWindow(workspace)
    window.show()
    return app.exec()
