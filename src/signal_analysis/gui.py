"""Independent signal analysis desktop."""
import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from common.gui import DesktopWindow
from .storage import Workspace
from .tasks import run_job

class MainWindow(DesktopWindow):
    page_title = "数据分析"
    run_task = staticmethod(run_job)

    def __init__(self, workspace):
        self.asset_limit = 100
        super().__init__(Workspace(workspace), "电磁信号分析 · SignalAnalysis", "离线数据 · 通用统计与时频展示 · 原生插件")
        self.refresh_assets()

    def build_page(self):
        return self.build_analysis()

    def job_buttons(self):
        return (self.demo_button, self.import_button, self.analyze_button, self.native_button)

    def result_ready(self, result):
        self.refresh_assets()
        if "kind" in result:
            self.display_result(result)
        else:
            for row in range(self.assets.count()):
                item = self.assets.item(row)
                if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == result["id"]:
                    self.assets.setCurrentItem(item)

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
        elif result["kind"] == "native":
            self.tab_results[0] = result
            self.wave.clear()
            self.spectrum.clear()
            self.tf_image.clear()
            self.waterfall_image.clear()
            self.summary.setPlainText(f"原生复制完成：{result['plugin']['id']}\n"
                                      f"输出资产：{result['derived_asset_id']}\n请选择输出资产进行分析。")
            self.tabs.setCurrentIndex(0)



def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    window.show()
    return app.exec()
