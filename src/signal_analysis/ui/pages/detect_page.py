"""信号检测页（mixin）：传统与 AI（含逐跳模型）检测的控件与渲染。"""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ...contracts.iq import IQ_WAVEFORM_CONTRACT
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


class DetectPageMixin:
    def build_detect(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "能量检测与参数估计（基线算法 energy_detect_v1）：对所选 IQ 记录做短时傅里叶变换，"
            "以中位数 + MAD 估计噪声本底，高于检测门限的连续频段判为目标；带宽按较低的带宽门限在"
            "同一连通区内测量，从而保留 AM 载波两侧较弱的边带。输出中心频率（功率重心所在频段的"
            "中点）、占用带宽、起止时间、带内功率与带内信噪比（信号功率 / 同频段噪声功率）。"
            "生成器产出的数据自带真值，可直接给出匹配、漏警、虚警与参数误差。"
            "IQ 为复基带记录：中心频率指基带频率偏移，不是射频载频。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        from common.gui import TaskBanner
        self.detect_banner = TaskBanner("信号检测", "取消本次检测")
        self.register_task_banner("信号检测", self.detect_banner)
        layout.addWidget(self.detect_banner)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("STFT 点数"))
        self.detect_nfft = QtWidgets.QComboBox()
        self.detect_nfft.addItems(["128", "256", "512", "1024", "2048", "4096"])
        self.detect_nfft.setCurrentText("512")
        self.detect_nfft.setToolTip("频点分辨率 = 采样率 / STFT 点数；点数越大频率越精细、时间粒度越粗")
        bar.addWidget(self.detect_nfft)
        bar.addWidget(QtWidgets.QLabel("检测门限"))
        threshold_row, self.detect_threshold = _plain_spin(0.5, 60.0, 3.0, 1, "dB")
        self.detect_threshold.setToolTip("高于噪声本底该值（dB）的频点参与检测，默认 3 dB")
        bar.addWidget(threshold_row)
        bar.addWidget(QtWidgets.QLabel("带宽门限"))
        band_row, self.detect_band_threshold = _plain_spin(0.0, 60.0, 1.5, 1, "dB")
        self.detect_band_threshold.setToolTip("测量占用带宽时使用的较低门限，不得超过检测门限；"
                                              "越小越能保留弱边带，但噪声也更容易把频段抬宽")
        bar.addWidget(band_row)
        self.detect_band_auto = QtWidgets.QCheckBox("自动")
        self.detect_band_auto.setChecked(True)
        self.detect_band_auto.setToolTip("勾选时取检测门限的一半；取消后可自行指定，但不得超过检测门限")
        self.detect_band_auto.toggled.connect(self.update_detect_controls)
        self.detect_threshold.valueChanged.connect(self.update_detect_controls)
        bar.addWidget(self.detect_band_auto)
        bar.addWidget(QtWidgets.QLabel("最小带宽"))
        width_row, self.detect_min_bandwidth = _freq_spin(0.0, 1e9, 0.0, 1)
        self.detect_min_bandwidth.setToolTip("小于该占用带宽的频段视为噪声，0 表示自动取 3 个频点")
        bar.addWidget(width_row)
        bar.addWidget(QtWidgets.QLabel("最小时长"))
        duration_row, self.detect_min_duration = _plain_spin(0.0, 3600.0, 0.0, 4, "s")
        self.detect_min_duration.setToolTip("持续时间短于该值的目标将被丢弃，0 表示不限制")
        bar.addWidget(duration_row)
        bar.addWidget(QtWidgets.QLabel("最多目标"))
        self.detect_max = QtWidgets.QSpinBox()
        self.detect_max.setRange(1, 256)
        self.detect_max.setValue(32)
        self.detect_max.setToolTip("按带内功率从大到小保留的目标数")
        bar.addWidget(self.detect_max)
        bar.addWidget(QtWidgets.QLabel("平滑半径"))
        self.detect_merge = QtWidgets.QComboBox()
        self.detect_merge.addItems(["自动", "0", "1", "2", "4", "8", "16"])
        self.detect_merge.setToolTip("形态学闭运算半径（频点），用于把同一目标被衰落切开的频段合并")
        bar.addWidget(self.detect_merge)
        bar.addWidget(QtWidgets.QLabel("频率显示"))
        self.detect_freq_view = QtWidgets.QComboBox()
        self.detect_freq_view.addItems(["自动", "双边", "仅正频率"])
        self.detect_freq_view.setToolTip(
            "横轴频率范围：自动 = 频谱关于 0 Hz 镜像（实数记录）时只显示非负频率，"
            "复数 IQ 保留双边；仅正频率适合只用正半轴的复基带信号。"
            "平均功率谱密度、时频图与检测框始终共用同一频率范围")
        self.detect_freq_view.currentIndexChanged.connect(
            lambda *_: self._apply_detect_display_mode())
        bar.addWidget(self.detect_freq_view)
        self.detect_button = QtWidgets.QPushButton("检测所选数据")
        self.detect_button.setObjectName("primary")
        self.detect_button.clicked.connect(self.detect_selected)
        bar.addWidget(self.detect_button)
        self.adopt_detect_button = QtWidgets.QPushButton("采纳为参数标注")
        self.adopt_detect_button.setEnabled(False)
        self.adopt_detect_button.setToolTip(
            "把当前检测结果写入目标参考参数（source=algorithm）；集合已有检测标注集时"
            "同时追加标签。全部为追加式：旧版本与旧标签保留，重复采纳产生新版本。")
        self.adopt_detect_button.clicked.connect(
            lambda: self.adopt_page_result("信号检测"))
        bar.addWidget(self.adopt_detect_button)
        bar.addStretch(1)
        layout.addLayout(bar)
        ai_bar = QtWidgets.QHBoxLayout()
        ai_bar.addWidget(QtWidgets.QLabel("AI 模型清单"))
        self.ml_manifest = QtWidgets.QLineEdit()
        self.ml_manifest.setPlaceholderText("选择 ml-manifest 生成的 JSON（含 ONNX 相对路径与 SHA-256）")
        self.ml_manifest.setToolTip("清单声明输入图像尺寸、STFT 点数与动态范围；推理时与清单不一致会直接报错")
        ai_bar.addWidget(self.ml_manifest, 1)
        self.ml_choose = QtWidgets.QPushButton("选择…")
        self.ml_choose.clicked.connect(self.choose_ml_manifest)
        ai_bar.addWidget(self.ml_choose)
        ai_bar.addWidget(QtWidgets.QLabel("置信度阈值"))
        self.ml_score = QtWidgets.QDoubleSpinBox()
        self.ml_score.setRange(0.01, 0.99)
        self.ml_score.setSingleStep(0.05)
        self.ml_score.setDecimals(2)
        self.ml_score.setValue(0.25)
        self.ml_score.setToolTip("低于该置信度的模型候选框被丢弃，默认 0.25")
        ai_bar.addWidget(self.ml_score)
        ai_bar.addWidget(QtWidgets.QLabel("去重 IoU"))
        self.ml_iou = QtWidgets.QDoubleSpinBox()
        self.ml_iou.setRange(0.0, 1.0)
        self.ml_iou.setSingleStep(0.05)
        self.ml_iou.setDecimals(2)
        self.ml_iou.setValue(0.5)
        self.ml_iou.setToolTip("重叠度超过该值的同类别候选框只保留置信度最高的一个，默认 0.5")
        ai_bar.addWidget(self.ml_iou)
        self.ml_compare = QtWidgets.QCheckBox("并排对比传统检测")
        self.ml_compare.setChecked(True)
        self.ml_compare.setToolTip("勾选时在同一次任务里跑一遍能量检测作为基线，并给出两项指标对照")
        ai_bar.addWidget(self.ml_compare)
        self.ml_button = QtWidgets.QPushButton("AI 检测所选数据")
        self.ml_button.setObjectName("primary")
        self.ml_button.clicked.connect(self.ml_detect_selected)
        ai_bar.addWidget(self.ml_button)
        self.ml_status = QtWidgets.QLabel()
        ai_bar.addWidget(self.ml_status)
        ai_bar.addStretch(1)
        layout.addLayout(ai_bar)
        self.update_ml_controls()
        grid = QtWidgets.QGridLayout()
        self.detect_spectrum = pg.PlotWidget(title="平均功率谱密度与检测门限")
        self.detect_spectrum.setLabel("bottom", "基带频率偏移", units="Hz")
        self.detect_spectrum.setLabel("left", "PSD（dB，参考 1 任意单位²/Hz）")
        self.detect_tf = pg.PlotWidget(title="时频图与检测框")
        self.detect_tf.setLabel("bottom", "基带频率偏移", units="Hz")
        self.detect_tf.setLabel("left", "时间", units="s")
        self.detect_tf_image = pg.ImageItem(axisOrder="row-major")
        self.detect_tf_image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.detect_tf.addItem(self.detect_tf_image)
        grid.addWidget(self.detect_spectrum, 0, 0)
        grid.addWidget(self.detect_tf, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        self._detect_items = []
        lower = QtWidgets.QHBoxLayout()
        self.detect_table = QtWidgets.QTableWidget(0, 9)
        self.detect_table.setHorizontalHeaderLabels(
            ["编号", "中心频率", "占用带宽", "频段范围", "时间范围", "功率 dBFS",
             "带内 SNR", "会话", "备注"])
        self.detect_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.detect_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.detect_table.horizontalHeader().setStretchLastSection(True)
        self.detect_table.setMaximumHeight(180)
        lower.addWidget(self.detect_table, 1)
        self.detect_summary = QtWidgets.QPlainTextEdit()
        self.detect_summary.setReadOnly(True)
        self.detect_summary.setMaximumHeight(180)
        self.detect_summary.setPlaceholderText("检测后显示噪声本底、门限、目标数量与参数误差。")
        lower.addWidget(self.detect_summary, 1)
        layout.addLayout(lower)
        self.update_detect_controls()
        return box


    def update_detect_controls(self, *_):
        """带宽门限默认跟随检测门限的一半；手动模式只做上界约束。"""
        auto = self.detect_band_auto.isChecked()
        threshold = float(self.detect_threshold.value())
        self.detect_band_threshold.setEnabled(not auto)
        if auto:
            self.detect_band_threshold.setValue(max(0.1, 0.5 * threshold))
        elif self.detect_band_threshold.value() > threshold:
            self.detect_band_threshold.setValue(max(0.1, 0.5 * threshold))


    def update_ml_controls(self):
        """推理运行时缺失时禁用 AI 入口并给出安装提示（传统路径不受影响）。

        tab 2 先于 tab 5 构建，所以这里按名字取控件：还没建的先跳过，
        待对应标签页构建完成时再调一次即可。
        """
        from ...inference.runtime import runtime_version

        version = runtime_version()
        ready = version is not None
        for name in ("ml_button", "ml_choose", "ml_score", "ml_iou", "ml_compare",
                     "hops_ml_button", "hops_ml_choose", "hops_ml_traditional"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(ready)
        note = (f"onnxruntime {version}" if ready else
                "未安装 onnxruntime：pip install '.[ml]' 后可用")
        for name in ("ml_status", "hops_ml_status"):
            label = getattr(self, name, None)
            if label is not None:
                label.setText(note)


    def choose_ml_manifest(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 AI 检测模型清单", str(self.workspace.root), "模型清单 (*.json)")
        if path:
            self.ml_manifest.setText(path)


    def ml_detect_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        manifest = self.ml_manifest.text().strip()
        if not manifest:
            self.status.setText("请先选择 AI 模型清单（ml-manifest 生成）")
            return
        config = {"threshold_db": float(self.detect_threshold.value()),
                  "score_threshold": float(self.ml_score.value()),
                  "iou_threshold": float(self.ml_iou.value()),
                  "max_detections": int(self.detect_max.value())}
        if self.detect_min_bandwidth.value() > 0:
            config["min_bandwidth_hz"] = float(self.detect_min_bandwidth.value())
        if self.detect_min_duration.value() > 0:
            config["min_duration_s"] = float(self.detect_min_duration.value())
        # nfft 与图像尺寸由清单声明，界面不覆盖，避免训练/推理口径分叉
        self.start_job("ml_detect", owner="信号检测",
                       label=f"AI 检测 · {asset['name']}", cancel_text="取消本次检测",
                       asset_id=asset["id"], manifest=manifest,
                       config=config, with_baseline=bool(self.ml_compare.isChecked()))


    def detect_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        config = {"nfft": int(self.detect_nfft.currentText()),
                  "threshold_db": float(self.detect_threshold.value()),
                  "band_threshold_db": float(self.detect_band_threshold.value()),
                  "max_detections": int(self.detect_max.value())}
        if self.detect_min_bandwidth.value() > 0:
            config["min_bandwidth_hz"] = float(self.detect_min_bandwidth.value())
        if self.detect_min_duration.value() > 0:
            config["min_duration_s"] = float(self.detect_min_duration.value())
        if self.detect_merge.currentText() != "自动":
            config["merge_bins"] = int(self.detect_merge.currentText())
        self.start_job("detect", owner="信号检测",
                       label=f"能量检测 · {asset['name']}", cancel_text="取消本次检测",
                       asset_id=asset["id"], config=config)


    def _frequency_mask(self, choice, frequencies, spectrum_db):
        """按“频率显示”选择给出频率轴索引掩码；``slice(None)`` 表示双边。

        「自动」按频谱是否关于 0 Hz 镜像判断：实数记录的正负半轴互为镜像，
        只看正半轴即可；复数 IQ 的双边谱都可能有信号，保持双边。检测页与
        跳频页共用这一判据，避免再读一遍原始样本来测 ``imag == 0``。
        """
        if choice == "仅正频率" or (choice == "自动" and _mirrored_spectrum(spectrum_db)):
            return frequencies >= 0
        return slice(None)

    def _render_detect(self, result, arrays):
        summary = result["summary"]
        detections = summary["detections"]
        f = arrays["frequency"]
        t = arrays["frame_time"]
        df = float(summary["freq_resolution_hz"])
        dt = float(summary["hop_samples"]) / float(summary["sample_rate_hz"])
        mask = self._frequency_mask(self.detect_freq_view.currentText(), f,
                                    arrays["spectrum_db"])
        positive_only = isinstance(mask, np.ndarray)
        fv = f[mask]
        matrix = arrays["spectrogram_db"][:, mask]
        boxes = arrays["detection_boxes"].reshape(-1, 4)
        baseline_boxes = (arrays["baseline_detection_boxes"].reshape(-1, 4)
                          if "baseline_detection_boxes" in arrays else [])
        is_ml = bool(summary.get("model"))
        threshold_db = float(arrays["threshold_db"].ravel()[0])
        noise_db = float(arrays["noise_floor_db"].ravel()[0])
        high = float(matrix.max()) if matrix.size else -120.0
        if arrays["spectrum_db"].size:
            high = max(high, float(arrays["spectrum_db"][mask].max()))
        # 平均功率谱密度：检测依据，与本底/门限一起显示；中位数 PSD 作对照
        self.detect_spectrum.clear()
        self.detect_spectrum.plot(fv, arrays["spectrum_median_db"][mask],
                                  pen="#9fb6cd", name="中位数 PSD（对照）")
        self.detect_spectrum.plot(fv, arrays["spectrum_db"][mask],
                                  pen="#2365b3", name="平均 PSD（检测依据）")
        self.detect_spectrum.addItem(pg.InfiniteLine(
            pos=threshold_db, angle=0, movable=False, pen=pg.mkPen("#e2564a", width=2)))
        self.detect_spectrum.addItem(pg.InfiniteLine(
            pos=noise_db, angle=0, movable=False,
            pen=pg.mkPen("#7d8fa1", style=QtCore.Qt.PenStyle.DashLine)))
        for item in detections:
            self.detect_spectrum.addItem(pg.LinearRegionItem(
                values=(item["f_low_hz"], item["f_high_hz"]), movable=False,
                brush=pg.mkBrush(35, 101, 179, 45), pen=pg.mkPen("#2365b3")))
        # 时频图叠加检测框（横轴频率、纵轴时间）：ImageItem 为 row-major，
        # 数组第一轴落在纵向，因此这里必须给「行=时间、列=频率」的原始矩阵；
        # 传转置矩阵会把时频图横过来，看起来像凭空多出负频率。
        levels = [high - 80.0, high]
        self.detect_tf_image.setImage(matrix, levels=levels, autoLevels=False)
        self.detect_tf_image.setRect(QtCore.QRectF(fv[0] - df / 2, t[0] - dt / 2,
                                                   df * len(fv), dt * len(t)))
        view_box = self.detect_tf.getPlotItem().getViewBox()
        for item in self._detect_items:
            item.setParentItem(None)
            self.detect_tf.scene().removeItem(item)
        self._detect_items = []
        for f_low, f_high, t_start, t_end in boxes:
            rectangle = QtWidgets.QGraphicsRectItem(QtCore.QRectF(
                float(f_low), float(t_start), float(f_high - f_low),
                max(float(t_end - t_start), dt)))
            rectangle.setPen(pg.mkPen("#e2564a", width=2))
            rectangle.setParentItem(view_box)
            self._detect_items.append(rectangle)
        for f_low, f_high, t_start, t_end in baseline_boxes:
            rectangle = QtWidgets.QGraphicsRectItem(QtCore.QRectF(
                float(f_low), float(t_start), float(f_high - f_low),
                max(float(t_end - t_start), dt)))
            rectangle.setPen(pg.mkPen("#7d8fa1", width=1, style=QtCore.Qt.PenStyle.DashLine))
            rectangle.setParentItem(view_box)
            self._detect_items.append(rectangle)
        low = 0.0 if positive_only else -float(summary["sample_rate_hz"]) / 2.0
        high_edge = float(summary["sample_rate_hz"]) / 2.0
        side = "仅正频率" if positive_only else "双边"
        self.detect_spectrum.setXRange(low, high_edge, padding=0.0)
        self.detect_spectrum.setYRange(max(noise_db - 5.0, high - 80.0), high + 3.0, padding=0.0)
        self.detect_tf.setXRange(low, high_edge, padding=0.0)
        self.detect_tf.setYRange(float(t[0]) - dt / 2, float(t[-1]) + dt / 2, padding=0.0)
        self.detect_tf.setTitle(f"时频图与检测框 · {side} {_fmt_hz(low)}～{_fmt_hz(high_edge)}"
                                f" · 动态范围 80 dB"
                                + (" · 红框 AI 检出，灰虚线传统基线" if len(baseline_boxes) else ""))
        self.detect_spectrum.setTitle(
            f"平均功率谱密度与检测门限 · {side} {_fmt_hz(low)}～{_fmt_hz(high_edge)}"
            f" · 本底 {noise_db:.1f} dB/Hz · 门限 {threshold_db:.1f} dB/Hz"
            f" · 带宽门限 {summary['config']['band_threshold_db']:.1f} dB")
        self.detect_table.setRowCount(len(detections))
        for row, item in enumerate(detections):
            if is_ml:
                note = f"{item.get('label', '目标')} · 置信度 {item['confidence']:.2f}"
                if item["hopping"]:
                    note += (f" · 跳频会话 · 子带 {item['sub_bands']} 个")
            else:
                note = (f"跳频会话 · 子带 {item['sub_bands']} 个" if item["hopping"]
                        else f"频点 {item['bin_count']} 个")
            values = [str(item["id"]), _fmt_hz(item["center_hz"]), _fmt_hz(item["bandwidth_hz"]),
                      f"{_fmt_hz(item['f_low_hz'])} ～ {_fmt_hz(item['f_high_hz'])}",
                      f"{item['t_start_s']:.4f} ～ {item['t_end_s']:.4f} s",
                      f"{item['power_dbfs']:.2f}", f"{item['snr_db']:.2f} dB",
                      "-" if item["session_id"] is None else str(item["session_id"]), note]
            for column, text in enumerate(values):
                self.detect_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        lines = [
            f"数据：{result.get('asset_name', result['asset_id'])}  |  "
            f"采样数 {summary['sample_count']:,}  |  时长 {summary['duration_s']:.6f} s  |  "
            f"采样率 {_fmt_hz(summary['sample_rate_hz'])}",
            f"算法 {summary['algorithm']}（契约 {summary['contract']}）· "
            f"带内信噪比定义 {summary['snr_definition']} · 频率参考 {summary['frequency_reference']}",
            f"STFT {summary['nfft']} 点（频点 {summary['freq_resolution_hz']:g} Hz）· "
            f"{summary['frame_count']} 帧 · 噪声本底 {summary['noise_floor_dbfs_per_hz']:.1f} dB/Hz · "
            f"检测门限 {summary['threshold_dbfs_per_hz']:.1f} dB/Hz · "
            f"最小带宽 {_fmt_hz(summary['config']['min_bandwidth_hz'])} · "
            f"平滑半径 {summary['config']['merge_bins']} 频点",
            f"检出目标 {len(detections)} 个（跳频会话 {sum(1 for item in detections if item['hopping'])} 个）",
        ]
        if is_ml:
            model = summary["model"]
            lines.insert(1, f"模型 {model['id']}@{model['version']} · 摘要 {str(model['sha256'])[:12]}… · "
                            f"输入 {summary['image']['size']}² 灰度时频图（动态范围 "
                            f"{summary['image']['db_ceiling'] - summary['image']['db_floor']:.0f} dB）· "
                            f"置信度阈值 {summary['config']['score_threshold']:.2f} · 去重 IoU "
                            f"{summary['config']['iou_threshold']:.2f}")
            lines.append(f"模型候选 {summary['raw_boxes']['candidates']} 个（输出 "
                         f"{summary['raw_boxes']['rows']} 行）· 上下文 "
                         f"{summary['timing']['context_ms']:.1f} ms · 推理 "
                         f"{summary['timing']['inference_ms']:.1f} ms · 合计 "
                         f"{summary['timing']['total_ms']:.1f} ms")
        metrics = result.get("metrics")
        if metrics:
            lines.append(
                f"真值 {metrics['true']} 个：匹配 {metrics['matched']} · 漏警 {metrics['missed']} · "
                f"虚警 {metrics['false_alarm']} · 精确率 {_fmt_metric(metrics['precision'], '4g')} · "
                f"召回 {_fmt_metric(metrics['recall'], '4g')} · F1 {_fmt_metric(metrics['f1'], '4g')}")
            lines.append(
                f"参数误差：中心频率 MAE {_fmt_metric(metrics['center_mae_hz'], '.1f')} Hz · "
                f"带宽相对误差 {_fmt_metric(metrics['bandwidth_mape'], '.1%')} · "
                f"带内信噪比 MAE {_fmt_metric(metrics['snr_mae_db'], '.2f')} dB")
            for entry in result["truth"]:
                lines.append(
                    f"  真值 #{entry['index']} {entry['mode']}{'（跳频会话）' if entry['hopping'] else ''}："
                    f"中心 {_fmt_hz(entry['center_hz'])} · 带宽 {_fmt_hz(entry['bandwidth_hz'])} · "
                    f"带内信噪比 {_fmt_metric(entry['snr_inband_db'], '.2f')} dB")
            if result.get("baseline_metrics"):
                lines.append(_comparison_line("AI 检测", metrics,
                                              "传统基线", result["baseline_metrics"]))
        else:
            lines.append("该数据没有生成器真值（导入或原生插件产出），只给出检测结果，不计算误差指标。")
        self.detect_summary.setPlainText("\n".join(lines))
