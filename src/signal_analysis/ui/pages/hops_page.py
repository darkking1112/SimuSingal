"""跳频参数页（mixin）：逐跳估计的控件与渲染。"""
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
from ..constants import (IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NAME, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..dialogs import SignalParamsDialog
from ..helpers import (_AMC_SOURCE_TEXT, _asset_format, _comparison_line,
                       _fmt_hz, _fmt_metric, _fmt_span, _mirrored_spectrum)
from ..runner import _run_task
from ..model_picker import ModelPicker
from ..widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class HopsPageMixin:
    def build_hops(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "逐跳参数估计（契约 fh_hops_v1，算法 hop_track_v1）：在短时傅里叶变换的时频图上把相邻帧里"
            "同频段的能量连成轨迹，再对每条轨迹用较低门限在原框架上收敛一次频带、用时域带内功率"
            "收敛一次起止时间，得到每一跳的中心频率、单跳带宽、驻留时间与带内信噪比；一跳内的功率"
            "重心（质心）与中点一并给出，便于判断跳变帧是否被包进来。逐跳结果再按时间连续性聚成会话，"
            "给出跳速（跳起点间隔中位数的倒数）、跳频点数、跳频跨度与占空比。"
            "与会话级“信号检测”是两个粒度：那里只给整条跳频链路的频带，这里给每一跳的参数。"
            "同一页也可以渲染 AI 逐跳结果（算法 ml_detect_hops）：逐跳模型只在时频图上定位跳的"
            "频段与粗糙时间，驻留、带宽、功率与逐跳 SNR 仍在原始 PSD 上用同一套门限重测，"
            "所以两种算法的物理量口径一致、可以并排比较。"
            "IQ 为复基带记录：中心频率指基带频率偏移，不是射频载频。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        from common.gui import TaskBanner
        self.hops_banner = TaskBanner("跳频参数", "取消本次逐跳估计")
        self.register_task_banner("跳频参数", self.hops_banner)
        layout.addWidget(self.hops_banner)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("STFT 点数"))
        self.hops_nfft = QtWidgets.QComboBox()
        self.hops_nfft.addItems(["128", "256", "512", "1024", "2048", "4096"])
        self.hops_nfft.setCurrentText("512")
        self.hops_nfft.setToolTip("时间频率折衷：点数越大频率越精细，但每跳可用的帧数越少；"
                                  "可分辨的最快跳速 = 采样率 / (4 × 点数)")
        bar.addWidget(self.hops_nfft)
        bar.addWidget(QtWidgets.QLabel("逐帧门限"))
        threshold_row, self.hops_threshold = _plain_spin(0.5, 60.0, 6.0, 1, "dB")
        self.hops_threshold.setToolTip("逐帧判定跳频信号是否存在的门限（高于本底多少 dB），默认 6 dB；"
                                       "比会话级检测更保守，以抑制跳变帧的展宽")
        bar.addWidget(threshold_row)
        bar.addWidget(QtWidgets.QLabel("时间平滑"))
        self.hops_smooth = QtWidgets.QSpinBox()
        self.hops_smooth.setRange(1, 64)
        self.hops_smooth.setValue(4)
        self.hops_smooth.setToolTip("功率谱在时间方向上的滑动平均帧数，用于抑制噪声引起的虚假跳变")
        bar.addWidget(self.hops_smooth)
        bar.addWidget(QtWidgets.QLabel("最小带宽"))
        width_row, self.hops_min_bandwidth = _freq_spin(0.0, 1e9, 0.0, 1)
        self.hops_min_bandwidth.setToolTip("小于该带宽的频段不构成一跳，0 表示自动取 3 个频点")
        bar.addWidget(width_row)
        bar.addWidget(QtWidgets.QLabel("最小驻留"))
        dwell_row, self.hops_min_dwell = _plain_spin(0.0, 3600.0, 0.0, 4, "s")
        self.hops_min_dwell.setToolTip("驻留时间短于该值的轨迹被丢弃，0 表示自动取 4 帧")
        bar.addWidget(dwell_row)
        bar.addWidget(QtWidgets.QLabel("最多跳数"))
        self.hops_max = QtWidgets.QSpinBox()
        self.hops_max.setRange(1, 256)
        self.hops_max.setValue(256)
        self.hops_max.setToolTip("按带内功率从大到小保留的跳数上限")
        bar.addWidget(self.hops_max)
        bar.addWidget(QtWidgets.QLabel("粘合丢帧"))
        self.hops_gap = QtWidgets.QComboBox()
        self.hops_gap.addItems(["自动", "0", "1", "2", "4", "8", "16"])
        self.hops_gap.setToolTip("轨迹在时间上允许粘合的最大丢帧间隔，0 表示不允许（保守模式）")
        bar.addWidget(self.hops_gap)
        self.hops_sessions = QtWidgets.QCheckBox("附带会话基线")
        self.hops_sessions.setChecked(True)
        self.hops_sessions.setToolTip("勾选时在同一次任务里跑一遍会话级能量检测，给出两种粒度的指标对照")
        bar.addWidget(self.hops_sessions)
        bar.addWidget(QtWidgets.QLabel("频率显示"))
        self.hops_freq_view = QtWidgets.QComboBox()
        self.hops_freq_view.addItems(["自动", "双边", "仅正频率"])
        self.hops_freq_view.setToolTip(
            "横轴频率范围：自动 = 频谱关于 0 Hz 镜像（实数记录）时只显示非负频率，"
            "复数 IQ 保留双边；仅正频率适合只用正半轴的复基带信号。"
            "平均功率谱密度、时频图与逐跳框始终共用同一频率范围")
        self.hops_freq_view.currentIndexChanged.connect(
            lambda *_: self._apply_hops_display_mode())
        bar.addWidget(self.hops_freq_view)
        self.hops_button = QtWidgets.QPushButton("估计逐跳参数")
        self.hops_button.setObjectName("primary")
        self.hops_button.clicked.connect(self.hops_selected)
        bar.addWidget(self.hops_button)
        self.adopt_hops_button = QtWidgets.QPushButton("采纳为参数标注")
        self.adopt_hops_button.setEnabled(False)
        self.adopt_hops_button.setToolTip(
            "把当前逐跳结果写回会话与逐跳目标的参考参数（source=algorithm）；"
            "集合已有逐跳标注集时同时追加标签，全部为追加式。")
        self.adopt_hops_button.clicked.connect(
            lambda: self.adopt_page_result("跳频参数"))
        bar.addWidget(self.adopt_hops_button)
        bar.addStretch(1)
        layout.addLayout(bar)
        ai_bar = QtWidgets.QHBoxLayout()
        ai_bar.addWidget(QtWidgets.QLabel("逐跳模型清单"))
        # 路径由模型下拉写入，界面不再重复放输入框与“选择…”按钮（见 detect_page 同款说明）
        self.hops_manifest = QtWidgets.QLineEdit(self)
        self.hops_manifest.hide()
        self.hops_picker = ModelPicker(
            self.hops_manifest, "hop", self, browse=self.choose_hops_manifest,
            hint="只有逐跳标签（per_hop_v1）训练的模型可以走这条通路；会话级模型会被直接拒绝并提示"
                 "改用“信号检测”页，避免把整条跳频链路当成一跳。逐跳模型必选，"
                 "选“默认”时点「AI 估计逐跳参数」会被拦下。")
        self.register_model_picker(self.hops_picker)
        ai_bar.addWidget(self.hops_picker, 1)
        self.hops_ml_traditional = QtWidgets.QCheckBox("并排对比传统逐跳")
        self.hops_ml_traditional.setChecked(True)
        self.hops_ml_traditional.setToolTip(
            "勾选时在同一次任务里按同样的参数跑一遍能量逐跳（hop_track_v1）作为基线；"
            "两条通路的驻留、带宽、功率与 SNR 都在原始 PSD 上重测，可直接比较")
        ai_bar.addWidget(self.hops_ml_traditional)
        self.hops_ml_button = QtWidgets.QPushButton("AI 估计逐跳参数")
        self.hops_ml_button.setObjectName("primary")
        self.hops_ml_button.clicked.connect(self.ml_hops_selected)
        ai_bar.addWidget(self.hops_ml_button)
        self.hops_ml_status = QtWidgets.QLabel()
        ai_bar.addWidget(self.hops_ml_status)
        ai_bar.addStretch(1)
        layout.addLayout(ai_bar)
        grid = QtWidgets.QGridLayout()
        self.hops_spectrum = pg.PlotWidget(title="平均功率谱密度与逐跳频带")
        self.hops_spectrum.setLabel("bottom", "基带频率偏移", units="Hz")
        self.hops_spectrum.setLabel("left", "PSD（dB，参考 1 任意单位²/Hz）")
        self.hops_tf = pg.PlotWidget(title="时频图与逐跳框")
        self.hops_tf.setLabel("bottom", "基带频率偏移", units="Hz")
        self.hops_tf.setLabel("left", "时间", units="s")
        self.hops_tf_image = pg.ImageItem(axisOrder="row-major")
        self.hops_tf_image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.hops_tf.addItem(self.hops_tf_image)
        grid.addWidget(self.hops_spectrum, 0, 0)
        grid.addWidget(self.hops_tf, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        self._hops_items = []
        lower = QtWidgets.QHBoxLayout()
        self.hops_table = QtWidgets.QTableWidget(0, 10)
        self.hops_table.setHorizontalHeaderLabels(
            ["跳号", "会话", "中心频率", "单跳带宽", "频段范围", "时间范围", "驻留",
             "功率 dBFS", "逐跳 SNR", "备注"])
        self.hops_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hops_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.hops_table.horizontalHeader().setStretchLastSection(True)
        self.hops_table.setMaximumHeight(180)
        lower.addWidget(self.hops_table, 3)
        self.hops_session_table = QtWidgets.QTableWidget(0, 9)
        self.hops_session_table.setHorizontalHeaderLabels(
            ["会话", "跳数", "跳速", "跳周期", "驻留中位", "占空比", "跳频点数", "跳频跨度", "会话带宽"])
        self.hops_session_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hops_session_table.horizontalHeader().setStretchLastSection(True)
        self.hops_session_table.setMaximumHeight(180)
        lower.addWidget(self.hops_session_table, 2)
        layout.addLayout(lower)
        self.hops_summary = QtWidgets.QPlainTextEdit()
        self.hops_summary.setReadOnly(True)
        self.hops_summary.setMaximumHeight(205)
        self.hops_summary.setPlaceholderText("估计后显示本底与门限、可分辨上限、逐跳统计与真值误差。")
        layout.addWidget(self.hops_summary)
        self.update_hops_controls()
        self.update_ml_controls()
        return box


    def choose_hops_manifest(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 AI 逐跳模型清单（per_hop_v1）", str(self.workspace.root),
            "模型清单 (*.json)")
        if path:
            self.hops_manifest.setText(path)


    def ml_hops_selected(self):
        """AI 逐跳参数估计：模型只在时频图上定位跳，物理量仍在原始 PSD 上重测。"""
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        manifest = self.hops_manifest.text().strip()
        if not manifest:
            self.status.setText(
                "请先选择逐跳模型清单（需声明 label_semantics=per_hop_v1）")
            return
        config = {"score_threshold": float(self.ml_score.value()),
                  "iou_threshold": float(self.ml_iou.value()),
                  "threshold_db": float(self.hops_threshold.value()),
                  "smooth_frames": int(self.hops_smooth.value()),
                  "max_hops": int(self.hops_max.value())}
        if self.hops_min_bandwidth.value() > 0:
            config["min_bandwidth_hz"] = float(self.hops_min_bandwidth.value())
        if self.hops_min_dwell.value() > 0:
            config["min_dwell_s"] = float(self.hops_min_dwell.value())
        if self.hops_gap.currentText() != "自动":
            config["max_gap_frames"] = int(self.hops_gap.currentText())
        # nfft 与图像尺寸由清单声明，界面不覆盖：模型输入口径必须与训练时一致
        self.start_job("ml_detect_hops", owner="跳频参数",
                       label=f"AI 逐跳估计 · {asset['name']}", cancel_text="取消本次逐跳估计",
                       asset_id=asset["id"], manifest=manifest,
                       config=config, with_sessions=bool(self.hops_sessions.isChecked()),
                       with_traditional=bool(self.hops_ml_traditional.isChecked()))


    def update_hops_controls(self, *_):
        """占位：逐跳参数没有联动约束，保留与其他页一致的刷新入口。"""
        return


    def hops_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        config = {"nfft": int(self.hops_nfft.currentText()),
                  "threshold_db": float(self.hops_threshold.value()),
                  "smooth_frames": int(self.hops_smooth.value()),
                  "max_hops": int(self.hops_max.value())}
        if self.hops_min_bandwidth.value() > 0:
            config["min_bandwidth_hz"] = float(self.hops_min_bandwidth.value())
        if self.hops_min_dwell.value() > 0:
            config["min_dwell_s"] = float(self.hops_min_dwell.value())
        if self.hops_gap.currentText() != "自动":
            config["max_gap_frames"] = int(self.hops_gap.currentText())
        self.start_job("detect_hops", owner="跳频参数",
                       label=f"逐跳估计 · {asset['name']}", cancel_text="取消本次逐跳估计",
                       asset_id=asset["id"], config=config,
                       with_sessions=bool(self.hops_sessions.isChecked()))


    def _render_hops(self, result, arrays):
        summary = result["summary"]
        hops = result["hops"]
        sessions = result["sessions"]
        f = arrays["frequency"]
        t = arrays["frame_time"]
        df = float(summary["freq_resolution_hz"])
        dt = float(summary["frame_interval_s"])
        mask = self._frequency_mask(self.hops_freq_view.currentText(), f,
                                    arrays["spectrum_db"])
        positive_only = isinstance(mask, np.ndarray)
        fv = f[mask]
        matrix = arrays["spectrogram_db"][:, mask]
        boxes = arrays["hop_boxes"].reshape(-1, 4)
        hop_ids = arrays["hop_id"].ravel().astype(int)
        hop_sessions = arrays["hop_session_id"].ravel().astype(int)
        threshold_db = float(arrays["threshold_db"].ravel()[0])
        noise_db = float(arrays["noise_floor_db"].ravel()[0])
        high = float(matrix.max()) if matrix.size else -120.0
        if arrays["spectrum_db"].size:
            high = max(high, float(arrays["spectrum_db"][mask].max()))
        colours = ["#2365b3", "#e2564a", "#2f8f5b", "#b3732a", "#7a53a8", "#3f9fb5"]
        # 平均功率谱密度：与会话级检测同一张图，逐跳频带按会话着色区分
        self.hops_spectrum.clear()
        self.hops_spectrum.plot(fv, arrays["spectrum_median_db"][mask],
                                pen="#9fb6cd", name="中位数 PSD（对照）")
        self.hops_spectrum.plot(fv, arrays["spectrum_db"][mask],
                                pen="#2365b3", name="平均 PSD（检测依据）")
        self.hops_spectrum.addItem(pg.InfiniteLine(
            pos=threshold_db, angle=0, movable=False, pen=pg.mkPen("#e2564a", width=2)))
        self.hops_spectrum.addItem(pg.InfiniteLine(
            pos=noise_db, angle=0, movable=False,
            pen=pg.mkPen("#7d8fa1", style=QtCore.Qt.PenStyle.DashLine)))
        for index, item in enumerate(hops):
            colour = colours[(int(hop_sessions[index]) - 1) % len(colours)]
            self.hops_spectrum.addItem(pg.LinearRegionItem(
                values=(item["f_low_hz"], item["f_high_hz"]), movable=False,
                brush=pg.mkBrush(colour), pen=pg.mkPen(colour)))
        # 时频图叠加逐跳框：同一会话同色，框内时间范围就是驻留时间。
        # 与会话级检测同口径：row-major 的 ImageItem 纵向吃数组第一轴，
        # 所以必须给「行=时间、列=频率」的原始矩阵，转置会让图横过来。
        levels = [high - 80.0, high]
        self.hops_tf_image.setImage(matrix, levels=levels, autoLevels=False)
        self.hops_tf_image.setRect(QtCore.QRectF(fv[0] - df / 2, t[0] - dt / 2,
                                                 df * len(fv), dt * len(t)))
        view_box = self.hops_tf.getPlotItem().getViewBox()
        for item in self._hops_items:
            item.setParentItem(None)
            self.hops_tf.scene().removeItem(item)
        self._hops_items = []
        for index, (f_low, f_high, t_start, t_end) in enumerate(boxes):
            colour = colours[(int(hop_sessions[index]) - 1) % len(colours)]
            rectangle = QtWidgets.QGraphicsRectItem(QtCore.QRectF(
                float(f_low), float(t_start), float(f_high - f_low),
                max(float(t_end - t_start), dt)))
            rectangle.setPen(pg.mkPen(colour, width=2))
            rectangle.setParentItem(view_box)
            self._hops_items.append(rectangle)
        low = 0.0 if positive_only else -float(summary["sample_rate_hz"]) / 2.0
        high_edge = float(summary["sample_rate_hz"]) / 2.0
        side = "仅正频率" if positive_only else "双边"
        self.hops_spectrum.setXRange(low, high_edge, padding=0.0)
        self.hops_spectrum.setYRange(max(noise_db - 5.0, high - 80.0), high + 3.0, padding=0.0)
        self.hops_tf.setXRange(low, high_edge, padding=0.0)
        self.hops_tf.setYRange(float(t[0]) - dt / 2, float(t[-1]) + dt / 2, padding=0.0)
        self.hops_spectrum.setTitle(
            f"平均功率谱密度与逐跳频带 · {side} {_fmt_hz(low)}～{_fmt_hz(high_edge)}"
            f" · 本底 {noise_db:.1f} dB/Hz · 逐帧门限 {threshold_db:.1f} dB/Hz"
            f" · 频点 {df:g} Hz")
        self.hops_tf.setTitle(f"时频图与逐跳框 · {side} {_fmt_hz(low)}～{_fmt_hz(high_edge)}"
                              f" · 动态范围 80 dB · 帧间隔 {dt * 1e3:.3f} ms"
                              f" · 同色为同一会话")
        self.hops_table.setRowCount(len(hops))
        for row, item in enumerate(hops):
            note = (f"频点 {item['bin_count']} 个 · 细化 FFT {item['band_nfft']} 点 · "
                    f"质心偏移 {_fmt_hz(item['centroid_hz'] - item['center_hz'])}")
            if item.get("model_confidence") is not None:
                # 模型分数与置信度不是一回事：前者是“这一跳存在吗”，后者由带内 SNR 换算
                note += (f" · 模型置信度 {item['model_confidence']:.3f}"
                         f"（{item.get('model_label') or 'emitter'}）")
            values = [str(item["id"]), str(item["session_id"]), _fmt_hz(item["center_hz"]),
                      _fmt_hz(item["bandwidth_hz"]),
                      f"{_fmt_hz(item['f_low_hz'])} ～ {_fmt_hz(item['f_high_hz'])}",
                      f"{item['t_start_s']:.5f} ～ {item['t_end_s']:.5f} s",
                      f"{(item['dwell_s'] or 0.0) * 1e3:.3f} ms",
                      f"{item['power_dbfs']:.2f}", f"{item['snr_db']:.2f} dB", note]
            for column, text in enumerate(values):
                self.hops_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self.hops_session_table.setRowCount(len(sessions))
        for row, item in enumerate(sessions):
            values = [str(item["session_id"]), str(item["hop_count"]),
                      _fmt_metric(item["hop_rate_hz"], ".2f"), _fmt_metric(item["hop_period_s"], ".6f"),
                      _fmt_metric(item["dwell_median_s"], ".6f"), _fmt_metric(item["duty_cycle"], ".3f"),
                      str(len(item["hop_frequencies_hz"])), _fmt_hz(item["hop_span_hz"]),
                      _fmt_hz(item["bandwidth_hz"])]
            for column, text in enumerate(values):
                self.hops_session_table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        config = summary["config"]
        lines = [
            f"数据：{result.get('asset_name', result['asset_id'])}  |  "
            f"采样数 {summary['sample_count']:,}  |  时长 {summary['duration_s']:.6f} s  |  "
            f"采样率 {_fmt_hz(summary['sample_rate_hz'])}",
            f"算法 {summary['algorithm']}（契约 {summary['contract']}）· "
            f"带内信噪比定义 {summary['snr_definition']} · 频率参考 {summary['frequency_reference']}",
            f"STFT {summary['nfft']} 点（频点 {df:g} Hz，帧 {summary['frame_count']} 个，"
            f"跳变帧 {summary['transition_frames']} 个）· 噪声本底 "
            f"{summary['noise_floor_dbfs_per_hz']:.1f} dB/Hz · 逐帧门限 "
            f"{summary['threshold_dbfs_per_hz']:.1f} dB/Hz · 时间平滑 {config['smooth_frames']} 帧",
            f"可分辨性：最短驻留过滤 {config['min_dwell_s'] * 1e3:.3f} ms · "
            f"可分辨门限 {summary['dwell_limit_s'] * 1e3:.3f} ms（帧间距 × 4）· "
            f"理论最快跳速 {summary['hop_rate_limit_hz']:.1f} Hz · "
            f"{'可分辨' if summary['resolvable'] else '不可分辨'}",
            f"逐跳结果：{len(hops)} 跳 · {len(sessions)} 个会话",
        ]
        if summary.get("reason"):
            lines.append(f"提示：{summary['reason']}")
        model = summary.get("model")
        if isinstance(model, dict) and model.get("id"):
            lines.append(
                f"逐跳模型：{model['id']}@{model.get('version') or '--'} · "
                f"运行时 {model.get('runtime_version') or '--'} · "
                f"图像 {summary.get('image_size', '--')} 像素 · "
                f"原始候选框 {summary.get('raw_boxes', '--')} 个（模型只定位跳，物理量重测）")
        metrics = result.get("metrics")
        if metrics:
            lines.append(
                f"逐跳真值 {metrics['true']} 个：匹配 {metrics['matched']} · 漏警 {metrics['missed']} · "
                f"虚警 {metrics['false_alarm']} · 精确率 {_fmt_metric(metrics['precision'], '4g')} · "
                f"召回 {_fmt_metric(metrics['recall'], '4g')} · F1 {_fmt_metric(metrics['f1'], '4g')}")
            lines.append(
                f"参数误差：中心频率 MAE {_fmt_metric(metrics['center_mae_hz'], '.1f')} Hz · "
                f"单跳带宽相对误差 {_fmt_metric(metrics['bandwidth_mape'], '.1%')} · "
                f"逐跳信噪比 MAE {_fmt_metric(metrics['snr_mae_db'], '.2f')} dB")
        else:
            truth = result.get("truth")
            reason = truth.get("reason") if isinstance(truth, dict) else "没有生成器真值"
            lines.append(f"逐跳真值：不适用 —— {reason}；结果仍然给出，不丢弃。")
        for item in sessions:
            lines.append(
                f"  会话 {item['session_id']}：{item['hop_count']} 跳 · 跳速 "
                f"{_fmt_metric(item['hop_rate_hz'], '.2f')} Hz"
                f"（周期 {_fmt_metric(item['hop_period_s'], '.6f')} s · 驻留中位 "
                f"{_fmt_metric(item['dwell_median_s'], '.6f')} s · 占空比 "
                f"{_fmt_metric(item['duty_cycle'], '.3f')}）· "
                f"跳频点 {len(item['hop_frequencies_hz'])} 个"
                f"（{', '.join(_fmt_hz(value) for value in item['hop_frequencies_hz'][:8])}"
                f"{'…' if len(item['hop_frequencies_hz']) > 8 else ''}）· "
                f"跨度 {_fmt_hz(item['hop_span_hz'])} · 会话带宽 {_fmt_hz(item['bandwidth_hz'])} · "
                f"带内信噪比 {_fmt_metric(item['snr_db'], '.2f')} dB · "
                f"会话级检出 {item['session_detection_id'] or '未关联'}")
        if result.get("traditional_metrics"):
            lines.append(_comparison_line(
                "AI 逐跳" if summary.get("model") else "逐跳口径", metrics, "传统逐跳",
                result["traditional_metrics"]))
        if result.get("baseline_metrics"):
            lines.append(_comparison_line("逐跳口径", metrics, "会话口径",
                                          result["baseline_metrics"]))
        self.hops_summary.setPlainText("\n".join(lines))
