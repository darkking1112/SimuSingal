"""Independent signal analysis desktop."""
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from common.gui import DesktopWindow
from common.reports import amc_metrics, detection_metrics
from .core_api import MAX_SAMPLES, plan_signal, spectrum_row
from .storage import Workspace
from .tasks import run_job

MODE_CHOICES = [("am", "AM 调幅"), ("fm", "FM 调频"), ("ssb", "SSB 单边带"),
                ("ask2", "2ASK 二进制幅移键控"), ("qpsk", "QPSK 四相相移键控"),
                ("qam16", "16QAM 正交幅度调制"), ("qam64", "64QAM 正交幅度调制"),
                ("fh_rc", "跳频 · 遥控链路 (FH-2FSK)"), ("fh_video", "跳频 · 图传链路 (FH-OFDM)")]
MODE_SHORT = {"am": "AM", "fm": "FM", "ssb": "SSB", "ask2": "2ASK", "qpsk": "QPSK",
              "qam16": "16QAM", "qam64": "64QAM", "fh_rc": "FH遥控", "fh_video": "FH图传"}
EXPORT_FORMATS = [("不导出（仅内部资产 .npy）", ""), ("NPY 格式 (.npy)", "npy"),
                  ("CSV 两列 I,Q (.csv)", "csv"), ("交织 IQ · int16 (.bin)", "iq16"),
                  ("交织 IQ · float32 (.bin)", "iq32"),
                  ("SigMF 双文件 (.sigmf-meta + .sigmf-data)", "sigmf")]
# 滚动瀑布图：时间窗内最多保留的帧数与单次刷新最多计算的帧数。
PLAY_MAX_ROWS = 360
PLAY_MAX_ROWS_PER_TICK = 64
# 波形图单次刷新最多绘制的样点数（超出则等间隔抽取）。
PLAY_WAVE_POINTS = 2000


def _run_task(request, cancel=None):
    # 16M 样本的生成与导出可能明显超过默认 30 s 子进程超时。
    timeout = 600.0 if request.get("action") == "generate" else 30.0
    return run_job(request, timeout=timeout, cancel=cancel)


class UnitSpinBox(QtWidgets.QDoubleSpinBox):
    """Stores Hz internally; displays the value in kHz / MHz / GHz.

    The unit is shown in a QLabel beside the box (``unit_label``), not inside
    the input; it switches automatically when the value crosses a boundary.
    """

    _UNITS = (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3), ("Hz", 1.0))

    def __init__(self, minimum, maximum, value, decimals):
        self._scale = 1.0  # 必须先于 setRange/setDecimals：Qt 内部会调用 textFromValue
        super().__init__()
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.unit_label = QtWidgets.QLabel("Hz")
        self.setValue(value)
        self._sync_unit()
        self.setAccelerated(True)
        self.valueChanged.connect(self._sync_unit)

    def _sync_unit(self, *_):
        unit, scale = self._UNITS[-1]
        for candidate, factor in self._UNITS:
            if abs(self.value()) >= factor:
                unit, scale = candidate, factor
                break
        if scale != self._scale:
            self._scale = scale
            self.unit_label.setText(unit)
            self.setSingleStep(scale)
            self.update()

    def textFromValue(self, value):
        return f"{value / self._scale:.{self.decimals()}f}"

    def valueFromText(self, text):
        return float(text.strip()) * self._scale


def _fmt_hz(value):
    """Format a Hz value with a readable kHz / MHz / GHz unit."""
    value = float(value)
    for unit, factor in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if abs(value) >= factor:
            return f"{value / factor:g} {unit}"
    return f"{value:g} Hz"


def _fmt_span(seconds):
    """Format a duration in ms / s."""
    seconds = float(seconds)
    return f"{seconds * 1000:g} ms" if seconds < 1.0 else f"{seconds:g} s"


def _fmt_metric(value, spec):
    """Format an optional metric; missing values (无真值/未定义) show as “--”。"""
    return "--" if value is None else format(float(value), spec)


def _comparison_line(left_label, left, right_label, right):
    """One-line side-by-side detection metrics (AI vs traditional baseline)."""
    fields = (("匹配", "matched", "g"), ("漏警", "missed", "g"), ("虚警", "false_alarm", "g"),
              ("中心 MAE", "center_mae_hz", ".1f"), ("带宽相对误差", "bandwidth_mape", ".3f"),
              ("信噪比 MAE", "snr_mae_db", ".2f"))
    parts = [f"{name} {_fmt_metric(left.get(key), spec)} / {_fmt_metric(right.get(key), spec)}"
             for name, key, spec in fields]
    return (f"并排对比（{left_label} / {right_label}）：" + " · ".join(parts))


_AMC_SOURCE_TEXT = {"builtin": "内置基线", "file": "指定模型文件", "onnx": "ONNX 分类器",
                   "inline": "内存模型"}


def _unit_row(spin):
    """Place a unit label beside a spin box (unit outside the input)."""
    container = QtWidgets.QWidget()
    row = QtWidgets.QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(spin)
    row.addWidget(spin.unit_label)
    return container


def _plain_spin(minimum, maximum, value, decimals, unit=None):
    """Plain double spin; with ``unit`` the label is placed outside the box."""
    spin = QtWidgets.QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(decimals)
    spin.setValue(value)
    if unit is None:
        return spin
    spin.unit_label = QtWidgets.QLabel(unit)
    return _unit_row(spin), spin


def _freq_spin(minimum, maximum, value, decimals):
    """Frequency spin in Hz with a kHz / MHz / GHz unit label outside."""
    spin = UnitSpinBox(minimum, maximum, value, decimals)
    return _unit_row(spin), spin


class BinaryImportDialog(QtWidgets.QDialog):
    """Explicit datatype/endian choice for headerless interleaved IQ files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("交织 IQ 二进制参数")
        layout = QtWidgets.QFormLayout(self)
        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItem("int16（有符号 16 位，按 1/32768 缩放）", "int16")
        self.dtype.addItem("float32（单精度浮点）", "float32")
        layout.addRow("数据类型", self.dtype)
        self.endian = QtWidgets.QComboBox()
        self.endian.addItem("小端（little-endian）", "little")
        self.endian.addItem("大端（big-endian）", "big")
        layout.addRow("字节序", self.endian)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self):
        return self.dtype.currentData(), self.endian.currentData()


class SignalParamsDialog(QtWidgets.QDialog):
    """Per-mode parameter editor with a live auto-parameter preview."""

    def __init__(self, mode, rate, spec=None, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.rate = rate
        spec = dict(spec or {})
        self._source_spec = spec
        self._last_spec = None
        self.setWindowTitle(f"信号参数 · {MODE_SHORT[mode]}")
        form = QtWidgets.QFormLayout(self)
        offset_row, self.offset = _freq_spin(-rate / 2, rate / 2, spec.get("offset", rate * 0.1), 2)
        form.addRow("频点（基带偏移）", offset_row)
        power_row, self.power = _plain_spin(-200.0, 0.0, spec.get("power_dbfs", -10.0), 1, "dBFS")
        form.addRow("功率", power_row)
        bw_row, self.bandwidth = _freq_spin(1.0, rate, spec.get("bandwidth", rate / 10.0), 2)
        if mode in ("fh_rc", "fh_video"):
            form.addRow(QtWidgets.QLabel("整体频带范围（含最外侧信道边缘）"), bw_row)
        else:
            form.addRow("目标带宽", bw_row)
        if mode == "am":
            self.depth = _plain_spin(0.01, 1.0, spec.get("depth", 0.8), 2)
            form.addRow("调制深度", self.depth)
        elif mode == "fm":
            msg_auto, msg_row, self.message_bandwidth = self._auto_spin(
                "消息带宽自动（=带宽/4）", spec, "message_bandwidth",
                1.0, rate, lambda: self.bandwidth.value() / 4.0)
            form.addRow(msg_auto, msg_row)
            dev_auto, dev_row, self.deviation = self._auto_spin(
                "频偏自动（RMS，标定使占用带宽≈目标带宽）", spec, "deviation",
                1.0, rate, lambda: 0.19 * self.bandwidth.value())
            form.addRow(dev_auto, dev_row)
        elif mode == "ssb":
            self.side = QtWidgets.QComboBox()
            self.side.addItem("上边带 USB", "usb")
            self.side.addItem("下边带 LSB", "lsb")
            self.side.setCurrentIndex(0 if spec.get("side", "usb") == "usb" else 1)
            form.addRow("边带", self.side)
        elif mode == "ask2":
            self.pulse = QtWidgets.QComboBox()
            self.pulse.addItem("RRC 成形", "rrc")
            self.pulse.addItem("矩形脉冲", "rect")
            self.pulse.setCurrentIndex(0 if spec.get("pulse", "rrc") == "rrc" else 1)
            form.addRow("脉冲成形", self.pulse)
            self.alpha = _plain_spin(0.05, 1.0, spec.get("alpha", 0.35), 2)
            form.addRow("滚降系数 α", self.alpha)
        elif mode in ("qpsk", "qam16", "qam64"):
            self.alpha = _plain_spin(0.05, 1.0, spec.get("alpha", 0.35), 2)
            form.addRow("滚降系数 α", self.alpha)
        elif mode in ("fh_rc", "fh_video"):
            hop_row, self.hop_rate = _plain_spin(
                0.001, rate, spec.get("hop_rate", 100.0 if mode == "fh_rc" else 50.0), 1, "hop/s")
            form.addRow("跳速", hop_row)
            self.hop_auto = QtWidgets.QCheckBox("频点集合自动均布（两端频点由中心跨度推导）")
            self.hop_auto.setChecked(spec.get("hop_points") is None)
            self.hop_count = QtWidgets.QSpinBox()
            self.hop_count.setRange(2, 64)
            self.hop_count.setValue(int(spec.get("hop_count", 8)))
            self.hop_count.setEnabled(self.hop_auto.isChecked())
            form.addRow(self.hop_auto, self.hop_count)
            self.hop_points_edit = QtWidgets.QLineEdit()
            self.hop_points_edit.setPlaceholderText("如：50000,120000,190000")
            if spec.get("hop_points") is not None:
                self.hop_points_edit.setText(", ".join(str(point) for point in spec["hop_points"]))
            self.hop_points_edit.setEnabled(not self.hop_auto.isChecked())
            form.addRow("频点列表（Hz）", self.hop_points_edit)
            span_auto, span_row, self.hop_span = self._auto_spin(
                "跳频中心跨度自动（=整体频带范围−单跳带宽）", spec, "hop_span",
                1.0, rate, self._hop_span_default)
            form.addRow(span_auto, span_row)
            hopbw_auto, hopbw_row, self.hop_bandwidth = self._auto_spin(
                "单跳带宽自动（=中心跨度/频点数）", spec, "hop_bandwidth",
                1.0, rate, self._hop_bandwidth_default)
            form.addRow(hopbw_auto, hopbw_row)
            self.hop_auto.toggled.connect(self.hop_count.setEnabled)
            self.hop_auto.toggled.connect(lambda checked: self.hop_points_edit.setEnabled(not checked))
            self.hop_auto.toggled.connect(self.refresh_auto)
            self.hop_rate.valueChanged.connect(self.refresh_auto)
            self.hop_count.valueChanged.connect(self.refresh_auto)
            self.hop_points_edit.textChanged.connect(self.refresh_auto)
            if mode == "fh_rc":
                dev_auto, dev_row, self.deviation = self._auto_spin(
                    "每跳 2FSK 频偏自动（=每跳带宽/4）", spec, "deviation",
                    1.0, rate, lambda: self._hop_bandwidth_now() / 4.0)
                form.addRow(dev_auto, dev_row)
                rs_auto, rs_row, self.symbol_rate = self._auto_spin(
                    "符号速率自动（=每跳带宽/2）", spec, "symbol_rate",
                    1.0, rate, lambda: self._hop_bandwidth_now() / 2.0)
                form.addRow(rs_auto, rs_row)
            else:
                self.subcarriers = QtWidgets.QSpinBox()
                self.subcarriers.setRange(8, 1024)
                self.subcarriers.setValue(int(spec.get("subcarriers", 64)))
                form.addRow("OFDM 子载波数", self.subcarriers)
                self.cp_ratio = _plain_spin(0.01, 0.5, spec.get("cp_ratio", 0.25), 2)
                form.addRow("循环前缀比例", self.cp_ratio)
        self.auto_label = QtWidgets.QLabel()
        self.auto_label.setWordWrap(True)
        self.auto_label.setStyleSheet("background:white;border:1px solid #d8e1ec;border-radius:5px;padding:8px;")
        form.addRow("自动参数", self.auto_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        for widget in (self.offset, self.power, self.bandwidth):
            widget.valueChanged.connect(self.refresh_auto)
        self.refresh_auto()

    def _auto_spin(self, auto_text, spec, key, minimum, maximum, default_fn):
        auto = QtWidgets.QCheckBox(auto_text)
        auto.setChecked(key not in spec)
        spin = UnitSpinBox(minimum, maximum, 0.0, 2)
        if key in spec:
            spin.setValue(spec[key])
        else:
            try:
                spin.setValue(default_fn())
            except ValueError:
                spin.setValue(minimum)
        spin.setEnabled(key in spec)
        auto.toggled.connect(lambda checked, spin=spin, default_fn=default_fn: self._toggle_auto(spin, default_fn, checked))
        spin.valueChanged.connect(lambda *_: self.refresh_auto())
        return auto, _unit_row(spin), spin

    def _toggle_auto(self, spin, default_fn, checked):
        if not checked:
            try:
                spin.setValue(default_fn())
            except ValueError:
                spin.setValue(spin.minimum())
        spin.setEnabled(not checked)
        self.refresh_auto()

    def _hop_points_now(self):
        if self.hop_auto.isChecked():
            return None
        text = self.hop_points_edit.text()
        points = [float(part) for part in text.replace("，", ",").split(",") if part.strip()]
        if not points:
            raise ValueError("请填写频点列表（Hz，逗号分隔）")
        return points

    def _planned_fh(self):
        """当前对话框状态（合并原始 spec 中的手动键）经 plan_signal 推导的 FH 规划。"""
        merged = dict(self._source_spec)
        merged.update(self.spec())
        return plan_signal(merged, self.rate)

    def _hop_span_default(self):
        return self._planned_fh()["hop_span"]

    def _hop_bandwidth_default(self):
        return self._planned_fh()["hop_bandwidth"]

    def _hop_bandwidth_now(self):
        try:
            return self._planned_fh()["hop_bandwidth"]
        except ValueError:
            points = self._hop_points_now()
            count = self.hop_count.value() if points is None else len(points)
            return self.bandwidth.value() / max(2, count)

    def _add_fh_keys(self, common):
        span_spin = getattr(self, "hop_span", None)
        if span_spin is not None and span_spin.isEnabled():
            common["hop_span"] = span_spin.value()
        bw_spin = getattr(self, "hop_bandwidth", None)
        if bw_spin is not None and bw_spin.isEnabled():
            common["hop_bandwidth"] = bw_spin.value()

    def spec(self):
        common = {"mode": self.mode, "offset": self.offset.value(),
                  "power_dbfs": self.power.value(), "bandwidth": self.bandwidth.value()}
        if self.mode == "am":
            common["depth"] = self.depth.value()
        elif self.mode == "fm":
            if self.message_bandwidth.isEnabled():
                common["message_bandwidth"] = self.message_bandwidth.value()
            if self.deviation.isEnabled():
                common["deviation"] = self.deviation.value()
        elif self.mode == "ssb":
            common["side"] = self.side.currentData()
        elif self.mode == "ask2":
            common["pulse"] = self.pulse.currentData()
            if common["pulse"] == "rrc":
                common["alpha"] = self.alpha.value()
        elif self.mode in ("qpsk", "qam16", "qam64"):
            common["alpha"] = self.alpha.value()
        elif self.mode == "fh_rc":
            common["hop_rate"] = self.hop_rate.value()
            points = self._hop_points_now()
            if points is None:
                common["hop_count"] = self.hop_count.value()
            else:
                common["hop_points"] = points
            dev = getattr(self, "deviation", None)
            if dev is not None and dev.isEnabled():
                common["deviation"] = dev.value()
            rs = getattr(self, "symbol_rate", None)
            if rs is not None and rs.isEnabled():
                common["symbol_rate"] = rs.value()
            self._add_fh_keys(common)
        elif self.mode == "fh_video":
            common["hop_rate"] = self.hop_rate.value()
            points = self._hop_points_now()
            if points is None:
                common["hop_count"] = self.hop_count.value()
            else:
                common["hop_points"] = points
            if hasattr(self, "subcarriers"):
                common["subcarriers"] = self.subcarriers.value()
                common["cp_ratio"] = self.cp_ratio.value()
            self._add_fh_keys(common)
        return common

    def refresh_auto(self, *_):
        try:
            plan = plan_signal(self.spec(), self.rate)
        except ValueError as exc:
            self.auto_label.setStyleSheet(
                "background:#fdecea;border:1px solid #e0a4a1;border-radius:5px;padding:8px;color:#a03a35;")
            self.auto_label.setText(f"参数无效：{exc}")
            return
        self.auto_label.setStyleSheet(
            "background:white;border:1px solid #d8e1ec;border-radius:5px;padding:8px;")
        mode = plan["mode"]
        if mode == "am":
            text = (f"消息带宽 {_fmt_hz(plan['message_bandwidth'])} → "
                    f"实际带宽 {_fmt_hz(plan['bandwidth_actual'])}")
        elif mode == "fm":
            text = (f"消息带宽 {_fmt_hz(plan['message_bandwidth'])}，RMS 频偏 {_fmt_hz(plan['deviation'])} → "
                    f"占用带宽 ≈ {_fmt_hz(plan['bandwidth_actual'])}")
        elif mode == "ssb":
            text = f"{plan['side'].upper()}，实际带宽 {_fmt_hz(plan['bandwidth_actual'])}"
        elif mode in ("ask2", "qpsk", "qam16", "qam64"):
            text = (f"{plan['pulse'].upper()}，符号速率 {_fmt_hz(plan['symbol_rate'])}，"
                    f"每符号 {plan['sps']} 个采样 → 实际带宽 {_fmt_hz(plan['bandwidth_actual'])}")
        elif mode == "fh_rc":
            text = (f"{len(plan['hop_points'])} 个频点，中心跨度 {_fmt_hz(plan['hop_span'])}，"
                    f"单跳带宽 {_fmt_hz(plan['hop_bandwidth'])} → 整体频带范围 {_fmt_hz(plan['bandwidth_actual'])}；"
                    f"2FSK 频偏 {_fmt_hz(plan['deviation'])}，符号速率 {_fmt_hz(plan['symbol_rate'])}")
        else:
            text = (f"{len(plan['hop_points'])} 个频点，中心跨度 {_fmt_hz(plan['hop_span'])}，"
                    f"单跳带宽 {_fmt_hz(plan['hop_bandwidth'])} → 整体频带范围 {_fmt_hz(plan['bandwidth_actual'])}；"
                    f"{plan['subcarriers']} 个子载波（QPSK），间隔 {_fmt_hz(plan['subcarrier_spacing'])}，"
                    f"FFT {plan['fft_size']} 点 + CP {plan['cp_samples']} 点")
        self.auto_label.setText(f"自动推导：{text}")

    def accept(self):
        try:
            spec = self.spec()
            plan_signal(spec, self.rate)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数无效", str(exc))
            return
        self._last_spec = spec
        super().accept()

    def result(self):
        return dict(self._last_spec or {})


class MainWindow(DesktopWindow):
    page_title = "数据分析"
    run_task = staticmethod(_run_task)

    def __init__(self, workspace):
        self.asset_limit = 100
        super().__init__(Workspace(workspace), "电磁信号分析 · SignalAnalysis",
                         "离线数据 · 通用统计与时频展示 · IQ 信号生成 · 信号检测与调制识别 · "
                         "算法对比与离线报告 · 原生插件")
        self.tabs.insertTab(1, self.build_generator(), "IQ 信号生成")
        self.tabs.insertTab(2, self.build_detect(), "信号检测")
        self.tabs.insertTab(3, self.build_amc(), "调制识别")
        self.tabs.insertTab(4, self.build_compare(), "算法对比")
        self.last_result = None
        self._play_data = None
        self._play_rate = 1.0
        self._play_pos = 0
        self._play_last = 0.0
        self._play_paused = False
        self._play_buffer = None
        self._play_times = []
        self._play_real = False
        self._play_level_hi = None
        self._range_syncing = False
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._on_playback_tick)
        self.refresh_assets()

    def build_page(self):
        return self.build_analysis()

    def job_buttons(self):
        return (self.demo_button, self.import_button, self.analyze_button, self.native_button,
                self.generate_button, self.detect_button, self.ml_button, self.amc_button)

    def result_ready(self, result):
        self.refresh_assets()
        if result.get("kind") == "generate":
            self.last_result = result
            self.show_generation_result(result)
            for row in range(self.assets.count()):
                item = self.assets.item(row)
                if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == result["id"]:
                    self.assets.setCurrentItem(item)
                    break
        elif "kind" in result:
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
        layout.addWidget(QtWidgets.QLabel("导入 / 演示采样率"))
        rate_row, self.sample_rate = _freq_spin(1, 1e9, 48000, 2)
        layout.addWidget(rate_row)
        self.import_button = QtWidgets.QPushButton("导入 IQ / SigMF")
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


    def build_generator(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel("生成用于测试检测、参数估计与调制识别算法的 IQ 基带信号。"
                                 "IQ 为复基带记录，不设置载频：\"频点\"指基带频率偏移；"
                                 "频率值以 Hz / kHz / MHz 显示（单位在输入框外）。"
                                 "可一次包含多种信号并独立设置参数，自动按目标带宽推导调制参数。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_global_row())
        layout.addWidget(self._build_noise_group())
        layout.addWidget(self._build_signal_table(), 1)
        layout.addWidget(self._build_export_row())
        self.gen_result = QtWidgets.QLabel("尚未生成")
        self.gen_result.setWordWrap(True)
        self.gen_result.setStyleSheet(
            "background:white;border:1px solid #d8e1ec;border-radius:5px;padding:8px;")
        layout.addWidget(self.gen_result)
        self.iq_signals = []
        self._last_suggested_name = ""
        self.update_gen_controls()
        return box


    def _build_global_row(self):
        group = QtWidgets.QGroupBox("全局参数")
        row = QtWidgets.QHBoxLayout(group)
        rate_label = QtWidgets.QLabel("采样率")
        rate_row, self.gen_rate = _freq_spin(1.0, 1e9, 1_000_000.0, 2)
        self.gen_rate.setToolTip("所有信号的统一采样率")
        duration_label = QtWidgets.QLabel("持续时间")
        duration_row, self.gen_duration = _plain_spin(0.000001, 3600.0, 0.1, 6, "s")
        self.gen_count_label = QtWidgets.QLabel()
        seed_label = QtWidgets.QLabel("随机种子")
        self.gen_seed = _plain_spin(0.0, 2 ** 32 - 1, 0.0, 0)
        self.gen_seed.setToolTip("相同种子产生完全一致的信号")
        for widget in (rate_label, rate_row, duration_label, duration_row,
                       self.gen_count_label, seed_label, self.gen_seed):
            row.addWidget(widget)
        row.addStretch()
        self.gen_rate.valueChanged.connect(self.update_gen_count)
        self.gen_duration.valueChanged.connect(self.update_gen_count)
        return group


    def _build_noise_group(self):
        group = QtWidgets.QGroupBox("背景噪声")
        row = QtWidgets.QHBoxLayout(group)
        self.gen_noise_enabled = QtWidgets.QCheckBox("启用")
        self.gen_noise_enabled.setChecked(True)
        self.gen_noise_enabled.toggled.connect(self.update_gen_controls)
        row.addWidget(self.gen_noise_enabled)
        row.addWidget(QtWidgets.QLabel("噪声带宽"))
        noise_bw_row, self.gen_noise_bw = _freq_spin(1.0, 1e9, 1_000_000.0, 2)
        self.gen_noise_bw.setToolTip("双侧带限带宽；等于采样率时为全带白噪声")
        row.addWidget(noise_bw_row)
        row.addWidget(QtWidgets.QLabel("带内 SNR（相对最强信号）"))
        snr_row, self.gen_snr = _plain_spin(-10.0, 80.0, 20.0, 1, "dB")
        self.gen_snr.setToolTip("信号平均功率 ÷ 同占用带宽内的噪声功率；噪声按功率谱密度折算，\n"
                               "噪声带宽须覆盖最强信号的占用频带")
        row.addWidget(snr_row)
        row.addWidget(QtWidgets.QLabel("噪声功率（无信号时）"))
        noise_power_row, self.gen_noise_power = _plain_spin(-200.0, 0.0, -20.0, 1, "dBFS")
        self.gen_noise_power.setEnabled(False)
        row.addWidget(noise_power_row)
        row.addStretch()
        return group


    def _build_signal_table(self):
        group = QtWidgets.QGroupBox("信号列表（最多 16 个，各信号独立参数）")
        layout = QtWidgets.QVBoxLayout(group)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("添加样式"))
        self.gen_mode_combo = QtWidgets.QComboBox()
        for key, label in MODE_CHOICES:
            self.gen_mode_combo.addItem(label, key)
        bar.addWidget(self.gen_mode_combo)
        add = QtWidgets.QPushButton("添加信号…")
        add.clicked.connect(self.add_signal_clicked)
        bar.addWidget(add)
        edit = QtWidgets.QPushButton("编辑参数…")
        edit.clicked.connect(self.edit_signal_clicked)
        bar.addWidget(edit)
        remove = QtWidgets.QPushButton("删除选中")
        remove.clicked.connect(self.remove_signal_clicked)
        bar.addWidget(remove)
        bar.addStretch()
        layout.addLayout(bar)
        self.gen_signals = QtWidgets.QTableWidget(0, 5)
        self.gen_signals.setHorizontalHeaderLabels(["调制样式", "频点", "功率", "带宽", "参数摘要"])
        self.gen_signals.horizontalHeader().setStretchLastSection(True)
        self.gen_signals.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.gen_signals.horizontalHeader().setSectionResizeMode(
            4, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.gen_signals.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.gen_signals.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.gen_signals.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.gen_signals, 1)
        return group


    def _build_export_row(self):
        group = QtWidgets.QGroupBox("资产与导出")
        row = QtWidgets.QHBoxLayout(group)
        row.addWidget(QtWidgets.QLabel("资产名称"))
        self.gen_name = QtWidgets.QLineEdit()
        self.gen_name.setPlaceholderText("留空自动命名，如：IQ 生成 · QPSK + AM")
        row.addWidget(self.gen_name, 1)
        row.addWidget(QtWidgets.QLabel("导出格式"))
        self.gen_export_format = QtWidgets.QComboBox()
        for label, fmt in EXPORT_FORMATS:
            self.gen_export_format.addItem(label, fmt)
        self.gen_export_format.currentIndexChanged.connect(self.update_gen_controls)
        row.addWidget(self.gen_export_format)
        row.addWidget(QtWidgets.QLabel("字节序"))
        self.gen_endian = QtWidgets.QComboBox()
        self.gen_endian.addItem("小端", "little")
        self.gen_endian.addItem("大端", "big")
        self.gen_endian.setEnabled(False)
        row.addWidget(self.gen_endian)
        self.generate_button = QtWidgets.QPushButton("生成 IQ 信号")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate_iq_clicked)
        row.addWidget(self.generate_button)
        return group


    def update_gen_controls(self, *_):
        has_signals = bool(self.iq_signals)
        self.gen_snr.setEnabled(self.gen_noise_enabled.isChecked() and has_signals)
        self.gen_noise_power.setEnabled(self.gen_noise_enabled.isChecked() and not has_signals)
        fmt = self.gen_export_format.currentData()
        self.gen_endian.setEnabled(fmt in ("iq16", "iq32"))
        self._suggest_name()
        self.update_gen_count()


    def update_gen_count(self, *_):
        rate = self.gen_rate.value()
        count = int(round(rate * self.gen_duration.value()))
        valid = 1 <= count <= MAX_SAMPLES
        color = "#23374d" if valid else "#c0392b"
        self.gen_count_label.setText(f"预计 {count:,} 个复采样（上限 {MAX_SAMPLES:,}）")
        self.gen_count_label.setStyleSheet(f"color:{color};")
        if self.iq_signals:
            for row in range(self.gen_signals.rowCount()):
                item = self.gen_signals.item(row, 4)
                if item is not None:
                    item.setText(self._describe_signal(self.iq_signals[row]))


    def _suggest_name(self):
        if not self.iq_signals:
            return
        styles = " + ".join(sorted(MODE_SHORT[signal["mode"]] for signal in self.iq_signals))
        suggestion = f"IQ 生成 · {styles}"
        if not self.gen_name.text() or self.gen_name.text() == self._last_suggested_name:
            self.gen_name.setText(suggestion)
        self._last_suggested_name = suggestion


    def _describe_signal(self, spec):
        try:
            plan = plan_signal(spec, self.gen_rate.value())
        except ValueError as exc:
            return f"参数无效：{exc}"
        if plan["mode"] == "am":
            return (f"深度 {plan['depth']:g} · 消息带宽 {_fmt_hz(plan['message_bandwidth'])} · "
                    f"实际带宽 {_fmt_hz(plan['bandwidth_actual'])}")
        if plan["mode"] == "fm":
            return (f"消息带宽 {_fmt_hz(plan['message_bandwidth'])} · RMS 频偏 {_fmt_hz(plan['deviation'])} · "
                    f"占用带宽 ≈ {_fmt_hz(plan['bandwidth_actual'])}")
        if plan["mode"] == "ssb":
            return f"{plan['side'].upper()} · 实际带宽 {_fmt_hz(plan['bandwidth_actual'])}"
        if plan["mode"] in ("ask2", "qpsk", "qam16", "qam64"):
            return (f"{plan['pulse'].upper()} α={plan['alpha']:g} · 符号速率 {_fmt_hz(plan['symbol_rate'])} · "
                    f"实际带宽 {_fmt_hz(plan['bandwidth_actual'])}")
        if plan["mode"] == "fh_rc":
            return (f"跳速 {plan['hop_rate']:g} hop/s · {len(plan['hop_points'])} 频点 · "
                    f"中心跨度 {_fmt_hz(plan['hop_span'])} · 单跳带宽 {_fmt_hz(plan['hop_bandwidth'])} · "
                    f"2FSK 频偏 {_fmt_hz(plan['deviation'])}")
        return (f"跳速 {plan['hop_rate']:g} hop/s · {len(plan['hop_points'])} 频点 · "
                f"中心跨度 {_fmt_hz(plan['hop_span'])} · 单跳带宽 {_fmt_hz(plan['hop_bandwidth'])} · "
                f"{plan['subcarriers']} 子载波")


    def add_signal_clicked(self):
        mode = self.gen_mode_combo.currentData()
        dialog = SignalParamsDialog(mode, self.gen_rate.value(), parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.add_iq_signal(dialog.result())


    def edit_signal_clicked(self):
        row = self.gen_signals.currentRow()
        if row < 0:
            self.status.setText("请先在信号表中选择一行")
            return
        spec = self.iq_signals[row]
        dialog = SignalParamsDialog(spec["mode"], self.gen_rate.value(), spec, parent=self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.iq_signals[row] = dialog.result()
            self._fill_signal_row(row, self.iq_signals[row])
            self.update_gen_controls()


    def remove_signal_clicked(self):
        row = self.gen_signals.currentRow()
        if row < 0:
            self.status.setText("请先在信号表中选择一行")
            return
        self.gen_signals.removeRow(row)
        del self.iq_signals[row]
        self.update_gen_controls()


    def add_iq_signal(self, spec):
        if len(self.iq_signals) >= 16:
            self.status.setText("一次最多生成 16 个信号")
            return
        self.iq_signals.append(dict(spec))
        row = self.gen_signals.rowCount()
        self.gen_signals.insertRow(row)
        self._fill_signal_row(row, spec)
        self.update_gen_controls()


    def _fill_signal_row(self, row, spec):
        mode_item = QtWidgets.QTableWidgetItem(MODE_SHORT.get(spec["mode"], spec["mode"]))
        mode_item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(spec))
        self.gen_signals.setItem(row, 0, mode_item)
        self.gen_signals.setItem(row, 1, QtWidgets.QTableWidgetItem(_fmt_hz(spec.get("offset", 0))))
        self.gen_signals.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{spec.get('power_dbfs', -10):g} dBFS"))
        self.gen_signals.setItem(row, 3, QtWidgets.QTableWidgetItem(_fmt_hz(spec.get("bandwidth", 0))))
        self.gen_signals.setItem(row, 4, QtWidgets.QTableWidgetItem(self._describe_signal(spec)))


    def generate_iq_clicked(self):
        rate = self.gen_rate.value()
        duration = self.gen_duration.value()
        count = int(round(rate * duration))
        if not 1 <= count <= MAX_SAMPLES:
            self.status.setText(f"采样点数 {count:,} 超出 1～{MAX_SAMPLES:,} 范围，请调整采样率或持续时间")
            return
        noise = None
        if self.gen_noise_enabled.isChecked():
            noise = {"enabled": True, "bandwidth": self.gen_noise_bw.value()}
            if self.iq_signals:
                noise["snr_db"] = self.gen_snr.value()
            else:
                noise["power_dbfs"] = self.gen_noise_power.value()
        export = None
        fmt = self.gen_export_format.currentData()
        if fmt:
            export = {"format": fmt, "endian": self.gen_endian.currentData()}
        self.start_job("generate", sample_rate=rate, duration=duration,
                       seed=int(self.gen_seed.value()), signals=self.iq_signals,
                       noise=noise, name=self.gen_name.text().strip() or None, export=export)


    def show_generation_result(self, result):
        summary = result["summary"]
        lines = [f"已生成资产：{result['name']}",
                 f"采样率 {_fmt_hz(summary['sample_rate_hz'])} · {summary['sample_count']:,} 个复采样 · "
                 f"时长 {summary['duration_s']:g} s · 峰值 {summary['peak_dbfs']:.1f} dBFS"]
        for entry in summary["signals"]:
            style = MODE_SHORT.get(entry["mode"], entry["mode"])
            line = (f"  {style}：频点 {_fmt_hz(entry['offset'])} · 目标功率 {entry['power_dbfs']:g} dBFS"
                    f"（实测 {entry['power_dbfs_actual']:.2f} dBFS）· 目标带宽 {_fmt_hz(entry['bandwidth'])}")
            if entry.get("snr_inband_db") is not None:
                line += f" · 带内 SNR {entry['snr_inband_db']:.2f} dB"
            lines.append(line)
        noise = summary["noise"]
        psd = noise.get("power_dbfs_per_hz")
        psd_text = "" if psd is None else f" · 功率谱密度 {psd:.1f} dB/Hz"
        if noise["enabled"]:
            if noise["snr_db"] is not None:
                reference = noise.get("snr_reference_index")
                ref_text = "" if reference is None else f"（参考信号 #{reference + 1}）"
                lines.append(f"  背景噪声：带宽 {_fmt_hz(noise['bandwidth'])} · 带内 SNR {noise['snr_db']:g} dB"
                             f"{ref_text} · 总功率 {noise['power_dbfs']:.1f} dBFS{psd_text}")
            elif noise["power_dbfs"] is not None:
                lines.append(f"  背景噪声：带宽 {_fmt_hz(noise['bandwidth'])} · "
                             f"总功率 {noise['power_dbfs']:.1f} dBFS{psd_text}")
        if result["export_path"]:
            lines.append(f"导出文件：{result['export_path']}（格式 {result['export_format']}）")
            if result.get("export_data_path"):
                lines.append(f"IQ 数据文件：{result['export_data_path']}")
        self.gen_result.setText("\n".join(lines))


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
        if self._play_data is not None:
            self._stop_playback()
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
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择离线数据", "",
                                                        "数据 (*.npy *.csv *.bin *.raw *.iq *.sigmf-meta *.sigmf-data)")
        if not path:
            return
        request = {"sample_rate": self.sample_rate.value()}
        if Path(path).suffix.lower() in (".sigmf-meta", ".sigmf-data"):
            request = {}  # The recording owns its sample rate, not the manual input.
        if Path(path).suffix.lower() in (".bin", ".raw", ".iq"):
            dialog = BinaryImportDialog(self)
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            request["binary_dtype"], request["endian"] = dialog.values()
        self.start_job("import", path=path, **request)

    def analyze_selected(self):
        if self.analyze_mode.currentText() == "实时播放":
            self._start_playback()
            return
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
        self.detect_button = QtWidgets.QPushButton("检测所选数据")
        self.detect_button.setObjectName("primary")
        self.detect_button.clicked.connect(self.detect_selected)
        bar.addWidget(self.detect_button)
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


    def build_amc(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "A09 六类调制识别（AMC）：FM、SSB、2ASK、QPSK、16QAM、64QAM。在指定分析频带内提取确定性"
            "NumPy 特征（包络统计、谱平坦度、瞬时频率与相位统计、高阶累积量、峰值幅度直方图模板、"
            "带内信噪比粗估），再由线性判别模型给出六类概率。默认使用随包分发的合成数据基线模型，"
            "也可以选择自训练模型 JSON 或 ONNX 分类器清单。识别准确率的合格门限尚未确认；低信噪比"
            "（<5 dB）或最高类概率偏低时会标注“仅供参考”，无法映射到六类的样式（如 AM、跳频会话）"
            "按“不适用”计数，不丢弃样本。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("分析中心"))
        center_row, self.amc_offset = _freq_spin(-1e9, 1e9, 0.0, 1)
        self.amc_offset.setToolTip("信号占用的中心频率（相对基带的频率偏移），0 表示基带中心")
        bar.addWidget(center_row)
        bar.addWidget(QtWidgets.QLabel("分析带宽"))
        width_row, self.amc_bandwidth = _freq_spin(0.0, 1e9, 0.0, 1)
        self.amc_bandwidth.setToolTip("信号占用带宽（Hz），0 表示使用整段采样带宽（不做抽取）")
        bar.addWidget(width_row)
        self.amc_button = QtWidgets.QPushButton("识别所选数据")
        self.amc_button.setObjectName("primary")
        self.amc_button.clicked.connect(self.amc_classify_selected)
        bar.addWidget(self.amc_button)
        self.amc_from_detect = QtWidgets.QPushButton("取用检测结果频带")
        self.amc_from_detect.setToolTip(
            "把最近一次检测结果中功率最大的目标中心频率与带宽填进左侧输入框")
        self.amc_from_detect.clicked.connect(self.use_detected_band)
        bar.addWidget(self.amc_from_detect)
        bar.addStretch(1)
        layout.addLayout(bar)
        model_bar = QtWidgets.QHBoxLayout()
        model_bar.addWidget(QtWidgets.QLabel("识别模型"))
        self.amc_model = QtWidgets.QLineEdit()
        self.amc_model.setPlaceholderText("留空使用内置线性基线；也可选择自训练模型 JSON 或 ONNX 清单")
        self.amc_model.setToolTip("模型 JSON 为 amc_model_v1；ONNX 清单为 amc-manifest 生成的 JSON")
        model_bar.addWidget(self.amc_model, 1)
        self.amc_choose = QtWidgets.QPushButton("选择…")
        self.amc_choose.clicked.connect(self.choose_amc_model)
        model_bar.addWidget(self.amc_choose)
        self.amc_model_status = QtWidgets.QLabel()
        model_bar.addWidget(self.amc_model_status)
        model_bar.addStretch(1)
        layout.addLayout(model_bar)
        grid = QtWidgets.QGridLayout()
        self.amc_plot = pg.PlotWidget(title="六类后验概率")
        self.amc_plot.setLabel("left", "概率")
        self.amc_bars = None
        grid.addWidget(self.amc_plot, 0, 0)
        self.amc_table = QtWidgets.QTableWidget(0, 2)
        self.amc_table.setHorizontalHeaderLabels(["特征", "取值"])
        self.amc_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.amc_table.horizontalHeader().setStretchLastSection(True)
        self.amc_table.setMaximumWidth(320)
        grid.addWidget(self.amc_table, 0, 1)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid, 1)
        self.amc_summary = QtWidgets.QPlainTextEdit()
        self.amc_summary.setReadOnly(True)
        self.amc_summary.setMaximumHeight(215)
        self.amc_summary.setPlaceholderText("识别后显示模型来源、六类概率、可信度提示与真值对照。")
        layout.addWidget(self.amc_summary)
        self.update_amc_controls()
        return box


    def update_amc_controls(self):
        """显示内置模型是否随包分发（缺失时给出自训练指引，不影响手动选模型）。"""
        from .ml import default_model_path

        present = default_model_path().is_file()
        self.amc_model_status.setText(
            "内置线性基线可用" if present else
            "未找到内置模型，请先运行 training/train_amc.py，或手动选择模型文件")


    def build_compare(self):
        """算法对比页：同一份数据上传统基线与 AI 路径的指标并排。"""
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "把同一个数据资产上的两条路径排在一起：信号检测侧的“检测结果 / 传统基线”取自 AI 检测运行时"
            "同步跑的能量检测，两者使用同一套真值、同一套会话合并口径；调制识别侧列出模型输出、"
            "生成器真值与命中情况。没有生成器真值（导入或原生插件产出）时只列结果、不计算指标，"
            "按“不适用”计数而不是静默丢弃。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        bar = QtWidgets.QHBoxLayout()
        self.compare_refresh = QtWidgets.QPushButton("刷新对比")
        self.compare_refresh.setToolTip("使用最近的检测与调制识别结果重新填表")
        self.compare_refresh.clicked.connect(self.compare_from_detect)
        bar.addWidget(self.compare_refresh)
        bar.addStretch(1)
        layout.addLayout(bar)
        self.compare_table = QtWidgets.QTableWidget(0, 4)
        self.compare_table.setHorizontalHeaderLabels(["环节", "对象", "指标", "取值"])
        self.compare_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.compare_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.compare_table, 3)
        self.compare_summary = QtWidgets.QPlainTextEdit()
        self.compare_summary.setReadOnly(True)
        self.compare_summary.setPlaceholderText(
            "运行“信号检测”（或 AI 检测）与“调制识别”后，这里显示并排对比与未确认项。")
        layout.addWidget(self.compare_summary, 2)
        return box


    def compare_from_detect(self):
        self._render_compare(switch=True)


    def _object_label(self, prefix, result):
        """表格里的对象名：有模型就带模型标识，否则用路径名。"""
        model = (result or {}).get("model") or {}
        if not model.get("id"):
            return prefix
        return f"{prefix} · {model['id']}@{model.get('version', '--')}"


    def _render_compare(self, switch=False):
        """把最近一次检测（AI/传统）与调制识别结果整理成并排表格。"""
        detect = self.tab_results.get(2)
        amc = self.tab_results.get(3)
        rows = []
        lines = []
        if detect:
            summary = detect.get("summary") or {}
            metrics = detect.get("metrics")
            baseline = detect.get("baseline_metrics")
            main_label = self._object_label(
                "AI 检测" if summary.get("model") else "能量检测", summary)
            lines = [
                f"检测数据：{detect.get('asset_name', detect.get('asset_id'))}  |  "
                f"算法 {detect.get('algorithm')}（契约 {detect.get('contract')}）",
                f"检出目标 {len(summary.get('detections') or [])} 个 · "
                f"门限 {_fmt_metric(summary.get('threshold_dbfs_per_hz'), '.1f')} dB/Hz",
            ]
            if metrics:
                for name, value in detection_metrics(metrics):
                    rows.append(("信号检测", main_label, name, value))
                if baseline:
                    for name, value in detection_metrics(baseline):
                        rows.append(("信号检测", "传统基线（能量检测）", name, value))
                    lines.append(_comparison_line(main_label, metrics,
                                                  "传统基线", baseline))
                else:
                    lines.append("本次检测没有同步运行传统基线，无法并排展示；"
                                 "AI 检测页勾选“同时跑能量检测”即可对照。")
            else:
                lines.append("该数据没有生成器真值，只列检测结果，不计算指标（不适用）。")
        if amc:
            prediction = amc.get("prediction") or {}
            model = amc.get("model") or {}
            truth = amc.get("truth") or {}
            label = self._object_label("调制识别", {"model": model})
            for name, value in amc_metrics(prediction):
                rows.append(("调制识别", label, name, value))
            if truth.get("available"):
                hit = amc.get("truth_hit")
                rows.append(("调制识别", "生成器真值", "真值类别 / 样式",
                             f"{truth.get('class')} / {truth.get('mode')}"))
                rows.append(("调制识别", "生成器真值", "识别命中",
                             "命中" if hit else "未命中"))
            else:
                rows.append(("调制识别", "生成器真值", "对照", truth.get("reason") or "不适用"))
            lines.append(f"识别模型来源 {model.get('source', '--')}（{model.get('id', '--')}）· "
                         f"带内信噪比粗估 {_fmt_metric(amc.get('snr_estimate_db'), '.2f')} dB")
            for item in amc.get("pending") or []:
                lines.append(f"待确认项：{item}")
        self.compare_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                self.compare_table.setItem(row, column, QtWidgets.QTableWidgetItem(str(text)))
        self.compare_table.resizeColumnsToContents()
        if not lines:
            lines = ["还没有可对比的结果：请先在“信号检测”或“调制识别”标签页运行一次。"]
        self.compare_summary.setPlainText("\n".join(lines))
        if switch:
            self.tabs.setCurrentIndex(4)
            self.status.setText(f"算法对比已刷新（{len(rows)} 行指标）")


    def choose_amc_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择调制识别模型", str(self.workspace.root), "模型文件 (*.json *.onnx)")
        if path:
            self.amc_model.setText(path)


    def use_detected_band(self):
        """把检测结果里最强目标的中心频率与带宽填进识别输入框（可手动再改）。"""
        result = self.tab_results.get(2) or {}
        detections = (result.get("summary") or {}).get("detections") or []
        if not detections:
            self.status.setText("还没有检测结果：请先在“信号检测”标签页运行一次检测")
            return
        target = max(detections, key=lambda item: float(item.get("power_dbfs", -1e9)))
        self.amc_offset.setValue(float(target["center_hz"]))
        self.amc_bandwidth.setValue(float(target["bandwidth_hz"]))
        self.status.setText(
            f"已取用检测结果 #{target['id']} 的频带（中心 {_fmt_hz(target['center_hz'])}、"
            f"带宽 {_fmt_hz(target['bandwidth_hz'])}）")


    def amc_classify_selected(self):
        asset = self.selected_asset()
        if not asset:
            self.status.setText("请先导入并选择数据")
            return
        config = {}
        offset = float(self.amc_offset.value())
        if offset:
            config["offset_hz"] = offset
        bandwidth = float(self.amc_bandwidth.value())
        if bandwidth > 0:
            config["bandwidth_hz"] = bandwidth
        model = self.amc_model.text().strip() or None
        self.start_job("amc_classify", asset_id=asset["id"], config=config, model=model)


    def _render_amc(self, result):
        summary = result["summary"]
        prediction = result["prediction"]
        classes = summary["classes"]
        labels = summary["labels"]
        features = summary["features"]
        scores = prediction["scores"]
        values = [float(scores.get(name, 0.0)) for name in classes]
        positions = np.arange(len(classes), dtype=float)
        if self.amc_bars is not None:
            self.amc_plot.removeItem(self.amc_bars)
        self.amc_bars = pg.BarGraphItem(x=positions, height=values, width=0.62,
                                        brush="#2365b3", pen=pg.mkPen("#183c65"))
        self.amc_plot.addItem(self.amc_bars)
        self.amc_plot.getAxis("bottom").setTicks(
            [[(float(position), labels[name]) for position, name in zip(positions, classes)]])
        self.amc_plot.setYRange(0.0, max(1.0, max(values) * 1.15), padding=0.0)
        self.amc_plot.setTitle(f"六类后验概率 · 预测 {prediction['label_text']}"
                               f"（{prediction['confidence']:.2f}）")
        self.amc_table.setRowCount(len(features))
        for row, name in enumerate(features):
            self.amc_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.amc_table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{float(features[name]):.6g}"))
        band = summary["band"]
        model = summary["model"] or {}
        source = _AMC_SOURCE_TEXT.get(str(model.get("source")), str(model.get("source")))
        model_line = (f"模型 {model.get('id')}@{model.get('version')} · 来源 {source}"
                      + (f" · 摘要 {str(model['sha256'])[:12]}…" if model.get("sha256") else "")
                      + (f" · 训练集内准确率 {_fmt_metric(model.get('accuracy_in_sample'), '.4f')}"
                         if model.get("accuracy_in_sample") is not None else ""))
        lines = [
            f"数据：{result.get('asset_name', result['asset_id'])}  |  采样率 "
            f"{_fmt_hz(band['sample_rate_hz'])}  |  分析频带 中心 {_fmt_hz(band['center_hz'])}"
            f" · 带宽 {_fmt_hz(band['bandwidth_hz'])}  |  分析样本 {band['analysis_samples']:,}"
            f"（抽样比 {band['decimation']}）· 带内功率 {band['power_dbfs']:.2f} dBFS",
            f"算法 {summary['algorithm']}（契约 {summary['contract']}）· 特征契约 "
            f"{summary['feature_contract']}（{len(features)} 维）· 峰值样本 {band['peak_samples']} 个",
            model_line,
            f"预测：{prediction['label_text']}（概率 {prediction['confidence']:.4f} · 与次高类差值 "
            f"{_fmt_metric(prediction.get('margin'), '.4f')}）· 带内信噪比粗估 "
            f"{_fmt_metric(summary['snr_estimate_db'], '.1f')} dB",
        ]
        if prediction["reliable"]:
            lines.append("可信度：未发现低信噪比或区分度不足的提示；" + prediction["snr_note"])
        else:
            lines.append(f"可信度：仅供参考 —— {prediction['reason']}；{prediction['snr_note']}")
        baseline = summary["baseline"]
        if baseline["classification"]:
            lines.append(f"传统对照（{baseline['algorithm']}）：判定 {baseline['classification']}"
                         f" · 估计簇数 {baseline['cluster_estimate']}"
                         "（该启发式只区分数字/模拟，不对应六类）")
        else:
            lines.append(f"传统对照（{baseline['algorithm']}）：不可用 —— {baseline['note']}")
        truth = result.get("truth") or {}
        if truth.get("available"):
            lines.append(
                f"生成器真值：{truth['class']}（样式 {truth['mode']}）· 带内信噪比 "
                f"{_fmt_metric(truth['snr_inband_db'], '.2f')} dB · 识别"
                f"{'命中' if result.get('truth_hit') else '未命中'}")
        else:
            lines.append(f"生成器真值：不适用 —— {truth.get('reason', '没有真值')}；样本仍计入统计，不丢弃")
        for item in summary["pending"]:
            lines.append(f"待确认：{item}")
        self.amc_summary.setPlainText("\n".join(lines))


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
        """推理运行时缺失时禁用 AI 入口并给出安装提示（传统路径不受影响）。"""
        from .ml.runtime import runtime_version

        version = runtime_version()
        ready = version is not None
        for widget in (self.ml_button, self.ml_choose, self.ml_score, self.ml_iou,
                       self.ml_compare):
            widget.setEnabled(ready)
        self.ml_status.setText(f"onnxruntime {version}" if ready else
                               "未安装 onnxruntime：pip install '.[ml]' 后可用")


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
        self.start_job("ml_detect", asset_id=asset["id"], manifest=manifest,
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
        self.start_job("detect", asset_id=asset["id"], config=config)


    def _render_detect(self, result, arrays):
        summary = result["summary"]
        detections = summary["detections"]
        f = arrays["frequency"]
        t = arrays["frame_time"]
        matrix = arrays["spectrogram_db"]
        boxes = arrays["detection_boxes"].reshape(-1, 4)
        baseline_boxes = (arrays["baseline_detection_boxes"].reshape(-1, 4)
                          if "baseline_detection_boxes" in arrays else [])
        is_ml = bool(summary.get("model"))
        threshold_db = float(arrays["threshold_db"].ravel()[0])
        noise_db = float(arrays["noise_floor_db"].ravel()[0])
        df = float(summary["freq_resolution_hz"])
        dt = float(summary["hop_samples"]) / float(summary["sample_rate_hz"])
        high = float(matrix.max()) if matrix.size else -120.0
        if arrays["spectrum_db"].size:
            high = max(high, float(arrays["spectrum_db"].max()))
        # 平均功率谱密度：检测依据，与本底/门限一起显示；中位数 PSD 作对照
        self.detect_spectrum.clear()
        self.detect_spectrum.plot(f, arrays["spectrum_median_db"], pen="#9fb6cd", name="中位数 PSD（对照）")
        self.detect_spectrum.plot(f, arrays["spectrum_db"], pen="#2365b3", name="平均 PSD（检测依据）")
        self.detect_spectrum.addItem(pg.InfiniteLine(
            pos=threshold_db, angle=0, movable=False, pen=pg.mkPen("#e2564a", width=2)))
        self.detect_spectrum.addItem(pg.InfiniteLine(
            pos=noise_db, angle=0, movable=False,
            pen=pg.mkPen("#7d8fa1", style=QtCore.Qt.PenStyle.DashLine)))
        for item in detections:
            self.detect_spectrum.addItem(pg.LinearRegionItem(
                values=(item["f_low_hz"], item["f_high_hz"]), movable=False,
                brush=pg.mkBrush(35, 101, 179, 45), pen=pg.mkPen("#2365b3")))
        # 时频图叠加检测框（横轴频率、纵轴时间，与图像坐标系一致）
        levels = [high - 80.0, high]
        self.detect_tf_image.setImage(matrix.T, levels=levels, autoLevels=False)
        self.detect_tf_image.setRect(QtCore.QRectF(f[0] - df / 2, t[0] - dt / 2,
                                                   df * len(f), dt * len(t)))
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
        span = float(f[-1] - f[0]) / 2.0 + df
        self.detect_spectrum.setXRange(float(f[0]) - df / 2, float(f[-1]) + df / 2, padding=0.0)
        self.detect_spectrum.setYRange(max(noise_db - 5.0, high - 80.0), high + 3.0, padding=0.0)
        self.detect_tf.setXRange(float(f[0]) - df / 2, float(f[-1]) + df / 2, padding=0.0)
        self.detect_tf.setYRange(float(t[0]) - dt / 2, float(t[-1]) + dt / 2, padding=0.0)
        self.detect_tf.setTitle(f"时频图与检测框 · 动态范围 80 dB · ±{_fmt_hz(span)}"
                                + (" · 红框 AI 检出，灰虚线传统基线" if baseline_boxes else ""))
        self.detect_spectrum.setTitle(
            f"平均功率谱密度与检测门限 · 本底 {noise_db:.1f} dB/Hz · 门限 {threshold_db:.1f} dB/Hz"
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


    def display_result(self, result):
        self.last_result = result
        if result["kind"] == "analysis":
            self.tab_results[0] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_analysis(result, arrays)
            self.tabs.setCurrentIndex(0)
        elif result["kind"] in ("detect", "ml_detect"):
            self.tab_results[2] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_detect(result, arrays)
            self._render_compare()
            self.tabs.setCurrentIndex(2)
        elif result["kind"] == "amc_classify":
            self.tab_results[3] = result
            self._render_amc(result)
            self._render_compare()
            self.tabs.setCurrentIndex(3)
        elif result["kind"] == "native":
            self.tab_results[0] = result
            self.wave.clear()
            self.spectrum.clear()
            self.tf_image.clear()
            self.waterfall_image.clear()
            self.const_scatter.clear()
            self.tf_stack.setCurrentWidget(self.time_frequency)
            self.summary.setPlainText(f"原生复制完成：{result['plugin']['id']}\n"
                                      f"输出资产：{result['derived_asset_id']}\n请选择输出资产进行分析。")
            self.tabs.setCurrentIndex(0)

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



def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    window.show()
    return app.exec()
