"""对话框：“信号与采样”配置弹框（SignalParamsDialog，生成页使用）。"""

import math

from PySide6 import QtWidgets

from ..core_api import plan_signal
from .constants import MODE_SHORT
from .helpers import _fmt_hz
from .widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


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
        power_row, self.power = _plain_spin(-200.0, 0.0, spec.get("power_dbfs", -10.0), 1, "dBFS")
        bw_row, self.bandwidth = _freq_spin(1.0, rate, spec.get("bandwidth", rate / 10.0), 2)
        if mode == "noise":
            # 自定义噪声是全带白噪声：频点固定 0、带宽固定采样率，所以只填功率。
            form.addRow("功率", power_row)
        else:
            form.addRow("频点（基带偏移）", offset_row)
            form.addRow("功率", power_row)
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
        elif mode == "noise":
            note = QtWidgets.QLabel("自定义噪声为全带白噪声：频点固定 0、带宽固定为采样率，只填功率；"
                                    "不参与「背景噪声」的带内 SNR 折算。")
            note.setWordWrap(True)
            form.addRow(note)
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
        if self.mode == "noise":
            # 全带白噪声：频点/带宽由 plan_signal 固定（0 与采样率），只提交功率。
            return {"mode": "noise", "power_dbfs": self.power.value()}
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
        elif mode == "noise":
            text = (f"全带白噪声，带宽 = 采样率 {_fmt_hz(self.rate)}，"
                    f"功率谱密度 {plan['power_dbfs'] - 10.0 * math.log10(self.rate):.1f} dBFS/Hz")
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
