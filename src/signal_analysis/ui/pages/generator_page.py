"""IQ 信号生成页（mixin）：单个信号生成与集合生成子页接线。"""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
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
from .collection_gen_page import CollectionGenPanel


class GeneratorPageMixin:
    def build_generator(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("gen_sections")
        layout.addWidget(tabs)
        single = QtWidgets.QWidget()
        single_layout = QtWidgets.QVBoxLayout(single)
        intro = QtWidgets.QLabel("生成用于测试检测、参数估计与调制识别算法的 IQ 基带信号。"
                                 "IQ 为复基带记录，不设置载频：\"频点\"指基带频率偏移；"
                                 "频率值以 Hz / kHz / MHz 显示（单位在输入框外）。"
                                 "可一次包含多种信号并独立设置参数，自动按目标带宽推导调制参数。")
        intro.setWordWrap(True)
        single_layout.addWidget(intro)
        from common.gui import TaskBanner
        self.generator_banner = TaskBanner("IQ 信号生成", "取消本次生成")
        self.register_task_banner("IQ 信号生成", self.generator_banner)
        single_layout.addWidget(self.generator_banner)
        single_layout.addWidget(self._build_global_row())
        single_layout.addWidget(self._build_collection_row())
        single_layout.addWidget(self._build_noise_group())
        single_layout.addWidget(self._build_signal_table(), 1)
        single_layout.addWidget(self._build_export_row())
        self.gen_result = QtWidgets.QLabel("尚未生成")
        self.gen_result.setWordWrap(True)
        self.gen_result.setStyleSheet(
            "background:white;border:1px solid #d8e1ec;border-radius:5px;padding:8px;")
        single_layout.addWidget(self.gen_result)
        tabs.addTab(single, "单个信号生成")
        self.gen_panel = CollectionGenPanel(self)
        tabs.addTab(self.gen_panel, "信号集合生成")
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


    def _build_collection_row(self):
        """生成产物的目标集合（新建或追加）与初始标注开关（方案 §7.2）。"""
        group = QtWidgets.QGroupBox("目标集合")
        row = QtWidgets.QHBoxLayout(group)
        self.gen_collection = QtWidgets.QComboBox()
        self.gen_collection.setToolTip("生成后把资产加入所选集合；选“新建集合…”时填名称（同名沿用）")
        self.gen_collection.addItem("不加入集合", None)
        self.gen_collection.addItem("新建集合…", "__new__")
        self.gen_collection.currentIndexChanged.connect(
            lambda *_: self._gen_collection_changed())
        row.addWidget(self.gen_collection, 1)
        self.gen_collection_name = QtWidgets.QLineEdit()
        self.gen_collection_name.setPlaceholderText("新集合名称")
        self.gen_collection_name.setVisible(False)
        row.addWidget(self.gen_collection_name)
        self.gen_initial_labels = QtWidgets.QCheckBox("同时生成初始标注")
        self.gen_initial_labels.setToolTip(
            "按目标参考参数写入初始标注：集合无标注集时自动建立默认检测/AMC 标注集")
        row.addWidget(self.gen_initial_labels)
        return group

    def _gen_collection_changed(self):
        self.gen_collection_name.setVisible(self.gen_collection.currentData() == "__new__")

    def _gen_collection_request(self):
        scope = self.gen_collection.currentData()
        request = {"collection_id": scope if scope not in (None, "__new__") else None,
                   "initial_labels": self.gen_initial_labels.isChecked()}
        if scope == "__new__":
            name = self.gen_collection_name.text().strip()
            if not name:
                raise ValueError("已选择“新建集合”，请填写集合名称")
            request["collection_name"] = name
        return request

    def _build_noise_group(self):
        """背景噪声底：全带白噪声，只在存在调制信号时可用，按最强调制信号的带内 SNR 定标。

        纯噪声记录请用“添加样式 → 自定义噪声”，那是独立信号行（只填功率），与这里的噪声底互不影响。
        """
        group = QtWidgets.QGroupBox("背景噪声")
        row = QtWidgets.QHBoxLayout(group)
        self.gen_noise_enabled = QtWidgets.QCheckBox("启用")
        self.gen_noise_enabled.setChecked(True)
        self.gen_noise_enabled.setToolTip("背景噪声底为全带白噪声（带宽 = 采样率），以最强调制信号为参考；\n"
                                          "没有调制信号时整组不可用，纯噪声请添加“自定义噪声”样式")
        self.gen_noise_enabled.toggled.connect(self.update_gen_controls)
        row.addWidget(self.gen_noise_enabled)
        row.addWidget(QtWidgets.QLabel("带内 SNR（参考：最强调制信号）"))
        snr_row, self.gen_snr = _plain_spin(-10.0, 80.0, 20.0, 1, "dB")
        self.gen_snr.setToolTip("最强调制信号（实测平均功率最大的调制信号）的带内 SNR = 其平均功率 ÷ 同带宽内的噪声功率；\n"
                                "噪声为全带白噪声，按双侧功率谱密度 N0 折算，其余信号按各自带宽折算、可能低于填写值\n"
                                "（窄带信号反而更高）")
        row.addWidget(snr_row)
        self.gen_noise_hint = QtWidgets.QLabel("需先添加调制信号；纯噪声请用“添加样式 → 自定义噪声”")
        self.gen_noise_hint.setStyleSheet("color:#7a8798;")
        row.addWidget(self.gen_noise_hint)
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
        self.gen_signals.setHorizontalHeaderLabels(["样式", "频点", "功率", "带宽", "参数摘要"])
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
        row.addWidget(QtWidgets.QLabel("信号名称"))
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
        # 背景噪声底以最强调制信号为参考：没有调制信号（或只有自定义噪声行）时整组不可用。
        has_modulation = any(spec["mode"] != "noise" for spec in self.iq_signals)
        self.gen_noise_enabled.setEnabled(has_modulation)
        active_noise = self.gen_noise_enabled.isChecked() and has_modulation
        self.gen_snr.setEnabled(active_noise)
        self.gen_noise_hint.setVisible(not has_modulation)
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
        if plan["mode"] == "noise":
            psd = plan["power_dbfs"] - 10.0 * np.log10(self.gen_rate.value())
            return f"全带白噪声 · 功率谱密度 {psd:.1f} dBFS/Hz"
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
        if spec.get("mode") == "noise":
            # 自定义噪声是全带白噪声，没有频点/带宽可填。
            offset_text = "—"
            bandwidth_text = f"全带（{_fmt_hz(self.gen_rate.value())}）"
        else:
            offset_text = _fmt_hz(spec.get("offset", 0))
            bandwidth_text = _fmt_hz(spec.get("bandwidth", 0))
        self.gen_signals.setItem(row, 1, QtWidgets.QTableWidgetItem(offset_text))
        self.gen_signals.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{spec.get('power_dbfs', -10):g} dBFS"))
        self.gen_signals.setItem(row, 3, QtWidgets.QTableWidgetItem(bandwidth_text))
        self.gen_signals.setItem(row, 4, QtWidgets.QTableWidgetItem(self._describe_signal(spec)))


    def generate_iq_clicked(self):
        rate = self.gen_rate.value()
        duration = self.gen_duration.value()
        count = int(round(rate * duration))
        if not 1 <= count <= MAX_SAMPLES:
            self.status.setText(f"采样点数 {count:,} 超出 1～{MAX_SAMPLES:,} 范围，请调整采样率或持续时间")
            return
        noise = None
        if self.gen_noise_enabled.isChecked() and any(
                spec["mode"] != "noise" for spec in self.iq_signals):
            # 背景噪声是全带白噪声（带宽 = 采样率），只发带内 SNR。
            noise = {"enabled": True, "snr_db": self.gen_snr.value()}
        export = None
        fmt = self.gen_export_format.currentData()
        if fmt:
            export = {"format": fmt, "endian": self.gen_endian.currentData()}
        try:
            collection = self._gen_collection_request()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.start_job("generate", owner="IQ 信号生成", label="生成信号",
                       cancel_text="取消本次生成",
                       sample_rate=rate, duration=duration,
                       seed=int(self.gen_seed.value()), signals=self.iq_signals,
                       noise=noise, name=self.gen_name.text().strip() or None,
                       export=export, **collection)


    def show_generation_result(self, result):
        summary = result["summary"]
        lines = [f"已生成资产：{result['name']}",
                 f"采样率 {_fmt_hz(summary['sample_rate_hz'])} · {summary['sample_count']:,} 个复采样 · "
                 f"时长 {summary['duration_s']:g} s · 峰值 {summary['peak_dbfs']:.1f} dBFS"]
        for entry in summary["signals"]:
            style = MODE_SHORT.get(entry["mode"], entry["mode"])
            if entry["mode"] == "noise":
                lines.append(f"  {style}：全带白噪声（带宽 = 采样率）· "
                             f"功率 {entry['power_dbfs']:g} dBFS（实测 {entry['power_dbfs_actual']:.2f} dBFS）")
                continue
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
                ref_text = "" if reference is None else f"（参考信号 #{reference + 1}，实测功率最大）"
                lines.append(f"  背景噪声：带宽 {_fmt_hz(noise['bandwidth'])} · 带内 SNR {noise['snr_db']:g} dB"
                             f"{ref_text} · 总功率 {noise['power_dbfs']:.1f} dBFS{psd_text}")
            elif noise["power_dbfs"] is not None:
                lines.append(f"  背景噪声：带宽 {_fmt_hz(noise['bandwidth'])} · "
                             f"总功率 {noise['power_dbfs']:.1f} dBFS{psd_text}")
        if result["export_path"]:
            lines.append(f"导出文件：{result['export_path']}（格式 {result['export_format']}）")
            if result.get("export_data_path"):
                lines.append(f"IQ 数据文件：{result['export_data_path']}")
        targets = result.get("targets") or {}
        if targets.get("sessions"):
            lines.append(f"目标参考参数：会话 {targets['sessions']} 个"
                         + (f" · 逐跳 {targets['hops']} 个" if targets.get("hops") else ""))
        if result.get("collection_id"):
            labels = result.get("initial_labels") or 0
            lines.append(f"目标集合：{result['collection_name']}"
                         + (f" · 初始标注 {labels} 条" if labels else ""))
        self.gen_result.setText("\n".join(lines))
