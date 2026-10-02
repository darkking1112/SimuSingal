"""算法对比页（mixin）：AI 与基线检测结果的并排对比。"""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from common.reports import amc_metrics, detection_metrics

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


class ComparePageMixin:
    def build_compare(self):
        """算法对比页：同一份数据上传统基线与 AI 路径的指标并排。"""
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "把同一个数据资产上的两条路径排在一起：信号检测侧的“检测结果 / 传统基线”取自 AI 检测运行时"
            "同步跑的能量检测，两者使用同一套真值、同一套会话合并口径；逐跳参数侧把“AI 逐跳 /"
            "传统逐跳 / 会话口径”三列排在一起，三者的驻留、带宽、功率与 SNR 都在原始 PSD 上重测；"
            "调制识别侧列出模型输出、生成器真值与命中情况。没有生成器真值（导入或原生插件产出）时"
            "只列结果、不计算指标，按“不适用”计数而不是静默丢弃。"
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
            "运行“信号检测”（或 AI 检测）、“跳频参数”（或 AI 逐跳）与“调制识别”后，"
            "这里显示并排对比与未确认项。")
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
        """把最近一次检测（AI/传统）、逐跳与调制识别结果整理成并排表格。"""
        detect = self.tab_results.get("信号检测")
        amc = self.tab_results.get("调制识别")
        hops = self.tab_results.get("跳频参数")
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
        if hops:
            summary = hops.get("summary") or {}
            metrics = hops.get("metrics")
            if metrics:
                main_label = self._object_label(
                    "AI 逐跳" if hops.get("model") else "逐跳", hops)
                for name, value in detection_metrics(metrics):
                    rows.append(("跳频参数", main_label, name, value))
                traditional = hops.get("traditional_metrics")
                if traditional:
                    for name, value in detection_metrics(traditional):
                        rows.append(("跳频参数", "传统逐跳（hop_track_v1）", name, value))
                    lines.append(_comparison_line(main_label, metrics, "传统逐跳", traditional))
                else:
                    lines.append("本次逐跳估计没有同步运行传统基线：跳频参数页勾选"
                                 "“并排对比传统逐跳”即可对照。")
                baseline = hops.get("baseline_metrics")
                if baseline:
                    for name, value in detection_metrics(baseline):
                        rows.append(("跳频参数", "会话口径（能量检测基线）", name, value))
                    lines.append(_comparison_line("逐跳口径", metrics, "会话口径", baseline))
                lines.append(f"逐跳结果：{len(hops.get('hops') or [])} 跳 · "
                             f"{len(hops.get('sessions') or [])} 个会话 · "
                             f"可分辨 {summary.get('hop_rate_limit_hz', '--')} Hz 以内")
            else:
                truth = hops.get("truth") or {}
                lines.append("该数据没有逐跳真值，只列逐跳结果，不计算指标（不适用）："
                             + str(truth.get("reason") or "不适用"))
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
            lines.append(f"识别模型 {model.get('id')}@{model.get('version', '--')} · "
                         + (f"来源 {_AMC_SOURCE_TEXT.get(str(model.get('source')), model.get('source'))} · "
                            if model.get("source") else
                            f"契约 {amc.get('contract', '--')}（标签集合 {amc.get('class_set', '--')}）· ")
                         + f"带内信噪比粗估 {_fmt_metric(amc.get('snr_estimate_db'), '.2f')} dB")
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
            self.tabs.setCurrentIndex(self._page_index("算法对比"))
            self.status.setText(f"算法对比已刷新（{len(rows)} 行指标）")
