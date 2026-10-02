"""调制识别页（mixin）：特征模型与原始 IQ 两条通路、模型清单选择与结果渲染。"""
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
                         IMPORT_COL_NOTE, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_RF_CENTER, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..dialogs import ImportBatchDialog, SignalParamsDialog
from ..helpers import (_AMC_SOURCE_TEXT, _asset_exports, _asset_format, _comparison_line,
                       _fmt_hz, _fmt_metric, _fmt_span, _iq_binary_kind, _mirrored_spectrum)
from ..runner import _run_task
from ..widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class AmcPageMixin:
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
            "若所选清单声明 contract = iq_waveform_v1，本页自动改走原始 IQ 通路：在同一分析频带内取"
            "定长复基带窗口（单位 RMS 归一化）直接交给 1D CNN／TCN，由网络自行学习调制特征；标签集合"
            "由清单决定（A09 六类或更宽的独立字典），结果契约 amc_iq_classify_v1，与特征通路互不影响。"
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
        self.amc_button.setToolTip(
            "对左侧选中的数据资产按当前分析频带开始调制识别；模型清单若声明"
            " iq_waveform_v1 则自动走原始 IQ 通路")
        self.amc_button.clicked.connect(self.amc_classify_selected)
        self.amc_from_detect = QtWidgets.QPushButton("取用检测结果频带")
        self.amc_from_detect.setToolTip(
            "把最近一次检测结果中功率最大的目标中心频率与带宽填进左侧输入框")
        self.amc_from_detect.clicked.connect(self.use_detected_band)
        bar.addWidget(self.amc_from_detect)
        self.adopt_amc_button = QtWidgets.QPushButton("采纳为参数标注")
        self.adopt_amc_button.setEnabled(False)
        self.adopt_amc_button.setToolTip(
            "把当前识别结论写入目标参考参数（modulation，source=algorithm）；"
            "集合已有 AMC 标注集时同时追加标签（字典外保留原始类名）。")
        self.adopt_amc_button.clicked.connect(
            lambda: self.adopt_page_result("调制识别"))
        bar.addWidget(self.adopt_amc_button)
        bar.addStretch(1)
        layout.addLayout(bar)
        model_bar = QtWidgets.QHBoxLayout()
        model_bar.addWidget(QtWidgets.QLabel("识别模型"))
        self.amc_model = QtWidgets.QLineEdit()
        self.amc_model.setPlaceholderText("留空使用内置线性基线；也可选择自训练模型 JSON 或 ONNX 清单")
        self.amc_model.setToolTip("模型 JSON 为 amc_model_v1；ONNX 清单为 amc-manifest（特征通路）或"
                                  "amc-iq-manifest（原始 IQ 通路）生成的 JSON，本页按清单契约自动分流")
        model_bar.addWidget(self.amc_model, 1)
        self.amc_choose = QtWidgets.QPushButton("选择…")
        self.amc_choose.clicked.connect(self.choose_amc_model)
        model_bar.addWidget(self.amc_choose)
        model_bar.addWidget(self.amc_button)  # 紧邻模型选择：选完模型即可点它开始识别
        self.amc_model_status = QtWidgets.QLabel()
        model_bar.addWidget(self.amc_model_status)
        model_bar.addStretch(1)
        layout.addLayout(model_bar)
        grid = QtWidgets.QGridLayout()
        self.amc_plot = pg.PlotWidget(title="六类后验概率")
        self.amc_plot.setLabel("left", "概率")
        # 类别轴与概率轴都是固定语义：禁用鼠标缩放/平移，避免类别名被移出视野
        self.amc_plot.setMouseEnabled(x=False, y=False)
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
        from ...algorithms.amc.feature_model import default_model_path

        present = default_model_path().is_file()
        self.amc_model_status.setText(
            "内置线性基线可用" if present else
            "未找到内置模型，请先运行 training/train_amc.py，或手动选择模型文件")


    def choose_amc_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择调制识别模型", str(self.workspace.root), "模型文件 (*.json *.onnx)")
        if path:
            self.amc_model.setText(path)


    def _amc_model_contract(self):
        """读取所选清单声明的契约；不是 JSON 清单或读不动时返回 ``None``。

        只做分流判断，真正的校验交给识别入口：这里不能替用户“宽容”非法清单，
        所以任何读取失败都按“非 IQ 清单”处理，再由 amc_classify 给出明确报错。
        """
        path = self.amc_model.text().strip()
        if not path or Path(path).suffix.lower() != ".json":
            return None
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return str(payload.get("contract") or "") or None


    def _use_iq_branch(self):
        """所选清单是否声原始 IQ 通路（决定调 amc_iq_classify 还是 amc_classify）。"""
        from ...contracts.iq import IQ_WAVEFORM_CONTRACT

        return self._amc_model_contract() == IQ_WAVEFORM_CONTRACT


    def use_detected_band(self):
        """把检测结果里最强目标的中心频率与带宽填进识别输入框（可手动再改）。"""
        result = self.tab_results.get("信号检测") or {}
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
        action = "amc_iq_classify" if self._use_iq_branch() else "amc_classify"
        self.start_job(action, asset_id=asset["id"], config=config, model=model)


    def _plot_amc_scores(self, classes, labels, scores, title):
        """画出类别后验概率柱状图（特征通路与 IQ 通路共用）。"""
        values = [float(scores.get(name, 0.0)) for name in classes]
        positions = np.arange(len(classes), dtype=float)
        if self.amc_bars is not None:
            self.amc_plot.removeItem(self.amc_bars)
        self.amc_bars = pg.BarGraphItem(x=positions, height=values, width=0.62,
                                        brush="#2365b3", pen=pg.mkPen("#183c65"))
        self.amc_plot.addItem(self.amc_bars)
        self.amc_plot.getAxis("bottom").setTicks(
            [[(float(position), labels[name]) for position, name in zip(positions, classes)]])
        self.amc_plot.setXRange(-0.5, len(classes) - 0.5, padding=0.0)
        self.amc_plot.setYRange(0.0, max(1.0, max(values) * 1.15), padding=0.0)
        self.amc_plot.setTitle(title)


    def _render_amc(self, result):
        summary = result["summary"]
        prediction = result["prediction"]
        classes = summary["classes"]
        labels = summary["labels"]
        features = summary["features"]
        self._plot_amc_scores(classes, labels, prediction["scores"],
                              f"六类后验概率 · 预测 {prediction['label_text']}"
                              f"（{prediction['confidence']:.2f}）")
        self.amc_table.setHorizontalHeaderLabels(["特征", "取值"])
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


    def _render_amc_iq(self, result):
        """原始 IQ 通路的展示：类别后验概率 + 输入窗口口径（没有 34 维特征表）。

        右侧表格改成“输入口径/取值”：这条通路没有可逐项对照的确定性特征，
        列无可列的假特征反而是误导，因此只列真正决定模型输入的那几个量。
        """
        summary = result["summary"]
        prediction = result["prediction"]
        classes = summary["classes"]
        labels = summary["labels"]
        waveform = summary["waveform"]
        model = summary["model"] or {}
        self._plot_amc_scores(classes, labels, prediction["scores"],
                              f"类别后验概率 · 预测 {prediction['label_text']}"
                              f"（{prediction['confidence']:.2f}）")
        rows = [("输入契约", waveform["contract"]), ("通道排布", waveform["layout"]),
                ("窗口采样点", f"{waveform['samples']:,}"),
                ("归一化", waveform["normalization"]),
                ("分析率 / Hz", _fmt_hz(waveform["analysis_rate_hz"])),
                ("抽样比", waveform["decimation"]),
                ("窗口起点（分析后）", f"{waveform['window_start']:,}"),
                ("源样本 / 分析样本", f"{waveform['source_samples']:,} / "
                                      f"{waveform['analysis_samples']:,}"),
                ("带内功率 / dBFS", f"{waveform['power_dbfs']:.2f}"),
                ("窗口 RMS / 峰均比", f"{waveform['rms']:.4f} / {waveform['crest_factor']:.4f}"),
                ("带内信噪比粗估 / dB", _fmt_metric(waveform.get("snr_estimate_db"), ".2f"))]
        self.amc_table.setHorizontalHeaderLabels(["输入口径", "取值"])
        self.amc_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.amc_table.setItem(index, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.amc_table.setItem(index, 1, QtWidgets.QTableWidgetItem(str(value)))
        timing = summary["timing"]
        lines = [
            f"数据：{result.get('asset_name', result['asset_id'])}  |  采样率 "
            f"{_fmt_hz(waveform['sample_rate_hz'])}  |  分析频带 中心 {_fmt_hz(waveform['offset_hz'])}"
            f" · 带宽 {_fmt_hz(waveform['bandwidth_hz'])}  |  输入窗口 {waveform['samples']:,} 点"
            f"（I/Q 两通道 · 单位 RMS）",
            f"算法 {summary['algorithm']}（契约 {summary['contract']}）· 标签集合 "
            f"{summary['class_set']}（{len(classes)} 类）· 预处理 {timing['preprocess_ms']} ms + "
            f"推理 {timing['inference_ms']} ms = {timing['total_ms']} ms",
            f"模型 {model.get('id')}@{model.get('version')}"
            + (f" · 摘要 {str(model['sha256'])[:12]}…" if model.get("sha256") else "")
            + (f" · 运行时 onnxruntime {model['runtime_version']}"
               if model.get("runtime_version") else "")
            + (f" · 训练 {model['training']}" if model.get("training") else ""),
            f"预测：{prediction['label_text']}（概率 {prediction['confidence']:.4f} · 与次高类差值 "
            f"{_fmt_metric(prediction.get('margin'), '.4f')}）· 带内信噪比粗估 "
            f"{_fmt_metric(summary['snr_estimate_db'], '.1f')} dB",
        ]
        if prediction["reliable"]:
            lines.append("可信度：未发现低信噪比或区分度不足的提示；" + prediction["snr_note"])
        else:
            lines.append(f"可信度：仅供参考 —— {prediction['reason']}；{prediction['snr_note']}")
        lines.append("对照说明：原始 IQ 通路没有传统启发式基线（传统判定只对确定性特征有意义），"
                     "因此这里不列对照行，也不拿它与特征通路的结果互相顶替。")
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
