"""Shared widgets, export flow, history, and lifecycle hooks for training pages."""
import json
import os
from pathlib import Path
import sys

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from common.gui import TaskBanner, direct_entry
from ...services.training_jobs import decode_worker_log, list_experiments
from ..training_runner import TrainingRunner


STATUS = {"running": "运行中", "success": "成功", "failed": "失败", "stopped": "已停止",
          "waiting": "等待", "interrupted": "已中断"}


def _percent(value):
    return "未提供" if value is None else f"{100 * value:.2f}%"


def _snr_lines(metrics):
    """按 SNR 分档统计：训练侧写 ``per_snr``（字典或列表两种形状）。"""
    if isinstance(metrics, dict):
        return [f"{bucket}：{_percent(value)}" for bucket, value in metrics.items()]
    lines = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        high = row.get("high_db")
        bucket = (f"{row.get('low_db'):g}~{high:g} dB" if high is not None
                  else f"≥{row.get('low_db'):g} dB")
        count = row.get("count")
        lines.append(f"{bucket}：{_percent(row.get('accuracy'))}"
                     + (f"（{count} 个样本）" if count is not None else ""))
    return lines


def metric_text(report):
    percent = _percent

    if "validation" in report:
        val = report["validation"]
        lines = [f"验证集准确率：{percent(val.get('accuracy'))}"]
        if val.get("macro_f1") is not None:
            lines.append(f"Macro-F1：{percent(val['macro_f1'])}")
        labels = val.get("labels", [])
        confusion = val.get("confusion", [])
        if labels or confusion:
            lines.extend(["混淆矩阵（行：真值；列：预测）",
                          "类别顺序：" + (" / ".join(labels) if labels else "未提供")])
            lines.extend(f"{label}: " + "  ".join(map(str, row))
                         for label, row in zip(labels, confusion))
        snr_metrics = val.get("per_snr") or val.get("by_snr")
        if snr_metrics:
            snr_lines = _snr_lines(snr_metrics)
            if snr_lines:
                lines.append("按 SNR 统计：")
                lines.extend(f"  {line}" for line in snr_lines)
        return "\n".join(lines)

    lines = ["按保存的检测框评分（IoU ≥ 0.5，置信度 ≥ 0.25）"]
    for split, metrics in report.get("splits", {}).items():
        name = {"val": "验证集", "test": "测试集"}.get(split, split)
        samples = metrics.get("samples", "未提供")
        lines.append(
            f"{name}：{samples} 个样本 · 精确率 {percent(metrics.get('precision'))}"
            f" · 召回率 {percent(metrics.get('recall'))} · F1 {percent(metrics.get('f1'))}")
        lines.append(
            f"  无信号样本误报率：{percent(metrics.get('noise_false_alarm_rate'))}"
            f" · CPU 推理与去重 P50：{metrics.get('latency_ms_p50', '未提供')} ms")
    return "\n".join(lines)


class TrainingPageBase(QtWidgets.QWidget):
    """Common training UI; task-specific behavior lives in the concrete page."""

    TASK = ""
    OWNER = ""
    TITLE = ""
    MODEL_OPTIONS = ()
    HISTORY_TASKS = ()
    HAS_ANNOTATIONS = False
    ANNOTATION_TITLE = "数据标注"

    def __init__(self, window, coordinator):
        super().__init__(window)
        self.window = window
        self.coordinator = coordinator
        self.root = window.workspace.root / "training"
        self.root.mkdir(parents=True, exist_ok=True)
        self.annotations_tab = None
        self.internal_models = (not getattr(sys, "frozen", False)
                                and os.environ.get("SIMUSIGNAL_RELEASE") != "1")
        self._sidebar_collection_id = None
        self._path_rows = {}
        self.runner = TrainingRunner(window, self, coordinator)

        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel(self.TITLE)
        title.setObjectName("title")
        layout.addWidget(title)
        self.sections = QtWidgets.QTabWidget()
        layout.addWidget(self.sections)

        self.config_tab = self.build_training()
        self.annotations_tab = self.build_annotations() if self.HAS_ANNOTATIONS else None
        self.monitor_tab = self.build_monitor()
        if self.annotations_tab is not None:
            self.sections.addTab(self.annotations_tab, self.ANNOTATION_TITLE)
        self.sections.addTab(self.config_tab, "训练配置")
        self.sections.addTab(self.monitor_tab, "实验与日志")

        self.runner.log_line.connect(self.log.appendPlainText)
        self.runner.stage_changed.connect(self._on_stage_changed)
        self.runner.metrics_changed.connect(self.draw_metrics)
        self.runner.state_changed.connect(self._on_runner_state)
        self.runner.run_finished.connect(self._on_run_finished)
        self.coordinator.slot_changed.connect(self.refresh_controls)
        self.window.tasks.started.connect(self.refresh_controls)
        self.window.tasks.finished.connect(self.refresh_controls)
        self.update_data_summary()
        self.refresh_history()
        self.refresh_controls()

    def path_field(self, form, title, default="", kind="directory"):
        row = QtWidgets.QWidget()
        horizontal = QtWidgets.QHBoxLayout(row)
        horizontal.setContentsMargins(0, 0, 0, 0)
        field = QtWidgets.QLineEdit(default)
        button = QtWidgets.QPushButton("选择…")
        horizontal.addWidget(field, 1)
        horizontal.addWidget(button)

        def choose():
            if kind == "directory":
                result = QtWidgets.QFileDialog.getExistingDirectory(self, title, field.text())
            else:
                result, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, field.text())
            if result:
                field.setText(result)

        button.clicked.connect(choose)
        form.addRow(title, row)
        self._path_rows[title] = row
        return field

    @staticmethod
    def spin(form, title, value, low, high):
        field = QtWidgets.QSpinBox()
        field.setRange(low, high)
        field.setValue(value)
        form.addRow(title, field)
        return field

    def build_training(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        self.config_inputs = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.config_inputs)
        candidate = Path(__file__).resolve().parents[4]
        self.repository = self.path_field(form, "训练源码根目录", str(candidate))
        default_python = "" if getattr(sys, "frozen", False) else sys.executable
        self.python = self.path_field(form, "训练环境 Python", default_python, "file")

        self.arch = QtWidgets.QComboBox()
        self.arch.addItems(self.model_options())
        form.addRow("模型", self.arch)
        self.train_collection = QtWidgets.QComboBox()
        self.train_collection.setToolTip("训练集：集合内已标注的信号就是本次训练的输入")
        self.val_collection = QtWidgets.QComboBox()
        self.val_collection.setToolTip("验证集：必须与训练集不同，且不要与训练集同源")
        form.addRow("训练集（信号集合）", self.train_collection)
        form.addRow("验证集（信号集合）", self.val_collection)
        self.collection_hint = QtWidgets.QLabel()
        self.collection_hint.setWordWrap(True)
        form.addRow(self.collection_hint)
        self.refresh_collections()

        self.task_banner = TaskBanner(self.OWNER, "取消当前任务")
        self.window.register_task_banner(self.OWNER, self.task_banner)
        self.config_status = QtWidgets.QLabel("")
        self.config_status.setWordWrap(True)
        self.config_status.setStyleSheet("color: #b00020;")
        self._hint_key = None

        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("训练设备", self.device)
        self.epochs = self.spin(form, "轮数", 20, 1, 10000)
        self.batch = self.spin(form, "批大小", 8, 1, 65536)
        self.lr = QtWidgets.QDoubleSpinBox()
        self.lr.setDecimals(7)
        self.lr.setRange(.0000001, 1)
        self.lr.setValue(.001)
        direct_entry(self.lr)
        self.lr.setSingleStep(10.0 ** -self.lr.decimals())
        form.addRow("学习率", self.lr)
        self.seed = self.spin(form, "随机种子", 7, 0, 2147483647)
        self.add_task_fields(form)

        self.train_button = QtWidgets.QPushButton("开始训练 → 验收")
        self.train_button.setObjectName("primary")
        self.train_button.clicked.connect(self.start_training)
        form.addRow(self.train_button)
        note = QtWidgets.QLabel(self.page_note())
        note.setWordWrap(True)
        form.addRow(note)
        form.addRow(self.config_status)
        self.train_collection.currentIndexChanged.connect(self.refresh_controls)
        self.val_collection.currentIndexChanged.connect(self.refresh_controls)
        self.arch.currentIndexChanged.connect(self.refresh_controls)

        body_layout.addWidget(self.config_inputs)
        body_layout.addWidget(self.task_banner)
        scroll.setWidget(body)
        return scroll

    def add_task_fields(self, form):
        pass

    def model_options(self):
        if self.TASK == "detection" and not self.internal_models:
            return [model for model in self.MODEL_OPTIONS if model != "yolo26s"]
        return list(self.MODEL_OPTIONS)

    def page_note(self):
        return ("训练数据直接取自信号集合：选择训练集与验证集集合即可，页内不生成数据、"
                "不导出训练集。训练依赖安装在所选 Python 环境。")

    def build_annotations(self):
        raise NotImplementedError

    def build_monitor(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.history = QtWidgets.QComboBox()
        self.history.currentIndexChanged.connect(self.show_experiment)
        layout.addWidget(self.history)
        self.status = QtWidgets.QLabel("就绪")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.results = QtWidgets.QPlainTextEdit()
        self.results.setReadOnly(True)
        self.results.setMaximumHeight(170)
        self.results.setPlaceholderText("训练完成后显示验收指标；原始报告保存在实验目录。")
        layout.addWidget(self.results)
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.curves = pg.PlotWidget(title=self.curve_title())
        self.curves.addLegend()
        self.loss_curve = self.curves.plot(pen="y", name="loss")
        self.accuracy_curve = self.curves.plot(pen="c", name="验证准确率")
        layout.addWidget(self.curves, 1)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        layout.addWidget(self.log, 1)
        row = QtWidgets.QHBoxLayout()
        self.stop_button = QtWidgets.QPushButton("停止任务")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.runner.stop)
        row.addWidget(self.stop_button)
        self.load_button = QtWidgets.QPushButton("加载验收通过的模型")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_model)
        row.addWidget(self.load_button)
        layout.addLayout(row)
        self._update_collection_hint()
        return widget

    def curve_title(self):
        return "训练 loss"

    def configuration(self):
        return {
            "repository": self.repository.text().strip(),
            "python": self.python.text().strip(),
            "task": self.TASK,
            "arch": self.arch.currentText(),
            "workspace": str(self.window.workspace.root),
            "train_collection_id": self.train_collection.currentData(),
            "val_collection_id": self.val_collection.currentData(),
            "weights": "",
            "framework_path": "",
            "framework_config": "",
            "device": self.device.currentText(),
            "epochs": self.epochs.value(),
            "batch": self.batch.value(),
            "lr": self.lr.value(),
            "seed": self.seed.value(),
        }

    def refresh_collections(self):
        """重填训练集/验证集下拉；尽量保留原选择，训练集默认跟随侧栏集合。

        验证集默认不预设（保持空选）：必须由用户显式选择，避免与训练集悄悄相同。
        """
        if not hasattr(self, "train_collection"):
            return
        collections = self.window.workspace.list_collections()
        for combo, default, fallback_last in ((self.train_collection,
                                               self._sidebar_collection_id, True),
                                              (self.val_collection, None, False)):
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for collection in collections:
                combo.addItem(f"{collection['name']}（{collection['asset_count']}）",
                              collection["id"])
            wanted = previous or default
            index = combo.findData(wanted) if wanted else -1
            if index < 0 and fallback_last and combo.count():
                index = combo.count() - 1
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self.update_data_summary()
        self.refresh_controls()

    def sync_train_collection(self, collection_id):
        """侧栏选中集合变化时同步「训练集」；页内改选在下次侧栏切换前保持不变。"""
        self._sidebar_collection_id = collection_id
        if collection_id is None or not hasattr(self, "train_collection"):
            return
        index = self.train_collection.findData(collection_id)
        if index >= 0 and self.train_collection.currentIndex() != index:
            self.train_collection.setCurrentIndex(index)

    def task_buttons(self):
        return (self.train_button,)

    def task_block_reason(self, action):
        return self.window.tasks.blocked_reason(self.OWNER, action)

    def refresh_controls(self, *_):
        if not hasattr(self, "stop_button"):
            return
        slot = self.coordinator.slot
        own_slot = slot is not None and slot["owner"] == self.OWNER
        self.config_inputs.setEnabled(not own_slot)
        if self.annotations_tab is not None:
            self.annotations_tab.setEnabled(not own_slot)
        train_reason = self.task_block_reason("external_training")
        collections_ready = bool(self.train_collection.currentData()
                                 and self.val_collection.currentData())
        can_train = (slot is None and train_reason is None and not self.runner.active
                     and collections_ready)
        self.train_button.setEnabled(can_train)
        self.train_collection.setEnabled(not own_slot)
        self.val_collection.setEnabled(not own_slot)
        self.stop_button.setEnabled(self.runner.active)
        self.update_task_controls()
        self._update_collection_hint()

    #: 标签状态 → 提示里的中文名（训练页说明"为什么样本比资产少"）
    _LABEL_STATE_NAMES = {"unlabeled": "未标注", "unknown": "未知调制",
                          "out_of_taxonomy": "字典外类别"}

    def _update_collection_hint(self):
        """在集合下拉下面给出本页当前选择的可用样本数与被跳过标签，避免"静默少样本"。"""
        if not hasattr(self, "collection_hint"):
            return
        train_id = self.train_collection.currentData()
        val_id = self.val_collection.currentData()
        if not train_id or not val_id:
            self.collection_hint.setText("选择训练集与验证集后显示本次训练可用的样本数。")
            return
        key = (train_id, val_id)
        if key == getattr(self, "_hint_key", None):
            return
        self._hint_key = key
        from ...services.training_inputs import inputs_plan
        try:
            plan = inputs_plan(self.window.workspace, self.TASK, train_id, val_id)
        except ValueError as exc:
            self.collection_hint.setText(f"当前选择不可训练：{exc}")
            return
        parts = []
        for label, section in (("训练集", plan["train"]), ("验证集", plan["val"])):
            counts = section["counts"]
            text = f"{label} {counts['samples']} 个样本（{counts['assets']} 条资产）"
            states = section["stats"].get("label_states") or {}
            skipped = "、".join(f"{self._LABEL_STATE_NAMES.get(state, state)} {count}"
                                for state, count in states.items() if count)
            if skipped:
                text += f"，跳过 {skipped}"
            parts.append(text)
        self.collection_hint.setText("；".join(parts) + "。")

    def update_task_controls(self):
        pass

    def _data_path_changed(self, *_):
        if hasattr(self, "refresh_controls"):
            self.update_data_summary()
            self.refresh_controls()

    def update_data_summary(self):
        pass

    def report_error(self, error):
        text = str(error)
        self.status.setText(text)
        self.config_status.setText(text)
        annotation_status = getattr(self, "annotation_status", None)
        if annotation_status is not None:
            annotation_status.setText(text)

    def prepare_start(self):
        return self.save_if_dirty()

    def save_if_dirty(self):
        return True

    def prepare_shutdown(self):
        return self.save_if_dirty()

    def start_training(self):
        if not self.prepare_start():
            return
        config = self.configuration()
        self.runner.start(config, owner=self.OWNER, task_kind="training",
                          validate_config=self.validate_training_config)

    def validate_training_config(self, config):
        if config.get("task") != self.TASK:
            raise ValueError(f"{self.TITLE} 只接受 {self.TASK} 训练任务")
        if config.get("arch") not in self.model_options():
            raise ValueError("请选择当前页面支持的模型")
        train_id = config.get("train_collection_id")
        val_id = config.get("val_collection_id")
        if not train_id or not val_id:
            raise ValueError("请选择训练集和验证集信号集合")
        if train_id == val_id:
            raise ValueError("训练集与验证集不能是同一个集合")
        from ...services.training_inputs import inputs_plan
        # 预检与 worker 首阶段共用同一份集合输入装配：训练集→train、验证集→val
        self.inputs_plan = inputs_plan(self.window.workspace,
                                       "amc" if self.TASK == "iq" else "detection",
                                       train_id, val_id)
        self.validate_task_config(config)

    def validate_task_config(self, config):
        pass

    def update_models(self):
        self.refresh_controls()

    def training_label(self, config):
        return f"{self.TITLE}：{config['arch']}"

    def draw_metrics(self, metrics):
        losses = [item for item in metrics if "loss" in item]
        accuracy = [item for item in metrics if "validation_accuracy" in item]
        self.loss_curve.setData([item["epoch"] for item in losses],
                                [item["loss"] for item in losses])
        self.accuracy_curve.setData([item["epoch"] for item in accuracy],
                                    [item["validation_accuracy"] for item in accuracy])

    def _on_stage_changed(self, stage):
        self.status.setText(stage)

    def _on_runner_state(self, state):
        name = state.get("state", "idle")
        if name in ("starting", "running", "stopping", "finalizing"):
            self.sections.setCurrentWidget(self.monitor_tab)
            self.status.setText({
                "starting": "正在启动训练环境…",
                "running": "训练运行中",
                "stopping": "正在停止训练…",
                "finalizing": "正在确认训练进程树已退出…",
            }[name])
        self.refresh_controls()

    def _on_run_finished(self, record):
        self.progress.setRange(0, 1)
        self.progress.setValue(int(record.get("status") == "success"))
        if record.get("id"):
            self.refresh_history(select_id=record["id"])
        self.refresh_controls()

    def refresh_history(self, *_args, select_id=None):
        if not hasattr(self, "history"):
            return
        previous = select_id
        if previous is None and self.history.currentData():
            previous = self.history.currentData().get("id")
        active = self.runner.active
        self.history.blockSignals(True)
        self.history.clear()
        records = list_experiments(self.root / "runs")
        records = [record for record in records
                   if record.get("config", {}).get("task") in self.HISTORY_TASKS
                   and (self.internal_models or record.get("config", {}).get("arch") != "yolo26s")]
        for record in records:
            state = record["status"]
            if state == "running" and record.get("pid"):
                try:
                    os.kill(record["pid"], 0)
                except ProcessLookupError:
                    state = "interrupted"
                    record["status"] = state
                except PermissionError:
                    pass
            task = record.get("config", {}).get("task")
            label = "资产转样本" if task == "asset" else record["config"].get("arch", "")
            self.history.addItem(
                f"{record['id']} · {label} · {STATUS.get(state, state)}", record)
        index = next((i for i in range(self.history.count())
                      if self.history.itemData(i).get("id") == previous), -1)
        if index >= 0:
            self.history.setCurrentIndex(index)
        self.history.blockSignals(False)
        if not active:
            self.show_experiment()

    def show_experiment(self, *_):
        if self.runner.active:
            return
        record = self.history.currentData()
        self.load_button.setEnabled(False)
        self.results.clear()
        if not record:
            self.log.clear()
            self.status.setText("就绪")
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.draw_metrics([])
            return
        directory = Path(record["directory"])
        path = directory / "worker.log"
        if path.is_file():
            start = max(0, path.stat().st_size - 200000)
            with path.open("rb") as stream:
                stream.seek(start)
                data = stream.read()
            if start and b"\n" in data:
                data = data.split(b"\n", 1)[1]
            self.log.setPlainText(decode_worker_log(data))
        else:
            self.log.clear()
        self.draw_metrics(record.get("metrics", []))
        report = directory / "model" / "metrics.json"
        if report.is_file():
            try:
                result = json.loads(report.read_text(encoding="utf-8"))
                result.pop("history", None)
                self.results.setPlainText(metric_text(result))
            except (OSError, ValueError, TypeError) as exc:
                self.results.setPlainText(f"指标文件无法读取：{exc}")
        task = record.get("config", {}).get("task")
        self.status.setText(
            f"{STATUS.get(record['status'], record['status'])} · "
            f"{record.get('error') or directory}")
        manifest_name = "iq_manifest.json" if task == "iq" else "detector.json"
        manifest = directory / "model" / manifest_name
        self.load_button.setEnabled(
            task in self.HISTORY_TASKS and task != "asset"
            and record["status"] == "success" and manifest.is_file())

    def load_model(self):
        record = self.history.currentData()
        if (not record or record.get("status") != "success"
                or record.get("config", {}).get("task") not in self.HISTORY_TASKS
                or record.get("config", {}).get("task") == "asset"):
            return
        directory = Path(record["directory"]) / "model"
        task = record["config"]["task"]
        try:
            if task == "iq":
                path = directory / "iq_manifest.json"
                if not path.is_file():
                    raise ValueError("验收通过的 IQ 模型清单不存在")
                self.window.amc_model.setText(str(path))
                page_name = "调制识别"
            else:
                path = directory / "detector.json"
                if not path.is_file():
                    raise ValueError("验收通过的检测模型清单不存在")
                manifest = json.loads(path.read_text(encoding="utf-8"))
                hop = manifest.get("training", {}).get("label_semantics") == "per_hop_v1"
                (self.window.hops_manifest if hop else self.window.ml_manifest).setText(str(path))
                page_name = "跳频参数" if hop else "信号检测"
            self.window.tabs.setCurrentIndex(self.window._page_index(page_name))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.report_error(f"无法加载模型：{exc}")
