"""Training workbench: annotation editing and external-process training."""
import codecs
import json
import os
from pathlib import Path
import signal
import sys
import uuid
from datetime import datetime, timezone

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .annotations import AnnotationDataset, create_dataset
from .training_jobs import iq_plan, list_experiments, save_record


STATUS = {"running": "运行中", "success": "成功", "failed": "失败", "stopped": "已停止",
          "waiting": "等待", "interrupted": "已中断"}

#: 强制结束信号；Windows 没有 SIGKILL，None 让 signal_process 直接走 QProcess.kill()
FORCE_KILL = getattr(signal, "SIGKILL", None)


def decode_worker_log(data):
    """解码 worker.log 字节：新实验为 UTF-8；历史实验由管道默认编码（GBK）写出，逐级回退。"""
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


def metric_text(report):
    def percent(value):
        return "不适用" if value is None else f"{100 * value:.2f}%"
    if "validation" in report:
        val = report["validation"]
        lines = [f"验证集准确率：{percent(val.get('accuracy'))}", "混淆矩阵（行：真值；列：预测）",
                 "类别顺序：" + " / ".join(val.get("labels", []))]
        for label, row in zip(val.get("labels", []), val.get("confusion", [])):
            lines.append(f"{label}: " + "  ".join(map(str, row)))
        return "\n".join(lines)
    lines = ["按保存的检测框评分（IoU ≥ 0.5，置信度 ≥ 0.25）"]
    for split, metrics in report.get("splits", {}).items():
        name = {"val": "验证集", "test": "测试集"}.get(split, split)
        lines.append(f"{name}：{metrics['samples']} 个样本 · 精确率 {percent(metrics['precision'])} · 召回率 {percent(metrics['recall'])}")
        lines.append(f"  无信号样本误报率：{percent(metrics['noise_false_alarm_rate'])} · CPU 推理与去重 P50：{metrics['latency_ms_p50']:.2f} ms")
    return "\n".join(lines)


class TrainingPage(QtWidgets.QWidget):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.root = window.workspace.root / "training"
        self.root.mkdir(exist_ok=True)
        self.process = None
        self.dataset = None
        self.current_index = -1
        self.rois = []
        self.selected_roi = None
        self.dirty = False
        self.record = None
        self.directory = None
        self.process_group = None
        self.internal_models = not getattr(sys, "frozen", False) and os.environ.get("SIMUSIGNAL_RELEASE") != "1"
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("模型训练 · 检测框标注 / IQ 分类 / 外部训练环境"))
        self.sections = QtWidgets.QTabWidget()
        layout.addWidget(self.sections)
        self.sections.addTab(self.build_annotations(), "数据标注")
        self.sections.addTab(self.build_training(), "训练配置")
        self.sections.addTab(self.build_monitor(), "实验与日志")
        self.refresh_history()

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
        return field

    def spin(self, form, title, value, low, high):
        field = QtWidgets.QSpinBox()
        field.setRange(low, high)
        field.setValue(value)
        form.addRow(title, field)
        return field

    def build_annotations(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        buttons = QtWidgets.QHBoxLayout()
        for text, callback in (("新建标注集", self.new_dataset), ("打开检测数据集", self.open_dataset),
                               ("添加左侧选中 IQ 资产", self.add_asset),
                               ("保存当前标注", self.save_annotation)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.dataset_label = QtWidgets.QLabel("新建后，先在左侧选择已导入或生成的 IQ 资产，再添加为时频图。")
        self.dataset_label.setWordWrap(True)
        layout.addWidget(self.dataset_label)
        split = QtWidgets.QSplitter()
        self.samples = QtWidgets.QListWidget()
        self.samples.currentRowChanged.connect(self.select_sample)
        split.addWidget(self.samples)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "时间 →（像素）")
        self.plot.setLabel("left", "频率递减 ↓（像素）")
        self.plot.invertY(True)
        self.plot.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        split.addWidget(self.plot)
        split.setSizes([220, 700])
        layout.addWidget(split, 1)
        row = QtWidgets.QHBoxLayout()
        self.split_choice = QtWidgets.QComboBox()
        for title, value in (("训练集", "train"), ("验证集", "val"), ("测试集（暂不参与训练）", "test")):
            self.split_choice.addItem(title, value)
        self.split_choice.currentIndexChanged.connect(self.mark_dirty)
        row.addWidget(self.split_choice)
        for text, callback in (("新增信号框", self.add_box), ("删除选中框", self.delete_box),
                               ("标记无信号", self.clear_boxes)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            row.addWidget(button)
        layout.addLayout(row)
        self.annotation_status = QtWidgets.QLabel("单类 emitter；拖动框移动，拖动角点缩放。修改后保存；空框保存表示确认无信号。")
        self.annotation_status.setWordWrap(True)
        layout.addWidget(self.annotation_status)
        return widget

    def build_training(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        candidate = Path(__file__).resolve().parents[2]
        self.repository = self.path_field(form, "训练源码根目录", str(candidate))
        default_python = "" if getattr(sys, "frozen", False) else sys.executable
        self.python = self.path_field(form, "训练环境 Python", default_python, "file")
        self.task = QtWidgets.QComboBox()
        self.task.addItem("时频图信号检测", "detection")
        self.task.addItem("原始 IQ 调制识别", "iq")
        form.addRow("任务", self.task)
        self.arch = QtWidgets.QComboBox()
        form.addRow("模型", self.arch)
        self.data = self.path_field(form, "训练数据集")
        self.source = QtWidgets.QComboBox()
        for title, value in (("已有 IQ 数据集", "existing"), ("生成 A09 数据", "generator"),
                             ("生成 A09 + 混入 TorchSig bundle", "bundle")):
            self.source.addItem(title, value)
        form.addRow("IQ 数据来源", self.source)
        self.bundle = self.path_field(form, "TorchSig bundle")
        self.mapping = self.path_field(form, "TorchSig → A09 映射 JSON", kind="file")
        self.weights = self.path_field(form, "初始权重（YOLO 空值使用 yolo26s.pt）", kind="file")
        self.framework = self.path_field(form, "RT-DETR rtdetrv2_pytorch 目录")
        self.framework_config = self.path_field(form, "RT-DETRv2 模型 YAML", kind="file")
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cpu", "cuda"])
        form.addRow("训练设备", self.device)
        self.epochs = self.spin(form, "轮数", 20, 1, 10000)
        self.batch = self.spin(form, "批大小", 8, 1, 65536)
        self.lr = QtWidgets.QDoubleSpinBox()
        self.lr.setDecimals(7)
        self.lr.setRange(.0000001, 1)
        self.lr.setValue(.001)
        form.addRow("学习率", self.lr)
        self.seed = self.spin(form, "随机种子", 7, 0, 2147483647)
        self.per_class = self.spin(form, "IQ 每类生成数量", 200, 2, 1000000)
        self.iq_samples = self.spin(form, "IQ 窗口采样点数", 1024, 64, 65536)
        self.snr_low = self.spin(form, "IQ 生成 SNR 下限（dB）", -5, -100, 100)
        self.snr_high = self.spin(form, "IQ 生成 SNR 上限（dB）", 30, -100, 100)
        self.count = self.spin(form, "检测数据生成条数", 64, 2, 1000000)
        self.size = QtWidgets.QComboBox()
        self.size.addItems(["128", "256", "512", "1024"])
        self.size.setCurrentText("1024")
        form.addRow("新建/生成检测图像边长", self.size)
        generate = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(generate)
        for title, kind in (("用本项目生成器准备检测数据", "generate"), ("用 TorchSig 准备检测数据", "torchsig")):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda checked=False, task=kind: self.start_generation(task))
            row.addWidget(button)
        form.addRow(generate)
        self.train_button = QtWidgets.QPushButton("开始训练 → 导出 → 验收")
        self.train_button.clicked.connect(self.start_training)
        form.addRow(self.train_button)
        note = QtWidgets.QLabel("验证集指标与当前资产的真值评分分别计算。无 generation 真值的资产仍可推理。\n"
                              "YOLO26s 供开发对照；RT-DETR 入口面向 lyuwenyu v2。训练依赖安装在所选 Python 环境。")
        note.setWordWrap(True)
        form.addRow(note)
        # 启动失败时就地提示：只写在其它标签页会让用户以为按钮没反应
        self.config_status = QtWidgets.QLabel("")
        self.config_status.setWordWrap(True)
        self.config_status.setStyleSheet("color: #b00020;")
        form.addRow(self.config_status)
        self.task.currentIndexChanged.connect(self.update_models)
        self.update_models()
        scroll.setWidget(body)
        return scroll

    def update_models(self):
        self.arch.clear()
        detection = self.task.currentData() == "detection"
        self.arch.addItems((["rtdetr", "yolo26s"] if self.internal_models else ["rtdetr"])
                           if detection else ["cnn", "tcn"])
        for field in (self.source, self.bundle, self.mapping, self.per_class, self.iq_samples,
                      self.snr_low, self.snr_high):
            field.setEnabled(not detection)
        for field in (self.weights, self.framework, self.framework_config):
            field.setEnabled(detection)

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
        self.results.setPlaceholderText("训练完成后显示验证/测试指标；原始报告保存在实验目录。")
        layout.addWidget(self.results)
        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.curves = pg.PlotWidget(title="训练 loss / IQ 验证准确率（0–1）")
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
        self.stop_button.clicked.connect(self.stop)
        row.addWidget(self.stop_button)
        self.load_button = QtWidgets.QPushButton("加载验收通过的模型")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_model)
        row.addWidget(self.load_button)
        layout.addLayout(row)
        return widget

    def report_error(self, error):
        text = str(error)
        self.annotation_status.setText(text)
        self.status.setText(text)
        status = getattr(self, "config_status", None)
        if status is not None:
            status.setText(text)

    def new_dataset(self):
        if self.process:
            return
        try:
            if not self.save_if_dirty():
                return
            directory = self.root / "datasets" / uuid.uuid4().hex
            create_dataset(directory, size=int(self.size.currentText()))
            self.set_dataset(directory)
        except (ValueError, OSError) as exc:
            self.report_error(exc)

    def open_dataset(self):
        if self.process or not self.save_if_dirty():
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "打开检测数据集")
        if path:
            self.set_dataset(path)

    def set_dataset(self, path):
        try:
            dataset = AnnotationDataset(path)
            if dataset.labels != ["emitter"]:
                raise ValueError("标注第一版仅支持单类 emitter 数据集")
            self.dataset = dataset
            self.current_index = -1
            self.dirty = False
            self.samples.clear()
            self.clear_rois()
            self.image_item.clear()
            self.samples.addItems([f"{i + 1:04d} · {Path(r['image']).name}" for i, r in enumerate(dataset.records)])
            self.dataset_label.setText(f"{path} · {len(dataset.records)} 个样本")
            self.data.setText(str(path))
            if dataset.records:
                self.samples.setCurrentRow(0)
        except (ValueError, OSError, KeyError) as exc:
            self.report_error(exc)

    def mark_dirty(self, *_):
        if self.current_index >= 0:
            self.dirty = True
            self.annotation_status.setText("当前标注已修改，尚未保存")

    def clear_rois(self):
        for roi in self.rois:
            self.plot.removeItem(roi)
        self.rois = []
        self.selected_roi = None

    def add_box(self, checked=False, box=None):
        if self.current_index < 0:
            return
        size = self.dataset.size
        x, y, w, h = (box or [.5, .5, .25, .15])[:4]
        roi = pg.RectROI(((x - w / 2) * size, (y - h / 2) * size), (w * size, h * size),
                         maxBounds=QtCore.QRectF(0, 0, size, size), pen=pg.mkPen("y", width=2))
        roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        roi.sigClicked.connect(lambda *args, item=roi: self.select_roi(item))
        roi.sigRegionChanged.connect(self.mark_dirty)
        self.plot.addItem(roi)
        self.rois.append(roi)
        self.select_roi(roi)
        self.mark_dirty()

    def select_roi(self, roi):
        self.selected_roi = roi
        for item in self.rois:
            item.setPen(pg.mkPen("c" if item is roi else "y", width=2))

    def delete_box(self):
        if self.selected_roi is not None:
            self.plot.removeItem(self.selected_roi)
            self.rois.remove(self.selected_roi)
            self.selected_roi = None
            self.mark_dirty()

    def clear_boxes(self):
        self.clear_rois()
        self.mark_dirty()

    def save_if_dirty(self):
        return self.save_annotation() if self.dirty else True

    def save_annotation(self):
        if self.current_index < 0 or self.dataset is None:
            return True
        try:
            size = self.dataset.size
            boxes = [[(r.pos().x() + r.size().x() / 2) / size,
                      (r.pos().y() + r.size().y() / 2) / size,
                      r.size().x() / size, r.size().y() / size, 1, 0] for r in self.rois]
            self.dataset.save(self.current_index, boxes, self.split_choice.currentData())
            self.samples.item(self.current_index).setText(
                f"{self.current_index + 1:04d} · {self.split_choice.currentText()} · 已确认 {len(boxes)} 框")
            self.dirty = False
            self.annotation_status.setText(f"已保存 · {len(boxes)} 个框 · {self.split_choice.currentText()}")
            return True
        except (ValueError, OSError) as exc:
            self.report_error(exc)
            return False

    def select_sample(self, index):
        if not self.save_if_dirty():
            self.samples.blockSignals(True)
            self.samples.setCurrentRow(self.current_index)
            self.samples.blockSignals(False)
            return
        if index < 0 or self.dataset is None:
            return
        try:
            array = self.dataset.image(index)
            record = self.dataset.record(index)
            self.clear_rois()
            self.current_index = index
            self.image_item.setImage(array, levels=(0, 1))
            self.plot.setRange(QtCore.QRectF(0, 0, self.dataset.size, self.dataset.size))
            self.split_choice.setCurrentIndex(self.split_choice.findData(record.get("split", "train")))
            for box in record.get("boxes", []):
                self.add_box(box=box)
            self.dirty = False
            self.annotation_status.setText(f"{len(self.rois)} 个框 · " +
                ("待标注/确认" if record.get("annotation_status") == "pending" else "已有标签"))
        except (ValueError, OSError, KeyError) as exc:
            self.report_error(exc)

    def configuration(self):
        return {"repository": self.repository.text().strip(), "python": self.python.text().strip(),
            "task": self.task.currentData(), "arch": self.arch.currentText(),
            "data": self.data.text().strip(), "source": self.source.currentData(),
            "bundle": self.bundle.text().strip(), "mapping": self.mapping.text().strip(),
            "weights": self.weights.text().strip(), "framework_path": self.framework.text().strip(),
            "framework_config": self.framework_config.text().strip(), "device": self.device.currentText(),
            "epochs": self.epochs.value(), "batch": self.batch.value(), "lr": self.lr.value(),
            "seed": self.seed.value(), "per_class": self.per_class.value(), "samples": self.iq_samples.value(),
            "snr_low": self.snr_low.value(), "snr_high": self.snr_high.value(),
            "count": self.count.value(), "image_size": int(self.size.currentText())}

    def add_asset(self):
        if self.process or not self.save_if_dirty():
            return
        asset = self.window.selected_asset()
        if self.dataset is None or asset is None:
            self.report_error("请先打开/新建标注集，并在左侧选择一个 IQ 资产")
            return
        config = self.configuration()
        config.update(task="asset", data=str(self.dataset.root),
                      workspace=str(self.window.workspace.root), asset_id=asset["id"])
        self.start(config)

    def start_generation(self, task):
        if not self.save_if_dirty():
            return
        config = self.configuration()
        config["task"] = task
        self.start(config)

    def start_training(self):
        if self.save_if_dirty():
            self.start(self.configuration())

    def start(self, config):
        if self.process:
            self.report_error("已有任务正在运行；请等待完成，或到“实验与日志”页停止后再开始")
            return
        try:
            import shutil
            python = shutil.which(config["python"])
            worker = Path(config["repository"]).expanduser().resolve() / "training/desktop_worker.py"
            if not python or not worker.is_file():
                raise ValueError("请选择有效的训练 Python 和包含 training/desktop_worker.py 的源码目录")
            if config["task"] == "iq":
                iq_plan(config, self.root / "preflight")
            elif config["task"] == "detection":
                if config["arch"] == "yolo26s" and not self.internal_models:
                    raise ValueError("发行配置不提供 YOLO26s 训练")
                AnnotationDataset(config["data"])
                if config["arch"] == "rtdetr" and (
                        not (Path(config["framework_path"]) / "src/core/yaml_config.py").is_file()
                        or not Path(config["framework_config"]).is_file()):
                    raise ValueError("RT-DETR 需要选择 rtdetrv2_pytorch 目录和模型 YAML")
            identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
            self.directory = self.root / "runs" / identifier
            self.directory.mkdir(parents=True)
            self.record = {"id": identifier, "status": "running", "config": config,
                           "stage": "启动", "metrics": [], "error": None}
            save_record(self.directory, self.record)
            self.stopping = False
            self.process_group = None
            self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self.buffer = ""
            self.log.clear()
            self.results.clear()
            self.loss_curve.clear()
            self.accuracy_curve.clear()
            self.process = QtCore.QProcess(self)
            self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            self.process.setWorkingDirectory(str(worker.parent.parent))
            environment = QtCore.QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONUNBUFFERED", "1")
            environment.insert("OMP_NUM_THREADS", "2")
            environment.insert("MKL_NUM_THREADS", "2")
            # stdout 不是控制台时 Python 按系统编码（中文 Windows=GBK）输出；
            # 这里强制子进程树输出 UTF-8，与 GUI 的 UTF-8 解码器（及日志文件）保持一致
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            self.process.setProcessEnvironment(environment)
            self.process.readyReadStandardOutput.connect(self.read_output)
            self.process.finished.connect(self.finished)
            self.process.errorOccurred.connect(self.process_error)
            self.process.started.connect(self.started)
            self.stop_button.setEnabled(True)
            self.train_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.history.setEnabled(False)
            self.sections.widget(0).setEnabled(False)
            self.sections.setCurrentIndex(2)
            self.progress.setRange(0, 0)
            self.config_status.clear()
            self.status.setText("启动训练环境…")
            self.process.start(python, ["-u", str(worker), str(self.directory / "experiment.json")])
        except (OSError, ValueError, KeyError) as exc:
            self.report_error(exc)

    def started(self):
        self.record["pid"] = int(self.process.processId())
        save_record(self.directory, self.record)

    def read_output(self):
        data = bytes(self.process.readAllStandardOutput())
        with (self.directory / "worker.log").open("ab") as handle:
            handle.write(data)
        self.buffer += self.decoder.decode(data)
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.log.appendPlainText(line)
            if line.startswith("TRAIN_EVENT "):
                try:
                    item = json.loads(line[len("TRAIN_EVENT "):])
                    if "stage" in item:
                        self.record["stage"] = item["stage"]
                        self.status.setText(item["stage"])
                        self.progress.setRange(0, 0)
                    if "epoch" in item:
                        self.record["metrics"].append(item)
                        self.progress.setRange(0, self.record["config"]["epochs"])
                        self.progress.setValue(item["epoch"])
                        self.draw_metrics(self.record["metrics"])
                    save_record(self.directory, self.record)
                except (ValueError, KeyError, TypeError):
                    pass

    def draw_metrics(self, metrics):
        loss = [m for m in metrics if "loss" in m]
        accuracy = [m for m in metrics if "validation_accuracy" in m]
        self.loss_curve.setData([m["epoch"] for m in loss], [m["loss"] for m in loss])
        self.accuracy_curve.setData([m["epoch"] for m in accuracy], [m["validation_accuracy"] for m in accuracy])

    def process_error(self, error):
        if self.process and error == QtCore.QProcess.ProcessError.FailedToStart:
            self.record["error"] = self.process.errorString()
            self.finished(-1, QtCore.QProcess.ExitStatus.CrashExit)

    def finished(self, code, exit_status):
        if self.process is None:
            return
        self.read_output()
        self.buffer += self.decoder.decode(b"", final=True)
        if self.buffer:
            self.log.appendPlainText(self.buffer)
        state = "stopped" if self.stopping else (
            "success" if code == 0 and exit_status == QtCore.QProcess.ExitStatus.NormalExit else "failed")
        if self.stopping and self.process_group is not None:
            try:
                os.killpg(self.process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.record.update(status=state, exit_code=code, finished=datetime.now(timezone.utc).isoformat())
        save_record(self.directory, self.record)
        self.process.deleteLater()
        self.process = None
        self.stop_button.setEnabled(False)
        self.train_button.setEnabled(True)
        self.history.setEnabled(True)
        self.sections.widget(0).setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(int(state == "success"))
        self.status.setText(f"{STATUS[state]} · {self.record['stage']} · {self.directory}")
        config = self.record["config"]
        if state == "success" and config["task"] in ("generate", "torchsig", "asset"):
            self.set_dataset(config["data"] if config["task"] == "asset" else self.directory / "data")
            self.sections.setCurrentIndex(0)
        self.refresh_history()

    def stop(self):
        if self.process is None:
            return
        self.stopping = True
        self.status.setText("正在停止任务…")
        self.signal_process(signal.SIGTERM)
        process = self.process
        QtCore.QTimer.singleShot(2000, lambda: self.signal_process(FORCE_KILL)
                                if self.process is process else None)

    def signal_process(self, sig):
        if self.process is None:
            return
        pid = int(self.process.processId())
        if pid <= 0:
            self.process.kill()
            return
        try:
            if os.name == "posix" and os.getpgid(pid) == pid:
                self.process_group = pid
                os.killpg(pid, sig)
            elif sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
        except ProcessLookupError:
            pass

    def shutdown(self):
        if not self.save_if_dirty():
            return False
        if self.process:
            self.stopping = True
            self.signal_process(signal.SIGTERM)
            self.process.waitForFinished(1000)
            if self.process:
                self.signal_process(FORCE_KILL)
                self.process.waitForFinished(1000)
        return self.process is None

    def refresh_history(self):
        self.history.blockSignals(True)
        self.history.clear()
        for record in list_experiments(self.root / "runs"):
            if not self.internal_models and record["config"].get("arch") == "yolo26s":
                continue
            state = record["status"]
            if state == "running":
                try:
                    if record.get("pid", 0) <= 0:
                        raise ProcessLookupError()
                    os.kill(record["pid"], 0)
                except ProcessLookupError:
                    state = "interrupted"
                    record["status"] = state
                except PermissionError:
                    pass
            self.history.addItem(f"{record['id']} · {record['config'].get('arch', '')} · {STATUS.get(state, state)}", record)
        self.history.blockSignals(False)
        self.show_experiment()

    def show_experiment(self, *_):
        if self.process:
            return
        record = self.history.currentData()
        self.load_button.setEnabled(False)
        if not record:
            return
        directory = Path(record["directory"])
        path = directory / "worker.log"
        if path.is_file():
            start = max(0, path.stat().st_size - 200000)
            with path.open("rb") as stream:
                stream.seek(start)
                data = stream.read()
            if start and b"\n" in data:  # 从中间截取时丢掉首个不完整行
                data = data.split(b"\n", 1)[1]
            self.log.setPlainText(decode_worker_log(data))
        else:
            self.log.clear()
        self.draw_metrics(record.get("metrics", []))
        report = directory / "model/metrics.json"
        self.results.clear()
        if report.is_file():
            try:
                result = json.loads(report.read_text(encoding="utf-8"))
                result.pop("history", None)
                self.results.setPlainText(metric_text(result))
            except (OSError, ValueError):
                self.results.setPlainText("指标文件无法读取，请查看实验日志")
        self.status.setText(f"{STATUS.get(record['status'], record['status'])} · {record.get('error') or directory}")
        manifest = directory / "model" / ("iq_manifest.json" if record["config"]["task"] == "iq" else "detector.json")
        self.load_button.setEnabled(record["status"] == "success" and manifest.is_file())

    def load_model(self):
        record = self.history.currentData()
        if not record or record["status"] != "success":
            return
        directory = Path(record["directory"]) / "model"
        if record["config"]["task"] == "iq":
            self.window.amc_model.setText(str(directory / "iq_manifest.json"))
            self.window.tabs.setCurrentIndex(self.window._page_index("调制识别"))
        else:
            path = directory / "detector.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            hop = manifest.get("training", {}).get("label_semantics") == "per_hop_v1"
            (self.window.hops_manifest if hop else self.window.ml_manifest).setText(str(path))
            self.window.tabs.setCurrentIndex(self.window._page_index("跳频参数" if hop else "信号检测"))
