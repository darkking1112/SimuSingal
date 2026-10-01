"""“信号集合生成”面板（生成页第二子页）：表单配置 → 内部生成参数 JSON → 执行器。

设计（方案 §6.3 与用户约定）：

* 界面是**表单**，配方（``gen_recipe_v1``）只是内部存储格式，用来保证可复现；
  生成时执行器自动把配方存进集合；
* 参数较多，明细放在**配置弹框**里（“信号与采样”），页面上只保留基本项、摘要与操作；
* 损伤项由 :mod:`signal_analysis.impairments` 声明，未实现的整组置灰（实现后自动启用）；
* JSON 导入/导出放在“高级”里，页面不需要手写或编辑 JSON；
* 检测标注粒度默认逐跳（``per_hop_v1``）：项目引擎生成跳频样式时同时写会话与逐跳目标；
* TorchSig 只能在 Linux 下生成，Windows 上直接提示；bundle 导入任何平台都能用。
"""
import json
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from . import impairments
from .core_api import MODE_NAMES
from .impairments import IMPAIRMENTS, RESERVED_SIGNAL_PARAMETERS
from .recipes import check_generator_support, validate_recipe
from .torchsig_support import is_linux, load_mapping, read_env, write_env

ENGINE_CHOICES = (("项目引擎", "project"), ("TorchSig（仅 Linux）", "torchsig"))
DETECTION_CHOICES = (("会话级（一个会话一个框）", "session_v1"),
                     ("逐跳（一跳一个框；默认）", "per_hop_v1"))
MODE_LABELS = list(MODE_NAMES.items())


def _line(lo, hi, unit="", digits=3):
    text = f"{lo:g}" if lo == hi else f"{lo:g}～{hi:g}"
    return f"{text} {unit}".strip()


class SignalParamsDialog(QtWidgets.QDialog):
    """“信号与采样”配置弹框：项目引擎与 TorchSig 共用，按引擎显示对应项。"""

    def __init__(self, parent, settings, engine):
        super().__init__(parent)
        self.setWindowTitle("生成参数 · 信号与采样")
        self.engine = engine
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.rate = QtWidgets.QDoubleSpinBox()
        self.rate.setRange(1e3, 1e9)
        self.rate.setDecimals(0)
        self.rate.setValue(float(settings["rate"]))
        self.rate.setToolTip("单条录制的采样率（Hz）；TorchSig 引擎用这一个值")
        form.addRow("采样率（Hz）", self.rate)
        self.duration_low, self.duration_high = self._range_row(
            form, "时长（s）", settings["duration"], 1e-3, 3600.0, 3)
        self.count_low, self.count_high = self._range_row(
            form, "每条的信号数", settings["signals"], 0, 16, 0)
        self.snr_low, self.snr_high = self._range_row(
            form, "SNR（dB）", settings["snr"], -60.0, 120.0, 1)
        self.bandwidth_low, self.bandwidth_high = self._range_row(
            form, "带宽占比（%）", settings["bandwidth_ratio"], 0.5, 100.0, 2)

        self.modes = QtWidgets.QListWidget()
        self.modes.setToolTip("参与轮换的调制类型；类别均衡时按样本序号轮换保证配比")
        for key, label in MODE_LABELS:
            item = QtWidgets.QListWidgetItem(f"{label}（{key}）")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if key in settings["modes"]
                               else QtCore.Qt.CheckState.Unchecked)
            self.modes.addItem(item)
        form.addRow("调制类型", self.modes)
        self.balanced = QtWidgets.QCheckBox("类别均衡（balanced：按序号轮换）")
        self.balanced.setChecked(bool(settings["balanced"]))
        form.addRow("", self.balanced)
        self.power_low, self.power_high = self._range_row(
            form, "功率（dBFS）", settings["power"], -200.0, 0.0, 1)
        self.hop_low, self.hop_high = self._range_row(
            form, "跳速（hop/s）", settings["hop"], 0.1, 1e6, 2)
        form.addRow(QtWidgets.QLabel("跳速只作用于跳频样式（FH-2FSK / FH-OFDM）。"))

        self.generators = QtWidgets.QLineEdit(settings["torchsig"]["generators"])
        self.generators.setToolTip("signal_generators 参数：all 或逗号分隔的类名")
        form.addRow("TorchSig 信号族", self.generators)
        self.impairment = QtWidgets.QComboBox()
        self.impairment.addItem("不加扰动（IQ 与元数据严格对应）", None)
        for level, label in ((0, "0 · 理想"), (1, "1 · 有线"), (2, "2 · 无线")):
            self.impairment.addItem(label, level)
        index = self.impairment.findData(settings["torchsig"]["impairment"])
        self.impairment.setCurrentIndex(max(index, 0))
        form.addRow("TorchSig 扰动档位", self.impairment)
        self.mapping = QtWidgets.QLineEdit(settings["torchsig"]["mapping"])
        self.mapping.setPlaceholderText("{\"TorchSig 类名\": \"A09 类别\"} 的 JSON 文件")
        choose_mapping = QtWidgets.QPushButton("选择…")
        choose_mapping.clicked.connect(self._choose_mapping)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.mapping, 1)
        box.addWidget(choose_mapping)
        form.addRow("类别映射 JSON", row)

        for widget in (self.modes, self.balanced, self.power_low, self.power_high,
                       self.hop_low, self.hop_high):
            self._set_enabled(widget, engine == "project")
        for widget in (self.generators, self.impairment, self.mapping, row):
            self._set_enabled(widget, engine == "torchsig")

        note = QtWidgets.QLabel(
            "项目引擎：数字样式的符号率由带宽与滚降系数推导；独立符号率与损伤项"
            "（频偏 / 相位噪声 / IQ 不平衡 / 多径）尚未实现，页面置灰，用到即报错。")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _set_enabled(widget, enabled):
        if isinstance(widget, tuple):
            for item in widget:
                item.setEnabled(enabled)
        else:
            widget.setEnabled(enabled)

    def _range_row(self, form, title, values, low, high, decimals):
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        left = QtWidgets.QDoubleSpinBox()
        right = QtWidgets.QDoubleSpinBox()
        for spin in (left, right):
            spin.setRange(low, high)
            spin.setDecimals(decimals)
        left.setValue(float(values[0]))
        right.setValue(float(values[1]))
        box.addWidget(left)
        box.addWidget(QtWidgets.QLabel("～"))
        box.addWidget(right)
        form.addRow(title, row)
        return left, right

    def _choose_mapping(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "类别映射 JSON", "",
                                                        "JSON (*.json)")
        if path:
            self.mapping.setText(path)

    def values(self):
        def pair(low, high):
            return (low.value(), high.value())

        modes = [self.modes.item(index).data(QtCore.Qt.ItemDataRole.UserRole)
                 for index in range(self.modes.count())
                 if self.modes.item(index).checkState() == QtCore.Qt.CheckState.Checked]
        return {
            "rate": self.rate.value(),
            "duration": pair(self.duration_low, self.duration_high),
            "signals": pair(self.count_low, self.count_high),
            "snr": pair(self.snr_low, self.snr_high),
            "bandwidth_ratio": (self.bandwidth_low.value() / 100.0,
                                self.bandwidth_high.value() / 100.0),
            "modes": modes, "balanced": self.balanced.isChecked(),
            "power": pair(self.power_low, self.power_high),
            "hop": pair(self.hop_low, self.hop_high),
            "torchsig": {"generators": self.generators.text().strip() or "all",
                         "impairment": self.impairment.currentData()},
            "torchsig_mapping": self.mapping.text().strip(),
        }


class TorchSigEnvDialog(QtWidgets.QDialog):
    """TorchSig 环境设置弹框：训练源码根目录 + 装有 torchsig 的 Python，可测试。"""

    def __init__(self, parent, panel):
        super().__init__(parent)
        self.setWindowTitle("TorchSig 环境")
        self.panel = panel
        env = read_env(panel.window.workspace)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        candidate = str(Path(__file__).resolve().parents[2])
        self.repository = QtWidgets.QLineEdit(env["repository"] or candidate)
        choose_repo = QtWidgets.QPushButton("选择…")
        choose_repo.clicked.connect(self._choose_repository)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.repository, 1)
        box.addWidget(choose_repo)
        form.addRow("训练源码根目录", row)
        self.python = QtWidgets.QLineEdit(env["python"] or sys.executable)
        choose_python = QtWidgets.QPushButton("选择…")
        choose_python.clicked.connect(self._choose_python)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.python, 1)
        box.addWidget(choose_python)
        form.addRow("TorchSig Python", row)
        self.test_button = QtWidgets.QPushButton("测试环境")
        self.test_button.clicked.connect(self.test)
        layout.addWidget(self.test_button)
        self.status = QtWidgets.QLabel("在 Linux 上测试：平台、Python、脚本与 import torchsig。")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                             | QtWidgets.QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_repository(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "训练源码根目录",
                                                          self.repository.text())
        if path:
            self.repository.setText(path)

    def _choose_python(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "TorchSig Python",
                                                        self.python.text())
        if path:
            self.python.setText(path)

    def environment(self):
        return {"repository": self.repository.text().strip(),
                "python": self.python.text().strip()}

    def test(self):
        if not is_linux():
            self.status.setText("当前系统不是 Linux：TorchSig 只能在 Linux 下生成数据。"
                               "可在 Linux/WSL/远程机生成 bundle 后回来导入。")
            return
        self.status.setText("正在测试…")
        self.test_button.setEnabled(False)
        self.panel.window.start_job("torchsig_probe", torchsig_env=self.environment())

    def show_probe(self, result):
        self.test_button.setEnabled(True)
        self.status.setText(result.get("message") or "无结果")

    def save(self):
        try:
            write_env(self.panel.window.workspace, self.repository.text(), self.python.text())
        except OSError as exc:
            self.status.setText(f"保存失败：{exc}")
            return
        self.panel.status.setText("TorchSig 环境设置已保存")
        self.accept()


class CollectionGenPanel(QtWidgets.QWidget):
    """生成页“信号集合生成”子页：表单 + 参数预览 + 开始生成 + 高级。"""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.env_dialog = None
        self.settings = {
            "count": 200, "seed": 7, "engine": "project",
            "rate": 1_000_000.0, "duration": (0.2, 0.5), "signals": (0, 3),
            "modes": ["fm", "ssb", "ask2", "qpsk", "qam16", "qam64", "fh_rc", "fh_video"],
            "balanced": True, "snr": (-5.0, 30.0), "bandwidth_ratio": (0.02, 0.3),
            "power": (-12.0, -3.0), "hop": (10.0, 200.0),
            "detection": "per_hop_v1", "amc": True,
            "torchsig": {"generators": "all", "impairment": None, "mapping": ""},
        }
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "按生成参数批量合成 IQ 基带信号：每次生成写入一个集合，配方自动存到集合上保证可复现；"
            "标注按生成的信号参数自动写入（AMC = 调制方式；检测 = 起止时间 / 中心频率 / 带宽 / SNR，"
            "检测框在构建数据版本时才算）。")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        layout.addWidget(self._build_basic_group())
        layout.addWidget(self._build_signal_group())
        layout.addWidget(self._build_labels_group())
        layout.addWidget(self._build_impairment_group())
        layout.addWidget(self._build_action_group())
        self.preview = QtWidgets.QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(150)
        self.preview.setPlaceholderText("参数预览（只抽参数、不合成 IQ）：样本数、各参数分布与分层格子。")
        layout.addWidget(self.preview)
        self.result_label = QtWidgets.QLabel("尚未生成")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet(
            "background:white;border:1px solid #d8e1ec;border-radius:5px;padding:8px;")
        layout.addWidget(self.result_label)
        self.status = QtWidgets.QLabel("就绪")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._sync()

    # ------------------------------------------------------------------ 界面
    def _build_basic_group(self):
        group = QtWidgets.QGroupBox("基本")
        form = QtWidgets.QFormLayout(group)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        self.count = QtWidgets.QSpinBox()
        self.count.setRange(1, 10_000_000)
        self.count.setValue(int(self.settings["count"]))
        self.count.setToolTip("本次生成的录制条数（每条一个资产）")
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 1)  # Qt 的 int 上限；配方本身允许 0～2**32-1
        self.seed.setValue(int(self.settings["seed"]))
        self.seed.setToolTip("同一种子 + 同一样本序号 = 完全一致的数据（含逐信号种子派生）")
        box.addWidget(QtWidgets.QLabel("数量"))
        box.addWidget(self.count)
        box.addWidget(QtWidgets.QLabel("种子"))
        box.addWidget(self.seed)
        box.addStretch()
        form.addRow("", row)
        self.engine = QtWidgets.QComboBox()
        for label, value in ENGINE_CHOICES:
            self.engine.addItem(label, value)
        self.engine.setCurrentIndex(self.engine.findData(self.settings["engine"]))
        self.engine.currentIndexChanged.connect(self._engine_changed)
        form.addRow("生成引擎", self.engine)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        self.collection = QtWidgets.QComboBox()
        self.collection.setToolTip("生成后把资产加入所选集合；选“新建集合…”时填名称（同名沿用）")
        self.collection_name = QtWidgets.QLineEdit()
        self.collection_name.setPlaceholderText("新集合名称")
        self.collection_name.setVisible(False)
        self.collection.currentIndexChanged.connect(
            lambda *_: self.collection_name.setVisible(
                self.collection.currentData() == "__new__"))
        box.addWidget(self.collection, 1)
        box.addWidget(self.collection_name)
        form.addRow("目标集合", row)
        return group

    def _build_signal_group(self):
        group = QtWidgets.QGroupBox("信号与采样")
        row = QtWidgets.QHBoxLayout(group)
        self.signal_summary = QtWidgets.QLabel()
        self.signal_summary.setWordWrap(True)
        row.addWidget(self.signal_summary, 1)
        self.signal_button = QtWidgets.QPushButton("配置…")
        self.signal_button.clicked.connect(self.configure_signals)
        row.addWidget(self.signal_button)
        return group

    def _build_labels_group(self):
        group = QtWidgets.QGroupBox("标注（生成后自动写入）")
        row = QtWidgets.QHBoxLayout(group)
        self.detection = QtWidgets.QCheckBox("检测")
        self.detection.setChecked(self.settings["detection"] is not None)
        self.detection.setToolTip("检测标注 = 起止时间（采样点，显示时长）、中心频率、带宽、SNR；"
                                  "检测框在构建数据版本时才计算")
        self.detection.toggled.connect(self._sync)
        row.addWidget(self.detection)
        self.detection_scope = QtWidgets.QComboBox()
        for label, value in DETECTION_CHOICES:
            self.detection_scope.addItem(label, value)
        self.detection_scope.setCurrentIndex(
            max(self.detection_scope.findData(self.settings["detection"]), 0))
        self.detection_scope.setToolTip("默认逐跳：生成跳频样式时同时写会话与逐跳目标；"
                                        "TorchSig 没有跳频，只能选会话级")
        self.detection_scope.currentIndexChanged.connect(self._engine_changed)
        row.addWidget(self.detection_scope)
        self.amc = QtWidgets.QCheckBox("AMC（标注即调制方式，如 QPSK、16QAM）")
        self.amc.setChecked(bool(self.settings["amc"]))
        row.addWidget(self.amc)
        row.addStretch()
        return group

    def _build_impairment_group(self):
        """损伤（折叠）：未实现的项置灰，实现在 impairments 里完成后自动启用。"""
        group = QtWidgets.QGroupBox("损伤（实现后启用；接口已预留）")
        group.setCheckable(False)
        row = QtWidgets.QHBoxLayout(group)
        self.impairment_checks = {}
        for item in IMPAIRMENTS.values():
            check = QtWidgets.QCheckBox(item.label)
            check.setEnabled(item.implemented)
            check.setToolTip(item.reason if not item.implemented
                             else f"{item.summary}；施加顺序：{item.stage}")
            self.impairment_checks[item.name] = check
            row.addWidget(check)
        row.addWidget(QtWidgets.QLabel("（全部未实现，置灰；配方里用到会直接报错，不静默忽略）"), 1)
        return group

    def _build_action_group(self):
        group = QtWidgets.QGroupBox("操作")
        column = QtWidgets.QVBoxLayout(group)
        row = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("参数预览")
        self.preview_button.setToolTip("只抽参数、不合成 IQ：检查各参数分布与分层格子计数")
        self.preview_button.clicked.connect(self.preview_recipe)
        row.addWidget(self.preview_button)
        self.generate_button = QtWidgets.QPushButton("开始生成")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.start_generation)
        row.addWidget(self.generate_button)
        row.addStretch()
        column.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("高级"))
        load = QtWidgets.QPushButton("载入生成参数 JSON…")
        load.setToolTip("导入仅用于迁移或外部工具交接；日常在表单里配置")
        load.clicked.connect(self.load_recipe)
        row.addWidget(load)
        save = QtWidgets.QPushButton("导出生成参数 JSON…")
        save.clicked.connect(self.export_recipe)
        row.addWidget(save)
        self.bundle_button = QtWidgets.QPushButton("导入 TorchSig bundle…")
        self.bundle_button.setToolTip("任何平台可用：把 Linux 下生成的 bundle 导入为集合并建立目标/标注")
        self.bundle_button.clicked.connect(self.import_bundle)
        row.addWidget(self.bundle_button)
        self.env_button = QtWidgets.QPushButton("TorchSig 环境…")
        self.env_button.clicked.connect(self.open_environment)
        row.addWidget(self.env_button)
        row.addStretch()
        column.addLayout(row)
        return group

    def action_buttons(self):
        """需要随“任务运行中”禁用/恢复的按钮（主窗口统一管理）。"""
        return (self.preview_button, self.generate_button, self.bundle_button,
                self.signal_button)

    def refresh_collections(self):
        previous = self.collection.currentData()
        self.collection.blockSignals(True)
        self.collection.clear()
        self.collection.addItem("新建集合…", "__new__")
        for collection in self.window.workspace.list_collections():
            self.collection.addItem(f"{collection['name']}（{collection['asset_count']}）",
                                    collection["id"])
        index = self.collection.findData(previous) if previous else -1
        self.collection.setCurrentIndex(index if index >= 0 else 0)
        self.collection.blockSignals(False)
        self.collection_name.setVisible(self.collection.currentData() == "__new__")

    def _engine_changed(self, *_):
        torchsig = self.engine.currentData() == "torchsig"
        if torchsig:
            index = self.detection_scope.findData("session_v1")
            self.detection_scope.setCurrentIndex(index)
        self.detection_scope.setToolTip(
            "TorchSig 没有跳频，不能产生逐跳标注（固定会话级）" if torchsig
            else "默认逐跳：生成跳频样式时同时写会话与逐跳目标；非跳频样式仍是一个会话框")
        self._sync()
        self.status.setText("TorchSig 只能在 Linux 下生成；Windows 可用“导入 TorchSig bundle…”"
                            if torchsig and not is_linux() else "就绪")

    def configure_signals(self):
        dialog = SignalParamsDialog(self, self.settings, self.engine.currentData())
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            values = dialog.values()
            self.settings.update(values)
            self.settings["torchsig"]["mapping"] = values["torchsig_mapping"]
            self._sync()

    def open_environment(self):
        if self.env_dialog is None:
            self.env_dialog = TorchSigEnvDialog(self, self)
        self.env_dialog.show()

    def show_probe(self, result):
        if self.env_dialog is not None:
            self.env_dialog.show_probe(result)

    def _sync(self):
        """把表单/设置同步成摘要文案与配方一致性。"""
        settings = self.settings
        settings["count"] = self.count.value() if hasattr(self, "count") else settings["count"]
        settings["engine"] = self.engine.currentData()
        settings["detection"] = (self.detection_scope.currentData()
                                 if self.detection.isChecked() else None)
        settings["amc"] = self.amc.isChecked()
        modes = ", ".join(MODE_NAMES.get(key, key) for key in settings["modes"]) \
            or "（未选择）"
        balance = "类别均衡" if settings["balanced"] else "随机选择"
        parts = [f"每次 {_line(*settings['signals'], '个信号')}",
                 f"调制：{modes}（{balance}）",
                 f"SNR {_line(*settings['snr'], 'dB')}",
                 f"带宽占比 {settings['bandwidth_ratio'][0] * 100:g}%～"
                 f"{settings['bandwidth_ratio'][1] * 100:g}%",
                 f"功率 {_line(*settings['power'], 'dBFS')}",
                 f"采样率 {_line(*[settings['rate']] * 2, 'Hz')}",
                 f"时长 {_line(*settings['duration'], 's')}"]
        if settings["engine"] == "project":
            parts.append(f"跳速 {_line(*settings['hop'], 'hop/s')}")
        else:
            level = settings["torchsig"]["impairment"]
            parts.append(f"信号族 {settings['torchsig']['generators']}"
                         f" · 扰动 {level if level is not None else '无'}")
        self.signal_summary.setText(" · ".join(parts))

    # ------------------------------------------------------------------ 配方
    @staticmethod
    def _range(values):
        low, high = values
        return {"fixed": low} if low == high else {"uniform": [low, high]}

    @staticmethod
    def _int_range(values):
        low, high = int(values[0]), int(values[1])
        return {"fixed": low} if low == high else {"choice": [low, high]}

    def recipe(self):
        """表单 → 内部生成参数（``gen_recipe_v1``）。"""
        self._sync()
        settings = self.settings
        record = {"sample_rate_hz": {"fixed": float(settings["rate"])},
                  "duration_s": self._range(settings["duration"])}
        signals = {"count": self._int_range(settings["signals"]),
                   "snr_db": self._range(settings["snr"]),
                   "bandwidth_ratio": self._range(settings["bandwidth_ratio"])}
        labels = {"detection": settings["detection"], "amc": bool(settings["amc"])}
        recipe = {"contract": "gen_recipe_v1", "engine": settings["engine"],
                  "base_seed": int(self.seed.value()), "count": int(self.count.value()),
                  "record": record, "signals": signals, "labels": labels}
        if settings["engine"] == "project":
            signals["power_dbfs"] = self._range(settings["power"])
            if not settings["modes"]:
                raise ValueError("请至少选择一种调制类型（“配置…”里勾选）")
            key = "balanced" if settings["balanced"] else "choice"
            signals["mode"] = {key: list(settings["modes"])}
            recipe["hopping"] = {"hop_rate_hz": self._range(settings["hop"])}
        else:
            options = {"signal_generators": {"fixed": settings["torchsig"]["generators"]}}
            if settings["torchsig"]["impairment"] is not None:
                options["impairment_level"] = {"fixed": int(settings["torchsig"]["impairment"])}
            recipe["torchsig"] = options
            if not settings["torchsig"]["mapping"]:
                raise ValueError("TorchSig 引擎需要给出类别映射（“配置…”→“类别映射 JSON”）："
                                 "格式为 {TorchSig 类名: A09 类别} 的 JSON")
        return recipe

    def _collection_request(self):
        scope = self.collection.currentData()
        if scope is None:
            raise ValueError("请选择目标集合（新建或追加）")
        if scope == "__new__":
            name = self.collection_name.text().strip()
            if not name:
                raise ValueError("已选择“新建集合…”，请填写集合名称")
            return {"collection_name": name}
        return {"collection_id": scope}

    def _check(self, recipe):
        if recipe["engine"] == "torchsig" and not is_linux():
            raise ValueError("TorchSig 只能在 Linux 环境下使用；Windows 上可导入已生成的 "
                             "bundle（“高级 → 导入 TorchSig bundle…”）")
        validate_recipe(recipe)
        check_generator_support(recipe)

    def preview_recipe(self):
        try:
            recipe = self.recipe()
            self._check(recipe)
        except ValueError as exc:
            self.status.setText(f"生成参数无效：{exc}")
            return
        self.window.start_job("recipe_preview", recipe=recipe,
                              samples=min(int(recipe["count"]), 10_000))

    def start_generation(self):
        try:
            recipe = self.recipe()
            self._check(recipe)
            collection = self._collection_request()
        except ValueError as exc:
            self.status.setText(f"无法开始：{exc}")
            return
        request = {"recipe": recipe, **collection}
        if recipe["engine"] == "torchsig":
            mapping = load_mapping(self.settings["torchsig"]["mapping"])
            request["mapping"] = mapping
            request["torchsig_env"] = read_env(self.window.workspace)
        self.window.start_job("generate_collection", **request)

    def show_preview(self, result):
        lines = [f"样本数 {result['count']} · 引擎 {result['engine']} · "
                 f"分层格子 {result['cells']['total']}（低于 min_per_cell 的 "
                 f"{result['cells']['below_min_per_cell']} 个，min_per_cell="
                 f"{result['cells']['min_per_cell']}）"]
        for axis in result["axes"]:
            values = "、".join(f"{item['label']} {item['count']}"
                               for item in axis["values"][:10])
            lines.append(f"{axis['axis']}：{values}")
        for sample in result["preview"][:3]:
            pairs = list(sample.items())[:8]
            lines.append("示例：" + " · ".join(f"{key}={value}" for key, value in pairs))
        self.preview.setPlainText("\n".join(lines))

    def show_generation(self, result):
        lines = [f"请求 {result['requested']} 条 · 新建资产 {result['created']}"
                 f"（跳过 {result.get('failed', 0)}）· 分片 {result.get('shards', 0)}"
                 f" · 用时 {result.get('elapsed_s', 0):g} s"]
        if result.get("stopped") == "cancelled":
            lines[0] = "已取消：" + lines[0].replace("新建资产", "已保留资产")
        elif result.get("stopped") == "time_limit":
            lines[0] = "达到时间上限：" + lines[0]
        lines.append(f"目标集合：{result['collection_name']}"
                     + (f" · 配方 {result['recipe_id'][:8]}" if result.get("recipe_id") else ""))
        targets = (f"目标参考参数：会话 {result.get('sessions', 0)}"
                   + (f" · 逐跳 {result.get('hops')}" if result.get("hops") else ""))
        lines.append(targets)
        labels = result.get("labels") or {}
        if labels:
            lines.append(f"初始标注：检测 {labels.get('detection', 0)} · "
                         f"AMC {labels.get('amc', 0)} 条")
        if result.get("unmapped"):
            extra = "、".join(f"{name}×{count}" for name, count in result["unmapped"].items())
            lines.append(f"类别映射之外（AMC 记为字典外）：{extra}")
        for item in result.get("failure_reasons") or []:
            lines.append(f"跳过 {item['count']} 条：{item['reason']}")
        self.result_label.setText("\n".join(lines))

    # ------------------------------------------------------------------ 高级
    def load_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "载入生成参数 JSON", "",
                                                        "生成参数 JSON (*.json)")
        if not path:
            return
        try:
            recipe = json.loads(Path(path).read_text(encoding="utf-8"))
            validate_recipe(recipe)
            check_generator_support(recipe)
        except (OSError, ValueError) as exc:
            self.status.setText(f"生成参数无效：{exc}")
            return
        self._apply_recipe(recipe)
        self.status.setText(f"已载入 {Path(path).name}（可在表单里继续调整）")

    def _apply_recipe(self, recipe):
        settings = self.settings
        settings["engine"] = recipe["engine"]
        self.engine.setCurrentIndex(max(self.engine.findData(recipe["engine"]), 0))
        self.seed.setValue(int(recipe["base_seed"]))
        self.count.setValue(int(recipe["count"]))
        record = recipe.get("record") or {}

        def _pair(node, fallback):
            if not isinstance(node, dict):
                return fallback
            if "fixed" in node:
                return (float(node["fixed"]), float(node["fixed"]))
            for key in ("uniform", "loguniform"):
                if key in node:
                    low, high = node[key]
                    return (float(low), float(high))
            values = node.get("choice") or node.get("balanced")
            if values:
                return (min(values), max(values))
            return fallback

        settings["rate"] = _pair(record.get("sample_rate_hz"),
                                 (settings["rate"], settings["rate"]))[0]
        settings["duration"] = _pair(record.get("duration_s"), settings["duration"])
        signals = recipe.get("signals") or {}
        settings["signals"] = _pair(signals.get("count"), settings["signals"])
        settings["snr"] = _pair(signals.get("snr_db"), settings["snr"])
        settings["bandwidth_ratio"] = _pair(signals.get("bandwidth_ratio"),
                                            settings["bandwidth_ratio"])
        settings["power"] = _pair(signals.get("power_dbfs"), settings["power"])
        settings["hop"] = _pair((recipe.get("hopping") or {}).get("hop_rate_hz"),
                                settings["hop"])
        mode_node = signals.get("mode") or {}
        if mode_node.get("balanced"):
            settings["modes"] = list(mode_node["balanced"])
            settings["balanced"] = True
        elif mode_node.get("choice"):
            settings["modes"] = list(mode_node["choice"])
            settings["balanced"] = False
        labels = recipe.get("labels") or {}
        settings["detection"] = labels.get("detection")
        settings["amc"] = bool(labels.get("amc"))
        options = recipe.get("torchsig") or {}
        node = options.get("signal_generators")
        settings["torchsig"]["generators"] = (node or {}).get("fixed", "all")
        level = (options.get("impairment_level") or {}).get("fixed")
        settings["torchsig"]["impairment"] = int(level) if level is not None else None
        self.detection.setChecked(settings["detection"] is not None)
        if settings["detection"] is not None:
            self.detection_scope.setCurrentIndex(
                max(self.detection_scope.findData(settings["detection"]), 0))
        self.amc.setChecked(settings["amc"])
        self._sync()

    def export_recipe(self):
        try:
            recipe = self.recipe()
            self._check(recipe)
        except ValueError as exc:
            self.status.setText(f"生成参数无效：{exc}")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出生成参数 JSON",
                                                        "生成参数.json", "JSON (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(recipe, ensure_ascii=False, indent=2, sort_keys=True),
                              encoding="utf-8")
        self.status.setText(f"生成参数已导出：{path}（JSON 仅用于迁移/交接，日常在表单里配置）")

    def import_bundle(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择 torchsig_bundle_v1 清单", "", "bundle 清单 (manifest.json)")
        if not path:
            return
        bundle = Path(path).parent
        dialog = _BundleImportDialog(self, bundle)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        try:
            request = dialog.request()
        except ValueError as exc:
            self.status.setText(f"无法导入：{exc}")
            return
        self.window.start_job("torchsig_import", **request)


class _BundleImportDialog(QtWidgets.QDialog):
    """导入 TorchSig bundle：目标集合 + 类别映射 + 标注勾选。"""

    def __init__(self, parent, bundle):
        super().__init__(parent)
        self.bundle = bundle
        self.setWindowTitle(f"导入 TorchSig bundle · {bundle.name}")
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)
        self.collection = QtWidgets.QComboBox()
        self.collection.addItem("新建集合…", "__new__")
        for collection in parent.window.workspace.list_collections():
            self.collection.addItem(f"{collection['name']}（{collection['asset_count']}）",
                                    collection["id"])
        self.collection_name = QtWidgets.QLineEdit(bundle.name)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.collection, 1)
        box.addWidget(self.collection_name)
        form.addRow("目标集合", row)
        self.collection.currentIndexChanged.connect(
            lambda *_: self.collection_name.setVisible(self.collection.currentData() == "__new__"))
        self.collection_name.setVisible(self.collection.currentData() == "__new__")
        self.mapping = QtWidgets.QLineEdit(parent.settings["torchsig"]["mapping"])
        choose = QtWidgets.QPushButton("选择…")
        choose.clicked.connect(self._choose_mapping)
        row = QtWidgets.QWidget()
        box = QtWidgets.QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.mapping, 1)
        box.addWidget(choose)
        form.addRow("类别映射 JSON", row)
        self.detection = QtWidgets.QCheckBox("检测（会话级；TorchSig 没有跳频）")
        self.detection.setChecked(True)
        self.amc = QtWidgets.QCheckBox("AMC（映射到 A09；映射不到的记字典外）")
        self.amc.setChecked(True)
        form.addRow("", self.detection)
        form.addRow("", self.amc)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_mapping(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "类别映射 JSON", "",
                                                        "JSON (*.json)")
        if path:
            self.mapping.setText(path)

    def request(self):
        scope = self.collection.currentData()
        request = {"bundle_path": str(self.bundle),
                   "detection": self.detection.isChecked(), "amc": self.amc.isChecked()}
        if scope == "__new__":
            name = self.collection_name.text().strip()
            if not name:
                raise ValueError("请填写新集合名称")
            request["collection_name"] = name
        else:
            request["collection_id"] = scope
        text = self.mapping.text().strip()
        if text:
            request["mapping_path"] = text
            request["mapping"] = load_mapping(text)
        return request
