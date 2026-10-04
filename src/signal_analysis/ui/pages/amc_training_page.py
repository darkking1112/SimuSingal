"""AMC (IQ waveform) training page: collection labels plus external training.

训练数据直接取自信号集合（拆分设计第 10 节）。「信号标注」子页查看并修改**当前集合**内
左侧选中资产的标注：调制类别与目标参考参数（时间范围／频率范围／SNR／调制）。
"""
import json

from PySide6 import QtCore, QtWidgets

from common.gui import direct_entry
from ...contracts.iq import (CLASS_SET_A09, DEFAULT_IQ_SAMPLES, IQ_INPUT_CHANNELS,
                             IQ_NORMALIZATION)
from ...data.targets import carryover_fields
from ...services.training_inputs import collection_task_set
from ...services.training_jobs import iq_tuning_args
from .training_common import TrainingPageBase

#: AMC 类别状态：与数据层 ``CLASS_STATES`` 一致。
CLASS_STATES = (("known", "字典内类别"), ("unknown", "未知调制"),
                ("out_of_taxonomy", "字典外类别"))
COLUMNS = ("目标", "频率范围", "时间范围", "SNR", "调制", "类别状态", "当前类别")


class AmcTrainingPage(TrainingPageBase):
    TASK = "iq"
    OWNER = "AMC 识别训练"
    TITLE = "AMC 识别训练"
    MODEL_OPTIONS = ("cnn", "tcn")
    HISTORY_TASKS = ("iq",)
    HAS_ANNOTATIONS = True
    ANNOTATION_TITLE = "信号标注"

    def page_note(self):
        return ("训练数据取自信号集合：选择训练集与验证集集合即可，页内不生成数据、"
                "不导出训练集。“信号标注”查看并修改当前集合内左侧选中资产的参数与调制类别。")

    def curve_title(self):
        return "AMC 训练 loss / IQ 验证准确率（0–1）"

    def add_task_fields(self, form):
        self.iq_summary = QtWidgets.QLabel()
        self.iq_summary.setWordWrap(True)
        self.iq_summary.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("IQ 数据契约（只读）", self.iq_summary)

        # 高级训练参数：默认关闭 = 用 train_iq.py 的默认结构超参；勾选后才写进配置
        self.advanced_toggle = QtWidgets.QCheckBox("覆盖默认训练参数（高级）")
        self.advanced_toggle.setToolTip(
            "默认关闭：结构超参用 train_iq.py 的默认值（cnn 32,64,128 / 核长 7；"
            "tcn 64 / 核长 3、dropout 0.1、权重衰减 1e-4、早停 8 轮）。勾选后下列字段"
            "才写入训练配置；结构超参没有搜索证据，改动即视为新的实验口径，需要重新验收。")
        self.advanced_toggle.toggled.connect(self._sync_advanced_enabled)
        form.addRow(self.advanced_toggle)

        self.iq_channels = QtWidgets.QLineEdit()
        self.iq_channels.setPlaceholderText("留空 = 默认；cnn 需 3 个宽度（如 64,128,256），tcn 只用第 1 个")
        form.addRow("卷积通道", self.iq_channels)

        self.iq_kernel = QtWidgets.QSpinBox()
        self.iq_kernel.setRange(0, 65)
        self.iq_kernel.setSpecialValueText("按结构默认（cnn 7 / tcn 3）")
        self.iq_kernel.setToolTip("cnn 需不小于 3 的奇数；tcn 为正整数")
        form.addRow("卷积核长", self.iq_kernel)

        self.iq_dropout = QtWidgets.QDoubleSpinBox()
        self.iq_dropout.setRange(0.0, 0.9)
        self.iq_dropout.setDecimals(2)
        self.iq_dropout.setSingleStep(0.05)
        self.iq_dropout.setValue(0.10)
        form.addRow("Dropout", self.iq_dropout)

        self.iq_weight_decay = QtWidgets.QDoubleSpinBox()
        self.iq_weight_decay.setRange(0.0, 1.0)
        self.iq_weight_decay.setDecimals(7)
        self.iq_weight_decay.setSingleStep(0.0001)
        self.iq_weight_decay.setValue(0.0001)
        direct_entry(self.iq_weight_decay)
        form.addRow("权重衰减", self.iq_weight_decay)

        self.iq_patience = QtWidgets.QSpinBox()
        self.iq_patience.setRange(1, 1000)
        self.iq_patience.setValue(8)
        self.iq_patience.setToolTip("验证准确率连续多少轮不提升就早停（默认 8）")
        form.addRow("早停轮数", self.iq_patience)

        self._advanced_fields = (self.iq_channels, self.iq_kernel, self.iq_dropout,
                                 self.iq_weight_decay, self.iq_patience)
        self._sync_advanced_enabled(False)

    def _sync_advanced_enabled(self, enabled):
        """未勾选"覆盖默认训练参数"时字段置灰（配置里也不出现这些键）。"""
        for field in getattr(self, "_advanced_fields", ()):
            field.setEnabled(bool(enabled))

    def configuration(self):
        config = super().configuration()
        if self.advanced_toggle.isChecked():
            text = self.iq_channels.text().strip()
            if text:
                config["channels"] = text
            if self.iq_kernel.value() > 0:
                config["kernel"] = int(self.iq_kernel.value())
            config["dropout"] = float(self.iq_dropout.value())
            config["weight_decay"] = float(self.iq_weight_decay.value())
            config["patience"] = int(self.iq_patience.value())
        return config

    def validate_task_config(self, config):
        classes = json.loads(self.window.workspace.get_taxonomy(
            self.inputs_plan["train"]["task_set"]["taxonomy_id"])["classes_json"])
        if len(classes) < 2:
            raise ValueError("AMC 训练至少需要 2 个类别：请检查训练集的类别字典")
        # 高级训练参数在这里先校验一遍（iq_plan 会在起任务前复核同一份逻辑）
        iq_tuning_args(config, config.get("arch", "cnn"))

    # ------------------------------------------------------------------ 摘要
    def update_data_summary(self):
        if not hasattr(self, "iq_summary"):
            return
        classes = list(CLASS_SET_A09)
        source = "未选择集合：类别字典显示 A09 默认值"
        collection_id = self.window.current_collection_id()
        if collection_id is not None:
            try:
                task_set = collection_task_set(self.window.workspace, collection_id, "amc")
            except ValueError as exc:
                source = str(exc)
            else:
                classes = json.loads(self.window.workspace.get_taxonomy(
                    task_set["taxonomy_id"])["classes_json"])
                source = f"类别字典来自集合『{task_set['name']}』的 AMC 标注集"
        self.iq_summary.setText(
            f"窗口长度：{DEFAULT_IQ_SAMPLES} samples · 通道：{IQ_INPUT_CHANNELS}"
            f" · 归一化：{IQ_NORMALIZATION}\n类别顺序：{' / '.join(map(str, classes))}\n{source}")
        if hasattr(self, "target_table"):
            self._reload_annotation()

    def sync_train_collection(self, collection_id):
        super().sync_train_collection(collection_id)
        if hasattr(self, "target_table"):
            self._reload_annotation()

    def _reload_annotation(self):
        """按当前上下文重画标注表格：先保存未保存的修改，保存失败则保留现有内容。"""
        if self.dirty and not self.save_annotation():
            return False
        self.refresh_annotation()
        return True

    # -------------------------------------------------------------- 信号标注
    def build_annotations(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        header = QtWidgets.QHBoxLayout()
        self.asset_label = QtWidgets.QLabel("在左侧选择一个数据资产")
        self.asset_label.setWordWrap(True)
        header.addWidget(self.asset_label, 1)
        self.save_button = QtWidgets.QPushButton("保存当前标注")
        self.save_button.clicked.connect(self.save_annotation)
        header.addWidget(self.save_button)
        layout.addLayout(header)

        self.target_table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.target_table.setHorizontalHeaderLabels(COLUMNS)
        self.target_table.verticalHeader().setVisible(False)
        self.target_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.target_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.target_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.target_table.itemSelectionChanged.connect(self._target_selection_changed)
        self.target_table.setMaximumHeight(200)
        layout.addWidget(self.target_table)

        form = QtWidgets.QFormLayout()
        self.class_state = QtWidgets.QComboBox()
        for value, text in CLASS_STATES:
            self.class_state.addItem(text, value)
        form.addRow("类别状态", self.class_state)
        self.class_name = QtWidgets.QComboBox()
        self.class_name.setEditable(True)
        form.addRow("调制类别", self.class_name)
        self.f_low = QtWidgets.QLineEdit()
        self.f_high = QtWidgets.QLineEdit()
        for field, title, hint in ((self.f_low, "频率下限（Hz）", "留空表示未提供"),
                                   (self.f_high, "频率上限（Hz）", "留空表示未提供")):
            field.setPlaceholderText(hint)
            form.addRow(title, field)
        self.sample_start = QtWidgets.QLineEdit()
        self.sample_end = QtWidgets.QLineEdit()
        for field, title in ((self.sample_start, "起始采样点"), (self.sample_end, "结束采样点")):
            field.setPlaceholderText("留空表示未提供")
            form.addRow(title, field)
        self.snr = QtWidgets.QLineEdit()
        self.snr.setPlaceholderText("留空表示未提供")
        form.addRow("带内 SNR（dB）", self.snr)
        self.modulation = QtWidgets.QLineEdit()
        self.modulation.setPlaceholderText("留空表示未提供，如 QPSK")
        form.addRow("原始调制标注", self.modulation)
        layout.addLayout(form)

        self.annotation_status = QtWidgets.QLabel(
            "选择目标后修改类别与参数；“保存当前标注”以追加新版本的方式写入，历史版本保留。")
        self.annotation_status.setWordWrap(True)
        layout.addWidget(self.annotation_status)
        layout.addStretch(1)

        for signal in (self.class_state.currentIndexChanged, self.class_name.editTextChanged,
                       self.f_low.textChanged, self.f_high.textChanged,
                       self.sample_start.textChanged, self.sample_end.textChanged,
                       self.snr.textChanged, self.modulation.textChanged):
            signal.connect(self.mark_dirty)

        self._targets = []
        self._task_set_id = None
        self._asset_id = None
        self._loaded_target_id = None
        self._loading = False
        self._switching_target = False
        self._switching_asset = False
        self.dirty = False
        self.window.assets.currentItemChanged.connect(self._asset_selection_changed)
        return widget

    # ------------------------------------------------------------ 选中与切换
    def _asset_selection_changed(self, current, previous):
        """切换资产前先保存标注；保存失败则留在原资产（第 11 节）。"""
        if self._switching_asset or self._loading:
            return
        asset = current.data(QtCore.Qt.ItemDataRole.UserRole) if current else None
        if asset is not None and asset["id"] == self._asset_id:
            return  # 列表刷新重选同一资产：保留未保存修改，不重画
        if self.dirty and not self.save_annotation():
            asset = previous.data(QtCore.Qt.ItemDataRole.UserRole) if previous else None
            reason = self.annotation_status.text()
            self._switching_asset = True
            try:
                restored = bool(asset) and self.window.select_asset(asset["id"])
            finally:
                self._switching_asset = False
            self.annotation_status.setText(
                f"{reason}；未切换资产，修正后重新保存" if restored else
                f"{reason}；原资产已不在左侧列表中，请重新选择该资产后再保存")
            return
        self.refresh_annotation()

    def _target_selection_changed(self):
        """切换目标前先保存；保存失败则把选中行恢复为原目标，避免静默丢弃修改。"""
        if self._loading or self._switching_target:
            return
        target = self.selected_target()
        if target is not None and target["id"] == self._loaded_target_id:
            return
        if self.dirty and not self.save_annotation():
            reason = self.annotation_status.text()
            row = next((index for index, item in enumerate(self._targets)
                        if item["id"] == self._loaded_target_id), -1)
            self._switching_target = True
            try:
                if row >= 0:
                    self.target_table.selectRow(row)
            finally:
                self._switching_target = False
            self.annotation_status.setText(
                f"{reason}；未切换目标，修正后重新保存" if row >= 0
                else f"{reason}；请重新选择目标后再保存")
            return
        self.load_target()

    def refresh_annotation(self, *_args, select_id=None):
        if not hasattr(self, "target_table"):
            return
        collection_id = self.window.current_collection_id()
        asset = self.window.selected_asset()
        self._loading = True
        self.target_table.setRowCount(0)
        self._targets = []
        self._task_set_id = None
        self._asset_id = asset["id"] if asset else None
        self._loaded_target_id = None
        self.dirty = False
        self._clear_form()
        if collection_id is None or asset is None:
            self.asset_label.setText("在左侧选择一个集合与资产" if collection_id is None
                                     else "在左侧选择一个数据资产")
            self.annotation_status.setText("")
            self._loading = False
            return
        try:
            task_set = collection_task_set(self.window.workspace, collection_id, "amc")
        except ValueError as exc:
            self.asset_label.setText(f"{asset['name']} · 当前集合未启用 AMC 标注")
            self.annotation_status.setText(str(exc))
            self._loading = False
            return
        self._task_set_id = task_set["id"]
        workspace = self.window.workspace
        targets = [target for target in
                   workspace.list_targets(asset["id"], with_current=True)
                   if target["for_amc"]]
        self._targets = targets
        self.target_table.setRowCount(len(targets))
        for row, target in enumerate(targets):
            target["label"] = workspace.current_label("amc", task_set["id"], target["id"])
            for column, text in enumerate(self._row_text(target)):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                              | QtCore.Qt.ItemFlag.ItemIsSelectable)
                self.target_table.setItem(row, column, item)
        self.asset_label.setText(f"{asset['name']} · 集合『{task_set['name']}』")
        pending = sum(1 for target in targets if target["label"] is None)
        self.annotation_status.setText(
            f"适用 AMC 的目标 {len(targets)} 个" + (
                f" · 待标注 {pending} 个" if targets else
                "；该资产没有适用 AMC 的目标"))
        self._loading = False
        row = next((index for index, target in enumerate(targets)
                    if target["id"] == select_id), 0 if targets else -1)
        if row >= 0:
            self.target_table.selectRow(row)

    def _row_text(self, target):
        version = target["current"] or {}
        label = target["label"] or {}
        band = ("未提供" if version.get("f_low_hz") is None else
                f"{version['f_low_hz']:.4g} ~ {version['f_high_hz']:.4g} Hz")
        times = ("未提供" if version.get("sample_start") is None else
                 f"{int(version['sample_start'])} ~ {int(version['sample_end'])} 采样点")
        snr = "未提供" if version.get("snr_db") is None else f"{version['snr_db']:.2f} dB"
        state = dict(CLASS_STATES).get(label.get("class_state"), "未标注")
        return (f"{target['target_key']}（{target['scope']}）", band, times, snr,
                version.get("modulation") or "未提供", state,
                label.get("class_name") or "未标注")

    def load_target(self):
        target = self.selected_target()
        if target is None:
            return
        version = target["current"] or {}
        label = target["label"] or {}
        self._loading = True
        self._fill_class_choices(label.get("class_name"))
        index = self.class_state.findData(label.get("class_state") or "unknown")
        self.class_state.setCurrentIndex(max(0, index))
        self.f_low.setText(self._format(version.get("f_low_hz")))
        self.f_high.setText(self._format(version.get("f_high_hz")))
        self.sample_start.setText(self._format(version.get("sample_start")))
        self.sample_end.setText(self._format(version.get("sample_end")))
        self.snr.setText(self._format(version.get("snr_db")))
        self.modulation.setText(version.get("modulation") or "")
        self._loading = False
        self.dirty = False
        self._loaded_target_id = target["id"]

    def _fill_class_choices(self, current):
        classes = []
        if self._task_set_id is not None:
            task_set = self.window.workspace.get_task_set(self._task_set_id)
            classes = json.loads(self.window.workspace.get_taxonomy(
                task_set["taxonomy_id"])["classes_json"])
        self.class_name.blockSignals(True)
        self.class_name.clear()
        self.class_name.addItems([str(name) for name in classes])
        self.class_name.setCurrentText(str(current or ""))
        self.class_name.blockSignals(False)

    def _clear_form(self):
        self._loading = True
        self.class_name.clear()
        for field in (self.f_low, self.f_high, self.sample_start, self.sample_end,
                      self.snr, self.modulation):
            field.clear()
        self.class_state.setCurrentIndex(
            max(0, self.class_state.findData("unknown")))
        self._loading = False

    @staticmethod
    def _format(value):
        return "" if value is None else f"{value:g}"

    def selected_target(self):
        row = self.target_table.currentRow() if hasattr(self, "target_table") else -1
        if row < 0 or row >= len(self._targets):
            return None
        return self._targets[row]

    def loaded_target(self):
        """表单当前承载的目标：保存/校验都以它为准，而不是新选中的行。"""
        return next((target for target in self._targets
                     if target["id"] == self._loaded_target_id), None)

    def mark_dirty(self, *_):
        if self._loading:
            return
        self.dirty = True
        self.annotation_status.setText("标注已修改，尚未保存")

    def save_if_dirty(self):
        return self.save_annotation() if self.dirty else True

    def save_annotation(self):
        target = self.loaded_target()
        if target is None or self._task_set_id is None:
            return True
        if not self.dirty:
            return True
        try:
            self._write_target(target)
        except (ValueError, TypeError) as exc:
            self.annotation_status.setText(f"保存失败：{exc}")
            self.report_error(f"保存失败：{exc}")
            return False
        self.dirty = False
        self.refresh_annotation(select_id=target["id"])
        self.annotation_status.setText("已保存 · 类别与参数已追加新版本")
        self.window.asset_changed()
        return True

    def _write_target(self, target):
        """参数版本与 AMC 标签同一事务写入：校验失败不留下半成品（第 11 节）。"""
        workspace = self.window.workspace
        # 表单只编辑这 6 个字段，其余（跳频标记、波形模式、符号率、功率等）沿用旧版本
        carried = carryover_fields(target["current"],
                                  exclude=("f_low_hz", "f_high_hz", "sample_start",
                                           "sample_end", "snr_db", "modulation"))
        workspace.append_amc_annotation(
            self._task_set_id, target["id"], source="manual",
            class_state=self.class_state.currentData(),
            class_name=(self.class_name.currentText().strip() or None),
            sample_start=self._int_or_none(self.sample_start, "起始采样点"),
            sample_end=self._int_or_none(self.sample_end, "结束采样点"),
            f_low_hz=self._float_or_none(self.f_low, "频率下限"),
            f_high_hz=self._float_or_none(self.f_high, "频率上限"),
            snr_db=self._float_or_none(self.snr, "带内 SNR"),
            modulation=(self.modulation.text().strip() or None),
            **carried)

    @staticmethod
    def _int_or_none(field, title):
        text = field.text().strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{title}应为整数：{text}") from exc

    @staticmethod
    def _float_or_none(field, title):
        text = field.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{title}应为数值：{text}") from exc
