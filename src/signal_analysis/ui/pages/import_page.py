"""信号导入页（mixin）：文件清单、批量参数、CSV 标注清单与导入执行。"""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ..constants import (EXPORT_FORMATS, IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NOTE, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_RF_CENTER, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..dialogs import ImportBatchDialog, SignalParamsDialog
from ..helpers import (_asset_exports, _asset_format, _fmt_hz, _fmt_metric, _fmt_span,
                       _iq_binary_kind, _mirrored_spectrum)
from ..runner import _run_task
from ..widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class ImportPageMixin:
    def build_import(self):
        """独立导入页（方案 §7.1，2026-10 改版）：以文件清单为中心。

        每个文件在清单里保留自己的解析参数（采样率/类型/字节序）、调制与备注；
        缺参数的单元格标红、状态不为“就绪”的行不参与导入。成批参数用“批量设置”，
        成批真值用“导入标注清单 CSV”（按文件名匹配，未匹配行原样列出）。
        """
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "批量导入离线 IQ：先添加文件或文件夹，自动识别格式与已知参数；"
            "缺采样率/类型/字节序的行标红、不参与导入。SigMF 自动读采样率，"
            "二进制需显式给出类型与字节序。目标真值可用“导入标注清单 CSV…”集中申报。")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        from common.gui import TaskBanner
        self.import_banner = TaskBanner("信号导入", "取消本次导入")
        self.register_task_banner("信号导入", self.import_banner)
        layout.addWidget(self.import_banner)

        toolbar = QtWidgets.QHBoxLayout()
        self.import_add_files_button = QtWidgets.QPushButton("添加文件…")
        self.import_add_files_button.clicked.connect(self.choose_import_files)
        toolbar.addWidget(self.import_add_files_button)
        self.import_add_folder_button = QtWidgets.QPushButton("添加文件夹…")
        self.import_add_folder_button.setToolTip(
            "递归收集 .npy/.csv/.bin/.raw/.iq 与 SigMF 元数据文件（SigMF 数据文件自动去重）")
        self.import_add_folder_button.clicked.connect(self.choose_import_folder)
        toolbar.addWidget(self.import_add_folder_button)
        self.import_remove_button = QtWidgets.QPushButton("移除选中")
        self.import_remove_button.clicked.connect(self.remove_import_rows)
        toolbar.addWidget(self.import_remove_button)
        self.import_batch_set_button = QtWidgets.QPushButton("批量设置…")
        self.import_batch_set_button.setToolTip("对选中行统一填采样率/类型/字节序；SigMF 行采样率只读")
        self.import_batch_set_button.clicked.connect(self.batch_set_import_params)
        toolbar.addWidget(self.import_batch_set_button)
        self.import_recheck_button = QtWidgets.QPushButton("重新识别")
        self.import_recheck_button.setToolTip("对选中文件重读头部/元数据（文件被外部替换后使用）")
        self.import_recheck_button.clicked.connect(self.recheck_import_rows)
        toolbar.addWidget(self.import_recheck_button)
        self.import_csv_button = QtWidgets.QPushButton("导入标注清单 CSV…")
        self.import_csv_button.clicked.connect(self.import_csv_manifest)
        toolbar.addWidget(self.import_csv_button)
        self.import_csv_template_button = QtWidgets.QPushButton("导出模板")
        self.import_csv_template_button.clicked.connect(self.export_import_template)
        toolbar.addWidget(self.import_csv_template_button)
        toolbar.addStretch(1)
        self.import_count_label = QtWidgets.QLabel("清单为空")
        toolbar.addWidget(self.import_count_label)
        layout.addLayout(toolbar)

        self.import_table = self._make_table(
            ["文件名", "格式", "采样率", "类型", "字节序", "点数/时长", "射频中心",
             "调制", "备注", "状态"],
            editable=True)
        self.import_table.setMinimumHeight(220)
        self.import_table.itemChanged.connect(self._import_item_changed)
        layout.addWidget(self.import_table, 1)

        record_group = QtWidgets.QGroupBox("归属与录制信息（应用到本批）")
        record_row = QtWidgets.QHBoxLayout(record_group)
        record_row.addWidget(QtWidgets.QLabel("目标集合"))
        self.import_collection = QtWidgets.QComboBox()
        self.import_collection.addItem("不加入集合", None)
        self.import_collection.addItem("新建集合…", "__new__")
        self.import_collection.currentIndexChanged.connect(
            lambda *_: self._import_collection_changed())
        record_row.addWidget(self.import_collection, 1)
        self.import_collection_name = QtWidgets.QLineEdit()
        self.import_collection_name.setPlaceholderText("新集合名称")
        self.import_collection_name.setVisible(False)
        record_row.addWidget(self.import_collection_name)
        self.import_initial_labels = QtWidgets.QCheckBox("同时生成初始标注（来源 import）")
        self.import_initial_labels.setToolTip(
            "按目标参考参数为新资产写检测/AMC 初始标签；集合无标注集时自动建立")
        record_row.addWidget(self.import_initial_labels)
        record_row.addWidget(QtWidgets.QLabel("射频中心（默认）"))
        self.import_rf_center = QtWidgets.QLineEdit()
        self.import_rf_center.setPlaceholderText("Hz，本批默认；行内可覆盖")
        self.import_rf_center.setToolTip("仅用于清单中未单独填写射频中心的文件；每行可覆盖")
        self.import_rf_center.setMaximumWidth(150)
        record_row.addWidget(self.import_rf_center)
        layout.addWidget(record_group)

        params_bar = QtWidgets.QHBoxLayout()
        self.import_params_toggle = QtWidgets.QToolButton()
        self.import_params_toggle.setText("信号参数（可选）")
        self.import_params_toggle.setCheckable(True)
        self.import_params_toggle.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.import_params_toggle.toggled.connect(self._toggle_import_params)
        params_bar.addWidget(self.import_params_toggle)
        params_bar.addWidget(QtWidgets.QLabel(
            "默认不建目标（未知待标注）；多目标请用 CSV 标注清单"), 1)
        layout.addLayout(params_bar)

        self.import_params_body = QtWidgets.QGroupBox("整条为单一信号（应用到所选文件）")
        self.import_params_body.setVisible(False)
        params_layout = QtWidgets.QHBoxLayout(self.import_params_body)
        params_layout.addWidget(QtWidgets.QLabel("调制"))
        self.import_single_modulation = QtWidgets.QComboBox()
        self.import_single_modulation.addItems(list(IMPORT_MODULATION_CHOICES))
        self.import_single_modulation.setEditable(True)
        self.import_single_modulation.setInsertPolicy(
            QtWidgets.QComboBox.InsertPolicy.NoInsert)
        params_layout.addWidget(self.import_single_modulation)
        for title, name in (("频率下限 Hz", "import_single_f_low"),
                            ("频率上限 Hz", "import_single_f_high"),
                            ("SNR dB", "import_single_snr")):
            params_layout.addWidget(QtWidgets.QLabel(title))
            field = QtWidgets.QLineEdit()
            field.setMaximumWidth(110)
            setattr(self, name, field)
            params_layout.addWidget(field)
        self.import_apply_single_button = QtWidgets.QPushButton("应用到所选文件")
        self.import_apply_single_button.clicked.connect(self.apply_single_signal_targets)
        params_layout.addWidget(self.import_apply_single_button)
        params_layout.addStretch(1)
        layout.addWidget(self.import_params_body)

        advanced = QtWidgets.QGroupBox("高级")
        advanced_row = QtWidgets.QHBoxLayout(advanced)
        advanced_row.addWidget(QtWidgets.QLabel("写入方式"))
        self.import_shard_mode = QtWidgets.QComboBox()
        self.import_shard_mode.addItem("自动（按文件数）", "auto")
        self.import_shard_mode.addItem("强制独立文件（每条一个 NPY）", "file")
        self.import_shard_mode.addItem("强制写入分片", "shard")
        advanced_row.addWidget(self.import_shard_mode)
        advanced_row.addWidget(QtWidgets.QLabel("自动阈值"))
        self.import_shard_threshold = QtWidgets.QSpinBox()
        self.import_shard_threshold.setRange(2, 500)
        self.import_shard_threshold.setValue(20)
        self.import_shard_threshold.setSuffix(" 个文件")
        self.import_shard_threshold.setToolTip("文件数达到阈值时自动写入分片（assets/shards/）")
        self.import_shard_threshold.valueChanged.connect(
            lambda *_: self._update_import_controls())
        advanced_row.addWidget(self.import_shard_threshold)
        advanced_row.addStretch(1)
        layout.addWidget(advanced)

        action_bar = QtWidgets.QHBoxLayout()
        self.import_start_button = QtWidgets.QPushButton("开始导入")
        self.import_start_button.setObjectName("primary")
        self.import_start_button.setEnabled(False)
        self.import_start_button.clicked.connect(self.start_import_batch)
        action_bar.addWidget(self.import_start_button)
        self.import_manage_button = QtWidgets.QPushButton("去数据管理查看")
        self.import_manage_button.setVisible(False)
        self.import_manage_button.clicked.connect(
            lambda: self.tabs.setCurrentIndex(self._page_index("数据管理")))
        action_bar.addWidget(self.import_manage_button)
        self.import_status = QtWidgets.QLabel("")
        self.import_status.setWordWrap(True)
        action_bar.addWidget(self.import_status, 1)
        layout.addLayout(action_bar)
        self.import_csv_report = QtWidgets.QLabel("")
        self.import_csv_report.setWordWrap(True)
        layout.addWidget(self.import_csv_report)

        self._import_filling = False
        return box

    # ------------------------------------------------------------------ 文件清单
    def choose_import_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "选择要导入的 IQ 数据", "", IMPORT_FILE_FILTER)
        self._add_import_paths(list(paths))

    def choose_import_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择要导入的文件夹")
        if not folder:
            return
        found = sorted(str(item) for item in Path(folder).rglob("*")
                       if item.is_file() and item.suffix.lower() in IMPORT_SUFFIXES)
        # SigMF 双文件按一条记录算：有同名 .sigmf-meta 时忽略 .sigmf-data
        metas = {item[:-11] for item in found if item.endswith(".sigmf-meta")}
        found = [item for item in found
                 if not (item.endswith(".sigmf-data") and item[:-11] in metas)]
        if len(found) > 2000:
            found = found[:2000]
            self.status.setText("文件夹内文件超过 2000 个，本次只加入前 2000 个")
        self._add_import_paths(found)

    def _add_import_paths(self, paths):
        existing = {self._import_row_info(row)["path"]
                    for row in range(self.import_table.rowCount())}
        new = [str(item) for item in paths if str(item) not in existing]
        if not new:
            self.status.setText("没有新增文件（重复路径已跳过）")
            return
        if len(new) > 2000:
            new = new[:2000]
        self.start_job("import_inspect", owner="信号导入",
                       label=f"识别 {len(new)} 个文件", cancel_text="取消本次识别",
                       paths=new)

    def _apply_import_inspection(self, files):
        for info in files:
            row = self._import_row_by_path(info["path"])
            if row is None:
                self._append_import_row(info)
            else:
                self._replace_import_row(row, info)
        self._update_import_controls()
        self.status.setText(f"已识别 {len(files)} 个文件")

    def _append_import_row(self, info):
        row = self.import_table.rowCount()
        self.import_table.insertRow(row)
        self._fill_import_row(row, info)
        self._refresh_import_row(row)

    def _replace_import_row(self, row, info):
        """重新识别：保留用户已填的采样率与调制，只刷新文件自身的信息。"""
        previous = self._import_row_info(row)
        info.setdefault("targets", previous.get("targets") or [])
        info.setdefault("capture_started_at", previous.get("capture_started_at"))
        rate_text = self._import_rate_text(row)
        modulation = self._import_row_modulation(row)
        rf_text = self._import_rf_text(row)
        self._fill_import_row(row, info)
        if rate_text and info["format"] != "sigmf":
            self._set_rate_text(row, rate_text)
        if modulation:
            self._set_modulation_text(row, modulation)
        if rf_text:
            self.import_table.item(row, IMPORT_COL_RF_CENTER).setText(rf_text)
        self._refresh_import_row(row)

    def _fill_import_row(self, row, info):
        self._import_filling = True
        try:
            flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
            file_item = QtWidgets.QTableWidgetItem(info["name"])
            file_item.setFlags(flags)
            file_item.setToolTip(info["path"])
            file_item.setData(QtCore.Qt.ItemDataRole.UserRole, info)
            self.import_table.setItem(row, IMPORT_COL_FILE, file_item)
            for column, text in (
                    (IMPORT_COL_FORMAT, IMPORT_FORMAT_LABELS.get(info["format"], "—")),
                    (IMPORT_COL_POINTS, "—"), (IMPORT_COL_STATUS, "")):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(flags)
                self.import_table.setItem(row, column, item)
            rate_item = QtWidgets.QTableWidgetItem(
                f"{info['sample_rate']:g}" if info["format"] == "sigmf" else "")
            if info["format"] == "sigmf":
                rate_item.setFlags(flags)
                rate_item.setToolTip("SigMF 采样率取自元数据，只读")
            self.import_table.setItem(row, IMPORT_COL_RATE, rate_item)
            rf_item = QtWidgets.QTableWidgetItem("")
            rf_item.setToolTip("Hz；留空 = 使用下方“射频中心（默认）”；同文件的多个目标共用")
            self.import_table.setItem(row, IMPORT_COL_RF_CENTER, rf_item)
            note_item = QtWidgets.QTableWidgetItem("")
            note_item.setToolTip("写入资产备注（最多 200 字）")
            self.import_table.setItem(row, IMPORT_COL_NOTE, note_item)
            if info["format"] == "binary":
                self._install_combo(row, IMPORT_COL_DTYPE,
                                    (("未设置", None), ("int16", "int16"),
                                     ("float32", "float32")), None)
                self._install_combo(row, IMPORT_COL_ENDIAN,
                                    (("未设置", None), ("小端 little", "little"),
                                     ("大端 big", "big")), None)
            else:
                auto_text = info.get("dtype") or "解析确定"
                self._install_combo(row, IMPORT_COL_DTYPE, ((auto_text, None),), None,
                                    enabled=False)
                self._install_combo(row, IMPORT_COL_ENDIAN, (("自动", None),), None,
                                    enabled=False)
            modulation = QtWidgets.QComboBox()
            modulation.addItems(list(IMPORT_MODULATION_CHOICES))
            modulation.setEditable(True)
            modulation.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
            modulation.currentTextChanged.connect(
                lambda *_, r=row: self._refresh_import_row(r))
            self.import_table.setCellWidget(row, IMPORT_COL_MOD, modulation)
        finally:
            self._import_filling = False

    def _install_combo(self, row, column, items, current, *, enabled=True):
        combo = QtWidgets.QComboBox()
        for label, value in items:
            combo.addItem(label, value)
        if current is not None:
            index = combo.findData(current)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.setEnabled(enabled)
        combo.currentIndexChanged.connect(lambda *_, r=row: self._refresh_import_row(r))
        self.import_table.setCellWidget(row, column, combo)

    def _import_row_info(self, row):
        item = self.import_table.item(row, IMPORT_COL_FILE)
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else {}

    def _set_import_row_info(self, row, **changes):
        """写回行状态（``item.data()`` 返回的是副本，必须重新 setData 才生效）。"""
        info = dict(self._import_row_info(row))
        info.update(changes)
        item = self.import_table.item(row, IMPORT_COL_FILE)
        if item is None:
            return info
        item.setData(QtCore.Qt.ItemDataRole.UserRole, info)
        self._refresh_import_row(row)
        return info

    def _import_row_by_path(self, path):
        for row in range(self.import_table.rowCount()):
            if self._import_row_info(row).get("path") == path:
                return row
        return None

    def _import_rate_text(self, row):
        item = self.import_table.item(row, IMPORT_COL_RATE)
        return item.text().strip() if item else ""

    def _set_rate_text(self, row, text):
        self._import_filling = True
        try:
            item = self.import_table.item(row, IMPORT_COL_RATE)
            if item is not None:
                item.setText(text)
        finally:
            self._import_filling = False

    def _import_rf_text(self, row):
        item = self.import_table.item(row, IMPORT_COL_RF_CENTER)
        return item.text().strip() if item else ""

    def _import_row_rf_center(self, row):
        """返回 ``(数值, 错误)``；留空为 ``(None, None)``，由本批默认值回落。"""
        text = self._import_rf_text(row)
        if not text:
            return None, None
        try:
            value = float(text)
        except ValueError:
            return None, "射频中心应为数值"
        if not np.isfinite(value):
            return None, "射频中心必须为有限数值"
        return value, None

    def _set_modulation_text(self, row, text):
        combo = self.import_table.cellWidget(row, IMPORT_COL_MOD)
        if combo is not None:
            combo.setCurrentText(text)

    def _import_combo_value(self, row, column):
        combo = self.import_table.cellWidget(row, column)
        return combo.currentData() if combo is not None else None

    def _import_row_rate(self, row):
        info = self._import_row_info(row)
        if info.get("format") == "sigmf":
            return info.get("sample_rate")
        text = self._import_rate_text(row)
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _import_row_modulation(self, row):
        combo = self.import_table.cellWidget(row, IMPORT_COL_MOD)
        if combo is None:
            return None
        text = combo.currentText().strip()
        return None if text in ("", "未知") else text

    def _import_row_note(self, row):
        item = self.import_table.item(row, IMPORT_COL_NOTE)
        return item.text().strip() if item else ""

    def _import_samples(self, row):
        """返回 ``(点数, 错误)``：NPY/SigMF 直接可用，二进制按类型推算。"""
        info = self._import_row_info(row)
        if info.get("sample_count") is not None:
            return int(info["sample_count"]), None
        if info.get("format") == "binary":
            dtype = self._import_combo_value(row, IMPORT_COL_DTYPE)
            if dtype:
                width = 2 * int(np.dtype(dtype).itemsize)
                size = int(info.get("size_bytes") or 0)
                if size % width:
                    return None, f"字节数不是 {dtype} 交织 I/Q 的整数倍（可能被截断）"
                return size // width, None
        return None, None

    def _import_missing_fields(self, row):
        info = self._import_row_info(row)
        if info.get("error") or info.get("format") in ("sigmf", "unknown"):
            return []
        missing = []
        if self._import_row_rate(row) is None:
            missing.append("采样率")
        if info.get("format") == "binary":
            if self._import_combo_value(row, IMPORT_COL_DTYPE) is None:
                missing.append("类型")
            if self._import_combo_value(row, IMPORT_COL_ENDIAN) is None:
                missing.append("字节序")
        return missing

    def _import_row_problem(self, row):
        """返回该行的阻塞原因；``None`` = 就绪。"""
        info = self._import_row_info(row)
        if info.get("error"):
            return f"解析失败：{info['error']}"
        missing = self._import_missing_fields(row)
        if missing:
            return "缺" + "、".join(missing)
        _, rf_error = self._import_row_rf_center(row)
        if rf_error:
            return rf_error
        _, error = self._import_samples(row)
        return error

    def _import_row_ready(self, row):
        return bool(self._import_row_info(row)) and self._import_row_problem(row) is None

    def _refresh_import_row(self, row):
        if self._import_filling or not self._import_row_info(row):
            return
        info = self._import_row_info(row)
        count, _ = self._import_samples(row)
        rate = self._import_row_rate(row)
        if count is None:
            points = "—" if info.get("format") != "csv" else "解析时确定"
        elif rate:
            points = f"{count:,} 点 · {count / rate:g} s"
        else:
            points = f"{count:,} 点"
        problem = self._import_row_problem(row)
        targets = info.get("targets") or []
        suffix = f" · 目标 {len(targets)} 条" if targets else ""
        status = (problem or "就绪") + suffix
        item = self.import_table.item(row, IMPORT_COL_POINTS)
        if item is not None:
            item.setText(points)
        status_item = self.import_table.item(row, IMPORT_COL_STATUS)
        if status_item is not None:
            status_item.setText(status)
            if problem:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor("#b00020")))
                status_item.setToolTip(problem)
            else:
                status_item.setData(QtCore.Qt.ItemDataRole.ForegroundRole, None)
                status_item.setToolTip("；".join(
                    f"{target.get('scope')} {target.get('f_low_hz') or '—'}～"
                    f"{target.get('f_high_hz') or '—'} Hz" for target in targets) or "")
        missing = self._import_missing_fields(row)
        highlight = QtGui.QBrush(QtGui.QColor("#ffe0e0"))
        rate_item = self.import_table.item(row, IMPORT_COL_RATE)
        if rate_item is not None:
            rate_item.setData(QtCore.Qt.ItemDataRole.BackgroundRole,
                              highlight if "采样率" in missing else None)
        for column, field in ((IMPORT_COL_DTYPE, "类型"), (IMPORT_COL_ENDIAN, "字节序")):
            combo = self.import_table.cellWidget(row, column)
            if combo is not None and info.get("format") == "binary":
                combo.setStyleSheet("background:#ffe0e0" if field in missing else "")
        self._update_import_controls()

    def _import_has_ready(self):
        return any(self._import_row_ready(row)
                   for row in range(self.import_table.rowCount()))

    def _update_import_controls(self):
        total = self.import_table.rowCount()
        ready = sum(1 for row in range(total) if self._import_row_ready(row))
        self.import_count_label.setText(
            "清单为空" if not total else f"清单 {total} 个文件 · 就绪 {ready} 个")
        self.import_start_button.setEnabled(ready > 0 and not self.task_running("信号导入"))

    def _import_item_changed(self, item):
        if self._import_filling or item.column() not in (IMPORT_COL_RATE,
                                                         IMPORT_COL_RF_CENTER):
            return
        self._refresh_import_row(item.row())

    def recheck_import_rows(self):
        rows = sorted({index.row() for index in self.import_table.selectedIndexes()})
        if not rows:
            self.status.setText("请先在文件清单里选择要重新识别的行")
            return
        self.start_job("import_inspect", owner="信号导入",
                       label=f"重新识别 {len(rows)} 个文件", cancel_text="取消本次识别",
                       paths=[self._import_row_info(row)["path"] for row in rows])

    def remove_import_rows(self):
        rows = sorted({index.row() for index in self.import_table.selectedIndexes()},
                      reverse=True)
        if not rows:
            self.status.setText("请先选择要移除的行")
            return
        for row in rows:
            self.import_table.removeRow(row)
        self._update_import_controls()

    def _import_collection_changed(self):
        self.import_collection_name.setVisible(
            self.import_collection.currentData() == "__new__")

    def _toggle_import_params(self, visible):
        self.import_params_toggle.setArrowType(
            QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)
        self.import_params_body.setVisible(visible)

    def batch_set_import_params(self):
        rows = sorted({index.row() for index in self.import_table.selectedIndexes()})
        if not rows:
            self.status.setText("请先在文件清单里选择要设置的行")
            return
        dialog = ImportBatchDialog(self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._apply_batch_settings(rows, **dialog.values())

    def _apply_batch_settings(self, rows, *, sample_rate="", binary_dtype=None, endian=None):
        """把批量设置应用到给定行；SigMF 行的采样率保持只读。"""
        if sample_rate:
            try:
                value = float(sample_rate)
            except ValueError:
                self.status.setText(f"采样率应为数值：{sample_rate}")
                return
            if value <= 0:
                self.status.setText("采样率必须为正数")
                return
            for row in rows:
                if self._import_row_info(row).get("format") != "sigmf":
                    self._set_rate_text(row, f"{value:g}")
        for row in rows:
            if self._import_row_info(row).get("format") == "binary":
                if binary_dtype:
                    combo = self.import_table.cellWidget(row, IMPORT_COL_DTYPE)
                    if combo is not None:
                        combo.setCurrentIndex(combo.findData(binary_dtype))
                if endian:
                    combo = self.import_table.cellWidget(row, IMPORT_COL_ENDIAN)
                    if combo is not None:
                        combo.setCurrentIndex(combo.findData(endian))
            self._refresh_import_row(row)
        self.status.setText(f"批量设置已应用到 {len(rows)} 行")

    def _optional_field_float(self, field, name):
        text = field.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{name}应为数值") from exc

    def apply_single_signal_targets(self):
        """把“整条为单一信号”目标应用到所选文件（覆盖它们已有的目标行）。"""
        rows = sorted({index.row() for index in self.import_table.selectedIndexes()})
        if not rows:
            self.status.setText("请先在文件清单里选择要应用的文件")
            return
        modulation = self.import_single_modulation.currentText().strip()
        modulation = None if modulation in ("", "未知") else modulation
        try:
            low = self._optional_field_float(self.import_single_f_low, "频率下限")
            high = self._optional_field_float(self.import_single_f_high, "频率上限")
            snr = self._optional_field_float(self.import_single_snr, "SNR")
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if (low is None) != (high is None):
            self.status.setText("频率范围必须同时给出上下限")
            return
        if low is not None and high <= low:
            self.status.setText("频率上限必须大于下限")
            return
        for row in rows:
            count, _ = self._import_samples(row)
            target = {"scope": "whole_record", "start": 0, "end": count,
                      "start_unit": "samples", "f_low_hz": low,
                      "f_high_hz": high, "modulation": modulation,
                      "snr_db": snr, "note": None}
            self._set_import_row_info(row, targets=[target])
        self.status.setText(f"已为 {len(rows)} 个文件设置“整条为单一信号”目标")

    # ------------------------------------------------------------------ 标注清单
    def import_csv_manifest(self):
        if not self.import_table.rowCount():
            self.status.setText("请先添加文件，再导入标注清单")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择标注清单 CSV", "", "CSV (*.csv)")
        if not path:
            return
        paths = [self._import_row_info(row)["path"]
                 for row in range(self.import_table.rowCount())]
        self.start_job("import_manifest", owner="信号导入", label="解析标注清单",
                       cancel_text="取消本次解析", path=path, paths=paths)

    def export_import_template(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出标注清单模板", "标注清单模板.csv", "CSV (*.csv)")
        if not path:
            return
        import csv as _csv

        header = ["文件", "粒度", "起止单位", "起始", "结束", "频率下限Hz",
                  "频率上限Hz", "调制", "SNR", "备注", "采集时间"]
        examples = [["record_000.npy", "whole_record", "采样点", "0", "", "-20000",
                     "20000", "QPSK", "18", "整条为单一信号", "2026-09-01T08:30:00Z"],
                    ["record_001.npy", "session", "毫秒", "0", "120", "50000",
                     "150000", "AM", "", "同一文件多行 = 多目标", "2026-09-01T09:15:00Z"],
                    ["record_001.npy", "session", "毫秒", "130", "260", "-150000",
                     "-60000", "QPSK", "12", "时间可写毫秒或秒", "2026-09-01T09:15:00Z"]]
        try:
            with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
                writer = _csv.writer(handle)
                writer.writerow(header)
                writer.writerows(examples)
        except OSError as exc:
            self.status.setText(f"模板导出失败：{exc}")
            return
        self.status.setText(f"已导出标注清单模板：{path}")

    def _apply_import_manifest(self, result):
        files = result.get("files") or {}
        captures = result.get("captures") or {}
        applied_rows = applied_targets = 0
        for row in range(self.import_table.rowCount()):
            info = self._import_row_info(row)
            changes = {}
            targets = files.get(info["path"])
            if targets:
                changes["targets"] = list(targets)
                applied_rows += 1
                applied_targets += len(targets)
            capture = captures.get(info["path"])
            if capture:
                changes["capture_started_at"] = capture
            if changes:
                self._set_import_row_info(row, **changes)
        unmatched = result.get("unmatched") or []
        text = f"标注清单：已挂接 {applied_targets} 条目标（{applied_rows} 个文件）"
        if captures:
            text += f"；{len(captures)} 个文件带采集时间"
        if unmatched:
            lines = [f"第 {item['line']} 行 · {item['file']}：{item['error']}"
                     for item in unmatched[:15]]
            text += f"；未匹配/非法 {len(unmatched)} 行：\n" + "\n".join(lines)
        self.import_csv_report.setText(text)
        self.status.setText(f"标注清单解析完成：挂接 {applied_targets} 条 · "
                            f"未匹配 {len(unmatched)} 行")

    # ------------------------------------------------------------------ 导入
    def _resolve_batch_shard(self, file_count):
        mode = self.import_shard_mode.currentData()
        if mode == "file":
            return False
        if mode == "shard":
            return True
        return file_count >= self.import_shard_threshold.value()

    def start_import_batch(self):
        if self.task_running("信号导入"):
            self.status.setText("“信号导入”已有任务在运行；请等待完成或先取消")
            return
        rows = [row for row in range(self.import_table.rowCount())
                if self._import_row_ready(row)]
        if not rows:
            self.status.setText("没有参数完备（就绪）的文件可导入")
            return
        scope = self.import_collection.currentData()
        collection_name = None
        if scope == "__new__":
            collection_name = self.import_collection_name.text().strip()
            if not collection_name:
                self.status.setText("已选择“新建集合”，请填写集合名称")
                return
        files = []
        for row in rows:
            info = self._import_row_info(row)
            entry = {
                "path": info["path"],
                "sample_rate": (None if info.get("format") == "sigmf"
                                else self._import_row_rate(row)),
                "binary_dtype": self._import_combo_value(row, IMPORT_COL_DTYPE),
                "endian": self._import_combo_value(row, IMPORT_COL_ENDIAN),
                "label": self._import_row_note(row) or None,
                "targets": list(info.get("targets") or [])}
            rf_center, _ = self._import_row_rf_center(row)
            if rf_center is not None:
                entry["rf_center_hz"] = rf_center  # 行内优先；缺省回落本批默认
            capture = info.get("capture_started_at")
            if capture:
                entry["capture_started_at"] = capture  # 来自标注清单；缺省由服务记导入时间
            files.append(entry)
        self.import_manage_button.setVisible(False)
        self.start_job("import_files", owner="信号导入",
                       label=f"导入 {len(files)} 个文件", cancel_text="取消本次导入",
                       files=files,
                       batch_shard=self._resolve_batch_shard(len(files)),
                       rf_center_hz=self.import_rf_center.text().strip() or None,
                       collection_id=scope if scope not in (None, "__new__") else None,
                       collection_name=collection_name,
                       initial_labels=self.import_initial_labels.isChecked())

    def _render_import_batch(self, result):
        """成功行移出清单；失败行带原因留下，修正后可再次导入。"""
        by_path = {item["path"]: item for item in result["results"]}
        for row in range(self.import_table.rowCount() - 1, -1, -1):
            info = self._import_row_info(row)
            item = by_path.get(info["path"])
            if item is None:
                continue
            if item["ok"]:
                self.import_table.removeRow(row)
            else:
                self._set_import_row_info(row, error=item["error"])
        self.import_status.setText(self.result_status(result))
        self.import_manage_button.setVisible(bool(result.get("collection_id")))
        self.refresh_assets()
        self._collections_changed()
        # 刷新下拉之后再切换：新集合只在任务里创建，刷新前下拉里还没有它
        if result.get("collection_id"):
            index = self.collection_combo.findData(result["collection_id"])
            if index >= 0:
                self.collection_combo.setCurrentIndex(index)
        self._update_import_controls()
