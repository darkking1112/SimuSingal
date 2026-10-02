"""信号导入页（mixin）：文件清单、信号参数编辑、CSV 标注清单与导入执行。

2026-10 改版：表格只读，编辑统一经由下方“信号参数编辑”面板（应用到所选信号）；
清单新增“信号名称”列（显式名 > 调制 · 文件名前 5 字符 > 文件名，自动命名由
``services.imports.import_signal_name`` 统一提供）；导入仅登记 AMC 目标（调制 +
可选 SNR，写目标参考参数 ``snr_db``，口径 ``declared``），频率/射频中心/备注
不再由本页编辑或写入。缺参数的行标红、不参与导入。
"""
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ...services.imports import import_signal_name
from ..constants import (EXPORT_FORMATS, IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NAME, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_SNR, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..dialogs import SignalParamsDialog
from ..helpers import (_asset_exports, _asset_format, _fmt_hz, _fmt_metric, _fmt_span,
                       _iq_binary_kind, _mirrored_spectrum)
from ..runner import _run_task
from ..widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


def _canonical_import_paths(paths, existing=()):
    """导入清单路径归一：SigMF 双文件按一条记录算，只登记元数据文件。

    ``x.sigmf-data`` 在同名 ``x.sigmf-meta`` 存在时归一为该元数据路径——同批
    既选两个文件、或先后分两次添加，都只留一条记录；``existing``（清单已有
    路径）参与去重。元数据缺失的数据文件保持原样，由识别阶段给出明确错误。
    """
    seen = {str(item) for item in existing}
    result = []
    for raw in paths:
        item = Path(str(raw))
        if item.suffix.lower() == ".sigmf-data":
            meta = item.with_suffix(".sigmf-meta")
            if meta.is_file():
                item = meta
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


class ImportPageMixin:
    def build_import(self):
        """独立导入页（方案 §7.1，2026-10 改版）：以文件清单为中心。

        表格只读展示识别结果与已填参数；“信号参数编辑”面板对所选文件统一填
        信号名称/采样率/调制/SNR/类型/字节序，点击“应用到所选信号”写回表格。
        缺参数的单元格标红、状态不为“就绪”的行不参与导入；批量真值用
        “导入标注清单 CSV”（按文件名匹配、支持信号名称，未匹配行原样列出）。
        """
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "批量导入离线 IQ：先添加文件或文件夹，自动识别格式与已知参数；"
            "缺采样率/类型/字节序的行标红、不参与导入。SigMF 自动读采样率，"
            "IQ 二进制需显式给出类型与字节序。在清单中选择文件后，用“信号参数编辑”"
            "统一填写信号名称与参数（AMC：调制 + 可选 SNR）；多目标请用"
            "“导入标注清单 CSV…”集中申报。")
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
            "递归收集 .npy/.csv/.bin/.raw/.iq 与 SigMF 元数据文件；"
            "SigMF 双文件按一条记录计（数据文件自动配对，不单独列出）")
        self.import_add_folder_button.clicked.connect(self.choose_import_folder)
        toolbar.addWidget(self.import_add_folder_button)
        self.import_remove_button = QtWidgets.QPushButton("移除选中")
        self.import_remove_button.clicked.connect(self.remove_import_rows)
        toolbar.addWidget(self.import_remove_button)
        self.import_recheck_button = QtWidgets.QPushButton("重新识别")
        self.import_recheck_button.setToolTip("对选中文件重读头部/元数据（文件被外部替换后使用）")
        self.import_recheck_button.clicked.connect(self.recheck_import_rows)
        toolbar.addWidget(self.import_recheck_button)
        self.import_csv_button = QtWidgets.QPushButton("导入标注清单 CSV…")
        self.import_csv_button.clicked.connect(self.import_csv_manifest)
        toolbar.addWidget(self.import_csv_button)
        toolbar.addStretch(1)
        self.import_count_label = QtWidgets.QLabel("清单为空")
        toolbar.addWidget(self.import_count_label)
        layout.addLayout(toolbar)

        self.import_table = self._make_table(
            ["文件名", "格式", "信号名称", "采样率", "类型", "字节序", "点数/时长",
             "调制", "SNR", "状态"])
        self.import_table.setMinimumHeight(220)
        # Windows 资源管理器风格：选中行（含多选整行）浅蓝底、深色字。
        # 注：windows11 原生样式会忽略 selection-background-color，
        # 必须用 ::item:selected 才生效；窗口失焦时用更浅的蓝。
        self.import_table.setStyleSheet(
            "QTableWidget { selection-background-color:#cce8ff;"
            " selection-color:#12324d; }"
            "QTableWidget::item:selected { background:#cce8ff; color:#12324d; }"
            "QTableWidget::item:selected:!active { background:#e5f1fb;"
            " color:#12324d; }")
        layout.addWidget(self.import_table, 1)

        record_group = QtWidgets.QGroupBox("归属与采集信息（应用到本批）")
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
        self.import_initial_labels = QtWidgets.QCheckBox("同时生成初始标注")
        self.import_initial_labels.setToolTip(
            "按目标参考参数为新资产写检测/AMC 初始标签；集合无标注集时自动建立")
        record_row.addWidget(self.import_initial_labels)
        record_row.addStretch(1)
        layout.addWidget(record_group)

        self.import_params_body = QtWidgets.QGroupBox("信号参数编辑（应用到所选信号）")
        params_layout = QtWidgets.QHBoxLayout(self.import_params_body)
        params_layout.addWidget(QtWidgets.QLabel("信号名称"))
        self.import_name = QtWidgets.QLineEdit()
        self.import_name.setPlaceholderText("留空不修改；自动命名见提示")
        self.import_name.setToolTip(
            "显式填写优先生效；留空时自动命名：有调制 → “调制 · 文件名前 5 字符”，无调制 → 文件名")
        self.import_name.setMaximumWidth(220)
        params_layout.addWidget(self.import_name)
        params_layout.addWidget(QtWidgets.QLabel("采样率"))
        self.import_rate = QtWidgets.QLineEdit()
        self.import_rate.setPlaceholderText("Hz")
        self.import_rate.setMaximumWidth(110)
        params_layout.addWidget(self.import_rate)
        params_layout.addWidget(QtWidgets.QLabel("调制"))
        self.import_modulation = QtWidgets.QComboBox()
        self.import_modulation.addItems(list(IMPORT_MODULATION_CHOICES))
        self.import_modulation.setEditable(True)
        self.import_modulation.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.import_modulation.setToolTip(
            "留空不修改；“未知”表示不建目标（清除该行已有目标）；"
            "其余值登记为整条 AMC 目标，可自由输入字典外类名")
        self.import_modulation.setCurrentText("")
        params_layout.addWidget(self.import_modulation)
        params_layout.addWidget(QtWidgets.QLabel("SNR"))
        self.import_snr = QtWidgets.QLineEdit()
        self.import_snr.setPlaceholderText("dB（留空不改）")
        self.import_snr.setMaximumWidth(110)
        self.import_snr.setToolTip(
            "AMC 参考参数（可选）：数值写入所有选中行目标的参考参数；\n"
            "留空不修改；所选行还没有目标时先填写调制")
        params_layout.addWidget(self.import_snr)
        params_layout.addWidget(QtWidgets.QLabel("类型"))
        self.import_dtype = QtWidgets.QComboBox()
        self.import_dtype.addItem("不修改", None)
        self.import_dtype.addItem("int16", "int16")
        self.import_dtype.addItem("float32", "float32")
        params_layout.addWidget(self.import_dtype)
        params_layout.addWidget(QtWidgets.QLabel("字节序"))
        self.import_endian = QtWidgets.QComboBox()
        self.import_endian.addItem("不修改", None)
        self.import_endian.addItem("小端", "little")
        self.import_endian.addItem("大端", "big")
        params_layout.addWidget(self.import_endian)
        self.import_apply_button = QtWidgets.QPushButton("应用到所选信号")
        self.import_apply_button.setToolTip(
            "把非空字段应用到所选文件；类型/字节序仅当所选文件全部为 IQ 二进制时可编辑")
        self.import_apply_button.clicked.connect(self.apply_signal_params)
        params_layout.addWidget(self.import_apply_button)
        params_layout.addStretch(1)
        layout.addWidget(self.import_params_body)
        self.import_table.itemSelectionChanged.connect(self._load_signal_params)

        advanced = QtWidgets.QGroupBox("写入方式")
        advanced_row = QtWidgets.QHBoxLayout(advanced)
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
        action_bar.addStretch(1)
        layout.addLayout(action_bar)
        self.import_csv_report = QtWidgets.QLabel("")
        self.import_csv_report.setWordWrap(True)
        layout.addWidget(self.import_csv_report)
        # 分组框统一黑色细边框（页面级样式：仅作用于本页 QGroupBox）
        box.setStyleSheet(
            "QGroupBox { border:1px solid #000000; border-radius:4px;"
            " margin-top:0.7em; }"
            "QGroupBox::title { subcontrol-origin:margin;"
            " subcontrol-position:top left; left:8px; padding:0 3px; }")
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
        # SigMF 双文件按一条记录算：只登记元数据文件（数据文件自动配对）
        found = _canonical_import_paths(found)
        if len(found) > 2000:
            found = found[:2000]
            self.status.setText("文件夹内文件超过 2000 个，本次只加入前 2000 个")
        self._add_import_paths(found)

    def _add_import_paths(self, paths):
        existing = {self._import_row_info(row)["path"]
                    for row in range(self.import_table.rowCount())}
        new = _canonical_import_paths(paths, existing)
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
        self._drop_shadowed_sigmf_rows()
        self._update_import_controls()
        self.status.setText(f"已识别 {len(files)} 个文件")

    def _drop_shadowed_sigmf_rows(self):
        """SigMF 双文件按一条记录计：元数据行已在清单时，移除同名的数据文件行。"""
        paths = {str(self._import_row_info(row).get("path") or "")
                 for row in range(self.import_table.rowCount())}
        for row in range(self.import_table.rowCount() - 1, -1, -1):
            item = Path(str(self._import_row_info(row).get("path") or ""))
            if item.suffix.lower() == ".sigmf-data" and \
                    str(item.with_suffix(".sigmf-meta")) in paths:
                self.import_table.removeRow(row)

    def _append_import_row(self, info):
        row = self.import_table.rowCount()
        self.import_table.insertRow(row)
        self._fill_import_row(row, info)
        self._refresh_import_row(row)

    def _replace_import_row(self, row, info):
        """重新识别：保留本页已填/已申报的行参数，只刷新文件自身的信息。"""
        previous = self._import_row_info(row)
        for key in ("signal_name", "modulation", "binary_dtype", "endian",
                    "capture_started_at"):
            if key not in info:
                info[key] = previous.get(key)
        info["targets"] = list(previous.get("targets") or [])
        if info.get("format") != "sigmf" and not info.get("sample_rate"):
            info["sample_rate"] = previous.get("sample_rate")
        self._fill_import_row(row, info)

    def _fill_import_row(self, row, info):
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        file_item = QtWidgets.QTableWidgetItem(info["name"])
        file_item.setFlags(flags)
        file_item.setToolTip(info["path"])
        file_item.setData(QtCore.Qt.ItemDataRole.UserRole, info)
        self.import_table.setItem(row, IMPORT_COL_FILE, file_item)
        for column, text in (
                (IMPORT_COL_FORMAT, IMPORT_FORMAT_LABELS.get(info["format"], "—")),
                (IMPORT_COL_NAME, ""), (IMPORT_COL_RATE, ""), (IMPORT_COL_DTYPE, ""),
                (IMPORT_COL_ENDIAN, ""), (IMPORT_COL_POINTS, "—"),
                (IMPORT_COL_MOD, ""), (IMPORT_COL_SNR, ""), (IMPORT_COL_STATUS, "")):
            item = QtWidgets.QTableWidgetItem(text)
            item.setFlags(flags)
            self.import_table.setItem(row, column, item)
        rate_item = self.import_table.item(row, IMPORT_COL_RATE)
        if info.get("format") == "sigmf":
            rate_item.setToolTip("SigMF 采样率取自元数据，只读")
        else:
            rate_item.setToolTip("Hz；由“信号参数编辑”填写")
        dtype_item = self.import_table.item(row, IMPORT_COL_DTYPE)
        if info.get("format") == "binary":
            dtype_item.setToolTip("交织 IQ 数据类型：int16（按 1/32768 缩放）或 float32")
        else:
            dtype_item.setToolTip("由文件（头）确定，不可修改")
        endian_item = self.import_table.item(row, IMPORT_COL_ENDIAN)
        if info.get("format") == "binary":
            endian_item.setToolTip("IQ 二进制字节序：小端或大端；由“信号参数编辑”选择")
        else:
            endian_item.setToolTip("非二进制格式无需字节序")
        snr_item = self.import_table.item(row, IMPORT_COL_SNR)
        snr_item.setToolTip("AMC 参考参数（dB）：由“信号参数编辑”的 SNR 或标注清单写入")
        self._refresh_import_row(row)

    def _refresh_import_row(self, row):
        info = self._import_row_info(row)
        if not info:
            return
        self._set_cell(row, IMPORT_COL_NAME, self._display_signal_name(info))
        rate = self._import_row_rate(row)
        self._set_cell(row, IMPORT_COL_RATE, f"{rate:g}" if rate else "")
        self._set_cell(row, IMPORT_COL_DTYPE, self._dtype_text(info))
        self._set_cell(row, IMPORT_COL_ENDIAN, self._endian_text(info))
        count, _ = self._import_samples(row)
        if count is None:
            points = "—" if info.get("format") != "csv" else "解析时确定"
        elif rate:
            points = f"{count:,} 点 · {count / rate:g} s"
        else:
            points = f"{count:,} 点"
        self._set_cell(row, IMPORT_COL_POINTS, points)
        self._set_cell(row, IMPORT_COL_MOD, info.get("modulation") or "—")
        targets = info.get("targets") or []
        self._set_cell(row, IMPORT_COL_SNR, self._display_target_snr(targets))
        problem = self._import_row_problem(row)
        suffix = f" · 目标 {len(targets)} 条" if targets else ""
        status_item = self.import_table.item(row, IMPORT_COL_STATUS)
        if status_item is not None:
            status_item.setText((problem or "就绪") + suffix)
            if problem:
                status_item.setForeground(QtGui.QBrush(QtGui.QColor("#b00020")))
                status_item.setToolTip(problem)
            else:
                status_item.setData(QtCore.Qt.ItemDataRole.ForegroundRole, None)
                status_item.setToolTip("；".join(
                    self._target_label(target) for target in targets) or "")
        missing = self._import_missing_fields(row)
        highlight = QtGui.QBrush(QtGui.QColor("#ffe0e0"))
        for column, field in ((IMPORT_COL_RATE, "采样率"), (IMPORT_COL_DTYPE, "类型"),
                              (IMPORT_COL_ENDIAN, "字节序")):
            item = self.import_table.item(row, column)
            if item is not None:
                item.setData(QtCore.Qt.ItemDataRole.BackgroundRole,
                             highlight if field in missing else None)
        self._update_import_controls()

    def _set_cell(self, row, column, text):
        item = self.import_table.item(row, column)
        if item is not None:
            item.setText(text)

    @staticmethod
    def _dtype_text(info):
        if info.get("format") == "binary":
            return info.get("binary_dtype") or "未设置"
        return info.get("dtype") or "解析确定"

    @staticmethod
    def _endian_text(info):
        if info.get("format") == "binary":
            labels = {"little": "小端", "big": "大端"}
            return labels.get(info.get("endian") or "", "未设置")
        return "自动"

    @staticmethod
    def _display_signal_name(info):
        """清单展示名：显式信号名 > 自动命名（调制 · 文件名前 5 字符）> 文件名。"""
        return info.get("signal_name") or import_signal_name(
            info.get("name") or "", info.get("modulation"))

    @staticmethod
    def _target_snr_value(target):
        """目标 SNR 的显示/比较值：兼容清单阶段的字符串数值与空值。"""
        value = target.get("snr_db")
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _target_common_snr(cls, targets):
        """目标列表的公共 SNR：全部一致时返回该值（含全空 = None），否则 None。"""
        values = {cls._target_snr_value(target) for target in (targets or [])}
        if len(values) == 1:
            return next(iter(values))
        return None

    @classmethod
    def _display_target_snr(cls, targets):
        """清单 SNR 列：无目标/全未知为空；一致显示数值；参差显示“混合”。"""
        if not targets:
            return ""
        values = {cls._target_snr_value(target) for target in targets}
        if len(values) == 1:
            value = next(iter(values))
            return "" if value is None else f"{value:g}"
        return "混合"

    @classmethod
    def _target_label(cls, target):
        """状态提示里的目标一行文案：粒度 · 调制（· SNR x dB）。"""
        snr = cls._target_snr_value(target)
        text = f"{target.get('scope')} · {target.get('modulation') or '调制未知'}"
        return text + (f" · SNR {snr:g} dB" if snr is not None else "")

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

    def _import_row_rate(self, row):
        return self._import_row_info(row).get("sample_rate")

    def _import_row_modulation(self, row):
        return self._import_row_info(row).get("modulation")

    def _import_samples(self, row):
        """返回 ``(点数, 错误)``：NPY/SigMF 直接可用，二进制按类型推算。"""
        info = self._import_row_info(row)
        if info.get("sample_count") is not None:
            return int(info["sample_count"]), None
        if info.get("format") == "binary" and info.get("binary_dtype"):
            width = 2 * int(np.dtype(info["binary_dtype"]).itemsize)
            size = int(info.get("size_bytes") or 0)
            if size % width:
                return None, f"字节数不是 {info['binary_dtype']} 交织 I/Q 的整数倍（可能被截断）"
            return size // width, None
        return None, None

    def _import_missing_fields(self, row):
        info = self._import_row_info(row)
        if info.get("error") or info.get("format") in ("sigmf", "unknown"):
            return []
        missing = []
        if not info.get("sample_rate"):
            missing.append("采样率")
        if info.get("format") == "binary":
            if not info.get("binary_dtype"):
                missing.append("类型")
            if not info.get("endian"):
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
        _, error = self._import_samples(row)
        return error

    def _import_row_ready(self, row):
        return bool(self._import_row_info(row)) and self._import_row_problem(row) is None

    def _import_has_ready(self):
        return any(self._import_row_ready(row)
                   for row in range(self.import_table.rowCount()))

    def _update_import_controls(self):
        total = self.import_table.rowCount()
        ready = sum(1 for row in range(total) if self._import_row_ready(row))
        self.import_count_label.setText(
            "清单为空" if not total else f"清单 {total} 个文件 · 就绪 {ready} 个")
        self.import_start_button.setEnabled(ready > 0 and not self.task_running("信号导入"))

    # ------------------------------------------------------------ 信号参数编辑
    def _selected_import_rows(self):
        return sorted({index.row() for index in self.import_table.selectedIndexes()})

    def _load_signal_params(self):
        """按所选行载入公共值；多值留空（留空 = 不修改）。"""
        rows = [row for row in self._selected_import_rows() if self._import_row_info(row)]
        if not rows:
            return
        infos = [self._import_row_info(row) for row in rows]
        names = {info.get("signal_name") or "" for info in infos}
        self.import_name.setText(next(iter(names)) if len(names) == 1 else "")
        non_sigmf = [info for info in infos if info.get("format") != "sigmf"]
        self.import_rate.setEnabled(bool(non_sigmf))
        self.import_rate.setToolTip("" if non_sigmf else "SigMF 采样率取自元数据，不可修改")
        rates = {float(info["sample_rate"]) for info in non_sigmf
                 if info.get("sample_rate")}
        self.import_rate.setText(f"{next(iter(rates)):g}" if len(rates) == 1 else "")
        mods = {info.get("modulation") or "" for info in infos}
        self.import_modulation.setCurrentText(next(iter(mods)) if len(mods) == 1 else "")
        snrs = {self._target_common_snr(info.get("targets") or []) for info in infos}
        common_snr = next(iter(snrs)) if len(snrs) == 1 else None
        self.import_snr.setText("" if common_snr is None else f"{common_snr:g}")
        binary_all = all(info.get("format") == "binary" for info in infos)
        for combo, key in ((self.import_dtype, "binary_dtype"),
                           (self.import_endian, "endian")):
            combo.setEnabled(binary_all)
            combo.setToolTip("" if binary_all
                             else "仅当所选文件全部为 IQ 二进制时可编辑")
            if binary_all:
                values = {info.get(key) for info in infos if info.get(key)}
                combo.setCurrentIndex(
                    combo.findData(next(iter(values)) if len(values) == 1 else None))
            else:
                combo.setCurrentIndex(0)

    def apply_signal_params(self):
        """把面板非空字段应用到所选文件；调制非空时替换为一条整条 AMC 目标。

        SNR 写入目标的参考参数：有目标则更新，
        同一次应用里新建的目标直接带上；所选行尚无目标时提示先申报调制。
        """
        rows = self._selected_import_rows()
        if not rows:
            self.status.setText("请先在文件清单里选择要编辑的文件")
            return
        name = self.import_name.text().strip()
        if len(name) > 200:
            self.status.setText("信号名称最多 200 个字符")
            return
        rate = None
        rate_text = self.import_rate.text().strip()
        if rate_text:
            try:
                rate = float(rate_text)
            except ValueError:
                self.status.setText(f"采样率应为数值：{rate_text}")
                return
            if not np.isfinite(rate) or rate <= 0:
                self.status.setText("采样率必须为正数（Hz）")
                return
        snr = None
        snr_text = self.import_snr.text().strip()
        if snr_text:
            try:
                snr = float(snr_text)
            except ValueError:
                self.status.setText(f"SNR 应为数值（dB）：{snr_text}")
                return
            if not np.isfinite(snr):
                self.status.setText("SNR 必须为有限数值（dB）")
                return
        modulation = self.import_modulation.currentText().strip()
        binary_all = all(self._import_row_info(row).get("format") == "binary"
                         for row in rows)
        binary_dtype = self.import_dtype.currentData() if binary_all else None
        endian = self.import_endian.currentData() if binary_all else None
        sigmf_skipped = 0
        snr_applied = snr_pending = 0
        applied = []
        for row in rows:
            info = self._import_row_info(row)
            targets = list(info.get("targets") or [])
            changes = {}
            if name:
                changes["signal_name"] = name
            if rate is not None:
                if info.get("format") == "sigmf":
                    sigmf_skipped += 1
                else:
                    changes["sample_rate"] = rate
            if binary_dtype:
                changes["binary_dtype"] = binary_dtype
            if endian:
                changes["endian"] = endian
            if modulation:
                if modulation == "未知":
                    changes["modulation"] = None
                    changes["targets"] = []
                else:
                    count, _ = self._import_samples(row)
                    carried = (snr if snr is not None
                               else self._target_common_snr(targets))
                    target = {"scope": "whole_record", "modulation": modulation}
                    if carried is not None:
                        target["snr_db"] = carried
                    if count:
                        target.update({"start": 0, "end": int(count),
                                       "start_unit": "samples"})
                    changes["modulation"] = modulation
                    changes["targets"] = [target]
                    if snr is not None:
                        snr_applied += 1
            elif snr is not None:
                if targets:
                    changes["targets"] = [{**target, "snr_db": snr}
                                          for target in targets]
                    snr_applied += 1
                else:
                    snr_pending += 1
            if changes:
                self._set_import_row_info(row, **changes)
        if name:
            applied.append("信号名称")
        if rate is not None:
            applied.append("采样率")
        if modulation:
            applied.append("调制")
        if snr is not None and snr_applied:
            applied.append("SNR")
        if binary_dtype or endian:
            applied.append("类型/字节序")
        if applied:
            text = f"已应用 {'、'.join(applied)} 到 {len(rows)} 个文件"
            if snr_pending:
                text += f"；{snr_pending} 行无目标，SNR 未应用（请先申报调制）"
        elif snr is not None and snr_pending:
            text = "SNR 未应用：所选行尚无目标（请先申报调制）"
        else:
            text = "未应用任何字段（留空表示不修改）"
        if sigmf_skipped:
            text += f"；{sigmf_skipped} 个 SigMF 行采样率取自元数据，已跳过"
        self.status.setText(text)
        self._load_signal_params()

    def recheck_import_rows(self):
        rows = self._selected_import_rows()
        if not rows:
            self.status.setText("请先在文件清单里选择要重新识别的行")
            return
        self.start_job("import_inspect", owner="信号导入",
                       label=f"重新识别 {len(rows)} 个文件", cancel_text="取消本次识别",
                       paths=[self._import_row_info(row)["path"] for row in rows])

    def remove_import_rows(self):
        rows = sorted(self._selected_import_rows(), reverse=True)
        if not rows:
            self.status.setText("请先选择要移除的行")
            return
        for row in rows:
            self.import_table.removeRow(row)
        self._update_import_controls()

    def _import_collection_changed(self):
        self.import_collection_name.setVisible(
            self.import_collection.currentData() == "__new__")

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

    def _apply_import_manifest(self, result):
        files = result.get("files") or {}
        captures = result.get("captures") or {}
        names = result.get("names") or {}
        ignored = result.get("ignored_columns") or []
        applied_rows = applied_targets = applied_names = 0
        for row in range(self.import_table.rowCount()):
            info = self._import_row_info(row)
            changes = {}
            targets = files.get(info["path"])
            if targets:
                changes["targets"] = list(targets)
                applied_rows += 1
                applied_targets += len(targets)
                modulation = next((target.get("modulation") for target in targets
                                   if target.get("modulation")), None)
                if modulation:
                    changes["modulation"] = modulation
            capture = captures.get(info["path"])
            if capture:
                changes["capture_started_at"] = capture
            name = names.get(info["path"])
            if name:
                changes["signal_name"] = name
                applied_names += 1
            if changes:
                self._set_import_row_info(row, **changes)
        unmatched = result.get("unmatched") or []
        with_snr = sum(1 for targets in files.values()
                       for target in targets if target.get("snr_db") is not None)
        text = f"标注清单：已挂接 {applied_targets} 条目标（{applied_rows} 个文件）"
        if with_snr:
            text += f"；{with_snr} 条目标带 SNR"
        if applied_names:
            text += f"；{applied_names} 个文件带信号名称"
        if captures:
            text += f"；{len(captures)} 个文件带采集时间"
        if ignored:
            text += f"；已忽略列：{'、'.join(ignored)}（导入仅支持 AMC 标注）"
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
                                else info.get("sample_rate")),
                "binary_dtype": info.get("binary_dtype"),
                "endian": info.get("endian"),
                "name": info.get("signal_name") or None,
                "targets": list(info.get("targets") or [])}
            capture = info.get("capture_started_at")
            if capture:
                entry["capture_started_at"] = capture  # 来自标注清单；缺省由服务记导入时间
            files.append(entry)
        self.import_manage_button.setVisible(False)
        self.start_job("import_files", owner="信号导入",
                       label=f"导入 {len(files)} 个文件", cancel_text="取消本次导入",
                       files=files,
                       batch_shard=self._resolve_batch_shard(len(files)),
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
        self.import_manage_button.setVisible(bool(result.get("collection_id")))
        self.refresh_assets()
        self._collections_changed()
        # 刷新下拉之后再切换：新集合只在任务里创建，刷新前下拉里还没有它
        if result.get("collection_id"):
            index = self.collection_combo.findData(result["collection_id"])
            if index >= 0:
                self.collection_combo.setCurrentIndex(index)
        self._update_import_controls()
