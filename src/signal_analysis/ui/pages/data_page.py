"""数据管理页（mixin）：容量盘点、一致性检查、安全清理与“信号集合”子页。"""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ...storage.maintenance import RUN_KIND_LABELS, format_bytes, read_settings, write_settings
from ...data import Workspace
from ..constants import (EXPORT_FORMATS, IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NOTE, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_RF_CENTER, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..helpers import (_asset_exports, _asset_format, _fmt_hz, _fmt_metric, _fmt_span,
                       _iq_binary_kind, _mirrored_spectrum)
from ..runner import _run_task


class DataPageMixin:
    # ------------------------------------------------------------ 数据管理页
    _STORAGE_HEAD = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
        "<title>数据盘点报告</title><style>"
        "body{font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#23374d;"
        "background:#f3f6fa;margin:0;padding:28px 34px;}"
        "h1{font-size:22px;color:#183c65;margin:0 0 4px;}"
        "h2{font-size:16px;color:#183c65;margin:26px 0 8px;}"
        "p{margin:4px 0;font-size:13px;}"
        "table{border-collapse:collapse;background:white;font-size:13px;margin-top:6px;}"
        "th,td{border:1px solid #d8e1ec;padding:5px 10px;text-align:left;}"
        "th{background:#e4edf8;} ul{margin:6px 0;font-size:13px;}"
        "</style></head><body>"
    )

    @staticmethod
    def _make_table(headers, editable=False):
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        if not editable:
            table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    @staticmethod
    def _set_rows(table, rows):
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, text in enumerate(values):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(str(text)))
        table.resizeColumnsToContents()

    @staticmethod
    def _count_text(shown, truncated):
        return f"{shown:,}+" if truncated else f"{shown:,}"

    @staticmethod
    def _time_text(value):
        text = str(value or "")
        return text[:19].replace("T", " ") if text else "--"

    def build_data_management(self):
        """数据管理页：盘点工作区占用、核对索引一致性，并给出可安全清理的条目。"""
        settings = read_settings(self.workspace)
        self._extra_dirs = list(settings["extra_dirs"])
        self._storage_report = None
        self._cleanup_ready = False
        self._pending_preview = False

        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "把 IQ 生成、检测、分析、调制识别与训练相关的结果统一盘点：按目录统计占用，核对索引与磁盘"
            "是否一致（未入库文件、缺失文件、孤儿运行目录、源资产已删除的运行记录），并列出可安全清理的"
            "条目。清理只覆盖“已结束且超过保留天数”的任务工件与无主文件；已入库的信号资产、运行目录与 "
            "exports/ 永远不在可删范围内。扫描与导出本身是只读的，不会写入运行记录。额外目录（如 training/data、"
            "training/runs 或模型目录）只参与容量统计，不做索引一致性检查。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        bar = QtWidgets.QHBoxLayout()
        self.storage_scan_button = QtWidgets.QPushButton("扫描 / 刷新")
        self.storage_scan_button.setObjectName("primary")
        self.storage_scan_button.setToolTip("重新统计工作区与额外目录的占用（只读）")
        self.storage_scan_button.clicked.connect(lambda: self.scan_storage())
        bar.addWidget(self.storage_scan_button)
        self.storage_open_button = QtWidgets.QPushButton("打开工作目录")
        self.storage_open_button.setToolTip("用系统文件管理器打开工作区目录")
        self.storage_open_button.clicked.connect(self.open_workspace_folder)
        bar.addWidget(self.storage_open_button)
        self.storage_export_button = QtWidgets.QPushButton("导出报告")
        self.storage_export_button.setToolTip("把本次盘点写成 HTML 或 JSON 快照（不写入运行记录）")
        self.storage_export_button.clicked.connect(self.export_storage_report)
        bar.addWidget(self.storage_export_button)
        bar.addSpacing(18)
        bar.addWidget(QtWidgets.QLabel("额外目录"))
        self.storage_extra_combo = QtWidgets.QComboBox()
        self.storage_extra_combo.setMinimumWidth(240)
        self.storage_extra_combo.currentIndexChanged.connect(lambda _: self._sync_extra_buttons())
        bar.addWidget(self.storage_extra_combo, 2)
        self.storage_extra_add = QtWidgets.QPushButton("添加目录…")
        self.storage_extra_add.setToolTip("选择要纳入容量统计的目录（建议 training/data、training/runs）")
        self.storage_extra_add.clicked.connect(self.add_extra_dir)
        bar.addWidget(self.storage_extra_add)
        self.storage_extra_remove = QtWidgets.QPushButton("移除")
        self.storage_extra_remove.clicked.connect(self.remove_extra_dir)
        bar.addWidget(self.storage_extra_remove)
        bar.addStretch(1)
        layout.addLayout(bar)

        bar2 = QtWidgets.QHBoxLayout()
        bar2.addWidget(QtWidgets.QLabel("任务保留"))
        self.storage_retention = QtWidgets.QSpinBox()
        self.storage_retention.setRange(0, 3650)
        self.storage_retention.setSuffix(" 天")
        self.storage_retention.setValue(int(settings["job_retention_days"]))
        self.storage_retention.setToolTip("已结束的任务工件超过该天数才允许清理；0 表示全部可选")
        bar2.addWidget(self.storage_retention)
        self.storage_retention.valueChanged.connect(self._retention_changed)
        self.storage_retention.editingFinished.connect(self._retention_saved)
        self.storage_preview_button = QtWidgets.QPushButton("预览清理")
        self.storage_preview_button.setToolTip("按当前保留天数重新计算候选清单，仍不删除任何文件")
        self.storage_preview_button.clicked.connect(self.preview_storage_cleanup)
        bar2.addWidget(self.storage_preview_button)
        self.storage_cleanup_button = QtWidgets.QPushButton("执行清理")
        self.storage_cleanup_button.setEnabled(False)
        self.storage_cleanup_button.setToolTip("需先“预览清理”，再勾选要删除的条目；删除前会二次确认")
        self.storage_cleanup_button.clicked.connect(self.apply_storage_cleanup)
        bar2.addWidget(self.storage_cleanup_button)
        self.storage_cleanup_hint = QtWidgets.QLabel("尚未扫描。")
        bar2.addWidget(self.storage_cleanup_hint, 1)
        layout.addLayout(bar2)

        self.storage_overview = QtWidgets.QLabel("点击“扫描 / 刷新”统计工作区占用与一致性。")
        self.storage_overview.setWordWrap(True)
        self.storage_overview.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(self.storage_overview)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.storage_chart = pg.PlotWidget()
        self.storage_chart.setBackground("#ffffff")
        self.storage_chart.setMinimumHeight(150)
        self.storage_chart.setLabel("left", "占用", units="MiB")
        self.storage_chart.showGrid(x=False, y=True, alpha=.15)
        self.storage_chart.setMenuEnabled(False)
        splitter.addWidget(self.storage_chart)

        self.storage_tables = QtWidgets.QTabWidget()
        self.storage_asset_table = self._make_table(
            ["名称", "采样点", "采样率 Hz", "占用", "来源", "创建时间", "样本文件"])
        self.storage_run_table = self._make_table(
            ["类型", "占用", "索引冗余", "创建时间", "源资产", "结果文件"])
        self.storage_job_table = self._make_table(
            ["任务", "动作", "状态", "占用", "结束时间", "年龄(天)"])
        self.storage_export_table = self._make_table(["文件", "占用", "修改时间"])
        self.storage_issue_table = self._make_table(["问题", "路径", "占用", "说明"])
        self.storage_cleanup_table = self._make_table(
            ["选择", "类别", "路径", "占用", "判定依据"], editable=True)
        self.storage_cleanup_table.itemChanged.connect(lambda _: self._sync_cleanup_button())
        for widget, title in ((self.storage_asset_table, "信号资产"),
                              (self.storage_run_table, "运行记录"),
                              (self.storage_job_table, "任务工件"),
                              (self.storage_export_table, "导出文件"),
                              (self.storage_issue_table, "一致性"),
                              (self.storage_cleanup_table, "可清理项")):
            self.storage_tables.addTab(widget, title)
        self.storage_tables.addTab(self._build_collections_tab(), "信号集合")
        splitter.addWidget(self.storage_tables)
        splitter.setSizes([170, 640])
        layout.addWidget(splitter, 1)

        self._reload_extra_combo()
        self.refresh_collections_panel()
        return box

    # ------------------------------------------------------------- 信号集合子页
    def _build_collections_tab(self):
        """数据管理页的“信号集合”子页：成员增删、标注进度与参数统计。"""
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        left = QtWidgets.QVBoxLayout()
        self.collection_panel = QtWidgets.QListWidget()
        self.collection_panel.currentItemChanged.connect(self._collection_selected)
        left.addWidget(self.collection_panel, 1)
        buttons = QtWidgets.QHBoxLayout()
        for name, text, callback, tip in (
                ("new_collection_button", "新建集合…", self.new_collection,
                 "创建一个手工集合"),
                ("add_member_button", "加入所选资产", self.add_selected_asset_to_collection,
                 "把左侧当前选中的数据资产加入该集合"),
                ("remove_member_button", "移除所选资产",
                 self.remove_selected_asset_from_collection,
                 "只删除成员关系；资产、标注与数据版本不受影响"),
                ("archive_collection_button", "归档集合", self.archive_selected_collection,
                 "归档后集合不再出现在选择器中，成员关系保留"),
                ("migrate_button", "登记历史数据…", self.migrate_legacy,
                 "扫描工作区 datasets/ 与 training/runs/，把旧标注数据集与实验登记进索引（幂等）")):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            setattr(self, name, button)
            buttons.addWidget(button)
        left.addLayout(buttons)
        hint = QtWidgets.QLabel("集合成员来自左侧“数据资产”的当前选择；同一资产可属于多个集合，"
                                "标注挂在目标上、全集合共享同一份当前标签。")
        hint.setWordWrap(True)
        left.addWidget(hint)
        layout.addLayout(left, 2)
        self.collection_detail = QtWidgets.QPlainTextEdit()
        self.collection_detail.setReadOnly(True)
        self.collection_detail.setPlaceholderText("选择集合后显示概览")
        layout.addWidget(self.collection_detail, 3)
        return box

    def refresh_collections_panel(self, *_):
        current = self.collection_panel.currentItem()
        keep = current.data(QtCore.Qt.ItemDataRole.UserRole) if current else None
        self.collection_panel.clear()
        selected_row = -1
        for index, collection in enumerate(self.workspace.list_collections()):
            item = QtWidgets.QListWidgetItem(
                f"{collection['name']} · {collection['asset_count']} 个资产 · "
                f"{_SOURCE_KIND_LABELS.get(collection['source_kind'], collection['source_kind'])}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, collection["id"])
            self.collection_panel.addItem(item)
            if collection["id"] == keep:
                selected_row = index
        if selected_row >= 0:
            self.collection_panel.setCurrentRow(selected_row)
        elif self.collection_panel.count():
            self.collection_panel.setCurrentRow(0)
        else:
            self._collection_selected()

    def _selected_collection_id(self):
        item = self.collection_panel.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _collection_selected(self, *_):
        collection_id = self._selected_collection_id()
        if not collection_id:
            self.collection_detail.setPlainText(
                "尚未创建集合。点击「新建集合…」后，把左侧选中的数据资产加入。")
            return
        summary = self.workspace.collection_summary(collection_id)
        lines = [f"集合：{summary['name']}"
                 f"（{_SOURCE_KIND_LABELS.get(summary['source_kind'], summary['source_kind'])}）",
                 f"资产 {summary['asset_count']} · 目标 {summary['target_count']}"
                 f"（其中逐跳 {summary['hop_count']}）",
                 "来源分布：" + ("、".join(
                     f"{_SOURCE_KIND_LABELS.get(key, key)} {value}"
                     for key, value in summary["source_kinds"].items()) or "—"),
                 "", "任务标注进度："]
        if summary["task_sets"]:
            for progress in summary["task_sets"]:
                task_name = "检测" if progress["task"] == "detection" else "AMC"
                line = (f"· {task_name}「{progress['name']}」：已标注 {progress['labeled']}/"
                        f"{progress['targets']} · 待标注 {progress['pending']} · "
                        f"重新标注 {progress['reannotated']}")
                if progress.get("label_semantics"):
                    line += f" · 粒度 {progress['label_semantics']}"
                lines.append(line)
                coverage = progress["coverage"]
                lines.append(f"    覆盖度：完整 {coverage['complete']} · 部分 {coverage['partial']} · "
                             f"未标记 {coverage['unmarked']}")
        else:
            lines.append("· 该集合还没有任务标注集（检测 / AMC 标注集随标注或配方生成创建）")
        lines.append("")
        lines.append("参数统计（信号级，按目标数）：")
        for axis, title in (("modulation", "调制"), ("waveform_mode", "样式"),
                            ("is_hopping", "跳频"), ("snr_db", "SNR")):
            stats = self.workspace.target_axis_stats(collection_id, axis, scope="signal",
                                                     top=8)
            if stats:
                lines.append("· " + title + "：" + "、".join(
                    f"{item['label']} {item['count']}" for item in stats))
        self.collection_detail.setPlainText("\n".join(lines))

    def _collections_changed(self):
        self.refresh_collections()
        self.refresh_assets()
        self.refresh_collections_panel()

    def new_collection(self):
        name, accepted = QtWidgets.QInputDialog.getText(self, "新建信号集合", "集合名称")
        if not accepted or not name.strip():
            return
        try:
            collection = self.workspace.create_collection(name.strip(), created_by="gui")
        except ValueError as exc:
            self.status.setText(f"无法新建集合：{exc}")
            return
        self._collections_changed()
        for row in range(self.collection_panel.count()):
            if self.collection_panel.item(row).data(
                    QtCore.Qt.ItemDataRole.UserRole) == collection["id"]:
                self.collection_panel.setCurrentRow(row)
                break
        self.status.setText(f"已新建集合「{collection['name']}」")

    def add_selected_asset_to_collection(self):
        asset = self.selected_asset()
        collection_id = self._selected_collection_id()
        if asset is None:
            self.status.setText("请先在左侧选择数据资产")
            return
        if not collection_id:
            self.status.setText("请先选择或新建集合")
            return
        added = self.workspace.add_collection_member(collection_id, asset["id"],
                                                     added_by="gui")
        self._collections_changed()
        self.status.setText("已加入集合" if added else "该资产已在集合中")

    def remove_selected_asset_from_collection(self):
        asset = self.selected_asset()
        collection_id = self._selected_collection_id()
        if asset is None:
            self.status.setText("请先在左侧选择数据资产")
            return
        if not collection_id:
            self.status.setText("请先选择集合")
            return
        removed = self.workspace.remove_collection_member(collection_id, asset["id"])
        self._collections_changed()
        self.status.setText("已从集合移除（资产与标注保留）" if removed
                            else "该资产不在所选集合中")

    def archive_selected_collection(self):
        collection_id = self._selected_collection_id()
        if not collection_id:
            self.status.setText("请先选择集合")
            return
        collection = self.workspace.get_collection(collection_id)
        answer = QtWidgets.QMessageBox.question(
            self, "归档信号集合",
            f"归档「{collection['name']}」？\n集合将不再出现在选择器中，成员关系与标注保留，"
            "之后可重新启用。")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.workspace.archive_collection(collection_id)
        self._collections_changed()
        self.status.setText(f"已归档集合「{collection['name']}」")

    def migrate_legacy(self):
        """把旧标注数据集与旧实验登记进新索引（幂等，可重复点击）。"""
        self.start_job("migrate_legacy")

    def scan_storage(self, preview=False):
        """扫描工作区；``preview=True`` 时视为一次显式清理预览（解锁删除按钮）。"""
        self._pending_preview = bool(preview)
        self.start_job("storage_report", extra_dirs=list(self._extra_dirs),
                       job_retention_days=int(self.storage_retention.value()))

    def _retention_changed(self, _value):
        self._cleanup_ready = False
        if hasattr(self, "storage_cleanup_hint"):
            self.storage_cleanup_hint.setText("保留天数已改变：请重新“预览清理”后再删除。")
            self._sync_cleanup_button()

    def _retention_saved(self):
        try:
            write_settings(self.workspace, job_retention_days=int(self.storage_retention.value()))
        except (OSError, ValueError) as exc:
            self.status.setText(f"保留天数保存失败：{exc}")

    def _reload_extra_combo(self):
        self.storage_extra_combo.blockSignals(True)
        self.storage_extra_combo.clear()
        self.storage_extra_combo.addItems(self._extra_dirs)
        self.storage_extra_combo.blockSignals(False)
        self._sync_extra_buttons()

    def _sync_extra_buttons(self):
        self.storage_extra_remove.setEnabled(self.storage_extra_combo.count() > 0)

    def _persist_extra_dirs(self):
        try:
            write_settings(self.workspace, extra_dirs=list(self._extra_dirs))
        except (OSError, ValueError) as exc:
            self.status.setText(f"额外目录保存失败：{exc}")

    def add_extra_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择要纳入统计的额外目录", str(self.workspace.root.parent))
        if not path:
            return
        resolved = str(Path(path).expanduser().resolve())
        if Path(resolved) == Path(self.workspace.root).resolve():
            self.status.setText("工作区目录已包含在统计中，无需重复添加")
            return
        if any(Path(item).expanduser().resolve() == Path(resolved) for item in self._extra_dirs):
            self.status.setText("该目录已在统计列表中")
            return
        self._extra_dirs.append(resolved)
        self._persist_extra_dirs()
        self._reload_extra_combo()
        self.scan_storage()

    def remove_extra_dir(self):
        index = self.storage_extra_combo.currentIndex()
        if not 0 <= index < len(self._extra_dirs):
            return
        self._extra_dirs.pop(index)
        self._persist_extra_dirs()
        self._reload_extra_combo()
        self.scan_storage()

    def open_workspace_folder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.workspace.root)))

    def _checked_cleanup_paths(self):
        table = self.storage_cleanup_table
        paths = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.checkState() == QtCore.Qt.CheckState.Checked:
                paths.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return paths

    def _sync_cleanup_button(self):
        self.storage_cleanup_button.setEnabled(
            bool(self._cleanup_ready) and bool(self._checked_cleanup_paths()))

    def _render_storage_chart(self, report):
        self.storage_chart.clear()
        items = [(item["label"], item["bytes"]) for item in report["categories"] if item["bytes"]]
        items += [(f"额外 · {Path(row['path']).name}", row["bytes"])
                  for row in report.get("extra_dirs") or [] if row.get("bytes")]
        if not items:
            return
        values = [value / (1024 ** 2) for _, value in items]
        bars = pg.BarGraphItem(x=np.arange(len(items), dtype=float), height=np.asarray(values),
                               width=0.6, brush="#4c86c9", pen=pg.mkPen("#2c5f9e"))
        self.storage_chart.addItem(bars)
        self.storage_chart.getAxis("bottom").setTicks(
            [[(index, label) for index, (label, _) in enumerate(items)]])
        self.storage_chart.setYRange(0, max(max(values) * 1.15, 1e-6), padding=0)

    def _fill_storage_tables(self, report):
        tables = report["tables"]
        asset_rows = [(item["name"], f"{item['sample_count']:,}", f"{item['sample_rate_hz']:g}",
                       format_bytes(item["bytes"]), item["source"],
                       self._time_text(item["created_at"]),
                       "正常" if item["exists"] else "缺失")
                      for item in tables["assets"]]
        self._set_rows(self.storage_asset_table, asset_rows or [("（无）", "", "", "0 B", "", "", "")])

        run_rows = [(RUN_KIND_LABELS.get(item["kind"], item["kind"]), format_bytes(item["bytes"]),
                     format_bytes(item["sqlite_bytes"]), self._time_text(item["created_at"]),
                     (item["source_id"] or "--")[:8], "正常" if item["exists"] else "缺失")
                    for item in tables["runs"]]
        self._set_rows(self.storage_run_table, run_rows or [("（无）", "0 B", "0 B", "", "", "")])

        job_rows = [(item["id"][:8], item["action"], item["state"], format_bytes(item["bytes"]),
                     self._time_text(item["finished"]), f"{item['age_days']:.1f}")
                    for item in tables["jobs"]]
        self._set_rows(self.storage_job_table, job_rows or [("（无）", "", "", "0 B", "", "")])

        export_rows = [(item["name"], format_bytes(item["bytes"]),
                        self._time_text(item["modified"])) for item in tables["exports"]]
        self._set_rows(self.storage_export_table, export_rows or [("（无）", "0 B", "")])

        consistency = report["consistency"]
        issues = []
        for item in consistency["orphan_files"]:
            issues.append(("未入库文件", item["path"], item["bytes"], item["reason"]))
        for item in consistency["tmp_files"]:
            issues.append(("原子写残留", item["path"], item["bytes"], item["reason"]))
        for item in consistency["missing_files"]:
            issues.append(("入库但文件缺失", item["path"], item["bytes"], item["reason"]))
        for item in consistency["orphan_run_dirs"]:
            issues.append(("未入库运行目录", item["path"], item["bytes"], item["reason"]))
        for item in tables["orphan_runs"]:
            issues.append(("源资产已删除", item["path"], item["bytes"], item["reason"]))
        self._set_rows(self.storage_issue_table,
                       [(kind, path, format_bytes(size), reason)
                        for kind, path, size, reason in issues]
                       or [("无", "--", "0 B", "索引与磁盘一致")])
        self.storage_tables.setTabText(4, f"一致性（{len(issues)}）")

    def _fill_cleanup_table(self, report):
        targets = report["cleanup"]["targets"]
        table = self.storage_cleanup_table
        kind_labels = {"job": "过期任务", "orphan_file": "无主文件", "orphan_run": "无主目录"}
        table.blockSignals(True)
        table.setRowCount(len(targets))
        for row, item in enumerate(targets):
            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
                           | QtCore.Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(QtCore.Qt.CheckState.Unchecked)
            check.setData(QtCore.Qt.ItemDataRole.UserRole, item["path"])
            table.setItem(row, 0, check)
            for column, text in enumerate((kind_labels.get(item["kind"], item["kind"]),
                                           item["path"], format_bytes(item["bytes"]),
                                           item["reason"]), start=1):
                table.setItem(row, column, QtWidgets.QTableWidgetItem(str(text)))
        table.resizeColumnsToContents()
        table.blockSignals(False)
        hint = (f"候选 {len(targets)} 项 · 共 {format_bytes(report['cleanup']['bytes'])} · "
                f"运行中/未结束任务已跳过 {report['cleanup']['skipped_running']} 个")
        if not targets:
            hint = "没有可清理项（任务都未超过保留天数，且没有无主文件）。"
        elif not self._cleanup_ready:
            hint += " · 先点“预览清理”解锁删除"
        else:
            hint += " · 勾选后点“执行清理”"
        self.storage_cleanup_hint.setText(hint)
        self.storage_tables.setTabText(5, f"可清理项（{len(targets)}）")
        self._sync_cleanup_button()

    def _render_data_management(self, report):
        self._storage_report = report
        self._cleanup_ready = bool(self._pending_preview)
        self._pending_preview = False
        totals = report["totals"]
        overview = [
            f"<b>工作区</b> {report['root']}",
            f"<b>总占用 {format_bytes(totals['bytes'])}</b> · {totals['files']:,} 个文件"
            + (f" · 其中额外目录 {format_bytes(totals['extra_bytes'])}" if totals.get("extra_bytes") else ""),
            " · ".join(f"{item['label']} {format_bytes(item['bytes'])}"
                       for item in report["categories"] if item["bytes"]) or "（工作区为空）",
            f"索引冗余 {format_bytes(report['catalog']['index_bytes'])}（结果 JSON 在 SQLite 与磁盘各存一份）· "
            f"资产 {report['catalog']['assets']} · 运行 {report['catalog']['runs']} · "
            f"任务 {self._count_text(len(report['tables']['jobs']), report['tables']['jobs_truncated'])} · "
            f"导出 {self._count_text(len(report['tables']['exports']), report['tables']['exports_truncated'])}"
        ]
        for row in report.get("extra_dirs") or []:
            overview.append(f"　额外目录 {row['path']} "
                            + (f"{format_bytes(row['bytes'])} · {row['files']:,} 个文件"
                               if row["exists"] else "不存在"))
        for item in report["warnings"]:
            overview.append(f"⚠ {item}")
        self.storage_overview.setText("<br>".join(overview))
        self._render_storage_chart(report)
        self._fill_storage_tables(report)
        self._fill_cleanup_table(report)
        self.tabs.setCurrentIndex(self._page_index("数据管理"))

    def preview_storage_cleanup(self):
        self.scan_storage(preview=True)

    def apply_storage_cleanup(self):
        paths = self._checked_cleanup_paths()
        if not paths:
            self.status.setText("请先在“可清理项”里勾选要删除的条目")
            return
        cached = {item["path"]: item for item in (self._storage_report or {}).get(
            "cleanup", {}).get("targets", [])}
        total = sum(cached.get(path, {}).get("bytes", 0) for path in paths)
        listing = "\n".join(paths[:12]) + ("\n…" if len(paths) > 12 else "")
        answer = QtWidgets.QMessageBox.question(
            self, "确认清理",
            f"将永久删除 {len(paths)} 个条目，共 {format_bytes(total)}：\n\n{listing}\n\n"
            "已入库的信号资产、运行目录与 exports/ 不在可删范围内；该操作不可撤销。",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.setText("已取消清理")
            return
        self._cleanup_ready = False
        self._sync_cleanup_button()
        self.start_job("storage_cleanup", targets=paths,
                       job_retention_days=int(self.storage_retention.value()))

    def _render_storage_cleanup(self, result):
        removed = result.get("removed") or []
        skipped = result.get("skipped") or []
        errors = result.get("errors") or []
        lines = [f"已删除 {len(removed)} 个条目，释放 {format_bytes(result.get('bytes'))}"]
        if skipped:
            lines.append(f"跳过 {len(skipped)} 项（已不再是候选或不在可删范围）")
        if errors:
            lines.append(f"失败 {len(errors)} 项：{errors[0].get('error')}")
        detail = ""
        if removed:
            detail = "\n\n" + "\n".join(item["path"] for item in removed[:12])
            if len(removed) > 12:
                detail += "\n…"
        QtWidgets.QMessageBox.information(self, "清理完成", "\n".join(lines) + detail)
        self.storage_overview.setText("<br>".join(lines) + "<br>正在重新扫描…")
        self._cleanup_ready = False
        self.scan_storage()

    def export_storage_report(self):
        report = self._storage_report
        if report is None:
            self.status.setText("请先扫描一次，再导出盘点报告")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出数据盘点报告", str(self.workspace.root / "storage_report.html"),
            "HTML (*.html);;JSON (*.json)")
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".json":
                Path(path).write_text(
                    json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2),
                    encoding="utf-8")
            else:
                Path(path).write_text(self._storage_report_html(report), encoding="utf-8")
            self.status.setText(f"盘点报告已导出：{path}")
        except (OSError, ValueError) as exc:
            self.status.setText(f"导出失败：{exc}")

    def _storage_report_html(self, report):
        import html as html_module

        def esc(value):
            return html_module.escape(str(value))

        def table(headers, rows):
            body = "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>"
                           for row in rows)
            head = "".join(f"<th>{esc(cell)}</th>" for cell in headers)
            return f"<table><tr>{head}</tr>{body}</table>"

        totals = report["totals"]
        parts = [self._STORAGE_HEAD, "<h1>数据盘点报告</h1>",
                 f"<p>工作区：{esc(report['root'])}</p>",
                 f"<p>生成时间：{esc(report['generated_at'])}</p>",
                 f"<p>总占用：<b>{esc(format_bytes(totals['bytes']))}</b> · "
                 f"{esc(totals['files'])} 个文件（其中额外目录 "
                 f"{esc(format_bytes(totals['extra_bytes']))}）</p>",
                 "<h2>目录占用</h2>",
                 table(["目录", "文件数", "占用"],
                       [(item["label"], item["files"], format_bytes(item["bytes"]))
                        for item in report["categories"]]
                       + [(f"额外 · {row['path']}",
                           row["files"] if row["exists"] else "--",
                           format_bytes(row["bytes"]) if row["exists"] else "不存在")
                          for row in report.get("extra_dirs") or []])]
        parts.append("<h2>按类型统计的运行记录</h2>")
        parts.append(table(["类型", "条数", "磁盘占用", "索引冗余"],
                           [(item["label"], item["count"], format_bytes(item["disk_bytes"]),
                             format_bytes(item["sqlite_bytes"])) for item in report["runs_by_kind"]]
                           or [("（无）", 0, "0 B", "0 B")]))
        parts.append(f"<h2>可清理项（保留 {esc(report['cleanup']['retention_days'])} 天）</h2>")
        parts.append(table(["路径", "占用", "判定依据"],
                           [(item["path"], format_bytes(item["bytes"]), item["reason"])
                            for item in report["cleanup"]["targets"]]
                           or [("（无）", "0 B", "没有可清理项")]))
        parts.append("<h2>提示</h2><ul>"
                     + ("".join(f"<li>{esc(item)}</li>" for item in report["warnings"])
                        or "<li>索引与磁盘一致，没有需要关注的问题。</li>")
                     + "</ul>")
        parts.append("<p>本报告由“数据管理”页生成，扫描过程为只读，未写入运行记录。</p>")
        parts.append("</body></html>")
        return "".join(parts)
