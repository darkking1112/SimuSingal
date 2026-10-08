"""数据管理页（mixin）：容量盘点、一致性检查、安全清理与“信号集合”子页。"""
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ...core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ...storage.maintenance import RUN_KIND_LABELS, format_bytes, read_settings, write_settings
from ...data import Workspace
from ...services import transfer
from ..constants import (IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                         IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                         IMPORT_COL_NAME, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                         IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                         IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                         IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                         PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                         _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)
from ..helpers import (_asset_format, _fmt_hz, _fmt_metric, _fmt_span,
                       _mirrored_spectrum)
from ..runner import _run_task


class DataPageMixin:
    # ------------------------------------------------------------ 数据管理页
    #: 集合多样性分析维度：(统计轴, 显示标题)；轴名与 ``target_axis_stats`` 一致。
    _DIVERSITY_AXES = (("waveform_mode", "信号样式"), ("modulation", "调制"),
                       ("snr_db", "SNR"), ("is_hopping", "跳频"),
                       ("bandwidth_hz", "带宽"), ("symbol_rate_baud", "符号率"))

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

    def _set_tab_text(self, widget, title):
        """按控件定位页签下标，避免页签顺序变动后写错页签标题。"""
        index = self.storage_tables.indexOf(widget)
        if index >= 0:
            self.storage_tables.setTabText(index, title)

    @staticmethod
    def _count_text(shown, truncated):
        return f"{shown:,}+" if truncated else f"{shown:,}"

    @staticmethod
    def _time_text(value):
        text = str(value or "")
        return text[:19].replace("T", " ") if text else "--"

    def build_data_management(self):
        """数据管理页：盘点当前工作目录占用、核对索引一致性，并给出可安全清理的条目。"""
        settings = read_settings(self.workspace)
        self._storage_report = None
        self._cleanup_ready = False
        self._pending_preview = False

        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        intro = QtWidgets.QLabel(
            "把 IQ 生成、检测、分析、调制识别与训练相关的结果统一盘点：统计当前工作目录"
            "（workspace_data）的占用，核对索引与磁盘是否一致（未入库文件、缺失文件、孤儿运行目录、"
            "源资产已删除的运行记录），并列出可安全清理的条目；同时在“信号集合”子页维护集合成员、"
            "查看标注进度与多样性，在“模型管理”子页查看训练产出的模型（训练时间、大小与参数）"
            "并重命名或删除。清理只覆盖“已结束且超过保留天数”的任务工件与无主文件；"
            "已入库的信号资产与运行目录永远不在可删范围内。扫描本身是只读的，不会写入运行记录。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        from common.gui import TaskBanner
        self.data_banner = TaskBanner("数据管理", "取消本次任务")
        self.register_task_banner("数据管理", self.data_banner)
        layout.addWidget(self.data_banner)

        bar = QtWidgets.QHBoxLayout()
        self.storage_scan_button = QtWidgets.QPushButton("扫描 / 刷新")
        self.storage_scan_button.setObjectName("primary")
        self.storage_scan_button.setToolTip("重新统计当前工作目录的占用（只读）")
        self.storage_scan_button.clicked.connect(lambda: self.scan_storage())
        bar.addWidget(self.storage_scan_button)
        self.storage_open_button = QtWidgets.QPushButton("打开工作目录")
        self.storage_open_button.setToolTip("用系统文件管理器打开当前工作区目录")
        self.storage_open_button.clicked.connect(self.open_workspace_folder)
        bar.addWidget(self.storage_open_button)
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
        self.storage_run_table = self._make_table(
            ["类型", "占用", "索引冗余", "创建时间", "源资产", "结果文件"])
        self.storage_job_table = self._make_table(
            ["任务", "动作", "状态", "占用", "结束时间", "年龄(天)"])
        self.storage_issue_table = self._make_table(["问题", "路径", "占用", "说明"])
        self.storage_cleanup_table = self._make_table(
            ["选择", "类别", "路径", "占用", "判定依据"], editable=True)
        self.storage_cleanup_table.itemChanged.connect(lambda _: self._sync_cleanup_button())
        self.storage_tables.addTab(self._build_collections_tab(), "信号集合")
        self.storage_tables.addTab(self._build_models_tab(), "模型管理")
        for widget, title in ((self.storage_run_table, "运行记录"),
                              (self.storage_job_table, "任务工件"),
                              (self.storage_issue_table, "一致性"),
                              (self.storage_cleanup_table, "可清理项")):
            self.storage_tables.addTab(widget, title)
        splitter.addWidget(self.storage_tables)
        splitter.setSizes([170, 640])
        layout.addWidget(splitter, 1)

        self.refresh_collections_panel()
        return box

    # ------------------------------------------------------------- 信号集合子页
    def _build_collections_tab(self):
        """数据管理页的“信号集合”子页：成员增删、标注进度与多样性分析。"""
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
                 "归档后集合不再出现在选择器中，成员关系保留，可随时恢复"),
                ("restore_collection_button", "恢复集合", self.restore_selected_collection,
                 "把选中的已归档集合恢复为正常状态"),
                ("export_collection_button", "导出集合包…", self.export_collection_package,
                 "把选中集合的资产数据与标注（目标、参考参数、标签、覆盖度、类别字典）"
                 "打包成 zip，可在另一台电脑用「导入集合包…」重建"),
                ("import_collection_button", "导入集合包…", self.import_collection_package,
                 "从集合包重建集合：先体检，通过后新建集合与资产（新 id，不覆盖已有数据）")):
            button = QtWidgets.QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(callback)
            setattr(self, name, button)
            buttons.addWidget(button)
        left.addLayout(buttons)
        self.collection_show_archived = QtWidgets.QCheckBox("显示已归档")
        self.collection_show_archived.setToolTip("勾选后在列表里一并列出已归档集合，便于恢复")
        self.collection_show_archived.toggled.connect(
            lambda _: self.refresh_collections_panel())
        left.addWidget(self.collection_show_archived)
        hint = QtWidgets.QLabel("集合成员来自左侧“数据资产”的当前选择；同一资产可属于多个集合，"
                                "标注挂在目标上、全集合共享同一份当前标签。归档集合可随时恢复。")
        hint.setWordWrap(True)
        left.addWidget(hint)
        layout.addLayout(left, 2)
        right = QtWidgets.QVBoxLayout()
        self.collection_detail = QtWidgets.QPlainTextEdit()
        self.collection_detail.setReadOnly(True)
        self.collection_detail.setPlaceholderText("选择集合后显示概览")
        right.addWidget(self.collection_detail, 3)
        diversity = QtWidgets.QGroupBox("多样性分析")
        diversity_layout = QtWidgets.QVBoxLayout(diversity)
        picker = QtWidgets.QHBoxLayout()
        picker.addWidget(QtWidgets.QLabel("维度"))
        self.diversity_axis = QtWidgets.QComboBox()
        for axis, title in self._DIVERSITY_AXES:
            self.diversity_axis.addItem(title, axis)
        self.diversity_axis.currentIndexChanged.connect(lambda _: self._render_diversity())
        picker.addWidget(self.diversity_axis, 1)
        diversity_layout.addLayout(picker)
        self.diversity_chart = pg.PlotWidget()
        self.diversity_chart.setBackground("#ffffff")
        self.diversity_chart.setMinimumHeight(150)
        self.diversity_chart.setLabel("left", "目标数")
        self.diversity_chart.showGrid(x=False, y=True, alpha=.15)
        self.diversity_chart.setMenuEnabled(False)
        diversity_layout.addWidget(self.diversity_chart)
        self.diversity_summary = QtWidgets.QLabel("选择集合后显示多样性指标。")
        self.diversity_summary.setWordWrap(True)
        diversity_layout.addWidget(self.diversity_summary)
        right.addWidget(diversity, 2)
        layout.addLayout(right, 3)
        return box

    def refresh_collections_panel(self, *_):
        current = self.collection_panel.currentItem()
        keep = current.data(QtCore.Qt.ItemDataRole.UserRole) if current else None
        show_archived = bool(getattr(self, "collection_show_archived", None)
                             and self.collection_show_archived.isChecked())
        self.collection_panel.clear()
        selected_row = -1
        for index, collection in enumerate(
                self.workspace.list_collections(include_archived=show_archived)):
            text = (f"{collection['name']} · {collection['asset_count']} 个资产 · "
                    f"{_SOURCE_KIND_LABELS.get(collection['source_kind'], collection['source_kind'])}")
            if collection.get("archived_at"):
                text += " · 已归档"
            item = QtWidgets.QListWidgetItem(text)
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

    def _sync_collection_buttons(self):
        collection_id = self._selected_collection_id()
        archived = bool(collection_id
                        and self.workspace.get_collection(collection_id).get("archived_at"))
        self.archive_collection_button.setEnabled(bool(collection_id) and not archived)
        self.restore_collection_button.setEnabled(archived)
        self.export_collection_button.setEnabled(bool(collection_id))

    def _collection_selected(self, *_):
        collection_id = self._selected_collection_id()
        self._sync_collection_buttons()
        if not collection_id:
            self.collection_detail.setPlainText(
                "尚未创建集合。点击「新建集合…」后，把左侧选中的数据资产加入。")
            self._render_diversity()
            return
        summary = self.workspace.collection_summary(collection_id)
        lines = [f"集合：{summary['name']}"
                 f"（{_SOURCE_KIND_LABELS.get(summary['source_kind'], summary['source_kind'])}）"
                 + ("　[已归档]" if summary.get("archived_at") else ""),
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
        self._render_diversity()

    # --------------------------------------------------------- 集合包迁移
    def export_collection_package(self):
        """把选中的集合（资产数据 + 标注）导出成一个集合包（zip）。"""
        collection_id = self._selected_collection_id()
        if not collection_id:
            self.collection_detail.setPlainText("请先在列表里选择一个集合再导出。")
            return
        collection = self.workspace.get_collection(collection_id)
        target = Path(self.workspace.root) / "exports"
        target.mkdir(parents=True, exist_ok=True)
        default = str(target / transfer.default_package_name(f"集合包-{collection['name']}"))
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出集合包", default, "集合包 (*.zip)")
        if not path:
            return
        try:
            result = self._with_wait_cursor(
                lambda: transfer.export_collection(self.workspace, collection_id, path))
        except (transfer.TransferError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.collection_detail.appendPlainText(
            f"\n[导出] {result['path']}（{format_bytes(result['size_bytes'])}）"
            f"：资产 {result['assets']} · 目标 {result['targets']} · 标签 {result['labels']}"
            f" · 覆盖度 {result['coverage']}；在另一台电脑用「导入集合包…」重建。")

    def import_collection_package(self):
        """从集合包重建集合：先体检（格式与摘要），再新建集合与资产。"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "导入集合包", str(Path(self.workspace.root) / "exports"), "集合包 (*.zip)")
        if not path:
            return
        try:
            report = transfer.inspect_package(path)
        except (transfer.TransferError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "集合包不可用", str(exc))
            return
        if "collection" not in report:
            QtWidgets.QMessageBox.warning(self, "集合包不可用", "这不是一个集合包（zip）")
            return
        counts = report.get("counts") or {}
        if not report.get("ok"):
            QtWidgets.QMessageBox.warning(
                self, "集合包校验不通过", "；".join(report.get("problems") or ["内容不完整"]))
            return
        info = report.get("collection") or {}
        if QtWidgets.QMessageBox.question(
                self, "导入集合包",
                f"集合：{info.get('name')}\n"
                f"资产 {counts.get('assets')} · 目标 {counts.get('targets')} · "
                f"标签 {counts.get('labels')} · 覆盖度 {counts.get('coverage')}\n"
                f"标注集：" + "、".join(f"{item['task']}/{item['name']}"
                                       for item in report.get("task_sets") or []) +
                "\n\n将新建一个集合（重名自动加序号），已有集合与资产不会被修改。继续？") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            result = self._with_wait_cursor(
                lambda: transfer.import_collection(self.workspace, path))
        except (transfer.TransferError, OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.refresh_collections_panel()
        self.refresh_assets()
        for index in range(self.collection_panel.count()):
            item = self.collection_panel.item(index)
            if item.data(QtCore.Qt.ItemDataRole.UserRole) == result["collection_id"]:
                self.collection_panel.setCurrentRow(index)
                break
        self.status.setText(
            f"已导入集合「{result['collection']}」：资产 {result['assets']} · "
            f"目标 {result['targets']} · 标签 {result['labels']}"
            + (f"（跳过 {len(result['skipped_labels'])} 条标签）"
               if result.get("skipped_labels") else ""))

    def _with_wait_cursor(self, action):
        """跑可能耗时的打包/解包：显示等待光标并禁用集合操作按钮。"""
        buttons = [getattr(self, name, None) for name in
                   ("export_collection_button", "import_collection_button",
                    "new_collection_button", "add_member_button", "remove_member_button",
                    "archive_collection_button", "restore_collection_button")]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        for button in buttons:
            if button is not None:
                button.setEnabled(False)
        try:
            return action()
        finally:
            for button in buttons:
                if button is not None:
                    button.setEnabled(True)
            QtWidgets.QApplication.restoreOverrideCursor()
            self._sync_collection_buttons()

    def _render_diversity(self):
        """把所选集合在“多样性分析”维度上的分布画成柱形图并给出指标。"""
        self.diversity_chart.clear()
        collection_id = self._selected_collection_id()
        if not collection_id:
            self.diversity_summary.setText("选择集合后显示多样性指标。")
            return
        report = self.workspace.collection_diversity(collection_id)
        overall = report["overall"]
        if not overall["targets"]:
            self.diversity_summary.setText("该集合没有可统计的目标（信号级）。")
            return
        info = report["axes"].get(self.diversity_axis.currentData()) or {}
        items = list(info.get("items") or [])
        if info.get("unknown"):
            items.append(("未知", info["unknown"]))
        if items:
            values = [count for _, count in items]
            bars = pg.BarGraphItem(x=np.arange(len(items), dtype=float),
                                   height=np.asarray(values, dtype=float), width=0.6,
                                   brush="#4c86c9", pen=pg.mkPen("#2c5f9e"))
            self.diversity_chart.addItem(bars)
            self.diversity_chart.getAxis("bottom").setTicks(
                [[(index, label) for index, (label, _) in enumerate(items)]])
            self.diversity_chart.setYRange(0, max(max(values) * 1.15, 1), padding=0)
        self.diversity_summary.setText(
            f"取值数 {info['distinct']} · 归一化多样性 {info['diversity']:.2f} · "
            f"最高占比 {info['top_share']:.0%} · 未知 {info['unknown_share']:.0%}"
            f"　|　综合多样性 {overall['diversity']:.2f}"
            f"（有效维度 {overall['effective_axes']}/{len(self._DIVERSITY_AXES)}）"
            f" · 组合变体 {overall['combination_variants']}/{overall['targets']}"
            f"（覆盖 {overall['combination_share']:.0%}）")

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
            "勾选“显示已归档”后可随时恢复。")
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.workspace.archive_collection(collection_id)
        self._collections_changed()
        self.status.setText(f"已归档集合「{collection['name']}」")

    def restore_selected_collection(self):
        collection_id = self._selected_collection_id()
        if not collection_id:
            self.status.setText("请先选择集合")
            return
        collection = self.workspace.get_collection(collection_id)
        if not collection.get("archived_at"):
            self.status.setText(f"集合「{collection['name']}」未归档，无需恢复")
            return
        self.workspace.archive_collection(collection_id, archived=False)
        self._collections_changed()
        self.status.setText(f"已恢复集合「{collection['name']}」")

    def scan_storage(self, preview=False):
        """扫描工作区；``preview=True`` 时视为一次显式清理预览（解锁删除按钮）。"""
        self._pending_preview = bool(preview)
        self.start_job("storage_report", owner="数据管理", label="数据盘点",
                       cancel_text="取消本次盘点",
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
        run_rows = [(RUN_KIND_LABELS.get(item["kind"], item["kind"]), format_bytes(item["bytes"]),
                     format_bytes(item["sqlite_bytes"]), self._time_text(item["created_at"]),
                     (item["source_id"] or "--")[:8], "正常" if item["exists"] else "缺失")
                    for item in tables["runs"]]
        self._set_rows(self.storage_run_table, run_rows or [("（无）", "0 B", "0 B", "", "", "")])

        job_rows = [(item["id"][:8], item["action"], item["state"], format_bytes(item["bytes"]),
                     self._time_text(item["finished"]), f"{item['age_days']:.1f}")
                    for item in tables["jobs"]]
        self._set_rows(self.storage_job_table, job_rows or [("（无）", "", "", "0 B", "", "")])

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
        self._set_tab_text(self.storage_issue_table, f"一致性（{len(issues)}）")

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
        self._set_tab_text(self.storage_cleanup_table, f"可清理项（{len(targets)}）")
        self._sync_cleanup_button()

    def _render_data_management(self, report):
        self._storage_report = report
        self._cleanup_ready = bool(self._pending_preview)
        self._pending_preview = False
        totals = report["totals"]
        overview = [
            f"<b>总占用 {format_bytes(totals['bytes'])}</b> · {totals['files']:,} 个文件",
            " · ".join(f"{item['label']} {format_bytes(item['bytes'])}"
                       for item in report["categories"] if item["bytes"]) or "（工作区为空）",
            f"索引冗余 {format_bytes(report['catalog']['index_bytes'])}（结果 JSON 在 SQLite 与磁盘各存一份）· "
            f"资产 {report['catalog']['assets']} · 运行 {report['catalog']['runs']} · "
            f"任务 {self._count_text(len(report['tables']['jobs']), report['tables']['jobs_truncated'])}"
        ]
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
            "已入库的信号资产与运行目录不在可删范围内；该操作不可撤销。",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status.setText("已取消清理")
            return
        self._cleanup_ready = False
        self._sync_cleanup_button()
        self.start_job("storage_cleanup", owner="数据管理", label="数据清理",
                       cancel_text="取消本次清理",
                       targets=paths,
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
