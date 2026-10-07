"""数据管理页的「模型管理」子页：查看、重命名、删除与跳转使用训练产出的模型。"""
from PySide6 import QtCore, QtGui, QtWidgets

from ...services import model_store
from ...storage.maintenance import format_bytes

STATUS_TEXT = {"ok": "正常", "missing": "文件缺失", "changed": "文件已变动", "corrupt": "元数据损坏"}
COLUMNS = ("名称", "类型", "用途", "训练时间", "大小", "参数摘要", "来源实验", "状态")


class ModelsPageMixin:
    """模型库（``<workspace>/training/models``）的只读盘点 + 重命名/删除/跳转使用。"""

    def _build_models_tab(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        bar = QtWidgets.QHBoxLayout()
        self.models_scan_button = QtWidgets.QPushButton("扫描既有模型")
        self.models_scan_button.setObjectName("primary")
        self.models_scan_button.setToolTip(
            "把训练目录里已成功完成、但还没登记的模型按默认命名补进模型库（幂等，可反复点）")
        self.models_scan_button.clicked.connect(self.scan_model_library)
        bar.addWidget(self.models_scan_button)
        self.models_open_button = QtWidgets.QPushButton("打开模型库目录")
        self.models_open_button.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(
                str(model_store.models_root(self.workspace.root)))))
        bar.addWidget(self.models_open_button)
        bar.addStretch(1)
        layout.addLayout(bar)

        self.models_table = self._make_table(COLUMNS)
        self.models_table.setToolTip("训练成功即自动入库；双击一行可在对应页面直接使用该模型")
        self.models_table.itemSelectionChanged.connect(self._models_selection_changed)
        self.models_table.itemDoubleClicked.connect(lambda _: self.use_selected_model())
        layout.addWidget(self.models_table, 1)

        self.model_hint = QtWidgets.QLabel(
            "模型库位于工作区 training/models，训练成功自动入库，名称默认为"
            "「模型类型-训练用途-时间」。删除只移除模型库条目，训练运行目录与实验记录保留。")
        self.model_hint.setWordWrap(True)
        layout.addWidget(self.model_hint)

        detail_row = QtWidgets.QHBoxLayout()
        self.model_detail = QtWidgets.QPlainTextEdit()
        self.model_detail.setReadOnly(True)
        self.model_detail.setPlaceholderText("选择模型后显示训练时间、大小、参数与指标。")
        detail_row.addWidget(self.model_detail, 1)
        actions = QtWidgets.QVBoxLayout()
        self.model_use_button = QtWidgets.QPushButton("在页面中使用")
        self.model_use_button.setToolTip("切到对应页面并选中该模型清单")
        self.model_use_button.clicked.connect(self.use_selected_model)
        actions.addWidget(self.model_use_button)
        self.model_rename_button = QtWidgets.QPushButton("重命名…")
        self.model_rename_button.clicked.connect(self.rename_selected_model)
        actions.addWidget(self.model_rename_button)
        self.model_delete_button = QtWidgets.QPushButton("删除模型")
        self.model_delete_button.clicked.connect(self.delete_selected_model)
        actions.addWidget(self.model_delete_button)
        actions.addStretch(1)
        detail_row.addLayout(actions)
        layout.addLayout(detail_row, 1)
        self.refresh_models_panel()
        return box

    # ------------------------------------------------------------------ 刷新
    def refresh_models_panel(self, *_):
        keep = self._selected_model_name() if hasattr(self, "models_table") else None
        models = model_store.list_models(self.workspace.root)
        table = self.models_table
        table.blockSignals(True)
        table.setRowCount(len(models))
        selected_row = -1
        for row, entry in enumerate(models):
            values = (entry["name"], entry.get("model_type") or "", entry.get("purpose_title") or "",
                      self._time_text(entry.get("created_at")), format_bytes(entry.get("size_bytes")),
                      self._params_text(entry), entry.get("source_run") or "--",
                      STATUS_TEXT.get(entry.get("status"), entry.get("status") or ""))
            for column, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(text))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, entry["name"])
                if entry.get("status") != "ok":
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#b00020")))
                table.setItem(row, column, item)
            if entry["name"] == keep:
                selected_row = row
        table.blockSignals(False)
        if selected_row < 0 and table.rowCount():
            selected_row = 0
        if selected_row >= 0:
            table.selectRow(selected_row)
        table.resizeColumnsToContents()
        self._models_selection_changed()

    @staticmethod
    def _params_text(entry):
        params = entry.get("params") or {}
        if entry.get("purpose") == "amc":
            parts = [f"{params.get('model')}" if params.get("model") else "",
                     f"{params.get('window_samples')} 点" if params.get("window_samples") else "",
                     f"{params.get('channels')} 通道" if params.get("channels") else "",
                     f"{len(params.get('classes') or [])} 类" if params.get("classes") else "",
                     f"{params.get('epochs')} 轮" if params.get("epochs") else "",
                     f"结构版本 {params.get('model_revision')}"
                     if params.get("model_revision") else ""]
        else:
            size = params.get("image_size")
            parts = [f"{size}×{size}" if size else "",
                     f"nfft {params.get('spectrogram_nfft')}" if params.get("spectrogram_nfft") else "",
                     f"{len(params.get('labels') or [])} 类" if params.get("labels") else "",
                     f"{params.get('dataset_samples')} 样本" if params.get("dataset_samples") else ""]
        return " · ".join(part for part in parts if part) or "--"

    def _selected_model_name(self):
        items = self.models_table.selectedItems() if hasattr(self, "models_table") else []
        return items[0].data(QtCore.Qt.ItemDataRole.UserRole) if items else None

    def _models_selection_changed(self, *_):
        name = self._selected_model_name()
        has = name is not None
        self.model_use_button.setEnabled(has)
        self.model_rename_button.setEnabled(has)
        self.model_delete_button.setEnabled(has)
        if not has:
            self.model_detail.clear()
            return
        entry = model_store.load_model(self.workspace.root, name)
        self.model_detail.setPlainText(self._model_detail_text(entry))

    @staticmethod
    def _model_detail_text(entry):
        lines = [f"名称：{entry.get('name')}",
                 f"类型：{entry.get('model_type') or '--'} · 用途：{entry.get('purpose_title') or '--'}",
                 f"训练时间：{entry.get('created_at') or '--'}",
                 f"大小：{format_bytes(entry.get('size_bytes'))}"
                 f"（{entry.get('library') or '--'}）",
                 f"状态：{STATUS_TEXT.get(entry.get('status'), entry.get('status') or '')}"
                 + (f"（{entry['reason']}）" if entry.get("reason") else ""),
                 f"来源实验：{entry.get('source_run') or '--'}",
                 f"清单：{entry.get('path')}",
                 f"SHA-256：{entry.get('sha256') or '--'}"]
        if entry.get("contract"):
            lines.append(f"契约：{entry['contract']}")
        params = entry.get("params") or {}
        if params:
            lines.append("参数：" + " · ".join(f"{key}={value}" for key, value in params.items()))
        metrics = entry.get("metrics") or {}
        if metrics:
            lines.append("指标：" + " · ".join(f"{key}={value}" for key, value in metrics.items()))
        return "\n".join(lines)

    # ------------------------------------------------------------------ 操作
    def scan_model_library(self):
        """把训练目录里已成功完成但未登记的模型补进模型库。"""
        try:
            result = model_store.reconcile(self.workspace.root)
        except OSError as exc:
            self.model_hint.setText(f"模型扫描失败：{exc}")
            return
        self.refresh_models_panel()
        self.refresh_model_choices()
        parts = []
        if result["added"]:
            parts.append(f"新登记 {len(result['added'])} 个：" + "、".join(result["added"]))
        if result["updated"]:
            parts.append(f"更新 {len(result['updated'])} 个：" + "、".join(result["updated"]))
        if result["failed"]:
            parts.append(f"失败 {len(result['failed'])} 个（"
                         + "；".join(f"{item['run']}：{item['error']}" for item in result["failed"])
                         + "）")
        self.model_hint.setText("；".join(parts) or "没有发现新的已训练模型。")

    def use_selected_model(self):
        name = self._selected_model_name()
        if name is None:
            self.model_hint.setText("请先在表格里选择一个模型。")
            return
        entry = model_store.load_model(self.workspace.root, name)
        if entry.get("status") != "ok":
            self.model_hint.setText(
                f"模型「{name}」状态为 {STATUS_TEXT.get(entry.get('status'), '')}，"
                f"不能用于推理：{entry.get('reason') or ''}")
            return
        purpose = entry.get("purpose")
        widget = {"detect": "ml_manifest", "hop": "hops_manifest", "amc": "amc_model"}.get(purpose)
        if widget is None:
            self.model_hint.setText(f"模型用途无法识别，不能跳转到页面：{name}")
            return
        getattr(self, widget).setText(entry["path"])
        self.tabs.setCurrentIndex(self._page_index(model_store.PURPOSE_TITLES[purpose]))
        self.status.setText(f"已选择模型：{name}")

    def rename_selected_model(self):
        name = self._selected_model_name()
        if name is None:
            return
        new_name, accepted = QtWidgets.QInputDialog.getText(
            self, "重命名模型", "新的模型名称：", text=name)
        if not accepted or new_name.strip() == name:
            return
        try:
            entry = model_store.rename_model(self.workspace.root, name, new_name)
        except (model_store.ModelError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "重命名失败", str(exc))
            return
        self.refresh_models_panel()
        self.refresh_model_choices()
        self.status.setText(f"模型已重命名为：{entry['name']}")

    def delete_selected_model(self):
        name = self._selected_model_name()
        if name is None:
            return
        entry = model_store.load_model(self.workspace.root, name)
        if QtWidgets.QMessageBox.question(
                self, "删除模型",
                f"删除模型「{name}」？\n\n目录：{entry['directory']}\n"
                "只删除模型库里的这一个条目，训练运行目录与实验记录保留。") \
                != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            model_store.delete_model(self.workspace.root, name)
        except (model_store.ModelError, OSError) as exc:
            QtWidgets.QMessageBox.warning(self, "删除失败", str(exc))
            return
        self.refresh_models_panel()
        self.refresh_model_choices()
        self.status.setText(f"已删除模型：{name}")
