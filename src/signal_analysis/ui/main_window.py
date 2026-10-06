"""Independent signal analysis desktop."""
import json
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from common.gui import DesktopWindow
from common.reports import amc_metrics, detection_metrics
from .pages.collection_gen_page import CollectionGenPanel
from ..core_api import MAX_SAMPLES, plan_signal, spectrum_row
from ..storage.maintenance import RUN_KIND_LABELS, format_bytes, read_settings, write_settings
from ..data import Workspace
from ..tasks import run_job

from .dialogs import SignalParamsDialog  # noqa: F401  对话框（保持 ui.main_window 旧导入面）
from .pages.import_page import ImportPageMixin
from .pages.generator_page import GeneratorPageMixin
from .pages.compare_page import ComparePageMixin
from .pages.analysis_page import AnalysisPageMixin
from .pages.detect_page import DetectPageMixin
from .pages.hops_page import HopsPageMixin
from .pages.amc_page import AmcPageMixin
from .pages.history_page import HistoryPageMixin
from .pages.data_page import DataPageMixin
from .pages.models_page import ModelsPageMixin
from .training_coordinator import TrainingCoordinator
from .constants import (IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                        IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                        IMPORT_COL_NAME, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                        IMPORT_COL_SNR, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                        IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                        IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                        PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                        _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)


from .helpers import (_AMC_SOURCE_TEXT, _asset_format, _comparison_line,
                      _fmt_hz, _fmt_metric, _fmt_span, _mirrored_spectrum)
from .runner import _run_task
from .widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class MainWindow(ImportPageMixin, GeneratorPageMixin, AnalysisPageMixin, ComparePageMixin, DetectPageMixin, HopsPageMixin, AmcPageMixin, HistoryPageMixin, DataPageMixin, ModelsPageMixin, DesktopWindow):
    run_task = staticmethod(_run_task)

    def __init__(self, workspace):
        self.asset_limit = 100
        self.asset_page = 0
        self.adopt_run_ids = {}
        #: 侧栏集合范围上次成功应用的值；标注保存失败时回滚到它（第 11 节）。
        self._applied_collection_scope = None
        self._restoring_scope = False
        super().__init__(Workspace(workspace), "电磁信号分析 · SignalAnalysis",
                         "离线数据 · 通用统计与时频展示 · IQ 信号生成 · 信号检测与调制识别 · "
                         "算法对比与离线报告 · 原生插件")
        #: 模型下拉：页面创建时登记，「数据管理 → 模型管理」增删后统一刷新。
        self._model_pickers = []
        self.training_coordinator = TrainingCoordinator(self)
        self.training_pages_by_owner = {}
        # 页面注册表：顺序即界面顺序，也是全项目唯一决定标签下标的地方。
        # 其余代码一律用 _page_index("页面名") 取下标、用页面名作 tab_results 的键，
        # 这样调整页序不会出现“静默跳到错误页面”或“读到别的页面的结果”。
        pages = {
            "信号导入": self.build_import,
            "IQ 信号生成": self.build_generator,
            "态势显示": self.build_analysis,
            "信号检测": self.build_detect,
            "调制识别": self.build_amc,
            "跳频参数": self.build_hops,
            "信号检测训练": self.build_detection_training,
            "AMC 识别训练": self.build_amc_training,
            "数据管理": self.build_data_management,
            "算法对比": self.build_compare,
            "运行记录": self.build_history,
        }
        self.page_index = {title: index for index, title in enumerate(pages)}
        # 页面构建前先登记既有训练模型，页面里的模型下拉首屏就能列出它们（幂等）
        self.reconcile_models()
        for title, builder in pages.items():
            self.tabs.addTab(builder(), title)
        self.training_pages = tuple(self.training_pages_by_owner.values())
        self.last_result = None
        self._play_data = None
        self._play_rate = 1.0
        self._play_pos = 0
        self._play_last = 0.0
        self._play_paused = False
        self._play_buffer = None
        self._play_times = []
        self._play_real = False
        self._play_level_hi = None
        self._range_syncing = False
        self.play_timer = QtCore.QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._on_playback_tick)
        self.tasks.started.connect(self._refresh_task_controls)
        self.tasks.finished.connect(self._refresh_task_controls)
        self.refresh_history()
        self.refresh_collections()
        self.refresh_assets()
        self._update_training_tab_access()

    def build_detection_training(self):
        from .pages.detection_training_page import DetectionTrainingPage
        self.detection_training_page = DetectionTrainingPage(
            self, self.training_coordinator)
        self.training_pages_by_owner[self.detection_training_page.OWNER] = (
            self.detection_training_page)
        return self.detection_training_page

    def build_amc_training(self):
        from .pages.amc_training_page import AmcTrainingPage
        self.amc_training_page = AmcTrainingPage(self, self.training_coordinator)
        self.training_pages_by_owner[self.amc_training_page.OWNER] = self.amc_training_page
        return self.amc_training_page

    def closeEvent(self, event):
        for page in getattr(self, "training_pages", ()):
            if not page.prepare_shutdown():
                event.ignore()
                return
        for page in getattr(self, "training_pages", ()):
            if not page.runner.shutdown():
                event.ignore()
                return
        super().closeEvent(event)

    def _page_index(self, title):
        """按页面名取标签下标；下标只由 __init__ 的页面注册表决定。"""
        return self.page_index[title]

    # ------------------------------------------------------------- 模型库同步
    def reconcile_models(self):
        """启动时把已训练但未登记的模型补进模型库；失败不阻塞窗口启动。"""
        from ..services import model_store
        try:
            return model_store.reconcile(self.workspace.root)
        except (model_store.ModelError, OSError, ValueError):
            return {"added": [], "updated": [], "failed": []}

    def register_model_picker(self, picker):
        self._model_pickers.append(picker)

    def refresh_model_choices(self):
        """模型库变化后统一刷新：各页模型下拉 + 数据管理页的模型管理表格。"""
        for picker in getattr(self, "_model_pickers", ()):
            picker.refresh()
        if getattr(self, "models_table", None) is not None:
            self.refresh_models_panel()


    def job_buttons(self):
        return (self.analyze_button, self.native_button,
                self.generate_button, self.detect_button, self.hops_button, self.ml_button,
                self.hops_ml_button, self.amc_button, self.storage_scan_button,
                self.storage_preview_button, self.storage_cleanup_button,
                self.import_start_button,
                *self.gen_panel.action_buttons())

    def owner_buttons(self, owner):
        """按页面返回任务期间要禁用的按钮；未登记的 owner 回退为整窗按钮集合。"""
        adopt = self._adopt_buttons()
        training_page = self.training_pages_by_owner.get(owner)
        if training_page is not None:
            return training_page.task_buttons()
        mapping = {
            "信号导入": (self.import_add_files_button, self.import_add_folder_button,
                        self.import_remove_button, self.import_recheck_button,
                        self.import_csv_button, self.import_apply_button,
                        self.import_start_button),
            "IQ 信号生成": (self.generate_button,
                           *self.gen_panel.action_buttons()),
            "信号集合生成": tuple(self.gen_panel.action_buttons()),
            "态势显示": (self.analyze_button, self.native_button),
            "信号检测": (self.detect_button, self.ml_button, adopt["信号检测"]),
            "跳频参数": (self.hops_button, self.hops_ml_button, adopt["跳频参数"]),
            "调制识别": (self.amc_button, adopt["调制识别"]),
            "数据管理": (self.storage_scan_button, self.storage_preview_button,
                        self.storage_cleanup_button),
        }
        return mapping.get(owner, tuple(self.job_buttons()))

    def task_finished_text(self, task):
        """任务结束文案：有结果的任务复用 result_status 的统计文案。"""
        if task.owner in self.training_pages_by_owner:
            if task.state == "cancelled":
                return f"{task.label} · 已取消"
            if task.state == "failed":
                return f"{task.label} · 失败：{task.error or ''}"
            if task.result is not None:
                return self.result_status(task.result)
            return f"{task.label} · 已完成"
        if task.result is not None:
            try:
                return self.result_status(task.result)
            except Exception:
                pass
        return super().task_finished_text(task)

    def _refresh_task_controls(self, *_):
        """任务收尾后重算采纳按钮与导入按钮的可用性（任务期间由 owner 按钮集合禁用）。"""
        for page, button in self._adopt_buttons().items():
            if button is not None:
                button.setEnabled(page in self.adopt_run_ids)
        if hasattr(self, "import_start_button"):
            self._update_import_controls()
        for page in getattr(self, "training_pages", ()):
            page.refresh_controls()

    def _adopt_buttons(self):
        return {"信号检测": getattr(self, "adopt_detect_button", None),
                "跳频参数": getattr(self, "adopt_hops_button", None),
                "调制识别": getattr(self, "adopt_amc_button", None)}

    def result_status(self, result):
        """数据盘点/清理不写运行记录，状态栏不能沿用“结果已保存”文案。"""
        kind = result.get("kind")
        if kind == "storage_cleanup":
            return "清理完成 · 正在重新扫描（未写入运行记录）"
        if kind == "storage_report":
            return "数据盘点完成（只读，未写入运行记录）"
        if kind == "recipe_preview":
            return f"参数预览完成：{result.get('count')} 组参数（未合成 IQ）"
        if kind == "generate_collection":
            engine = {"project": "项目引擎", "torchsig": "TorchSig",
                      "torchsig_import": "TorchSig 导入"}.get(result.get("engine"),
                                                              result.get("engine"))
            text = (f"集合生成完成（{engine}）：新建 {result.get('created', 0)}"
                    f" / 请求 {result.get('requested', 0)} 条")
            if result.get("failed"):
                text += f" · 跳过 {result['failed']} 条"
            if result.get("stopped") == "cancelled":
                text = text.replace("完成", "已取消")
            elif result.get("stopped") == "time_limit":
                text += " · 达到时间上限"
            text += (f" · 会话 {result.get('sessions', 0)}"
                     + (f" · 逐跳 {result.get('hops')}" if result.get("hops") else ""))
            if result.get("collection_name"):
                text += f" · 集合「{result['collection_name']}」"
            return text
        if kind == "torchsig_probe":
            if result.get("ok"):
                return "TorchSig 环境可用：" + str(result.get("version"))
            return f"TorchSig 环境不可用：{result.get('message')}"
        if kind == "import_inspect":
            if result.get("cancelled"):
                return (f"识别已取消：已完成 {len(result.get('files', []))}"
                        f" · 未处理 {result.get('unprocessed', 0)}")
            return f"已识别 {len(result.get('files', []))} 个文件（只读头部，未写入资产）"
        if kind == "import_manifest":
            return (f"标注清单解析完成：挂接 {result.get('matched', 0)} 条 · "
                    f"未匹配 {len(result.get('unmatched', []))} 行（未写入资产）")
        if kind == "import_files":
            if result.get("cancelled"):
                text = (f"导入已取消：成功 {result.get('created', 0)}"
                        f" · 失败 {result.get('failed', 0)}"
                        f" · 未处理 {result.get('unprocessed', 0)}")
                if result.get("collection_name"):
                    text += f" · 集合「{result['collection_name']}」"
                return text
            shard_ids = result.get("shard_ids") or []
            if shard_ids:
                storage = f"分片 ×{len(shard_ids)}"
            else:
                storage = "分片" if result.get("shard_id") else "独立文件"
            text = (f"导入完成：成功 {result.get('created', 0)} · 失败 {result.get('failed', 0)}"
                    f" · 目标 {result.get('targets_total', 0)} 条 · {storage}")
            if result.get("collection_name"):
                text += f" · 集合「{result['collection_name']}」"
            if result.get("initial_labels"):
                text += f" · 初始标注 {result['initial_labels']} 条"
            return text
        if kind == "adopt_result":
            return (f"已采纳为参数标注：新建目标 {result.get('created_targets', 0)} · "
                    f"更新 {result.get('updated_targets', 0)} · 标签 {result.get('labels', 0)}"
                    + (f"（{'、'.join(result['task_sets'])}）" if result.get("task_sets") else ""))
        return super().result_status(result)

    def result_ready(self, result):
        self.refresh_history()
        self.refresh_assets()
        if result.get("kind") == "recipe_preview":
            self.gen_panel.show_preview(result)
        elif result.get("kind") == "generate_collection":
            self.last_result = result
            self.gen_panel.show_generation(result)
            self._collections_changed()
        elif result.get("kind") == "torchsig_probe":
            self.gen_panel.show_probe(result)
        elif result.get("kind") == "adopt_result":
            # 目标参考参数与标签已变：刷新侧栏只读详情与集合面板进度
            self.asset_changed()
            self.refresh_collections_panel()
        elif result.get("kind") == "import_inspect":
            self._apply_import_inspection(result["files"])
        elif result.get("kind") == "import_manifest":
            self._apply_import_manifest(result)
        elif result.get("kind") == "import_files":
            self._render_import_batch(result)
        elif result.get("kind") == "storage_report":
            self._render_data_management(result)
        elif result.get("kind") == "storage_cleanup":
            self._render_storage_cleanup(result)
        elif result.get("kind") == "generate":
            self.last_result = result
            self.show_generation_result(result)
            if result.get("collection_id"):
                self._collections_changed()
            for row in range(self.assets.count()):
                item = self.assets.item(row)
                if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == result["id"]:
                    self.assets.setCurrentItem(item)
                    break
        elif "kind" in result:
            self.display_result(result)
        else:
            for row in range(self.assets.count()):
                item = self.assets.item(row)
                if item.data(QtCore.Qt.ItemDataRole.UserRole)["id"] == result["id"]:
                    self.assets.setCurrentItem(item)

    def build_sidebar(self):
        box = QtWidgets.QWidget()
        box.setMinimumWidth(250)
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("信号集合"))
        self.collection_combo = QtWidgets.QComboBox()
        self.collection_combo.setToolTip("按集合筛选资产；零散资产指不在任何未归档集合中的资产")
        self.collection_combo.addItem("全部资产", None)
        self.collection_combo.addItem("零散资产", "__scattered__")
        self.collection_combo.currentIndexChanged.connect(self._sidebar_collection_changed)
        layout.addWidget(self.collection_combo)
        layout.addWidget(QtWidgets.QLabel("数据资产"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("按信号名称查找")
        self.search.textChanged.connect(self.refresh_assets)
        layout.addWidget(self.search)
        self.label_filter = QtWidgets.QComboBox()
        self.label_filter.setToolTip(
            "按标注状态过滤：只保留仍有目标缺少该任务标签的资产；\n"
            "没有目标的资产（纯噪声负样本、已确认无信号）不在此列")
        self.label_filter.addItem("全部", None)
        self.label_filter.addItem("未标注检测信息", "detection")
        self.label_filter.addItem("未标注AMC信息", "amc")
        self.label_filter.currentIndexChanged.connect(self.refresh_assets)
        layout.addWidget(self.label_filter)
        self.assets = QtWidgets.QListWidget()
        self.assets.currentItemChanged.connect(self.asset_changed)
        layout.addWidget(self.assets, 1)
        page_row = QtWidgets.QHBoxLayout()
        self.prev_page = QtWidgets.QPushButton("上一页")
        self.prev_page.clicked.connect(lambda: self.change_asset_page(-1))
        page_row.addWidget(self.prev_page)
        self.page_label = QtWidgets.QLabel("第 0/0 页")
        self.page_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        page_row.addWidget(self.page_label, 1)
        self.next_page = QtWidgets.QPushButton("下一页")
        self.next_page.clicked.connect(lambda: self.change_asset_page(1))
        page_row.addWidget(self.next_page)
        layout.addLayout(page_row)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("每页"))
        self.page_size = QtWidgets.QComboBox()
        self.page_size.addItems(["100", "200", "500"])
        self.page_size.currentIndexChanged.connect(lambda *_: self.change_asset_page(0, reset=True))
        size_row.addWidget(self.page_size)
        self.asset_total = QtWidgets.QLabel("共 0 条")
        size_row.addWidget(self.asset_total, 1)
        layout.addLayout(size_row)
        self.asset_info = QtWidgets.QLabel("尚未选择数据")
        self.asset_info.setWordWrap(True)
        layout.addWidget(self.asset_info)
        layout.addWidget(QtWidgets.QLabel("目标与标注（只读）"))
        self.target_list = QtWidgets.QListWidget()
        self.target_list.setMaximumHeight(150)
        self.target_list.setToolTip("目标参考参数与标注状态；参数标注在“信号导入”页维护")
        layout.addWidget(self.target_list)
        return box


    def selected_asset(self):
        item = self.assets.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def refresh_collections(self, *_):
        """刷新侧栏集合下拉：全部资产 / 零散资产 / 各集合（附资产数）。"""
        previous = self.collection_combo.currentData() if hasattr(
            self, "collection_combo") else None
        self.collection_combo.blockSignals(True)
        self.collection_combo.clear()
        self.collection_combo.addItem("全部资产", None)
        self.collection_combo.addItem("零散资产", "__scattered__")
        for collection in self.workspace.list_collections():
            self.collection_combo.addItem(f"{collection['name']}（{collection['asset_count']}）",
                                          collection["id"])
        index = self.collection_combo.findData(previous) if previous else 0
        self.collection_combo.setCurrentIndex(index if index >= 0 else 0)
        self.collection_combo.blockSignals(False)
        if hasattr(self, "import_collection"):
            previous_scope = self.import_collection.currentData()
            self.import_collection.blockSignals(True)
            self.import_collection.clear()
            self.import_collection.addItem("不加入集合", None)
            self.import_collection.addItem("新建集合…", "__new__")
            for collection in self.workspace.list_collections():
                self.import_collection.addItem(collection["name"], collection["id"])
            scope_index = (self.import_collection.findData(previous_scope)
                           if previous_scope else 0)
            self.import_collection.setCurrentIndex(scope_index if scope_index >= 0 else 0)
            self.import_collection.blockSignals(False)
            self._import_collection_changed()
        if hasattr(self, "gen_panel"):
            self.gen_panel.refresh_collections()
        for page in getattr(self, "training_pages", ()):
            page.refresh_collections()
        self.update_gen_collection_scope()
        self._update_training_tab_access()
        self._applied_collection_scope = self.collection_combo.currentData()

    def _asset_scope(self):
        data = self.collection_combo.currentData()
        if data == "__scattered__":
            return {"scattered": True}
        if data:
            return {"collection_id": data}
        return {}

    #: 训练页注册名；只在侧栏选中具体集合时可用（见第 10.2 节）。
    TRAINING_TAB_TITLES = ("信号检测训练", "AMC 识别训练")

    def current_collection_id(self):
        """侧栏当前选中的集合 ID；「全部资产」「零散资产」或未初始化时返回 ``None``。"""
        if not hasattr(self, "collection_combo"):
            return None
        data = self.collection_combo.currentData()
        return data if isinstance(data, str) and data != "__scattered__" else None

    def select_asset(self, asset_id):
        """按资产 ID 选中左侧列表项（用于保存失败时回滚选择）；不在列表中返回 False。"""
        for row in range(self.assets.count()):
            asset = self.assets.item(row).data(QtCore.Qt.ItemDataRole.UserRole)
            if asset and asset["id"] == asset_id:
                self.assets.setCurrentItem(self.assets.item(row))
                return True
        return False

    def _sidebar_collection_changed(self, *_):
        """侧栏集合范围变化：先保存训练页的标注，保存失败则回滚选择（第 11 节）。

        名字必须避开页面 mixin 的 ``_collection_selected``（数据管理页的信号集合子页用它）。
        """
        if self._restoring_scope:
            return
        if not self._flush_annotation_edits():
            self._restore_collection_scope()
            return
        self._applied_collection_scope = self.collection_combo.currentData()
        self.refresh_assets()
        self._collection_scope_changed()

    def _flush_annotation_edits(self):
        """保存两个训练页里未保存的标注；任一保存失败返回 False。"""
        for page in getattr(self, "training_pages", ()):
            if not page.save_if_dirty():
                return False
        return True

    def _restore_collection_scope(self):
        """把侧栏集合恢复到上次成功应用的值（标注保存失败时保持上下文不变）。"""
        index = self.collection_combo.findData(self._applied_collection_scope)
        self._restoring_scope = True
        try:
            self.collection_combo.setCurrentIndex(max(0, index))
            self.refresh_assets()
        finally:
            self._restoring_scope = False

    def _collection_scope_changed(self, *_):
        """侧栏集合范围变化：刷新训练页可用性并把「训练集」同步到当前集合。"""
        collection_id = self.current_collection_id()
        for page in getattr(self, "training_pages", ()):
            page.sync_train_collection(collection_id)
        self._update_training_tab_access()

    def _update_training_tab_access(self):
        """训练页需要具体集合：全部资产/零散资产时置灰，并跳回“信号导入”。"""
        index_map = getattr(self, "page_index", None)
        if not index_map or not hasattr(self, "tabs"):
            return
        enabled = self.current_collection_id() is not None
        guarded = [index_map[title] for title in self.TRAINING_TAB_TITLES
                   if title in index_map]
        current = self.tabs.currentIndex()
        # 先离开训练页再置灰：Qt 会把禁用标签页上的当前页挪到相邻页，而不是我们要求的首页
        if not enabled and current in guarded:
            self.tabs.setCurrentIndex(self._page_index("信号导入"))
        for index in guarded:
            self.tabs.setTabEnabled(index, enabled)

    def change_asset_page(self, delta, reset=False):
        """翻页或重置到第一页；页码越界时由 refresh_assets 夹取。"""
        self.asset_page = 0 if reset else max(0, self.asset_page + delta)
        self.refresh_assets()

    def refresh_assets(self, *_):
        selected = self.selected_asset()
        scope = self._asset_scope()
        missing_labels = self.label_filter.currentData()
        self.update_gen_collection_scope()
        total = self.workspace.count_assets(self.search.text(), **scope,
                                           missing_labels=missing_labels)
        self.asset_limit = int(self.page_size.currentText())
        pages = max(1, -(-total // self.asset_limit))
        self.asset_page = min(self.asset_page, pages - 1)
        self.assets.clear()
        for asset in self.workspace.list_assets(self.search.text(), self.asset_limit,
                                                self.asset_page * self.asset_limit,
                                                **scope, missing_labels=missing_labels):
            item = QtWidgets.QListWidgetItem(asset["name"])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, asset)
            self.assets.addItem(item)
            if selected and selected["id"] == asset["id"]:
                self.assets.setCurrentItem(item)
        self.page_label.setText(f"第 {self.asset_page + 1}/{pages} 页")
        self.asset_total.setText(f"共 {total} 条" + (
            f" · 已过滤「{self.label_filter.currentText()}」"
            if missing_labels is not None else ""))
        if not total and missing_labels is not None:
            self.asset_info.setText("当前过滤下没有资产：这些资产的目标都已有该任务标签，"
                                    "或资产没有目标（负样本/无信号）")
        self.prev_page.setEnabled(self.asset_page > 0)
        self.next_page.setEnabled(self.asset_page + 1 < pages)
        if self.assets.currentItem() is None and self.assets.count():
            self.assets.setCurrentRow(0)

    def _append_target_rows(self, asset):
        """侧栏只读目标列表：适用标记、标注状态、当前参数与来源。"""
        self.target_list.clear()
        try:
            targets = self.workspace.list_targets(asset["id"], with_current=True)
            status = self.workspace.asset_label_status(asset["id"])
        except ValueError:
            return
        for target in targets:
            current = target["current"] or {}
            flags = (f"检测{'✓' if target['for_detection'] else '—'} · "
                     f"AMC{'✓' if target['for_amc'] else '—'}")
            marks = []
            if target["id"] in status["detection"]:
                marks.append("检测已标注")
            if target["id"] in status["amc"]:
                marks.append("AMC已标注")
            if not marks:
                marks.append("待标注")
            scope = _SCOPE_LABELS.get(target["scope"], target["scope"])
            modulation = current.get("modulation") or "调制未知"
            source = _VERSION_SOURCE_LABELS.get(current.get("source"),
                                                 current.get("source") or "—")
            text = (f"{target['target_key']} · {scope} · {flags}\n"
                    f"    {modulation} · {' / '.join(marks)} · 来源 {source}")
            item = QtWidgets.QListWidgetItem(text)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, target["id"])
            low, high = current.get("f_low_hz"), current.get("f_high_hz")
            band = (f"{low:g}～{high:g} Hz" if low is not None and high is not None else "频带未知")
            detail = f"{target['target_key']}：{band}"
            snr = current.get("snr_db")
            if snr is not None:
                detail += f" · SNR {snr:.1f} dB"
            item.setToolTip(detail)
            self.target_list.addItem(item)

    def asset_changed(self, *_):
        if self._play_data is not None:
            self._stop_playback()
        asset = self.selected_asset()
        if asset:
            storage = "分片" if asset.get("storage_kind") == "shard" else "文件"
            source_kind = _SOURCE_KIND_LABELS.get(asset.get("source_kind"), "未知")
            self.asset_info.setText(
                f"{asset['sample_count']:,} 个复采样\n{asset['sample_rate']:g} Hz\n"
                f"来源：{source_kind} · 存储：{storage}")
            self.set_status_detail(self._asset_status_text(asset))
            self._append_target_rows(asset)
        else:
            self.asset_info.setText("尚未选择数据")
            self.set_status_detail("")
            self.target_list.clear()

    def _asset_status_text(self, asset):
        """状态栏摘要：文件名 / 相对位置 / 大小 / 资产（工作区存储格式与来源）。

        路径一律相对工作目录，不重复写出工作目录本身；文件缺失时明确提示而不是
        抛异常。资产格式按 catalog 登记的 ``storage_format``/``endian`` 描述，
        SigMF 资产的两个文件只报一次。
        """
        path = self.workspace.root / asset["path"]
        shard = asset.get("storage_kind") == "shard"
        try:
            if shard:
                # 分片资产只报本条记录的份额（complex64 = 8 B/采样点）
                size = format_bytes(int(asset.get("shard_length") or 0) * 8) + "（分片份额）"
            else:
                size = format_bytes(path.stat().st_size)
        except OSError:
            size = "文件缺失"
        rate = float(asset["sample_rate"])
        duration = _fmt_span(asset["sample_count"] / rate) if rate else "--"
        return (f"文件名 {asset['name']}  ·  位置 {asset['path']}  ·  "
                f"大小 {size}（{asset['sample_count']:,} 复采样 @ {rate:g} Hz · {duration}）  ·  "
                f"资产：{_asset_format(asset)}")


    def adopt_page_result(self, page):
        """把当前页展示的结果采纳为目标参考参数与标签（追加式）。"""
        run_id = self.adopt_run_ids.get(page)
        if run_id is None:
            self.status.setText("当前页还没有可采纳的结果")
            return
        self.start_job("adopt_result", owner=page, label="采纳为参数标注",
                       cancel_text="取消本次采纳", run_id=run_id)

    def _set_adopt_run(self, page, result):
        """更新某页的采纳按钮可用状态；不可采纳的结果类型保持禁用。"""
        button = self._adopt_buttons().get(page)
        run_id = (result or {}).get("run_id")
        if run_id:
            self.adopt_run_ids[page] = run_id
            if button is not None:
                button.setEnabled(True)
        else:
            self.adopt_run_ids.pop(page, None)
            if button is not None:
                button.setEnabled(False)

    def display_result(self, result):
        """按结果类型写回对应页面（不切换当前标签页）；页面名与下标一律经由页面注册表解析。"""
        self.last_result = result
        if result["kind"] == "analysis":
            self.tab_results["态势显示"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_analysis(result, arrays)
        elif result["kind"] in ("detect", "ml_detect"):
            self.tab_results["信号检测"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_detect(result, arrays)
            self._render_compare()
            self._set_adopt_run("信号检测", result)
        elif result["kind"] in ("detect_hops", "ml_detect_hops"):
            self.tab_results["跳频参数"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_hops(result, arrays)
            self._render_compare()
            self._set_adopt_run("跳频参数", result)
        elif result["kind"] == "amc_classify":
            self.tab_results["调制识别"] = result
            self._render_amc(result)
            self._render_compare()
            self._set_adopt_run("调制识别", result)
        elif result["kind"] == "amc_iq_classify":
            self.tab_results["调制识别"] = result
            self._render_amc_iq(result)
            self._render_compare()
            self._set_adopt_run("调制识别", result)
        elif result["kind"] == "native":
            self.tab_results["态势显示"] = result
            self.wave.clear()
            self.spectrum.clear()
            self.waterfall_image.clear()
            self.const_scatter.clear()
            self.const_hint.setText(self._hint_html(
                "原生复制完成",
                "请选择输出资产后再点「分析所选数据」，届时按目标参考参数决定是否绘制星座图。"))
            self.const_stack.setCurrentWidget(self.const_hint)
            self.summary.setPlainText(f"原生复制完成：{result['plugin']['id']}\n"
                                      f"输出资产：{result['derived_asset_id']}\n请选择输出资产进行分析。")


def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    window.show()
    return app.exec()
