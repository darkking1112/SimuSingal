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

from .dialogs import ImportBatchDialog, SignalParamsDialog  # noqa: F401  两个对话框（保持 ui.main_window 旧导入面）
from .pages.import_page import ImportPageMixin
from .pages.generator_page import GeneratorPageMixin
from .pages.compare_page import ComparePageMixin
from .pages.analysis_page import AnalysisPageMixin
from .pages.detect_page import DetectPageMixin
from .pages.hops_page import HopsPageMixin
from .pages.amc_page import AmcPageMixin
from .pages.history_page import HistoryPageMixin
from .pages.data_page import DataPageMixin
from .constants import (EXPORT_FORMATS, IMPORT_COL_DTYPE, IMPORT_COL_ENDIAN,
                        IMPORT_COL_FILE, IMPORT_COL_FORMAT, IMPORT_COL_MOD,
                        IMPORT_COL_NOTE, IMPORT_COL_POINTS, IMPORT_COL_RATE,
                        IMPORT_COL_RF_CENTER, IMPORT_COL_STATUS, IMPORT_FILE_FILTER,
                        IMPORT_FORMAT_LABELS, IMPORT_MODULATION_CHOICES,
                        IMPORT_SUFFIXES, MODE_CHOICES, MODE_SHORT, PLAY_MAX_ROWS,
                        PLAY_MAX_ROWS_PER_TICK, PLAY_WAVE_POINTS, _SCOPE_LABELS,
                        _SOURCE_KIND_LABELS, _VERSION_SOURCE_LABELS)


from .helpers import (_AMC_SOURCE_TEXT, _asset_exports, _asset_format, _comparison_line,
                      _fmt_hz, _fmt_metric, _fmt_span, _iq_binary_kind, _mirrored_spectrum)
from .runner import _run_task
from .widgets import UnitSpinBox, _freq_spin, _plain_spin, _unit_row


class MainWindow(ImportPageMixin, GeneratorPageMixin, AnalysisPageMixin, ComparePageMixin, DetectPageMixin, HopsPageMixin, AmcPageMixin, HistoryPageMixin, DataPageMixin, DesktopWindow):
    run_task = staticmethod(_run_task)

    def __init__(self, workspace):
        self.asset_limit = 100
        self.asset_page = 0
        self.adopt_run_ids = {}
        super().__init__(Workspace(workspace), "电磁信号分析 · SignalAnalysis",
                         "离线数据 · 通用统计与时频展示 · IQ 信号生成 · 信号检测与调制识别 · "
                         "算法对比与离线报告 · 原生插件")
        # 页面注册表：顺序即界面顺序，也是全项目唯一决定标签下标的地方。
        # 其余代码一律用 _page_index("页面名") 取下标、用页面名作 tab_results 的键，
        # 这样调整页序不会出现“静默跳到错误页面”或“读到别的页面的结果”。
        pages = {
            "信号导入": self.build_import,
            "IQ 信号生成": self.build_generator,
            "数据分析": self.build_analysis,
            "信号检测": self.build_detect,
            "调制识别": self.build_amc,
            "跳频参数": self.build_hops,
            "模型训练": self.build_training,
            "数据管理": self.build_data_management,
            "算法对比": self.build_compare,
            "运行记录": self.build_history,
        }
        self.page_index = {title: index for index, title in enumerate(pages)}
        for title, builder in pages.items():
            self.tabs.addTab(builder(), title)
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
        self.refresh_history()
        self.refresh_collections()
        self.refresh_assets()

    def build_training(self):
        from .pages.training_page import TrainingPage
        self.training_page = TrainingPage(self)
        return self.training_page

    def closeEvent(self, event):
        if not self.training_page.shutdown():
            event.ignore()
            return
        super().closeEvent(event)

    def _page_index(self, title):
        """按页面名取标签下标；下标只由 __init__ 的页面注册表决定。"""
        return self.page_index[title]


    def job_buttons(self):
        return (self.demo_button, self.analyze_button, self.native_button,
                self.generate_button, self.detect_button, self.hops_button, self.ml_button,
                self.hops_ml_button, self.amc_button, self.storage_scan_button,
                self.storage_preview_button, self.storage_cleanup_button,
                self.migrate_button, self.import_start_button,
                *self.gen_panel.action_buttons())

    def set_busy(self, busy):
        """输入控件的启停由父类处理；采纳按钮与导入按钮的可用性由页面状态决定。"""
        super().set_busy(busy)
        for page, button in self._adopt_buttons().items():
            button.setEnabled(not busy and page in self.adopt_run_ids)
        if hasattr(self, "import_start_button"):
            self.import_start_button.setEnabled(not busy and self._import_has_ready())

    def _adopt_buttons(self):
        return {"信号检测": getattr(self, "adopt_detect_button", None),
                "跳频参数": getattr(self, "adopt_hops_button", None),
                "调制识别": getattr(self, "adopt_amc_button", None)}

    def result_status(self, result):
        """数据盘点/清理/历史登记不写运行记录，状态栏不能沿用“结果已保存”文案。"""
        kind = result.get("kind")
        if kind == "storage_cleanup":
            return "清理完成 · 正在重新扫描（未写入运行记录）"
        if kind == "storage_report":
            return "数据盘点完成（只读，未写入运行记录）"
        if kind == "legacy_migration":
            return (f"历史数据登记完成：数据集 {len(result['datasets']['registered'])} · "
                    f"实验 {len(result['experiments']['registered'])} · "
                    f"跳过 {len(result['datasets']['skipped'])} 项（幂等，可重复执行）")
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
        if kind == "training_export":
            name = {"detection": "检测", "iq": "AMC"}.get(result.get("task"))
            return f"{name}训练数据导出完成：{result.get('samples', 0)} 个样本"
        if kind == "import_inspect":
            return f"已识别 {len(result.get('files', []))} 个文件（只读头部，未写入资产）"
        if kind == "import_manifest":
            return (f"标注清单解析完成：挂接 {result.get('matched', 0)} 条 · "
                    f"未匹配 {len(result.get('unmatched', []))} 行（未写入资产）")
        if kind == "import_files":
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
        if result.get("kind") == "legacy_migration":
            self._collections_changed()
        elif result.get("kind") == "recipe_preview":
            self.gen_panel.show_preview(result)
        elif result.get("kind") == "generate_collection":
            self.last_result = result
            self.gen_panel.show_generation(result)
            self._collections_changed()
        elif result.get("kind") == "torchsig_probe":
            self.gen_panel.show_probe(result)
        elif result.get("kind") == "training_export":
            self.training_page.export_finished(result)
            self.tabs.setCurrentIndex(self._page_index("模型训练"))
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
        self.collection_combo.currentIndexChanged.connect(self.refresh_assets)
        layout.addWidget(self.collection_combo)
        layout.addWidget(QtWidgets.QLabel("数据资产"))
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("按文件名查找")
        self.search.textChanged.connect(self.refresh_assets)
        layout.addWidget(self.search)
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
        workspace_label = QtWidgets.QLabel(f"工作目录\n{self.workspace.root}")
        workspace_label.setWordWrap(True)
        layout.addWidget(workspace_label)
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
        if hasattr(self, "gen_collection"):
            previous_scope = self.gen_collection.currentData()
            self.gen_collection.blockSignals(True)
            self.gen_collection.clear()
            self.gen_collection.addItem("不加入集合", None)
            self.gen_collection.addItem("新建集合…", "__new__")
            for collection in self.workspace.list_collections():
                self.gen_collection.addItem(collection["name"], collection["id"])
            scope_index = (self.gen_collection.findData(previous_scope)
                           if previous_scope else 0)
            self.gen_collection.setCurrentIndex(scope_index if scope_index >= 0 else 0)
            self.gen_collection.blockSignals(False)
            self._gen_collection_changed()
        if hasattr(self, "gen_panel"):
            self.gen_panel.refresh_collections()
        if hasattr(self, "training_page"):
            self.training_page.refresh_collections()

    def _asset_scope(self):
        data = self.collection_combo.currentData()
        if data == "__scattered__":
            return {"scattered": True}
        if data:
            return {"collection_id": data}
        return {}

    def change_asset_page(self, delta, reset=False):
        """翻页或重置到第一页；页码越界时由 refresh_assets 夹取。"""
        self.asset_page = 0 if reset else max(0, self.asset_page + delta)
        self.refresh_assets()

    def refresh_assets(self, *_):
        selected = self.selected_asset()
        scope = self._asset_scope()
        total = self.workspace.count_assets(self.search.text(), **scope)
        self.asset_limit = int(self.page_size.currentText())
        pages = max(1, -(-total // self.asset_limit))
        self.asset_page = min(self.asset_page, pages - 1)
        self.assets.clear()
        for asset in self.workspace.list_assets(self.search.text(), self.asset_limit,
                                                self.asset_page * self.asset_limit,
                                                **scope):
            item = QtWidgets.QListWidgetItem(asset["name"])
            item.setData(QtCore.Qt.ItemDataRole.UserRole, asset)
            self.assets.addItem(item)
            if selected and selected["id"] == asset["id"]:
                self.assets.setCurrentItem(item)
        self.page_label.setText(f"第 {self.asset_page + 1}/{pages} 页")
        self.asset_total.setText(f"共 {total} 条")
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
        """状态栏摘要：文件名 / 相对位置 / 大小 / 资产（工作区存储格式）/ 导出。

        路径一律相对工作目录，不重复写出工作目录本身；文件缺失时明确提示而不是
        抛异常。资产与导出分开写：前者是工作区里的存储本体（独立 NPY 或分片记录），
        后者是 ``exports/`` 下本次实际生成的副本（没有就写“无”）。
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
        exports = _asset_exports(self.workspace.root / "exports", asset)
        return (f"文件名 {asset['name']}  ·  位置 {asset['path']}  ·  "
                f"大小 {size}（{asset['sample_count']:,} 复采样 @ {rate:g} Hz · {duration}）  ·  "
                f"资产：{_asset_format(asset)}  ·  "
                f"导出：{'、'.join(exports) if exports else '无'}")


    def adopt_page_result(self, page):
        """把当前页展示的结果采纳为目标参考参数与标签（追加式）。"""
        run_id = self.adopt_run_ids.get(page)
        if run_id is None:
            self.status.setText("当前页还没有可采纳的结果")
            return
        self.start_job("adopt_result", run_id=run_id)

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
        """按结果类型写回对应页面并切过去；页面名与下标一律经由页面注册表解析。"""
        self.last_result = result
        if result["kind"] == "analysis":
            self.tab_results["数据分析"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_analysis(result, arrays)
            self.tabs.setCurrentIndex(self._page_index("数据分析"))
        elif result["kind"] in ("detect", "ml_detect"):
            self.tab_results["信号检测"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_detect(result, arrays)
            self._render_compare()
            self._set_adopt_run("信号检测", result)
            self.tabs.setCurrentIndex(self._page_index("信号检测"))
        elif result["kind"] in ("detect_hops", "ml_detect_hops"):
            self.tab_results["跳频参数"] = result
            with np.load(self.workspace.root / result["plots_path"], allow_pickle=False) as arrays:
                self._render_hops(result, arrays)
            self._render_compare()
            self._set_adopt_run("跳频参数", result)
            self.tabs.setCurrentIndex(self._page_index("跳频参数"))
        elif result["kind"] == "amc_classify":
            self.tab_results["调制识别"] = result
            self._render_amc(result)
            self._render_compare()
            self._set_adopt_run("调制识别", result)
            self.tabs.setCurrentIndex(self._page_index("调制识别"))
        elif result["kind"] == "amc_iq_classify":
            self.tab_results["调制识别"] = result
            self._render_amc_iq(result)
            self._render_compare()
            self._set_adopt_run("调制识别", result)
            self.tabs.setCurrentIndex(self._page_index("调制识别"))
        elif result["kind"] == "native":
            self.tab_results["数据分析"] = result
            self.wave.clear()
            self.spectrum.clear()
            self.tf_image.clear()
            self.waterfall_image.clear()
            self.const_scatter.clear()
            self.tf_stack.setCurrentWidget(self.time_frequency)
            self.summary.setPlainText(f"原生复制完成：{result['plugin']['id']}\n"
                                      f"输出资产：{result['derived_asset_id']}\n请选择输出资产进行分析。")
            self.tabs.setCurrentIndex(self._page_index("数据分析"))


def launch(workspace):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(workspace)
    window.show()
    return app.exec()
