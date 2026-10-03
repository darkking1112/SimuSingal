"""Signal-detection training page: collection labels plus external training.

训练数据直接取自信号集合（拆分设计第 10 节）：本页只负责检测任务的标注与训练配置，
不生成、不导出任何数据集。「数据标注」子页编辑**当前集合**内左侧选中资产的检测标签。
"""
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ...algorithms.dsp.image import detection_image, spectral_context
from ...contracts.image import band_to_box, box_to_band
from ...data.datasets import detection_label_applies
from ...data.targets import carryover_fields
from ...services.training_inputs import collection_task_set
from .training_common import TrainingPageBase

#: 标注渲染与训练脚本读取的数据卡片共用同一 FFT 点数。
NFFT = 512


class DetectionTrainingPage(TrainingPageBase):
    TASK = "detection"
    OWNER = "信号检测训练"
    TITLE = "信号检测训练"
    MODEL_OPTIONS = ("rtdetr", "yolo26s")
    HISTORY_TASKS = ("detection", "asset")
    HAS_ANNOTATIONS = True

    def add_task_fields(self, form):
        self.weights = self.path_field(
            form, "初始权重（YOLO 空值使用 yolo26s.pt）", kind="file")
        self.rtdetr_fields = QtWidgets.QGroupBox("RT-DETR 配置")
        rtdetr_form = QtWidgets.QFormLayout(self.rtdetr_fields)
        self.framework = self.path_field(
            rtdetr_form, "rtdetrv2_pytorch 目录")
        self.framework_config = self.path_field(
            rtdetr_form, "RT-DETRv2 模型 YAML", kind="file")
        form.addRow(self.rtdetr_fields)
        self.size = QtWidgets.QComboBox()
        self.size.addItems(["128", "256", "512", "1024"])
        self.size.setCurrentText("1024")
        form.addRow("时频图边长（检测）", self.size)

    def page_note(self):
        return ("训练数据取自信号集合：选择训练集与验证集集合即可，页内不生成数据、不导出"
                "训练集。“数据标注”编辑的是当前集合内左侧选中资产的检测标签，"
                "时频图按上面的边长现算。")

    def curve_title(self):
        return "检测训练 loss"

    # ------------------------------------------------------------------ 标注页
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

        toolbar = QtWidgets.QHBoxLayout()
        for text, callback in (("新增信号框", self.add_box),
                               ("删除选中框", self.delete_box),
                               ("标记无信号", self.mark_no_signal)):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        self.enable_button = QtWidgets.QPushButton("启用检测标注")
        self.enable_button.setToolTip("把该资产的目标标记为适用于检测后再标注")
        self.enable_button.clicked.connect(self.enable_detection_targets)
        toolbar.addWidget(self.enable_button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "时间 →（像素）")
        self.plot.setLabel("left", "频率递减 ↓（像素）")
        self.plot.invertY(True)
        self.plot.setAspectLocked(True)
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot, 1)

        self.annotation_status = QtWidgets.QLabel(
            "单类 emitter；拖动框移动、拖动角点缩放。修改后点“保存当前标注”：有框的目标记为"
            "正样本，没有框的适用目标记为无信号，同时把该资产覆盖度置为完整（整幅时频图已确认）；"
            "确认真没有信号时用“标记无信号”。")
        self.annotation_status.setWordWrap(True)
        layout.addWidget(self.annotation_status)

        self.selected_roi = None
        self._boxes = []
        self._targets = []
        self._negative = set()
        self._asset_id = None
        self._task_set_id = None
        self._meta = None
        self._size = 1024
        self._skipped_flags = 0
        self._semantics = "session_v1"
        self._created_targets = 0
        self._loading = False
        self._switching_asset = False
        self.dirty = False

        self.window.assets.currentItemChanged.connect(self._asset_selection_changed)
        return widget

    def update_task_controls(self):
        if not hasattr(self, "rtdetr_fields"):
            return
        self.rtdetr_fields.setVisible(self.arch.currentText() == "rtdetr")

    def update_data_summary(self):
        if hasattr(self, "image_item"):
            self._reload_annotation()

    def sync_train_collection(self, collection_id):
        super().sync_train_collection(collection_id)
        if hasattr(self, "image_item"):
            self._reload_annotation()

    def _reload_annotation(self):
        """按当前上下文重画标注视图：先保存未保存的修改，保存失败则保留现有视图。"""
        if self.dirty and not self.save_annotation():
            return False
        self.refresh_annotation()
        return True

    def configuration(self):
        config = super().configuration()
        config.update(weights=self.weights.text().strip(),
                      framework_path=self.framework.text().strip(),
                      framework_config=self.framework_config.text().strip(),
                      image_size=int(self.size.currentText()),
                      nfft=NFFT)
        return config

    def validate_task_config(self, config):
        if config["arch"] == "yolo26s" and not self.internal_models:
            raise ValueError("发行配置不提供 YOLO26s 训练")
        if config["arch"] == "rtdetr" and (
                not (Path(config["framework_path"]) / "src/core/yaml_config.py").is_file()
                or not Path(config["framework_config"]).is_file()):
            raise ValueError("RT-DETR 需要选择 rtdetrv2_pytorch 目录和模型 YAML")

    # ------------------------------------------------------------ 资产与渲染
    def _asset_selection_changed(self, current, previous):
        """切换资产前先保存标注；保存失败则留在原资产（第 11 节）。"""
        if self._switching_asset:
            return
        asset = current.data(QtCore.Qt.ItemDataRole.UserRole) if current else None
        if asset is not None and asset["id"] == self._asset_id:
            return  # 列表刷新重选同一资产：保留未保存修改，不重画
        if self.dirty and not self.save_annotation():
            self._restore_asset(previous)
            return
        self.refresh_annotation()

    def _restore_asset(self, previous):
        """把左侧选中项恢复为切换前的资产；原资产已不在列表时只提示。"""
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

    def refresh_annotation(self, *_):
        """按当前集合与左侧选中资产重画标注视图（不保存，调用方先 save_if_dirty）。"""
        collection_id = self.window.current_collection_id()
        asset = self.window.selected_asset()
        self._loading = True
        self.clear_rois()
        self._boxes, self._targets, self._negative = [], [], set()
        self._asset_id = asset["id"] if asset else None
        self._task_set_id = None
        self._meta = None
        self.image_item.clear()
        self.dirty = False
        self._loading = False
        if collection_id is None or asset is None:
            self.asset_label.setText("在左侧选择一个集合与资产" if collection_id is None
                                     else "在左侧选择一个数据资产")
            self.annotation_status.setText("")
            return
        self._size = int(self.size.currentText())
        try:
            task_set = collection_task_set(self.window.workspace, collection_id,
                                           "detection")
        except ValueError as exc:
            self.asset_label.setText(f"{asset['name']} · 当前集合未启用检测标注")
            self.annotation_status.setText(str(exc))
            return
        self._task_set_id = task_set["id"]
        try:
            self._render(asset, task_set)
        except (ValueError, OSError, KeyError, IndexError) as exc:
            self.annotate_status_error(exc)

    def _render(self, asset, task_set):
        workspace = self.window.workspace
        asset, data = workspace.load_samples(asset["id"])
        rate = float(asset["sample_rate"])
        summary, arrays = spectral_context(data, rate, {"nfft": NFFT})
        image, meta = detection_image(arrays, summary, size=self._size,
                                      dynamic_range_db=60.0)
        self._meta = {**meta, "sample_rate_hz": rate}
        self.image_item.setImage(image, levels=(0, 1))
        self.plot.setRange(QtCore.QRectF(0, 0, self._size, self._size))

        semantics = task_set["label_semantics"] or "session_v1"
        self._semantics = semantics
        targets = workspace.list_targets(asset["id"], with_current=True)
        self._targets = [target for target in targets
                         if target["for_detection"]
                         and detection_label_applies(semantics, target)]
        for target in self._targets:
            target["label"] = workspace.current_label("detection", task_set["id"],
                                                      target["id"])
            label = target["label"]
            if label is None:
                continue
            if int(label.get("include", 1)) == 0:
                self._negative.add(target["id"])
                continue
            version = target["current"]
            if version is None or version.get("f_low_hz") is None:
                continue
            self._add_box(self._box_of(version, rate), target["id"])
        self.asset_label.setText(
            f"{asset['name']} · 集合『{task_set['name']}』（{semantics}）")
        self._skipped_flags = len(targets) - len(self._targets)
        self._refresh_status()

    def _box_of(self, version, rate):
        duration = self._meta["duration_s"]
        start = version.get("sample_start")
        end = version.get("sample_end")
        return band_to_box(self._meta, version["f_low_hz"], version["f_high_hz"],
                           0.0 if start is None else start / rate,
                           duration if end is None else end / rate)

    def _refresh_status(self):
        if not self._targets:
            hint = ("该资产没有适用于检测的目标"
                    + (f"（{self._skipped_flags} 个目标未标记为适用检测，"
                       "可点“启用检测标注”）" if self._skipped_flags else
                       "；可点“标记无信号”确认没有信号"))
            self.annotation_status.setText(hint)
            return
        pending = [target for target in self._targets if target["label"] is None]
        text = f"{len(self._boxes)} 个框 · 适用目标 {len(self._targets)} 个"
        if pending:
            text += f" · 待标注 {len(pending)} 个（保存后记为无信号）"
        if self._negative:
            text += f" · 已确认无信号目标 {len(self._negative)} 个"
        if self.dirty:
            text += " · 尚未保存"
        self.annotation_status.setText(text)

    def annotate_status_error(self, error):
        self.annotation_status.setText(str(error))
        self.report_error(error)

    # --------------------------------------------------------------- 框操作
    def clear_rois(self):
        for item in self._boxes:
            self.plot.removeItem(item["roi"])
        self._boxes = []
        self.selected_roi = None

    def _add_box(self, box, target_id=None):
        x, y, width, height = box[:4]
        roi = pg.RectROI(((x - width / 2) * self._size, (y - height / 2) * self._size),
                         (width * self._size, height * self._size),
                         maxBounds=QtCore.QRectF(0, 0, self._size, self._size),
                         pen=pg.mkPen("y", width=2))
        roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        roi.sigClicked.connect(lambda *args, item=roi: self.select_roi(item))
        roi.sigRegionChanged.connect(self.mark_dirty)
        self.plot.addItem(roi)
        self._boxes.append({"target_id": target_id, "roi": roi})
        self.select_roi(roi)
        return roi

    def add_box(self, checked=False, box=None):
        if self._asset_id is None or self._task_set_id is None:
            self.annotate_status_error("请先在左侧选择当前集合内的数据资产")
            return
        self._loading = True
        self._add_box(box or [.5, .5, .25, .15])
        self._loading = False
        self.mark_dirty()

    def select_roi(self, roi):
        self.selected_roi = roi
        for item in self._boxes:
            item["roi"].setPen(pg.mkPen("c" if item["roi"] is roi else "y", width=2))

    def delete_box(self):
        for item in self._boxes:
            if item["roi"] is self.selected_roi:
                self.plot.removeItem(item["roi"])
                self._boxes.remove(item)
                self.selected_roi = None
                self.mark_dirty()
                return

    def clear_boxes(self):
        self.clear_rois()
        self.mark_dirty()

    def mark_dirty(self, *_):
        if self._loading:
            return
        self.dirty = True
        if self._asset_id is not None:
            self.annotation_status.setText("当前标注已修改，尚未保存")

    # --------------------------------------------------------------- 保存
    def save_if_dirty(self):
        return self.save_annotation() if self.dirty else True

    def save_annotation(self, *, negative_kind=None):
        if self._asset_id is None or self._task_set_id is None or self._meta is None:
            return True
        if not self.dirty:
            return True
        try:
            kept = self._write_boxes()
            negatives, blocked = self._write_removals(kept)
            self._write_coverage(negative_kind)
        except (ValueError, OSError, KeyError) as exc:
            self.annotate_status_error(exc)
            return False
        self.dirty = False
        self._refresh_status()
        message = (f"已保存 · 正样本 {len(self._boxes)} 个"
                   + (f"（含新增目标 {self._created_targets} 个）"
                      if self._created_targets else "")
                   + f" · 无信号目标 {negatives} 个 · 覆盖度完整")
        if blocked:
            message += f"；{blocked} 个适用目标没有参考参数版本，训练仍会排除它们"
        self.annotation_status.setText(message)
        self._created_targets = 0
        self.window.asset_changed()
        return True

    def _write_coverage(self, negative_kind):
        """保存即确认「整幅时频图已看过一遍」：写入完整覆盖度（第 11 节）。"""
        self.window.workspace.set_asset_coverage(
            self._task_set_id, self._asset_id, "complete",
            negative_kind=negative_kind, source="manual")

    def _write_boxes(self):
        workspace = self.window.workspace
        asset = workspace.get_asset(self._asset_id)
        rate, count = float(asset["sample_rate"]), int(asset["sample_count"])
        used_keys = {target["target_key"] for target in self._targets}
        kept = set()
        created = 0
        for item in self._boxes:
            x, y, width, height = self._roi_box(item["roi"])
            band = box_to_band(self._meta, x, y, width, height)
            target_id = item["target_id"]
            if target_id is None:
                key = self._next_target_key(used_keys)
                used_keys.add(key)
                target_id = workspace.add_target(
                    self._asset_id, key, "segment", for_detection=True)["id"]
                item["target_id"] = target_id
                created += 1
            start = int(round(band["t_start_s"] * rate))
            end = int(round(band["t_end_s"] * rate))
            start = max(0, min(start, count - 1))
            end = max(start + 1, min(end, count))
            # 本页只编辑时间与频率：其余字段从上一版本沿用，避免被写成 NULL
            carried = carryover_fields(workspace.current_target_version(target_id))
            version = workspace.append_target_version(
                target_id, source="manual", sample_start=start, sample_end=end,
                f_low_hz=band["f_low_hz"], f_high_hz=band["f_high_hz"], **carried)
            workspace.append_detection_label(
                self._task_set_id, target_id, source="manual",
                target_version_id=version["id"], include=True)
            kept.add(target_id)
        self._created_targets = created
        return kept

    def _write_removals(self, kept):
        """保存时没有框的适用目标写负标注（追加版本，不就地覆盖）。

        以库里的目标为准而不是渲染时的快照：同一次会话中新增又删除的目标也要写负标注；
        从未标注过的目标在这里一并确认为「无信号」，否则它们会让整条资产进不了训练。
        返回 ``(写入条数, 因缺少参考参数版本而跳过的条数)``。
        """
        workspace = self.window.workspace
        written = skipped = 0
        for target in workspace.list_targets(self._asset_id, with_current=True):
            if target["id"] in kept or not target["for_detection"]:
                continue
            if not detection_label_applies(self._semantics, target):
                continue
            label = workspace.current_label("detection", self._task_set_id, target["id"])
            if label is not None and int(label.get("include", 1)) == 0:
                continue
            if target["current"] is None:
                # 没有参考参数版本的目标无法标注（训练侧同样按“缺少参数版本”排除）
                skipped += 1
                continue
            workspace.append_detection_label(self._task_set_id, target["id"],
                                            source="manual", include=False)
            self._negative.add(target["id"])
            written += 1
        return written, skipped

    def _next_target_key(self, used):
        index = len(used) + 1
        while f"m{index}" in used:
            index += 1
        return f"m{index}"

    def _roi_box(self, roi):
        size = float(self._size)
        position, dimensions = roi.pos(), roi.size()
        return ((position.x() + dimensions.x() / 2) / size,
                (position.y() + dimensions.y() / 2) / size,
                dimensions.x() / size, dimensions.y() / size)

    # --------------------------------------------------------------- 完成动作
    def mark_no_signal(self):
        if self._asset_id is None or self._task_set_id is None:
            self.annotate_status_error("请先在左侧选择当前集合内的数据资产")
            return
        self._loading = True
        self.clear_rois()
        self._loading = False
        self.dirty = True
        if not self.save_annotation(negative_kind="no_signal"):
            return
        self.annotation_status.setText("已保存 · 已标记该资产没有信号（负样本）")
        self.window.asset_changed()

    def enable_detection_targets(self):
        asset = self.window.selected_asset()
        if asset is None:
            self.annotate_status_error("请先在左侧选择数据资产")
            return
        if self.dirty and not self.save_annotation():
            return
        try:
            targets = [target for target in
                       self.window.workspace.list_targets(asset["id"])
                       if not target["for_detection"]]
            for target in targets:
                self.window.workspace.update_target_flags(target["id"], for_detection=True)
        except ValueError as exc:
            self.annotate_status_error(exc)
            return
        self.refresh_annotation()
        self.annotation_status.setText(
            f"已把 {len(targets)} 个目标标记为适用检测" if targets
            else "该资产的目标都已适用检测")
