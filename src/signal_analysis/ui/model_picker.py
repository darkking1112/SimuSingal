"""已登记模型下拉：选中即把清单路径写回配套的 QLineEdit。

页面上的 ``QLineEdit`` 仍是唯一数据源（``text()`` 就是清单路径），下拉只负责按用途
列出模型库里已登记的模型；「浏览本地文件…」保留选择未登记清单的旧用法。
"""
from pathlib import Path

from PySide6 import QtWidgets

from ..services import model_store

#: 「浏览本地文件…」项的哨兵数据；空选择项的数据为 ``None``。
BROWSE = object()

#: 未指定模型时的统一首项文案：三页都用它，保证「默认通路」的命名一致。
DEFAULT_EMPTY_TEXT = "默认（不使用已登记模型）"

_BASE_TOOLTIP = ("模型库（数据管理 → 模型管理）里登记且用途匹配的模型；"
                 "「浏览本地文件…」可继续使用未登记的清单")


class ModelPicker(QtWidgets.QComboBox):
    """按 ``purpose`` 过滤模型库条目的下拉；用途与页面对应关系见 ``model_store``。

    ``hint`` 是该页"默认通路是什么/对清单有什么额外要求"的补充说明，拼进 tooltip；
    选中项只把清单路径写回 ``line_edit``（页面用隐藏的 QLineEdit 存路径，界面只剩下拉）。
    """

    def __init__(self, line_edit, purpose, window, *, empty_text=DEFAULT_EMPTY_TEXT,
                 browse=None, hint="", parent=None):
        super().__init__(parent)
        self.line_edit = line_edit
        self.purpose = purpose
        self.window = window
        self.empty_text = empty_text
        self.browse = browse
        self._syncing = False
        self.setMinimumWidth(260)
        self.setToolTip(f"{hint}\n{_BASE_TOOLTIP}" if hint else _BASE_TOOLTIP)
        self.currentIndexChanged.connect(self._apply)
        line_edit.textChanged.connect(self.sync)
        self.refresh()

    # ------------------------------------------------------------------ 填充
    def refresh(self):
        """重填下拉；用于训练入库、模型管理增删之后同步界面。"""
        text = self.line_edit.text().strip()
        registered = [entry for entry in model_store.list_models(self.window.workspace.root)
                      if entry.get("purpose") == self.purpose and entry.get("status") == "ok"]
        self._syncing = True
        try:
            self.clear()
            self.addItem(self.empty_text, None)
            for entry in registered:
                self.addItem(self._label(entry), entry["path"])
            if text and self.findData(text) < 0:
                self.insertItem(1, f"当前：{Path(text).name}（未登记）", text)
            self.addItem("浏览本地文件…", BROWSE)
            index = self.findData(text) if text else 0
            self.setCurrentIndex(max(index, 0))
        finally:
            self._syncing = False

    def sync(self, *_):
        if not self._syncing:
            self.refresh()

    @staticmethod
    def _label(entry):
        parts = [str(entry.get("name") or ""), str(entry.get("model_type") or "")]
        created = str(entry.get("created_at") or "")[:10]
        if created:
            parts.append(created)
        metrics = entry.get("metrics") or {}
        if metrics.get("accuracy") is not None:
            parts.append(f"准确率 {100 * float(metrics['accuracy']):.1f}%")
        elif (metrics.get("val") or {}).get("f1") is not None:
            parts.append(f"验证 F1 {100 * float(metrics['val']['f1']):.1f}%")
        return " · ".join(part for part in parts if part)

    # ------------------------------------------------------------------ 选择
    def _apply(self, index):
        if self._syncing:
            return
        data = self.itemData(index)
        if data is BROWSE:
            if self.browse is not None:
                self.browse()
            self.refresh()
        elif data is None:
            if self.line_edit.text().strip():
                self.line_edit.setText("")
        elif data != self.line_edit.text().strip():
            self.line_edit.setText(data)
