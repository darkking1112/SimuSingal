"""运行记录页（mixin）：最近运行列表与历史结果回放。"""
from PySide6 import QtCore, QtWidgets


class HistoryPageMixin:
    def build_history(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("最近 100 次成功运行 · 双击查看或回放"))
        self.history = QtWidgets.QListWidget()
        self.history.itemDoubleClicked.connect(self.open_history)
        layout.addWidget(self.history)
        return box

    def refresh_history(self):
        self.history.clear()
        for result in self.workspace.list_runs():
            item = QtWidgets.QListWidgetItem(
                f"{result['created_at'][:19]}  ·  {result['kind']}  ·  {result['id'][:8]}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result["id"])
            self.history.addItem(item)

    def open_history(self, item):
        try:
            self.display_result(self.workspace.get_run(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            self.status.setText("已读取历史结果")
        except (ValueError, OSError) as exc:
            self.status.setText(f"无法读取历史结果：{exc}")
