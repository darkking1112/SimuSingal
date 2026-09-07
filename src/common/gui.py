"""Shared desktop shell and task widgets; no business imports."""
import threading
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from .reports import export_report

class JobSignals(QtCore.QObject):
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)


class JobRunner(QtCore.QRunnable):
    def __init__(self, request, executor):
        super().__init__()
        self.request = request
        self.executor = executor
        self.signals = JobSignals()
        self.cancelled = threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.completed.emit(self.executor(self.request, cancel=self.cancelled))
        except Exception as exc:
            self.signals.failed.emit(str(exc))


class DesktopWindow(QtWidgets.QMainWindow):
    def __init__(self, workspace, title, subtitle):
        super().__init__()
        self.workspace = workspace
        self.active_job = None
        self.last_result = None
        self.tab_results = {}
        self.setWindowTitle(title)
        self.resize(1440, 950)
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        pg.setConfigOptions(background="#ffffff", foreground="#34465a", antialias=True)
        self.setStyleSheet("""
            QMainWindow,QWidget {background:#f3f6fa;color:#23374d;font-size:13px;}
            QLineEdit,QDoubleSpinBox,QSpinBox,QListWidget,QPlainTextEdit,QTableWidget {
                background:white;border:1px solid #d8e1ec;border-radius:5px;padding:5px;}
            QPushButton {background:#e4edf8;border:0;border-radius:5px;padding:9px 14px;}
            QPushButton:hover {background:#cddff3;} QPushButton:disabled {color:#97a6b7;}
            QPushButton#primary {background:#2365b3;color:white;}
            QTabBar::tab {padding:11px 24px;} QTabBar::tab:selected {background:white;color:#2365b3;}
            QLabel#title {font-size:25px;font-weight:600;color:#183c65;}
        """)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(24, 18, 24, 16)
        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("title")
        layout.addWidget(title_label)
        layout.addWidget(QtWidgets.QLabel(subtitle))
        splitter = QtWidgets.QSplitter()
        sidebar = self.build_sidebar()
        if sidebar is not None:
            splitter.addWidget(sidebar)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.build_page(), self.page_title)
        self.tabs.addTab(self.build_history(), "运行记录")
        splitter.addWidget(self.tabs)
        splitter.setSizes([290, 1100])
        layout.addWidget(splitter, 1)
        bottom = QtWidgets.QHBoxLayout()
        self.status = QtWidgets.QLabel("就绪")
        bottom.addWidget(self.status, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        self.cancel_button = QtWidgets.QPushButton("取消任务")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_job)
        bottom.addWidget(self.cancel_button)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.refresh_history()


    def build_history(self):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(box)
        layout.addWidget(QtWidgets.QLabel("最近 100 次成功运行 · 双击查看或回放"))
        self.history = QtWidgets.QListWidget()
        self.history.itemDoubleClicked.connect(self.open_history)
        layout.addWidget(self.history)
        return box


    def start_job(self, action, **kwargs):
        if self.active_job is not None:
            return
        job = JobRunner({"workspace": str(self.workspace.root), "action": action, **kwargs}, self.run_task)
        self.active_job = job
        job.signals.completed.connect(self.job_completed)
        job.signals.failed.connect(self.job_failed)
        self.set_busy(True)
        self.status.setText("任务运行中，可取消…")
        self.pool.start(job)


    def set_busy(self, busy):
        for button in self.job_buttons():
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.progress.setRange(0, 0 if busy else 1)
        if not busy:
            self.progress.setValue(1)


    @QtCore.Slot(str)
    def job_failed(self, message):
        self.active_job = None
        self.set_busy(False)
        self.status.setText(f"任务未完成：{message}")


    def cancel_job(self):
        if self.active_job:
            self.active_job.cancelled.set()


    def refresh_history(self):
        self.history.clear()
        for result in self.workspace.list_runs():
            item = QtWidgets.QListWidgetItem(f"{result['created_at'][:19]}  ·  {result['kind']}  ·  {result['id'][:8]}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, result["id"])
            self.history.addItem(item)


    def open_history(self, item):
        try:
            self.display_result(self.workspace.get_run(item.data(QtCore.Qt.ItemDataRole.UserRole)))
            self.status.setText("已读取历史结果")
        except (ValueError, OSError) as exc:
            self.status.setText(f"无法读取历史结果：{exc}")


    def export_current(self):
        result = self.last_result
        if result is None:
            self.status.setText("请先完成一次分析或仿真")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "导出报告", "report.html", "HTML (*.html);;JSON (*.json)")
        if path:
            try:
                export_report(result, path)
                self.status.setText(f"报告已导出：{path}")
            except (ValueError, OSError) as exc:
                self.status.setText(f"导出失败：{exc}")


    def closeEvent(self, event):
        self.cancel_job()
        self.pool.waitForDone(5000)
        if self.pool.activeThreadCount():
            self.status.setText("正在结束后台任务，请稍后关闭")
            event.ignore()
        else:
            event.accept()


    def build_sidebar(self):
        return None

    @QtCore.Slot(object)
    def job_completed(self, result):
        self.active_job = None
        self.set_busy(False)
        self.refresh_history()
        try:
            self.result_ready(result)
            self.status.setText("任务完成 · 结果已保存到本项目工作目录")
        except Exception as exc:
            self.status.setText(f"结果已保存，但显示失败：{exc}")

