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


class ElidedLabel(QtWidgets.QLabel):
    """单行标签：过长时按中间省略，完整文本保留在 tooltip 与 ``text()``。

    ``QLabel`` 默认按 sizeHint 要求父窗口变宽，长绝对路径会把窗口顶大；把水平
    策略设为 ``Ignored`` 后改由布局分配宽度，超宽部分用 ``QFontMetrics.elidedText``
    截断，既不改动 ``text()`` 语义（仍是完整文本），也不会撑大窗口。
    """

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):
        """记录完整文本并按当前宽度重绘（真正写入控件的只是省略后的文本）。"""
        self._full = text or ""
        self.setToolTip(self._full)
        self._elide()

    def text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        metrics = self.fontMetrics()
        super().setText(metrics.elidedText(self._full, QtCore.Qt.TextElideMode.ElideMiddle,
                                           max(0, self.width())))


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
            QLabel#detail {background:white;border:1px solid #d8e1ec;border-radius:5px;
                padding:6px 10px;color:#4a5b6e;}
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
        splitter.addWidget(self.tabs)
        splitter.setSizes([290, 1100])
        layout.addWidget(splitter, 1)
        self.status_detail = ElidedLabel()
        self.status_detail.setObjectName("detail")
        self.status_detail.setVisible(False)
        layout.addWidget(self.status_detail)
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


    def set_status_detail(self, text):
        """写入或清空状态栏第二行；文案含义由各项目决定，空文本即隐藏该行。"""
        self.status_detail.setText(text or "")
        self.status_detail.setVisible(bool(self.status_detail.text()))


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

    def result_status(self, result):
        """任务完成后的状态栏文案；只读页面（不写运行记录）可覆盖本方法。"""
        return "任务完成 · 结果已保存到本项目工作目录"

    @QtCore.Slot(object)
    def job_completed(self, result):
        self.active_job = None
        self.set_busy(False)
        try:
            self.result_ready(result)
            self.status.setText(self.result_status(result))
        except Exception as exc:
            self.status.setText(f"结果已保存，但显示失败：{exc}")
