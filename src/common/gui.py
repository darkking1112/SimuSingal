"""Shared desktop shell and task widgets; no business imports.

任务模型（2026-10）：每个后台任务都有独立标识、归属页面（``owner``，如“信号导入”）、
取消事件与进度。同一 owner 同一时刻只允许一个任务；不同 owner 允许并发。
``storage_cleanup`` 与 ``migrate_legacy`` 是独占动作，与任何运行中任务互斥。
窗口底部只保留状态文字与“正在运行 N 个任务”汇总；单个任务自己的进度与取消
由发起页面内的 ``TaskBanner`` 提供（页面用 ``register_task_banner`` 注册）。

兼容接口（旧代码与另一项目仍在使用）：``window.active_job`` 属性、``cancel_job()``、
``set_busy()``、``job_buttons()`` 保持可用；``start_job()`` 新增可选参数
``owner / label / cancel_text / buttons``，并返回 ``(task, reason)``。
"""
import inspect
import threading
import time
import uuid

from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from .reports import export_report

#: 独占动作：与任何运行中任务互斥（清理会删除文件、迁移会批量改库）。
EXCLUSIVE_ACTIONS = frozenset({"storage_cleanup", "migrate_legacy"})


class UiTask:
    """界面层任务状态：独立标识 / 归属页面 / 独立取消事件与进度。"""

    def __init__(self, owner, label, action, cancel_text="取消任务", kind="job"):
        self.id = uuid.uuid4().hex[:12]
        self.owner = owner
        self.label = label
        self.action = action
        self.cancel_text = cancel_text
        self.kind = kind            # "job"=子进程任务 | "external"=页面自管进程
        self.state = "running"      # running / success / failed / cancelled
        self.progress = {}          # {done, total, message, stage, ...}
        self.cancel_event = threading.Event()
        self.started = time.monotonic()
        self.finished_at = None
        self.error = None
        self.result = None
        self.buttons = ()
        self.on_cancel = None

    @property
    def cancelled(self):
        """兼容旧接口：``window.active_job.cancelled.set()``。"""
        return self.cancel_event

    def cancel(self):
        self.cancel_event.set()
        if self.on_cancel is not None:
            self.on_cancel(self)


class JobSignals(QtCore.QObject):
    completed = QtCore.Signal(str, object)
    failed = QtCore.Signal(str, str)
    progress = QtCore.Signal(str, object)


class JobRunner(QtCore.QRunnable):
    """后台线程里执行一个子进程任务；进度/结果/失败经信号回到任务管理器。"""

    def __init__(self, request, executor, task_id="", cancel_event=None):
        super().__init__()
        self.request = request
        self.executor = executor
        self.task_id = task_id
        self.signals = JobSignals()
        self.cancelled = cancel_event if cancel_event is not None else threading.Event()

    @QtCore.Slot()
    def run(self):
        try:
            kwargs = {"cancel": self.cancelled}
            # 只有声明了 progress 形参的执行器才收到进度回调（其它项目的执行器签名不变）
            if "progress" in inspect.signature(self.executor).parameters:
                kwargs["progress"] = (lambda info, task_id=self.task_id:
                                      self.signals.progress.emit(task_id, info))
            self.signals.completed.emit(self.task_id, self.executor(self.request, **kwargs))
        except Exception as exc:
            self.signals.failed.emit(self.task_id, str(exc))


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


class TaskManager(QtCore.QObject):
    """窗口级多任务管理器：规则检查、启动、取消与结果路由。"""

    started = QtCore.Signal(object)
    progressed = QtCore.Signal(object)
    finished = QtCore.Signal(object)

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self._tasks = {}
        self._order = []

    # ------------------------------------------------------------------ 查询
    def running(self):
        return [self._tasks[item] for item in self._order if item in self._tasks]

    def count(self):
        return len(self._tasks)

    def latest(self):
        """最近启动的运行中任务；没有则 None（供 ``active_job`` 兼容属性）。"""
        for task_id in reversed(self._order):
            task = self._tasks.get(task_id)
            if task is not None:
                return task
        return None

    def owner_task(self, owner):
        return next((task for task in self._tasks.values() if task.owner == owner), None)

    def owner_busy(self, owner):
        return self.owner_task(owner) is not None

    # ------------------------------------------------------------------ 规则
    def blocked_reason(self, owner, action):
        """启动前检查：返回拒绝原因；``None`` 表示可以启动。"""
        owner_task = self.owner_task(owner)
        if owner_task is not None:
            return f"“{owner}”已有任务在运行（{owner_task.label}）；请等待完成或先在页面内取消"
        exclusive = next((task for task in self._tasks.values()
                          if task.action in EXCLUSIVE_ACTIONS), None)
        if action in EXCLUSIVE_ACTIONS and self._tasks:
            other = self.latest()
            return f"「{other.label}」正在运行；数据清理/历史登记需要独占执行，请等待完成"
        if exclusive is not None:
            return f"「{exclusive.label}」正在独占执行，请等待完成后再启动新任务"
        return None

    # ------------------------------------------------------------- 启动/取消
    def start(self, owner, action, *, label=None, cancel_text="取消任务",
              buttons=(), kind="job", on_cancel=None, **kwargs):
        """启动一个任务；返回 ``(task, None)`` 或 ``(None, 拒绝原因)``。"""
        reason = self.blocked_reason(owner, action)
        if reason is not None:
            return None, reason
        task = UiTask(owner, label or action, action, cancel_text=cancel_text, kind=kind)
        task.buttons = tuple(button for button in buttons if button is not None)
        task.on_cancel = on_cancel
        self._tasks[task.id] = task
        self._order.append(task.id)
        for button in task.buttons:
            button.setEnabled(False)
        self.window.status.setText(f"任务运行中：{task.label}")
        self._update_summary()
        self.started.emit(task)
        if kind == "job":
            request = {"workspace": str(self.window.workspace.root), "action": action, **kwargs}
            job = JobRunner(request, self.window.run_task, task.id, task.cancel_event)
            job.signals.completed.connect(self._on_job_completed)
            job.signals.failed.connect(self._on_job_failed)
            job.signals.progress.connect(self._on_job_progress)
            self.window.pool.start(job)
        return task, None

    def cancel(self, task_id):
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.cancel()
        return True

    def cancel_owner(self, owner):
        task = self.owner_task(owner)
        if task is None:
            return False
        task.cancel()
        return True

    def cancel_all(self):
        for task in self.running():
            task.cancel()

    # -------------------------------------------- 外部任务（页面自管进程）接口
    def notify_progress(self, task_id, info):
        """外部任务（如训练 QProcess）的进度上报入口。"""
        task = self._tasks.get(task_id)
        if task is None or not isinstance(info, dict):
            return
        task.progress.update(info)
        if info.get("message"):
            self.window.status.setText(str(info["message"]))
        self.progressed.emit(task)

    def finish_external(self, task_id, state="success", error=None):
        """外部任务收尾（无结果渲染，只更新列表/横幅/汇总）。"""
        task = self._release(task_id, state, error=error)
        if task is None:
            return None
        self.finished.emit(task)
        return task

    # ------------------------------------------------------------------ 内部
    def _update_summary(self):
        summary = getattr(self.window, "task_summary", None)
        if summary is not None:
            count = self.count()
            summary.setText(f"正在运行 {count} 个任务" if count else "无运行中任务")

    def _release(self, task_id, state, *, result=None, error=None):
        """把任务移出运行表、恢复它的按钮；返回任务对象（不存在则 None）。"""
        task = self._tasks.pop(task_id, None)
        if task is None:
            return None
        task.state = state
        task.result = result
        task.error = error
        task.finished_at = time.monotonic()
        for button in task.buttons:
            try:
                button.setEnabled(True)
            except RuntimeError:  # 控件已随页面销毁
                pass
        self._update_summary()
        return task

    @QtCore.Slot(str, object)
    def _on_job_progress(self, task_id, info):
        self.notify_progress(task_id, info)

    @QtCore.Slot(str, object)
    def _on_job_completed(self, task_id, result):
        state = "cancelled" if isinstance(result, dict) and result.get("cancelled") else "success"
        task = self._release(task_id, state, result=result)
        if task is None:
            return
        try:
            self.window.result_ready(result)
            self.window.status.setText(self.window.result_status(result))
        except Exception as exc:
            self.window.status.setText(f"结果已保存，但显示失败：{exc}")
        self.finished.emit(task)

    @QtCore.Slot(str, str)
    def _on_job_failed(self, task_id, message):
        running = self._tasks.get(task_id)
        state = "cancelled" if running is not None and running.cancel_event.is_set() else "failed"
        task = self._release(task_id, state, error=message)
        if task is None:
            return
        if state == "cancelled":
            self.window.status.setText(f"任务已取消：{task.label}")
        else:
            self.window.status.setText(f"任务未完成：{message}")
        self.finished.emit(task)


class TaskBanner(QtWidgets.QFrame):
    """页内任务横幅：本页任务的进度、阶段与取消按钮。

    有总量（``total`` > 0）时显示确定进度；只有阶段消息时显示忙碌动画并原文
    展示阶段名，不虚构百分比。任务结束后保留最终文案，直到本页启动下一个任务。
    """

    def __init__(self, owner, cancel_text="取消任务", parent=None):
        super().__init__(parent)
        self.owner = owner
        self._task = None
        self.setObjectName("taskbanner")
        self.setVisible(False)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(150)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.cancel_button = QtWidgets.QPushButton(cancel_text)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)
        layout.addWidget(self.cancel_button)

    # ------------------------------------------------------------- 状态回写
    def apply_started(self, task):
        self._task = task
        self.setVisible(True)
        self.cancel_button.setText(task.cancel_text)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setVisible(True)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.label.setText(task.label)

    def apply_progress(self, task):
        if self._task is None or task.id != self._task.id:
            return
        info = task.progress
        total = int(info.get("total") or 0)
        if total > 0:
            done = min(int(info.get("done") or 0), total)
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            text = f"{task.label}：已处理 {done}/{total}"
        else:
            self.progress.setRange(0, 0)
            text = task.label
        message = info.get("message") or info.get("stage")
        if message:
            text += f" · {message}"
        self.label.setText(text)

    def apply_finished(self, task, text):
        if self._task is not None and task.id != self._task.id:
            return
        self._task = None
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)
        self.setVisible(True)
        self.label.setText(text)

    def show_refusal(self, reason):
        """本页任务被拒绝启动时的提示（不影响其它页面正在显示的任务）。"""
        if self._task is not None:
            return
        self.setVisible(True)
        self.cancel_button.setVisible(False)
        self.progress.setVisible(False)
        self.label.setText(reason)

    def _cancel(self):
        if self._task is not None:
            self._task.cancel()


class DesktopWindow(QtWidgets.QMainWindow):
    def __init__(self, workspace, title, subtitle):
        super().__init__()
        self.workspace = workspace
        self.last_result = None
        self.tab_results = {}
        self.pool = QtCore.QThreadPool(self)
        # 每个任务只占一个"等待子进程"的轻线程；允许不同页面并发运行
        self.pool.setMaxThreadCount(4)
        self.tasks = TaskManager(self)
        self._task_banners = {}
        self.tasks.started.connect(self._on_task_started)
        self.tasks.progressed.connect(self._on_task_progressed)
        self.tasks.finished.connect(self._on_task_finished)
        self.setWindowTitle(title)
        self.resize(1440, 950)
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
            QFrame#taskbanner {background:#e8f1fb;border:1px solid #cfe0f3;border-radius:5px;}
            QFrame#taskbanner QLabel {background:transparent;color:#1d4e89;}
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
        self.task_summary = QtWidgets.QLabel("无运行中任务")
        bottom.addWidget(self.task_summary)
        layout.addLayout(bottom)
        self.setCentralWidget(central)

    # ------------------------------------------------------------- 任务横幅
    def register_task_banner(self, owner, banner):
        """注册页面/面板的任务横幅；该页面任务与拒绝提示都显示在横幅内。"""
        self._task_banners[owner] = banner

    @QtCore.Slot(object)
    def _on_task_started(self, task):
        banner = self._task_banners.get(task.owner)
        if banner is not None:
            banner.apply_started(task)

    @QtCore.Slot(object)
    def _on_task_progressed(self, task):
        banner = self._task_banners.get(task.owner)
        if banner is not None:
            banner.apply_progress(task)

    @QtCore.Slot(object)
    def _on_task_finished(self, task):
        banner = self._task_banners.get(task.owner)
        if banner is not None:
            banner.apply_finished(task, self.task_finished_text(task))

    def task_finished_text(self, task):
        """任务结束文案；页面可覆写（例如复用 ``result_status`` 的统计文案）。"""
        if task.state == "cancelled":
            return f"{task.label} · 已取消"
        if task.state == "failed":
            return f"{task.label} · 失败：{task.error or ''}"
        return f"{task.label} · 完成"

    # --------------------------------------------------------- 兼容旧接口
    @property
    def active_job(self):
        """兼容属性：无运行任务→None；否则最近启动的运行任务（含 ``.cancelled``）。"""
        return self.tasks.latest()

    def task_running(self, owner=None):
        """检查运行中任务：传入 owner 只看该页面；否则看全窗口。"""
        return self.tasks.owner_busy(owner) if owner is not None else bool(self.tasks.count())

    def start_job(self, action, *, owner=None, label=None, cancel_text="取消任务",
                  buttons=None, kind="job", on_cancel=None, **kwargs):
        """启动后台任务；返回 ``(task, reason)``，``reason`` 非空表示未启动及原因。"""
        owner = owner or "默认"
        if buttons is None:
            buttons = tuple(self.owner_buttons(owner))
        task, reason = self.tasks.start(owner, action, label=label, cancel_text=cancel_text,
                                        buttons=buttons, kind=kind, on_cancel=on_cancel, **kwargs)
        if reason is not None:
            self.status.setText(reason)
            banner = self._task_banners.get(owner)
            if banner is not None:
                banner.show_refusal(reason)
        return task, reason

    def owner_buttons(self, owner):
        """该 owner 在任务期间应禁用的按钮；默认沿用整窗按钮集合（旧行为）。"""
        return tuple(self.job_buttons())

    def job_buttons(self):
        """本窗口任务相关按钮（兼容旧接口；默认空）。"""
        return ()

    def cancel_job(self):
        """兼容旧接口：取消最近启动的运行中任务。"""
        task = self.tasks.latest()
        if task is not None:
            self.tasks.cancel(task.id)

    def cancel_tasks(self):
        """取消全部运行中任务（窗口关闭等场景）。"""
        self.tasks.cancel_all()

    def set_busy(self, busy):
        """兼容旧接口：整体启停任务按钮（新代码请用按页按钮集合）。"""
        for button in self.job_buttons():
            button.setEnabled(not busy)

    def set_status_detail(self, text):
        """写入或清空状态栏第二行；文案含义由各项目决定，空文本即隐藏该行。"""
        self.status_detail.setText(text or "")
        self.status_detail.setVisible(bool(self.status_detail.text()))

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
        self.cancel_tasks()
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

    def result_ready(self, result):
        """任务结果的页面渲染入口；子类覆盖。"""
