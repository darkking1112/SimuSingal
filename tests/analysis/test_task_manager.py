"""任务体系 P1：任务管理器——并发、同页单任务、独占规则、独立取消与兼容属性。"""
import os
import time

# 无显示环境：必须在导入 PySide6 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets

from signal_analysis.ui import MainWindow

# 本文件全部用例都要起 QApplication + MainWindow
pytestmark = pytest.mark.gui


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(app, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _fake_executor(holder, iterations=4000, step=0.005):
    """可控假执行器：循环检查取消事件；周期性上报进度。"""

    def run(request, cancel=None, progress=None):
        name = request.get("name", "任务")
        holder.setdefault("started", []).append(name)
        try:
            for index in range(iterations):
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("已取消")
                if progress is not None and index % 100 == 0:
                    progress({"done": index, "total": iterations,
                              "message": f"处理 {name} {index}/{iterations}"})
                time.sleep(step)
            holder.setdefault("finished", []).append(name)
            return {"kind": "fake_result", "name": name}
        finally:
            holder.setdefault("stopped", []).append(name)

    return run


def _make_window(tmp_path, holder):
    app = _app()
    window = MainWindow(tmp_path)
    window.run_task = _fake_executor(holder)
    window.show()
    return app, window


def _teardown(app, window):
    window.cancel_tasks()
    _wait(app, lambda: window.tasks.count() == 0)
    window.close()


def test_two_pages_run_concurrently_and_cancel_independently(tmp_path):
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        first, reason = window.start_job("fake", owner="页面A", label="任务A", name="A")
        assert reason is None and first is not None
        second, reason = window.start_job("fake", owner="页面B", label="任务B", name="B")
        assert reason is None and second is not None
        assert _wait(app, lambda: {"A", "B"} <= set(holder.get("started", [])))
        assert window.tasks.count() == 2
        assert window.task_summary.text() == "正在运行 2 个任务"
        assert window.active_job is not None

        # 进度写回任务对象与状态栏文字
        assert _wait(app, lambda: bool(first.progress) or bool(second.progress))
        assert "处理" in window.status.text()

        # 取消 A：B 不受影响
        window.tasks.cancel(first.id)
        assert _wait(app, lambda: first.state == "cancelled")
        assert window.tasks.count() == 1
        assert "B" not in holder.get("stopped", [])
        assert second.state == "running"

        # 取消 B：全部收尾
        window.tasks.cancel(second.id)
        assert _wait(app, lambda: second.state == "cancelled")
        assert window.tasks.count() == 0
        assert window.task_summary.text() == "无运行中任务"
        assert window.active_job is None
    finally:
        _teardown(app, window)


def test_same_page_second_task_refused(tmp_path):
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        first, reason = window.start_job("fake", owner="页面A", label="任务A", name="A")
        assert reason is None
        assert _wait(app, lambda: "A" in holder.get("started", []))

        second, reason = window.start_job("fake", owner="页面A", label="任务A2", name="A2")
        assert second is None
        assert "已有任务在运行" in reason
        assert window.status.text() == reason
        assert window.tasks.count() == 1

        window.tasks.cancel(first.id)
        assert _wait(app, lambda: window.tasks.count() == 0)
        third, reason = window.start_job("fake", owner="页面A", label="任务A3", name="A3")
        assert reason is None and third is not None
    finally:
        _teardown(app, window)


def test_exclusive_actions_block_and_are_blocked(tmp_path):
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        first, _ = window.start_job("fake", owner="页面A", label="任务A", name="A")
        assert _wait(app, lambda: "A" in holder.get("started", []))

        blocked, reason = window.start_job("storage_cleanup", owner="数据管理", label="数据清理")
        assert blocked is None
        assert "独占执行" in reason

        window.tasks.cancel(first.id)
        assert _wait(app, lambda: window.tasks.count() == 0)

        cleanup, reason = window.start_job("storage_cleanup", owner="数据管理", label="数据清理")
        assert reason is None and cleanup is not None
        assert _wait(app, lambda: window.tasks.count() == 1)

        other, reason = window.start_job("fake", owner="页面B", label="任务B", name="B")
        assert other is None
        assert "独占执行" in reason
    finally:
        _teardown(app, window)


def test_active_job_compat_property(tmp_path):
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        assert window.active_job is None
        task, reason = window.start_job("fake", owner="页面A", label="任务A", name="A")
        assert reason is None
        assert window.active_job is task
        assert not task.cancelled.is_set()

        # 旧接口用法：直接 set 取消事件（等价于点底部“取消任务”）
        task.cancelled.set()
        assert _wait(app, lambda: window.active_job is None)
        assert task.state == "cancelled"
    finally:
        _teardown(app, window)


def test_banners_show_concurrent_tasks_and_cancel_in_page(tmp_path):
    """页内横幅：并发任务各显示各的；页内取消只作用于本页任务；切页往返状态保留。"""
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        detect_task, _ = window.start_job("fake", owner="信号检测",
                                          label="能量检测 · A", name="D")
        analysis_task, _ = window.start_job("fake", owner="态势显示",
                                            label="态势显示 · B", name="A")
        assert _wait(app, lambda: {"D", "A"} <= set(holder.get("started", [])))
        detect_banner = window._task_banners["信号检测"]
        analysis_banner = window._task_banners["态势显示"]
        assert "能量检测" in detect_banner.label.text()
        assert "态势显示" in analysis_banner.label.text()

        # 页内横幅在各自页面可见（切到该页查看；未切到的页不抢焦点）
        window.tabs.setCurrentIndex(window._page_index("信号检测"))
        assert detect_banner.isVisible()
        assert detect_banner.cancel_button.isEnabled()

        # 点“信号检测”横幅的取消按钮：只取消该任务
        detect_banner.cancel_button.click()
        assert detect_task.cancel_event.is_set()
        assert not analysis_task.cancel_event.is_set()
        assert _wait(app, lambda: detect_task.state == "cancelled")
        assert analysis_task.state == "running"
        assert _wait(app, lambda: "已取消" in detect_banner.label.text())

        # 切到“态势显示”页：另一任务仍在运行且横幅可见；切页往返状态保留
        window.tabs.setCurrentIndex(window._page_index("态势显示"))
        assert analysis_banner.isVisible()
        window.tabs.setCurrentIndex(window._page_index("数据管理"))
        window.tabs.setCurrentIndex(window._page_index("态势显示"))
        assert analysis_banner.isVisible()
        assert "态势显示" in analysis_banner.label.text()

        window.tasks.cancel(analysis_task.id)
        assert _wait(app, lambda: analysis_task.state == "cancelled")
        assert _wait(app, lambda: "已取消" in analysis_banner.label.text())
    finally:
        _teardown(app, window)


def test_refusal_and_exclusive_conflict_show_in_banner(tmp_path):
    """同页重复启动返回明确原因；独占任务冲突时，空闲页面的横幅显示原因。"""
    holder = {}
    app, window = _make_window(tmp_path, holder)
    try:
        cleanup, reason = window.start_job("storage_cleanup", owner="数据管理",
                                           label="数据清理")
        assert reason is None and cleanup is not None
        assert _wait(app, lambda: window.tasks.count() == 1)

        # 同页重复启动：原因写回状态栏，横幅继续显示正在运行的任务
        again, reason = window.start_job("storage_cleanup", owner="数据管理",
                                         label="数据清理2")
        assert again is None
        assert "已有任务在运行" in reason
        assert window.status.text() == reason
        assert "数据清理" in window._task_banners["数据管理"].label.text()

        # 其它页面被独占规则拒绝：该页横幅（空闲）显示原因
        blocked, reason = window.start_job("fake", owner="信号检测",
                                           label="能量检测 · A", name="D")
        assert blocked is None
        assert "独占执行" in reason
        window.tabs.setCurrentIndex(window._page_index("信号检测"))
        detect_banner = window._task_banners["信号检测"]
        assert detect_banner.isVisible()
        assert "独占执行" in detect_banner.label.text()

        window.tasks.cancel(cleanup.id)
        assert _wait(app, lambda: window.tasks.count() == 0)
    finally:
        _teardown(app, window)


def test_finished_banner_uses_result_status_text(tmp_path):
    """真实任务收尾：横幅显示 result_status 的统计文案（结果回到发起页）。"""
    app = _app()
    window = MainWindow(tmp_path)
    window.show()
    try:
        task, reason = window.start_job("generate", owner="IQ 信号生成",
                                        label="生成信号",
                                        sample_rate=100_000.0, duration=0.04096,
                                        seed=0, signals=[],
                                        noise={"enabled": True, "bandwidth": 100_000.0,
                                               "power_dbfs": -20.0},
                                        name=None, export=None)
        assert reason is None and task is not None
        assert _wait(app, lambda: window.tasks.count() == 0, timeout=30)
        window.tabs.setCurrentIndex(window._page_index("IQ 信号生成"))
        banner = window._task_banners["IQ 信号生成"]
        assert banner.isVisible()
        assert "任务完成" in banner.label.text()
        assert banner.cancel_button.isVisible() is False
    finally:
        window.close()

