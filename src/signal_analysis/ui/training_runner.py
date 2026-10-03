"""External training-process lifecycle and experiment persistence."""
import codecs
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from PySide6 import QtCore

from ..services.training_jobs import save_record
from .training_process import PROCESS_BOOTSTRAP, ProcessTree


@dataclass
class _Run:
    token: str
    owner: str
    task_kind: str
    config: dict
    identifier: str
    task_id: str | None = None
    directory: Path | None = None
    record: dict | None = None
    process: QtCore.QProcess | None = None
    process_tree: ProcessTree | None = None
    decoder: object = None
    buffer: str = ""
    stopping: bool = False
    finalizing: bool = False
    force_sent: bool = False
    deadline: float | None = None
    exit_code: int | None = None
    exit_status: object = None
    error: str | None = None
    persistence_error: str | None = None
    unexpected_children: bool = False
    started: bool = False


class TrainingRunner(QtCore.QObject):
    """Own one page's QProcess while the coordinator owns the window-wide slot."""

    log_line = QtCore.Signal(str)
    stage_changed = QtCore.Signal(str)
    metrics_changed = QtCore.Signal(list)
    state_changed = QtCore.Signal(object)
    run_finished = QtCore.Signal(object)

    def __init__(self, window, page, coordinator):
        super().__init__(page)
        self.window = window
        self.page = page
        self.coordinator = coordinator
        self._run = None
        self._last_record = None
        self._last_directory = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_process_tree)

    @property
    def active(self):
        return self._run is not None

    @property
    def task_id(self):
        return None if self._run is None else self._run.task_id

    @property
    def record(self):
        return self._last_record if self._run is None else self._run.record

    @property
    def directory(self):
        return self._last_directory if self._run is None else self._run.directory

    def start(self, config, *, owner, task_kind, validate_config):
        if self._run is not None:
            self.page.report_error("本页已有训练任务正在收尾")
            return False
        token, reason = self.coordinator.acquire(owner, task_kind)
        if token is None:
            self.page.report_error(reason)
            return False

        run = _Run(token=token, owner=owner, task_kind=task_kind,
                   config=dict(config), identifier=(
                       datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                       + "-" + uuid.uuid4().hex[:8]))
        run.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._run = run
        self.state_changed.emit({"state": "starting", "owner": owner})
        try:
            validate_config(run.config)
            python = shutil.which(run.config["python"])
            repository = Path(run.config["repository"]).expanduser().resolve()
            worker = repository / "training" / "desktop_worker.py"
            if not python or not worker.is_file():
                raise ValueError("请选择有效的训练 Python 和包含 training/desktop_worker.py 的源码目录")

            run.process_tree = ProcessTree()
            task, reason = self.window.start_job(
                "external_training", owner=owner,
                label=self.page.training_label(run.config),
                cancel_text="取消本次训练", kind="external",
                buttons=self.page.task_buttons(),
                on_cancel=lambda _task, run_id=run.identifier: self._request_stop(run_id))
            if task is None:
                raise ValueError(reason or "训练任务未能登记")
            run.task_id = task.id
            self.coordinator.bind_task(token, task)

            run.directory = self.page.root / "runs" / run.identifier
            run.directory.mkdir(parents=True, exist_ok=False)
            run.record = {"id": run.identifier, "status": "running",
                          "config": run.config, "stage": "启动",
                          "metrics": [], "error": None,
                          "warnings": ([run.process_tree.warning]
                                       if run.process_tree.warning else [])}
            self._save_record(run)

            process = QtCore.QProcess(self)
            run.process = process
            process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
            process.setWorkingDirectory(str(repository))
            environment = QtCore.QProcessEnvironment.systemEnvironment()
            environment.insert("PYTHONUNBUFFERED", "1")
            environment.insert("OMP_NUM_THREADS", "2")
            environment.insert("MKL_NUM_THREADS", "2")
            environment.insert("PYTHONIOENCODING", "utf-8")
            environment.insert("PYTHONUTF8", "1")
            for name, value in run.process_tree.environment().items():
                environment.insert(name, value)
            process.setProcessEnvironment(environment)
            process.readyReadStandardOutput.connect(
                lambda run_id=run.identifier: self._read_output(run_id))
            process.finished.connect(
                lambda code, status, run_id=run.identifier:
                self._process_finished(run_id, code, status))
            process.errorOccurred.connect(
                lambda error, run_id=run.identifier:
                self._process_error(run_id, error))
            process.started.connect(lambda run_id=run.identifier: self._process_started(run_id))
            self._reset_monitor()
            if run.process_tree.warning:
                self.log_line.emit(f"警告：{run.process_tree.warning}")
            process.start(python, ["-u", "-c", PROCESS_BOOTSTRAP,
                                   str(worker), str(run.directory / "experiment.json")])
            self.state_changed.emit({"state": "running", "owner": owner})
            return True
        except Exception as exc:
            self._startup_failed(run, str(exc))
            return False

    def _process_started(self, run_id):
        run = self._current(run_id)
        if run is None or run.process is None:
            return
        try:
            run.started = True
            run.process_tree.attach(int(run.process.processId()))
            run.record["pid"] = int(run.process.processId())
            self._save_record(run)
            self.window.tasks.notify_progress(
                run.task_id, {"stage": "训练进程已启动", "message": "训练进程已启动"})
        except (OSError, RuntimeError) as exc:
            run.error = f"无法监管训练进程树：{exc}"
            self.page.report_error(run.error)
            self._request_stop(run_id)

    def _process_error(self, run_id, error):
        run = self._current(run_id)
        if run is None or run.finalizing:
            return
        if error == QtCore.QProcess.ProcessError.FailedToStart:
            message = run.process.errorString() if run.process is not None else "训练进程启动失败"
            run.error = message
            self._begin_finalizing(run, -1, QtCore.QProcess.ExitStatus.CrashExit)
        elif (error == QtCore.QProcess.ProcessError.Crashed and run.process is not None
              and run.process.state() == QtCore.QProcess.ProcessState.NotRunning):
            run.error = run.process.errorString() or "训练进程崩溃"
            self._begin_finalizing(run, run.process.exitCode(), run.process.exitStatus())
        elif run.process is not None:
            self.page.report_error(f"训练进程错误：{run.process.errorString()}")

    def _read_output(self, run_id):
        run = self._current(run_id)
        if run is None or run.process is None:
            return
        data = bytes(run.process.readAllStandardOutput())
        if not data:
            return
        try:
            with (run.directory / "worker.log").open("ab") as handle:
                handle.write(data)
        except OSError as exc:
            run.persistence_error = f"训练日志无法写入：{exc}"
            self.page.report_error(run.persistence_error)
        run.buffer += run.decoder.decode(data)
        while "\n" in run.buffer:
            line, run.buffer = run.buffer.split("\n", 1)
            self._handle_line(run, line)

    def _handle_line(self, run, line):
        self.log_line.emit(line)
        if not line.startswith("TRAIN_EVENT "):
            return
        try:
            item = json.loads(line[len("TRAIN_EVENT "):])
            if not isinstance(item, dict):
                raise ValueError("训练事件必须是 JSON 对象")
        except (ValueError, TypeError) as exc:
            self.page.report_error(f"忽略无效训练事件：{exc}")
            return
        if "stage" in item:
            stage = str(item["stage"])
            run.record["stage"] = stage
            self.stage_changed.emit(stage)
        if "epoch" in item:
            run.record["metrics"].append(item)
            self.metrics_changed.emit(list(run.record["metrics"]))
        try:
            self._save_record(run)
        except OSError:
            pass
        if run.task_id:
            self.window.tasks.notify_progress(run.task_id,
                                              self._progress_of(run, item))

    @staticmethod
    def _progress_of(run, item):
        """把阶段/轮次事件归一化成任务横幅的 ``done/total``。

        ``TaskManager`` 只做字典合并，所以换阶段必须显式覆盖 done/total，否则横幅会一直
        停在上一阶段（例如快照阶段的 2/2）；轮次事件只带 ``epoch``，总量取自运行配置。
        """
        if "epoch" in item:
            total = int(run.config.get("epochs") or 0)
            epoch = int(item["epoch"])
            loss = item.get("loss")
            progress = {"done": 0, "total": 0, "message": f"训练第 {epoch} 轮"}
            if total > 0:
                progress.update(done=min(epoch, total), total=total,
                                message=f"训练第 {epoch}/{total} 轮")
            if isinstance(loss, float) or isinstance(loss, int):
                progress["message"] += f" · loss {float(loss):.4f}"
            return progress
        stage = str(item.get("stage") or "")
        progress = {"stage": stage, "message": str(item.get("message") or stage)}
        if "total" in item or "done" in item:
            progress["done"] = int(item.get("done") or 0)
            progress["total"] = int(item.get("total") or 0)
        else:
            progress.update(done=0, total=0)
        return progress

    def _process_finished(self, run_id, code, exit_status):
        run = self._current(run_id)
        if run is None:
            return
        self._read_output(run_id)
        run.buffer += run.decoder.decode(b"", final=True)
        if run.buffer:
            self._handle_line(run, run.buffer)
            run.buffer = ""
        self._begin_finalizing(run, code, exit_status)

    def _begin_finalizing(self, run, code, exit_status):
        if run.finalizing:
            return
        run.finalizing = True
        run.exit_code = int(code)
        run.exit_status = exit_status
        if not run.stopping and (code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit):
            run.error = run.error or f"训练进程异常结束（退出码 {code}）"
        run.deadline = time.monotonic() + 2.0
        self.state_changed.emit({"state": "finalizing", "owner": run.owner})
        self._poll_timer.start()
        self._poll_process_tree()

    def _poll_process_tree(self):
        run = self._run
        if run is None or not run.finalizing:
            self._poll_timer.stop()
            return
        try:
            active = run.process_tree.active_count(run.process)
        except OSError as exc:
            self.page.report_error(f"无法确认训练子进程是否已退出：{exc}；训练运行槽保持占用")
            if run.deadline is not None and time.monotonic() >= run.deadline:
                self._force_stop(run.identifier)
            return
        if active == 0:
            self._finish_run(run)
            return
        if run.deadline is not None and time.monotonic() >= run.deadline:
            if not run.stopping:
                run.unexpected_children = True
            try:
                run.process_tree.terminate(run.process)
                if run.process is not None and run.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                    run.process.kill()
                run.force_sent = True
                run.deadline = time.monotonic() + 1.0
            except OSError as exc:
                self.page.report_error(f"无法结束训练子进程：{exc}；训练运行槽保持占用")

    def _request_stop(self, run_id):
        run = self._current(run_id)
        if run is None:
            return
        first_request = not run.stopping
        run.stopping = True
        run.deadline = time.monotonic() + 2.0
        if first_request:
            self.state_changed.emit({"state": "stopping", "owner": run.owner})
        try:
            if run.process is not None and run.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                run.process_tree.request_stop(run.process)
        except OSError as exc:
            self.page.report_error(f"请求停止训练失败：{exc}")
            if (run.process is not None
                    and run.process.state() != QtCore.QProcess.ProcessState.NotRunning):
                run.process.terminate()
        QtCore.QTimer.singleShot(
            2000, lambda run_id=run.identifier: self._force_stop(run_id))

    def _force_stop(self, run_id):
        run = self._current(run_id)
        if run is None or not run.stopping:
            return
        tree_error = None
        try:
            if run.process_tree is not None:
                run.process_tree.terminate(run.process)
        except OSError as exc:
            self.page.report_error(f"强制停止训练失败：{exc}；运行槽保持占用")
            tree_error = exc
        if (run.process is not None
                and run.process.state() != QtCore.QProcess.ProcessState.NotRunning):
            run.process.kill()
        if tree_error is not None:
            QtCore.QTimer.singleShot(
                1000, lambda run_id=run.identifier: self._force_stop(run_id))

    def _startup_failed(self, run, message):
        run.error = message
        self.page.report_error(message)
        process_running = (run.process is not None and
                           run.process.state() != QtCore.QProcess.ProcessState.NotRunning)
        if process_running:
            try:
                if run.process_tree is not None:
                    run.process_tree.terminate(run.process)
            except OSError as exc:
                self.page.report_error(f"启动失败后的进程清理失败：{exc}")
            if run.process.state() != QtCore.QProcess.ProcessState.NotRunning:
                run.process.kill()
            run.finalizing = True
            run.deadline = time.monotonic()
            self._poll_timer.start()
            return
        if run.process_tree is not None:
            try:
                if run.process_tree.active_count(run.process):
                    run.process_tree.terminate(run.process)
                    run.deadline = time.monotonic() + 1.0
                    run.finalizing = True
                    self._poll_timer.start()
                    return
                run.process_tree.close()
            except OSError as exc:
                self.page.report_error(f"启动失败后的进程树仍未确认退出：{exc}")
                run.finalizing = True
                run.deadline = time.monotonic() + 1.0
                self._poll_timer.start()
                return
        self._finish_run(run, force_state="failed")

    def _finish_run(self, run, force_state=None):
        if self._current(run.identifier) is None:
            return
        self._poll_timer.stop()
        task = self.window.tasks.owner_task(run.owner) if run.task_id else None
        cancelled = bool(task is not None and task.id == run.task_id
                         and task.cancel_event.is_set())
        task_state = force_state or ("cancelled" if cancelled else
                                     "success" if run.error is None and
                                     run.persistence_error is None and
                                     not run.unexpected_children and
                                     run.exit_code == 0 and
                                     run.exit_status == QtCore.QProcess.ExitStatus.NormalExit
                                     else "failed")
        record_state = "stopped" if task_state == "cancelled" else task_state
        if run.record is not None:
            run.record.update(status=record_state, exit_code=run.exit_code,
                              finished=datetime.now(timezone.utc).isoformat())
            errors = [item for item in (run.error, run.persistence_error) if item]
            if run.unexpected_children:
                errors.append("训练 worker 退出时仍有子进程；已终止残留进程")
            run.record["error"] = "；".join(errors) or None
            try:
                self._save_record(run)
            except OSError as exc:
                task_state = "failed"
                run.record["status"] = "failed"
                run.record["error"] = f"{run.record.get('error') or ''}；实验记录无法保存：{exc}".strip("；")
                self.page.report_error(run.record["error"])
        try:
            if run.process_tree is not None:
                run.process_tree.close()
        except OSError as exc:
            self.page.report_error(f"进程树句柄清理失败：{exc}")
            run.deadline = time.monotonic() + 1.0
            self._poll_timer.start()
            return

        record = run.record
        self._last_record = record
        self._last_directory = run.directory
        if run.process is not None:
            run.process.deleteLater()
        self._run = None
        self.state_changed.emit({"state": "idle", "owner": run.owner})
        if run.task_id:
            self.window.tasks.finish_external(run.task_id, task_state,
                                              error=run.record.get("error") if run.record else run.error)
        self.coordinator.release(run.token)
        self.run_finished.emit(record or {"status": task_state, "config": run.config})

    def _save_record(self, run):
        if run.record is None or run.directory is None:
            return
        try:
            save_record(run.directory, run.record)
        except OSError as exc:
            run.persistence_error = f"实验记录无法保存：{exc}"
            self.page.report_error(run.persistence_error)
            if run.record.get("error") is None:
                run.record["error"] = run.persistence_error
            raise

    def stop(self):
        run = self._run
        if run is None or not run.task_id:
            return False
        return self.window.tasks.cancel(run.task_id)

    def shutdown(self):
        run = self._run
        if run is None:
            return True
        if run.task_id:
            self.window.tasks.cancel(run.task_id)
        process = run.process
        if process is not None and process.state() != QtCore.QProcess.ProcessState.NotRunning:
            process.waitForFinished(1000)
        current = self._current(run.identifier)
        if current is None:
            return True
        if process is not None and process.state() != QtCore.QProcess.ProcessState.NotRunning:
            try:
                run.process_tree.terminate(process)
            except OSError as exc:
                self.page.report_error(f"关窗时停止训练失败：{exc}")
            if process.state() != QtCore.QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(1000)
        try:
            if run.process_tree is not None and run.process_tree.active_count(process) == 0:
                self._finish_run(run)
        except OSError as exc:
            self.page.report_error(f"关窗时无法确认进程树已退出：{exc}")
        return self._current(run.identifier) is None

    def _reset_monitor(self):
        self.page.log.clear()
        self.page.results.clear()
        self.page.loss_curve.clear()
        self.page.accuracy_curve.clear()
        self.page.progress.setRange(0, 0)

    def _current(self, run_id):
        if self._run is not None and self._run.identifier == run_id:
            return self._run
        return None
