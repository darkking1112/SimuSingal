"""worker 侧进度与取消通道：写 ``progress.json``、读 ``cancel.flag``。

父进程（``common/tasks.py::run_job``）以 0.2 s 间隔轮询 ``progress.json`` 并回调
进度；``cancel_grace`` > 0 时，取消由父进程写 ``cancel.flag``，worker 在任务边界
（如逐个文件、逐条录制）检查后优雅收尾并返回部分结果。
"""
import json
import time
from pathlib import Path


class Reporter:
    """进度与取消通道：worker 写 ``progress.json``，父进程写 ``cancel.flag``。"""

    def __init__(self, job_dir):
        self.directory = Path(job_dir) if job_dir else None
        self._last = 0.0

    def emit(self, done, total, message, *, force=False, **extra):
        if self.directory is None:
            return
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        path = self.directory / "progress.json"
        temporary = path.with_suffix(".tmp")
        payload = {"done": int(done), "total": int(total), "message": message, **extra}
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def cancelled(self):
        return self.directory is not None and (self.directory / "cancel.flag").exists()
