"""worker 侧进度与取消通道：写 ``progress.json``、读 ``cancel.flag``。

父进程（``common/tasks.py::run_job``）以 0.2 s 间隔轮询 ``progress.json`` 并回调
进度；``cancel_grace`` > 0 时，取消由父进程写 ``cancel.flag``，worker 在任务边界
（如逐个文件、逐条录制）检查后优雅收尾并返回部分结果。
"""
import json
import time
from pathlib import Path

#: 原子替换进度文件的重试次数与间隔。父进程每 0.2 s 打开一次 ``progress.json`` 读，
#: 每次读句柄只存活几十微秒；Windows 上 ``os.replace`` 撞上这个读句柄会以
#: ``WinError 5``（拒绝访问）/``WinError 32``（共享冲突）失败——重试几十毫秒即可
#: 错开，否则整个任务会因为一条进度写不进去而崩掉。
_REPLACE_ATTEMPTS = 6
_REPLACE_DELAY_S = 0.02
#: 允许重试的 Windows 错误码：ERROR_ACCESS_DENIED / ERROR_SHARING_VIOLATION。
_SHARING_WINERRORS = (5, 32)


def _is_sharing_error(exc):
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in _SHARING_WINERRORS


def _replace_with_retry(temporary, path):
    """原子替换 ``temporary`` → ``path``；被并发读句柄挡住时短暂重试。"""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if not _is_sharing_error(exc) or attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY_S * (attempt + 1))


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
        _replace_with_retry(temporary, path)

    def cancelled(self):
        return self.directory is not None and (self.directory / "cancel.flag").exists()
