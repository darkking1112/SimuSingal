"""进度通道（``services/progress.py``）的写入健壮性与取消语义（不需要 GUI）。

父进程按 0.2 s 轮询 ``progress.json``，worker 用原子替换写它。Windows 上这次替换
一旦撞上父进程的读句柄就会以 ``WinError 5`` 失败，把 worker 打崩——并发轮询的那个
用例就是把该场景固定成回归测试。
"""
import json
import threading
import time
from pathlib import Path

import pytest

from signal_analysis.services import progress
from signal_analysis.services.progress import Reporter


class _Reader:
    """模拟父进程轮询：反复打开 → 读 → 关闭 ``progress.json``。

    ``hold`` 让每次读多占用一点时间（真实轮询只有几十微秒），确保与写入方重叠。
    """

    def __init__(self, path, hold=0.001):
        self.path = path
        self.hold = hold
        self.errors = []      # 意外错误：不该出现
        self.tolerated = []   # 共享类错误：轮询如实容错（等下一轮再读）
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                with self.path.open("rb") as stream:
                    stream.read()
                    time.sleep(self.hold)
            except FileNotFoundError:
                pass  # 还没写过或正在替换
            except OSError as exc:
                if progress._is_sharing_error(exc):
                    self.tolerated.append(exc)
                else:
                    self.errors.append(exc)
            time.sleep(0.002)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=5)
        return False


def test_emit_survives_concurrent_polling(tmp_path):
    """并发轮询期间必须能写完每一条进度（Windows 上不重试会 WinError 5 直接崩）。"""
    reporter = Reporter(tmp_path)
    with _Reader(tmp_path / "progress.json") as reader:
        for index in range(1, 26):
            reporter.emit(index, 25, f"第 {index} 条", force=True)
    assert reader.errors == []  # 只有共享类错误可以被轮询容错，别的错误说明测试本身有问题
    assert json.loads((tmp_path / "progress.json").read_text(encoding="utf-8")) == {
        "done": 25, "total": 25, "message": "第 25 条"}
    assert list(tmp_path.glob("*.tmp")) == []  # 临时文件都被替换走了，不留残渣


def test_replace_with_retry_recovers_from_a_held_read_handle(tmp_path, monkeypatch):
    """读句柄挡住替换时重试到成功：前两次报共享错误，第三次放行。"""
    temporary = tmp_path / "progress.tmp"
    temporary.write_text("x", encoding="utf-8")
    destination = tmp_path / "progress.json"
    real_replace = Path.replace
    attempts = []

    def flaky(self, target):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError(13, "Permission denied")
        return real_replace(self, target)

    monkeypatch.setattr(progress.Path, "replace", flaky)
    progress._replace_with_retry(temporary, destination)
    assert len(attempts) == 3
    assert destination.read_text(encoding="utf-8") == "x"


def test_replace_with_retry_gives_up_after_the_budget(tmp_path, monkeypatch):
    """一直挡住就如实抛出（不静默吞掉）。"""
    temporary = tmp_path / "progress.tmp"
    temporary.write_text("x", encoding="utf-8")

    def blocked(self, target):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(progress.Path, "replace", blocked)
    monkeypatch.setattr(progress, "_REPLACE_DELAY_S", 0.0)
    with pytest.raises(PermissionError):
        progress._replace_with_retry(temporary, tmp_path / "progress.json")


def test_replace_with_retry_does_not_swallow_other_errors(tmp_path, monkeypatch):
    """非共享类错误（如磁盘满）立刻抛出，不重试。"""
    temporary = tmp_path / "progress.tmp"
    temporary.write_text("x", encoding="utf-8")
    attempts = []

    def full_disk(self, target):
        attempts.append(1)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(progress.Path, "replace", full_disk)
    with pytest.raises(OSError, match="No space left"):
        progress._replace_with_retry(temporary, tmp_path / "progress.json")
    assert len(attempts) == 1


def test_emit_is_throttled_unless_forced(tmp_path):
    """默认 0.25 s 节流；``force=True`` 时每次都写。"""
    reporter = Reporter(tmp_path)
    reporter.emit(1, 3, "开始", force=True)
    reporter.emit(2, 3, "被节流")
    assert json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))["done"] == 1
    reporter.emit(3, 3, "强制写入", force=True)
    assert json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))["done"] == 3


def test_emit_carries_extra_fields(tmp_path):
    reporter = Reporter(tmp_path)
    reporter.emit(1, 2, "分块", force=True, shard_ids=["a", "b"])
    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["shard_ids"] == ["a", "b"] and payload["done"] == 1


def test_reporter_without_job_dir_is_a_noop():
    reporter = Reporter(None)
    reporter.emit(1, 2, "无目录")
    assert reporter.directory is None and not reporter.cancelled()


def test_cancelled_follows_the_flag_file(tmp_path):
    reporter = Reporter(tmp_path)
    assert not reporter.cancelled()
    (tmp_path / "cancel.flag").write_text("", encoding="utf-8")
    assert reporter.cancelled()
