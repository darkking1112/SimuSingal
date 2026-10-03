"""Process-tree supervision for external training workers.

bootstrap 只负责 Windows 的启动门事件与转交 worker：POSIX 上的新会话由
``training/desktop_worker.py`` 自己建立（只在一处 setsid，避免第二次调用因为进程已是
会话首进程而失败）。
"""
import ctypes
import os
import signal
import uuid
from ctypes import wintypes


PROCESS_BOOTSTRAP = r"""
import ctypes
import os
import runpy
import sys
from ctypes import wintypes

if os.name == "nt":
    name = os.environ.pop("SIMUSIGNAL_TRAINING_GATE_EVENT")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenEventW.argtypes = [ctypes.c_ulong, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.OpenEventW.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    event = kernel32.OpenEventW(0x00100000, False, name)
    if not event:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        result = kernel32.WaitForSingleObject(event, 0xFFFFFFFF)
        if result != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(event)

worker, experiment = sys.argv[1:3]
sys.argv = [worker, experiment]
runpy.run_path(worker, run_name="__main__")
"""

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_TH32CS_SNAPPROCESS = 0x00000002
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x00100000
_ERROR_NO_MORE_FILES = 18
_ERROR_INVALID_PARAMETER = 87
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_WAIT_FAILED = 0xFFFFFFFF


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class ProcessTree:
    """Own a POSIX process group or a Windows Job Object for one worker run."""

    def __init__(self):
        self.pid = None
        self._job = None
        self._event = None
        self._event_name = None
        self._attached = False
        self._descendants = {}
        self.warning = None
        if os.name == "nt":
            self._prepare_windows_job()

    def _prepare_windows_job(self):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL,
                                          wintypes.LPCWSTR]
        kernel32.CreateEventW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                      ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        kernel32.Process32FirstW.argtypes = [wintypes.HANDLE,
                                             ctypes.POINTER(_ProcessEntry32W)]
        kernel32.Process32FirstW.restype = wintypes.BOOL
        kernel32.Process32NextW.argtypes = [wintypes.HANDLE,
                                            ctypes.POINTER(_ProcessEntry32W)]
        kernel32.Process32NextW.restype = wintypes.BOOL
        self._kernel32 = kernel32

        self._job = kernel32.CreateJobObjectW(None, None)
        if not self._job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
                self._job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits), ctypes.sizeof(limits)):
            error_code = ctypes.get_last_error()
            if error_code != _ERROR_INVALID_PARAMETER:
                error = ctypes.WinError(error_code)
                self.close()
                raise error
            limits.BasicLimitInformation.LimitFlags = 0
            if not kernel32.SetInformationJobObject(
                    self._job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits), ctypes.sizeof(limits)):
                error = ctypes.WinError(ctypes.get_last_error())
                self.close()
                raise error
            self.warning = (
                "当前 Windows 作业环境不支持句柄关闭时自动终止；"
                "训练正常取消和关窗仍会显式终止整个进程树")

        self._event_name = f"Local\\SimuSignalTraining-{uuid.uuid4().hex}"
        self._event = kernel32.CreateEventW(None, True, False, self._event_name)
        if not self._event:
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def environment(self):
        if self._event_name is None:
            return {}
        return {"SIMUSIGNAL_TRAINING_GATE_EVENT": self._event_name}

    def attach(self, pid):
        self.pid = int(pid)
        if os.name != "nt":
            return
        kernel32 = self._kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.SetEvent.argtypes = [wintypes.HANDLE]
        kernel32.SetEvent.restype = wintypes.BOOL
        process = kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_QUERY_LIMITED_INFORMATION,
            False, self.pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not kernel32.AssignProcessToJobObject(self._job, process):
                raise ctypes.WinError(ctypes.get_last_error())
            self._attached = True
            if not kernel32.SetEvent(self._event):
                raise ctypes.WinError(ctypes.get_last_error())
        except OSError:
            if self._attached:
                self.terminate()
            raise
        finally:
            kernel32.CloseHandle(process)

    def active_count(self, process=None):
        if os.name == "nt":
            self._refresh_descendants()
            if not self._attached:
                return int(process is not None and
                           process.state() != process.ProcessState.NotRunning)
            info = _BasicAccountingInformation()
            returned = wintypes.DWORD()
            self._kernel32.QueryInformationJobObject.argtypes = [
                wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD)]
            self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
            if not self._kernel32.QueryInformationJobObject(
                    self._job, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
                    ctypes.byref(info), ctypes.sizeof(info), ctypes.byref(returned)):
                raise ctypes.WinError(ctypes.get_last_error())
            active_external = 0
            for pid, tracked in tuple(self._descendants.items()):
                result = self._kernel32.WaitForSingleObject(tracked["handle"], 0)
                if result == _WAIT_OBJECT_0:
                    self._kernel32.CloseHandle(tracked["handle"])
                    del self._descendants[pid]
                elif result == _WAIT_TIMEOUT and not tracked["in_job"]:
                    active_external += 1
                elif result == _WAIT_FAILED:
                    raise ctypes.WinError(ctypes.get_last_error())
            return int(info.ActiveProcesses) + active_external
        if self.pid is None:
            return int(process is not None and
                       process.state() != process.ProcessState.NotRunning)
        try:
            os.killpg(self.pid, 0)
        except ProcessLookupError:
            return 0
        return 1

    def terminate(self, process=None):
        if os.name == "nt":
            self._refresh_descendants()
            if self._attached:
                self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
                self._kernel32.TerminateJobObject.restype = wintypes.BOOL
                if not self._kernel32.TerminateJobObject(self._job, 1):
                    error = ctypes.WinError(ctypes.get_last_error())
                else:
                    error = None
            elif process is not None:
                process.kill()
                error = None
            else:
                error = None
            for tracked in self._descendants.values():
                if tracked["in_job"]:
                    continue
                result = self._kernel32.WaitForSingleObject(tracked["handle"], 0)
                if result == _WAIT_TIMEOUT:
                    self._kernel32.TerminateProcess.argtypes = [
                        wintypes.HANDLE, wintypes.UINT]
                    self._kernel32.TerminateProcess.restype = wintypes.BOOL
                    if not self._kernel32.TerminateProcess(tracked["handle"], 1):
                        if error is None:
                            error = ctypes.WinError(ctypes.get_last_error())
                elif result != _WAIT_OBJECT_0 and error is None:
                    error = ctypes.WinError(ctypes.get_last_error())
            if error is not None:
                raise error
            return
        if self.pid is None:
            if process is not None:
                process.terminate()
            return
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def request_stop(self, process):
        if os.name == "nt":
            self._refresh_descendants()
            if process.state() != process.ProcessState.NotRunning:
                process.terminate()
            return
        if self.pid is None:
            process.terminate()
            return
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def close(self):
        if os.name == "nt":
            for tracked in self._descendants.values():
                self._kernel32.CloseHandle(tracked["handle"])
            self._descendants.clear()
        if self._event:
            self._kernel32.CloseHandle(self._event)
            self._event = None
        if self._job:
            self._kernel32.CloseHandle(self._job)
            self._job = None
        self._attached = False

    def _refresh_descendants(self):
        """Track descendants too; Windows hosts may not inherit nested job membership."""
        if os.name != "nt" or self.pid is None:
            return
        kernel32 = self._kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if snapshot == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            parents = {}
            entry = _ProcessEntry32W()
            entry.dwSize = ctypes.sizeof(entry)
            if not kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                raise ctypes.WinError(ctypes.get_last_error())
            while True:
                parents.setdefault(int(entry.th32ParentProcessID), []).append(
                    int(entry.th32ProcessID))
                entry.dwSize = ctypes.sizeof(entry)
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    error_code = ctypes.get_last_error()
                    if error_code != _ERROR_NO_MORE_FILES:
                        raise ctypes.WinError(error_code)
                    break
        finally:
            kernel32.CloseHandle(snapshot)

        descendants = set()
        pending = [self.pid]
        while pending:
            for pid in parents.get(pending.pop(), ()):
                if pid not in descendants and pid != self.pid:
                    descendants.add(pid)
                    pending.append(pid)

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        for pid in descendants.difference(self._descendants):
            handle = kernel32.OpenProcess(
                _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION | _SYNCHRONIZE,
                False, pid)
            if not handle:
                error_code = ctypes.get_last_error()
                if error_code == _ERROR_INVALID_PARAMETER:
                    continue
                raise ctypes.WinError(error_code)
            in_job = wintypes.BOOL()
            if not kernel32.IsProcessInJob(handle, self._job, ctypes.byref(in_job)):
                error = ctypes.WinError(ctypes.get_last_error())
                kernel32.CloseHandle(handle)
                raise error
            self._descendants[pid] = {"handle": handle, "in_job": bool(in_job.value)}
