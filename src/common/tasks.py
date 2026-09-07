"""Subprocess tasks with deadlines, cooperative parent cancellation and status files."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .storage import utc_now


class JobError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def worker_command(request_path, response_path, worker_module):
    if getattr(sys, "frozen", False):
        return [sys.executable, "worker", str(request_path), str(response_path)]
    return [sys.executable, "-m", worker_module, "worker", str(request_path), str(response_path)]


def run_job(request, *, worker_module, timeout=30.0, cancel=None):
    if not 0 < timeout <= 3600:
        raise ValueError("任务超时应在 (0,3600] 秒内")
    workspace_root = Path(request["workspace"]).expanduser().resolve()
    (workspace_root / "jobs").mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    folder = workspace_root / "jobs" / job_id
    folder.mkdir()
    request_path, response_path = folder / "request.json", folder / "response.json"
    write_json(request_path, {**request, "workspace": str(workspace_root)})
    status = {"job_id": job_id, "action": request["action"], "state": "running", "started": utc_now()}
    write_json(folder / "status.json", status)
    env = os.environ.copy()
    if not getattr(sys, "frozen", False):
        # Preserve source-checkout execution without depending on caller cwd.
        root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    process = None
    try:
        with (folder / "worker.log").open("wb") as log:
            process = subprocess.Popen(worker_command(request_path, response_path, worker_module),
                                       stdout=log, stderr=log, env=env)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if cancel is not None and cancel.is_set():
                    raise JobError("cancelled", "任务已取消")
                if time.monotonic() >= deadline:
                    raise JobError("timeout", "任务超过允许时间")
                if (folder / "worker.log").stat().st_size > 1024 * 1024:
                    raise JobError("log_limit", "工作进程日志超过 1 MiB，任务已终止")
                time.sleep(0.03)
            if process.returncode != 0:
                raise JobError("worker_crashed", f"工作进程异常退出：{process.returncode}")
        if not response_path.exists() or response_path.stat().st_size > 4 * 1024 * 1024:
            raise JobError("invalid_response", "工作进程结果缺失或过大")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise JobError("task_failed", response.get("error", "任务失败") if isinstance(response, dict) else "结果格式错误")
        status["state"] = "success"
        return response["result"]
    except BaseException as exc:
        status.update(state="cancelled" if isinstance(exc, JobError) and exc.code == "cancelled" else "failed",
                      error=str(exc), error_code=getattr(exc, "code", type(exc).__name__))
        raise
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        status["finished"] = utc_now()
        write_json(folder / "status.json", status)


def worker_main(request_path, response_path, execute):
    # Suppress core dump files for the trusted native-demo crash tests on Unix.
    if os.name == "posix":
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        response = {"ok": True, "result": execute(request)}
    except Exception as exc:
        response = {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    write_json(response_path, response)
    return 0
