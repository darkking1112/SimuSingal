"""Bind common task machinery to this project's worker entry."""
from common.tasks import JobError, run_job as run_process, worker_main as run_worker
from .data import Workspace

def run_job(request, timeout=30.0, cancel=None, progress=None, cancel_grace=0.0):
    Workspace(request["workspace"])
    return run_process(request, worker_module="signal_analysis", timeout=timeout, cancel=cancel,
                       progress=progress, cancel_grace=cancel_grace)

def worker_main(request_path, response_path):
    from .services import execute
    return run_worker(request_path, response_path, execute)
