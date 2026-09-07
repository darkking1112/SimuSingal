"""Bind common task machinery to this project's worker entry."""
from common.tasks import JobError, run_job as run_process, worker_main as run_worker
from .storage import Workspace

def run_job(request, timeout=30.0, cancel=None):
    Workspace(request["workspace"])
    return run_process(request, worker_module="signal_analysis", timeout=timeout, cancel=cancel)

def worker_main(request_path, response_path):
    from .services import execute
    return run_worker(request_path, response_path, execute)
