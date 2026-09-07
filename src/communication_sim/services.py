"""Simulation application operations, independent of signal analysis."""
from .simulation import Scenario, simulate
from .storage import Workspace

def execute(request):
    if request["action"] != "simulate":
        raise ValueError(f"不支持的仿真任务：{request['action']}")
    workspace = Workspace(request["workspace"])
    return workspace.save_run("simulation", simulate(Scenario(**request.get("scenario", {}))))
