"""Application operations executed by the worker; shared by GUI and CLI."""

from pathlib import Path

from .core_api import analyze, make_demo
from .dataio import read_samples
from .plugins import call_demo_plugin
from .storage import Workspace


def execute(request):
    workspace = Workspace(request["workspace"])
    action = request["action"]
    if action == "demo":
        rate = request.get("sample_rate", 48000.0)
        return workspace.add_samples(make_demo(rate, request.get("count", 8192)),
                                     rate, "数学双音演示", "generated:tones_v1")
    if action == "import":
        path = Path(request["path"])
        return workspace.add_samples(read_samples(path), request["sample_rate"],
                                     path.name, str(path.resolve()))
    if action == "analyze":
        asset, data = workspace.load_samples(request["asset_id"])
        summary, arrays = analyze(data, asset["sample_rate"], request.get("nfft", 256))
        return workspace.save_run("analysis", {"summary": summary, "asset_name": asset["name"]}, arrays, asset["id"])
    if action == "native":
        asset, data = workspace.load_samples(request["asset_id"])
        manifest, result = call_demo_plugin(request["manifest"], data)
        derived = workspace.add_samples(result, asset["sample_rate"],
                                        asset["name"] + " · 原生复制", f"parent:{asset['id']}")
        return workspace.save_run("native", {"plugin": manifest,
                                             "derived_asset_id": derived["id"]}, asset_id=asset["id"])
    raise ValueError(f"不支持的任务类型：{action}")
