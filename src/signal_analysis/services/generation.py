"""生成与分析服务：生成/频谱分析、配方预览与保存、集合生成与 TorchSig 接入。"""

from ..core_api import analyze as core_analyze, generate_iq
from ..data.io import resolve_storage_format
from .imports import _ensure_initial_labels, _resolve_collection
from .truth import _generated_name, _generation_targets


def _storage_request(request):
    """资产格式与字节序：新键优先，兼容旧的 ``export.format``/``export.endian``。"""
    export = request.get("export") if isinstance(request.get("export"), dict) else {}
    return resolve_storage_format(request.get("storage_format") or export.get("format") or "npy",
                                  request.get("endian") or export.get("endian") or "little")


def generate(workspace, request):
    rate = request["sample_rate"]
    duration = request.get("duration", 0.1)
    signals = request.get("signals", [])
    seed = request.get("seed", 0)
    storage_format, endian = _storage_request(request)
    samples, summary = generate_iq(rate, duration, signals, request.get("noise"), seed)
    name = request.get("name") or _generated_name(signals)
    mode = str(signals[0].get("mode", "noise")) if signals else "noise"
    asset = workspace.add_samples(samples, rate, name, f"generated:iq_{mode}_v1",
                                  metadata={"generation": summary},
                                  storage_format=storage_format, endian=endian)
    # 生成产物自带参考参数（方案 §7.2，存储层按摘要一次写全）；
    # 集合的“初始标注”也从这里引导。
    sessions, hops = _generation_targets(workspace, asset)
    collection = _resolve_collection(workspace, request, source_kind="generated")
    labels = 0
    if collection is not None:
        workspace.add_collection_member(collection["id"], asset["id"])
        if request.get("initial_labels"):
            labels = _ensure_initial_labels(workspace, collection, [asset["id"]])
    return {"kind": "generate", "id": asset["id"], "name": asset["name"],
            "sample_rate": asset["sample_rate"], "summary": summary,
            "targets": {"sessions": sessions, "hops": hops},
            "collection_id": collection["id"] if collection else None,
            "collection_name": collection["name"] if collection else None,
            "initial_labels": labels,
            "storage_format": asset["storage_format"], "endian": asset["endian"],
            "asset_path": asset["path"]}


def analyze(workspace, request):
    asset, data = workspace.load_samples(request["asset_id"])
    summary, arrays = core_analyze(data, asset["sample_rate"], request.get("nfft", 256))
    return workspace.save_run("analysis", {"summary": summary, "asset_name": asset["name"]},
                              arrays, asset["id"])


def recipe_preview(workspace, request):
    from ..algorithms.generation.recipes import preview_recipe

    return {"kind": "recipe_preview",
            **preview_recipe(request["recipe"], samples=request.get("samples"))}


def recipe_save(workspace, request):
    row = workspace.save_recipe(request["name"], request.get("engine", "project"),
                                request["recipe"], created_by=request.get("created_by"))
    return {"kind": "recipe_saved", **row}


def generate_collection(workspace, request):
    from .collection_gen import generate_collection as run_generate_collection

    return run_generate_collection(
        workspace, request,
        lambda: _resolve_collection(workspace, request, source_kind="generated"))


def torchsig_import(workspace, request):
    from ..integrations.torchsig import import_action

    return import_action(
        workspace, request,
        lambda: _resolve_collection(workspace, request, source_kind="generated"))


def torchsig_probe(workspace, request):
    from ..integrations.torchsig import probe

    return probe(workspace, request)
