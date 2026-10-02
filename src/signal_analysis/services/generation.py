"""生成与分析服务：演示/生成/频谱分析、配方预览与保存、集合生成与 TorchSig 接入。"""

from ..core_api import analyze as core_analyze, generate_iq, make_demo
from ..data.io import write_samples
from .imports import _ensure_initial_labels, _resolve_collection
from .truth import _generated_name, _generation_targets


def demo(workspace, request):
    rate = request.get("sample_rate", 48000.0)
    return workspace.add_samples(make_demo(rate, request.get("count", 8192)),
                                 rate, "数学双音演示", "generated:tones_v1")


def generate(workspace, request):
    rate = request["sample_rate"]
    duration = request.get("duration", 0.1)
    signals = request.get("signals", [])
    seed = request.get("seed", 0)
    samples, summary = generate_iq(rate, duration, signals, request.get("noise"), seed)
    name = request.get("name") or _generated_name(signals)
    mode = str(signals[0].get("mode", "noise")) if signals else "noise"
    asset = workspace.add_samples(samples, rate, name, f"generated:iq_{mode}_v1",
                                  metadata={"generation": summary})
    # 生成产物自带参考参数（方案 §7.2，存储层按摘要一次写全）；
    # 集合的“初始标注”也从这里引导。
    sessions, hops = _generation_targets(workspace, asset)
    collection = _resolve_collection(workspace, request, source_kind="generated")
    labels = 0
    if collection is not None:
        workspace.add_collection_member(collection["id"], asset["id"])
        if request.get("initial_labels"):
            labels = _ensure_initial_labels(workspace, collection, [asset["id"]])
    result = {"kind": "generate", "id": asset["id"], "name": asset["name"],
              "sample_rate": asset["sample_rate"], "summary": summary,
              "targets": {"sessions": sessions, "hops": hops},
              "collection_id": collection["id"] if collection else None,
              "collection_name": collection["name"] if collection else None,
              "initial_labels": labels,
              "export_path": None, "export_format": None}
    export = request.get("export")
    if export:
        fmt = str(export.get("format", ""))
        if fmt:
            exports = workspace.root / "exports"
            exports.mkdir(exist_ok=True)
            extension = ".sigmf-meta" if fmt == "sigmf" else (".bin" if fmt in ("iq16", "iq32") else "." + fmt)
            path = write_samples(exports / f"{asset['id']}{extension}",
                                 samples, fmt, export.get("endian", "little"),
                                 sample_rate=rate, description=name, generation=summary)
            result["export_path"] = str(path)
            result["export_format"] = fmt
            if fmt == "sigmf":
                result["export_data_path"] = str(path.with_suffix(".sigmf-data"))
    return result


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
