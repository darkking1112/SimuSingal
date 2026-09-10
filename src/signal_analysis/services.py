"""Application operations executed by the worker; shared by GUI and CLI."""

from pathlib import Path

from .core_api import MODE_NAMES, analyze, generate_iq, make_demo
from .dataio import read_samples, write_samples
from .sigmf_io import SIGMF_EXTENSIONS, read_sigmf
from .plugins import call_demo_plugin
from .storage import Workspace


def _generated_name(signals):
    styles = sorted({MODE_NAMES.get(str(signal.get("mode", "")), str(signal.get("mode", "")))
                     for signal in signals})
    return f"IQ 生成 · {' + '.join(styles)}"


def execute(request):
    workspace = Workspace(request["workspace"])
    action = request["action"]
    if action == "demo":
        rate = request.get("sample_rate", 48000.0)
        return workspace.add_samples(make_demo(rate, request.get("count", 8192)),
                                     rate, "数学双音演示", "generated:tones_v1")
    if action == "import":
        path = Path(request["path"])
        metadata = None
        if path.suffix.lower() in SIGMF_EXTENSIONS:
            samples, rate, metadata = read_sigmf(path)
            supplied_rate = request.get("sample_rate")
            if supplied_rate is not None and float(supplied_rate) != rate:
                raise ValueError("指定采样率与 SigMF 元数据不一致")
        else:
            if request.get("sample_rate") is None:
                raise ValueError("非 SigMF 格式必须指定采样率 --sample-rate")
            samples = read_samples(path, binary_dtype=request.get("binary_dtype"),
                                   endian=request.get("endian", "little"))
            rate = request["sample_rate"]
        return workspace.add_samples(samples, rate, path.name, str(path.resolve()),
                                     metadata={"sigmf": metadata} if metadata is not None else None)
    if action == "generate":
        rate = request["sample_rate"]
        duration = request.get("duration", 0.1)
        signals = request.get("signals", [])
        seed = request.get("seed", 0)
        samples, summary = generate_iq(rate, duration, signals, request.get("noise"), seed)
        name = request.get("name") or _generated_name(signals)
        mode = str(signals[0].get("mode", "noise")) if signals else "noise"
        asset = workspace.add_samples(samples, rate, name, f"generated:iq_{mode}_v1",
                                      metadata={"generation": summary})
        result = {"kind": "generate", "id": asset["id"], "name": asset["name"],
                  "sample_rate": asset["sample_rate"], "summary": summary,
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
