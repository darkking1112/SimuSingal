"""Application operations executed by the worker; shared by GUI and CLI."""

from pathlib import Path

from .core_api import MODE_NAMES, analyze, detect_signals, generate_iq, make_demo
from .dataio import read_samples, write_samples
from .evaluation import evaluate_detections, signal_truth
from .sigmf_io import SIGMF_EXTENSIONS, read_sigmf
from .plugins import call_demo_plugin
from .storage import Workspace


def _generated_name(signals):
    styles = sorted({MODE_NAMES.get(str(signal.get("mode", "")), str(signal.get("mode", "")))
                     for signal in signals})
    return f"IQ 生成 · {' + '.join(styles)}"


def _attach_truth(payload, workspace, asset_id, summary):
    """给检测结果补上生成器真值与评分（AI 路径同时给出传统基线评分）。"""
    truth = signal_truth(workspace.get_metadata(asset_id).get("generation"))
    if not truth:
        return payload
    payload["truth"] = truth
    payload["metrics"] = evaluate_detections(truth, summary["detections"])
    baseline = summary.get("baseline")
    if isinstance(baseline, dict) and baseline.get("detections"):
        payload["baseline_metrics"] = evaluate_detections(truth, baseline["detections"])
    return payload


def _attach_amc_truth(payload, workspace, asset_id):
    """给调制识别结果附上生成器真值。

    只有“生成器产出且仅含单个信号”时才能与真值直接比对；其余情况显式标注为
    “不适用”（而不是静默跳过），与 A09 的“不适用必须计数”一致。
    """
    from .ml import mode_to_class

    truth = signal_truth(workspace.get_metadata(asset_id).get("generation"))
    prediction = payload["prediction"]
    if not truth:
        payload["truth"] = {"available": False,
                            "reason": "数据没有生成器真值（导入或原生插件产出），不计算识别正误"}
        payload["truth_hit"] = None
        return payload
    if len(truth) != 1:
        payload["truth"] = {
            "available": False, "count": len(truth),
            "reason": f"生成数据含 {len(truth)} 个信号，单频带识别结果不与真值直接比对"}
        payload["truth_hit"] = None
        return payload
    entry = truth[0]
    truth_class = mode_to_class(entry["mode"])
    payload["truth"] = {
        "available": truth_class is not None,
        "mode": entry["mode"],
        "class": truth_class,
        "center_hz": entry["center_hz"],
        "bandwidth_hz": entry["bandwidth_hz"],
        "snr_inband_db": entry["snr_inband_db"],
        "reason": None if truth_class is not None else
                  f"生成样式 {entry['mode']} 不在 A09 六类字典内，按“不适用”计",
    }
    payload["truth_hit"] = (None if truth_class is None
                            else prediction["label"] == truth_class)
    return payload


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
    if action == "detect":
        asset, data = workspace.load_samples(request["asset_id"])
        summary, arrays = detect_signals(data, asset["sample_rate"], request.get("config"))
        payload = {"summary": summary, "asset_name": asset["name"],
                   "contract": summary["contract"], "algorithm": summary["algorithm"],
                   "snr_definition": summary["snr_definition"]}
        _attach_truth(payload, workspace, asset["id"], summary)
        return workspace.save_run("detect", payload, arrays, asset["id"])
    if action == "ml_detect":
        from .ml import ml_detect

        asset, data = workspace.load_samples(request["asset_id"])
        summary, arrays = ml_detect(data, asset["sample_rate"], request.get("config"),
                                    model=request.get("manifest"),
                                    threads=request.get("threads"),
                                    with_baseline=request.get("with_baseline", True))
        payload = {"summary": summary, "asset_name": asset["name"],
                   "contract": summary["contract"], "algorithm": summary["algorithm"],
                   "snr_definition": summary["snr_definition"], "model": summary["model"]}
        # 同一份数据上的传统基线：AI 与会话合并口径一致，可直接并排比较
        _attach_truth(payload, workspace, asset["id"], summary)
        return workspace.save_run("ml_detect", payload, arrays, asset["id"])
    if action == "amc_classify":
        from .ml import amc_classify

        asset, data = workspace.load_samples(request["asset_id"])
        result = amc_classify(data, asset["sample_rate"], request.get("config"),
                              model=request.get("model"), threads=request.get("threads"))
        payload = {"summary": result, "asset_name": asset["name"],
                   "contract": result["contract"], "algorithm": result["algorithm"],
                   "feature_contract": result["feature_contract"],
                   "snr_estimate_db": result["snr_estimate_db"],
                   "prediction": result["prediction"], "model": result["model"],
                   "band": result["band"], "pending": result["pending"]}
        _attach_amc_truth(payload, workspace, asset["id"])
        return workspace.save_run("amc_classify", payload, None, asset["id"])
    if action == "native":
        asset, data = workspace.load_samples(request["asset_id"])
        manifest, result = call_demo_plugin(request["manifest"], data)
        derived = workspace.add_samples(result, asset["sample_rate"],
                                        asset["name"] + " · 原生复制", f"parent:{asset['id']}")
        return workspace.save_run("native", {"plugin": manifest,
                                             "derived_asset_id": derived["id"]}, asset_id=asset["id"])
    raise ValueError(f"不支持的任务类型：{action}")
