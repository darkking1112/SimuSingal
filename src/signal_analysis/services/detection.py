"""检测与逐跳估计服务：传统与 AI 两条通路的结果编排（真值评分在 ``truth``）。"""

from ..algorithms.detection.symbol_rate import (SYMBOL_RATE_METHOD,
                                                estimate_symbol_rate)
from ..core_api import detect_hops as core_detect_hops, detect_signals as core_detect_signals
from .truth import _attach_hop_truth, _attach_truth


def _annotate_symbol_rate(detection, samples, rate):
    """给单个会话级检出补符号率字段；跳频会话不估，明确标不适用。"""
    if detection.get("hopping"):
        detection.update({
            "symbol_rate_baud": None,
            "symbol_rate_confidence": 0.0,
            "symbol_rate_method": SYMBOL_RATE_METHOD,
            "symbol_rate_applicable": False,
            "symbol_rate_reason": "hopping",
        })
        return
    low = detection.get("occupied_f_low_hz")
    high = detection.get("occupied_f_high_hz")
    bandwidth = (float(high) - float(low)
                 if low is not None and high is not None and high > low
                 else detection.get("bandwidth_hz"))
    detection.update(estimate_symbol_rate(
        samples, rate, detection.get("center_hz", 0.0), bandwidth,
        start_s=detection.get("t_start_s"), end_s=detection.get("t_end_s")))


def _annotate_symbol_rates(summary, samples, rate):
    """传统/AI 共用的后处理：逐目标估计符号率，并同步 AI 的传统基线。"""
    for detection in summary.get("detections") or []:
        if isinstance(detection, dict):
            _annotate_symbol_rate(detection, samples, rate)
    baseline = summary.get("baseline")
    if isinstance(baseline, dict):
        _annotate_symbol_rates(baseline, samples, rate)


def detect(workspace, request):
    asset, data = workspace.load_samples(request["asset_id"])
    summary, arrays = core_detect_signals(data, asset["sample_rate"], request.get("config"))
    _annotate_symbol_rates(summary, data, asset["sample_rate"])
    payload = {"summary": summary, "asset_name": asset["name"],
               "contract": summary["contract"], "algorithm": summary["algorithm"],
               "snr_definition": summary["snr_definition"],
               "symbol_rate_method": SYMBOL_RATE_METHOD}
    _attach_truth(payload, workspace, asset["id"], summary)
    return workspace.save_run("detect", payload, arrays, asset["id"])


def detect_hops(workspace, request):
    asset, data = workspace.load_samples(request["asset_id"])
    summary, arrays = core_detect_hops(data, asset["sample_rate"], request.get("config"),
                                       request.get("with_sessions", True))
    payload = {"summary": summary, "asset_name": asset["name"],
               "contract": summary["contract"], "algorithm": summary["algorithm"],
               "snr_definition": summary["snr_definition"],
               "resolvable": summary["resolvable"], "reason": summary["reason"],
               "hops": summary["hops"], "sessions": summary["sessions"]}
    _attach_hop_truth(payload, workspace, asset["id"], summary)
    return workspace.save_run("detect_hops", payload, arrays, asset["id"])


def ml_detect(workspace, request):
    from ..algorithms.detection.ai import ml_detect as run_ml_detect

    asset, data = workspace.load_samples(request["asset_id"])
    summary, arrays = run_ml_detect(data, asset["sample_rate"], request.get("config"),
                                    model=request.get("manifest"),
                                    threads=request.get("threads"),
                                    with_baseline=request.get("with_baseline", True))
    _annotate_symbol_rates(summary, data, asset["sample_rate"])
    payload = {"summary": summary, "asset_name": asset["name"],
               "contract": summary["contract"], "algorithm": summary["algorithm"],
               "snr_definition": summary["snr_definition"], "model": summary["model"],
               "symbol_rate_method": SYMBOL_RATE_METHOD}
    # 同一份数据上的传统基线：AI 与会话合并口径一致，可直接并排比较
    _attach_truth(payload, workspace, asset["id"], summary)
    return workspace.save_run("ml_detect", payload, arrays, asset["id"])


def ml_detect_hops(workspace, request):
    from ..algorithms.detection.ai import ml_detect_hops as run_ml_detect_hops

    asset, data = workspace.load_samples(request["asset_id"])
    summary, arrays = run_ml_detect_hops(data, asset["sample_rate"], request.get("config"),
                                         model=request.get("manifest"),
                                         threads=request.get("threads"),
                                         with_sessions=request.get("with_sessions", True),
                                         with_traditional=request.get("with_traditional", True))
    payload = {"summary": summary, "asset_name": asset["name"],
               "contract": summary["contract"], "algorithm": summary["algorithm"],
               "snr_definition": summary["snr_definition"],
               "resolvable": summary["resolvable"], "reason": summary["reason"],
               "hops": summary["hops"], "sessions": summary["sessions"],
               "model": summary["model"]}
    # 逐跳真值 + 会话基线评分 + 传统逐跳基线评分（三者粒度各不同，不混用）
    _attach_hop_truth(payload, workspace, asset["id"], summary)
    return workspace.save_run("ml_detect_hops", payload, arrays, asset["id"])
