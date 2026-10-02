"""调制识别服务：特征模型（A09）与原始 IQ 模型两条通路。"""

from .truth import _attach_amc_truth, _attach_iq_truth


def amc_classify(workspace, request):
    from ..algorithms.amc.feature_model import amc_classify as run_amc_classify

    asset, data = workspace.load_samples(request["asset_id"])
    result = run_amc_classify(data, asset["sample_rate"], request.get("config"),
                              model=request.get("model"), threads=request.get("threads"))
    payload = {"summary": result, "asset_name": asset["name"],
               "contract": result["contract"], "algorithm": result["algorithm"],
               "feature_contract": result["feature_contract"],
               "snr_estimate_db": result["snr_estimate_db"],
               "prediction": result["prediction"], "model": result["model"],
               "band": result["band"], "pending": result["pending"]}
    _attach_amc_truth(payload, workspace, asset["id"])
    return workspace.save_run("amc_classify", payload, None, asset["id"])


def amc_iq_classify(workspace, request):
    from ..algorithms.amc.iq_model import amc_iq_classify as run_amc_iq_classify

    asset, data = workspace.load_samples(request["asset_id"])
    result = run_amc_iq_classify(data, asset["sample_rate"], request.get("config"),
                                 model=request.get("model"), threads=request.get("threads"))
    payload = {"summary": result, "asset_name": asset["name"],
               "contract": result["contract"], "algorithm": result["algorithm"],
               "classes": result["classes"], "class_set": result["class_set"],
               "labels": result["labels"], "waveform": result["waveform"],
               "snr_estimate_db": result["snr_estimate_db"],
               "prediction": result["prediction"], "model": result["model"],
               "timing": result["timing"], "pending": result["pending"]}
    _attach_iq_truth(payload, workspace, asset["id"])
    return workspace.save_run("amc_iq_classify", payload, None, asset["id"])
