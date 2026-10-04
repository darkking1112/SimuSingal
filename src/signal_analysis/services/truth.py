"""目标参考参数真值：会话/逐跳真值读取，以及检测/识别结果的真值与评分附加。"""

from ..core_api import MODE_NAMES
from ..evaluation import HOP_CONTRACT, evaluate_detections, hop_truth, signal_truth


def _generation_targets(workspace, asset):
    """回读生成器在存储层已登记的目标数：会话与逐跳（方案 §7.2 与 §12.3）。

    目标与参考参数由 ``Workspace._register_generation_targets`` 在 ``add_samples``
    写库时按生成摘要一次写全（真值行就是 ``evaluation.signal_truth`` 的行），
    这里只做只读汇总，供结果面板与测试核对，不再重复建目标。
    """
    targets = workspace.list_targets(asset["id"])
    sessions = sum(1 for target in targets if target["scope"] != "hop")
    hops = sum(1 for target in targets if target["scope"] == "hop")
    return sessions, hops


def _generated_name(signals):
    """按调制样式拼自动名；没有信号行时是“仅噪声”，别留下悬空的分隔符。"""
    styles = sorted({MODE_NAMES.get(str(signal.get("mode", "")), str(signal.get("mode", "")))
                     for signal in signals})
    return f"IQ 生成 · {' + '.join(styles)}" if styles else "IQ 生成 · 仅噪声"


#: 规范调制名 → A09 类别；跳频等复合样式没有单一调制名（先查调制、再查样式）。
_MODULATION_TO_CLASS = {"FM": "fm", "SSB": "ssb", "2ASK": "ask2", "QPSK": "qpsk",
                        "16QAM": "qam16", "64QAM": "qam64"}


def _signal_targets(workspace, asset_id):
    """资产内非逐跳目标（会话/整条/片段）及当前参数版本。"""
    return [target for target in workspace.list_targets(asset_id, with_current=True)
            if target["scope"] != "hop" and target["current"] is not None]


def _target_signal_truth(workspace, asset_id):
    """目标参考参数 → 与 :func:`signal_truth` 同形的会话级真值。

    比较口径**改读目标参考参数**（方案 §3.3、C6）：重新标注频带后，算法对比与
    真值评分跟随新版本；没有可用参数（未知占位目标、空目标）时返回空列表，
    由调用方决定是否退化为生成器摘要。
    """
    asset = workspace.get_asset(asset_id)
    rate = float(asset["sample_rate"])
    truth = []
    for target in _signal_targets(workspace, asset_id):
        version = target["current"]
        if version.get("f_low_hz") is None or version.get("f_high_hz") is None:
            continue
        start, end = version.get("sample_start"), version.get("sample_end")
        truth.append({
            "index": len(truth), "mode": version.get("waveform_mode") or "",
            "hopping": bool(version.get("is_hopping")),
            "nominal_offset_hz": version.get("nominal_center_hz"),
            "nominal_bandwidth_hz": version.get("nominal_bandwidth_hz"),
            "center_hz": version.get("center_hz"),
            "bandwidth_hz": version.get("bandwidth_hz"),
            "f_low_hz": version.get("f_low_hz"), "f_high_hz": version.get("f_high_hz"),
            "t_start_s": (start / rate) if start is not None else None,
            "t_end_s": (end / rate) if end is not None else None,
            "power_dbfs": version.get("power_dbfs"),
            "snr_inband_db": version.get("snr_db"),
            "session_id": len(truth), "visited_channels": None,
        })
    return truth


def _target_hop_truth(workspace, asset_id):
    """目标参考参数 → 与 :func:`hop_truth` 同形的逐跳真值。"""
    asset = workspace.get_asset(asset_id)
    rate = float(asset["sample_rate"])
    rows = workspace.list_targets(asset_id, with_current=True)
    session_order, hop_counts = {}, {}
    for target in rows:
        if target["scope"] == "hop":
            parent = target["parent_target_id"]
            hop_counts[parent] = hop_counts.get(parent, 0) + 1
        else:
            session_order[target["id"]] = len(session_order)
    truth = []
    for target in rows:
        if target["scope"] != "hop":
            continue
        version = target["current"]
        if version is None or version.get("f_low_hz") is None \
                or version.get("sample_start") is None or version.get("sample_end") is None:
            continue
        start, end = version["sample_start"], version["sample_end"]
        parent = target["parent_target_id"]
        truth.append({
            "index": len(truth), "mode": version.get("waveform_mode") or "",
            "session_index": session_order.get(parent, 0),
            "hop_index": target["hop_index"], "hop_count": hop_counts.get(parent, 1),
            "center_hz": version.get("center_hz"),
            "bandwidth_hz": version.get("bandwidth_hz"),
            "f_low_hz": version.get("f_low_hz"), "f_high_hz": version.get("f_high_hz"),
            "t_start_s": start / rate, "t_end_s": end / rate,
            "dwell_s": (end - start) / rate,
            "hop_rate_hz": version.get("hop_rate_hz"),
            "power_dbfs": version.get("power_dbfs"),
            "snr_inband_db": version.get("snr_db"),
        })
    return truth


def _class_from_version(version):
    """目标参考参数 → A09 类别：先查规范调制名，再查生成样式。"""
    from ..algorithms.amc.feature_model import mode_to_class

    mapped = _MODULATION_TO_CLASS.get(str(version.get("modulation") or "").upper())
    if mapped:
        return mapped
    return mode_to_class(str(version.get("waveform_mode") or "").lower())


#: 已知调制名/生成样式 → 数模家族；键为去掉大小写与分隔符后的文本。
_ANALOG_FAMILY = frozenset({"am", "amdsb", "dsb", "amssb", "ssb", "usb", "lsb", "fm", "pm"})
_DIGITAL_FAMILY = frozenset({"ook", "ask", "ask2", "2ask", "bpsk", "qpsk", "dqpsk", "psk",
                             "8psk", "fsk", "2fsk", "4fsk", "gfsk", "msk", "gmsk", "qam",
                             "qam16", "16qam", "qam32", "32qam", "qam64", "64qam",
                             "qam256", "256qam", "ofdm", "fhrc", "fhvideo"})


def modulation_family(text):
    """调制名或生成样式 → ``"analog"``/``"digital"``；未收录的文本返回 ``None``。"""
    key = "".join(ch for ch in str(text).casefold() if ch.isalnum())
    if key in _DIGITAL_FAMILY:
        return "digital"
    if key in _ANALOG_FAMILY:
        return "analog"
    return None


def _version_modulation(version):
    """目标版本的调制文本：规范调制名优先，其次生成样式。"""
    return (str(version.get("modulation") or "").strip()
            or str(version.get("waveform_mode") or "").strip())


def _signal_versions(workspace, asset_id):
    """参与星座图判定的信号目标当前版本：不含逐跳目标，跳过纯噪声行。"""
    versions = []
    for target in _signal_targets(workspace, asset_id):
        version = target["current"]
        noise_row = (str(version.get("waveform_mode") or "") == "noise"
                     and not str(version.get("modulation") or ""))
        if not noise_row:
            versions.append(version)
    return versions


def _version_band(version):
    """目标版本的频带口径：中心频率与占用带宽，缺省由频率上下限派生。"""
    center = version.get("nominal_center_hz")
    bandwidth = version.get("nominal_bandwidth_hz")
    low, high = version.get("f_low_hz"), version.get("f_high_hz")
    if low is not None and high is not None:
        if center is None:
            center = (float(low) + float(high)) / 2.0
        if bandwidth is None:
            bandwidth = float(high) - float(low)
    return center, bandwidth


def constellation_plan(workspace, asset):
    """左下角星座图的绘制计划：元数据里的调制样式与特征参数是否够画符号级星座图。

    返回 ``{"plotted", "reason", "family", "label", "sources", "targets",
    "symbol_rate_baud", "center_hz", "bandwidth_hz", "points", "timing_phase"}``。
    ``plotted`` 为真表示可按 ``center_hz``/``symbol_rate_baud``（可选 ``bandwidth_hz``）
    抽取符号级星座点；为假时 ``reason`` 给出原因码：``analog``（模拟信号）、
    ``unknown``（调制未知或含未标注信号）、``multiple``（记录含多个信号）、
    ``hopping``（跳频/复合样式）、``symbol_rate``（缺符号率）、``center``（缺中心
    频率）、``aliasing``（符号率高于采样率一半）。

    判定只读目标参考参数（生成元数据登记、识别采纳、人工标注追加的版本），不消费
    原始 IQ 的启发式判定，也不猜频偏与符号率。
    """
    rate = float(asset["sample_rate"])
    versions = _signal_versions(workspace, asset["id"])
    families = {modulation_family(_version_modulation(version)) for version in versions}
    labels, sources = [], []
    for version in versions:
        modulation = str(version.get("modulation") or "").strip()
        mode = str(version.get("waveform_mode") or "").strip()
        if modulation or mode:
            labels.append(modulation or MODE_NAMES.get(mode, mode))
            sources.append(str(version.get("source") or ""))
    family = families.pop() if len(families) == 1 and None not in families else None
    unique = sorted(set(labels))
    plan = {"plotted": False, "reason": "unknown", "family": family,
            "label": " + ".join(unique) if unique else None,
            "sources": sorted(source for source in sources if source),
            "targets": len(versions), "symbol_rate_baud": None, "center_hz": None,
            "bandwidth_hz": None, "points": 0, "timing_phase": None}
    if family == "analog":
        plan["reason"] = "analog"
        return plan
    if family is None:
        return plan
    if len(versions) != 1:
        plan["reason"] = "multiple"
        return plan
    version = versions[0]
    symbol_rate = version.get("symbol_rate_baud")
    center, bandwidth = _version_band(version)
    plan.update({"symbol_rate_baud": symbol_rate, "center_hz": center,
                 "bandwidth_hz": bandwidth})
    if version.get("is_hopping"):
        plan["reason"] = "hopping"
    elif symbol_rate is None or float(symbol_rate) <= 0.0:
        plan["reason"] = "symbol_rate"
    elif center is None:
        plan["reason"] = "center"
    elif rate / float(symbol_rate) < 2.0:
        plan["reason"] = "aliasing"
    else:
        plan["plotted"] = True
        plan["reason"] = None
    return plan


def _attach_truth(payload, workspace, asset_id, summary):
    """给检测结果补上目标参考参数真值与评分（AI 路径同时给出传统基线评分）。"""
    generation = workspace.get_metadata(asset_id).get("generation")
    truth = _target_signal_truth(workspace, asset_id) or signal_truth(generation)
    if not truth:
        return payload
    payload["truth"] = truth
    payload["metrics"] = evaluate_detections(truth, summary["detections"])
    baseline = summary.get("baseline")
    if isinstance(baseline, dict) and baseline.get("detections"):
        payload["baseline_metrics"] = evaluate_detections(truth, baseline["detections"])
    return payload


def _attach_hop_truth(payload, workspace, asset_id, summary):
    """给逐跳参数估计结果补上逐跳真值与评分（真值同样读目标参考参数）。

    逐跳真值来自 ``scope='hop'`` 的目标参数；没有逐跳目标时退化为生成器摘要，
    仍没有则显式标注“不适用”。会话级基线（同一帧上的 ``detect_signals``）与
    传统逐跳基线照常按各自口径评分，便于对照两种粒度。
    """
    metadata = workspace.get_metadata(asset_id)
    generation = metadata.get("generation")
    truth = _target_hop_truth(workspace, asset_id) or hop_truth(generation)
    if not truth:
        payload["truth"] = {"available": False,
                            "reason": "该资产不是跳频生成样式，逐跳真值不适用"}
        payload["metrics"] = None
    else:
        payload["truth"] = truth
        payload["metrics"] = evaluate_detections(truth, summary["hops"],
                                                  contract=HOP_CONTRACT)
    baseline = summary.get("baseline")
    if isinstance(baseline, dict) and baseline.get("detections"):
        session_truth = _target_signal_truth(workspace, asset_id) or signal_truth(generation)
        if session_truth:
            payload["baseline_metrics"] = evaluate_detections(session_truth,
                                                              baseline["detections"])
    # AI 逐跳路径还会带上同配置的传统逐跳结果（``summary["traditional"]``）：
    # 两条通路的物理量口径相同，因此这一行是真正可比的逐跳基线，而不是同义反复。
    traditional = summary.get("traditional")
    if truth and isinstance(traditional, dict) and traditional.get("hops"):
        payload["traditional_metrics"] = evaluate_detections(truth, traditional["hops"],
                                                              contract=HOP_CONTRACT)
    return payload


def _attach_amc_truth(payload, workspace, asset_id):
    """给调制识别结果附上目标参考参数真值。

    真值优先读目标的当前参数版本（重新标注后比较自动跟随）；只有单目标录制才
    与真值直接比对，其余情况显式标注为“不适用”（而不是静默跳过）。历史数据
    没有目标参数时退化为生成器摘要。
    """
    prediction = payload["prediction"]
    signals = _signal_targets(workspace, asset_id)
    if len(signals) > 1:
        payload["truth"] = {
            "available": False, "count": len(signals),
            "reason": f"数据含 {len(signals)} 个信号目标，单频带识别结果不与真值直接比对"}
        payload["truth_hit"] = None
        return payload
    if len(signals) == 1:
        version = signals[0]["current"]
        if version.get("waveform_mode") or version.get("modulation"):
            mode = version.get("waveform_mode") or ""
            mapped = _class_from_version(version)
            payload["truth"] = {
                "available": mapped is not None,
                "mode": mode,
                "class": mapped,
                "center_hz": version.get("center_hz"),
                "bandwidth_hz": version.get("bandwidth_hz"),
                "snr_inband_db": version.get("snr_db"),
                "reason": None if mapped is not None else
                          f"生成样式 {mode} 不在 A09 六类字典内，按“不适用”计",
            }
            payload["truth_hit"] = prediction["label"] == mapped if mapped is not None else None
            return payload
    # 退化路径：历史数据（没有目标参考参数）仍按生成器摘要比对
    from ..algorithms.amc.feature_model import mode_to_class

    truth = signal_truth(workspace.get_metadata(asset_id).get("generation"))
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


def _attach_iq_truth(payload, workspace, asset_id):
    """给原始 IQ 识别结果附上目标参考参数真值。

    IQ 分支的标签集合由模型清单声明（可以是 A09 六类，也可以是更宽的独立
    字典），所以比对的前提是"生成样式能映射到该模型的某个类别"。真值优先读
    目标参考参数；映射不到时显式标"不适用"，不把不可比的样本计入准确率。
    """
    classes = list(payload["summary"]["classes"])
    prediction = payload["prediction"]
    signals = _signal_targets(workspace, asset_id)
    if len(signals) > 1:
        payload["truth"] = {
            "available": False, "count": len(signals),
            "reason": f"数据含 {len(signals)} 个信号目标，单窗口识别结果不与真值直接比对"}
        payload["truth_hit"] = None
        return payload
    if len(signals) == 1:
        version = signals[0]["current"]
        if version.get("waveform_mode") or version.get("modulation"):
            mode = version.get("waveform_mode") or ""
            mapped = _class_from_version(version)
            usable = mapped is not None and mapped in classes
            if usable:
                reason = None
            elif mapped is None:
                reason = f"生成样式 {mode} 不在 A09 六类字典内，按“不适用”计"
            else:
                reason = (f"生成样式 {mode} 映射到 {mapped}，但该模型标签集合不含此类，"
                          "按“不适用”计")
            payload["truth"] = {
                "available": usable, "mode": mode, "class": mapped, "classes": classes,
                "center_hz": version.get("center_hz"),
                "bandwidth_hz": version.get("bandwidth_hz"),
                "snr_inband_db": version.get("snr_db"),
                "reason": reason,
            }
            payload["truth_hit"] = prediction["label"] == mapped if usable else None
            return payload
    # 退化路径：历史数据（没有目标参考参数）仍按生成器摘要比对
    from ..algorithms.amc.feature_model import mode_to_class

    truth = signal_truth(workspace.get_metadata(asset_id).get("generation"))
    if not truth:
        payload["truth"] = {"available": False,
                            "reason": "数据没有生成器真值（导入或原生插件产出），不计算识别正误"}
        payload["truth_hit"] = None
        return payload
    if len(truth) != 1:
        payload["truth"] = {
            "available": False, "count": len(truth),
            "reason": f"生成数据含 {len(truth)} 个信号，单窗口识别结果不与真值直接比对"}
        payload["truth_hit"] = None
        return payload
    entry = truth[0]
    mapped = mode_to_class(entry["mode"])
    usable = mapped is not None and mapped in classes
    if usable:
        reason = None
    elif mapped is None:
        reason = f"生成样式 {entry['mode']} 不在 A09 六类字典内，按“不适用”计"
    else:
        reason = (f"生成样式 {entry['mode']} 映射到 {mapped}，但该模型标签集合不含此类，"
                  "按“不适用”计")
    payload["truth"] = {
        "available": usable,
        "mode": entry["mode"],
        "class": mapped,
        "classes": classes,
        "center_hz": entry["center_hz"],
        "bandwidth_hz": entry["bandwidth_hz"],
        "snr_inband_db": entry["snr_inband_db"],
        "reason": reason,
    }
    payload["truth_hit"] = prediction["label"] == mapped if usable else None
    return payload
