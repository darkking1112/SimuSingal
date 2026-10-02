"""把算法结论“采纳为参数标注”（方案 §4.2）。

采纳 = 在**目标参考参数**上追加 ``source=algorithm`` 的新版本，并在资产所属集合
已有对应任务标注集时同步追加标签（``note`` 记录来源运行，对应设计里的
``source_run_id`` 口径）。全部为追加式：旧版本与旧标签不修改、不删除；重复采纳
会产生新版本（符合“保存即生效、可回看历史”的单人标注模型）。

匹配规则（检测/逐跳）：

* 已存在目标时按“中心频率距离 + 时间重叠”选最接近的一个（门限为两者带宽较大值
  的一半；时间重叠不低于 30%）；
* 匹配不到则新建目标：会话结果建 ``scope='session'``，逐跳结果挂到会话下
  ``scope='hop'``（``target_key`` 形如 ``s1.h3``）；
* 采纳只写测量得到的物理量（时间、频带、SNR、功率），名义值与调制样式从旧版本
  继承，避免覆盖用户已确认的信息。

分类结果（AMC / IQ）更新目标的 ``modulation``；类别名按任务字典记为 ``known``
或 ``out_of_taxonomy``（字典外的原始类名保留，不猜测）。
"""
import json

#: 采纳时从旧版本继承的字段（算法测不到的名义值、样式与上层属性）。
_CARRY_FIELDS = ("signal_type", "waveform_mode", "modulation", "nominal_center_hz",
                 "nominal_bandwidth_hz", "symbol_rate_baud", "hop_rate_hz", "is_hopping")

#: A09 类别 → 规范调制名；字典外的类别名按大写原样记录（不猜测语义）。
_CLASS_TO_MODULATION = {"fm": "FM", "ssb": "SSB", "ask2": "2ASK", "qpsk": "QPSK",
                        "qam16": "16QAM", "qam64": "64QAM"}


def _next_target_index(sessions):
    indexes = []
    for target in sessions:
        key = str(target.get("target_key") or "")
        if key.startswith("s") and "." not in key:
            try:
                indexes.append(int(key[1:]))
            except ValueError:
                continue
    return max(indexes, default=-1) + 1


def _next_hop_index(children):
    indexes = [int(target["hop_index"]) for target in children
               if target.get("hop_index") is not None]
    return max(indexes, default=-1) + 1


def _clamp_span(start_s, end_s, rate, count):
    """时间区间 → ``[start, end)`` 采样点（夹在资产范围内，至少 1 个采样点）。"""
    start = 0 if start_s is None else int(round(float(start_s) * rate))
    end = count if end_s is None else int(round(float(end_s) * rate))
    start = max(0, min(start, count - 1))
    end = max(start + 1, min(end, count))
    return start, end


def _overlap(first, second):
    if first is None or second is None:
        return 1.0  # 缺时间信息时不把候选排除
    low, high = max(first[0], second[0]), min(first[1], second[1])
    span = min(first[1] - first[0], second[1] - second[0])
    if span <= 0:
        return 0.0
    return max(0.0, (high - low) / span)


def _time_range(version, rate):
    start, end = version.get("sample_start"), version.get("sample_end")
    if start is None or end is None:
        return None
    return (start / rate, end / rate)


def _match_target(targets, *, center_hz, bandwidth_hz, time_range, rate,
                  min_overlap=0.3):
    """在候选目标中选“中心最近 + 时间重叠”最优者；没有合格候选返回 ``None``。"""
    best, best_score = None, None
    for target in targets:
        version = target.get("current") or {}
        if version.get("center_hz") is None:
            continue
        gate = max(0.5 * max(abs(version.get("bandwidth_hz") or 0.0),
                             abs(bandwidth_hz or 0.0)), 1.0)
        distance = abs(float(version["center_hz"]) - float(center_hz))
        if distance > gate:
            continue
        overlap = _overlap(_time_range(version, rate), time_range)
        if overlap < min_overlap:
            continue
        score = distance / gate + (1.0 - overlap)
        if best_score is None or score < best_score:
            best, best_score = target, score
    return best


def _carry_fields(previous, override=None):
    fields = {}
    if previous:
        for name in _CARRY_FIELDS:
            if previous.get(name) is not None:
                fields[name] = previous[name]
    if override:
        fields.update(override)
    return fields


def adopt_detections(workspace, asset_id, detections, run_id, *, note_label="检测"):
    """会话级检测结果 → 目标参考参数版本；返回统计与采纳的目标。"""
    asset = workspace.get_asset(asset_id)
    rate = float(asset["sample_rate"])
    count = int(asset["sample_count"])
    rows = workspace.list_targets(asset_id, with_current=True)
    sessions = [target for target in rows if target["scope"] != "hop"]
    created = updated = 0
    adopted = []
    for detection in detections or []:
        center = detection.get("center_hz")
        if center is None:
            continue
        bandwidth = detection.get("bandwidth_hz")
        f_low = detection.get("f_low_hz")
        f_high = detection.get("f_high_hz")
        if f_low is None or f_high is None:
            if bandwidth is None:
                continue
            f_low, f_high = float(center) - abs(bandwidth) / 2.0, float(center) + abs(bandwidth) / 2.0
        if f_high <= f_low:
            continue
        start, end = _clamp_span(detection.get("t_start_s"), detection.get("t_end_s"),
                                 rate, count)
        match = _match_target(sessions, center_hz=float(center),
                              bandwidth_hz=bandwidth, time_range=(start / rate, end / rate),
                              rate=rate)
        if match is None:
            match = workspace.add_target(asset_id, f"s{_next_target_index(sessions)}",
                                         "session", for_detection=1, for_amc=0)
            sessions.append(match)
            created += 1
        else:
            updated += 1
            sessions = [target for target in sessions if target["id"] != match["id"]]
        previous = match.get("current") or workspace.current_target_version(match["id"])
        workspace.append_target_version(
            match["id"], source="algorithm",
            note=f"采纳{note_label}结果 · 来源运行 {run_id}",
            sample_start=start, sample_end=end, f_low_hz=f_low, f_high_hz=f_high,
            snr_db=detection.get("snr_db"), snr_definition="inband_snr_v1",
            power_dbfs=detection.get("power_dbfs"),
            **_carry_fields(previous))
        match = workspace.get_target(match["id"])
        match["current"] = workspace.current_target_version(match["id"])
        sessions.append(match)
        adopted.append(match)
    labels, task_sets = _append_detection_labels(workspace, asset_id, adopted, run_id)
    return {"created_targets": created, "updated_targets": updated,
            "targets": adopted, "labels": labels, "task_sets": task_sets}


def adopt_hops(workspace, asset_id, hops, sessions, run_id):
    """逐跳结果 → 会话目标 + 逐跳目标（``scope='hop'``）的参考参数版本。"""
    asset = workspace.get_asset(asset_id)
    rate = float(asset["sample_rate"])
    count = int(asset["sample_count"])
    rows = workspace.list_targets(asset_id, with_current=True)
    session_targets = [target for target in rows if target["scope"] != "hop"]
    hops = [hop for hop in hops or [] if hop.get("center_hz") is not None]
    if not hops:
        raise ValueError("逐跳结果为空，没有可采纳的内容")
    session_entry = next((item for item in sessions or []
                          if item.get("center_hz") is not None), None)
    spans = [(hop.get("t_start_s"), hop.get("t_end_s")) for hop in hops]
    lows = [hop["f_low_hz"] for hop in hops
            if hop.get("f_low_hz") is not None]
    highs = [hop["f_high_hz"] for hop in hops
             if hop.get("f_high_hz") is not None]
    if session_entry is None and lows and highs:
        times = [(start, end) for start, end in spans
                 if start is not None and end is not None]
        session_entry = {"center_hz": (min(lows) + max(highs)) / 2.0,
                         "bandwidth_hz": max(highs) - min(lows)}
        if times:
            session_entry["t_start_s"] = min(start for start, _ in times)
            session_entry["t_end_s"] = max(end for _, end in times)
    start, end = _clamp_span(session_entry.get("t_start_s") if session_entry else None,
                             session_entry.get("t_end_s") if session_entry else None,
                             rate, count)
    match = None
    if session_entry is not None:
        match = _match_target(session_targets, center_hz=float(session_entry["center_hz"]),
                              bandwidth_hz=session_entry.get("bandwidth_hz"),
                              time_range=(start / rate, end / rate), rate=rate)
    created = updated = 0
    if match is None:
        match = workspace.add_target(asset_id, f"s{_next_target_index(session_targets)}",
                                     "session", for_detection=1, for_amc=0)
        session_targets.append(match)
        created += 1
    else:
        updated += 1
    adopted = []
    if session_entry is not None:
        previous = match.get("current") or workspace.current_target_version(match["id"])
        low = session_entry.get("f_low_hz")
        high = session_entry.get("f_high_hz")
        if low is None or high is None:
            center = float(session_entry["center_hz"])
            width = abs(float(session_entry.get("bandwidth_hz")
                              or ((max(highs) - min(lows)) if lows and highs else 0.0)))
            if width > 0:
                low, high = center - width / 2.0, center + width / 2.0
        if low is not None and high is not None and high > low:
            hop_rate = session_entry.get("hop_rate_hz") or next(
                (hop.get("hop_rate_hz") for hop in hops
                 if hop.get("hop_rate_hz") is not None), None)
            workspace.append_target_version(
                match["id"], source="algorithm",
                note=f"采纳逐跳结果的会话范围 · 来源运行 {run_id}",
                sample_start=start, sample_end=end, f_low_hz=low, f_high_hz=high,
                snr_db=session_entry.get("snr_db"), snr_definition="inband_snr_v1",
                power_dbfs=session_entry.get("power_dbfs"),
                **_carry_fields(previous, {"is_hopping": 1, "hop_rate_hz": hop_rate}))
            match = workspace.get_target(match["id"])
            match["current"] = workspace.current_target_version(match["id"])
            adopted.append(match)
    children = [target for target in workspace.list_targets(asset_id)
                if target["scope"] == "hop" and target["parent_target_id"] == match["id"]]
    for hop in hops:
        hop_start, hop_end = _clamp_span(hop.get("t_start_s"), hop.get("t_end_s"),
                                         rate, count)
        time_range = (hop_start / rate, hop_end / rate)
        child = _match_target(children, center_hz=float(hop["center_hz"]),
                              bandwidth_hz=hop.get("bandwidth_hz"),
                              time_range=time_range, rate=rate)
        if child is None:
            hop_index = hop.get("hop_index")
            if hop_index is None or any(target.get("hop_index") == hop_index
                                        for target in children):
                hop_index = _next_hop_index(children)
            child = workspace.add_target(asset_id, f"{match['target_key']}.h{hop_index}",
                                         "hop", parent_target_id=match["id"],
                                         hop_index=hop_index, for_detection=1, for_amc=0)
            created += 1
        else:
            updated += 1
        low, high = hop.get("f_low_hz"), hop.get("f_high_hz")
        if low is None or high is None:
            width = abs(float(hop.get("bandwidth_hz") or 0.0))
            if width <= 0:
                continue
            low, high = float(hop["center_hz"]) - width / 2.0, float(hop["center_hz"]) + width / 2.0
        previous = child.get("current") or workspace.current_target_version(child["id"])
        override = {"is_hopping": 1}
        if hop.get("hop_rate_hz") is not None:
            override["hop_rate_hz"] = hop.get("hop_rate_hz")
        workspace.append_target_version(
            child["id"], source="algorithm",
            note=f"采纳逐跳结果 · 来源运行 {run_id}",
            sample_start=hop_start, sample_end=hop_end, f_low_hz=low, f_high_hz=high,
            snr_db=hop.get("snr_db"), snr_definition="inband_snr_v1",
            power_dbfs=hop.get("power_dbfs"),
            **_carry_fields(previous, override))
        child = workspace.get_target(child["id"])
        child["current"] = workspace.current_target_version(child["id"])
        children.append(child)
        adopted.append(child)
    labels, task_sets = _append_detection_labels(workspace, asset_id, adopted, run_id)
    return {"created_targets": created, "updated_targets": updated,
            "targets": adopted, "labels": labels, "task_sets": task_sets}


def adopt_classification(workspace, asset_id, prediction, run_id, *, band=None):
    """AMC / IQ 分类结果 → 目标的 ``modulation``（没有目标时建整条记录目标）。"""
    label_id = str((prediction or {}).get("label") or "").strip()
    if not label_id:
        raise ValueError("识别结果没有类别标签，无法采纳")
    asset = workspace.get_asset(asset_id)
    rate = float(asset["sample_rate"])
    count = int(asset["sample_count"])
    rows = workspace.list_targets(asset_id, with_current=True)
    signals = [target for target in rows if target["scope"] != "hop"]
    target = None
    created = 0
    if band and band.get("center_hz") is not None:
        target = _match_target(signals, center_hz=float(band["center_hz"]),
                               bandwidth_hz=band.get("bandwidth_hz"),
                               time_range=None, rate=rate, min_overlap=0.0)
    if target is None and len(signals) == 1:
        target = signals[0]
    if target is None and not signals:
        target = workspace.add_target(asset_id, "s0", "whole_record",
                                      for_detection=1, for_amc=1)
        signals.append(target)
        created = 1
    if target is None:
        raise ValueError("资产含多个目标且无法按分析频带定位；请先采纳检测结果或调整分析频带")
    modulation = _CLASS_TO_MODULATION.get(label_id, label_id.upper())
    previous = target.get("current") or workspace.current_target_version(target["id"])
    fields = _carry_fields(previous, {"modulation": modulation})
    extra = {}
    if previous is None and band and band.get("center_hz") is not None \
            and band.get("bandwidth_hz"):
        width = abs(float(band["bandwidth_hz"]))
        extra = {"f_low_hz": float(band["center_hz"]) - width / 2.0,
                 "f_high_hz": float(band["center_hz"]) + width / 2.0}
    workspace.append_target_version(target["id"], source="algorithm",
                                    note=f"采纳识别结果（{label_id}）· 来源运行 {run_id}",
                                    snr_db=None, snr_definition=None, **fields, **extra)
    target = workspace.get_target(target["id"])
    target["current"] = workspace.current_target_version(target["id"])
    labels, task_sets = _append_amc_labels(workspace, asset_id, target["id"], label_id,
                                           run_id)
    return {"created_targets": created, "updated_targets": 1 - created,
            "targets": [target], "class_name": label_id, "modulation": modulation,
            "labels": labels, "task_sets": task_sets}


def _append_detection_labels(workspace, asset_id, targets, run_id):
    added, task_sets = 0, []
    for collection in workspace.collections_of_asset(asset_id):
        for task_set in workspace.list_task_sets(collection["id"], "detection"):
            semantics = task_set["label_semantics"] or "session_v1"
            for target in targets:
                if (semantics == "per_hop_v1") != (target["scope"] == "hop"):
                    continue
                workspace.append_detection_label(task_set["id"], target["id"],
                                                 source="algorithm",
                                                 note=f"来源运行 {run_id}")
                added += 1
                task_sets.append(task_set["name"])
    return added, sorted(set(task_sets))


def _append_amc_labels(workspace, asset_id, target_id, label_id, run_id):
    added, task_sets = 0, []
    for collection in workspace.collections_of_asset(asset_id):
        for task_set in workspace.list_task_sets(collection["id"], "amc"):
            taxonomy = workspace.get_taxonomy(task_set["taxonomy_id"])
            classes = json.loads(taxonomy["classes_json"])
            state = "known" if label_id in classes else "out_of_taxonomy"
            workspace.append_amc_label(task_set["id"], target_id, source="algorithm",
                                       class_state=state, class_name=label_id,
                                       note=f"来源运行 {run_id}")
            added += 1
            task_sets.append(task_set["name"])
    return added, sorted(set(task_sets))


def adopt_run_result(workspace, run_id):
    """按运行记录类型分派采纳；返回统一统计（供 GUI/CLI 展示）。

    各采纳函数自行写入目标版本与标签，本函数只负责分派与汇总。
    """
    run = workspace.get_run(run_id)
    kind = str(run.get("kind") or "")
    asset_id = run.get("asset_id") or run.get("source_id")
    if not asset_id:
        raise ValueError("该运行记录没有关联资产，无法采纳")
    if kind in ("detect", "ml_detect"):
        detections = (run.get("summary") or {}).get("detections") or []
        result = adopt_detections(workspace, asset_id, detections, run_id)
    elif kind in ("detect_hops", "ml_detect_hops"):
        result = adopt_hops(workspace, asset_id, run.get("hops") or [],
                            run.get("sessions") or [], run_id)
    elif kind in ("amc_classify", "amc_iq_classify"):
        result = adopt_classification(workspace, asset_id, run.get("prediction") or {},
                                      run_id, band=run.get("band"))
    else:
        raise ValueError(f"“{kind}”结果不支持采纳为参数标注")
    targets = result.pop("targets", [])
    return {
        "kind": "adopt_result", "run_id": run_id, "asset_id": asset_id,
        "source_kind": kind, "created_targets": result.get("created_targets", 0),
        "updated_targets": result.get("updated_targets", 0),
        "adopted_targets": len(targets), "labels": result.get("labels", 0),
        "task_sets": result.get("task_sets", []),
    }
