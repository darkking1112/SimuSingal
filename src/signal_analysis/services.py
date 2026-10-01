"""Application operations executed by the worker; shared by GUI and CLI."""

import csv
from pathlib import Path

import numpy as np

from .core_api import MODE_NAMES, analyze, detect_hops, detect_signals, generate_iq, make_demo
from .dataio import read_samples, write_samples
from .evaluation import HOP_CONTRACT, evaluate_detections, hop_truth, signal_truth
from .maintenance import apply_cleanup, build_report
from .sigmf_io import SIGMF_EXTENSIONS, read_sigmf, read_sigmf_metadata
from .plugins import call_demo_plugin
from .storage import Workspace


def _parse_import_file(path, request):
    """按扩展名解析单个导入文件；返回 ``(samples, rate, metadata)``。"""
    path = Path(path)
    if path.suffix.lower() in SIGMF_EXTENSIONS:
        samples, rate, metadata = read_sigmf(path)
        supplied_rate = request.get("sample_rate")
        if supplied_rate is not None and float(supplied_rate) != rate:
            raise ValueError("指定采样率与 SigMF 元数据不一致")
        return samples, rate, {"sigmf": metadata}
    if request.get("sample_rate") is None:
        raise ValueError("非 SigMF 格式必须指定采样率 --sample-rate")
    samples = read_samples(path, binary_dtype=request.get("binary_dtype"),
                           endian=request.get("endian", "little"))
    return samples, request["sample_rate"], None


def _resolve_collection(workspace, request, *, source_kind):
    """按请求解析目标集合：给定 id 直接用；给定名称则新建或复用（同名的沿用）。"""
    collection_id = request.get("collection_id")
    if collection_id:
        return workspace.get_collection(collection_id)
    name = str(request.get("collection_name") or "").strip()
    if not name:
        return None
    existing = next((item for item in
                     workspace.list_collections(include_archived=True)
                     if item["name"] == name), None)
    if existing is not None:
        return existing
    return workspace.create_collection(name, source_kind=source_kind,
                                       created_by=request.get("created_by"))


#: 标注清单的起止单位（中文与英文）→ 内部名；缺省为采样点。
_UNIT_NAMES = {"": "samples", "samples": "samples", "sample": "samples", "采样点": "samples",
               "点": "samples", "s": "s", "sec": "s", "second": "s",
               "秒": "s", "ms": "ms", "msec": "ms", "millisecond": "ms", "毫秒": "ms"}

#: 导入页可申报的目标粒度；逐跳目标需要父会话与跳序号，不在导入路径内。
_IMPORT_SCOPES = ("whole_record", "session", "segment")


def _target_unit_name(value):
    text = str(value or "").strip().lower()
    if text in _UNIT_NAMES:
        return _UNIT_NAMES[text]
    raise ValueError(f"起止单位应为 采样点/秒/毫秒，收到 {value!r}")


def _target_scope_name(value, where):
    scope = str(value or "session").strip()
    if scope in _IMPORT_SCOPES:
        return scope
    if scope == "hop":
        raise ValueError(f"{where}：逐跳目标需要父会话与跳序号；请在检测页逐跳标注，"
                         "或对逐跳估计结果使用“采纳为参数标注”")
    raise ValueError(f"{where}粒度无效：{scope}")


def _normalized_target_rows(rows, *, rate, sample_count, prefix="目标", row_label=None):
    """逐文件目标行 → 采样点口径的参考参数（支持采样点/秒/毫秒三种起止单位）。

    与方案 §7.1/§12.3 一致：空行跳过；时间与频率都必须成对给出；调制为自由
    文本（A09 之外的规范名如 AM 也保留，类别映射在 AMC 侧记 ``out_of_taxonomy``）。
    ``rate=None`` 表示“仅结构校验”（标注清单解析阶段还不知道每个文件的采样率）：
    秒/毫秒只校验数值合法性，换算留给导入阶段按文件采样率完成。
    """
    cleaned = []
    for index, row in enumerate(rows or []):
        where = row_label or f"{prefix}第 {index + 1} 行"
        values = {key: row.get(key) for key in
                  ("scope", "sample_start", "sample_end", "start", "end", "start_unit",
                   "f_low_hz", "f_high_hz", "modulation", "snr_db", "note")}
        if all(value in (None, "") for value in values.values()):
            continue
        scope = _target_scope_name(values["scope"], where)
        unit = _target_unit_name(values["start_unit"])
        raw_start = (values["start"] if values["start"] not in (None, "")
                     else values["sample_start"])
        raw_end = (values["end"] if values["end"] not in (None, "")
                   else values["sample_end"])
        if (raw_start in (None, "")) != (raw_end in (None, "")):
            raise ValueError(f"{where}：时间范围必须同时给出起止")
        start = end = None
        for raw, name in ((raw_start, "起点"), (raw_end, "终点")):
            if raw in (None, ""):
                continue
            if unit == "samples":
                value = _optional_int(raw, f"{where}{name}")
            elif rate is None:
                _optional_float(raw, f"{where}{name}")  # 仅校验数值；换算在导入时做
                value = None
            else:
                seconds = _optional_float(raw, f"{where}{name}")
                value = int(round(seconds * (float(rate) if unit == "s"
                                             else float(rate) / 1000.0)))
            if name == "起点":
                start = value
            else:
                end = value
        if start is not None:
            if end <= start:
                raise ValueError(f"{where}：终点必须大于起点")
            if start < 0 or (sample_count is not None and end > int(sample_count)):
                raise ValueError(f"{where}：时间范围超出该文件（0～{int(sample_count):,} 采样点）")
        low = _optional_float(values["f_low_hz"], f"{where}频率下限")
        high = _optional_float(values["f_high_hz"], f"{where}频率上限")
        if (low is None) != (high is None):
            raise ValueError(f"{where}：频率范围必须同时给出上下限")
        if low is not None and high <= low:
            raise ValueError(f"{where}：频率上限必须大于下限")
        cleaned.append({"scope": scope, "sample_start": start, "sample_end": end,
                        "f_low_hz": low, "f_high_hz": high,
                        "modulation": _optional_text_value(values["modulation"]),
                        "snr_db": _optional_float(values["snr_db"], f"{where}SNR"),
                        "note": _optional_text_value(values["note"])})
    return cleaned


def _optional_text_value(value):
    """目标行文本列的可选值：缺列/空串/纯空白都归一为 None（不写字符串 "None"）。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value, name):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}应为整数") from exc


def _optional_float(value, name):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}应为数值") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name}必须为有限数值")
    return number


def _apply_import_targets(workspace, asset, targets):
    """按目标行建立目标与参考参数（资料不足 = 未知待标注，不建目标）。

    适用标记按方案 §12.3：``for_detection`` 只在给出频带时置 1；``for_amc``
    只在给出调制时置 1（字典外的规范名如 AM 同样置 1，类别映射在 AMC 侧记
    ``out_of_taxonomy`` 并保留原名）。
    """
    for index, row in enumerate(targets):
        snr = row.get("snr_db")
        target = workspace.add_target(
            asset["id"], f"s{index}", row["scope"],
            for_detection=1 if row["f_low_hz"] is not None else 0,
            for_amc=1 if row["modulation"] else 0)
        workspace.append_target_version(
            target["id"], source="import", note=row["note"],
            sample_start=row["sample_start"], sample_end=row["sample_end"],
            f_low_hz=row["f_low_hz"], f_high_hz=row["f_high_hz"],
            modulation=row["modulation"], snr_db=snr,
            snr_definition="inband_snr_v1" if snr is not None else None)
    return len(targets)


def _inspect_import_file(path):
    """只读文件头部/元数据，供导入清单自动识别（不解析全文）。"""
    path = Path(str(path)).expanduser()
    info = {"path": str(path), "name": path.name, "format": "unknown",
            "sample_rate": None, "sample_count": None, "size_bytes": None,
            "dtype": None, "complex": None, "error": None}
    try:
        if not path.is_file():
            raise ValueError("文件不存在")
        size = path.stat().st_size
        info["size_bytes"] = int(size)
        if size <= 0:
            raise ValueError("文件为空")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1:
                raise ValueError("NPY 应为 1 维 I/Q 采样")
            info.update(format="npy", dtype=str(array.dtype),
                        complex=bool(np.iscomplexobj(array)),
                        sample_count=int(array.shape[0]))
        elif suffix == ".csv":
            info["format"] = "csv"  # 无表头两列 I,Q；点数与类型在解析时确定
        elif suffix in (".bin", ".raw", ".iq"):
            info["format"] = "binary"  # 需要类型与字节序后才能算出点数
        elif suffix in SIGMF_EXTENSIONS:
            meta = read_sigmf_metadata(path)
            info.update(format="sigmf", sample_rate=float(meta["sample_rate"]),
                        sample_count=int(meta["sample_count"]), dtype=meta["datatype"])
        else:
            raise ValueError("不支持的扩展名（支持 .npy/.csv/.bin/.raw/.iq 与 SigMF 双文件）")
    except (ValueError, OSError) as exc:
        info["error"] = str(exc)
    return info


#: 标注清单 CSV 的列名（中文与英文）→ 内部名。
_MANIFEST_COLUMNS = {"文件": "file", "文件名": "file", "file": "file", "path": "file",
                     "粒度": "scope", "scope": "scope",
                     "起止单位": "unit", "单位": "unit", "unit": "unit",
                     "起始": "start", "start": "start",
                     "结束": "end", "end": "end",
                     "频率下限": "f_low", "频率下限hz": "f_low", "f_low": "f_low",
                     "f_low_hz": "f_low", "频率上限": "f_high", "频率上限hz": "f_high",
                     "f_high": "f_high", "f_high_hz": "f_high",
                     "调制": "modulation", "modulation": "modulation",
                     "snr": "snr", "snr_db": "snr", "备注": "note", "note": "note"}


def _import_manifest(request):
    """解析标注清单 CSV：按文件名（或完整路径）匹配当前清单，未匹配行原样退回。

    只做结构校验（粒度/单位/成对/数值）与匹配，不写任何资产；时间单位换算留给
    导入时按每个文件的采样率完成（``_normalized_target_rows``）。
    """
    path = Path(str(request.get("path") or ""))
    if not path.is_file():
        raise ValueError("标注清单文件不存在")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("标注清单必须为 UTF-8 编码（在 Excel 里另存为 CSV UTF-8）") from exc
    lines = list(csv.reader(text.splitlines()))
    if not lines:
        raise ValueError("标注清单为空")
    header = [_MANIFEST_COLUMNS.get(str(item).strip().lower(), "") for item in lines[0]]
    if "file" not in header:
        raise ValueError("标注清单缺少“文件”列")
    index_of = {name: index for index, name in enumerate(header) if name}
    paths = [Path(str(item)) for item in (request.get("paths") or [])]
    by_name = {}
    for item in paths:
        by_name.setdefault(item.name, []).append(str(item))
    ambiguous = {name for name, items in by_name.items() if len(items) > 1}
    known_paths = {str(item) for item in paths}
    files, unmatched = {}, []

    def cell(row, key):
        index = index_of.get(key)
        return row[index].strip() if index is not None and index < len(row) else ""

    for line_no, row in enumerate(lines[1:], start=2):
        if not any(str(item).strip() for item in row):
            continue
        key = cell(row, "file")
        try:
            if not key:
                raise ValueError("缺少文件名")
            if key in known_paths:
                matched = key
            elif key in by_name:
                if key in ambiguous:
                    raise ValueError(f"文件名 {key} 在当前清单里重复，请写完整路径")
                matched = by_name[key][0]
            else:
                raise ValueError("当前文件清单中没有该文件")
            unit = _target_unit_name(cell(row, "unit") or "samples")
            scope = _target_scope_name(cell(row, "scope"), f"清单第 {line_no} 行")
            target = {"scope": scope, "start": cell(row, "start"), "end": cell(row, "end"),
                      "start_unit": unit, "f_low_hz": cell(row, "f_low"),
                      "f_high_hz": cell(row, "f_high"), "modulation": cell(row, "modulation"),
                      "snr_db": cell(row, "snr"), "note": cell(row, "note")}
            # 结构校验（成对/数值/大小关系）；单位换算与文件采样率绑定，导入时再做
            _normalized_target_rows([target], rate=None, sample_count=None,
                                    row_label=f"清单第 {line_no} 行")
            files.setdefault(matched, []).append(target)
        except ValueError as exc:
            unmatched.append({"line": line_no, "file": key, "error": str(exc)})
    return {"kind": "import_manifest", "path": str(path),
            "matched": sum(len(items) for items in files.values()),
            "files": files, "unmatched": unmatched}


def _import_files(workspace, request):
    """批量导入：逐文件可覆盖参数与目标行，可写分片/加入集合/写备注与初始标注。

    请求优先用 ``files``（GUI 清单：``path/sample_rate/binary_dtype/endian/label/
    targets``）；``paths`` + 公共参数仍作为回落，命令行与旧调用不受影响。
    """
    entries = [dict(item) for item in (request.get("files") or [])]
    if not entries:
        entries = [{"path": item, "targets": request.get("targets")}
                   for item in (request.get("paths") or [])]
    if not entries:
        raise ValueError("未选择任何文件")
    if len(entries) > 500:
        raise ValueError("单次导入最多 500 个文件")
    collection = _resolve_collection(workspace, request, source_kind="imported")
    writer = (workspace.create_shard(name=request.get("shard_name")
                                     or f"导入批次（{len(entries)} 个文件）",
                                     created_by=request.get("created_by"))
              if request.get("batch_shard") else None)
    results, imported_ids, total_targets = [], [], 0
    for index, entry in enumerate(entries):
        path = Path(str(entry.get("path") or ""))
        try:
            if not path.name:
                raise ValueError("文件路径为空")
            label = _optional_text_value(entry.get("label"))
            if label and len(label) > 200:
                raise ValueError("备注最多 200 个字符")
            file_request = {
                "sample_rate": entry.get("sample_rate", request.get("sample_rate")),
                "binary_dtype": entry.get("binary_dtype") or request.get("binary_dtype"),
                "endian": entry.get("endian") or request.get("endian", "little"),
            }
            samples, rate, metadata = _parse_import_file(path, file_request)
            targets = _normalized_target_rows(entry.get("targets") or [], rate=rate,
                                              sample_count=int(samples.size))
            common = {
                "source_kind": "imported",
                "rf_center_hz": entry.get("rf_center_hz", request.get("rf_center_hz")),
                "capture_started_at": entry.get("capture_started_at",
                                                request.get("capture_started_at")),
            }
            if writer is not None:
                asset = writer.append(samples, rate, path.name, str(path.resolve()),
                                      metadata=metadata, **common)
            else:
                asset = workspace.add_samples(samples, rate, path.name,
                                              str(path.resolve()), metadata=metadata,
                                              created_by=request.get("created_by"),
                                              **common)
            total_targets += _apply_import_targets(workspace, asset, targets)
            if label:
                workspace.set_label(asset["id"], label)
            if collection is not None:
                workspace.add_collection_member(collection["id"], asset["id"])
            imported_ids.append(asset["id"])
            results.append({"path": str(path), "name": path.name, "ok": True,
                            "asset_id": asset["id"], "targets": len(targets)})
        except (ValueError, OSError) as exc:
            results.append({"path": str(path), "name": path.name or f"第 {index + 1} 项",
                            "ok": False, "error": str(exc)})
    shard = None
    if writer is not None:
        shard = writer.seal() if writer.count else None
        if writer.count == 0:
            writer.abort()
    labels = 0
    if collection is not None and request.get("initial_labels") and imported_ids:
        labels = _ensure_initial_labels(workspace, collection, imported_ids)
    return {"kind": "import_files", "results": results,
            "created": sum(1 for item in results if item["ok"]),
            "failed": sum(1 for item in results if not item["ok"]),
            "collection_id": collection["id"] if collection else None,
            "collection_name": collection["name"] if collection else None,
            "shard_id": shard["id"] if shard else None,
            "targets_total": total_targets, "initial_labels": labels}


def _ensure_initial_labels(workspace, collection, asset_ids):
    """生成页的“初始标注”：无标注集则建默认检测/AMC 标注集，再按参考参数写标签。"""
    from .datasets import bootstrap_labels_from_versions

    labels = 0
    task_sets = workspace.list_task_sets(collection["id"])
    detection = next((item for item in task_sets if item["task"] == "detection"), None)
    if detection is None:
        detection = workspace.create_task_set(collection["id"], "detection", name="检测标注")
    labels += bootstrap_labels_from_versions(workspace, detection["id"],
                                             asset_ids=asset_ids)["created"]
    has_amc = any(target.get("current") for target in
                  workspace.collection_targets(collection["id"], task="amc")
                  if target["asset_id"] in set(asset_ids))
    if has_amc:
        amc = next((item for item in task_sets if item["task"] == "amc"), None)
        if amc is None:
            amc = workspace.create_task_set(collection["id"], "amc", name="AMC 标注")
        labels += bootstrap_labels_from_versions(workspace, amc["id"],
                                                 asset_ids=asset_ids)["created"]
    return labels


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
    from .ml import mode_to_class

    mapped = _MODULATION_TO_CLASS.get(str(version.get("modulation") or "").upper())
    if mapped:
        return mapped
    return mode_to_class(str(version.get("waveform_mode") or "").lower())


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
    from .ml import mode_to_class

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
    from .ml import mode_to_class

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


def execute(request):
    workspace = Workspace(request["workspace"])
    action = request["action"]
    if action == "demo":
        rate = request.get("sample_rate", 48000.0)
        return workspace.add_samples(make_demo(rate, request.get("count", 8192)),
                                     rate, "数学双音演示", "generated:tones_v1")
    if action == "import":
        path = Path(request["path"])
        samples, rate, metadata = _parse_import_file(path, request)
        return workspace.add_samples(samples, rate, path.name, str(path.resolve()),
                                     metadata=metadata)
    if action == "import_inspect":
        paths = request.get("paths") or []
        if not paths:
            raise ValueError("未选择任何文件")
        if len(paths) > 2000:
            raise ValueError("单次识别最多 2000 个文件")
        return {"kind": "import_inspect",
                "files": [_inspect_import_file(item) for item in paths]}
    if action == "import_manifest":
        return _import_manifest(request)
    if action == "import_files":
        return _import_files(workspace, request)
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
    if action == "detect_hops":
        asset, data = workspace.load_samples(request["asset_id"])
        summary, arrays = detect_hops(data, asset["sample_rate"], request.get("config"),
                                      request.get("with_sessions", True))
        payload = {"summary": summary, "asset_name": asset["name"],
                   "contract": summary["contract"], "algorithm": summary["algorithm"],
                   "snr_definition": summary["snr_definition"],
                   "resolvable": summary["resolvable"], "reason": summary["reason"],
                   "hops": summary["hops"], "sessions": summary["sessions"]}
        _attach_hop_truth(payload, workspace, asset["id"], summary)
        return workspace.save_run("detect_hops", payload, arrays, asset["id"])
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
    if action == "ml_detect_hops":
        from .ml import ml_detect_hops

        asset, data = workspace.load_samples(request["asset_id"])
        summary, arrays = ml_detect_hops(data, asset["sample_rate"], request.get("config"),
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
    if action == "amc_iq_classify":
        from .ml import amc_iq_classify

        asset, data = workspace.load_samples(request["asset_id"])
        result = amc_iq_classify(data, asset["sample_rate"], request.get("config"),
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
    if action == "native":
        asset, data = workspace.load_samples(request["asset_id"])
        manifest, result = call_demo_plugin(request["manifest"], data)
        derived = workspace.add_samples(result, asset["sample_rate"],
                                        asset["name"] + " · 原生复制", f"parent:{asset['id']}")
        return workspace.save_run("native", {"plugin": manifest,
                                             "derived_asset_id": derived["id"]}, asset_id=asset["id"])
    if action == "storage_report":
        # 只读盘点：刻意不写入运行记录，否则报告本身会撑大存储。
        return build_report(workspace, extra_dirs=request.get("extra_dirs"),
                            job_retention_days=request.get("job_retention_days"))
    if action == "storage_cleanup":
        targets = request.get("targets")
        if not isinstance(targets, list) or not targets:
            raise ValueError("未指定要清理的条目")
        if len(targets) > 5000:
            raise ValueError("单次清理条目过多，请分批执行")
        return apply_cleanup(workspace, targets, extra_dirs=request.get("extra_dirs"),
                             job_retention_days=request.get("job_retention_days"))
    if action == "recipe_preview":
        from .recipes import preview_recipe

        return {"kind": "recipe_preview",
                **preview_recipe(request["recipe"], samples=request.get("samples"))}
    if action == "recipe_save":
        row = workspace.save_recipe(request["name"], request.get("engine", "project"),
                                    request["recipe"], created_by=request.get("created_by"))
        return {"kind": "recipe_saved", **row}
    if action == "dataset_build":
        from .datasets import build_dataset_version

        return build_dataset_version(
            workspace, request["task_set_id"], fractions=request.get("fractions"),
            seed=int(request.get("seed", 0)), holdout=request.get("holdout"),
            preprocessing=request.get("preprocessing"))
    if action == "dataset_verify":
        from .datasets import verify_dataset_version

        version = workspace.get_dataset_version(request["dataset_version_id"])
        if version["status"] != "ready":
            raise ValueError(f"数据版本状态为 {version['status']}，尚不可用于训练")
        return {"kind": "dataset_verify", "dataset_version_id": version["id"],
                "samples": verify_dataset_version(workspace, version),
                "task_set_id": version["task_set_id"]}
    if action == "dataset_bootstrap_labels":
        from .datasets import bootstrap_labels_from_versions

        return bootstrap_labels_from_versions(
            workspace, request["task_set_id"], asset_ids=request.get("asset_ids"),
            source=request.get("source", "generator"))
    if action == "adopt_result":
        from .adoption import adopt_run_result

        return adopt_run_result(workspace, request["run_id"])
    if action == "generate_collection":
        from .collection_gen import generate_collection

        return generate_collection(
            workspace, request,
            lambda: _resolve_collection(workspace, request, source_kind="generated"))
    if action == "torchsig_import":
        from .torchsig_support import import_action

        return import_action(
            workspace, request,
            lambda: _resolve_collection(workspace, request, source_kind="generated"))
    if action == "torchsig_probe":
        from .torchsig_support import probe

        return probe(workspace, request)
    if action == "export_training_data":
        from .training_export import export_training_data

        return export_training_data(workspace, request)
    if action == "migrate_legacy":
        from .maintenance import register_legacy_experiments, scan_legacy_datasets

        datasets = scan_legacy_datasets(workspace, request.get("folders"))
        experiments = register_legacy_experiments(workspace, request.get("experiments"))
        return {"kind": "legacy_migration", "datasets": datasets, "experiments": experiments}
    raise ValueError(f"不支持的任务类型：{action}")
