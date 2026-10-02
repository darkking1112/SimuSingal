"""导入链服务：文件解析、标注清单匹配、批量导入与初始标注（GUI/CLI 共用）。"""

import csv
from pathlib import Path

import numpy as np

from common.storage import utc_now

from ..data.io import read_samples
from ..data.sigmf import SIGMF_EXTENSIONS, read_sigmf, read_sigmf_metadata
from .progress import Reporter


def import_signal_name(file_name, modulation):
    """导入资产的自动命名（GUI 预览与服务落库共用同一条规则）。

    显式填写的名称优先（调用方自行传入）；本函数只算回退名：
    有调制 → ``调制 · 文件名主体前 5 字符``；无调制（或“未知”）→ 文件名。
    """
    text = str(file_name or "")
    mod = str(modulation or "").strip()
    if not mod or mod == "未知":
        return text
    stem = Path(text).stem or text
    return f"{mod} · {stem[:5]}"


def import_file(workspace, request):
    """单文件导入（旧版入口）：可选 ``name`` 覆盖资产名，缺省用文件名。"""
    path = Path(request["path"])
    samples, rate, metadata = _parse_import_file(path, request)
    name = _optional_text_value(request.get("name")) or path.name
    return workspace.add_samples(samples, rate, name, str(path.resolve()),
                                 metadata=metadata,
                                 capture_started_at=request.get("capture_started_at")
                                 or utc_now())


def import_inspect(request):
    paths = request.get("paths") or []
    if not paths:
        raise ValueError("未选择任何文件")
    if len(paths) > 2000:
        raise ValueError("单次识别最多 2000 个文件")
    reporter = Reporter(request.get("job_dir"))
    files = []
    cancelled = False
    for index, item in enumerate(paths):
        if reporter.cancelled():
            cancelled = True
            break
        reporter.emit(index, len(paths), f"正在识别 {Path(item).name or item}")
        files.append(_inspect_import_file(item))
    reporter.emit(len(files), len(paths), "已取消" if cancelled else "识别完成", force=True)
    return {"kind": "import_inspect", "files": files,
            "cancelled": cancelled, "unprocessed": len(paths) - len(files)}


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
    """按目标行建立目标与参考参数（导入仅支持 AMC 标注）。

    目标必须给出调制；``for_amc`` 恒为 1、``for_detection`` 恒为 0（检测标注
    不再经导入入口建立）。旧调用传入的频率/SNR 不写入参考参数，由调用方回报。
    """
    for index, row in enumerate(targets):
        if not row["modulation"]:
            raise ValueError("导入仅支持 AMC 标注：目标行必须给出调制")
        target = workspace.add_target(asset["id"], f"s{index}", row["scope"],
                                      for_detection=0, for_amc=1)
        workspace.append_target_version(
            target["id"], source="import", note=row["note"],
            sample_start=row["sample_start"], sample_end=row["sample_end"],
            modulation=row["modulation"])
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
                     "信号名称": "name", "name": "name", "signal_name": "name",
                     "粒度": "scope", "scope": "scope",
                     "起止单位": "unit", "单位": "unit", "unit": "unit",
                     "起始": "start", "start": "start",
                     "结束": "end", "end": "end",
                     "频率下限": "f_low", "频率下限hz": "f_low", "f_low": "f_low",
                     "f_low_hz": "f_low", "频率上限": "f_high", "频率上限hz": "f_high",
                     "f_high": "f_high", "f_high_hz": "f_high",
                     "调制": "modulation", "modulation": "modulation",
                     "snr": "snr", "snr_db": "snr", "备注": "note", "note": "note",
                     "采集时间": "capture", "capture": "capture",
                     "capture_started_at": "capture"}

#: 旧清单里已被弃用的列（导入仅支持 AMC 标注）：出现时在报告里提示已忽略。
_MANIFEST_IGNORED_KEYS = ("f_low", "f_high", "snr", "note")


def import_manifest(request):
    """解析标注清单 CSV：按文件名（或完整路径）匹配当前清单，未匹配行原样退回。

    只做结构校验（粒度/单位/数值）与匹配，不写任何资产；时间单位换算留给导入时
    按每个文件的采样率完成（``_normalized_target_rows``）。“信号名称”与“采集
    时间”为文件级：同一文件各行必须一致，导入时随资产写入。导入仅支持 AMC
    标注：目标行必须给出“调制”；旧列（频率下限/上限、SNR、备注）不再解析，
    出现时记入 ``ignored_columns`` 由调用方提示，不静默丢弃。
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
    raw_header = [str(item).strip() for item in lines[0]]
    header = [_MANIFEST_COLUMNS.get(item.lower(), "") for item in raw_header]
    if "file" not in header:
        raise ValueError("标注清单缺少“文件”列")
    index_of = {name: index for index, name in enumerate(header) if name}
    ignored_columns = [label for label, key in zip(raw_header, header)
                       if key in _MANIFEST_IGNORED_KEYS]
    paths = [Path(str(item)) for item in (request.get("paths") or [])]
    by_name = {}
    for item in paths:
        by_name.setdefault(item.name, []).append(str(item))
    ambiguous = {name for name, items in by_name.items() if len(items) > 1}
    known_paths = {str(item) for item in paths}
    files, captures, names, unmatched = {}, {}, {}, []

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
            name = cell(row, "name")
            if name and len(name) > 200:
                raise ValueError("信号名称最多 200 个字符")
            capture = cell(row, "capture")
            if capture and len(capture) > 64:
                raise ValueError("采集时间最长 64 个字符")
            has_target = any(cell(row, key) for key in ("scope", "start", "end",
                                                        "modulation"))
            if has_target:
                if not cell(row, "modulation"):
                    raise ValueError("导入仅支持 AMC 标注：目标行必须给出“调制”")
                unit = _target_unit_name(cell(row, "unit") or "samples")
                scope = _target_scope_name(cell(row, "scope"), f"清单第 {line_no} 行")
                target = {"scope": scope, "start": cell(row, "start"),
                          "end": cell(row, "end"), "start_unit": unit,
                          "modulation": cell(row, "modulation")}
                # 结构校验（成对/数值/大小关系）；单位换算与文件采样率绑定，导入时再做
                _normalized_target_rows([target], rate=None, sample_count=None,
                                        row_label=f"清单第 {line_no} 行")
                files.setdefault(matched, []).append(target)
            if capture:
                previous = captures.get(matched)
                if previous is not None and previous != capture:
                    raise ValueError(f"采集时间与同一文件的其他行不一致（已填 {previous}）")
                captures[matched] = capture
            if name:
                previous = names.get(matched)
                if previous is not None and previous != name:
                    raise ValueError(f"信号名称与同一文件的其他行不一致（已填 {previous}）")
                names[matched] = name
            if not has_target and not capture and not name:
                raise ValueError("行内没有可申报的内容（目标列与采集时间/信号名称都为空）")
        except ValueError as exc:
            unmatched.append({"line": line_no, "file": key, "error": str(exc)})
    return {"kind": "import_manifest", "path": str(path),
            "matched": sum(len(items) for items in files.values()),
            "files": files, "captures": captures, "names": names,
            "ignored_columns": ignored_columns, "unmatched": unmatched}


def import_files(workspace, request):
    """批量导入：逐文件可覆盖参数与 AMC 目标行，可写分片/加入集合与初始标注。

    请求优先用 ``files``（GUI 清单：``path/name/sample_rate/binary_dtype/endian/
    capture_started_at/targets``）；``paths`` + 公共参数仍作为回落，命令行与旧调用
    不受影响。``name`` 缺省按 :func:`import_signal_name` 自动命名（有调制时用
    “调制 · 文件名前 5 字符”）；目标只登记 AMC（调制必填、``for_detection`` 恒为
    0），行内若带旧检测字段（频率/SNR）则忽略并在结果 ``ignored`` 里回报。
    ``rf_center_hz``/``label`` 保留用于 CLI 兼容（页面不再提供入口）；
    ``capture_started_at`` 逐文件优先、其次请求级、缺省记本批导入时间。
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
    imported_at = utc_now()  # 未提供采集时间的文件统一记本批导入时间
    reporter = Reporter(request.get("job_dir"))
    stopped = False
    for index, entry in enumerate(entries):
        path = Path(str(entry.get("path") or ""))
        if reporter.cancelled():
            # 文件边界协作取消：下方照常封存已写分片，未处理项计数返回
            stopped = True
            break
        reporter.emit(index, len(entries), f"正在导入 {path.name or path}")
        try:
            if not path.name:
                raise ValueError("文件路径为空")
            label = _optional_text_value(entry.get("label"))
            if label and len(label) > 200:
                raise ValueError("备注最多 200 个字符")
            signal_name = _optional_text_value(entry.get("name"))
            if signal_name and len(signal_name) > 200:
                raise ValueError("信号名称最多 200 个字符")
            file_request = {
                "sample_rate": entry.get("sample_rate", request.get("sample_rate")),
                "binary_dtype": entry.get("binary_dtype") or request.get("binary_dtype"),
                "endian": entry.get("endian") or request.get("endian", "little"),
            }
            samples, rate, metadata = _parse_import_file(path, file_request)
            targets = _normalized_target_rows(entry.get("targets") or [], rate=rate,
                                              sample_count=int(samples.size))
            for row in targets:
                # 先校验再落盘：避免“资产已入库但目标非法”的半写状态
                if not row["modulation"]:
                    raise ValueError("导入仅支持 AMC 标注：目标行必须给出调制")
            modulation = next((row.get("modulation") for row in targets
                               if row.get("modulation")), None)
            asset_name = signal_name or import_signal_name(path.name, modulation)
            ignored = [title for field, title in
                       (("f_low_hz", "频率下限"), ("f_high_hz", "频率上限"),
                        ("snr_db", "SNR"))
                       if any(row.get(field) is not None for row in targets)]
            common = {
                "source_kind": "imported",
                "rf_center_hz": entry.get("rf_center_hz", request.get("rf_center_hz")),
                "capture_started_at": (entry.get("capture_started_at")
                                       or request.get("capture_started_at")
                                       or imported_at),
            }
            if writer is not None:
                asset = writer.append(samples, rate, asset_name, str(path.resolve()),
                                      metadata=metadata, **common)
            else:
                asset = workspace.add_samples(samples, rate, asset_name,
                                              str(path.resolve()), metadata=metadata,
                                              created_by=request.get("created_by"),
                                              **common)
            total_targets += _apply_import_targets(workspace, asset, targets)
            if label:
                workspace.set_label(asset["id"], label)
            if collection is not None:
                workspace.add_collection_member(collection["id"], asset["id"])
            imported_ids.append(asset["id"])
            results.append({"path": str(path), "name": asset_name, "ok": True,
                            "asset_id": asset["id"], "targets": len(targets),
                            "ignored": ignored})
        except (ValueError, OSError) as exc:
            results.append({"path": str(path), "name": path.name or f"第 {index + 1} 项",
                            "ok": False, "error": str(exc)})
    reporter.emit(len(results), len(entries), "已取消" if stopped else "导入完成", force=True)
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
            "targets_total": total_targets, "initial_labels": labels,
            "cancelled": stopped, "unprocessed": len(entries) - len(results)}


def _ensure_initial_labels(workspace, collection, asset_ids):
    """生成页的“初始标注”：无标注集则建默认检测/AMC 标注集，再按参考参数写标签。"""
    from ..data.datasets import bootstrap_labels_from_versions

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
