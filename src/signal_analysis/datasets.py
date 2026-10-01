"""数据版本构建（方案文档 §8）：固定成员、标签版本、划分与预处理的不可变清单。

流程：

1. 选定**任务标注集**，取集合内参与该任务的全部目标及其**当前（最新）标签版本**；
2. 检测任务只纳入 ``coverage=complete`` 的录制：目标标签缺一不可，否则整条
   录制跳过（在卡片里记录原因，避免把未标注信号当负样本）；
3. 生成清单 ``manifest.jsonl``（每行一个训练样本），先写临时文件、校验后
   原子换名；失败不留下半成品，也不会写入 ``dataset_versions`` 行；
4. 划分按 ``origin_group_id`` **整组**分配（同一次采集/同一生成场景不跨
   划分），``holdout`` 指定参数范围的组只进指定划分；随后写入
   ``dataset_versions``（``status='ready'`` 之后才允许训练）。

清单行字段（``dataset_manifest_v1``）::

    {"sample_id": "...", "task": "detection", "asset_id": "...",
     "asset_sha256": "...", "origin_group_id": "...", "split": "train",
     "target_id": "...", "target_key": "s0.h3", "scope": "hop",
     "label_revision_id": "...", "label_revision_no": 1, "class_name": "emitter",
     "include": true, "negative": false,
     "extraction": {"sample_start": 0, "sample_end": 8192, "f_low_hz": -1e5,
                    "f_high_hz": 1e5}}

AMC 行额外携带 ``amc`` 字段（``class_state`` 与窗口范围）。
"""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np

MANIFEST_CONTRACT = "dataset_manifest_v1"
SPLITS = ("train", "val", "test")
DEFAULT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

#: 规范调制名（目标参数里的 ``modulation``）与 A09 类别互转；``mode_to_class`` 只认生成器样式名，
#: 对 2ASK/16QAM/64QAM 不成立，所以这里单独维护。
MODULATION_TO_CLASS = {"FM": "fm", "SSB": "ssb", "2ASK": "ask2", "QPSK": "qpsk",
                       "16QAM": "qam16", "64QAM": "qam64"}
CLASS_TO_MODULATION = {value: key for key, value in MODULATION_TO_CLASS.items()}


def detection_label_applies(semantics, target):
    """该目标在给定检测标签粒度下是否需要标签。

    * ``session_v1``：一个会话一个框，逐跳子目标不标；
    * ``per_hop_v1``：跳频会话改为一跳一个框（跳频父会话不标），非跳频会话仍然一个框。
    """
    if target["scope"] == "hop":
        return semantics == "per_hop_v1"
    if semantics == "per_hop_v1":
        return not (target.get("current") or {}).get("is_hopping")
    return True


def _validate_fractions(fractions):
    if fractions is None:
        return dict(DEFAULT_FRACTIONS)
    if not isinstance(fractions, dict) or not fractions:
        raise ValueError("划分比例必须是非空对象")
    kept = {}
    for key, value in fractions.items():
        if key not in SPLITS:
            raise ValueError(f"未知划分：{key}")
        number = float(value)
        if not np.isfinite(number) or number < 0:
            raise ValueError("划分比例必须为非负有限数")
        kept[key] = number
    if not any(value > 0 for value in kept.values()):
        raise ValueError("划分比例之和必须大于 0")
    return kept


def _version_axis(version, axis):
    """holdout 轴取值：优先按字段名取当前参数版本；``abs:`` 前缀取绝对值。"""
    name = axis
    absolute = False
    if name.startswith("abs:"):
        absolute = True
        name = name[4:]
    if version is None:
        return None
    if name not in version:
        raise ValueError(f"holdout 轴不存在：{axis}")
    value = version.get(name)
    if value is None:
        return None
    number = float(value)
    return abs(number) if absolute else number


def _forced_split(targets_versions, holdout):
    """按 holdout 规则求组的强制划分；没有命中返回 ``None``。"""
    for rule in holdout or []:
        low, high = float(rule["range"][0]), float(rule["range"][1])
        for version in targets_versions:
            value = _version_axis(version, str(rule["axis"]))
            if value is not None and low <= value <= high:
                return str(rule["split"])
    return None


def build_rows(workspace, task_set_id):
    """收集数据版本候选行与排除统计；不写任何文件。"""
    task_set = workspace.get_task_set(task_set_id)
    task = task_set["task"]
    collection_id = task_set["collection_id"]
    targets = workspace.collection_targets(collection_id, task=task, with_current=True)
    if task == "detection":
        # 只有适用于该标签粒度的目标才要求标签（跳频父会话在逐跳粒度下不标，反之亦然）
        semantics = task_set["label_semantics"] or "session_v1"
        targets = [target for target in targets
                   if detection_label_applies(semantics, target)]
    assets = {}
    total = workspace.count_assets(collection_id=collection_id)
    for offset in range(0, total, 500):
        for asset in workspace.list_assets(limit=500, offset=offset,
                                           collection_id=collection_id):
            assets[asset["id"]] = asset
    labels = {row["target_id"]: row
              for row in workspace.current_labels(task, task_set_id, include_all=True)}
    coverage_rows = {row["asset_id"]: row for row in workspace.list_asset_coverage(task_set_id)}
    rows, excluded = [], []
    missing_versions = 0
    by_asset = {}
    for target in targets:
        by_asset.setdefault(target["asset_id"], []).append(target)

    def target_row(target, label):
        asset = assets[target["asset_id"]]
        row = {
            "task": task, "asset_id": asset["id"], "asset_sha256": asset["sha256"],
            "origin_group_id": asset.get("origin_group_id") or asset["id"],
            "target_id": target["id"], "target_key": target["target_key"],
            "scope": target["scope"], "target_version_id": target["current"]["id"],
            "label_revision_id": label["id"],
            "label_revision_no": int(label["revision_no"]),
            "class_name": label.get("class_name"), "include": True, "negative": False,
            "extraction": _extraction(target["current"], label),
            "_version": target["current"],
        }
        if task == "amc":
            row["amc"] = {"class_state": label.get("class_state"),
                          "window_start": label.get("window_start"),
                          "window_end": label.get("window_end"),
                          "analysis_center_hz": label.get("analysis_center_hz"),
                          "analysis_bandwidth_hz": label.get("analysis_bandwidth_hz")}
        return row

    def negative_row(asset):
        return {"task": task, "asset_id": asset["id"], "asset_sha256": asset["sha256"],
                "origin_group_id": asset.get("origin_group_id") or asset["id"],
                "target_id": None, "target_key": None, "scope": None,
                "target_version_id": None, "label_revision_id": None,
                "label_revision_no": None, "class_name": None, "include": False,
                "negative": True, "_version": None,
                "extraction": {"sample_start": 0,
                               "sample_end": int(asset["sample_count"])}}

    if task == "detection":
        for asset_id, asset_targets in by_asset.items():
            asset = assets.get(asset_id)
            if asset is None or asset.get("archived_at"):
                continue
            missing_versions += sum(1 for target in asset_targets
                                    if target.get("current") is None)
            coverage = coverage_rows.get(asset_id)
            coverage_value = coverage["coverage"] if coverage else "unknown"
            if coverage_value != "complete":
                excluded.append({"asset_id": asset_id, "name": asset["name"],
                                 "reason": f"覆盖度 {coverage_value}（仅纳入 complete）"})
                continue
            unlabeled = [target for target in asset_targets
                         if target["id"] not in labels or target.get("current") is None]
            if unlabeled:
                excluded.append({"asset_id": asset_id, "name": asset["name"],
                                 "reason": f"{len(unlabeled)} 个目标尚未标注或缺少参数版本"})
                continue
            included = [row for row in
                        (target_row(target, labels[target["id"]]) for target in asset_targets
                         if int(labels[target["id"]].get("include", 1)) == 1)]
            rows.extend(included)
            if not included:
                rows.append(negative_row(asset))
        for asset_id, coverage in coverage_rows.items():
            if coverage["coverage"] != "complete" or asset_id in by_asset:
                continue
            asset = assets.get(asset_id)
            if asset is not None and not asset.get("archived_at"):
                rows.append(negative_row(asset))
    else:
        for target in targets:
            asset = assets.get(target["asset_id"])
            if asset is None or asset.get("archived_at") or target.get("current") is None:
                missing_versions += 1
                continue
            label = labels.get(target["id"])
            if label is None or label.get("class_state") != "known":
                continue  # 未标注 / 未知 / 字典外不进入训练（统计另行报告）
            rows.append(target_row(target, label))
    stats = {"excluded": excluded, "excluded_count": len(excluded),
             "targets": {"total": len(targets), "labeled": len(labels),
                         "missing_versions": missing_versions}}
    return rows, stats


def _extraction(version, label):
    return {
        "sample_start": version.get("sample_start"),
        "sample_end": version.get("sample_end"),
        "f_low_hz": version.get("f_low_hz"),
        "f_high_hz": version.get("f_high_hz"),
        "window_start": label.get("window_start") if label else None,
        "window_end": label.get("window_end") if label else None,
    }


def assign_splits(rows, *, fractions=None, seed=0, holdout=None):
    """按 ``origin_group_id`` 整组分配 ``split``；返回 ``(rows, summary)``。

    每组先检查留出轴（命中则整组进入指定划分）；其余组按样本数缺口贪心分配，
    保证训练/验证/测试的样本比例接近目标，且同组样本永不跨划分。
    """
    fractions = _validate_fractions(fractions)
    groups = {}
    for row in rows:
        groups.setdefault(row["origin_group_id"], []).append(row)
    rng = np.random.default_rng(int(seed))
    order = sorted(groups)
    order = [order[index] for index in rng.permutation(len(order))]
    total = sum(len(group_rows) for group_rows in groups.values())
    targets = {name: total * share for name, share in fractions.items()}
    assigned = {name: 0 for name in fractions}
    forced_counts = {}
    pending = []
    for group_id in order:
        versions = [row["_version"] for row in groups[group_id] if row.get("_version")]
        forced = _forced_split(versions, holdout)
        if forced is not None and forced in fractions:
            forced_counts[forced] = forced_counts.get(forced, 0) + 1
            for row in groups[group_id]:
                row["split"] = forced
            assigned[forced] += len(groups[group_id])
        else:
            pending.append(group_id)
    for group_id in pending:
        deficits = {name: targets[name] - assigned[name] for name in fractions}
        split = max(deficits, key=lambda name: (deficits[name], _priority(name)))
        for row in groups[group_id]:
            row["split"] = split
        assigned[split] += len(groups[group_id])
    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    warnings = []
    for name in fractions:
        if fractions[name] > 0 and counts.get(name, 0) == 0 and len(groups) > 2:
            warnings.append(f"划分 {name} 没有样本（组数不足或留出轴占满）")
    summary = {"fractions": fractions, "seed": int(seed),
               "groups": len(groups), "counts": counts, "targets": total,
               "forced_groups": forced_counts, "warnings": warnings}
    return rows, summary


def _priority(name):
    return {"train": 2, "val": 1, "test": 0}.get(name, 0)


def write_manifest(rows, directory):
    """流式写入清单；返回 ``(path, sha256, counts)``。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.jsonl"
    temporary = directory / "manifest.jsonl.tmp"
    digest = hashlib.sha256()
    counts = {}
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            clean["sample_id"] = _sample_id(clean)
            text = json.dumps(clean, ensure_ascii=False, sort_keys=True, allow_nan=False)
            payload = (text + "\n").encode("utf-8")
            stream.write(text + "\n")
            digest.update(payload)
            counts[clean["split"]] = counts.get(clean["split"], 0) + 1
    temporary.replace(path)
    return path, digest.hexdigest(), counts


def _sample_id(row):
    target = row.get("target_id") or "negative"
    return f"{row['asset_id']}:{target}"


def build_dataset_version(workspace, task_set_id, *, fractions=None, seed=0, holdout=None,
                          preprocessing=None):
    """构建不可变数据版本；先写清单再入库，失败不留下半成品。"""
    task_set = workspace.get_task_set(task_set_id)
    rows, stats = build_rows(workspace, task_set_id)
    if not rows:
        raise ValueError("没有可纳入的数据：请先完成标注并确认录制覆盖度")
    rows, split_summary = assign_splits(rows, fractions=fractions, seed=seed,
                                        holdout=holdout)
    asset_ids = {row["asset_id"] for row in rows}
    version_id = uuid.uuid4().hex
    directory = workspace.root / "datasets" / version_id
    taxonomy = workspace.get_taxonomy(task_set["taxonomy_id"])
    card = {
        "contract": MANIFEST_CONTRACT, "task": task_set["task"],
        "task_set_id": task_set_id, "collection_id": task_set["collection_id"],
        "label_semantics": task_set["label_semantics"],
        "splits": split_summary, "excluded": stats["excluded"][:200],
        "excluded_count": len(stats["excluded"]), "targets": stats["targets"],
        "holdout": holdout or [],
    }
    try:
        path, digest, counts = write_manifest(rows, directory)
        (directory / "dataset.json").write_text(
            json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8")
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    relative = path.relative_to(workspace.root).as_posix()
    version = workspace.add_dataset_version(
        task_set_id, status="ready", manifest_path=relative, manifest_sha256=digest,
        sample_count=len(rows), asset_count=len(asset_ids),
        split=counts, preprocessing=preprocessing,
        taxonomy_snapshot={"name": taxonomy["name"], "version": taxonomy["version"],
                           "classes": json.loads(taxonomy["classes_json"])},
        card=card)
    return {"version": version, "card": card}


def read_manifest(workspace, dataset_version, *, limit=None):
    """读取数据版本清单（逐行 yield dict）；供训练消费与检查使用。"""
    if dataset_version.get("manifest_path") is None:
        raise ValueError("数据版本没有清单文件")
    path = workspace.root / dataset_version["manifest_path"]
    if not path.is_file():
        raise ValueError("数据版本清单文件缺失")
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def verify_dataset_version(workspace, dataset_version):
    """核对清单哈希与样本数；训练前调用。"""
    path = workspace.root / dataset_version["manifest_path"]
    if not path.is_file():
        raise ValueError("数据版本清单文件缺失")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != dataset_version["manifest_sha256"]:
        raise ValueError("数据版本清单校验失败（文件已被修改）")
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if count != dataset_version["sample_count"]:
        raise ValueError("数据版本清单行数与记录不一致")
    return count


def bootstrap_labels_from_versions(workspace, task_set_id, *, asset_ids=None,
                                   source="generator"):
    """从目标参考参数生成初始标注（生成页与测试用）。

    * 检测任务：为 ``for_detection`` 的目标按参数版本追加标签（会话/逐跳按
      ``task_sets.label_semantics`` 对应；``per_hop_v1`` 只为 ``scope='hop'``
      的目标写标签）；
    * AMC 任务：``modulation`` 能映射到类别字典的写 ``known``，其余按
      ``out_of_taxonomy``（保留原始名）或 ``unknown`` 记录。
    已有当前标签的目标跳过（幂等，重复调用不产生新版本）。
    """
    task_set = workspace.get_task_set(task_set_id)
    task = task_set["task"]
    collection_id = task_set["collection_id"]
    taxonomy = workspace.get_taxonomy(task_set["taxonomy_id"])
    classes = json.loads(taxonomy["classes_json"])
    selected = set(asset_ids) if asset_ids is not None else None
    created = skipped = 0
    for target in workspace.collection_targets(collection_id, task=task, with_current=True):
        if selected is not None and target["asset_id"] not in selected:
            continue
        if target.get("current") is None:
            skipped += 1
            continue
        existing = workspace.current_label(task, task_set_id, target["id"])
        if existing is not None:
            skipped += 1
            continue
        if task == "detection":
            if not detection_label_applies(task_set["label_semantics"] or "session_v1", target):
                skipped += 1
                continue
            workspace.append_detection_label(task_set_id, target["id"], source=source)
        else:
            modulation = (target["current"].get("modulation") or "").strip()
            mapped = _class_for_modulation(modulation, classes)
            if mapped is not None:
                workspace.append_amc_label(task_set_id, target["id"], source=source,
                                           class_state="known", class_name=mapped)
            elif modulation:
                workspace.append_amc_label(task_set_id, target["id"], source=source,
                                           class_state="out_of_taxonomy",
                                           class_name=modulation)
            else:
                workspace.append_amc_label(task_set_id, target["id"], source=source,
                                           class_state="unknown")
        created += 1
    return {"created": created, "skipped": skipped}


def _class_for_modulation(modulation, classes):
    from .ml.amc import mode_to_class

    text = str(modulation or "").strip()
    mapped = MODULATION_TO_CLASS.get(text.upper()) or mode_to_class(text.lower())
    return mapped if mapped in classes else None


def label_dataset_asset(workspace, task_set_id, asset_id, *, coverage="complete",
                        negative_kind=None, source="manual"):
    """便捷封装：为单条资产登记覆盖度。"""
    return workspace.set_asset_coverage(task_set_id, asset_id, coverage,
                                        negative_kind=negative_kind, source=source)
