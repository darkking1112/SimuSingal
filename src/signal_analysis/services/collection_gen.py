"""信号集合生成执行器（方案 §6.3、§7.2）：按生成参数批量合成、写分片、入集合并自动标注。

这是纯 NumPy 的后台任务，和主程序同一个 Python 环境，由应用已有的工作进程
（``tasks.run_job``）运行，不再借训练页的外部进程。只有 TorchSig 引擎要依赖
``torchsig`` 且只能在 Linux 下运行，它走外部进程生成 bundle，再由
:mod:`signal_analysis.integrations.torchsig` 导入。

约定：

* **集合 + 配方**：生成时自动把配方存进 ``recipes`` 表并绑定到集合（集合还没有
  配方时；追加到已有集合不改它原来的配方），每条资产的元数据里记配方号与样本序号，
  同一份配方 + 序号即可复现；
* **自动标注**：项目引擎知道全部真值。存储层在写分片时已经按生成摘要登记了目标与
  参考参数（会话；跳频样式同时有逐跳子目标），这里只负责建标注集、设覆盖度并
  从参考参数写初始标签；
* **进度与取消**：进度写 ``<job_dir>/progress.json``；取消由父进程写
  ``<job_dir>/cancel.flag``，本任务在两条录制之间检查，收尾后返回部分结果；
* **失败不硬凑**：某条录制参数放不下（频带装不进采样带宽等）就跳过并按原因计数，
  不会悄悄改参数补足数量。分层配额的“补抽”尚未实现，预览只报告不足的格子。
"""
import json
import time
from pathlib import Path

import numpy as np

from ..algorithms.generation import impairments
from ..core_api import generate_iq, occupied_interval, plan_signal
from ..data.datasets import bootstrap_labels_from_versions
from ..evaluation import hop_truth, signal_truth
from ..algorithms.generation.recipes import check_generator_support, draw_record, sample_seed, validate_recipe

#: 单个分片封存前的目标字节数；分片越小，中途取消时未封存的部分越少。
SHARD_BYTES = 256 * 1024 * 1024
#: 信号频带之间、频带与奈奎斯特边缘之间的保护间隔（占采样率的比例）。
GUARD_RATIO = 0.01
#: 跳频样式：与 ``plan_signal`` 的 ``hop_rate`` 对应。
HOPPING_MODES = ("fh_rc", "fh_video")
#: 同一条录制放置信号的最大重试次数。
PLACEMENT_TRIES = 64


class Reporter:
    """进度与取消通道：worker 写 ``progress.json``，父进程写 ``cancel.flag``。"""

    def __init__(self, job_dir):
        self.directory = Path(job_dir) if job_dir else None
        self._last = 0.0

    def emit(self, done, total, message, *, force=False, **extra):
        if self.directory is None:
            return
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        path = self.directory / "progress.json"
        temporary = path.with_suffix(".tmp")
        payload = {"done": int(done), "total": int(total), "message": message, **extra}
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def cancelled(self):
        return self.directory is not None and (self.directory / "cancel.flag").exists()


# ---------------------------------------------------------------------------
# 标注集与自动标注
# ---------------------------------------------------------------------------


def prepare_task_sets(workspace, collection, labels):
    """按“标注”勾选建立（或沿用）集合的检测/AMC 标注集；粒度冲突直接报错。"""
    labels = labels or {}
    result = {}
    existing = workspace.list_task_sets(collection["id"])
    wanted = labels.get("detection")
    if wanted:
        current = [item for item in existing if item["task"] == "detection"]
        match = next((item for item in current
                      if (item["label_semantics"] or "session_v1") == wanted), None)
        if match is None and current:
            raise ValueError(
                f"集合「{collection['name']}」已有检测标注集「{current[0]['name']}」"
                f"（{current[0]['label_semantics'] or 'session_v1'}），与本次选择的 {wanted} "
                "不一致；请换一个集合，或改用相同的检测标注粒度")
        result["detection"] = match or workspace.create_task_set(
            collection["id"], "detection", name="检测标注", label_semantics=wanted)
    if labels.get("amc"):
        match = next((item for item in existing if item["task"] == "amc"), None)
        result["amc"] = match or workspace.create_task_set(collection["id"], "amc",
                                                           name="AMC 标注")
    return result


def annotate_assets(workspace, task_sets, asset_ids, *, source, noise_only=()):
    """对新资产设覆盖度并从目标参考参数写初始标签；返回各任务新增的标签数。

    生成器（以及带完整实例元数据的 TorchSig bundle）列全了每条录制里的信号，
    所以检测覆盖度记为 ``complete``；没有任何信号的录制是纯噪声负样本。
    """
    created = {"detection": 0, "amc": 0}
    asset_ids = list(asset_ids)
    if not asset_ids:
        return created
    detection = task_sets.get("detection")
    if detection is not None:
        noise = set(noise_only)
        for asset_id in asset_ids:
            workspace.set_asset_coverage(detection["id"], asset_id, "complete",
                                         negative_kind="noise_only" if asset_id in noise
                                         else None, source=source)
        created["detection"] = bootstrap_labels_from_versions(
            workspace, detection["id"], asset_ids=asset_ids, source=source)["created"]
    amc = task_sets.get("amc")
    if amc is not None:
        created["amc"] = bootstrap_labels_from_versions(
            workspace, amc["id"], asset_ids=asset_ids, source=source)["created"]
    return created


# ---------------------------------------------------------------------------
# 项目引擎：配方 → 一条录制
# ---------------------------------------------------------------------------


def _place_signals(specs, rate, rng):
    """给每个信号选互不重叠的频点；放不下就报错（不静默改参数）。"""
    guard = GUARD_RATIO * rate
    used = []
    for spec in specs:
        plan = plan_signal({**spec, "offset": 0.0}, rate)
        rel_low, rel_high = occupied_interval(0.0, plan["bandwidth_actual"], plan["mode"],
                                              plan.get("side"))
        low = -rate / 2.0 + guard / 2.0 - rel_low
        high = rate / 2.0 - guard / 2.0 - rel_high
        if plan["mode"] == "ssb":  # check_band 对 SSB 用 offset ± 带宽的保守校验
            low = max(low, -rate / 2.0 + plan["bandwidth"])
            high = min(high, rate / 2.0 - plan["bandwidth"])
        if low > high:
            raise ValueError("信号带宽过大，放不进采样带宽")
        for _ in range(PLACEMENT_TRIES):
            offset = float(rng.uniform(low, high))
            interval = (offset + rel_low, offset + rel_high)
            if all(interval[1] + guard <= other[0] or interval[0] - guard >= other[1]
                   for other in used):
                used.append(interval)
                spec["offset"] = offset
                break
        else:
            raise ValueError("无法把全部信号放进采样带宽而互不重叠（带宽占比过大或信号过多）")
    return specs


def synthesize_record(recipe, index):
    """按配方合成第 ``index`` 条录制；返回 ``(samples, summary, info)``。

    ``summary`` 是项目生成器的 ``iq_generator_v1`` 摘要（真值来源），``info`` 是本次
    抽到的参数与种子，供资产元数据记录。参数不合法抛 ``ValueError``。
    """
    drawn = draw_record(recipe, index)
    rate = float(drawn["record"].get("sample_rate_hz", 1_000_000.0))
    duration = float(drawn["record"].get("duration_s", 0.1))
    layout = np.random.default_rng(sample_seed(recipe["base_seed"], index, -1))
    specs = []
    for params in drawn["signals"]:
        mode = str(params["mode"])
        spec = {"mode": mode, "offset": 0.0,
                "bandwidth": float(params.get("bandwidth_ratio", 0.1)) * rate,
                "power_dbfs": float(params.get("power_dbfs", -10.0))}
        if mode in HOPPING_MODES:
            # 显式给出单跳带宽与跨度并留 2% 余量：plan_signal 的“全自动”取值在浮点上会
            # 恰好超出整体带宽，随带宽数值不同而间歇报错
            hop_bandwidth = spec["bandwidth"] / 10.0
            spec.update(hop_bandwidth=hop_bandwidth,
                        hop_span=0.98 * (spec["bandwidth"] - hop_bandwidth))
            if "hop_rate_hz" in params["hopping"]:
                spec["hop_rate"] = float(params["hopping"]["hop_rate_hz"])
        specs.append(spec)
    specs = _place_signals(specs, rate, layout)
    noise = {"enabled": True, "bandwidth": rate}
    if specs:
        strongest = int(np.argmax([spec["power_dbfs"] for spec in specs]))
        noise["snr_db"] = float(drawn["signals"][strongest].get("snr_db", 20.0))
    else:
        noise["power_dbfs"] = float(drawn["record"].get("noise_power_dbfs", -20.0))
    seed = sample_seed(recipe["base_seed"], index)
    samples, summary = generate_iq(rate, duration, specs, noise, seed)
    # 损伤的调用位置（方案 §6.4 约定 2）：目前全部未实现，抽到非空即报错
    samples = impairments.apply_impairments(samples, rate, drawn["impairments"],
                                            np.random.default_rng(seed), stage="record")
    return samples, summary, {"sample_seed": seed, "signal_count": len(specs)}


def _flush_labels(workspace, task_sets, pending, noise, source, totals):
    if not pending:
        return
    created = annotate_assets(workspace, task_sets, pending, source=source,
                              noise_only=noise)
    for key, value in created.items():
        totals[key] += value
    pending.clear()
    noise.clear()


def _run_project(workspace, recipe, collection, recipe_row, task_sets, reporter, request):
    total = int(recipe["count"])
    max_seconds = request.get("max_seconds")
    created_by = request.get("created_by")
    started = time.monotonic()
    writer, shard_bytes, shards = None, 0, 0
    pending, noise_ids = [], set()
    failures, totals = {}, {"detection": 0, "amc": 0}
    created = sessions = hops = 0
    stopped = None
    try:
        for index in range(total):
            if reporter.cancelled():
                stopped = "cancelled"
                break
            if max_seconds and time.monotonic() - started > float(max_seconds):
                stopped = "time_limit"
                break
            try:
                samples, summary, info = synthesize_record(recipe, index)
            except ValueError as exc:
                failures[str(exc)] = failures.get(str(exc), 0) + 1
                reporter.emit(index + 1, total, f"生成中 {index + 1}/{total}（成功 {created}）")
                continue
            if writer is None:
                shards += 1
                writer = workspace.create_shard(
                    name=f"{collection['name']} · 分片 {shards}", created_by=created_by)
            asset = writer.append(
                samples, summary["sample_rate_hz"], f"{collection['name']} · {index:06d}",
                f"generated:recipe_{recipe['engine']}_v1", source_kind="generated",
                metadata={"generation": summary,
                          "recipe": {"recipe_id": recipe_row["id"], "index": index, **info}})
            workspace.add_collection_member(collection["id"], asset["id"])
            created += 1
            sessions += len(signal_truth(summary))
            hops += len(hop_truth(summary))
            pending.append(asset["id"])
            if info["signal_count"] == 0:
                noise_ids.add(asset["id"])
            shard_bytes += int(samples.nbytes)
            if shard_bytes >= SHARD_BYTES:
                writer.seal()
                writer, shard_bytes = None, 0
                reporter.emit(index + 1, total, "写入标注…", force=True)
                _flush_labels(workspace, task_sets, pending, noise_ids, "generator", totals)
            reporter.emit(index + 1, total, f"生成中 {index + 1}/{total}（成功 {created}）")
    finally:
        if writer is not None:
            writer.seal()
    reporter.emit(total, total, "写入标注…", force=True)
    _flush_labels(workspace, task_sets, pending, noise_ids, "generator", totals)
    return {"created": created, "failures": failures, "sessions": sessions, "hops": hops,
            "labels": totals, "stopped": stopped, "shards": shards}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def generate_collection(workspace, request, resolve_collection):
    """按 ``request["recipe"]`` 批量生成到集合；``resolve_collection()`` 解析/新建目标集合。"""
    recipe = request["recipe"]
    validate_recipe(recipe)
    check_generator_support(recipe)
    if recipe["engine"] == "project" and "mode" not in (recipe.get("signals") or {}):
        raise ValueError("项目引擎的生成参数必须给出 signals.mode（调制类型）")
    labels = recipe.get("labels") or {}
    reporter = Reporter(request.get("job_dir"))
    started = time.monotonic()
    if recipe["engine"] == "torchsig":
        from ..integrations import torchsig

        torchsig.preflight(workspace, request)
    collection = resolve_collection()
    if collection is None:
        raise ValueError("请指定目标集合（新建或追加）")
    task_sets = prepare_task_sets(workspace, collection, labels)
    recipe_row = workspace.ensure_recipe(
        request.get("recipe_name") or f"{collection['name']} · 生成参数",
        recipe["engine"], recipe, created_by=request.get("created_by"))
    workspace.set_collection_recipe(collection["id"], recipe_row["id"])
    reporter.emit(0, int(recipe["count"]), "开始生成…", force=True)
    if recipe["engine"] == "project":
        outcome = _run_project(workspace, recipe, collection, recipe_row, task_sets,
                               reporter, request)
    else:
        from ..integrations import torchsig

        outcome = torchsig.run_generation(workspace, recipe, collection, recipe_row,
                                          task_sets, reporter, request)
    reasons = sorted(outcome["failures"].items(), key=lambda item: -item[1])
    failed = sum(outcome["failures"].values())
    return {"kind": "generate_collection", "engine": recipe["engine"],
            "requested": int(recipe["count"]), "created": outcome["created"],
            "failed": failed,
            "failure_reasons": [{"reason": reason, "count": count}
                                for reason, count in reasons[:5]],
            "stopped": outcome["stopped"], "collection_id": collection["id"],
            "collection_name": collection["name"], "recipe_id": recipe_row["id"],
            "sessions": outcome["sessions"], "hops": outcome["hops"],
            "labels": outcome["labels"], "shards": outcome["shards"],
            "elapsed_s": round(time.monotonic() - started, 3)}
