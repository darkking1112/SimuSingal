"""生成参数（内部格式 ``gen_recipe_v1``）：校验、样本级种子与参数预览。

方案文档 §6.3：生成参数在内部存为**不可变 JSON 配方**，用来保证可复现；界面上是
表单（``collection_gen_gui``），JSON 导入导出只放在“高级”里。本模块只负责
“抽参数、不合成 IQ”的部分：校验结构、按样本派生种子、试抽样并统计覆盖度、
对照生成器能力做启动前校验（:func:`check_generator_support`）。真正的 IQ 合成
由 ``collection_gen`` 执行器调用项目生成器完成；配方里用到但生成器尚未支持的
项（损伤、独立符号率）不做静默忽略，启动前直接报错。

分布写法（字段名与现有脚本参数一一对应）：

* ``{"fixed": value}``          固定值；
* ``{"choice": [...], "weights": [...]}``  离散选择（可选权重）；
* ``{"uniform": [low, high]}``  连续均匀；
* ``{"loguniform": [low, high]}``  对数均匀（low 必须大于 0）；
* ``{"balanced": [...]}``       均衡轮转（按样本序号轮换，保证类别配比）。

样本级种子：第 i 个样本的种子由 ``base_seed`` 与 ``i`` 哈希派生
（:func:`sample_seed`），不再共享一个顺序随机流，因此每个样本独立可复现、
支持断点续跑与并行。这会改变随机序列：同参数下新配方生成的数据与旧脚本
不逐位相同（方案文档 §6.3 已说明）。
"""
import hashlib
import json

import numpy as np

RECIPE_CONTRACT = "gen_recipe_v1"
ENGINES = ("project", "torchsig")
DISTRIBUTIONS = ("fixed", "choice", "uniform", "loguniform", "balanced")
SPLITS = ("train", "val", "test")
_STRUCTURAL_KEYS = {"contract", "engine", "base_seed", "count", "stratify", "holdout",
                    "labels", "note"}


def _fail(path, message):
    raise ValueError(f"配方字段 {'.'.join(path) or '<根>'}：{message}")


def _validate_distribution(path, node):
    dist_keys = [key for key in node if key in DISTRIBUTIONS]
    if len(dist_keys) != 1:
        _fail(path, "分布节点必须恰好包含一种分布（fixed / choice / uniform / "
                    "loguniform / balanced）")
    extra = [key for key in node if key not in DISTRIBUTIONS and key != "weights"]
    if extra:
        _fail(path, f"分布节点包含未知键：{' / '.join(sorted(extra))}")
    kind = dist_keys[0]
    value = node[kind]
    if "weights" in node and kind != "choice":
        _fail(path, "weights 只能与 choice 搭配")
    if kind == "fixed":
        return
    if kind in ("choice", "balanced"):
        if not isinstance(value, list) or not value:
            _fail(path, f"{kind} 必须是非空列表")
        if "weights" in node:
            weights = node["weights"]
            if (not isinstance(weights, list) or len(weights) != len(value)
                    or any((not isinstance(item, (int, float)) or isinstance(item, bool)
                            or not np.isfinite(item) or item <= 0) for item in weights)):
                _fail(path, "weights 必须是与 choice 等长的正数列表")
    else:
        if (not isinstance(value, list) or len(value) != 2
                or any(not isinstance(item, (int, float)) or isinstance(item, bool)
                       or not np.isfinite(item) for item in value)):
            _fail(path, f"{kind} 必须是 [下限, 上限] 两个有限数")
        low, high = float(value[0]), float(value[1])
        if low > high:
            _fail(path, "下限不能大于上限")
        if kind == "loguniform" and low <= 0:
            _fail(path, "loguniform 的下限必须大于 0")


def _iter_distributions(node, path=()):
    """深度优先找出全部参数分布节点；返回 ``[(path, kind, node)]``。

    分组字典的**每个值都必须是分布节点**：常量要用 ``{"fixed": …}`` 包装，
    拼写错误的分布名（如 ``{"gauss": …}``）在预览阶段就报错，不静默忽略。
    """
    if isinstance(node, dict):
        if any(key in DISTRIBUTIONS for key in node):
            _validate_distribution(path, node)
            kind = next(key for key in DISTRIBUTIONS if key in node)
            yield path, kind, node
            return
        for key, value in node.items():
            if not isinstance(value, dict):
                _fail((*path, str(key)),
                      "参数值必须是分布节点（fixed / choice / uniform / loguniform / "
                      "balanced），常量请用 fixed 包装")
            yield from _iter_distributions(value, (*path, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            if isinstance(value, (dict, list)):
                yield from _iter_distributions(value, (*path, str(index)))


def validate_recipe(recipe):
    """校验配方结构；失败抛 ``ValueError``（带字段路径），成功返回规范化信息。"""
    if not isinstance(recipe, dict):
        raise ValueError("配方必须是 JSON 对象")
    if recipe.get("contract") != RECIPE_CONTRACT:
        raise ValueError(f"配方契约应为 {RECIPE_CONTRACT}")
    engine = recipe.get("engine")
    if engine not in ENGINES:
        raise ValueError(f"生成引擎应为 {' / '.join(ENGINES)} 之一")
    seed = recipe.get("base_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("base_seed 应为 0～4294967295 的整数")
    count = recipe.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10_000_000:
        raise ValueError("count 应为 1～10000000 的整数")
    stratify = recipe.get("stratify")
    if stratify is not None:
        if not isinstance(stratify, dict) or not isinstance(stratify.get("axes"), list) \
                or not stratify["axes"]:
            raise ValueError("stratify 必须包含非空的 axes 列表")
        if any(not isinstance(axis, str) or not axis for axis in stratify["axes"]):
            raise ValueError("stratify.axes 的元素必须是非空文本")
        minimum = stratify.get("min_per_cell", 1)
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
            raise ValueError("stratify.min_per_cell 应为正整数")
    holdout = recipe.get("holdout")
    if holdout is not None:
        if not isinstance(holdout, list):
            raise ValueError("holdout 必须是列表")
        for index, item in enumerate(holdout):
            if (not isinstance(item, dict) or not isinstance(item.get("axis"), str)
                    or not isinstance(item.get("range"), list) or len(item["range"]) != 2
                    or item.get("split") not in SPLITS):
                raise ValueError(f"holdout[{index}] 应含 axis、range=[低, 高]、split"
                                 f"（{' / '.join(SPLITS)}）")
            low, high = item["range"]
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       for value in (low, high)) or low > high:
                raise ValueError(f"holdout[{index}].range 必须是递增的两个数值")
    labels = recipe.get("labels")
    if labels is not None:
        if not isinstance(labels, dict):
            raise ValueError("labels 必须是对象")
        detection = labels.get("detection")
        if detection not in (None, "session_v1", "per_hop_v1"):
            raise ValueError("labels.detection 应为 session_v1 / per_hop_v1 / null")
        if not isinstance(labels.get("amc", False), bool):
            raise ValueError("labels.amc 应为布尔值")
    parameters = []
    for path, kind, _node in _iter_distributions(
            {key: value for key, value in recipe.items() if key not in _STRUCTURAL_KEYS}):
        parameters.append({"path": list(path), "distribution": kind})
    if not parameters:
        raise ValueError("配方没有可抽样的参数分布")
    axes = list(stratify["axes"]) if stratify else []
    for axis in axes:
        name = axis.split(":", 1)[0]
        matches = [item for item in parameters if item["path"][-1] == name]
        if not matches:
            raise ValueError(f"stratify.axes 引用了不存在的参数：{name}")
    return {"contract": RECIPE_CONTRACT, "engine": engine, "count": count,
            "base_seed": seed, "axes": axes, "parameters": parameters}


def sample_seed(base_seed, index, salt=None):
    """第 ``index`` 个样本的独立种子（0～2**32-1）；同输入永远同输出。

    ``salt`` 用于同一样本内的子流（如第 k 个信号槽位）；缺省时与旧实现逐位一致。
    """
    text = f"{int(base_seed)}:{int(index)}" if salt is None \
        else f"{int(base_seed)}:{int(index)}:{int(salt)}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


class _FixedIndex(dict):
    """``balanced`` 的计数器：取样本（槽位）序号，而不是累加调用次数，样本间彼此独立。"""

    def __init__(self, index):
        super().__init__()
        self._index = int(index)

    def get(self, key, default=None):
        return self._index

    def __setitem__(self, key, value):
        pass


def _draw(kind_node, rng, counters, path):
    kind = next(key for key in DISTRIBUTIONS if key in kind_node)
    value = kind_node[kind]
    if kind == "fixed":
        return value
    if kind == "balanced":
        index = counters.get(path, 0)
        counters[path] = index + 1
        return value[index % len(value)]
    if kind == "choice":
        weights = kind_node.get("weights")
        if weights is not None:
            total = float(sum(weights))
            pick = float(rng.random()) * total
            accum = 0.0
            for item, weight in zip(value, weights):
                accum += float(weight)
                if pick <= accum:
                    return item
            return value[-1]
        return value[int(rng.integers(0, len(value)))]
    low, high = float(value[0]), float(value[1])
    if kind == "uniform":
        return float(rng.uniform(low, high))
    if low == high:
        return low
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _draw_tree(node, rng, counters, path=()):
    if isinstance(node, dict) and any(key in DISTRIBUTIONS for key in node):
        return _draw(node, rng, counters, path)
    if isinstance(node, dict):
        return {str(key): _draw_tree(value, rng, counters, (*path, str(key)))
                for key, value in node.items()}
    if isinstance(node, list):
        return [_draw_tree(value, rng, counters, (*path, str(index)))
                for index, value in enumerate(node)]
    return node


def draw_record(recipe, index):
    """执行器用的抽样：第 ``index`` 条录制的记录级参数与逐信号参数。

    ``signals`` 组里除 ``count`` 外的每个参数，对每个信号槽位**独立再抽一次**
    （槽位 k 的随机流由 ``sample_seed(base, index, k)`` 派生）；``balanced`` 按
    ``index + k`` 轮换，因此同一条录制里的多个信号在样式数足够时互不相同。
    ``hopping`` 组随信号槽位一起抽取，``impairments`` 组为记录级。
    返回 ``{"record": {...}, "impairments": {...}, "signals": [{..., "hopping": {...}}]}``。
    """
    base = recipe["base_seed"]
    rng = np.random.default_rng(sample_seed(base, index))
    counters = _FixedIndex(index)
    record = _draw_tree(recipe.get("record") or {}, rng, counters, ("record",))
    impairments = _draw_tree(recipe.get("impairments") or {}, rng, counters, ("impairments",))
    torchsig = _draw_tree(recipe.get("torchsig") or {}, rng, counters, ("torchsig",))
    slots = dict(recipe.get("signals") or {})
    count_node = slots.pop("count", None)
    count = int(_draw(count_node, rng, counters, ("signals", "count"))) if count_node else 1
    signals = []
    for slot in range(count):
        slot_rng = np.random.default_rng(sample_seed(base, index, slot))
        slot_counters = _FixedIndex(index + slot)
        params = _draw_tree(slots, slot_rng, slot_counters, ("signals",))
        params["hopping"] = _draw_tree(recipe.get("hopping") or {}, slot_rng,
                                       slot_counters, ("hopping",))
        signals.append(params)
    return {"record": record, "impairments": impairments, "torchsig": torchsig,
            "signals": signals}


def literal_values(node):
    """分布节点里可枚举的取值（``fixed`` / ``choice`` / ``balanced``）；连续分布返回 ``None``。"""
    for kind in ("fixed", "choice", "balanced"):
        if kind in node:
            return [node[kind]] if kind == "fixed" else list(node[kind])
    return None


def distribution_bounds(node):
    """数值分布的取值区间 ``(低, 高)``；用于把分布压成外部工具的“区间”参数。"""
    if "fixed" in node:
        value = float(node["fixed"])
        return value, value
    if "uniform" in node or "loguniform" in node:
        low, high = node.get("uniform") or node["loguniform"]
        return float(low), float(high)
    values = [float(item) for item in literal_values(node)]
    return min(values), max(values)


#: 项目引擎当前支持的参数（其余一律报错，不静默忽略）。
PROJECT_PARAMETERS = {
    "record": {"sample_rate_hz", "duration_s", "noise_power_dbfs"},
    "signals": {"count", "mode", "bandwidth_ratio", "snr_db", "power_dbfs"},
    "hopping": {"hop_rate_hz"},
}
#: TorchSig 引擎支持的参数：区间会被压成 ``build_torchsig.py`` 的命令行区间。
TORCHSIG_PARAMETERS = {
    "record": {"sample_rate_hz", "duration_s"},
    "signals": {"count", "bandwidth_ratio", "snr_db"},
    "torchsig": {"signal_generators", "impairment_level"},
}


def check_generator_support(recipe):
    """启动前对照生成器能力校验配方；不支持的项直接报错（方案 §6.4 约定 5）。

    ``validate_recipe`` 只管结构，这里才知道“生成器能不能做”：独立符号率与全部
    损伤项尚未实现（见 :mod:`signal_analysis.impairments`），TorchSig 没有跳频、
    没有逐跳标注，扰动只有 0/1/2 三档。
    """
    from . import impairments
    from .core_api import MODE_NAMES as MODES

    engine = recipe.get("engine")
    allowed = PROJECT_PARAMETERS if engine == "project" else TORCHSIG_PARAMETERS
    if engine == "project":
        impairments.ensure_supported(recipe.get("impairments"))
        if "symbol_rate_baud" in (recipe.get("signals") or {}):
            raise ValueError(impairments.RESERVED_SIGNAL_PARAMETERS["symbol_rate_baud"]["reason"])
    for group in ("record", "signals", "hopping", "impairments", "torchsig"):
        node = recipe.get(group)
        if node is None:
            continue
        if engine == "project" and group in ("impairments", "torchsig"):
            if group == "torchsig":
                raise ValueError("项目引擎不使用 torchsig 组参数")
            continue  # impairments 已由 ensure_supported 校验
        if group not in allowed:
            label = {"hopping": "跳频", "impairments": "损伤"}.get(group, group)
            raise ValueError(f"TorchSig 引擎不支持“{label}”参数（扰动只有 "
                             "torchsig.impairment_level 的 0/1/2 三档，且没有跳频）")
        unknown = sorted(set(node) - allowed[group])
        if unknown:
            raise ValueError(f"{engine} 引擎不支持参数：{group}.{' / '.join(unknown)}")
    if engine == "project":
        modes = literal_values((recipe.get("signals") or {}).get("mode") or {"fixed": None})
        bad = [item for item in (modes or []) if item not in MODES]
        if bad:
            raise ValueError(f"signals.mode 含不支持的样式：{' / '.join(map(str, bad))}；"
                             f"可选：{', '.join(MODES)}")
    else:
        record = recipe.get("record") or {}
        for key, label in (("sample_rate_hz", "采样率"), ("duration_s", "时长")):
            node = record.get(key)
            if node is not None and len(set(literal_values(node) or [0, 1])) != 1:
                raise ValueError(f"TorchSig 引擎的{label}只能是单个值")
        labels = recipe.get("labels") or {}
        if labels.get("detection") == "per_hop_v1":
            raise ValueError("TorchSig 没有跳频，不能产生逐跳标注（请选会话级）")


def _flatten(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten(value, (*path, str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _flatten(value, (*path, str(index)))
    else:
        yield path, node


def _bin_axis(values, bins):
    """数值轴等宽分箱；``bins`` 为箱数，返回 ``[(label, count)]``。"""
    numbers = [float(value) for value in values]
    low, high = min(numbers), max(numbers)
    if low == high:
        return [(f"{low:g}", len(numbers))]
    width = (high - low) / int(bins)
    counts = {}
    for number in numbers:
        index = min(int((number - low) / width), int(bins) - 1)
        counts[index] = counts.get(index, 0) + 1
    return [(f"[{low + index * width:g}, {low + (index + 1) * width:g})",
             counts.get(index, 0)) for index in range(int(bins))]


def preview_recipe(recipe, *, samples=None, preview_rows=5):
    """仅抽参数、不合成 IQ 的试抽样；返回参数样例、分布与分层格子计数。

    ``samples`` 默认等于配方 ``count``（建议预览时用较小的值，例如 1 万）。预览按
    参数逐项抽样（每条录制每个参数一个值）；执行器对每个信号槽位再独立抽取
    ``signals`` 组（:func:`draw_record`），分布与种子规则相同。
    """
    info = validate_recipe(recipe)
    count = int(samples or info["count"])
    if not 1 <= count <= 10_000_000:
        raise ValueError("预览样本数应为 1～10000000 的整数")
    param_root = {key: value for key, value in recipe.items()
                  if key not in _STRUCTURAL_KEYS}
    counters = {}
    rows = []
    for index in range(count):
        rng = np.random.default_rng(sample_seed(info["base_seed"], index))
        rows.append(_draw_tree(param_root, rng, counters))
    leaves = []
    for row in rows:
        leaves.append(dict(_flatten(row)))
    axis_reports = []
    axis_names = []
    stratify = recipe.get("stratify") or {}
    for axis in stratify.get("axes", []):
        name, _, raw_bins = axis.partition(":")
        matches = [item for item in info["parameters"] if item["path"][-1] == name]
        if not matches:
            raise ValueError(f"stratify.axes 引用了不存在的参数：{name}")
        path = tuple(matches[0]["path"])
        axis_names.append(name)
        values = [leaf.get(path) for leaf in leaves]
        if raw_bins and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            for value in values):
            report = [{"label": label, "count": count_}
                      for label, count_ in _bin_axis(values, int(raw_bins))]
        else:
            counts = {}
            for value in values:
                key = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) \
                    else value
                counts[key] = counts.get(key, 0) + 1
            report = [{"label": label, "count": count_}
                      for label, count_ in sorted(counts.items(),
                                                  key=lambda item: (-item[1], item[0]))]
        axis_reports.append({"axis": name, "path": list(path), "bins": raw_bins or None,
                             "values": report})
    cells = {}
    if axis_names:
        for leaf in leaves:
            key = tuple(_cell_value(leaf.get(tuple(
                next(item["path"] for item in info["parameters"]
                     if item["path"][-1] == name)))) for name in axis_names)
            cells[key] = cells.get(key, 0) + 1
    minimum = int(stratify.get("min_per_cell", 1)) if stratify else 1
    below = [(key, count_) for key, count_ in cells.items() if count_ < minimum]
    preview = []
    for index in range(min(preview_rows, count)):
        preview.append({".".join(path): value for path, value in leaves[index].items()})
    return {
        "contract": RECIPE_CONTRACT,
        "engine": info["engine"],
        "count": count,
        "cells": {
            "axes": axis_names,
            "total": len(cells),
            "below_min_per_cell": len(below),
            "min_per_cell": minimum,
            "examples": [{"cell": list(key), "count": count_} for key, count_ in below[:20]],
        },
        "axes": axis_reports,
        "preview": preview,
    }


def _cell_value(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) \
        if isinstance(value, (dict, list)) else value
