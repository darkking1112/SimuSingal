"""AMC 训练模型的纯元数据目录：模型说明、参数 schema 与校验。

本模块**不导入 torch**：桌面应用（冻结包不含 torch）用它展示模型列表与参数、
服务层用它做启动前校验，训练侧（``training/amc_models``，不随 wheel 分发）按
``spec.implementation`` 解析真正的网络实现。

目录版本 :data:`CATALOG_VERSION` 与每模型 ``model_revision`` 用于桌面应用与
训练源码目录之间的版本握手，以及历史实验的口径判定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

#: 目录版本：模型增删或参数 schema 变更时递增（应用与训练源码目录握手比对）
CATALOG_VERSION = 1

#: 参数类型；``int-list`` 的元素为整数
KINDS = ("int", "float", "int-list", "bool", "choice")


@dataclass(frozen=True)
class ParamSpec:
    """单个模型参数的声明。

    ``default`` 是默认值的**唯一事实源**：实现模块不得再写一套默认值，
    CLI 未显式给出的参数一律取这里声明的值。
    """

    name: str
    kind: str
    default: Any
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    odd: bool = False                    # 标量必须为奇数（如卷积核长）
    element_minimum: float | None = None
    element_odd: bool = False            # 列表元素必须为奇数
    length: int | None = None            # 列表长度（精确）
    minimum_length: int | None = None
    maximum_length: int | None = None
    choices: tuple[Any, ...] = ()
    divides: str | None = None           # 本参数必须整除另一整数参数（如头数整除隐层宽度）
    label: str = ""                      # 页面/帮助里的显示名（留空用 name）


@dataclass(frozen=True)
class ModelSpec:
    """一个可训练模型的目录条目。"""

    id: str                              # 与 --arch、实验记录、清单 training.arch 共用
    title: str
    summary: str
    implementation: str                  # 训练侧模块路径，如 "amc_models.cnn"
    model_revision: int                  # 结构或前向语义变更时 +1
    layers: tuple[str, ...]              # 架构展示行（GUI 只读文本）
    params: tuple[ParamSpec, ...] = ()
    samples: str = "any"                 # any / exact:<N> / min:<N>
    input_layout: str = "iq_channels_first_v1"
    requires: tuple[str, ...] = ("torch",)
    exportable: bool = True              # 是否允许进入产品推理链路（导出 ONNX）
    notes: str = ""
    cross_validate: Callable[[dict], None] | None = None


def validate_value(param: ParamSpec, value: Any) -> Any:
    """校验并规范化单个参数取值（返回可 JSON 化的形式）。"""
    label = param.label or param.name
    if param.kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} 应为整数")
        number: float = value
    elif param.kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} 应为数值")
        number = float(value)
    elif param.kind == "int-list":
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError(f"{label} 应为非空的整数列表")
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{label} 的每个元素都应为整数")
        if param.length is not None and len(value) != param.length:
            raise ValueError(f"{label} 需要 {param.length} 个正整数")
        if param.minimum_length is not None and len(value) < param.minimum_length:
            raise ValueError(f"{label} 至少需要 {param.minimum_length} 个正整数")
        if param.maximum_length is not None and len(value) > param.maximum_length:
            raise ValueError(f"{label} 最多 {param.maximum_length} 个正整数")
        if param.element_minimum is not None:
            if any(item < param.element_minimum for item in value):
                raise ValueError(f"{label} 的每个元素都不应小于 {param.element_minimum:g}")
        if param.element_odd and any(item % 2 == 0 for item in value):
            raise ValueError(f"{label} 的每个元素都应为奇数")
        return [int(item) for item in value]
    elif param.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{label} 应为布尔值")
        return bool(value)
    elif param.kind == "choice":
        if value not in param.choices:
            raise ValueError(f"{label} 只能是 {' / '.join(map(str, param.choices))}")
        return value
    else:
        raise ValueError(f"参数 {label} 的类型 {param.kind!r} 不受支持")
    if param.minimum is not None:
        if number < param.minimum or (param.exclusive_minimum and number == param.minimum):
            limit = f"大于 {param.minimum:g}" if param.exclusive_minimum else f"不小于 {param.minimum:g}"
            raise ValueError(f"{label} 应{limit}")
    if param.maximum is not None:
        if number > param.maximum or (param.exclusive_maximum and number == param.maximum):
            limit = f"小于 {param.maximum:g}" if param.exclusive_maximum else f"不大于 {param.maximum:g}"
            raise ValueError(f"{label} 应{limit}")
    if param.odd and int(number) % 2 == 0:
        raise ValueError(f"{label} 应为奇数")
    return int(number) if param.kind == "int" else float(number)


def param_hint(param: ParamSpec) -> str:
    """参数的约束说明（页面 tooltip 与 CLI 帮助共用）。"""
    parts = [param.help] if param.help else []
    if param.kind == "int-list":
        bounds = []
        if param.length is not None:
            bounds.append(f"恰好 {param.length} 个")
        elif param.minimum_length is not None or param.maximum_length is not None:
            low = param.minimum_length if param.minimum_length is not None else 1
            high = param.maximum_length if param.maximum_length is not None else "∞"
            bounds.append(f"{low}～{high} 个")
        if param.element_odd:
            bounds.append("奇数")
        elif param.element_minimum is not None:
            bounds.append(f"不小于 {param.element_minimum:g}")
        if bounds:
            parts.append("、".join(bounds))
    elif param.kind in ("int", "float"):
        if param.minimum is not None and param.maximum is not None:
            left = "(" if param.exclusive_minimum else "["
            right = ")" if param.exclusive_maximum else "]"
            parts.append(f"{left}{param.minimum:g}, {param.maximum:g}{right}")
        elif param.minimum is not None:
            parts.append(f"{'>' if param.exclusive_minimum else '≥'} {param.minimum:g}")
        if param.odd:
            parts.append("奇数")
    return "；".join(part for part in parts if part)


def validate_params(spec: ModelSpec, params: dict | None) -> dict:
    """校验参数字典：拒绝未知参数，套用默认值，执行交叉约束；返回完整参数字典。"""
    given = dict(params or {})
    names = [param.name for param in spec.params]
    unknown = sorted(set(given) - set(names))
    if unknown:
        raise ValueError(
            f"模型 {spec.id} 不认识参数：{'、'.join(unknown)}；可用参数："
            f"{'、'.join(names) if names else '无'}")
    resolved = {param.name: validate_value(param, given.get(param.name, param.default))
                for param in spec.params}
    for param in spec.params:
        if not param.divides:
            continue
        other = resolved.get(param.divides)
        value = resolved[param.name]
        if isinstance(other, int) and isinstance(value, int) and value > 0 and other % value:
            label = param.label or param.name
            raise ValueError(f"{label} 必须整除 {param.divides}（{other} 不能被 {value} 整除）")
    if spec.cross_validate is not None:
        spec.cross_validate(resolved)
    return resolved


def merge_param_sources(spec: ModelSpec, *, legacy: dict | None = None,
                        explicit: dict | None = None) -> dict:
    """合并"旧 CLI 单参数"与 ``--model-params``；同一参数两处都显式给出即报错。"""
    legacy = dict(legacy or {})
    explicit = dict(explicit or {})
    conflict = sorted(set(legacy) & set(explicit))
    if conflict:
        raise ValueError("同一参数不能同时写在旧参数与 --model-params 里："
                         f"{'、'.join(conflict)}")
    return validate_params(spec, {**legacy, **explicit})


def samples_constraint(spec: ModelSpec) -> tuple[str, int | None]:
    """解析 ``spec.samples``：返回 ``(kind, value)``，kind 为 any/exact/min。"""
    return parse_samples_constraint(spec.id, spec.samples)


def parse_samples_constraint(model_id: str, constraint: str) -> tuple[str, int | None]:
    """解析窗口约束文本：``"any"`` / ``"exact:<N>"`` / ``"min:<N>"``。"""
    text = str(constraint).strip().lower()
    if text == "any":
        return "any", None
    for kind in ("exact", "min"):
        prefix = f"{kind}:"
        if text.startswith(prefix):
            try:
                return kind, int(text[len(prefix):])
            except ValueError as exc:  # pragma: no cover - 目录数据错误
                raise ValueError(f"模型 {model_id} 的窗口约束 {constraint!r} 非法") from exc
    raise ValueError(f"模型 {model_id} 的窗口约束 {constraint!r} 非法")


def check_model_samples(model_id: str, constraint: str, samples: int) -> None:
    """校验窗口长度是否满足目录约束（不满足即报错，不做静默截断/补零）。"""
    kind, value = parse_samples_constraint(model_id, constraint)
    if kind == "any":
        return
    if kind == "exact" and int(samples) != value:
        raise ValueError(f"模型 {model_id} 固定要求 {value} 点窗口，数据集是 {int(samples)} 点")
    if kind == "min" and int(samples) < int(value):
        raise ValueError(f"模型 {model_id} 要求窗口不少于 {value} 点，数据集是 {int(samples)} 点")


def check_samples(spec: ModelSpec, samples: int) -> None:
    """校验窗口长度是否满足模型约束（不满足即报错，不做静默截断/补零）。"""
    check_model_samples(spec.id, spec.samples, samples)


def spec_json(spec: ModelSpec) -> dict:
    """目录条目的 JSON 形式（GUI 的元数据查询接口用，含构建控件所需的约束字段）。"""
    return {
        "id": spec.id,
        "title": spec.title,
        "summary": spec.summary,
        "model_revision": spec.model_revision,
        "implementation": spec.implementation,
        "layers": list(spec.layers),
        "samples": spec.samples,
        "input_layout": spec.input_layout,
        "requires": list(spec.requires),
        "exportable": spec.exportable,
        "notes": spec.notes,
        "params": [{"name": param.name, "kind": param.kind,
                    # 元组是目录里的写法，JSON 里统一成数组（GUI 直接当列表用）
                    "default": list(param.default) if isinstance(param.default, tuple)
                    else param.default,
                    "help": param.help, "hint": param_hint(param),
                    "label": param.label or param.name,
                    "minimum": param.minimum, "maximum": param.maximum,
                    "exclusive_minimum": param.exclusive_minimum,
                    "exclusive_maximum": param.exclusive_maximum,
                    "odd": param.odd, "choices": list(param.choices),
                    "length": param.length, "minimum_length": param.minimum_length,
                    "maximum_length": param.maximum_length,
                    "element_minimum": param.element_minimum,
                    "element_odd": param.element_odd, "divides": param.divides}
                   for param in spec.params],
    }


def describe_model(model: dict) -> str:
    """按目录条目（``spec_json`` 形式）渲染只读文本（GUI 展示、日志与文档共用）。"""
    lines = [f"{model['title']}（{model['id']} · 结构版本 {model['model_revision']}）",
             model["summary"], "", "结构："]
    lines.extend(f"  - {row}" for row in model["layers"])
    if model.get("params"):
        lines.extend(["", "参数："])
        lines.extend(f"  - {param['label']}（默认 {param['default']}）：{param['hint']}"
                     for param in model["params"])
    lines.extend(["", f"窗口：{model['samples']} · 输入布局：{model['input_layout']}"
                      f" · 依赖：{'、'.join(model['requires'])}"])
    if model.get("missing"):
        lines.append(f"当前训练环境不可用：缺少 {'、'.join(model['missing'])}")
    if model.get("notes"):
        lines.append(f"说明：{model['notes']}")
    return "\n".join(lines)


def describe(spec: ModelSpec) -> str:
    """模型架构与参数的只读文本（GUI 展示、日志与文档共用）。"""
    return describe_model(spec_json(spec))


def catalog_json(specs: dict[str, ModelSpec]) -> str:
    """整份目录的 JSON（应用与训练源码目录握手/展示用）。"""
    return json.dumps({"catalog_version": CATALOG_VERSION,
                       "models": [spec_json(spec) for spec in specs.values()]},
                      ensure_ascii=False, indent=2)
