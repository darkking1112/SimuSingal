"""适配器注册表：把 ``--arch`` 从硬编码分支变成可插拔的数据。

设计要点：

* 注册表**不导入 torch / onnx**，因此"列出有哪些框架"这件事在最小环境里也能用；
* 未知框架、缺少依赖、许可证不允许分发，三种失败都抛 :class:`SystemExit` 并
  给出**可执行的下一步**（安装命令或 ``training/README.md`` 的章节号），
  与 ``train_yolox.py`` 既有的报错风格保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass

APACHE = "Apache-2.0"
AGPL = "AGPL-3.0"

#: 具备"传染性"的许可证：权重随产品分发会产生开源义务（见 training/README.md §8）
COPYLEFT = frozenset({"AGPL-3.0", "AGPL-3.0-only", "AGPL-3.0-or-later", "GPL-3.0"})

READ_ME_HINT = "接入清单与许可证说明见 training/README.md §7 / §8"


@dataclass(frozen=True)
class AdapterInfo:
    """一个检测器适配器的静态描述（全部字段都应能离线给出）。"""

    name: str
    title: str
    license: str
    upstream: str
    notes: str = ""

    @property
    def distributable(self):
        """权重是否可以随产品分发（许可证层面）。"""
        return self.license not in COPYLEFT

    def as_dict(self):
        return {
            "name": self.name,
            "title": self.title,
            "license": self.license,
            "distributable": bool(self.distributable),
            "upstream": self.upstream,
            "notes": self.notes,
        }


_REGISTRY = {}


def register(adapter_cls):
    """类装饰器：把适配器注册进 ``--arch`` 可选值。"""
    adapter = adapter_cls()
    info = adapter.info
    if not isinstance(info, AdapterInfo):
        raise TypeError("适配器的 info 必须是 AdapterInfo")
    key = info.name.strip().lower()
    if key in _REGISTRY:
        raise ValueError(f"适配器名称重复：{info.name}")
    _REGISTRY[key] = adapter
    return adapter_cls


def names():
    """已注册的 ``--arch`` 取值（含排序，便于报错信息稳定）。"""
    return tuple(sorted(_REGISTRY))


def lookup(name):
    """按名字取适配器；未知名字时列出全部可用值。"""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        available = "、".join(names()) or "（无）"
        raise SystemExit(f"未知的检测器框架 --arch {name!r}；可用取值：{available}。\n{READ_ME_HINT}")
    return _REGISTRY[key]


def describe_all():
    """所有适配器的描述（供 ``--list-frameworks`` 与文档生成使用）。"""
    return [_REGISTRY[key].describe() for key in names()]


def require_available(adapter, allow_copyleft=False, runtime=True):
    """检查运行环境与许可证；不满足时抛出带下一步动作的 :class:`SystemExit`。

    ``runtime=False`` 用于「只改写已导出的 ONNX」的场景：此时不需要框架的 Python 包，
    但许可证门禁照常生效。
    """
    if runtime and not adapter.is_available():
        raise SystemExit(
            f"--arch {adapter.info.name} 需要 {adapter.info.title}，当前环境未检测到。\n"
            f"  {adapter.install_hint()}\n{READ_ME_HINT}")
    if adapter.info.license in COPYLEFT and not allow_copyleft:
        raise SystemExit(
            f"--arch {adapter.info.name} 使用 {adapter.info.license}（{adapter.info.title}），"
            "该许可证具有传染性：把权重随产品分发会产生开源义务。\n"
            "  如确认仅用于内网基线对照、不进入发行包，请显式加 --allow-copyleft。\n"
            f"{READ_ME_HINT}")
    return adapter
