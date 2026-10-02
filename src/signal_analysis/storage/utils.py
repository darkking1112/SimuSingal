"""数据分析工作区通用工具：文本/数值校验、JSON 清洗、来源分类与分箱。"""

import json
import re

import numpy as np

#: 目标键解析：``s3`` 与 ``s3.h2``（会话 3 的第 2 跳）。
_TARGET_KEY = re.compile(r"^s(\d+)(?:\.h(\d+))?$")


def _clean_text(value, name, maximum=200, allow_none=False):
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name}不能为空")
    if not isinstance(value, str):
        raise ValueError(f"{name}应为文本")
    text = value.strip()
    if not text and not allow_none:
        raise ValueError(f"{name}不能为空")
    if len(text) > maximum:
        raise ValueError(f"{name}最多 {maximum} 个字符")
    return text


def _optional_text(value, name, maximum=200):
    if value is None:
        return None
    return _clean_text(value, name, maximum, allow_none=True) or None


def _number(value, name, *, minimum=None, maximum=None):
    """有限数值或 ``None``；NaN/Inf 与越界直接报错。"""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name}必须为有限数值")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}不能大于 {maximum}")
    return number


def _integer(value, name, *, minimum=None, maximum=None):
    if value is None:
        return None
    if isinstance(value, bool) or value != int(value):
        raise ValueError(f"{name}应为整数")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name}不能小于 {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name}不能大于 {maximum}")
    return number


def _enum(value, allowed, name):
    if value not in allowed:
        raise ValueError(f"{name}应为 {' / '.join(allowed)} 之一")
    return value


def _json_text(value, name="内容"):
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}不是有效的 JSON：{exc}") from exc


def _json_safe(value):
    """递归去掉非有限浮点；返回可安全序列化的副本。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _classify_source(source):
    """按既有 ``source`` 文本推断来源分类与父资产；不猜测时记为 legacy。"""
    text = str(source or "")
    if text.startswith("parent:"):
        parent = text.split(":", 1)[1].strip()
        return "derived", (parent or None)
    if text.startswith("generated") or text == "generated":
        return "generated", None
    if text.startswith("imported") or re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("/"):
        return "imported", None
    return "legacy", None


def _target_sort_key(row):
    match = _TARGET_KEY.match(str(row.get("target_key", "")))
    if match:
        session = int(match.group(1))
        hop = int(match.group(2)) if match.group(2) is not None else -1
        return (session, hop, str(row.get("target_key", "")))
    return (10 ** 9, -1, str(row.get("target_key", "")))


def _ensure_columns(conn, table, columns):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, ddl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")


def _fmt_edge(value):
    return f"{float(value):g}"


def _bin_specs(edges):
    """分箱边界 → ``[(low, high, label)]``；``None`` 表示开放端。"""
    specs = []
    for low, high in zip(edges, edges[1:]):
        if low is None and high is None:
            label = "全部"
        elif low is None:
            label = f"< {_fmt_edge(high)}"
        elif high is None:
            label = f"≥ {_fmt_edge(low)}"
        else:
            label = f"[{_fmt_edge(low)}, {_fmt_edge(high)})"
        specs.append((low, high, label))
    return specs


class _null_context:
    """把已有连接包装成上下文（不关闭、不提交，由外层负责）。"""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, tb):
        return False
