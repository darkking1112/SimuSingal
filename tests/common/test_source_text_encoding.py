"""文本 IO 必须显式指定编码的全局守卫。

中文 Windows 上 ``Path.read_text()`` / ``write_text()`` 默认用 GBK，
而本仓库所有产品文件（JSON / YAML / 日志）都按 UTF-8 读写；两边一混就报
``UnicodeDecodeError``。这类问题只会在中文系统上偶发（字节恰好可被 GBK
解码时还会"碰巧通过"），曾经在训练链路炸过一次 ``data.yaml``（含中文注释）。

因此这里对所有会被用户直接运行的目录做静态守卫：``read_text`` / ``write_text``
调用必须带 ``encoding`` 关键字（或明确的位置参数）。二进制读写（bytes）不在此列。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCANNED = ("src", "training", "scripts")


def _text_io_offenders(path):
    """返回 [(行号, 方法名)]：未指定 encoding 的 read_text / write_text 调用。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("read_text", "write_text")):
            keywords = {item.arg for item in node.keywords}
            if "encoding" not in keywords and not node.args:
                yield node.lineno, node.func.attr


def test_text_io_calls_declare_encoding():
    offenders = []
    for folder in SCANNED:
        for path in sorted((ROOT / folder).rglob("*.py")):
            for lineno, attr in _text_io_offenders(path):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}:{attr}()")
    assert not offenders, (
        "以下文本读写未显式指定 encoding=\"utf-8\"（中文 Windows 默认 GBK 会读坏 UTF-8 文件）：\n"
        + "\n".join(offenders))
