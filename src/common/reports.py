"""Standalone escaped HTML or JSON report export."""

import html
import json
from pathlib import Path


def export_report(result, destination):
    destination = Path(destination)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if destination.suffix.lower() == ".json":
        text = payload
    elif destination.suffix.lower() == ".html":
        title = html.escape(f"SimuSignal · {result.get('kind', '实验')} · {result.get('run_id', '')}")
        text = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
                f"<title>{title}</title><style>body{{font:16px sans-serif;max-width:1100px;"
                "margin:40px auto;padding:20px;color:#183044;background:#f5f8fb}}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px}"
                f"</style><h1>{title}</h1><p>通用离线实验记录</p><pre>{html.escape(payload)}</pre></html>")
    else:
        raise ValueError("报告后缀应为 .json 或 .html")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return str(destination.resolve())
