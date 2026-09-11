"""Standalone escaped HTML or JSON report export.

HTML 报告在通用 JSON 转储之外，额外把**已冻结契约里的可比较指标**排成表格：检测类
结果给出“检测 / 传统基线”并排的评测表，调制识别结果给出预测与真值命中表。所有文本
都经 ``html.escape`` 转义，原始 JSON 仍完整保留在页尾，便于机器复用。
"""

import html
import json
from pathlib import Path

_KIND_LABELS = {"analysis": "数据分析", "detect": "信号检测", "ml_detect": "AI 信号检测",
                "amc_classify": "调制识别", "generate": "IQ 生成", "native": "原生插件"}

# (显示名, 指标键, 格式化函数)；键不存在或为 None 时显示 “--”，不写 0 以免误读
_DETECTION_FIELDS = (
    ("真实目标数", "true", lambda v: f"{int(v)}"),
    ("匹配", "matched", lambda v: f"{int(v)}"),
    ("漏警", "missed", lambda v: f"{int(v)}"),
    ("虚警", "false_alarm", lambda v: f"{int(v)}"),
    ("精确率", "precision", lambda v: f"{v:.4g}"),
    ("召回", "recall", lambda v: f"{v:.4g}"),
    ("F1", "f1", lambda v: f"{v:.4g}"),
    ("中心频率 MAE / Hz", "center_mae_hz", lambda v: f"{v:.1f}"),
    ("带宽相对误差", "bandwidth_mape", lambda v: f"{v:.3f}"),
    ("带内信噪比 MAE / dB", "snr_mae_db", lambda v: f"{v:.2f}"),
)

_AMC_FIELDS = (
    ("预测类别", "label", str),
    ("类别名称", "name", str),
    ("置信度", "confidence", lambda v: f"{v:.4f}"),
    ("置信度差（首选-次选）", "margin", lambda v: f"{v:.4f}"),
    ("可靠", "reliable", lambda v: "是" if v else "否"),
    ("判定说明", "reason", str),
)

_STYLE = ("body{font:16px sans-serif;max-width:1100px;margin:40px auto;padding:20px;"
          "color:#183044;background:#f5f8fb}"
          "h1{font-size:24px}h2{font-size:18px;margin-top:28px}"
          "table{border-collapse:collapse;background:white;width:100%;margin:8px 0}"
          "th,td{border:1px solid #d7e1ea;padding:6px 10px;text-align:left}"
          "th{background:#e9f1f8}"
          ".note{color:#5b7286}"
          "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px}")


def _cell(field, metrics):
    """按 ``(显示名, 键, 格式)`` 取一格文本，缺值显示 “--”。"""
    name, key, formatter = field
    if not isinstance(metrics, dict) or metrics.get(key) is None:
        return name, "--"
    try:
        return name, formatter(metrics[key])
    except (TypeError, ValueError):
        return name, "--"


def _table(headers, rows):
    head = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
                   for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def detection_metrics(metrics):
    """检测评测指标整理成 ``(显示名, 文本)`` 列表，供 HTML 报告与界面并排展示。"""
    return [_cell(field, metrics) for field in _DETECTION_FIELDS]


def amc_metrics(prediction):
    """调制识别预测字段整理成 ``(显示名, 文本)`` 列表。"""
    return [_cell(field, prediction) for field in _AMC_FIELDS]


def _detection_table(result):
    """检测类结果：检测算法与传统基线并排（缺哪一列就不显示哪一列）。"""
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        return ""
    columns = [("检测结果", metrics)]
    baseline = result.get("baseline_metrics")
    if isinstance(baseline, dict):
        columns.append(("传统基线", baseline))
    rows = [[_cell(field, columns[0][1])[0]] + [_cell(field, column)[1] for _, column in columns]
            for field in _DETECTION_FIELDS]
    text = ["<h2>检测评测（与生成器真值逐项对照）</h2>",
            _table(["指标"] + [label for label, _ in columns], rows)]
    if len(columns) == 2:
        text.append("<p class='note'>同一份数据、同一套会话合并口径下的并排结果："
                    "“传统基线”指能量检测路径，可与 AI 检测直接比较。</p>")
    return "".join(text)


def _amc_table(result):
    """调制识别结果：预测、置信度与生成器真值命中。"""
    prediction = result.get("prediction")
    if not isinstance(prediction, dict):
        return ""
    rows = [[name, value] for field in _AMC_FIELDS for name, value in [_cell(field, prediction)]]
    parts = ["<h2>调制识别结果</h2>", _table(["项目", "取值"], rows)]
    truth = result.get("truth")
    if isinstance(truth, dict) and truth.get("available"):
        hit = result.get("truth_hit")
        snr = truth.get("snr_inband_db")
        rows = [["真值样式", str(truth.get("mode"))],
                ["真值类别", str(truth.get("class"))],
                ["识别命中", "--" if hit is None else ("命中" if hit else "未命中")],
                ["真值带内信噪比 / dB", "--" if snr is None else f"{float(snr):.2f}"]]
        parts.append(_table(["项目", "取值"], rows))
    else:
        reason = (truth or {}).get("reason") if isinstance(truth, dict) else None
        parts.append("<p class='note'>真值：" + html.escape(str(reason or "不适用"))
                     + "（按“不适用”计数，不丢弃样本）</p>")
    pending = result.get("pending")
    if isinstance(pending, list) and pending:
        items = "".join(f"<li>{html.escape(str(item))}</li>" for item in pending)
        parts.append("<h2>尚未确认项</h2><ul>" + items + "</ul>")
    return "".join(parts)


def _metrics_section(result):
    """按结果类型挑选可展示的指标表；没有指标就返回空串。"""
    kind = result.get("kind")
    if kind in ("detect", "ml_detect"):
        return _detection_table(result)
    if kind == "amc_classify":
        return _amc_table(result)
    return ""


def export_report(result, destination):
    destination = Path(destination)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
    if destination.suffix.lower() == ".json":
        text = payload
    elif destination.suffix.lower() == ".html":
        kind = str(result.get("kind", "实验"))
        title = html.escape(f"SimuSignal · {_KIND_LABELS.get(kind, kind)} · {result.get('run_id', '')}")
        text = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'>"
                f"<title>{title}</title><style>{_STYLE}</style><h1>{title}</h1>"
                "<p class='note'>通用离线实验记录：指标表由已冻结契约中的字段生成，"
                "页尾保留完整原始 JSON。</p>"
                f"{_metrics_section(result)}"
                f"<h2>原始结果</h2><pre>{html.escape(payload)}</pre></html>")
    else:
        raise ValueError("报告后缀应为 .json 或 .html")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)
    return str(destination.resolve())
