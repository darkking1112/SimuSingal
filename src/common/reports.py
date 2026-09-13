"""Standalone escaped HTML or JSON report export.

HTML 报告在通用 JSON 转储之外，额外把**已冻结契约里的可比较指标**排成表格：检测类
结果给出“检测 / 传统基线”并排的评测表，逐跳参数估计给出“逐跳 / 会话”并排的评测表
加会话汇总与逐跳明细，调制识别结果给出预测与真值命中表。所有文本都经 ``html.escape``
转义，原始 JSON 仍完整保留在页尾，便于机器复用。
"""

import html
import json
from pathlib import Path

_KIND_LABELS = {"analysis": "数据分析", "detect": "信号检测", "ml_detect": "AI 信号检测",
                "detect_hops": "跳频参数估计", "ml_detect_hops": "AI 跳频参数估计",
                "amc_classify": "调制识别", "amc_iq_classify": "原始 IQ 调制识别",
                "generate": "IQ 生成", "native": "原生插件"}

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

# 逐跳参数（fh_hops_v1）：一跳一行的可比字段
_HOP_FIELDS = (
    ("跳号", "id", lambda v: f"{int(v)}"),
    ("会话", "session_id", lambda v: "--" if v is None else f"{int(v)}"),
    ("中心频率 / Hz", "center_hz", lambda v: f"{v:.1f}"),
    ("单跳带宽 / Hz", "bandwidth_hz", lambda v: f"{v:.1f}"),
    ("频段下限 / Hz", "f_low_hz", lambda v: f"{v:.1f}"),
    ("频段上限 / Hz", "f_high_hz", lambda v: f"{v:.1f}"),
    ("起始 / s", "t_start_s", lambda v: f"{v:.5f}"),
    ("结束 / s", "t_end_s", lambda v: f"{v:.5f}"),
    ("驻留 / ms", "dwell_s", lambda v: f"{v * 1e3:.3f}"),
    ("功率 / dBFS", "power_dbfs", lambda v: f"{v:.2f}"),
    ("逐跳 SNR / dB", "snr_db", lambda v: f"{v:.2f}"),
    ("置信度", "confidence", lambda v: f"{v:.2f}"),
)

# 会话级汇总：跳速与跳频点集合是这一层的关键结论
_SESSION_FIELDS = (
    ("会话", "session_id", lambda v: f"{int(v)}"),
    ("跳数", "hop_count", lambda v: f"{int(v)}"),
    ("跳速 / Hz", "hop_rate_hz", lambda v: f"{v:.3f}"),
    ("跳周期 / s", "hop_period_s", lambda v: f"{v:.6f}"),
    ("驻留中位 / s", "dwell_median_s", lambda v: f"{v:.6f}"),
    ("占空比", "duty_cycle", lambda v: f"{v:.3f}"),
    ("跳频点数", "hop_frequencies_hz", lambda v: f"{len(v)}"),
    ("跳频跨度 / Hz", "hop_span_hz", lambda v: f"{v:.1f}"),
    ("会话带宽 / Hz", "bandwidth_hz", lambda v: f"{v:.1f}"),
    ("会话带内 SNR / dB", "snr_db", lambda v: f"{v:.2f}"),
    ("会话级检出", "session_detection_id", lambda v: "--" if v is None else f"{int(v)}"),
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


def _hop_table(result):
    """逐跳参数估计：逐跳与会话两种口径的评测表，加会话汇总与逐跳明细。

    传统路径给出“逐跳 / 会话”两列；AI 逐跳路径（``ml_detect_hops``）再多一列同配置的
    传统逐跳结果，三列同源于一套物理量口径（驻留、带宽、功率、SNR 都在原始 PSD 上重测），
    因此可以直接并排比较“模型定位 + 重测”与“纯能量逐跳”。
    """
    parts = []
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        first = "AI 逐跳口径" if result.get("kind") == "ml_detect_hops" else "逐跳口径"
        columns = [(first, metrics)]
        traditional = result.get("traditional_metrics")
        if isinstance(traditional, dict):
            columns.append(("传统逐跳口径", traditional))
        baseline = result.get("baseline_metrics")
        if isinstance(baseline, dict):
            columns.append(("会话口径（能量检测基线）", baseline))
        rows = [[_cell(field, columns[0][1])[0]] + [_cell(field, column)[1]
                for _, column in columns] for field in _DETECTION_FIELDS]
        parts.append("<h2>逐跳评测（与生成器逐跳真值对照）</h2>")
        parts.append(_table(["指标"] + [label for label, _ in columns], rows))
        if len(columns) > 1:
            labels = "、".join(html.escape(label) for label, _ in columns)
            parts.append("<p class='note'>同一次分析按不同口径并排给出：" + labels
                         + "。逐跳口径按“一跳一条”匹配，会话口径即能量检测按“一条链路"
                           "一条”匹配，所以真实目标数不同、精确率/召回率也不同——这正是"
                           "逐跳输出要补充的信息。</p>")
    else:
        truth = result.get("truth")
        reason = truth.get("reason") if isinstance(truth, dict) else "没有生成器真值"
        parts.append("<h2>逐跳评测</h2><p class='note'>真值："
                     + html.escape(str(reason or "不适用"))
                     + "（按“不适用”计数，结果与原始 JSON 仍然保留）</p>")
    model = result.get("model")
    if isinstance(model, dict) and model.get("id"):
        parts.append("<p class='note'>逐跳模型："
                     + html.escape(f"{model.get('id')}@{model.get('version') or '-'}")
                     + html.escape(f"，推理运行时 {model.get('runtime_version') or '-'}")
                     + "。模型只负责在时频图上定位跳（频段 + 粗时间），驻留、带宽、"
                       "功率与逐跳 SNR 均在原始 PSD 上用统一门限重新测量。</p>")
    if result.get("reason"):
        parts.append("<p class='note'>提示：" + html.escape(str(result["reason"])) + "</p>")
    sessions = result.get("sessions")
    if isinstance(sessions, list) and sessions:
        rows = [[_cell(field, item)[1] for field in _SESSION_FIELDS] for item in sessions]
        parts.append("<h2>会话与跳参数</h2>")
        parts.append(_table([field[0] for field in _SESSION_FIELDS], rows))
        parts.append("<p class='note'>跳速取“跳起点间隔中位数”的倒数（发射机的跳频速率）；"
                     "驻留时间是一次跳频里信号真实存在的时间，OFDM 图传类样式每跳尾部"
                     "有空闲所以占空比小于 1。</p>")
    hops = result.get("hops")
    if isinstance(hops, list) and hops:
        # AI 逐跳通路的每条明细都带模型分数，传统通路没有 → 只在需要时加这一列
        fields = list(_HOP_FIELDS)
        if any(isinstance(item, dict) and item.get("model_confidence") is not None
               for item in hops):
            fields.append(("模型置信度", "model_confidence", lambda v: f"{v:.3f}"))
        rows = [[_cell(field, item)[1] for field in fields] for item in hops]
        parts.append(f"<h2>逐跳明细（{len(hops)} 跳）</h2>")
        parts.append(_table([field[0] for field in fields], rows))
    else:
        parts.append("<h2>逐跳明细</h2><p class='note'>本帧未检出任何跳：可能不是跳频样式，"
                     "也可能跳速超过当前 STFT 的时间分辨率。</p>")
    return "".join(parts)


def _metrics_section(result):
    """按结果类型挑选可展示的指标表；没有指标就返回空串。"""
    kind = result.get("kind")
    if kind in ("detect", "ml_detect"):
        return _detection_table(result)
    if kind == "detect_hops" or kind == "ml_detect_hops":
        return _hop_table(result)
    if kind == "amc_classify" or kind == "amc_iq_classify":
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
