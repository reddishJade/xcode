from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any

from .schema import EvalReport


def write_report_files(report: EvalReport) -> tuple[Path, Path]:
    """写入机器可读 JSON 和可浏览 HTML 报告。"""
    report.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = report.output_dir / "report.json"
    html_path = report.output_dir / "report.html"
    json_path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(report_to_html(report), encoding="utf-8")
    return json_path, html_path


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "success": report.success,
        "output_dir": str(report.output_dir),
        "metrics": report.metrics,
        "trials": [
            {
                "task_id": trial.task_id,
                "trial_id": trial.trial_id,
                "success": trial.success,
                "answer": trial.answer,
                "trace_path": str(trial.trace_path),
                "metrics": trial.metrics,
                "graders": [
                    {
                        "name": grader.name,
                        "passed": grader.passed,
                        "details": grader.details,
                    }
                    for grader in trial.graders
                ],
            }
            for trial in report.trials
        ],
    }


def report_to_html(report: EvalReport) -> str:
    rows = "\n".join(_trial_row(trial) for trial in report.trials)
    status = "PASS" if report.success else "FAIL"
    m = report.metrics
    passed = m.get("passed_trials", 0)
    total = m.get("trial_count", len(report.trials))
    grader_rate = m.get("grader_pass_rate")
    grader_pct = f"{grader_rate * 100:.1f}%" if grader_rate is not None else "—"
    total_llm = m.get("total_llm_calls", 0)
    total_tokens = m.get("total_estimated_tokens", 0)
    total_model_ms = m.get("total_model_ms", 0.0)
    grader_table = _grader_summary_table(m.get("per_grader_pass_rate", {}))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Xcode Eval Report {escape(report.run_id)}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #202124; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin: 28px 0 12px; }}
    .summary {{ display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }}
    .card {{ border: 1px solid #d0d7de; border-radius: 6px; padding: 12px 16px; min-width: 120px; }}
    .label {{ color: #57606a; font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #d8dee4; text-align: left; padding: 10px; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    .pass {{ color: #116329; font-weight: 600; }}
    .fail {{ color: #cf222e; font-weight: 600; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; margin: 0; max-height: 160px; overflow: auto; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }}
    .badge-pass {{ background: #dafbe1; color: #116329; }}
    .badge-fail {{ background: #ffebe9; color: #cf222e; }}
    .metric-pill {{ display: inline-block; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 12px; padding: 2px 10px; font-size: 12px; margin: 2px; }}
    .evidence-contains {{ color: #116329; }}
    .evidence-missing {{ color: #cf222e; }}
    .grader-summary {{ margin-top: 8px; }}
    .grader-bar {{ display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }}
    .grader-bar-pass {{ background: #116329; }}
    .grader-bar-fail {{ background: #cf222e; }}
  </style>
</head>
<body>
  <h1>Xcode Eval Report</h1>
  <div>Run ID: <code>{escape(report.run_id)}</code></div>
  <div class="summary">
    <div class="card"><div class="label">Status</div><div class="value {status.lower()}">{status}</div></div>
    <div class="card"><div class="label">Trials</div><div class="value">{passed}/{total}</div></div>
    <div class="card"><div class="label">Tasks</div><div class="value">{m.get("task_count", 0)}</div></div>
    <div class="card"><div class="label">Grader Pass Rate</div><div class="value">{grader_pct}</div></div>
    <div class="card"><div class="label">LLM Calls</div><div class="value">{total_llm}</div></div>
    <div class="card"><div class="label">Est. Tokens</div><div class="value">{total_tokens:,}</div></div>
    <div class="card"><div class="label">Model Latency</div><div class="value">{_fmt_ms(total_model_ms)}</div></div>
  </div>
  {grader_table}
  <h2>Trials</h2>
  <table>
    <thead>
      <tr><th>Trial</th><th>Status</th><th>Metrics</th><th>Graders</th><th>Trace</th><th>Answer</th></tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</body>
</html>
"""


def _fmt_ms(ms: float) -> str:
    """毫秒格式化为人类可读字符串。"""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _grader_summary_table(per_grader: dict[str, float]) -> str:
    """生成 grader 维度的通过率汇总表。"""
    if not per_grader:
        return ""
    rows = []
    for name, rate in sorted(per_grader.items()):
        pct = f"{rate * 100:.1f}%"
        bar_w = int(rate * 100)
        bar_html = (
            f'<span class="grader-bar grader-bar-pass" style="width:{bar_w}px"></span>'
            f'<span class="grader-bar grader-bar-fail" style="width:{100 - bar_w}px"></span>'
        )
        cls = "pass" if rate >= 1.0 else "fail"
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f'<td class="{cls}">{pct}</td>'
            f"<td>{bar_html}</td></tr>"
        )
    return (
        "<h2>Grader Summary</h2>"
        "<table><thead><tr><th>Grader</th><th>Pass Rate</th><th>Chart</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _trial_row(trial) -> str:
    status = "PASS" if trial.success else "FAIL"
    # grader 列表
    graders = "<br>".join(
        f'<span class="{"pass" if grader.passed else "fail"}">'
        f"{escape('PASS' if grader.passed else 'FAIL')}</span> "
        f"{escape(grader.name)}"
        f"{': ' + escape(grader.details) if grader.details else ''}"
        for grader in trial.graders
    )
    # 指标展示：过滤掉大数组字段，格式化延迟
    skip_keys = {"model_latencies_ms", "tool_latencies_ms", "file_evidence"}
    pills = []
    for key, value in sorted(trial.metrics.items()):
        if key in skip_keys:
            continue
        display = _fmt_metric_value(key, value)
        pills.append(f'<span class="metric-pill">{escape(key)}: {escape(display)}</span>')
    # 文件证据
    evidence_html = _file_evidence_html(trial.metrics.get("file_evidence"))
    metrics_html = " ".join(pills)
    if evidence_html:
        metrics_html += "<br>" + evidence_html
    return (
        "<tr>"
        f"<td>{escape(trial.trial_id)}</td>"
        f'<td class="{status.lower()}">{status}</td>'
        f"<td>{metrics_html}</td>"
        f"<td>{graders}</td>"
        f"<td><code>{escape(str(trial.trace_path))}</code></td>"
        f"<td><pre>{escape(trial.answer)}</pre></td>"
        "</tr>"
    )


def _fmt_metric_value(key: str, value: object) -> str:
    """格式化单个指标值用于展示。"""
    if key in ("model_total_ms", "tool_total_ms", "total_observed_ms"):
        return _fmt_ms(float(value))
    if key == "estimated_prompt_tokens":
        return f"{int(value):,}"
    return str(value)


def _file_evidence_html(evidence: list | None) -> str:
    """将文件证据渲染为可读的 HTML 片段。"""
    if not evidence:
        return ""
    parts = []
    for ev in evidence:
        path = ev.get("path", "?")
        contains = ev.get("contains", {})
        not_contains = ev.get("not_contains", {})
        changed = ev.get("sha1")
        items = []
        for k, v in contains.items():
            cls = "evidence-contains" if v else "evidence-missing"
            mark = "✓" if v else "✗"
            items.append(f'<span class="{cls}">{mark} {escape(k)}</span>')
        for k, v in not_contains.items():
            cls = "evidence-contains" if v else "evidence-missing"
            mark = "✓" if v else "✗"
            items.append(f'<span class="{cls}">{mark} !{escape(k)}</span>')
        if items:
            parts.append(f"<b>{escape(path)}</b>: " + " ".join(items))
    return "<br>".join(parts) if parts else ""
