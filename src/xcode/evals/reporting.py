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
    passed = report.metrics.get("passed_trials", 0)
    total = report.metrics.get("trial_count", len(report.trials))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Xcode Eval Report {escape(report.run_id)}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; color: #202124; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    .summary {{ display: flex; gap: 12px; margin: 20px 0; }}
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
  </style>
</head>
<body>
  <h1>Xcode Eval Report</h1>
  <div>Run ID: <code>{escape(report.run_id)}</code></div>
  <div class="summary">
    <div class="card"><div class="label">Status</div><div class="value {status.lower()}">{status}</div></div>
    <div class="card"><div class="label">Trials</div><div class="value">{passed}/{total}</div></div>
    <div class="card"><div class="label">Tasks</div><div class="value">{report.metrics.get("task_count", 0)}</div></div>
  </div>
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


def _trial_row(trial) -> str:
    status = "PASS" if trial.success else "FAIL"
    graders = "<br>".join(
        f'<span class="{"pass" if grader.passed else "fail"}">'
        f"{escape('PASS' if grader.passed else 'FAIL')}</span> "
        f"{escape(grader.name)}"
        f"{': ' + escape(grader.details) if grader.details else ''}"
        for grader in trial.graders
    )
    metrics = "<br>".join(
        f"{escape(str(key))}: {escape(str(value))}"
        for key, value in sorted(trial.metrics.items())
    )
    return (
        "<tr>"
        f"<td>{escape(trial.trial_id)}</td>"
        f'<td class="{status.lower()}">{status}</td>'
        f"<td>{metrics}</td>"
        f"<td>{graders}</td>"
        f"<td><code>{escape(str(trial.trace_path))}</code></td>"
        f"<td><pre>{escape(trial.answer)}</pre></td>"
        "</tr>"
    )
