from __future__ import annotations

import csv
from datetime import datetime, UTC
from html import escape
import io
import json
from pathlib import Path
from typing import Any

from .schema import (
    EvalReport,
    RUN_MANIFEST_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
)

_NUMERIC_FIELDS = (
    "llm_calls",
    "estimated_prompt_tokens",
    "model_total_ms",
    "tool_calls",
    "tool_errors",
    "steps",
    "memory_retrieval_count",
    "memory_injected_count",
    "memory_tool_search_count",
    "memory_injected_tokens",
    "memory_retrieval_latency_ms",
)


def write_report_files(
    report: EvalReport,
    *,
    started_at: datetime,
    completed_at: datetime | None = None,
    suite_name: str | None = None,
    task_source: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    report.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = report.output_dir / "report.json"
    html_path = report.output_dir / "report.html"
    csv_path = report.output_dir / "report.csv"
    manifest_path = report.output_dir / "run_manifest.json"
    json_path.write_text(
        json.dumps(report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path.write_text(report_to_html(report), encoding="utf-8")
    csv_path.write_text(write_csv_report(report), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            run_manifest_to_dict(
                report,
                started_at=started_at,
                completed_at=completed_at or datetime.now(UTC),
                suite_name=suite_name,
                task_source=task_source,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _update_run_history_index(
        report,
        manifest_path=manifest_path,
        started_at=started_at,
        completed_at=completed_at or datetime.now(UTC),
        suite_name=suite_name,
        task_source=task_source,
    )
    return manifest_path, json_path, html_path, csv_path


def write_csv_report(report: EvalReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "task_id",
            "trial_id",
            "success",
            "graders_pass",
            "graders_total",
            "graders_skipped",
            "tool_calls",
            "tool_errors",
            "llm_calls",
            "tokens",
            "model_ms",
        ]
    )
    for trial in report.trials:
        evaluated_graders = [grader for grader in trial.graders if not grader.skipped]
        g_pass = sum(1 for grader in evaluated_graders if grader.passed)
        g_total = len(evaluated_graders)
        g_skipped = sum(1 for grader in trial.graders if grader.skipped)
        writer.writerow(
            [
                trial.task_id,
                trial.trial_id,
                trial.success,
                g_pass,
                g_total,
                g_skipped,
                trial.metrics.get("tool_calls", ""),
                trial.metrics.get("tool_errors", ""),
                trial.metrics.get("llm_calls", ""),
                trial.metrics.get("estimated_prompt_tokens", ""),
                trial.metrics.get("model_total_ms", ""),
            ]
        )
    writer.writerow([])
    writer.writerow(["run_id", report.run_id, "success", report.success])
    writer.writerow(["schema_version", report.schema_version])
    for key, value in report.metrics.items():
        if isinstance(value, dict):
            continue
        writer.writerow([key, value])
    return buf.getvalue()


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "run_id": report.run_id,
        "success": report.success,
        "output_dir": str(report.output_dir),
        "metrics": report.metrics,
        "tasks": [task.model_dump(exclude_none=True) for task in report.tasks],
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
                        "skipped": grader.skipped,
                        "details": grader.details,
                        "score": grader.score,
                        "required": grader.required,
                        "weight": grader.weight,
                        "evidence": grader.evidence,
                        "failure_category": grader.failure_category,
                    }
                    for grader in trial.graders
                ],
            }
            for trial in report.trials
        ],
    }


def run_manifest_to_dict(
    report: EvalReport,
    *,
    started_at: datetime,
    completed_at: datetime,
    suite_name: str | None,
    task_source: str | None,
) -> dict[str, Any]:
    provider_models = sorted(
        {
            str(trial.metrics.get("provider_model"))
            for trial in report.trials
            if trial.metrics.get("provider_model")
        }
    )
    provider_types = sorted(
        {
            str(trial.metrics.get("provider_type"))
            for trial in report.trials
            if trial.metrics.get("provider_type")
        }
    )
    agent_configs = [
        trial.metrics.get("agent_config")
        for trial in report.trials
        if trial.metrics.get("agent_config") is not None
    ]
    total_wall_clock_ms = round(
        (completed_at - started_at).total_seconds() * 1000,
        1,
    )
    metrics = report.metrics
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "report_schema_version": report.schema_version,
        "run_id": report.run_id,
        "suite_name": suite_name,
        "task_source": task_source,
        "started_at": started_at.astimezone(UTC).isoformat(),
        "completed_at": completed_at.astimezone(UTC).isoformat(),
        "task_count": len(report.tasks),
        "trial_count": len(report.trials),
        "provider_models": provider_models or ["unavailable"],
        "provider_types": provider_types or ["unavailable"],
        "agent_config": agent_configs[0] if agent_configs else "unavailable",
        "task_ids": [task.id for task in report.tasks],
        "wall_clock_ms": total_wall_clock_ms,
        "tokens": _manifest_metric_or_unavailable(metrics, "total_estimated_tokens"),
        "model_latency_ms": _manifest_metric_or_unavailable(metrics, "total_model_ms"),
        "tool_calls": _manifest_metric_or_unavailable(metrics, "total_tool_calls"),
        "tool_errors": _manifest_metric_or_unavailable(metrics, "total_tool_errors"),
        "avg_tool_latency_ms": _manifest_distribution_mean_or_unavailable(
            metrics,
            "tool_total_ms_distribution",
        ),
        "avg_model_latency_ms": _manifest_distribution_mean_or_unavailable(
            metrics,
            "model_total_ms_distribution",
        ),
        "termination_reasons": _manifest_termination_reasons_or_unavailable(report),
        "token_cost": "unavailable",
    }


def _manifest_metric_or_unavailable(metrics: dict[str, Any], key: str) -> Any:
    value = metrics.get(key)
    return value if value not in (None, "") else "unavailable"


def _manifest_distribution_mean_or_unavailable(
    metrics: dict[str, Any],
    key: str,
) -> Any:
    value = metrics.get(key)
    if not isinstance(value, dict):
        return "unavailable"
    mean = value.get("mean")
    return mean if mean is not None else "unavailable"


def _manifest_termination_reasons_or_unavailable(report: EvalReport) -> Any:
    counts: dict[str, int] = {}
    for trial in report.trials:
        reason = trial.metrics.get("termination_reason")
        if not isinstance(reason, str) or not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts or "unavailable"


def report_to_html(report: EvalReport) -> str:
    rows = "\n".join(_trial_row(trial) for trial in report.trials)
    status = "PASS" if report.success else "FAIL"
    metrics = report.metrics
    passed = metrics.get("passed_trials", 0)
    total = metrics.get("trial_count", len(report.trials))
    grader_rate = metrics.get("grader_pass_rate")
    grader_pct = f"{grader_rate * 100:.1f}%" if grader_rate is not None else "—"
    pass_at_k = metrics.get("pass@k", "—")
    pass_k_rate = metrics.get("pass@k_rate")
    pass_k_pct = f"{pass_k_rate * 100:.0f}%" if pass_k_rate is not None else ""
    pass_pow_k = metrics.get("pass^k", "—")
    pass_pow_rate = metrics.get("pass^k_rate")
    pass_pow_pct = f"{pass_pow_rate * 100:.0f}%" if pass_pow_rate is not None else ""
    total_llm = metrics.get("total_llm_calls", 0)
    total_tokens = metrics.get("total_estimated_tokens", 0)
    total_model_ms = metrics.get("total_model_ms", 0.0)
    total_tool_calls = metrics.get("total_tool_calls", 0)
    total_tool_errors = metrics.get("total_tool_errors", 0)
    grader_table = _grader_summary_table(metrics.get("per_grader_pass_rate", {}))
    dist_table = _distribution_table_html(metrics)
    memory_ablation_table = _memory_ablation_table_html(metrics)
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
    .skip {{ color: #656d76; font-weight: 600; }}
    code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; margin: 0; max-height: 160px; overflow: auto; }}
    .metric-pill {{ display: inline-block; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 12px; padding: 2px 10px; font-size: 12px; margin: 2px; }}
    .evidence-contains {{ color: #116329; }}
    .evidence-missing {{ color: #cf222e; }}
    .grader-bar {{ display: inline-block; height: 10px; border-radius: 3px; vertical-align: middle; }}
    .grader-bar-pass {{ background: #116329; }}
    .grader-bar-fail {{ background: #cf222e; }}
  </style>
</head>
<body>
  <h1>Xcode Eval Report</h1>
  <div>Run ID: <code>{escape(report.run_id)}</code></div>
  <div>Schema Version: <code>{report.schema_version}</code></div>
  <div class="summary">
    <div class="card"><div class="label">Status</div><div class="value {status.lower()}">{status}</div></div>
    <div class="card"><div class="label">pass@k</div><div class="value">{pass_at_k} {pass_k_pct}</div></div>
    <div class="card"><div class="label">pass^k</div><div class="value">{pass_pow_k} {pass_pow_pct}</div></div>
    <div class="card"><div class="label">Trials</div><div class="value">{passed}/{total}</div></div>
    <div class="card"><div class="label">Tasks</div><div class="value">{metrics.get("task_count", 0)}</div></div>
    <div class="card"><div class="label">Grader Pass Rate</div><div class="value">{grader_pct}</div></div>
    <div class="card"><div class="label">Tool Calls</div><div class="value">{total_tool_calls}</div></div>
    <div class="card"><div class="label">Tool Errors</div><div class="value">{total_tool_errors}</div></div>
    <div class="card"><div class="label">LLM Calls</div><div class="value">{total_llm}</div></div>
    <div class="card"><div class="label">Est. Tokens</div><div class="value">{total_tokens:,}</div></div>
    <div class="card"><div class="label">Model Latency</div><div class="value">{_fmt_ms(total_model_ms)}</div></div>
  </div>
  {grader_table}
  {dist_table}
  {memory_ablation_table}
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
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _grader_summary_table(per_grader: dict[str, float]) -> str:
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
        css_class = "pass" if rate >= 1.0 else "fail"
        rows.append(
            f"<tr><td>{escape(name)}</td>"
            f'<td class="{css_class}">{pct}</td>'
            f"<td>{bar_html}</td></tr>"
        )
    return (
        "<h2>Grader Summary</h2>"
        "<table><thead><tr><th>Grader</th><th>Pass Rate</th><th>Chart</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _trial_row(trial) -> str:
    status = "PASS" if trial.success else "FAIL"
    grader_lines: list[str] = []
    for grader in trial.graders:
        if grader.skipped:
            css_class = "skip"
            label = "SKIP"
        elif grader.passed:
            css_class = "pass"
            label = "PASS"
        else:
            css_class = "fail"
            label = "FAIL"
        grader_lines.append(
            f'<span class="{css_class}">{label}</span> '
            f"{escape(grader.name)}"
            f"{': ' + escape(grader.details) if grader.details else ''}"
        )
    graders = "<br>".join(grader_lines)
    skip_keys = {
        "model_latencies_ms",
        "tool_latencies_ms",
        "file_evidence",
        "memory_trace",
        "agent_config",
    }
    pills = []
    for key, value in sorted(trial.metrics.items()):
        if key in skip_keys:
            continue
        display = _fmt_metric_value(key, value)
        pills.append(
            f'<span class="metric-pill">{escape(key)}: {escape(display)}</span>'
        )
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


def _fmt_metric_value(key: str, value: str | int | float) -> str:
    if value == "unavailable":
        return "unavailable"
    if key in (
        "model_total_ms",
        "tool_total_ms",
        "total_observed_ms",
        "memory_retrieval_latency_ms",
    ):
        return _fmt_ms(float(value))
    if key in {"estimated_prompt_tokens", "memory_injected_tokens"}:
        return f"{int(value):,}"
    if key.startswith("memory_") and key.endswith(("_rate", "_mrr", "_at_k")):
        return f"{float(value) * 100:.1f}%"
    return str(value)


def _file_evidence_html(evidence: list | None) -> str:
    if not evidence:
        return ""
    parts = []
    for ev in evidence:
        path = ev.get("path", "?")
        contains = ev.get("contains", {})
        not_contains = ev.get("not_contains", {})
        items = []
        for key, present in contains.items():
            css_class = "evidence-contains" if present else "evidence-missing"
            mark = "✓" if present else "✗"
            items.append(f'<span class="{css_class}">{mark} {escape(key)}</span>')
        for key, absent in not_contains.items():
            css_class = "evidence-contains" if absent else "evidence-missing"
            mark = "✓" if absent else "✗"
            items.append(f'<span class="{css_class}">{mark} !{escape(key)}</span>')
        if items:
            parts.append(f"<b>{escape(path)}</b>: " + " ".join(items))
    return "<br>".join(parts) if parts else ""


def _distribution_table_html(metrics: dict[str, Any]) -> str:
    rows = []
    for key in _NUMERIC_FIELDS:
        dist = metrics.get(f"{key}_distribution")
        if not dist:
            continue
        bar_w = int(dist.get("p50", 0) / max(dist.get("max", 1), 1) * 100)
        bar_w = min(bar_w, 100)
        label = key.replace("_", " ")
        rows.append(
            f"<tr><td>{escape(label)}</td>"
            f"<td>{dist['min']}</td>"
            f"<td>{dist['p50']}</td>"
            f"<td>{dist['p95']}</td>"
            f"<td>{dist['p99']}</td>"
            f"<td>{dist['max']}</td>"
            f"<td>{dist['mean']}</td>"
            f'<td><span class="grader-bar grader-bar-pass" style="width:{bar_w}px"></span></td>'
            f"</tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Distribution (p50/p95/p99)</h2>"
        "<table><thead><tr><th>Metric</th><th>Min</th><th>p50</th><th>p95</th><th>p99</th><th>Max</th><th>Mean</th><th>p50→max</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _memory_ablation_table_html(metrics: dict[str, Any]) -> str:
    pair_count = metrics.get("memory_ablation_pair_count")
    if not pair_count:
        return ""
    rows = [
        ("Pairs", str(pair_count)),
        ("Memory On Success", _fmt_ratio(metrics.get("memory_on_success_rate"))),
        ("Memory Off Success", _fmt_ratio(metrics.get("memory_off_success_rate"))),
        ("Success Delta", _fmt_ratio(metrics.get("memory_success_delta"))),
        ("Tool Call Delta", str(metrics.get("memory_tool_call_delta_mean", "—"))),
        (
            "Negative Migration",
            _fmt_ratio(metrics.get("memory_negative_migration_rate")),
        ),
    ]
    body = "".join(
        f"<tr><td>{escape(label)}</td><td>{escape(value)}</td></tr>"
        for label, value in rows
    )
    return (
        "<h2>Memory On/Off</h2>"
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _fmt_ratio(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value) * 100:.1f}%"


def _update_run_history_index(
    report: EvalReport,
    *,
    manifest_path: Path,
    started_at: datetime,
    completed_at: datetime,
    suite_name: str | None,
    task_source: str | None,
) -> None:
    history_dir = report.output_dir.parent
    history_dir.mkdir(parents=True, exist_ok=True)
    index_path = history_dir / "run_index.jsonl"
    trend_path = history_dir / "trend_summary.json"
    entry = {
        "run_id": report.run_id,
        "suite_name": suite_name or "ad-hoc",
        "task_source": task_source or "unknown",
        "started_at": started_at.astimezone(UTC).isoformat(),
        "completed_at": completed_at.astimezone(UTC).isoformat(),
        "success": report.success,
        "task_count": len(report.tasks),
        "trial_count": len(report.trials),
        "pass@k_rate": report.metrics.get("pass@k_rate", "unavailable"),
        "pass^k_rate": report.metrics.get("pass^k_rate", "unavailable"),
        "grader_pass_rate": report.metrics.get("grader_pass_rate", "unavailable"),
        "total_model_ms": report.metrics.get("total_model_ms", "unavailable"),
        "total_estimated_tokens": report.metrics.get(
            "total_estimated_tokens", "unavailable"
        ),
        "manifest_path": str(manifest_path),
        "report_path": str(report.output_dir / "report.json"),
    }
    with index_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    trend_path.write_text(
        json.dumps(_build_trend_summary(index_path), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_trend_summary(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {"runs": 0, "by_suite": {}}
    rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        by_suite.setdefault(str(row.get("suite_name", "ad-hoc")), []).append(row)
    summary: dict[str, Any] = {"runs": len(rows), "by_suite": {}}
    for suite_name, entries in sorted(by_suite.items()):
        latest = entries[-5:]
        summary["by_suite"][suite_name] = {
            "runs": len(entries),
            "recent_run_ids": [str(entry.get("run_id", "")) for entry in latest],
            "recent_pass@k_rate_mean": _mean_available(latest, "pass@k_rate"),
            "recent_grader_pass_rate_mean": _mean_available(latest, "grader_pass_rate"),
        }
    return summary


def _mean_available(rows: list[dict[str, Any]], key: str) -> Any:
    values = [
        float(value) for row in rows if isinstance((value := row.get(key)), int | float)
    ]
    if not values:
        return "unavailable"
    return round(sum(values) / len(values), 4)
