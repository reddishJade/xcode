"""从封存 Trial artifact 离线重建 Experiment 原始数据和报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .artifacts import ArtifactError, ArtifactStore
from .metrics import aggregate_experiment, TrialRecord
from .schema import Experiment, ExperimentSummary, Trial

EXPERIMENT_FILE = "experiment.json"
TRIALS_FILE = "trials.jsonl"
SUMMARY_FILE = "summary.json"
REPORT_FILE = "report.md"
CHECKSUMS_FILE = "experiment-checksums.json"


class ReportingError(RuntimeError):
    """Experiment 控制 artifact 缺失、损坏或不可重建。"""


class ExperimentArtifactStore:
    """保存 Experiment 声明，并由 Trial artifact 重建所有派生报告。"""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root.resolve()

    def begin(self, experiment: Experiment) -> Path:
        """在任何 Trial 前冻结 Experiment 声明。"""
        root = self._artifact_root / experiment.experiment_id
        declaration = root / EXPERIMENT_FILE
        if declaration.exists():
            existing = self.load_experiment(root, verify=True)
            if existing != experiment:
                raise ReportingError(
                    "experiment id already exists with a different declaration"
                )
            return root
        root.mkdir(parents=True, exist_ok=True)
        _write_json(declaration, experiment)
        _write_checksums(root, (EXPERIMENT_FILE,))
        return root

    def rebuild(self, root: Path) -> ExperimentSummary:
        """验证所有输入 artifact，并原子重写 JSONL、摘要和 Markdown。"""
        root = root.resolve()
        experiment = self.load_experiment(root, verify=True)
        records = self.load_records(root)
        summary = aggregate_experiment(experiment, records)
        _write_jsonl(root / TRIALS_FILE, records)
        _write_json(root / SUMMARY_FILE, summary)
        _write_text(root / REPORT_FILE, _render_markdown(summary))
        _write_checksums(
            root,
            (EXPERIMENT_FILE, TRIALS_FILE, SUMMARY_FILE, REPORT_FILE),
        )
        return summary

    def load_experiment(self, root: Path, *, verify: bool) -> Experiment:
        """读取冻结声明，可选校验控制文件哈希。"""
        if verify:
            _verify_control_checksums(root)
        try:
            return Experiment.model_validate_json(
                (root / EXPERIMENT_FILE).read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ReportingError(
                f"cannot load experiment declaration: {error}"
            ) from error

    def load_records(self, root: Path) -> tuple[TrialRecord, ...]:
        """只接受通过单 Trial checksum 校验的直接子目录。"""
        records: list[TrialRecord] = []
        trial_store = ArtifactStore(self._artifact_root)
        for child in sorted(root.iterdir()):
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            try:
                trial = Trial.model_validate_json(
                    (child / "trial.json").read_text(encoding="utf-8")
                )
                result = trial_store.load_result(child, verify=True)
            except (OSError, ValueError, ArtifactError) as error:
                raise ReportingError(
                    f"cannot rebuild trial {child.name}: {error}"
                ) from error
            records.append(TrialRecord(trial=trial, result=result))
        return tuple(records)


def main(argv: list[str] | None = None) -> int:
    """从已有 artifact 离线重建报告。"""
    parser = argparse.ArgumentParser(prog="xcode-eval-report")
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.experiment_root.resolve()
    summary = ExperimentArtifactStore(root.parent).rebuild(root)
    print(
        json.dumps(
            {
                "experiment_id": summary.experiment_id,
                "summary": str(root / SUMMARY_FILE),
                "report": str(root / REPORT_FILE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    _write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, records: tuple[TrialRecord, ...]) -> None:
    lines = [
        json.dumps(
            {
                "artifact_root": record.trial.trial_id,
                "trial": record.trial.model_dump(mode="json"),
                "result": record.result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in sorted(records, key=lambda item: item.trial.trial_id)
    ]
    _write_text(path, "".join(f"{line}\n" for line in lines))


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_checksums(root: Path, names: tuple[str, ...]) -> None:
    checksums = {name: _sha256(root / name) for name in names}
    _write_json(root / CHECKSUMS_FILE, checksums)


def _verify_control_checksums(root: Path) -> None:
    path = root / CHECKSUMS_FILE
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
        declaration_hash = expected[EXPERIMENT_FILE]
    except (OSError, ValueError, TypeError) as error:
        raise ReportingError(
            f"cannot verify experiment control files: {error}"
        ) from error
    if declaration_hash != _sha256(root / EXPERIMENT_FILE):
        raise ReportingError("experiment control artifact checksum verification failed")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_markdown(summary: ExperimentSummary) -> str:
    rows = [
        "| Variant | Valid / observed | Success | pass@k | pass^k | Input tokens | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in summary.variants:
        rows.append(
            "| "
            f"{variant.variant_id} | {variant.valid_trials} / {variant.observed_trials} "
            f"| {_percent(variant.success_rate)} | {_number(variant.pass_at_k)} "
            f"| {_number(variant.pass_power_k)} | {_number(variant.usage.input_tokens)} "
            f"| {variant.usage.wall_time_seconds:.2f}s |"
        )
    exclusions = []
    for variant in summary.variants:
        detail = ", ".join(
            f"{category.value}={count}"
            for category, count in sorted(
                variant.exclusions.items(), key=lambda item: item[0].value
            )
        )
        exclusions.append(
            f"- `{variant.variant_id}`: excluded={variant.excluded_trials}, "
            f"missing={variant.missing_trials}" + (f" ({detail})" if detail else "")
        )
    return "\n".join(
        [
            f"# Experiment {summary.experiment_id}",
            "",
            f"Dataset: `{summary.dataset_version}`; tasks: {len(summary.task_ids)}; "
            f"repetitions: {summary.repetitions}.",
            "",
            *rows,
            "",
            "## Exclusions and missing trials",
            "",
            *exclusions,
            "",
            "## Efficiency frontier",
            "",
            ", ".join(f"`{value}`" for value in summary.efficient_variant_ids)
            or "No comparable variant has complete token usage.",
            "",
            "All observed Trial costs, including failures and exclusions, are included "
            "in aggregate usage. See `trials.jsonl` for traceable raw records and "
            "`summary.json` for formulas and joint result/cost points.",
            "",
        ]
    )


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _number(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
