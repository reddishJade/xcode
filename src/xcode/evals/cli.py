"""Xcode Eval 的薄命令入口。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from xcode.harness.config import discover_runtime_config

from .artifacts import ArtifactStore
from .dataset import load_tasks
from .isolation import BubblewrapExecutor
from .policy import build_eval_runtime, EVAL_EXECUTION_MODE
from .schema import ModelConfig, Trial, Variant, VerifierSpec
from .trial_runner import TrialRunner


def main(argv: list[str] | None = None) -> int:
    """运行一个真实历史 Task；不把 Test 输出当作能力分。"""
    parser = argparse.ArgumentParser(prog="xcode-eval")
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args(argv)

    control_root = Path.cwd().resolve()
    tasks = {task.task_id: task for task in load_tasks(args.dataset)}
    task = tasks[args.task]
    runtime = build_eval_runtime(discover_runtime_config(control_root, args.config))
    profile = runtime.provider.model_profiles["main"]
    harness_revision = _git(control_root, "rev-parse", "HEAD")
    dirty = bool(_git(control_root, "status", "--porcelain"))
    sanitized_runtime = runtime.model_dump(mode="json")
    for model_profile in sanitized_runtime["provider"]["model_profiles"].values():
        model_profile["api_key"] = "<redacted-configured>"
    trial_id = f"{args.experiment_id}.{task.task_id}.full.r{args.repetition}"
    trial = Trial(
        trial_id=trial_id,
        experiment_id=args.experiment_id,
        task_id=task.task_id,
        dataset_version=task.dataset_version,
        variant=Variant(
            variant_id="full",
            harness_revision=harness_revision,
            capabilities={
                "context_assembly": True,
                "tools": True,
                "compaction": True,
                "error_recovery": True,
                "permission_feedback": True,
                "session": True,
                "mcp": True,
                "memory": True,
            },
            config={
                "runtime": sanitized_runtime,
                "execution_mode": EVAL_EXECUTION_MODE,
                "isolation": "bubblewrap",
                "external_network_tools": "denied",
            },
        ),
        model=ModelConfig(
            provider=profile.transport,
            model=profile.chat_model,
            options={
                "base_url": profile.base_url,
                "thinking": profile.thinking,
                "reasoning_effort": profile.reasoning_effort,
            },
        ),
        budget=task.budget,
        repetition=args.repetition,
        workspace_revision=task.source.revision,
        command=tuple(sys.argv),
    )
    hidden_root = (args.private_root / task.task_id).resolve()
    verifier_spec = VerifierSpec(
        verifier_id=task.verifier_id,
        version="v1",
        command=(str(control_root / ".venv/bin/python"), "verify.py", "{workspace}"),
        hidden_root=str(hidden_root),
        timeout_seconds=180,
    )
    python_runtime = Path(sys.executable).resolve().parents[1]
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise RuntimeError("uv executable is required for isolated Eval")
    executor = BubblewrapExecutor(
        xcode_source=control_root / "src/xcode",
        virtualenv=control_root / ".venv",
        python_runtime=python_runtime,
        uv_executable=Path(uv_executable),
    )
    artifact_root = args.run_root / "artifacts"
    runner = TrialRunner(
        repository=args.repository,
        workspace_root=args.run_root / "workspaces",
        artifact_store=ArtifactStore(artifact_root),
        executor=executor,
    )
    result = runner.run(
        task=task,
        trial=trial,
        verifier_spec=verifier_spec,
        runtime_config=runtime,
        environment={
            "captured_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "xcode_revision": harness_revision,
            "xcode_dirty": dirty,
            "workspace_revision": task.source.revision,
            "dataset_version": task.dataset_version,
            "provider": profile.transport,
            "model": profile.chat_model,
            "base_url": profile.base_url,
            "provider_auth_configured": bool(profile.api_key),
        },
    )
    print(
        json.dumps(
            {
                "trial_id": result.trial_id,
                "valid_trial": result.valid_trial,
                "success": result.success,
                "error_category": result.error_category,
                "termination_reason": result.termination_reason,
                "artifact_root": str(
                    artifact_root / args.experiment_id / trial.trial_id
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0 if result.valid_trial else 2


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
