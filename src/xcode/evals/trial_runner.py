"""串联 workspace、隔离 Agent、artifact 和隐藏 verifier 的单 Trial 路径。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any

from xcode.harness.config import XcodeRuntimeConfig

from .artifacts import ArtifactStore
from .isolation import BubblewrapExecutor, IsolationError
from .schema import (
    ErrorCategory,
    ResourceUsage,
    Task,
    Trial,
    TrialResult,
    VerifierSpec,
)
from .verifier import VerifierError, VerifierRunner
from .workspace import (
    changed_paths,
    GitWorkspaceFactory,
    WorkspaceError,
    workspace_patch,
)


class TrialRunner:
    """执行一次不可复用、可独立判分且证据完整的 Trial。"""

    def __init__(
        self,
        *,
        repository: Path,
        workspace_root: Path,
        artifact_store: ArtifactStore,
        executor: BubblewrapExecutor,
        verifier: VerifierRunner | None = None,
    ) -> None:
        self._workspace_factory = GitWorkspaceFactory(
            repository=repository,
            workspace_root=workspace_root,
        )
        self._artifact_store = artifact_store
        self._executor = executor
        self._verifier = verifier or VerifierRunner()

    def run(
        self,
        *,
        task: Task,
        trial: Trial,
        verifier_spec: VerifierSpec,
        runtime_config: XcodeRuntimeConfig,
        environment: dict[str, Any],
    ) -> TrialResult:
        """执行完整路径；所有基础设施失败均保存分类而非伪装成未解决。"""
        paths = self._artifact_store.begin(
            task=task,
            trial=trial,
            environment=environment,
        )
        started_at = datetime.now(UTC)
        usage = ResourceUsage(wall_time_seconds=0, model_calls=0, tool_calls=0)
        agent_completed = False
        workspace = None
        agent_patch = ""
        try:
            workspace = self._workspace_factory.create(task, trial.trial_id)
            if workspace.revision != trial.workspace_revision:
                raise WorkspaceError(
                    "trial workspace revision does not match restored commit"
                )
            with TemporaryDirectory() as output_value:
                output = Path(output_value)
                execution = self._executor.run(
                    task=task,
                    trial=trial,
                    runtime_config=runtime_config,
                    workspace=workspace.root,
                    output=output,
                )
                usage = execution.usage
                agent_completed = True
                shutil.copyfile(
                    output / "trace.jsonl",
                    paths.resolve(paths.manifest.trace),
                )
            changes = changed_paths(workspace.initial_files, workspace.root)
            agent_patch = workspace_patch(workspace.initial_files, workspace.root)
            termination_error = _termination_error(execution.termination_reason)
            verifier_result = (
                self._verifier.run(
                    spec=verifier_spec,
                    task=task,
                    workspace=workspace.root,
                    changed_paths=changes,
                    log_path=paths.resolve(paths.manifest.verifier_log),
                )
                if termination_error is None
                else None
            )
            over_budget = (
                usage.model_calls > trial.budget.model_calls
                or usage.tool_calls > trial.budget.tool_calls
                or (
                    trial.budget.input_tokens is not None
                    and usage.input_tokens is not None
                    and usage.input_tokens > trial.budget.input_tokens
                )
                or (
                    trial.budget.output_tokens is not None
                    and usage.output_tokens is not None
                    and usage.output_tokens > trial.budget.output_tokens
                )
            )
            error_category = (
                ErrorCategory.BUDGET_EXCEEDED if over_budget else termination_error
            )
            result = TrialResult(
                trial_id=trial.trial_id,
                started_at=execution.started_at,
                finished_at=execution.finished_at,
                agent_completed=termination_error is None,
                valid_trial=not over_budget and termination_error is None,
                verifier=verifier_result if termination_error is None else None,
                error_category=error_category,
                error_message=(
                    "declared resource budget exceeded"
                    if over_budget
                    else execution.error_detail
                ),
                termination_reason=(
                    "budget_exceeded" if over_budget else execution.termination_reason
                ),
                usage=usage,
                artifacts=paths.manifest,
            )
        except (WorkspaceError, IsolationError, VerifierError) as error:
            category = _error_category(error)
            result = TrialResult(
                trial_id=trial.trial_id,
                started_at=started_at,
                finished_at=datetime.now(UTC),
                agent_completed=agent_completed,
                valid_trial=False,
                verifier=None,
                error_category=category,
                error_message=str(error),
                termination_reason=category.value,
                usage=usage,
                artifacts=paths.manifest,
            )
        for relative in (paths.manifest.trace, paths.manifest.verifier_log):
            path = paths.resolve(relative)
            if not path.exists():
                path.touch()
        self._artifact_store.finish(paths=paths, patch=agent_patch, result=result)
        return result


def _error_category(error: Exception) -> ErrorCategory:
    if isinstance(error, WorkspaceError):
        return ErrorCategory.ENVIRONMENT_FAILURE
    if isinstance(error, IsolationError):
        return ErrorCategory.AGENT_FAILURE
    return ErrorCategory.VERIFIER_FAILURE


def _termination_error(reason: str) -> ErrorCategory | None:
    """把无法形成能力结果的 Agent final 状态映射为排除分类。"""
    if reason == "provider_error":
        return ErrorCategory.PROVIDER_FAILURE
    if reason == "cancelled":
        return ErrorCategory.AGENT_FAILURE
    return None
