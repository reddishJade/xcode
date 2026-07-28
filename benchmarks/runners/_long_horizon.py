"""上下文压缩消融实验的公共运行器。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Literal
from uuid import uuid4

from xcode.ai.events import Message, ProviderEvent, UsageUpdate
from xcode.ai.models import get_models, get_providers
from xcode.ai.providers.base import ModelProvider
from xcode.ai.types import StreamOptions, ToolDefinition
from xcode.agent.messages import AgentMessage, UserMessage
from xcode.coding_agent.app import XcodeApp, build_app
from xcode.harness.agent_runtime.compaction import (
    LayeredCompactor,
    build_compact_summarize_fn,
)
from xcode.harness.agent_runtime.result import AgentHarnessResult
from xcode.harness.config import (
    HooksRuntimeConfig,
    InlineInstructionSource,
    XcodeRuntimeConfig,
)
from xcode.harness.memory import load_session_checkpoint

from benchmarks.evaluators.state_retention import (
    capture_initial_state,
    evaluate_state_retention,
    retention_rate,
)
from benchmarks.evaluators.test_result import run_command
from benchmarks.models import LongHorizonTask
from benchmarks.runners.progress import ProgressStage, ProgressUpdate

Variant = Literal["baseline", "xcode"]
SummaryMode = Literal["model", "deterministic"]

_OVERFLOW_MARKERS = (
    "context length",
    "context window",
    "context_length_exceeded",
    "maximum context",
    "prompt is too long",
    "too many tokens",
)

_BENCHMARK_GIT_EXCLUDES = """\
.benchmark/
.local/
.pytest_cache/
**/__pycache__/
*.py[cod]
.coverage
"""
_BENCHMARK_GIT_DATE = "2000-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class RunOptions:
    """一次任务运行所需的稳定实验参数。"""

    output_dir: Path
    repeat: int
    temperature: float | None = None
    summary_mode: SummaryMode = "model"
    keep_workspace: bool = False
    progress_callback: Callable[[ProgressUpdate], None] | None = None


@dataclass(frozen=True)
class ProviderCallRecord:
    """一次 provider stream 的实际 usage。"""

    kind: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    has_usage: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class InstrumentedProvider:
    """固定采样参数并记录主调用与摘要调用。"""

    def __init__(
        self,
        delegate: ModelProvider,
        temperature: float | None,
        calls: list[ProviderCallRecord],
        progress_callback: Callable[[ProgressStage, str], None] | None = None,
    ) -> None:
        self._delegate = delegate
        self._temperature = temperature
        self._calls = calls
        self._progress_callback = progress_callback
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._delegate.model

    @property
    def base_url(self) -> str:
        return self._delegate.base_url

    @property
    def transport(self) -> str:
        return self._delegate.transport

    @property
    def thinking(self) -> bool:
        return self._delegate.thinking

    @property
    def reasoning_effort(self) -> str | None:
        return self._delegate.reasoning_effort

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        options: StreamOptions | None = None,
        **kwargs: object,
    ) -> AsyncIterator[ProviderEvent]:
        merged = options or StreamOptions()
        if self._temperature is not None:
            merged = replace(merged, temperature=self._temperature)
        started = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        has_usage = False
        error: str | None = None
        kind = "compaction summary" if not tools else "agent request"
        self._emit_progress("provider_started", f"{kind} · {self._delegate.model}")
        try:
            async for event in self._delegate.stream(
                messages,
                tools,
                options=merged,
                **kwargs,
            ):
                if isinstance(event, UsageUpdate):
                    input_tokens += event.input_tokens
                    output_tokens += event.output_tokens
                    has_usage = True
                yield event
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            record = ProviderCallRecord(
                kind="compaction_summary" if not tools else "agent",
                model=self._delegate.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_seconds=time.perf_counter() - started,
                has_usage=has_usage,
                error=error,
            )
            with self._lock:
                self._calls.append(record)
            detail = (
                f"{kind} · in={input_tokens} out={output_tokens} · "
                f"{record.duration_seconds:.1f}s"
            )
            self._emit_progress("provider_finished", detail)

    def reset_conversation_state(self) -> None:
        reset = getattr(self._delegate, "reset_conversation_state", None)
        if callable(reset):
            reset()

    def _emit_progress(self, stage: ProgressStage, detail: str) -> None:
        if self._progress_callback is not None:
            self._progress_callback(stage, detail)


def _prepare_workspace(source: Path, workspace: Path) -> str:
    """复制 fixture，并创建不受父仓库影响的确定性 Git 基线。"""
    shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(".git"))
    _run_git(workspace, "init", "--quiet")
    _run_git(workspace, "config", "user.name", "Xcode Benchmark")
    _run_git(workspace, "config", "user.email", "benchmark@local.invalid")
    _run_git(workspace, "config", "core.autocrlf", "false")
    _run_git(workspace, "config", "core.filemode", "false")
    _run_git(workspace, "config", "commit.gpgsign", "false")

    exclude_path = workspace / ".git" / "info" / "exclude"
    exclude_path.write_text(_BENCHMARK_GIT_EXCLUDES, encoding="utf-8")
    _run_git(workspace, "add", "--all", "--force", "--", ".")
    commit_env = {
        **os.environ,
        "GIT_AUTHOR_DATE": _BENCHMARK_GIT_DATE,
        "GIT_COMMITTER_DATE": _BENCHMARK_GIT_DATE,
    }
    _run_git(
        workspace,
        "-c",
        "core.hooksPath=.git/benchmark-hooks-disabled",
        "commit",
        "--quiet",
        "--allow-empty",
        "--no-gpg-sign",
        "-m",
        "benchmark: initial fixture",
        env=commit_env,
    )
    top_level = Path(_run_git(workspace, "rev-parse", "--show-toplevel")).resolve()
    if top_level != workspace.resolve():
        raise RuntimeError(
            "benchmark Git workspace escaped its fixture root: "
            f"expected {workspace.resolve()}, got {top_level}"
        )
    return _run_git(workspace, "rev-parse", "HEAD")


def _run_git(
    workspace: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> str:
    """在指定工作区运行 Git，并把初始化失败转换为清晰错误。"""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError as exc:
        raise RuntimeError(
            f"unable to initialize benchmark Git workspace: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        command = "git " + " ".join(args)
        raise RuntimeError(f"{command} failed in {workspace}: {detail}")
    return completed.stdout.strip()


def run_task(
    task: LongHorizonTask,
    variant: Variant,
    runtime_config: XcodeRuntimeConfig,
    options: RunOptions,
) -> dict[str, Any]:
    """在隔离工作区运行一个任务，并写出单次原始记录。"""
    current_turn = 0

    def emit_progress(
        stage: ProgressStage,
        detail: str = "",
        *,
        turn: int | None = None,
    ) -> None:
        callback = options.progress_callback
        if callback is None:
            return
        callback(
            ProgressUpdate(
                stage=stage,
                task_id=task.id,
                variant=variant,
                repeat=options.repeat,
                total_turns=len(task.turns),
                turn=turn,
                detail=detail,
            )
        )

    run_id = f"{task.id}-{variant}-r{options.repeat}-{uuid4().hex[:8]}"
    output_dir = options.output_dir.resolve()
    workspace = output_dir / "workspaces" / run_id
    if workspace.exists():
        raise ValueError(f"benchmark workspace already exists: {workspace}")
    emit_progress("run_started", f"workspace={workspace.name}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    baseline_commit = _prepare_workspace(task.workspace, workspace)

    initial_state = capture_initial_state(workspace, task.state_checks)
    calls: list[ProviderCallRecord] = []
    tool_calls: list[dict[str, object]] = []
    turn_records: list[dict[str, object]] = []
    runtime_errors: list[str] = []
    terminations: list[str] = []
    compactions = 0
    restarts = 0
    checkpoint_resumes = 0
    boundary_indices: dict[str, int] = {}
    session_id = f"bench-{task.id}-{variant}-r{options.repeat}"
    runtime_dir = output_dir / "runtime" / run_id
    configured = _benchmark_runtime_config(
        runtime_config,
        task,
        sessions_dir=runtime_dir / "sessions",
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    app = _build_benchmark_app(
        workspace,
        configured,
        variant,
        options,
        calls,
        session_id,
        progress_callback=lambda stage, detail: emit_progress(
            stage,
            detail,
            turn=current_turn or None,
        ),
    )

    try:
        for turn_index, turn in enumerate(task.turns, 1):
            current_turn = turn_index
            emit_progress("turn_started", turn.prompt, turn=turn_index)
            boundary_id = f"turn-{turn_index:04d}"
            boundary_indices[boundary_id] = len(app.agent.history_messages())
            app.agent.set_compaction_source_message_id(boundary_id)
            if turn.compact_before and variant == "xcode":
                app.agent.request_compaction()

            call_start = len(calls)
            turn_started = time.perf_counter()
            try:
                result, turn_compactions = _run_turn(
                    app,
                    turn.prompt,
                    progress_callback=lambda stage, detail: emit_progress(
                        stage,
                        detail,
                        turn=turn_index,
                    ),
                )
            except Exception as exc:
                emit_progress("error", str(exc), turn=turn_index)
                runtime_errors.append(f"turn {turn_index}: {exc}")
                turn_records.append(
                    {
                        "turn": turn_index,
                        "duration_seconds": time.perf_counter() - turn_started,
                        "error": str(exc),
                    }
                )
                break

            compactions += turn_compactions
            terminations.append(str(result.termination_reason))
            turn_tools = [
                {"name": call.name, "input": dict(call.input)}
                for call in result.tool_calls
            ]
            tool_calls.extend(turn_tools)
            turn_usage = calls[call_start:]
            turn_records.append(
                {
                    "turn": turn_index,
                    "duration_seconds": time.perf_counter() - turn_started,
                    "termination_reason": str(result.termination_reason),
                    "compactions": turn_compactions,
                    "provider_calls": [call.to_dict() for call in turn_usage],
                    "tool_calls": turn_tools,
                }
            )
            emit_progress(
                "turn_completed",
                f"termination={result.termination_reason}",
                turn=turn_index,
            )

            if turn.restart_after:
                emit_progress(
                    "restart", "checkpoint + transcript tail", turn=turn_index
                )
                history = app.agent.history_messages()
                rebuilt, used_checkpoint = _rebuild_history(
                    app,
                    variant,
                    session_id,
                    history,
                    boundary_indices,
                )
                app.close()
                app = _build_benchmark_app(
                    workspace,
                    configured,
                    variant,
                    options,
                    calls,
                    session_id,
                    progress_callback=lambda stage, detail: emit_progress(
                        stage,
                        detail,
                        turn=current_turn or None,
                    ),
                )
                app.agent.load_history(rebuilt)
                if result.run_state is not None:
                    app.agent.restore_run_state_metadata(result.run_state)
                app.agent.set_resumed_notice(
                    "The benchmark process restarted. Continue the same task "
                    "without repeating completed work."
                )
                restarts += 1
                checkpoint_resumes += int(used_checkpoint)
                boundary_indices = {}
    finally:
        app.close()

    emit_progress("verification", "task success command and state checks")
    verification = run_command(task.success_command, workspace)
    state_outcomes = evaluate_state_retention(
        workspace,
        task.state_checks,
        initial_state,
    )
    input_tokens = sum(call.input_tokens for call in calls)
    output_tokens = sum(call.output_tokens for call in calls)
    models_used = sorted({call.model for call in calls})
    prices_by_model = {
        model: prices
        for model in models_used
        if (prices := _model_prices(model)) is not None
    }
    pricing_complete = len(prices_by_model) == len(models_used)
    input_cost = _usage_cost(calls, prices_by_model, token_field="input")
    output_cost = _usage_cost(calls, prices_by_model, token_field="output")
    if not pricing_complete:
        input_cost = None
        output_cost = None
    error_text = "\n".join(
        [*runtime_errors, *(call.error or "" for call in calls if call.error)]
    )
    overflow = any(marker in error_text.casefold() for marker in _OVERFLOW_MARKERS)
    all_turns_completed = len(turn_records) == len(task.turns) and not runtime_errors
    normal_termination = all(reason == "completed" for reason in terminations)
    task_success = verification.passed
    long_session_completed = (
        task_success and all_turns_completed and normal_termination and not overflow
    )
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "task_id": task.id,
        "variant": variant,
        "repeat": options.repeat,
        "model": app.agent.provider.model,
        "temperature": options.temperature,
        "summary_mode": options.summary_mode,
        "execution_mode": "build",
        "baseline_commit": baseline_commit,
        "models_used": models_used,
        "started_at": started_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "turns_expected": len(task.turns),
        "turns_completed": len(turn_records) - int(bool(runtime_errors)),
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "peak_input_tokens": max((call.input_tokens for call in calls), default=0),
        "usage_complete": bool(calls) and all(call.has_usage for call in calls),
        "pricing_by_model": {
            model: {
                "input_per_million": prices[0],
                "output_per_million": prices[1],
            }
            for model, prices in prices_by_model.items()
        },
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "task_success": task_success,
        "long_session_completed": long_session_completed,
        "verification": verification.to_dict(),
        "state_retention": retention_rate(state_outcomes),
        "state_checks": [outcome.to_dict() for outcome in state_outcomes],
        "context_overflow": overflow,
        "runtime_errors": runtime_errors,
        "termination_reasons": terminations,
        "compactions": compactions,
        "restarts": restarts,
        "checkpoint_resumes": checkpoint_resumes,
        "provider_calls": [call.to_dict() for call in calls],
        "tool_calls_total": len(tool_calls),
        "repeated_read_calls": _repeated_read_calls(tool_calls),
        "turns": turn_records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / f"{run_id}.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    record["record_path"] = str(record_path)

    if not options.keep_workspace:
        shutil.rmtree(workspace)
        shutil.rmtree(runtime_dir, ignore_errors=True)
    else:
        record["workspace"] = str(workspace)
        record["runtime_dir"] = str(runtime_dir)
    emit_progress(
        "run_completed",
        f"success={task_success} input_tokens={input_tokens}",
    )
    return record


def _benchmark_runtime_config(
    base: XcodeRuntimeConfig,
    task: LongHorizonTask,
    *,
    sessions_dir: Path,
) -> XcodeRuntimeConfig:
    instruction = InlineInstructionSource(
        content=(
            "This is a controlled benchmark. Do not use subagents, web tools, "
            "or modify files outside the current workspace. Follow each user "
            "turn literally and preserve all stated constraints across restarts. "
            "The workspace is an isolated Git repository whose HEAD is the exact "
            "initial fixture. Use git diff HEAD to verify task changes and file "
            "preservation; do not inspect benchmark transcripts or create commits."
        ),
        priority="critical",
    )
    agent = base.agent.model_copy(
        update={
            "max_recent_messages": task.compaction.max_recent_messages,
            "keep_recent_tokens": task.compaction.keep_recent_tokens,
        }
    )
    paths = base.paths.model_copy(update={"sessions_dir": sessions_dir})
    prompt = base.prompt.model_copy(
        update={"instructions": (*base.prompt.instructions, instruction)}
    )
    return base.model_copy(
        update={
            "agent": agent,
            "paths": paths,
            "prompt": prompt,
            "hooks": HooksRuntimeConfig(),
        }
    )


def _build_benchmark_app(
    workspace: Path,
    runtime_config: XcodeRuntimeConfig,
    variant: Variant,
    options: RunOptions,
    calls: list[ProviderCallRecord],
    session_id: str,
    progress_callback: Callable[[ProgressStage, str], None] | None = None,
) -> XcodeApp:
    app = build_app(
        project_root=workspace,
        runtime_config=runtime_config,
        audit_path=None,
    )
    instrumented = InstrumentedProvider(
        app.agent.provider,
        options.temperature,
        calls,
        progress_callback,
    )
    app.agent.provider = instrumented
    if variant == "baseline":
        app.agent.compactor = None
    elif isinstance(app.agent.compactor, LayeredCompactor):
        app.agent.compactor.summarize_fn = (
            build_compact_summarize_fn(instrumented)
            if options.summary_mode == "model"
            else None
        )
    app.agent.session_id = session_id
    app.agent.set_history_session_id(session_id)
    return app


def _run_turn(
    app: XcodeApp,
    prompt: str,
    progress_callback: Callable[[ProgressStage, str], None] | None = None,
) -> tuple[AgentHarnessResult, int]:
    result: AgentHarnessResult | None = None
    compactions = 0
    # Benchmark 不提供交互审批；固定 Build 模式让工作区内写入和验证命令
    # 在两个消融分组中以相同策略自动执行。
    for event in app.agent.run_stream(prompt, mode="build"):
        if event.type == "compaction":
            compactions += 1
            if progress_callback is not None:
                progress_callback(
                    "compaction",
                    f"removed={event.data.messages_removed} "
                    f"remaining={event.data.messages_after}",
                )
        elif event.type == "tool_use" and progress_callback is not None:
            progress_callback("tool_started", _tool_call_detail(event.data))
        elif event.type == "final":
            result = event.data
    if result is None:
        raise RuntimeError("agent run ended without a final event")
    return result, compactions


def _tool_call_detail(call: object) -> str:
    name = str(getattr(call, "name", "tool"))
    raw_input = getattr(call, "input", None)
    if not isinstance(raw_input, dict):
        return name
    value = (
        raw_input.get("path")
        or raw_input.get("file_path")
        or raw_input.get("command")
        or raw_input.get("pattern")
    )
    return f"{name} · {value}" if value else name


def _rebuild_history(
    app: XcodeApp,
    variant: Variant,
    session_id: str,
    history: list[AgentMessage],
    boundary_indices: dict[str, int],
) -> tuple[list[AgentMessage], bool]:
    if variant == "baseline" or not isinstance(app.agent.compactor, LayeredCompactor):
        return history, False
    checkpoint_dir = app.agent.compactor.checkpoint_dir
    if not isinstance(checkpoint_dir, Path):
        return history, False
    checkpoint = load_session_checkpoint(checkpoint_dir, session_id)
    if checkpoint is None:
        return history, False
    boundary = boundary_indices.get(checkpoint.boundary_message_id)
    if boundary is None:
        return history, False
    rebuilt: list[AgentMessage] = [
        UserMessage(content=checkpoint.render_rebuild_prompt()),
        *history[boundary:],
    ]
    return rebuilt, True


def _model_prices(model_name: str) -> tuple[float, float] | None:
    normalized = model_name.casefold()
    for provider_name in get_providers():
        for model in get_models(provider_name):
            if model.id.casefold() == normalized:
                return model.cost.input, model.cost.output
    return None


def _usage_cost(
    calls: list[ProviderCallRecord],
    prices_by_model: dict[str, tuple[float, float]],
    *,
    token_field: Literal["input", "output"],
) -> float:
    index = 0 if token_field == "input" else 1
    return sum(
        (call.input_tokens if token_field == "input" else call.output_tokens)
        * prices_by_model.get(call.model, (0.0, 0.0))[index]
        / 1_000_000
        for call in calls
    )


def _repeated_read_calls(calls: list[dict[str, object]]) -> int:
    seen: set[tuple[str, str]] = set()
    repeated = 0
    for call in calls:
        name = str(call.get("name", ""))
        if name not in {"read", "read_file"}:
            continue
        raw_input = call.get("input")
        if not isinstance(raw_input, dict):
            continue
        path = str(
            raw_input.get("path")
            or raw_input.get("file_path")
            or raw_input.get("input")
            or ""
        )
        signature = (name, path)
        if signature in seen:
            repeated += 1
        seen.add(signature)
    return repeated
