from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xcode.ai.providers.protocol import StreamProvider

from ...agent.compaction import (
    extract_prompt_tokens_from_usage,
    get_model_soft_threshold,
)
from ...agent.config import AgentLoopConfig, AgentLoopTurnUpdate
from ...agent.context_assembly import DefaultContextAssembler
from ...agent.context_collector import (
    ActiveDiffCollector,
    ContextCollectorRegistry,
    NotesCollector,
    ProjectManifestCollector,
    RecentValidationCollector,
    TaskStateCollector,
)
from ...agent.history import apply_request_hygiene
from ...agent.message_converter import convert_to_llm
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from ...agent.protocols import AgentTool
from ..config import AgentConfig, ExecutionMode, RequestHygieneConfig
from ..observability import AuditRecord, HookManager, HookRecord, PermissionPolicy
from ..observability.permission_model import PolicyEvaluator
from ..observability.permission_model import ExternalDirectory
from ..skills import ApprovalCallback, ToolSpec
from .cancellation import CancellationToken
from .compaction import CompactController, estimate_message_tokens
from .execution_modes import ExecutionModeState, mode_notice
from .message_codec import messages_from_compacted_dicts
from .tool_gate import ToolGate

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


@dataclass(frozen=True)
class GateConfig:
    """ToolGate 配置：审批、权限、审计、Hook。"""

    approval_callback: ApprovalCallback | None = None
    permission_policy: PermissionPolicy | None = None
    restricted_dirs: tuple[str, ...] = ()
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = ()
    hook_manager: HookManager | None = None
    audit_logger: Callable[[AuditRecord], None] | None = None
    session_id: str = "local"
    external_directories: tuple[ExternalDirectory, ...] = ()


@dataclass
class AgentRuntimeConfig:
    """StructuredAgent 运行时基础设施配置。"""

    config: AgentConfig = field(default_factory=AgentConfig)
    compactor: StructuredCompactor | None = None
    compact_controller: CompactController | None = None
    cancellation_token: CancellationToken | None = None
    runtime_context_provider: RuntimeContextProvider | None = None
    fallback_provider: StreamProvider | None = None
    project_root: Path | None = None
    request_hygiene: RequestHygieneConfig | None = None
    skill_registry: Any | None = None


@dataclass(frozen=True)
class TurnSnapshot:
    config: AgentConfig
    registry: tuple[ToolSpec, ...]
    provider: StreamProvider
    runtime_context_provider: RuntimeContextProvider | None


def build_turn_snapshot(
    config: AgentConfig,
    registry: tuple[ToolSpec, ...],
    provider: StreamProvider,
    runtime_context_provider: RuntimeContextProvider | None,
) -> TurnSnapshot:
    return TurnSnapshot(
        config=config,
        registry=registry,
        provider=provider,
        runtime_context_provider=runtime_context_provider,
    )


def build_turn_context_messages(
    question: str,
    mode: ExecutionMode,
    snapshot: TurnSnapshot,
    resumed_notice: str | None,
) -> list[AgentMessage]:
    typed: list[AgentMessage] = []
    notice = mode_notice(mode)
    parts: list[str] = []
    if snapshot.runtime_context_provider is not None:
        parts = list(snapshot.runtime_context_provider(question))
    if resumed_notice is not None:
        parts.append(f"<session-notices>\n{resumed_notice}\n</session-notices>")
    if notice:
        parts.append(notice)
    if parts:
        typed.append(SystemMessage(content="\n\n".join(p for p in parts if p)))
    return typed


def _compact_and_emit(
    loop_messages: list[AgentMessage],
    compactor: StructuredCompactor | None,
    emit_hook: Callable[[HookRecord], None],
) -> list[AgentMessage]:
    """执行消息压缩并发射 Hook。"""
    emit_hook(HookRecord("on_compact", metadata={"messages": len(loop_messages)}))
    if compactor is None:
        return loop_messages
    dict_messages = [_to_dict_safe(m) for m in loop_messages]
    compacted = compactor(dict_messages)
    return messages_from_compacted_dicts(compacted)


def _to_dict_safe(message: AgentMessage) -> dict[str, Any]:
    from .agent_helpers import to_dict

    return to_dict(message)


def _build_before_provider_request_closure(
    emit_hook: Callable[[HookRecord], None],
    get_prompt_version: Callable[[], str],
) -> Callable[[list[dict[str, Any]], list[Any]], None]:
    """构建 provider 请求前的 hook 发射回调。"""

    def closure(msgs: list[dict[str, Any]], tools: list[Any]) -> None:
        system_prompt = "\n\n".join(
            str(message.get("content", ""))
            for message in msgs
            if message.get("role") == "system"
        )
        prompt_bytes = len(system_prompt.encode("utf-8"))
        prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        emit_hook(
            HookRecord(
                "before_provider_request",
                metadata={
                    "messages": msgs,
                    "tools": [tool_definition_to_dict(tool) for tool in tools],
                    "prompt_version": get_prompt_version(),
                    "prompt_sha256": prompt_sha,
                    "system_prompt_bytes": prompt_bytes,
                },
            )
        )

    return closure


def _build_task_state_provider(project_root: Path) -> Callable[[], str] | None:
    """返回一个可调用对象，每次调用时从 TaskStore 读取当前任务状态。"""
    try:
        from xcode.harness.task_store import TaskStore

        store = TaskStore(project_root)
        tasks = store.list()
        if not tasks:
            return None

        def provider() -> str:
            try:
                current = store.list()
            except Exception:
                return ""
            lines: list[str] = []
            for t in current:
                blocked_by = t.payload.get("blocked_by")
                block_info = f" [Blocked by: {blocked_by}]" if blocked_by else ""
                lines.append(f"  - #{t.id} [{t.status}]: {t.title}{block_info}")
            return "\n".join(lines)

        return provider
    except Exception:
        return None


def build_loop_config(
    mode: ExecutionMode,
    snapshot: TurnSnapshot,
    gate: ToolGate,
    registry: tuple[ToolSpec, ...],
    compactor: StructuredCompactor | None,
    manual_compact_requested: Callable[[], bool] | None,
    request_hygiene: RequestHygieneConfig,
    compact_controller: CompactController | None,
    last_prompt_tokens: int | None,
    tools_for_mode: Callable[[tuple[ToolSpec, ...], ExecutionMode], list[AgentTool]],
    steer: Callable[[AgentMessage], None],
    emit_hook: Callable[[HookRecord], None],
    mode_state: ExecutionModeState,
    get_prompt_version: Callable[[], str],
    project_root: Path | None = None,
    skill_registry: Any | None = None,
) -> AgentLoopConfig:
    gate_snapshot = gate.snapshot_for(registry)

    def should_compact_fn(loop_messages: list[AgentMessage]) -> bool:
        return _should_compact(
            loop_messages,
            compactor,
            manual_compact_requested,
            last_prompt_tokens,
            snapshot,
        )

    def compact_fn(loop_messages: list[AgentMessage]) -> list[AgentMessage]:
        return _compact_and_emit(loop_messages, compactor, emit_hook)

    def transform_fn(
        messages: list[AgentMessage],
        _signal: object,
    ) -> list[AgentMessage]:
        if not request_hygiene.enabled:
            return messages
        return apply_request_hygiene(
            messages,
            max_tool_result_bytes=request_hygiene.max_tool_result_bytes,
            max_tool_arg_length=request_hygiene.max_tool_arg_length,
            keep_head_lines=request_hygiene.keep_head_lines,
            keep_tail_lines=request_hygiene.keep_tail_lines,
        )

    def prepare_next_turn_fn() -> AgentLoopTurnUpdate | None:
        if gate.check_progress_reminder():
            steer(
                UserMessage(
                    content=(
                        "<reminder>You have gone several turns without updating "
                        "task progress. Use update_task or save_task_progress to "
                        "record progress before continuing.</reminder>"
                    )
                )
            )
        if mode_state.check_plan_timeout():
            steer(
                SystemMessage(
                    content=(
                        "<plan-timeout>\n"
                        "Plan Mode timed out after reaching the maximum number "
                        "of investigation turns. Returning to Act Mode.\n"
                        "</plan-timeout>"
                    )
                )
            )
        return None

    # 构建上下文收集器 + 组装器
    registry_: ContextCollectorRegistry | None = None
    assembler: DefaultContextAssembler | None = None
    if project_root is not None:
        from xcode.harness.skills_registry import (
            SkillIndexCollector,
            SkillRegistry,
            build_skill_search_dirs,
        )

        registry_ = ContextCollectorRegistry()
        registry_.register(ProjectManifestCollector(project_root))
        registry_.register(ActiveDiffCollector(project_root))
        registry_.register(RecentValidationCollector())
        task_provider = _build_task_state_provider(project_root)
        if task_provider is not None:
            registry_.register(TaskStateCollector(task_provider))
        registry_.register(NotesCollector(project_root))
        sr = skill_registry
        if sr is None:
            sr = SkillRegistry()
            sr.discover(build_skill_search_dirs(project_root))
        registry_.register(SkillIndexCollector(sr))
        assembler = DefaultContextAssembler()

    return AgentLoopConfig(
        provider=snapshot.provider,
        convert_to_llm=convert_to_llm,
        context_collectors=registry_,
        context_assembler=assembler,
        max_steps=snapshot.config.max_steps,
        max_step_retries=3,
        retry_backoff_base=0.5,
        max_tokens_continuation=True,
        max_consecutive_continuations=3,
        min_continuation_tokens=500,
        watchdog_repeated_tool_limit=snapshot.config.watchdog_repeated_tool_limit,
        max_consecutive_idle_steps=4,
        should_compact=should_compact_fn,
        compact=compact_fn,
        transform_context=transform_fn,
        is_tool_productive=gate.build_is_tool_productive_hook(gate_snapshot),
        before_tool_call=gate.build_before_tool_hook(gate_snapshot),
        after_tool_call=gate.build_after_tool_hook(gate_snapshot),
        before_provider_request=_build_before_provider_request_closure(
            emit_hook, get_prompt_version
        ),
        prepare_next_turn=prepare_next_turn_fn,
    )


def _should_compact(
    messages: list[AgentMessage],
    compactor: StructuredCompactor | None,
    manual_compact_requested: Callable[[], bool] | None,
    last_prompt_tokens: int | None,
    snapshot: TurnSnapshot,
) -> bool:
    if compactor is None:
        return False
    if manual_compact_requested and manual_compact_requested():
        return True
    if last_prompt_tokens is not None:
        model_name = getattr(snapshot.provider, "model", None)
        return last_prompt_tokens >= get_model_soft_threshold(
            str(model_name) if model_name is not None else None
        )
    from .agent_helpers import to_dict

    msg_dicts = [to_dict(m) for m in messages]
    return (
        snapshot.config.compact_threshold > 0
        and len(messages) > snapshot.config.compact_threshold
    ) or (
        snapshot.config.compact_token_threshold > 0
        and estimate_message_tokens(msg_dicts) > snapshot.config.compact_token_threshold
    )


def tool_definition_to_dict(tool: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(tool, "name", "")),
        "description": str(getattr(tool, "description", "")),
        "parameters": getattr(tool, "parameters", {}),
    }


def resolve_permission_policy(
    project_root: Path | None, base: PermissionPolicy | None
) -> PermissionPolicy | None:
    """返回静态权限策略，直接使用已通过 discover_runtime_config 合并的结果。

    各配置源的合并已在 config.discover_runtime_config() 中完成，
    无需在此处再次加载 .local/settings.json。
    """
    return base


def record_last_prompt_tokens(
    messages: list[AgentMessage],
) -> int | None:
    for message in reversed(messages):
        if not isinstance(message, AssistantMessage):
            continue
        prompt_tokens = extract_prompt_tokens_from_usage(message.usage)
        if prompt_tokens is not None:
            return prompt_tokens
    return None
