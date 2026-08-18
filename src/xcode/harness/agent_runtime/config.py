from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from xcode.ai.providers.base import ModelProvider

from xcode.ai.models import effective_compact_threshold
from ...agent._compaction import extract_prompt_tokens_from_usage
from ...agent.config import (
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    CompletionVerifier,
)
from ...agent.context import ContextCollectorRegistry, DefaultContextAssembler
from ...agent._codec import convert_to_llm as _convert_to_llm
from ...agent.request import (
    DefaultRequestAssembler,
    RequestAssembly,
    RequestHygiene,
)
from .prompting.citations import decorate_citable_messages
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from ..config import AgentConfig, RequestHygieneConfig
from ..security import PermissionDecision, PermissionPolicy
from ..observability import (
    AuditLogger,
    ExternalHookRunner,
    HookManager,
    HookRecord,
    RuntimeCorrelation,
    hook_correlation_fields,
)
from ..security.permission_model import (
    ExternalDirectory,
    GrantStore,
    PolicyEvaluator,
    PathExtractor,
    Rule,
    SensitivePathOverride,
)
from ...agent.types import ApprovalCallback, ToolSpec
from .cancellation import CancellationToken
from ..session.inbox import SessionInbox


from .compaction import CompactController, estimate_message_tokens
from ._mode_protocol import RuntimeModeState
from .message_codec import messages_from_compacted_dicts
from .tool_gate import ToolGate


def _convert_to_llm_with_citations(
    messages: list[AgentMessage],
) -> list[dict[str, Any]]:
    decorated = decorate_citable_messages(messages)
    return _convert_to_llm(decorated)


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
    external_hook_runner: ExternalHookRunner | None = None
    external_hooks_subagent: bool = False
    external_hooks_cwd: Path | None = None
    audit_logger: AuditLogger | None = None
    session_id: str = "local"
    external_directories: tuple[ExternalDirectory, ...] = ()
    sensitive_path_overrides: tuple[SensitivePathOverride, ...] = ()
    session_grant_store: GrantStore | None = None
    session_grant_store_provider: Callable[[], GrantStore | None] | None = None
    permanent_grant_store: GrantStore | None = None
    user_rulesets: dict[str, tuple[Rule, ...]] = field(default_factory=dict)
    default_mode_rulesets: dict[str, tuple[Rule, ...]] = field(default_factory=dict)
    mode_fallbacks: dict[str, PermissionDecision] = field(default_factory=dict)
    shell_unresolved_policies: dict[str, PermissionDecision] = field(
        default_factory=dict
    )
    tool_path_extractors: dict[str, PathExtractor] = field(default_factory=dict)
    correlation: RuntimeCorrelation | None = None


@dataclass
class AgentRuntimeConfig:
    """CodingAgentHarness 运行时基础设施配置。"""

    session_inbox: SessionInbox
    config: AgentConfig = field(default_factory=AgentConfig)
    compactor: StructuredCompactor | None = None
    compact_controller: CompactController | None = None
    cancellation_token: CancellationToken | None = None
    runtime_context_provider: RuntimeContextProvider | None = None
    fallback_provider: ModelProvider | None = None
    project_root: Path | None = None
    request_hygiene: RequestHygieneConfig | None = None
    context_collectors: ContextCollectorRegistry | None = None
    context_assembler: DefaultContextAssembler | None = None


@dataclass(frozen=True)
class TurnSnapshot:
    config: AgentConfig
    registry: tuple[ToolSpec, ...]
    provider: ModelProvider
    runtime_context_provider: RuntimeContextProvider | None


def build_turn_snapshot(
    config: AgentConfig,
    registry: tuple[ToolSpec, ...],
    provider: ModelProvider,
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
    snapshot: TurnSnapshot,
    resumed_notice: str | None,
    mode_notice: str | None = None,
    memory_overview: str | None = None,
) -> list[AgentMessage]:
    typed: list[AgentMessage] = []
    parts: list[str] = []
    if snapshot.runtime_context_provider is not None:
        parts = list(snapshot.runtime_context_provider(question))
    if resumed_notice is not None:
        parts.append(f"<session-notices>\n{resumed_notice}\n</session-notices>")
    if memory_overview:
        parts.append(memory_overview)
    if mode_notice:
        parts.append(mode_notice)
    if parts:
        typed.append(SystemMessage(content="\n\n".join(p for p in parts if p)))
    return typed


def _compact_and_emit(
    loop_messages: list[AgentMessage],
    compactor: StructuredCompactor | None,
    emit_hook: Callable[[HookRecord], None],
    correlation: RuntimeCorrelation,
) -> list[AgentMessage]:
    """执行消息压缩并发射 Hook。"""
    current = correlation.snapshot()
    emit_hook(
        HookRecord(
            "on_compact",
            metadata={"messages": len(loop_messages)},
            **hook_correlation_fields(current),
        )
    )
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
    correlation: RuntimeCorrelation,
    provider: ModelProvider,
) -> Callable[[RequestAssembly], None]:
    """构建 provider 请求前的 hook 发射回调。"""

    def closure(assembly: RequestAssembly) -> None:
        correlation.begin_turn()
        current = correlation.begin_request()
        messages = list(assembly.wire_messages)
        system_prompt = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "system"
        )
        prompt_bytes = len(system_prompt.encode("utf-8"))
        prompt_sha = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        provider_info = {
            "model": provider.model,
            "base_url": provider.base_url,
            "transport": provider.transport,
            "thinking": provider.thinking,
            "reasoning_effort": provider.reasoning_effort,
        }
        tool_payload = [tool_definition_to_dict(tool) for tool in assembly.tools]
        options_payload = _request_options_payload(assembly.options)
        request_bytes = json.dumps(
            {
                "messages": messages,
                "tools": tool_payload,
                "provider": provider_info,
                "options": options_payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        emit_hook(
            HookRecord(
                "before_provider_request",
                metadata={
                    "messages": messages,
                    "tools": tool_payload,
                    "provider": provider_info,
                    "options": options_payload,
                    "assembly": {
                        "current_step": assembly.current_step,
                        "hygiene_applied": assembly.hygiene_applied,
                        "context_trace": [
                            {
                                "source": trace.source,
                                "target": trace.target,
                                "block_id": trace.block_id,
                                "included": trace.included,
                                "token_count": trace.token_count,
                                "content_sha256": trace.content_sha256,
                            }
                            for trace in assembly.context_trace
                        ],
                    },
                    "prompt_version": get_prompt_version(),
                    "prompt_sha256": prompt_sha,
                    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "system_prompt_bytes": prompt_bytes,
                },
                **hook_correlation_fields(current),
            )
        )

    return closure


def _request_options_payload(options: object) -> dict[str, object]:
    if options is None or not is_dataclass(options):
        return {}
    payload: dict[str, object] = {}
    for option in fields(options):
        value = getattr(options, option.name)
        if value is not None:
            payload[option.name] = _request_option_value(value)
    return payload


def _request_option_value(value: object) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _request_option_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_request_option_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _request_option_value(getattr(value, item.name))
            for item in fields(value)
        }
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def build_loop_config(
    *,
    snapshot: TurnSnapshot,
    gate: ToolGate,
    registry: tuple[ToolSpec, ...],
    compactor: StructuredCompactor | None,
    manual_compact_requested: Callable[[], bool] | None,
    request_hygiene: RequestHygieneConfig,
    compact_controller: CompactController | None,
    last_prompt_tokens: int | None,
    steer: Callable[[AgentMessage], None],
    emit_hook: Callable[[HookRecord], None],
    get_prompt_version: Callable[[], str],
    context_collectors: ContextCollectorRegistry | None = None,
    context_assembler: DefaultContextAssembler | None = None,
    correlation: RuntimeCorrelation | None = None,
    # 领域扩展参数由上层装配后注入。
    mode_state: RuntimeModeState | None = None,
    watchdog_repeated_tool_skip: frozenset[str] | None = None,
    completion_verifier: CompletionVerifier | None = None,
) -> AgentLoopConfig:
    active_correlation = correlation or RuntimeCorrelation("local")
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
        return _compact_and_emit(
            loop_messages,
            compactor,
            emit_hook,
            active_correlation,
        )

    request_assembler = DefaultRequestAssembler(
        converter=_convert_to_llm_with_citations,
        context_collectors=context_collectors,
        context_assembler=context_assembler or DefaultContextAssembler(),
        hygiene=RequestHygiene(
            enabled=request_hygiene.enabled,
            max_tool_result_bytes=request_hygiene.max_tool_result_bytes,
            max_tool_arg_length=request_hygiene.max_tool_arg_length,
            keep_head_lines=request_hygiene.keep_head_lines,
            keep_tail_lines=request_hygiene.keep_tail_lines,
        ),
    )

    def prepare_next_turn_fn() -> AgentLoopTurnUpdate | None:
        if gate.check_progress_reminder():
            steer(
                UserMessage(
                    content=(
                        "<reminder>You have gone several turns without updating "
                        "task progress. Use todowrite to "
                        "record progress before continuing.</reminder>"
                    )
                )
            )
        if mode_state is not None and mode_state.check_plan_timeout():
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

    return AgentLoopConfig(
        provider=snapshot.provider,
        request_assembler=request_assembler,
        max_steps=snapshot.config.max_steps,
        tool_workers=snapshot.config.tool_workers,
        tool_timeout_seconds=float(snapshot.config.tool_timeout_seconds),
        max_step_retries=3,
        retry_backoff_base=0.5,
        max_tokens_continuation=True,
        max_consecutive_continuations=3,
        min_continuation_tokens=500,
        watchdog_repeated_tool_limit=snapshot.config.watchdog_repeated_tool_limit,
        watchdog_repeated_tool_skip=watchdog_repeated_tool_skip or frozenset(),
        max_consecutive_idle_steps=4,
        should_compact=should_compact_fn,
        compact=compact_fn,
        completion_verifier=completion_verifier,
        is_tool_productive=gate.build_is_tool_productive_hook(gate_snapshot),
        before_tool_call=gate.build_before_tool_hook(gate_snapshot),
        after_tool_call=gate.build_after_tool_hook(gate_snapshot),
        before_provider_request=_build_before_provider_request_closure(
            emit_hook,
            get_prompt_version,
            active_correlation,
            snapshot.provider,
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
        provider = snapshot.provider
        model_name = provider.model if isinstance(provider, ModelProvider) else None
        model_str = str(model_name) if model_name is not None else None
        # 使用 context_window - reserve_tokens 作为精确触发线
        trigger = effective_compact_threshold(
            model_str,
            reserve_tokens=snapshot.config.reserve_tokens,
            trigger_ratio=snapshot.config.compact_trigger_ratio,
        )
        return last_prompt_tokens >= trigger
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
