"""StructuredAgent — harness 层对 agent/Agent 的适配。

将 Xcode 特定的 ToolSpec、权限、审计、压缩等配置映射为 AgentLoopConfig，
委托给 agent/Agent.run() 执行。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, cast

from ...agent.agent import Agent
from ...agent.messages import convert_to_llm
from ...agent.config import (
    AgentLoopConfig,
    AgentLoopTurnUpdate,
)
from ...agent.messages import (
    AgentMessage,
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from ...agent.protocols import AgentTool, ContentBlock
from ...agent.types import TextContent, ToolCallContent
from xcode.ai.providers.protocol import ModelProvider
from .agent_helpers import run_coro_sync, aiter_to_sync_iter, to_dict
from .cancellation import CancellationToken
from .compaction import CompactController, estimate_message_tokens
from .event_translation import (
    _StreamTranslationState,
    _translate_event,
    StructuredAgentEvent,
)
from .fallback import _FallbackSwitchingProvider, _FallbackWithRetryPrimary
from .history_manager import HistoryManager
from .mode_manager import ModeManager
from .result import (
    _build_structured_result,
    _final_event,
    RunState,
    StructuredAgentResult,
)
from .tool_adapter import adapt_tool_specs
from .tool_gate import ToolGate, ToolGateSnapshot
from ..config import AgentConfig, ExecutionMode
from ..observability import HookManager, HookRecord, PermissionPolicy
from ..skills import ApprovalCallback, ToolSpec


__all__ = ["StructuredAgent"]

StructuredCompactor = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
RuntimeContextProvider = Callable[[str], list[str]]


@dataclass(frozen=True)
class TurnSnapshot:
    """单个 turn 使用的运行期快照。"""

    config: AgentConfig
    registry: tuple[ToolSpec, ...]
    provider: ModelProvider
    runtime_context_provider: RuntimeContextProvider | None


class StructuredAgent:
    """与 provider 解耦的结构化工具调用循环。

    harness 层适配器：将 Xcode 特定配置映射为 AgentLoopConfig，
    委托 agent 核心循环执行，通过事件翻译保持 StructuredAgentEvent
    接口不变。
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: tuple[ToolSpec, ...],
        config: AgentConfig | None = None,
        approval_callback: ApprovalCallback | None = None,
        compactor: StructuredCompactor | None = None,
        manual_compact_requested: Callable[[], bool] | None = None,
        compact_controller: CompactController | None = None,
        audit_logger: Callable[[Any], None] | None = None,
        session_id: str = "local",
        permission_policy: PermissionPolicy | None = None,
        high_risk_requires_approval: bool = True,
        hook_manager: HookManager | None = None,
        runtime_context_provider: RuntimeContextProvider | None = None,
        cancellation_token: CancellationToken | None = None,
        fallback_provider: ModelProvider | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.provider: ModelProvider = provider
        if fallback_provider is not None:
            self.provider = _FallbackWithRetryPrimary(provider, fallback_provider)
        self._original_provider = provider
        self.project_root = project_root
        self.registry = registry
        self.tool_map = {t.name: t for t in registry}
        self.config = config or AgentConfig()
        self.compactor = compactor
        self.manual_compact_requested = manual_compact_requested or (
            compact_controller.consume if compact_controller else None
        )
        self._compact_controller = compact_controller
        self.runtime_context_provider = runtime_context_provider
        self.cancellation_token = cancellation_token or CancellationToken()

        # 组件
        self._mode = ModeManager()
        self._gate = ToolGate(
            mode_manager=self._mode,
            approval_callback=approval_callback,
            permission_policy=_resolve_permission_policy(
                project_root, permission_policy
            ),
            high_risk_requires_approval=high_risk_requires_approval,
            hook_manager=hook_manager,
            audit_logger=audit_logger,
            session_id=session_id,
        )
        self.audit_logger = audit_logger
        self._history = HistoryManager()
        self._resumed_notice: str | None = None

        # 适配 ToolSpec → AgentTool，创建 Agent 实例
        adapted = adapt_tool_specs(
            registry,
            approval_callback=approval_callback,
            permission_policy=self._gate._permission_policy,
            high_risk_requires_approval=high_risk_requires_approval,
        )
        adapted.extend(self._build_mode_switch_agent_tools())
        self._agent = Agent(cast(list[AgentTool], adapted))

    # ── 公共 API ──

    def steer(self, msg: AgentMessage) -> None:
        self._agent.steer(msg)

    def follow_up(self, msg: AgentMessage) -> None:
        self._agent.follow_up(msg)

    def request_compaction(self) -> None:
        if self._compact_controller is not None:
            self._compact_controller.request()

    def confirm_plan(self) -> None:
        self._mode.confirm_plan()

    def clear_history(self) -> None:
        self._history.clear()
        self._reset_provider_conversation_state()

    def load_history(self, messages: list[AgentMessage]) -> None:
        self._history.load(messages)
        self._reset_provider_conversation_state()

    def set_resumed_notice(self, notice: str) -> None:
        """设置会话恢复通知，将在下一轮 system prompt 中注入一次。"""
        self._resumed_notice = notice

    def load_run_state(self, run_state: RunState) -> None:
        self._history.load_run_state(run_state)
        self._reset_provider_conversation_state()
        restored = self._history.restore_mode(run_state)
        if restored is not None:
            self._mode._current_mode = cast(ExecutionMode, restored)

    def history_messages(self) -> list[AgentMessage]:
        return self._history.messages()

    def run(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        return run_coro_sync(self.arun(question, mode=mode))

    async def run_async(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        return await self.arun(question, mode=mode)

    async def arun(
        self, question: str, mode: ExecutionMode | None = None
    ) -> StructuredAgentResult:
        result: StructuredAgentResult | None = None
        async for event in self.arun_stream(question, mode=mode):
            if event.type == "final":
                result = event.data
        assert result is not None
        return result

    def run_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> Iterator[StructuredAgentEvent]:
        yield from aiter_to_sync_iter(
            self.arun_stream(question, mode=mode), self.cancellation_token
        )

    async def arun_stream(
        self, question: str, mode: ExecutionMode | None = None
    ) -> AsyncIterator[StructuredAgentEvent]:
        snapshot = self._turn_snapshot()
        effective_mode = mode or snapshot.config.execution_mode
        self._mode._current_mode = effective_mode
        active_registry = self._mode.filter_tools_for_mode(snapshot.registry)
        self.cancellation_token.reset()

        context_messages = self._turn_context_messages(
            question, effective_mode, snapshot
        )
        history_messages = context_messages + self.history_messages()
        turn_messages: list[AgentMessage] = [UserMessage(content=question)]

        adapted = adapt_tool_specs(
            active_registry,
            approval_callback=self._gate._approval_callback,
            permission_policy=self._gate._permission_policy,
            high_risk_requires_approval=self._gate._high_risk_requires_approval,
        )
        self._agent = Agent(cast(list[AgentTool], adapted))

        loop_config = self._build_loop_config(effective_mode, snapshot)

        self._gate._emit_hook(
            HookRecord(
                "before_agent_start",
                metadata={"question": question, "mode": effective_mode},
            )
        )

        translation_state = _StreamTranslationState()

        async for event in self._agent.run_stream(
            turn_messages,
            loop_config,
            signal=self.cancellation_token,
            history=history_messages,
        ):
            translated = _translate_event(event, translation_state)
            if translated is not None:
                for te in translated if isinstance(translated, list) else [translated]:
                    yield te

        result = self._agent.last_result
        assert result is not None

        if isinstance(self.provider, _FallbackSwitchingProvider):
            wrapper = self.provider
            if wrapper._using_fallback:
                self._original_provider = wrapper._fallback
            else:
                self._original_provider = wrapper._primary

        self._history.save_turn(result.messages)

        visible_result = (
            replace(result, messages=context_messages + result.messages)
            if context_messages
            else result
        )
        final = _build_structured_result(
            visible_result, snapshot.config.max_steps, self._mode.current_mode
        )
        yield _final_event(result.steps, final)

    # ── 模式切换 ──

    def _build_mode_switch_agent_tools(self) -> list[Any]:
        plan_spec, act_spec = self._mode.build_mode_switch_tools()
        return adapt_tool_specs((plan_spec, act_spec))

    def _tools_for_mode(
        self, registry: tuple[ToolSpec, ...], mode: ExecutionMode
    ) -> list[Any]:
        from .execution_modes import policy_for_mode

        policy = policy_for_mode(mode)
        filtered = policy.filter_tools(registry)
        adapted = adapt_tool_specs(
            filtered,
            approval_callback=self._gate._approval_callback,
            permission_policy=self._gate._permission_policy,
            high_risk_requires_approval=self._gate._high_risk_requires_approval,
        )
        adapted.extend(self._build_mode_switch_agent_tools())
        return adapted

    # ── 配置构建 ──

    def _build_loop_config(
        self, mode: ExecutionMode, snapshot: TurnSnapshot
    ) -> AgentLoopConfig:
        tool_map = {t.name: t for t in snapshot.registry}
        gate_snapshot = ToolGateSnapshot(
            approval_callback=self._gate._approval_callback,
            permission_policy=self._gate._permission_policy,
            high_risk_requires_approval=self._gate._high_risk_requires_approval,
            tool_map=tool_map,
        )

        should_compact = self._loop_should_compact(snapshot) if self.compactor else None
        compact = self._loop_compact if self.compactor else None

        return AgentLoopConfig(
            provider=snapshot.provider,
            convert_to_llm=convert_to_llm,
            max_steps=snapshot.config.max_steps,
            max_step_retries=3,
            retry_backoff_base=0.5,
            max_tokens_continuation=True,
            max_consecutive_continuations=3,
            min_continuation_tokens=500,
            watchdog_repeated_tool_limit=snapshot.config.watchdog_repeated_tool_limit,
            max_consecutive_idle_steps=4,
            should_compact=should_compact,
            compact=compact,
            is_tool_productive=self._gate.build_is_tool_productive_hook(gate_snapshot),
            before_tool_call=self._gate.build_before_tool_hook(gate_snapshot),
            after_tool_call=self._gate.build_after_tool_hook(gate_snapshot),
            prepare_next_turn=self._loop_prepare_next_turn(snapshot),
        )

    # ── 辅助方法 ──

    def _loop_should_compact(
        self, snapshot: TurnSnapshot
    ) -> Callable[[list[AgentMessage]], bool]:
        def should_compact(messages: list[AgentMessage]) -> bool:
            return self._should_compact([to_dict(m) for m in messages], snapshot)

        return should_compact

    def _loop_compact(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        self._gate._emit_hook(
            HookRecord("on_compact", metadata={"messages": len(messages)})
        )
        if self.compactor is None:
            return messages
        dict_messages = [to_dict(m) for m in messages]
        compacted = self.compactor(dict_messages)
        return _messages_from_compacted_dicts(compacted)

    def _loop_prepare_next_turn(
        self, snapshot: TurnSnapshot
    ) -> Callable[[], AgentLoopTurnUpdate | None]:
        def prepare_next_turn() -> AgentLoopTurnUpdate | None:
            if self._gate.check_progress_reminder():
                self.steer(
                    UserMessage(
                        content=(
                            "<reminder>You have gone several turns without updating task progress. "
                            "Use update_task or save_task_progress to record progress before continuing.</reminder>"
                        )
                    )
                )

            if self._mode.check_plan_timeout():
                self._agent.update_tools(self._tools_for_mode(snapshot.registry, "act"))
                self.steer(
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

        return prepare_next_turn

    def _should_compact(
        self, messages: list[dict[str, Any]], snapshot: TurnSnapshot
    ) -> bool:
        if self.compactor is None:
            return False
        if self.manual_compact_requested and self.manual_compact_requested():
            return True
        return (
            snapshot.config.compact_threshold > 0
            and len(messages) > snapshot.config.compact_threshold
        ) or (
            snapshot.config.compact_token_threshold > 0
            and estimate_message_tokens(messages)
            > snapshot.config.compact_token_threshold
        )

    def _turn_context_messages(
        self,
        question: str,
        mode: ExecutionMode,
        snapshot: TurnSnapshot,
    ) -> list[AgentMessage]:
        from .execution_modes import mode_notice

        typed: list[AgentMessage] = []
        notice = mode_notice(mode)
        parts: list[str] = []
        if snapshot.runtime_context_provider is not None:
            parts = list(snapshot.runtime_context_provider(question))
        if self._resumed_notice is not None:
            parts.append(
                f"<session-notices>\n{self._resumed_notice}\n</session-notices>"
            )
            self._resumed_notice = None
        if notice:
            parts.append(notice)
        if parts:
            typed.append(SystemMessage(content="\n\n".join(p for p in parts if p)))
        return typed

    def _turn_snapshot(self) -> TurnSnapshot:
        return TurnSnapshot(
            config=self.config,
            registry=tuple(self.registry),
            provider=self.provider,
            runtime_context_provider=self.runtime_context_provider,
        )

    def _reset_provider_conversation_state(self) -> None:
        reset = getattr(self.provider, "reset_conversation_state", None)
        if callable(reset):
            reset()


# ── 模块级辅助 ──


def _resolve_permission_policy(
    project_root: Path | None, base: PermissionPolicy | None
) -> PermissionPolicy | None:
    if project_root is None:
        return base
    local = project_root / ".local" / "settings.json"
    root = project_root / "settings.json"
    settings_path = local if local.exists() else (root if root.exists() else None)
    if settings_path is None:
        return base
    from ..observability.permissions import (
        SettingsSandboxPermissionPolicy,
        CompositePermissionPolicy,
    )

    sandbox = SettingsSandboxPermissionPolicy(settings_path)
    return CompositePermissionPolicy(sandbox, base)


def _messages_from_compacted_dicts(
    messages: list[dict[str, Any]],
) -> list[AgentMessage]:
    """将压缩后的 provider 风格消息恢复为内部消息。"""
    restored: list[AgentMessage] = []
    for item in messages:
        message = _message_from_compacted_dict(item)
        if message is not None:
            restored.append(message)
    return restored


def _message_from_compacted_dict(item: dict[str, Any]) -> AgentMessage | None:
    """恢复单条压缩消息，未知格式保持为普通用户文本。"""
    role = str(item.get("role", ""))
    content = item.get("content", "")
    if role == "system":
        return SystemMessage(content=_content_to_text(content))
    if role == "user":
        return UserMessage(content=_content_to_text(content))
    if role == "assistant":
        return _assistant_from_compacted_dict(item)
    if role == "tool":
        return ToolResultMessage(
            tool_call_id=str(item.get("tool_call_id", "")),
            content=_tool_result_content_from_compacted(content),
        )
    return None


def _assistant_from_compacted_dict(item: dict[str, Any]) -> AssistantMessage:
    """恢复 assistant 文本和工具调用。"""
    content_blocks: list[ContentBlock] = []
    content = item.get("content")
    text = _content_to_text(content)
    if text:
        content_blocks.append(TextContent(text=text))

    tool_calls = item.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            parsed = _tool_call_from_compacted(tool_call)
            if parsed is not None:
                content_blocks.append(parsed)
    return AssistantMessage(content=content_blocks)


def _tool_call_from_compacted(item: object) -> ToolCallContent | None:
    """恢复 OpenAI 风格 tool_call。"""
    if not isinstance(item, dict):
        return None
    function = item.get("function", {})
    if not isinstance(function, dict):
        return None
    name = str(function.get("name", "")).strip()
    tool_call_id = str(item.get("id", "")).strip()
    if not name or not tool_call_id:
        return None
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            decoded = {}
        arguments = decoded
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCallContent(id=tool_call_id, name=name, arguments=arguments)


def _tool_result_content_from_compacted(content: object) -> str:
    """恢复工具结果文本。"""
    if not isinstance(content, list):
        return _content_to_text(content)

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "tool_result":
            parts.append(str(part.get("content", "")))
        else:
            parts.append(_content_to_text(part))
    return "".join(parts)


def _content_to_text(content: object) -> str:
    """将压缩中间格式转为稳定文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "tool_result":
                parts.append(str(part.get("content", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
