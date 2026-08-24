"""工具执行门控：审批、权限、准入决策。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from xcode.ai.events import ToolCall

from ...agent.config import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    IsToolProductiveHook,
)
from ...agent.types import AgentTool, AgentToolResult, CancellationSignal
from ...agent.types import TextContent, ToolCallContent, ToolSpecAdapter
from ._mode_protocol import ToolGateMode
from .tool_audit import build_audit_record, emit_audit
from .tool_hooks import emit_hook, emit_tool_hook, tool_result_text
from ..security import (
    PermissionEngine,
    PermissionEngineConfig,
    PermissionDecision,
    PermissionEngineResult,
    PermissionPolicy,
)
from ..observability import (
    AuditLogger,
    ExternalHookRunner,
    HookManager,
    HookRecord,
    RuntimeCorrelation,
    hook_correlation_fields,
    HookCorrelationFields,
)
from ..security.permission_model import (
    ExternalDirectory,
    GrantStore,
    PolicyEvaluator,
    PathExtractor,
    Rule,
    SensitivePathOverride,
)
from ..security.approval import ApprovalPolicy, ApprovalsReviewer
from ...agent.types import (
    ApprovalCallback,
    ToolSpec,
    stringify_tool_input,
)
from ..observability import redact_text


@runtime_checkable
class _ClearableGrantStore(Protocol):
    def clear(self) -> None: ...


# 核心工具 capability 映射，提供给权限引擎
_TOOL_ACTION_PROFILES: dict[str, tuple[str, str]] = {
    "read_file": ("read", "path"),
    "glob_files": ("read", "path"),
    "grep_search": ("read", "path"),
    "find_files": ("read", "path"),
    "list_dir": ("read", "path"),
    "search_tools": ("read", "none"),
    "write_file": ("write", "path"),
    "edit_file": ("edit", "path"),
    "apply_patch": ("patch", "path"),
    "bash": ("shell", "none"),
    "shell": ("shell", "none"),
    "load_skill": ("skill", "skill"),
    "todowrite": ("write", "none"),
    "webfetch": ("read", "none"),
    "websearch": ("read", "none"),
    "question": ("read", "none"),
}


class _RedactingAdapter(ToolSpecAdapter):
    """ToolSpecAdapter + 输出脱敏（生产环境用）。"""

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, object],
        signal: CancellationSignal | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        def _redacted_update(text: str) -> None:
            if on_update is not None:
                on_update(
                    AgentToolResult(content=[TextContent(text=redact_text(text))])
                )

        content = await asyncio.to_thread(
            self._spec.handler, dict(params), _redacted_update
        )
        metadata = getattr(content, "metadata", None)
        render_intent = getattr(content, "render_intent", None)
        return AgentToolResult(
            content=[TextContent(text=redact_text(str(content)))],
            details=metadata if isinstance(metadata, dict) else None,
            is_error=bool(getattr(content, "is_error", False)),
            render_intent=render_intent,
        )


@dataclass(frozen=True)
class ToolGateSnapshot:
    """ToolGate 在单个 turn 中使用的冻结配置。"""

    approval_callback: ApprovalCallback | None
    approvals_reviewer: ApprovalsReviewer
    permission_policy: PermissionPolicy | None
    approval_policy: ApprovalPolicy
    tool_map: dict[str, ToolSpec]
    restricted_dirs: tuple[str, ...] = ()
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = ()
    project_root: Path | None = None
    external_directories: tuple[ExternalDirectory, ...] = ()
    sensitive_path_overrides: tuple[SensitivePathOverride, ...] = ()
    session_grant_store: GrantStore | None = None
    permanent_grant_store: GrantStore | None = None
    mode_ruleset: tuple[Rule, ...] = ()
    user_ruleset: tuple[Rule, ...] = ()
    mode_fallback: PermissionDecision = "ask"
    shell_unresolved_policy: PermissionDecision = "ask"
    tool_path_extractors: dict[str, PathExtractor] = field(default_factory=dict)


class ToolGate:
    """工具执行门控：HITL 审批、权限检查、准入决策。"""

    PROGRESS_TOOL_NAMES = frozenset()

    def __init__(
        self,
        mode_state: ToolGateMode,
        user_approval_callback: ApprovalCallback | None,
        auto_approval_callback: ApprovalCallback | None,
        permission_policy: PermissionPolicy | None,
        hook_manager: HookManager | None,
        audit_logger: AuditLogger | None,
        session_id: str,
        approval_policy: ApprovalPolicy = "on-request",
        external_hook_runner: ExternalHookRunner | None = None,
        external_hooks_subagent: bool = False,
        external_hooks_cwd: Path | None = None,
        correlation: RuntimeCorrelation | None = None,
        restricted_dirs: tuple[str, ...] = (),
        hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
        project_root: Path | None = None,
        external_directories: tuple[ExternalDirectory, ...] = (),
        sensitive_path_overrides: tuple[SensitivePathOverride, ...] = (),
        session_grant_store: GrantStore | None = None,
        session_grant_store_provider: Callable[[], GrantStore | None] | None = None,
        permanent_grant_store: GrantStore | None = None,
        user_ruleset: tuple[Rule, ...] = (),
        user_rulesets: Mapping[str, tuple[Rule, ...]] | None = None,
        default_mode_rulesets: Mapping[str, tuple[Rule, ...]] | None = None,
        mode_fallbacks: Mapping[str, PermissionDecision] | None = None,
        shell_unresolved_policies: Mapping[str, PermissionDecision] | None = None,
        tool_path_extractors: Mapping[str, PathExtractor] | None = None,
    ) -> None:
        self._mode = mode_state
        self._user_ruleset = user_ruleset
        self._user_rulesets = user_rulesets or {}
        self._default_mode_rulesets = default_mode_rulesets or {}
        self._mode_fallbacks = mode_fallbacks or {}
        self._shell_unresolved_policies = shell_unresolved_policies or {}
        self._tool_path_extractors = tool_path_extractors or {}
        self._user_approval_callback = user_approval_callback
        self._auto_approval_callback = auto_approval_callback
        self._approval_policy: ApprovalPolicy = approval_policy
        self._permission_policy = permission_policy
        self._restricted_dirs = restricted_dirs
        self._hook_constraint_providers = hook_constraint_providers
        self._external_directories = external_directories
        self._sensitive_path_overrides = sensitive_path_overrides
        self._hook_manager = hook_manager
        self._external_hook_runner = external_hook_runner
        self._external_hooks_subagent = external_hooks_subagent
        self._external_hooks_cwd = external_hooks_cwd
        self._correlation = correlation or RuntimeCorrelation(session_id)
        self._audit_logger = audit_logger
        self._session_id = session_id
        self._project_root = project_root
        self._session_grant_store = session_grant_store
        self._session_grant_store_provider = session_grant_store_provider
        self._permanent_grant_store = permanent_grant_store
        self._progress_steps_without_update: int = 0
        self._last_perm_results: dict[str, PermissionEngineResult] = {}

    def _resolve_session_store(self) -> GrantStore | None:
        if self._session_grant_store_provider is not None:
            return self._session_grant_store_provider()
        return self._session_grant_store

    def _ruleset_for_mode(self, mode_name: str) -> tuple[Rule, ...]:
        configured = self._user_rulesets.get(mode_name)
        if configured is not None:
            return configured
        return self._user_ruleset

    def _default_ruleset_for_mode(self, mode_name: str) -> tuple[Rule, ...]:
        return self._default_mode_rulesets.get(mode_name, ())

    def _fallback_for_mode(self, mode_name: str) -> PermissionDecision:
        return self._mode_fallbacks.get(mode_name, "ask")

    def _shell_unresolved_policy_for_mode(self, mode_name: str) -> PermissionDecision:
        return self._shell_unresolved_policies.get(mode_name, "ask")

    def _approval_callback_for_mode(
        self, reviewer: ApprovalsReviewer
    ) -> ApprovalCallback | None:
        if reviewer == "auto_review":
            return self._auto_approval_callback
        return self._user_approval_callback

    def snapshot(self) -> ToolGateSnapshot:
        mode_name = self._mode.current_mode
        default_rules = self._default_ruleset_for_mode(mode_name)
        fallback = self._fallback_for_mode(mode_name)
        shell_unresolved_policy = self._shell_unresolved_policy_for_mode(mode_name)
        reviewer = self._mode.approvals_reviewer
        return ToolGateSnapshot(
            user_ruleset=self._ruleset_for_mode(mode_name),
            approval_callback=self._approval_callback_for_mode(reviewer),
            approvals_reviewer=reviewer,
            permission_policy=self._permission_policy,
            approval_policy=self._approval_policy,
            tool_map={},
            restricted_dirs=self._restricted_dirs,
            hook_constraint_providers=self._hook_constraint_providers,
            project_root=self._project_root,
            external_directories=self._external_directories,
            sensitive_path_overrides=self._sensitive_path_overrides,
            session_grant_store=self._resolve_session_store(),
            permanent_grant_store=self._permanent_grant_store,
            mode_ruleset=default_rules,
            mode_fallback=fallback,
            shell_unresolved_policy=shell_unresolved_policy,
            tool_path_extractors=dict(self._tool_path_extractors),
        )

    def snapshot_for(self, registry: tuple[ToolSpec, ...]) -> ToolGateSnapshot:
        """为单个 turn 创建包含工具映射的门控快照。"""
        mode_name = self._mode.current_mode
        default_rules = self._default_ruleset_for_mode(mode_name)
        fallback = self._fallback_for_mode(mode_name)
        shell_unresolved_policy = self._shell_unresolved_policy_for_mode(mode_name)
        reviewer = self._mode.approvals_reviewer
        return ToolGateSnapshot(
            user_ruleset=self._ruleset_for_mode(mode_name),
            approval_callback=self._approval_callback_for_mode(reviewer),
            approvals_reviewer=reviewer,
            permission_policy=self._permission_policy,
            approval_policy=self._approval_policy,
            tool_map={tool.name: tool for tool in registry},
            restricted_dirs=self._restricted_dirs,
            hook_constraint_providers=self._hook_constraint_providers,
            project_root=self._project_root,
            external_directories=self._external_directories,
            sensitive_path_overrides=self._sensitive_path_overrides,
            session_grant_store=self._resolve_session_store(),
            permanent_grant_store=self._permanent_grant_store,
            mode_ruleset=default_rules,
            mode_fallback=fallback,
            shell_unresolved_policy=shell_unresolved_policy,
            tool_path_extractors=dict(self._tool_path_extractors),
        )

    def adapt_tools(self, registry: tuple[ToolSpec, ...]) -> list[AgentTool]:
        missing_schema = [spec.name for spec in registry if spec.schema is None]
        if missing_schema:
            names = ", ".join(sorted(missing_schema))
            raise ValueError(f"tools must define JSON schemas: {names}")
        return [_RedactingAdapter(spec) for spec in registry]

    @property
    def current_approval_callback(self) -> ApprovalCallback | None:
        """返回当前 execution mode 实际使用的审批回调。"""
        return self._approval_callback_for_mode(self._mode.approvals_reviewer)

    @property
    def user_approval_callback(self) -> ApprovalCallback | None:
        return self._user_approval_callback

    @property
    def auto_approval_callback(self) -> ApprovalCallback | None:
        return self._auto_approval_callback

    @property
    def approval_policy(self) -> ApprovalPolicy:
        """返回 ask 的会话处理策略。"""
        return self._approval_policy

    @property
    def approvals_reviewer(self) -> ApprovalsReviewer:
        """返回当前 execution mode 的审批者类型。"""
        return self._mode.approvals_reviewer

    @property
    def correlation(self) -> RuntimeCorrelation:
        """返回这个 gate 自己的运行关联域。"""
        return self._correlation

    def set_user_approval_callback(
        self, approval_callback: ApprovalCallback | None
    ) -> None:
        """更新 Act 等人工审批模式使用的回调。"""
        self._user_approval_callback = approval_callback

    def set_permission_policy(self, policy: PermissionPolicy | None) -> None:
        """应用新 composition generation 的静态权限策略。"""
        self._permission_policy = policy

    def set_session_grant_store_provider(
        self, provider: Callable[[], GrantStore | None] | None
    ) -> None:
        """设置或清除 session grant store provider。"""
        self._session_grant_store_provider = provider

    def set_permanent_grant_store(self, store: GrantStore | None) -> None:
        """设置或清除 permanent grant store。"""
        self._permanent_grant_store = store

    def clear_session_grants(self) -> None:
        """清空当前 session grant store。"""
        store = self._resolve_session_store()
        if isinstance(store, _ClearableGrantStore):
            store.clear()

    def fork_for_subagent(
        self,
        session_id: str | None = None,
        hook_manager: HookManager | None = None,
    ) -> ToolGate:
        """派生独立 correlation、共享权限配置和 grant 的子代理门控。"""
        child_session_id = session_id or self._session_id
        return ToolGate(
            mode_state=self._mode,
            user_approval_callback=self._user_approval_callback,
            auto_approval_callback=self._auto_approval_callback,
            permission_policy=self._permission_policy,
            approval_policy=self._approval_policy,
            hook_manager=hook_manager,
            audit_logger=self._audit_logger,
            session_id=child_session_id,
            external_hook_runner=self._external_hook_runner,
            external_hooks_subagent=True,
            external_hooks_cwd=self._external_hooks_cwd,
            correlation=RuntimeCorrelation(child_session_id),
            restricted_dirs=self._restricted_dirs,
            hook_constraint_providers=self._hook_constraint_providers,
            project_root=self._project_root,
            external_directories=self._external_directories,
            sensitive_path_overrides=self._sensitive_path_overrides,
            session_grant_store=self._session_grant_store,
            session_grant_store_provider=self._session_grant_store_provider,
            permanent_grant_store=self._permanent_grant_store,
            user_ruleset=self._user_ruleset,
            user_rulesets=self._user_rulesets,
            default_mode_rulesets=self._default_mode_rulesets,
            mode_fallbacks=self._mode_fallbacks,
            shell_unresolved_policies=self._shell_unresolved_policies,
            tool_path_extractors=self._tool_path_extractors,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    # ── 钩子构建 ──

    def build_before_tool_hook(
        self, snapshot: ToolGateSnapshot
    ) -> Callable[
        [BeforeToolCallContext, CancellationSignal | None],
        BeforeToolCallResult | None,
    ]:
        def before_tool(
            ctx: BeforeToolCallContext, _signal: CancellationSignal | None
        ) -> BeforeToolCallResult | None:
            tool_call = ctx.tool_call
            args = ctx.args
            original_args = args

            decision = self._mode.check_call(
                ToolCall(id=tool_call.id, name=tool_call.name, input=args)
            )
            args, decision = self._apply_external_pre_hooks(
                tool_call.name,
                args,
                decision,
                tool_call.id,
            )
            permission_result = self._precheck_permission(
                tool_call.name,
                args,
                decision,
                snapshot,
                tool_call.id,
                approval_transcript=_approval_transcript(ctx),
                approval_turn_id=self._correlation.turn_id,
            )
            if permission_result is not None:
                perm_result = self._last_perm_results.pop(tool_call.id, None)
                if perm_result is not None:
                    self._audit_blocked(
                        tool_call,
                        args,
                        perm_result,
                    )
                return permission_result

            ctx.args = args
            emit_hook(
                self._hook_manager,
                HookRecord(
                    "pre_tool",
                    tool=tool_call.name,
                    input=stringify_tool_input(args),
                    **self._hook_correlation_fields(tool_call.id),
                ),
            )
            if args != original_args:
                return BeforeToolCallResult(args=args)
            return None

        return before_tool

    def build_after_tool_hook(
        self, snapshot: ToolGateSnapshot
    ) -> Callable[
        [AfterToolCallContext, CancellationSignal | None],
        AfterToolCallResult | None,
    ]:
        def after_tool(
            ctx: AfterToolCallContext, _signal: CancellationSignal | None
        ) -> AfterToolCallResult | None:
            if ctx.tool_call.name in self.PROGRESS_TOOL_NAMES:
                self._progress_steps_without_update = 0
            action_input = stringify_tool_input(ctx.args)
            result_text = tool_result_text(ctx)
            emit_tool_hook(
                self._hook_manager,
                ctx,
                action_input,
                result_text,
                self._hook_correlation_fields(ctx.tool_call.id),
            )
            perm_result = self._last_perm_results.pop(ctx.tool_call.id, None)
            permission_notice = _permission_notice(perm_result)
            if permission_notice is not None:
                details = (
                    dict(ctx.result.details)
                    if isinstance(ctx.result.details, dict)
                    else {}
                )
                details["permission_notice"] = permission_notice
                ctx.result.details = details
            emit_audit(
                self._audit_logger,
                self._session_id,
                ctx,
                action_input,
                result_text,
                perm_result=perm_result,
                correlation=self._hook_correlation_fields(ctx.tool_call.id),
            )
            return None

        return after_tool

    def build_is_tool_productive_hook(
        self, snapshot: ToolGateSnapshot
    ) -> IsToolProductiveHook:
        def is_productive(
            tool_calls: list[ToolCallContent],
            tool_results: list[Any],
        ) -> bool:
            if self._mode.current_mode == "plan":
                return True
            return _tool_results_count_as_progress(
                [
                    ToolCall(id="", name=tc.name, input=tc.arguments or {})
                    for tc in tool_calls
                ],
                tool_results,
                snapshot.tool_map,
            )

        return is_productive

    # ── 进度跟踪 ──

    def check_progress_reminder(self) -> bool:
        """检查是否需要发送进度提醒。返回 True 表示应发送提醒。"""
        self._progress_steps_without_update += 1
        if self._progress_steps_without_update >= 5:
            self._progress_steps_without_update = 0
            return True
        return False

    # ── 内部方法 ──

    def _apply_external_pre_hooks(
        self,
        tool_name: str,
        args: dict[str, Any],
        decision: PermissionDecision,
        tool_call_id: str,
    ) -> tuple[dict[str, Any], PermissionDecision]:
        """应用参数变换，并仅允许 hook 收紧准入决策。"""
        runner = self._external_hook_runner
        if runner is None:
            return args, decision
        executions = runner.execute(
            HookRecord(
                "pre_tool",
                tool=tool_name,
                input=stringify_tool_input(args),
                **self._hook_correlation_fields(tool_call_id),
            ),
            subagent=self._external_hooks_subagent,
            cwd=self._external_hooks_cwd,
        )
        transformed_args = args
        effective_decision = decision
        for execution in executions:
            if execution.status != "succeeded":
                continue
            response_args = execution.response.get("arguments")
            if isinstance(response_args, dict):
                transformed_args = cast(dict[str, Any], response_args)
            response_decision = execution.response.get("decision")
            if response_decision in {"allow", "deny", "ask"}:
                effective_decision = _stricter_decision(
                    effective_decision,
                    cast(PermissionDecision, response_decision),
                )
        return transformed_args, effective_decision

    def _hook_correlation_fields(self, tool_call_id: str = "") -> HookCorrelationFields:
        """返回当前工具 hook 的共享关联字段。"""
        return hook_correlation_fields(self._correlation.snapshot(tool_call_id))

    def _audit_blocked(
        self,
        tool_call: ToolCallContent,
        args: dict[str, Any],
        perm_result: PermissionEngineResult,
    ) -> None:
        if self._audit_logger is None:
            return
        action_input = stringify_tool_input(args)
        correlation = self._hook_correlation_fields(tool_call.id)
        self._audit_logger(
            build_audit_record(
                session_id=self._session_id,
                tool_call=tool_call,
                action_input=action_input,
                result_text="",
                final_status="blocked",
                perm_result=perm_result,
                correlation=correlation,
            )
        )

    def _precheck_permission(
        self,
        tool_name: str,
        args: dict[str, Any],
        execution_decision: PermissionDecision,
        snapshot: ToolGateSnapshot,
        tool_call_id: str,
        *,
        approval_transcript: str = "",
        approval_turn_id: str = "",
    ) -> BeforeToolCallResult | None:
        action_profiles: dict[str, tuple[str, str]] = {}
        path_extractors = dict(snapshot.tool_path_extractors)
        for spec in snapshot.tool_map.values():
            profile = spec.action_profile or _TOOL_ACTION_PROFILES.get(spec.name)
            if profile is not None:
                action_profiles[spec.name] = profile
            if spec.path_extractor is not None:
                path_extractors.setdefault(spec.name, spec.path_extractor)
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=snapshot.permission_policy,
                restricted_dirs=snapshot.restricted_dirs,
                hook_constraint_providers=snapshot.hook_constraint_providers,
                project_root=snapshot.project_root,
                external_directories=snapshot.external_directories,
                sensitive_path_overrides=snapshot.sensitive_path_overrides,
                session_grant_store=snapshot.session_grant_store,
                permanent_grant_store=snapshot.permanent_grant_store,
                tool_action_profiles=action_profiles,
                tool_path_extractors=path_extractors,
                mode_ruleset=snapshot.mode_ruleset,
                user_ruleset=snapshot.user_ruleset,
                mode_fallback=snapshot.mode_fallback,
                shell_unresolved_policy=snapshot.shell_unresolved_policy,
                approval_policy=snapshot.approval_policy,
            )
        )
        result = engine.decide(
            tool_name,
            args,
            execution_decision=execution_decision,
            tool_spec=snapshot.tool_map.get(tool_name),
            approval_callback=snapshot.approval_callback,
            approvals_reviewer=snapshot.approvals_reviewer,
            approval_transcript=approval_transcript,
            approval_turn_id=approval_turn_id,
        )
        self._last_perm_results[tool_call_id] = result
        if result.blocked:
            suggestion = result.remediation or ""
            if result.metadata:
                suggestion = str(result.metadata.get("suggestion", suggestion))
            return BeforeToolCallResult(
                block=True, reason=result.reason, suggestion=suggestion
            )
        return None


# ── 模块级辅助 ──


def _approval_transcript(ctx: BeforeToolCallContext) -> str:
    """构建 reviewer 使用的有界会话证据，保留角色和信任边界。"""
    entries: list[str] = []
    if ctx.context.system_prompt.strip():
        entries.append(
            _format_transcript_entry("system", ctx.context.system_prompt, 8_000)
        )
    messages = [
        *ctx.context.request_prefix,
        *ctx.context.messages,
        ctx.assistant_message,
    ]
    for message in messages[-40:]:
        role = str(getattr(message, "role", "unknown"))
        content = _transcript_message_text(message)
        if not content or (role == "user" and content.casefold() == "continue"):
            continue
        per_entry_limit = 4_000 if role == "tool_result" else 8_000
        entries.append(_format_transcript_entry(role, content, per_entry_limit))
    return _truncate_transcript("\n\n".join(entries), 40_000)


def _transcript_message_text(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, TextContent) and block.text:
                parts.append(block.text)
                continue
            if isinstance(block, ToolCallContent):
                arguments = json.dumps(
                    block.arguments or {},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                parts.append(f"tool_call {block.name}: {arguments}")
                continue
            text = getattr(block, "content", None) or getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip()
    summary = getattr(message, "summary", None)
    if isinstance(summary, str):
        return summary.strip()
    return ""


def _format_transcript_entry(role: str, content: str, limit: int) -> str:
    trust = "trusted" if role in {"system", "user"} else "untrusted"
    return f"<{role} trust={trust}>\n{_truncate_transcript(content, limit)}\n</{role}>"


def _truncate_transcript(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "<truncated />"
    remaining = max(0, limit - len(marker))
    prefix = remaining // 2
    suffix = remaining - prefix
    return f"{text[:prefix]}{marker}{text[-suffix:]}"


def _tool_results_count_as_progress(
    tool_uses: list[ToolCall],
    tool_results: list[Any],
    tool_map: dict[str, ToolSpec],
) -> bool:
    del tool_map
    if not tool_uses or not tool_results:
        return False
    results_by_id = {
        str(getattr(result, "tool_call_id", "")): result for result in tool_results
    }
    for tool_use, positional_result in zip(tool_uses, tool_results):
        result = results_by_id.get(tool_use.id, positional_result)
        is_error = getattr(result, "is_error", None)
        if isinstance(is_error, bool) and not is_error:
            return True
        if is_error is None and getattr(result, "status", None) == "ok":
            return True
    return False


def _stricter_decision(
    current: PermissionDecision,
    proposed: PermissionDecision,
) -> PermissionDecision:
    """合并准入决策，禁止外部 hook 放宽现有约束。"""
    priority: dict[PermissionDecision, int] = {
        "allow": 0,
        "ask": 1,
        "deny": 2,
    }
    return proposed if priority[proposed] > priority[current] else current


def _permission_notice(result: PermissionEngineResult | None) -> str | None:
    """为自动命中的 grant 生成用户可见说明。"""
    if result is None or result.approval_result is None:
        return None
    if result.approval_result.decision != "allow":
        return None
    metadata = result.metadata or {}
    if metadata.get("approval_reviewer") == "auto_review":
        rationale = str(metadata.get("approval_rationale") or "").strip()
        risk = str(metadata.get("approval_risk") or "unknown")
        authorization = str(metadata.get("approval_authorization") or "unknown")
        prefix = (
            "Automatic approval review approved "
            f"(risk: {risk}, authorization: {authorization})"
        )
        if rationale:
            return f"{prefix}: {rationale}"
        return prefix
    if result.matched_rule == "session_grant":
        return "Allowed by session grant"
    if result.matched_rule == "persistent_grant":
        return "Allowed by permanent grant"
    return None
