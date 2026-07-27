"""工具执行门控：审批、权限、准入决策。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
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
        return AgentToolResult(
            content=[TextContent(text=redact_text(str(content)))],
            details=metadata if isinstance(metadata, dict) else None,
            is_error=bool(getattr(content, "is_error", False)),
        )


@dataclass(frozen=True)
class ToolGateSnapshot:
    """ToolGate 在单个 turn 中使用的冻结配置。"""

    approval_callback: ApprovalCallback | None
    permission_policy: PermissionPolicy | None
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
        approval_callback: ApprovalCallback | None,
        permission_policy: PermissionPolicy | None,
        hook_manager: HookManager | None,
        audit_logger: AuditLogger | None,
        session_id: str,
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
        user_rulesets: dict[str, tuple[Rule, ...]] | None = None,
        default_mode_rulesets: dict[str, tuple[Rule, ...]] | None = None,
        mode_fallbacks: dict[str, PermissionDecision] | None = None,
        shell_unresolved_policies: dict[str, PermissionDecision] | None = None,
        tool_path_extractors: dict[str, PathExtractor] | None = None,
    ) -> None:
        self._mode = mode_state
        self._user_ruleset = user_ruleset
        self._user_rulesets = user_rulesets or {}
        self._default_mode_rulesets = default_mode_rulesets or {}
        self._mode_fallbacks = mode_fallbacks or {}
        self._shell_unresolved_policies = shell_unresolved_policies or {}
        self._tool_path_extractors = tool_path_extractors or {}
        self._approval_callback = approval_callback
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

    def snapshot(self) -> ToolGateSnapshot:
        mode_name = self._mode.current_mode
        default_rules = self._default_ruleset_for_mode(mode_name)
        fallback = self._fallback_for_mode(mode_name)
        shell_unresolved_policy = self._shell_unresolved_policy_for_mode(mode_name)
        return ToolGateSnapshot(
            user_ruleset=self._ruleset_for_mode(mode_name),
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
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
            tool_path_extractors=self._tool_path_extractors,
        )

    def snapshot_for(self, registry: tuple[ToolSpec, ...]) -> ToolGateSnapshot:
        """为单个 turn 创建包含工具映射的门控快照。"""
        mode_name = self._mode.current_mode
        default_rules = self._default_ruleset_for_mode(mode_name)
        fallback = self._fallback_for_mode(mode_name)
        shell_unresolved_policy = self._shell_unresolved_policy_for_mode(mode_name)
        return ToolGateSnapshot(
            user_ruleset=self._ruleset_for_mode(mode_name),
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
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
            tool_path_extractors=self._tool_path_extractors,
        )

    def adapt_tools(self, registry: tuple[ToolSpec, ...]) -> list[AgentTool]:
        missing_schema = [spec.name for spec in registry if spec.schema is None]
        if missing_schema:
            names = ", ".join(sorted(missing_schema))
            raise ValueError(f"tools must define JSON schemas: {names}")
        return [_RedactingAdapter(spec) for spec in registry]

    @property
    def approval_callback(self) -> ApprovalCallback | None:
        """返回当前 HITL 审批回调。"""
        return self._approval_callback

    def set_approval_callback(self, approval_callback: ApprovalCallback | None) -> None:
        """更新后续工具适配与前置检查使用的 HITL 回调。"""
        self._approval_callback = approval_callback

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

    def fork_for_subagent(self) -> ToolGate:
        """派生隔离运行状态、共享权限配置和 grant 的子代理门控。"""
        return ToolGate(
            mode_state=self._mode,
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
            hook_manager=self._hook_manager,
            audit_logger=self._audit_logger,
            session_id=self._session_id,
            external_hook_runner=self._external_hook_runner,
            external_hooks_subagent=True,
            external_hooks_cwd=self._external_hooks_cwd,
            correlation=self._correlation,
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
                tool_call.name, args, decision, snapshot, tool_call.id
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
    ) -> BeforeToolCallResult | None:
        action_profiles: dict[str, tuple[str, str]] = {}
        for spec in snapshot.tool_map.values():
            profile = _TOOL_ACTION_PROFILES.get(spec.name)
            if profile is not None:
                action_profiles[spec.name] = profile
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
                tool_path_extractors=snapshot.tool_path_extractors,
                mode_ruleset=snapshot.mode_ruleset,
                user_ruleset=snapshot.user_ruleset,
                mode_fallback=snapshot.mode_fallback,
                shell_unresolved_policy=snapshot.shell_unresolved_policy,
            )
        )
        result = engine.decide(
            tool_name,
            args,
            execution_decision=execution_decision,
            tool_spec=snapshot.tool_map.get(tool_name),
            approval_callback=snapshot.approval_callback,
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


def _tool_results_count_as_progress(
    tool_uses: list[ToolCall],
    tool_results: list[Any],
    tool_map: dict[str, ToolSpec],
) -> bool:
    for _, tool_result in zip(tool_uses, tool_results, strict=True):
        is_ok = (hasattr(tool_result, "is_error") and not tool_result.is_error) or (
            hasattr(tool_result, "status") and tool_result.status == "ok"
        )
        if not is_ok:
            continue
    return True


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
    if result.matched_rule == "session_grant":
        return "Allowed by session grant"
    if result.matched_rule == "persistent_grant":
        return "Allowed by permanent grant"
    return None
