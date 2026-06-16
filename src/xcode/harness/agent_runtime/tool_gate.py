"""工具执行门控：审批、权限、准入决策。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xcode.ai.events import ToolCall

from ...agent.config import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    IsToolProductiveHook,
)
from ...agent.protocols import AgentTool, CancellationSignal
from ...agent.types import ToolCallContent
from .execution_modes import ExecutionModeState, policy_for_mode
from .tool_adapter import adapt_tool_specs
from .tool_audit import emit_audit
from .tool_hooks import emit_hook, emit_tool_hook, tool_result_text
from ..observability import (
    AuditRecord,
    HookManager,
    HookRecord,
    PermissionEngine,
    PermissionEngineConfig,
    PermissionDecision,
    PermissionPolicy,
)
from ..observability.permission_model import (
    ExternalDirectory,
    GrantStore,
    PolicyEvaluator,
)
from ..skills import ApprovalCallback, ToolSpec, stringify_tool_input


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
    session_grant_store: GrantStore | None = None
    permanent_grant_store: GrantStore | None = None


class ToolGate:
    """工具执行门控：HITL 审批、权限检查、准入决策。"""

    PROGRESS_TOOL_NAMES = frozenset(
        {
            "save_task_progress",
            "resume_task_progress",
            "update_task",
            "create_task",
        }
    )

    def __init__(
        self,
        mode_state: ExecutionModeState,
        approval_callback: ApprovalCallback | None,
        permission_policy: PermissionPolicy | None,
        hook_manager: HookManager | None,
        audit_logger: Callable[[AuditRecord], None] | None,
        session_id: str,
        restricted_dirs: tuple[str, ...] = (),
        hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
        project_root: Path | None = None,
        external_directories: tuple[ExternalDirectory, ...] = (),
        session_grant_store: GrantStore | None = None,
        session_grant_store_provider: Callable[[], GrantStore | None] | None = None,
        permanent_grant_store: GrantStore | None = None,
    ) -> None:
        self._mode = mode_state
        self._approval_callback = approval_callback
        self._permission_policy = permission_policy
        self._restricted_dirs = restricted_dirs
        self._hook_constraint_providers = hook_constraint_providers
        self._external_directories = external_directories
        self._hook_manager = hook_manager
        self._audit_logger = audit_logger
        self._session_id = session_id
        self._project_root = project_root
        self._session_grant_store = session_grant_store
        self._session_grant_store_provider = session_grant_store_provider
        self._permanent_grant_store = permanent_grant_store
        self._progress_steps_without_update: int = 0

    def _resolve_session_store(self) -> GrantStore | None:
        if self._session_grant_store_provider is not None:
            return self._session_grant_store_provider()
        return self._session_grant_store

    def snapshot(self) -> ToolGateSnapshot:
        return ToolGateSnapshot(
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
            tool_map={},
            restricted_dirs=self._restricted_dirs,
            hook_constraint_providers=self._hook_constraint_providers,
            project_root=self._project_root,
            external_directories=self._external_directories,
            session_grant_store=self._resolve_session_store(),
            permanent_grant_store=self._permanent_grant_store,
        )

    def snapshot_for(self, registry: tuple[ToolSpec, ...]) -> ToolGateSnapshot:
        """为单个 turn 创建包含工具映射的门控快照。"""
        return ToolGateSnapshot(
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
            tool_map={tool.name: tool for tool in registry},
            restricted_dirs=self._restricted_dirs,
            hook_constraint_providers=self._hook_constraint_providers,
            project_root=self._project_root,
            external_directories=self._external_directories,
            session_grant_store=self._resolve_session_store(),
            permanent_grant_store=self._permanent_grant_store,
        )

    def adapt_tools(self, registry: tuple[ToolSpec, ...]) -> list[AgentTool]:
        """将 ToolSpec 注册表适配为 AgentTool。

        权限门控由 build_before_tool_hook 中的 _precheck_permission 处理，
        不在 ToolSpecAdapter 中执行。
        """
        return list(adapt_tool_specs(registry))

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

            effective_policy = policy_for_mode(self._mode.current_mode)
            decision = effective_policy.check_call(
                ToolCall(id=tool_call.id, name=tool_call.name, input=args)
            )
            permission_result = self._precheck_permission(
                tool_call.name, args, decision, snapshot
            )
            if permission_result is not None:
                return permission_result

            emit_hook(
                self._hook_manager,
                HookRecord(
                    "pre_tool",
                    tool=tool_call.name,
                    input=stringify_tool_input(args),
                ),
            )
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
            emit_tool_hook(self._hook_manager, ctx, action_input, result_text)
            emit_audit(
                self._audit_logger,
                self._session_id,
                ctx,
                action_input,
                result_text,
                snapshot.tool_map,
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

    def _precheck_permission(
        self,
        tool_name: str,
        args: dict[str, Any],
        execution_decision: PermissionDecision,
        snapshot: ToolGateSnapshot,
    ) -> BeforeToolCallResult | None:
        engine = PermissionEngine(
            PermissionEngineConfig(
                static_policy=snapshot.permission_policy,
                restricted_dirs=snapshot.restricted_dirs,
                hook_constraint_providers=snapshot.hook_constraint_providers,
                project_root=snapshot.project_root,
                external_directories=snapshot.external_directories,
                session_grant_store=snapshot.session_grant_store,
                permanent_grant_store=snapshot.permanent_grant_store,
            )
        )
        result = engine.decide(
            tool_name,
            stringify_tool_input(args),
            execution_decision=execution_decision,
            tool_spec=snapshot.tool_map.get(tool_name),
            tool_input=args,
            approval_callback=snapshot.approval_callback,
        )
        if result.blocked:
            return BeforeToolCallResult(block=True, reason=result.reason)
        return None


# ── 模块级辅助 ──


def _tool_results_count_as_progress(
    tool_uses: list[ToolCall],
    tool_results: list[Any],
    tool_map: dict[str, ToolSpec],
) -> bool:
    for tool_use, tool_result in zip(tool_uses, tool_results, strict=True):
        is_ok = (hasattr(tool_result, "is_error") and not tool_result.is_error) or (
            hasattr(tool_result, "status") and tool_result.status == "ok"
        )
        if not is_ok:
            continue
        spec = tool_map.get(tool_use.name)
        if spec and spec.counts_as_progress is not None:
            return spec.counts_as_progress
        if spec and spec.read_only:
            return True
    return False
