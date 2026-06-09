"""工具执行门控：审批、权限、钩子、审计。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xcode.ai.events import ToolCall as ToolUseBlock

from ...agent.config import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    IsToolProductiveHook,
)
from ...agent.types import ToolCallContent
from .execution_modes import policy_for_mode
from .mode_manager import ModeManager
from ..observability import AuditRecord, HookManager, HookRecord, PermissionPolicy, redact_text
from ..skills import ApprovalCallback, ToolSpec, stringify_tool_input


@dataclass(frozen=True)
class ToolGateSnapshot:
    """ToolGate 在单个 turn 中使用的冻结配置。"""

    approval_callback: ApprovalCallback | None
    permission_policy: PermissionPolicy | None
    high_risk_requires_approval: bool
    tool_map: dict[str, ToolSpec]


class ToolGate:
    """工具执行门控：HITL 审批、权限检查、钩子发射、审计记录。"""

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
        mode_manager: ModeManager,
        approval_callback: ApprovalCallback | None,
        permission_policy: PermissionPolicy | None,
        high_risk_requires_approval: bool,
        hook_manager: HookManager | None,
        audit_logger: Callable[[AuditRecord], None] | None,
        session_id: str,
    ) -> None:
        self._mode = mode_manager
        self._approval_callback = approval_callback
        self._permission_policy = permission_policy
        self._high_risk_requires_approval = high_risk_requires_approval
        self._hook_manager = hook_manager
        self._audit_logger = audit_logger
        self._session_id = session_id
        self._progress_steps_without_update: int = 0

    def snapshot(self) -> ToolGateSnapshot:
        return ToolGateSnapshot(
            approval_callback=self._approval_callback,
            permission_policy=self._permission_policy,
            high_risk_requires_approval=self._high_risk_requires_approval,
            tool_map={},
        )

    # ── 钩子构建 ──

    def build_before_tool_hook(
        self, snapshot: ToolGateSnapshot
    ) -> Callable[[BeforeToolCallContext, Any], BeforeToolCallResult | None]:
        def before_tool(
            ctx: BeforeToolCallContext, _signal: Any
        ) -> BeforeToolCallResult | None:
            tool_call = ctx.tool_call
            args = ctx.args

            if self._mode.is_plan_confirmation_required(tool_call.name, args):
                self._mode.confirm_plan()
                return BeforeToolCallResult(
                    block=True,
                    reason=(
                        f"tool {tool_call.name} requires plan confirmation. "
                        "Present the plan to the user for approval before executing write tools."
                    ),
                )

            effective_policy = policy_for_mode(self._mode.current_mode)
            decision = effective_policy.check_call(
                ToolUseBlock(id=tool_call.id, name=tool_call.name, input=args)
            )
            if decision == "deny":
                return BeforeToolCallResult(
                    block=True,
                    reason=f"tool not allowed in {self._mode.current_mode} mode: {tool_call.name}",
                )
            if decision == "ask":
                approval = self._request_approval(tool_call.name, args, snapshot)
                if approval is not None:
                    return approval

            self._emit_hook(
                HookRecord("pre_tool", tool=tool_call.name, input=stringify_tool_input(args))
            )
            return None

        return before_tool

    def build_after_tool_hook(
        self, snapshot: ToolGateSnapshot
    ) -> Callable[[AfterToolCallContext, Any], AfterToolCallResult | None]:
        def after_tool(
            ctx: AfterToolCallContext, _signal: Any
        ) -> AfterToolCallResult | None:
            if ctx.tool_call.name in self.PROGRESS_TOOL_NAMES:
                self._progress_steps_without_update = 0
            action_input = stringify_tool_input(ctx.args)
            result_text = _tool_result_text(ctx)
            self._emit_tool_hook(ctx, action_input, result_text)
            self._emit_audit(ctx, action_input, result_text, snapshot)
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
                    ToolUseBlock(id="", name=tc.name, input=tc.arguments or {})
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

    def _request_approval(
        self, tool_name: str, args: dict[str, Any], snapshot: ToolGateSnapshot
    ) -> BeforeToolCallResult | None:
        if snapshot.approval_callback is None or tool_name not in snapshot.tool_map:
            return BeforeToolCallResult(
                block=True, reason=f"tool requires approval: {tool_name}"
            )
        hitl = snapshot.approval_callback(snapshot.tool_map[tool_name], args)
        if hitl.decision == "deny":
            return BeforeToolCallResult(
                block=True, reason=f"tool {tool_name} denied by user"
            )
        return None

    def _emit_tool_hook(
        self,
        ctx: AfterToolCallContext,
        action_input: str,
        result_text: str,
    ) -> None:
        tool_call = ctx.tool_call
        if ctx.is_error:
            self._emit_hook(
                HookRecord("on_error", tool=tool_call.name, input=action_input, error=result_text)
            )
            return
        self._emit_hook(
            HookRecord("post_tool", tool=tool_call.name, input=action_input, output=result_text)
        )

    def _emit_audit(
        self,
        ctx: AfterToolCallContext,
        action_input: str,
        result_text: str,
        snapshot: ToolGateSnapshot,
    ) -> None:
        if self._audit_logger is None:
            return
        tool_call = ctx.tool_call
        spec = snapshot.tool_map.get(tool_call.name)
        self._audit_logger(
            AuditRecord(
                session_id=self._session_id,
                tool=tool_call.name,
                static_risk=(spec.risk if spec else None) or "low",
                dynamic_decision="allow",
                policy_decision=None,
                final_status="error" if ctx.is_error else "ok",
                approved=True,
                redacted_input=redact_text(action_input),
                redacted_output=redact_text(result_text),
            )
        )

    def _emit_hook(self, record: HookRecord) -> None:
        if self._hook_manager is not None:
            self._hook_manager.emit(record)


# ── 模块级辅助 ──


def _tool_result_text(ctx: AfterToolCallContext) -> str:
    """从 AfterToolCallContext 提取结果文本。"""
    from ...agent.types import TextContent

    if not ctx.result or not ctx.result.content:
        return ""
    return "".join(c.text for c in ctx.result.content if isinstance(c, TextContent))


def _tool_results_count_as_progress(
    tool_uses: list[ToolUseBlock],
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
