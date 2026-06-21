"""Hook 发射与工具回调组织。"""

from __future__ import annotations

from ...agent.config import AfterToolCallContext
from ...agent.types import TextContent
from ..observability import HookCorrelationFields, HookManager, HookRecord


def emit_hook(
    hook_manager: HookManager | None,
    record: HookRecord,
) -> None:
    if hook_manager is not None:
        hook_manager.emit(record)


def emit_tool_hook(
    hook_manager: HookManager | None,
    ctx: AfterToolCallContext,
    action_input: str,
    result_text: str,
    correlation: HookCorrelationFields,
) -> None:
    tool_call = ctx.tool_call
    if ctx.is_error:
        emit_hook(
            hook_manager,
            HookRecord(
                "on_error",
                tool=tool_call.name,
                input=action_input,
                error=result_text,
                **correlation,
            ),
        )
        return
    emit_hook(
        hook_manager,
        HookRecord(
            "post_tool",
            tool=tool_call.name,
            input=action_input,
            output=result_text,
            **correlation,
        ),
    )


def tool_result_text(ctx: AfterToolCallContext) -> str:
    if not ctx.result or not ctx.result.content:
        return ""
    return "".join(c.text for c in ctx.result.content if isinstance(c, TextContent))
