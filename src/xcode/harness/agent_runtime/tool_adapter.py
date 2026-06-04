"""ToolSpec → AgentTool 适配器。

将 harness 层的 ToolSpec 适配为 agent 层的 AgentTool protocol，
使 StructuredAgent 可以将 ToolSpec 注册表传给 agent 核心循环。
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...agent.types import (
    AgentToolResult,
    CancellationSignal,
    TextContent,
    ToolExecutionMode,
    ToolUpdateCallback,
)
from ..skills import ToolSpec, stringify_tool_input
from ..observability import (
    PermissionCheckResult,
    PermissionPolicy,
    check_tool_permission,
    redact_text,
)


class ToolSpecAdapter:
    """将 harness ToolSpec 适配为 agent AgentTool protocol。

    依赖方向：harness -> agent（正确）。
    此类在 harness/ 层，实现 agent/ 层定义的 protocol。
    """

    def __init__(
        self,
        spec: ToolSpec,
        *,
        approval_callback: Any | None = None,
        permission_policy: PermissionPolicy | None = None,
    ) -> None:
        self._spec = spec
        self._approval_callback = approval_callback
        self._permission_policy = permission_policy

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def label(self) -> str:
        return self._spec.name

    @property
    def description(self) -> str:
        return self._spec.description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._spec.schema or {"type": "object"}

    @property
    def examples(self) -> list[dict[str, Any]]:
        return list(self._spec.examples)

    @property
    def execution_mode(self) -> ToolExecutionMode | None:
        if self._spec.execution_mode is not None:
            return self._spec.execution_mode
        if (
            self._spec.read_only
            and self._spec.concurrency_safe
            and self._spec.risk != "high"
        ):
            return "parallel"
        return "sequential"

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        signal: CancellationSignal | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult[None]:
        action_input = stringify_tool_input(params)
        result: PermissionCheckResult = check_tool_permission(
            self._spec.name,
            action_input,
            permission_policy=self._permission_policy,
            approval_callback=self._approval_callback,
            tool_spec=self._spec,
            tool_input=params,
        )
        if result.blocked:
            return AgentToolResult(content=[TextContent(text=result.reason)])

        content = await asyncio.to_thread(self._spec.handler, params)
        return AgentToolResult(content=[TextContent(text=redact_text(content))])


def adapt_tool_specs(
    specs: tuple[ToolSpec, ...],
    *,
    approval_callback: Any | None = None,
    permission_policy: PermissionPolicy | None = None,
) -> list[ToolSpecAdapter]:
    """批量将 ToolSpec 适配为 AgentTool。"""
    return [
        ToolSpecAdapter(
            spec,
            approval_callback=approval_callback,
            permission_policy=permission_policy,
        )
        for spec in specs
    ]
