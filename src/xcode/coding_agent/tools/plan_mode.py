"""Plan mode 工具：让 AI 主动请求进入/退出 plan mode。"""

from __future__ import annotations

from pathlib import Path

from xcode.harness.skills import ToolInput, ToolSpec
from xcode.harness.observability.permissions import PermissionDecision


def build_plan_mode_tools(project_root: Path) -> tuple[ToolSpec, ...]:
    """构建 plan mode 相关工具。

    - request_plan_mode: AI 主动请求进入 plan mode（需用户批准）
    - exit_plan_mode: AI 完成计划，请求退出 plan mode 并提交审核
    """
    _ = project_root  # 未来可能需要用到项目路径

    def request_plan_mode(data: ToolInput) -> str:
        reason = str(data.get("reason", "")).strip()
        if not reason:
            raise ValueError("reason is required")

        # 此函数只有在用户批准后才会执行
        return (
            f"Plan mode request approved by user.\n"
            f"Reason: {reason}\n\n"
            f"You may now focus on reading code, understanding requirements, "
            f"and designing the implementation approach. "
            f"File modifications and shell commands are restricted in plan mode."
        )

    def exit_plan_mode(data: ToolInput) -> str:
        plan = str(data.get("plan", "")).strip()
        if not plan:
            raise ValueError("plan is required")

        return (
            f"Plan submitted for review.\n\n"
            f"The plan has been presented to the user. "
            f"They will decide whether to:\n"
            f"  1) Approve and proceed to implementation\n"
            f"  2) Request modifications\n"
            f"  3) Reject the plan\n\n"
            f"Plan content:\n{plan}"
        )

    def _plan_mode_risk_evaluator(tool_input: dict) -> PermissionDecision:
        """request_plan_mode 必须经过用户确认。"""
        return "ask"

    return (
        ToolSpec(
            name="request_plan_mode",
            description=(
                "Request user permission to enter plan mode for complex tasks. "
                "Use when the task involves: architectural changes, multi-step refactoring, "
                "multiple valid approaches, or needs user agreement on the approach before implementation. "
                "This tool requires user approval and cannot be executed automatically."
            ),
            input_hint='JSON: {"reason": "This task requires architectural design decisions"}',
            handler=request_plan_mode,
            risk="high",
            risk_evaluator=_plan_mode_risk_evaluator,
            read_only=True,
            schema={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explanation of why plan mode is needed for this task.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            group="core",
        ),
        ToolSpec(
            name="exit_plan_mode",
            description=(
                "Submit the completed plan for user review and approval. "
                "Use when you have finished designing the implementation approach and "
                "are ready to get user approval before proceeding to implementation."
            ),
            input_hint='JSON: {"plan": "## Implementation Plan\\n\\n1. ..."}',
            handler=exit_plan_mode,
            risk="low",
            read_only=True,
            schema={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The complete implementation plan in markdown format.",
                    }
                },
                "required": ["plan"],
                "additionalProperties": False,
            },
            group="core",
        ),
    )
