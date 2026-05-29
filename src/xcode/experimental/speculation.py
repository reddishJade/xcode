from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculationEvent:
    kind: str
    reason: str
    tool: str | None = None


class FinishKindTracker:
    """识别最近一次可观察 Agent 步骤，用于安全的界面预热。"""

    def classify(self, tool_name: str | None, status: str) -> str:
        if status in {"error", "denied", "approval_required", "interrupted"}:
            return status
        if tool_name == "bash":
            return "bash"
        if tool_name == "edit_file":
            return "file_edit"
        if tool_name:
            return "tool_use"
        return "normal_text"


class SpeculationPlanner:
    """规划安全的本地准备动作，不执行有副作用的工具。"""

    def __init__(self, tracker: FinishKindTracker | None = None) -> None:
        self.tracker = tracker or FinishKindTracker()

    def plan(self, tool_name: str | None, status: str) -> SpeculationEvent | None:
        finish_kind = self.tracker.classify(tool_name, status)
        if finish_kind == "file_edit":
            return SpeculationEvent(
                "prepare_diff_view", "file edit completed", tool_name
            )
        if finish_kind == "bash":
            return SpeculationEvent(
                "prepare_terminal_buffer", "bash completed", tool_name
            )
        if finish_kind in {"error", "denied", "approval_required", "interrupted"}:
            return SpeculationEvent("prepare_recovery_hint", finish_kind, tool_name)
        return None
