from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

"""当前任务上下文的轻量记录与提示词渲染。"""


@dataclass(frozen=True)
class RecentToolResult:
    tool: str
    summary: str


@dataclass(frozen=True)
class RecentToolCall:
    tool: str
    input_brief: str
    status: str
    risk: str
    approval_scope: str | None
    target_path: str | None
    timestamp: str


class ContextualRetrievalState:
    """记录已明确相关的文件和工具结果，供下一轮 system context 使用。"""

    def __init__(
        self,
        project_root: Path,
        # 上下文预算限制（基于 system prompt token 预算设计）
        max_files: int = 8,  # 约占 200 token 预算
        max_results: int = 6,  # 约占 150 token 预算
        max_tool_calls: int = 8,  # 约占 200 token 预算
    ) -> None:
        self.project_root = project_root.resolve()
        self.max_files = max_files
        self.max_results = max_results
        self._files: deque[str] = deque()
        self._file_set: set[str] = set()
        self._active_file: str | None = None
        self._tool_results: deque[RecentToolResult] = deque(maxlen=max_results)
        self._tool_calls: deque[RecentToolCall] = deque(maxlen=max_tool_calls)
        self._render_cache: str | None = None
        self._dirty = True

    def record_file(self, path: Path | str) -> None:
        """记录文件为当前上下文相关，去重并维护 LRU 队列。"""
        text = self._display(path)
        if not text:
            return
        if text in self._file_set:
            self._files.remove(text)
        else:
            self._file_set.add(text)
        self._files.append(text)
        self._active_file = text
        while len(self._files) > self.max_files:
            removed = self._files.popleft()
            self._file_set.discard(removed)
            if self._active_file == removed:
                self._active_file = self._files[-1] if self._files else None
        self._dirty = True

    def record_tool_result(self, tool: str, content: str) -> None:
        """记录工具结果摘要，用于下一轮 system prompt。"""
        clean = " ".join(content.strip().split())
        if not clean:
            return
        if len(clean) > 240:
            clean = clean[:237] + "..."
        self._tool_results.append(RecentToolResult(tool=tool, summary=clean))
        self._dirty = True

    def record_tool_call(
        self,
        *,
        tool: str,
        input_brief: str,
        status: str,
        risk: str,
        approval_scope: str | None = None,
        target_path: str | None = None,
    ) -> None:
        """记录工具调用历史，包含状态、风险级别和审批范围。"""
        clean_input = " ".join(input_brief.strip().split())
        if len(clean_input) > 160:
            clean_input = clean_input[:157] + "..."
        self._tool_calls.append(
            RecentToolCall(
                tool=tool,
                input_brief=clean_input,
                status=status,
                risk=risk,
                approval_scope=approval_scope,
                target_path=target_path,
                timestamp=datetime.now().isoformat(timespec="seconds"),
            )
        )
        self._dirty = True

    def render(self) -> str:
        """渲染为 system prompt 的 contextual-retrieval 块。"""
        if not self._dirty and self._render_cache is not None:
            return self._render_cache
        lines = [
            "<contextual-retrieval>",
            "This block contains only context already made relevant by the current task.",
            "Use it to orient tool choices; do not treat it as a replacement for exact search or file reads.",
            "For ambiguous references such as 'this file' or 'it', prefer active_file as the first candidate, then verify before editing.",
        ]
        if self._active_file:
            lines.append(f"active_file: {self._active_file}")
        if self._files:
            lines.append("recent_files:")
            lines.extend(f"- {path}" for path in self._files)
        if self._tool_results:
            lines.append("recent_tool_results:")
            lines.extend(
                f"- {result.tool}: {result.summary}" for result in self._tool_results
            )
        if self._tool_calls:
            lines.append("recent_tool_calls:")
            for call in self._tool_calls:
                approval = (
                    f" approval={call.approval_scope}" if call.approval_scope else ""
                )
                target = f" target={call.target_path}" if call.target_path else ""
                lines.append(
                    f"- {call.tool} status={call.status} risk={call.risk}{approval}{target}: {call.input_brief}"
                )
        lines.append("</contextual-retrieval>")
        rendered = "\n".join(lines)
        self._render_cache = rendered
        self._dirty = False
        return rendered

    def _display(self, path: Path | str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return ""
