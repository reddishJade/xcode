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
        max_files: int = 8,
        max_results: int = 6,
        max_tool_calls: int = 8,
    ) -> None:
        self.project_root = project_root.resolve()
        self.max_files = max_files
        self.max_results = max_results
        self._files: deque[str] = deque()
        self._file_set: set[str] = set()
        self._tool_results: deque[RecentToolResult] = deque(maxlen=max_results)
        self._tool_calls: deque[RecentToolCall] = deque(maxlen=max_tool_calls)

    def record_file(self, path: Path | str) -> None:
        text = self._display(path)
        if not text or text in self._file_set:
            return
        self._files.append(text)
        self._file_set.add(text)
        while len(self._files) > self.max_files:
            removed = self._files.popleft()
            self._file_set.discard(removed)

    def record_tool_result(self, tool: str, content: str) -> None:
        clean = " ".join(content.strip().split())
        if not clean:
            return
        if len(clean) > 240:
            clean = clean[:237] + "..."
        self._tool_results.append(RecentToolResult(tool=tool, summary=clean))

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

    def render(self) -> str:
        lines = [
            "<contextual-retrieval>",
            "This block contains only context already made relevant by the current task.",
            "Use it to orient tool choices; do not treat it as a replacement for exact search or file reads.",
        ]
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
        return "\n".join(lines)

    def _display(self, path: Path | str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return ""
