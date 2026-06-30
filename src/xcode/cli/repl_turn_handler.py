"""Turn-渲染专用的处理器：ToolCallHandler 和 ReasoningHandler。"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.text import Text

from .commands import ReplState, VerbosityLevel
from .repl_rendering import (
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_THINKING,
    CLI_COLOR_TOOL,
    LiveReasoningPreview,
    format_elapsed,
    reasoning_preview_lines,
    should_print_reasoning_summary,
    single_line_preview,
)
from .repl_tools import (
    brief_input,
    print_tool_call_rich,
    print_tool_result_rich,
    summarize_intents,
    tool_intent,
)
from xcode.ai.events import ToolCall
from xcode.harness.agent_runtime.events import ToolResultBlock, ToolUpdateData


def _safe_write(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        safe_text = (
            text.replace("•", "*").replace("×", "x").replace("✘", "x").replace("⊘", "o")
        )
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(safe_text.encode(encoding, errors="replace").decode(encoding))
        sys.stdout.flush()


class ToolCallHandler:
    """追踪工具调用、结果和更新，管理工具组聚合渲染和进度显示。"""

    def __init__(self, state: ReplState, live_console: Console) -> None:
        self.state = state
        self.live_console = live_console
        self.tool_group: dict[str, Any] | None = None
        self.tool_call_labels: dict[str, str] = {}
        self._progress_tool_id: str | None = None

    def record_tool_call(self, event_data: ToolCall) -> None:
        label = brief_input(event_data.name, event_data.input)
        intent = tool_intent(event_data.name, event_data.input)
        self.tool_call_labels[event_data.id] = label
        if self.state.verbosity != "normal":
            print_tool_call_rich(label, self.live_console)
            return
        if self.tool_group is None:
            self.tool_group = {
                "intents": [],
                "calls": 0,
                "ok": 0,
                "errors": [],
            }
        self.tool_group["calls"] += 1
        if intent not in self.tool_group["intents"]:
            self.tool_group["intents"].append(intent)

    def record_tool_result(self, event_data: ToolResultBlock) -> None:
        if self.state.verbosity != "normal":
            print_tool_result_rich(event_data, self.state.verbosity, self.live_console)
            return
        if self.tool_group is None:
            return
        if event_data.status == "ok":
            self.tool_group["ok"] += 1
            return
        label = self.tool_call_labels.get(
            event_data.tool_use_id, event_data.tool_use_id
        )
        self.tool_group["errors"].append((label, event_data))

    def handle_tool_update(self, event_data: ToolUpdateData) -> None:
        tool_id = event_data.tool_call_id
        partial = event_data.partial_result
        if not tool_id or not partial:
            return
        if event_data.tool_name == "delegate_task":
            self.flush_group()
            self.clear_progress()
            for line in partial.splitlines():
                clean = line.strip()
                if clean:
                    self.live_console.print(
                        Text(f"  Subagent: {clean}", style=CLI_COLOR_TOOL)
                    )
            return
        if self._progress_tool_id != tool_id:
            self._clear_progress()
            self._progress_tool_id = tool_id
        lines = [line for line in partial.splitlines() if line.strip()]
        last_line = lines[-1] if lines else ""
        if len(last_line) > 100:
            last_line = last_line[:97] + "..."
        if last_line:
            _safe_write(f"\r\033[K\x1b[90m  {last_line}\x1b[0m")

    def clear_progress(self) -> None:
        if self._progress_tool_id is not None:
            self.clear_line()
            self._progress_tool_id = None

    def flush_group(self) -> None:
        if self.tool_group is None:
            return
        self.clear_progress()
        calls = int(self.tool_group["calls"])
        errors = list(self.tool_group["errors"])
        intents = list(self.tool_group["intents"])
        title = summarize_intents(intents)
        status = "failed" if errors else "done"
        style = CLI_COLOR_ERROR if errors else CLI_COLOR_SUCCESS
        self.live_console.print(Text(f"  • Explore: {title}", style=CLI_COLOR_TOOL))
        self.live_console.print(Text(f"    {status}: {calls} tools", style=style))
        for label, result in errors:
            summary = single_line_preview(str(result.content), width=120)
            self.live_console.print(
                Text(f"    error: {label}: {summary}", style=CLI_COLOR_ERROR)
            )
        self.tool_group = None

    def discard_group(self) -> None:
        self.clear_progress()
        self.tool_group = None

    def clear_line(self) -> None:
        _safe_write("\r\033[K")

    def _clear_progress(self) -> None:
        if self._progress_tool_id is not None:
            self.clear_line()
            self._progress_tool_id = None


class ReasoningHandler:
    """处理推理过程的 delta 流式事件，管理实时预览和摘要输出。"""

    def __init__(
        self, live_console: Console, verbosity: VerbosityLevel = "normal"
    ) -> None:
        self.live_console = live_console
        self.verbosity = verbosity
        self.reasoning_started_at: float | None = None
        self.reasoning_text = ""
        self.reasoning_preview = LiveReasoningPreview(live_console)

    def handle_delta(self, event_data: str) -> None:
        if self.reasoning_started_at is None:
            self.reasoning_started_at = time.perf_counter()
        self.reasoning_text += event_data
        preview = reasoning_preview_lines(self.reasoning_text)
        display = ["  Thinking..."] + (preview or [])
        self.reasoning_preview.update(display)

    def finish(self) -> None:
        if self.reasoning_started_at is None:
            return
        elapsed = time.perf_counter() - self.reasoning_started_at
        self.reasoning_preview.stop()
        if not should_print_reasoning_summary(self.reasoning_text, elapsed):
            self.reasoning_started_at = None
            self.reasoning_text = ""
            return
        self.live_console.print(
            Text(
                f"  Thought for {format_elapsed(elapsed)}",
                style=CLI_COLOR_THINKING,
            )
        )
        self.reasoning_started_at = None
        self.reasoning_text = ""

    @property
    def text(self) -> str:
        return self.reasoning_text
