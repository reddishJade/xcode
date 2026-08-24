"""Turn-渲染专用的处理器：ToolCallHandler 和 ReasoningHandler。"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.text import Text

from xcode.ai.events import ToolCall
from xcode.harness.agent_runtime.events import ToolResultBlock, ToolUpdateData

from .commands import ReplState
from .repl_rendering import (
    CLI_COLOR_DIM,
    CLI_COLOR_ERROR,
    CLI_COLOR_SUCCESS,
    CLI_COLOR_THINKING,
    CLI_COLOR_TOOL,
    LiveReasoningPreview,
)
from .repl_tools import (
    brief_input,
    print_tool_call_rich,
    print_tool_result_rich,
    summarize_intents,
    tool_call_text,
    tool_intent,
)
from .shared.thinking import (
    ReasoningCore,
    format_elapsed,
    should_print_reasoning_summary,
    single_line_preview,
)


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
    """追踪工具调用、结果和更新，管理批次聚合渲染和进度显示。"""

    def __init__(self, state: ReplState, live_console: Console) -> None:
        self.state = state
        self.live_console = live_console
        self.tool_group: dict[str, Any] | None = None
        self.tool_call_labels: dict[str, str] = {}
        self._progress_tool_id: str | None = None
        self._subagent_slots: dict[int, dict[str, str]] = {}
        self._subagent_live: Live | None = None

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
                "details": [],
            }
        self.tool_group["calls"] += 1
        # ponytail: 记录工具名称+参数摘要，方便 flush 时显示
        self.tool_group["details"].append(
            tool_call_text(event_data.name, label, event_data.input)
        )
        if intent not in self.tool_group["intents"]:
            self.tool_group["intents"].append(intent)

    def record_tool_result(self, event_data: ToolResultBlock) -> None:
        self.clear_progress()
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
        if self.state.tool_collapsed:
            return
        if event_data.tool_name == "subagent":
            self.flush_group()
            self._clear_progress()
            changed = False
            for line in partial.splitlines():
                changed = self._record_subagent_update(line.strip()) or changed
            if changed:
                self._render_subagent_live()
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
        self.clear_line()
        self._progress_tool_id = None

    def flush_group(self) -> None:
        if self.tool_group is None:
            return
        self.clear_progress()
        calls = int(self.tool_group["calls"])
        errors = list(self.tool_group["errors"])
        intents = list(self.tool_group["intents"])
        details = list(self.tool_group.get("details", []))
        title = summarize_intents(intents)
        if self.state.tool_collapsed:
            self.live_console.print(
                Text(f"  • {calls} tools — {title}", style=CLI_COLOR_TOOL)
            )
        else:
            # ponytail: 主 agent 工具每次调用一行，清晰可见
            for detail in details:
                self.live_console.print(detail)
            status = "failed" if errors else "done"
            style = CLI_COLOR_ERROR if errors else CLI_COLOR_SUCCESS
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
        self._stop_subagent_live()
        self._subagent_slots.clear()

    def _record_subagent_update(self, clean: str) -> bool:
        if not clean:
            return False
        match = re.match(r"\[(\d+)]( +)(.*)", clean)
        if match is None:
            return False
        index = int(match.group(1))
        gap = match.group(2)
        body = match.group(3)
        slot = self._subagent_slots.setdefault(index, {"task": "", "tool": ""})
        if len(gap) > 1:
            slot["tool"] = body.strip()
        else:
            slot["task"] = body
            if body.startswith(("✓", "✗")):
                slot["tool"] = ""
        return True

    def _render_subagent_live(self) -> None:
        text = Text("  Subagents", style=CLI_COLOR_TOOL)
        for index in sorted(self._subagent_slots):
            slot = self._subagent_slots[index]
            task = slot.get("task") or "waiting"
            tool = slot.get("tool", "")
            text.append("\n")
            text.append(f"    [{index}] ", style=CLI_COLOR_DIM)
            text.append(task, style=CLI_COLOR_TOOL)
            if tool:
                text.append("\n")
                text.append("        ", style=CLI_COLOR_DIM)
                text.append(tool, style=CLI_COLOR_DIM)
        if self._subagent_live is None:
            self._subagent_live = Live(
                text,
                console=self.live_console,
                refresh_per_second=12,
                transient=False,
            )
            self._subagent_live.start(refresh=True)
            return
        self._subagent_live.update(text, refresh=True)

    def _stop_subagent_live(self) -> None:
        if self._subagent_live is None:
            return
        self._subagent_live.stop()
        self._subagent_live = None

    def clear_line(self) -> None:
        _safe_write("\r\033[K")

    def _clear_progress(self) -> None:
        if self._progress_tool_id is not None:
            self.clear_line()
            self._progress_tool_id = None


class ReasoningHandler:
    """处理推理过程的 delta 流式事件，管理实时预览和摘要输出。

    基于输出无关的 ReasoningCore，叠加 Rich Live 实时显示。
    """

    def __init__(self, live_console: Console, state: ReplState) -> None:
        self.live_console = live_console
        self.state = state
        self.core = ReasoningCore()
        self.reasoning_preview = LiveReasoningPreview(live_console)

    def handle_delta(self, event_data: str) -> None:
        self.core.handle_delta(event_data)
        if not self.state.thinking_collapsed:
            display = ["  Thinking..."] + (self.core.preview_lines or [])
            self.reasoning_preview.update(display)

    def finish(self) -> None:
        if self.core.started_at is None:
            return
        self.core.finish()
        self.reasoning_preview.stop()
        if self.state.thinking_collapsed:
            self.core.reset()
            return
        elapsed = time.perf_counter() - self.core.started_at
        if not should_print_reasoning_summary(self.core.text, elapsed):
            self.core.reset()
            return
        self.live_console.print(
            Text(
                f"  Thought for {format_elapsed(elapsed)}",
                style=CLI_COLOR_THINKING,
            )
        )
        self.core.reset()

    @property
    def text(self) -> str:
        return self.core.text
