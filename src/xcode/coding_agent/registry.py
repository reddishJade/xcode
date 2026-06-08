"""Coding product tool registry builder.

Extracted from harness/assembly.py to give the product layer explicit ownership
of tool composition decisions (which tools belong to which group, how they are
constructed, and what runtime values they receive).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from xcode.harness.execution_env import ExecutionEnv
from xcode.harness.skills import ToolSpec
from xcode.coding_agent.tools import (
    build_bash_tool,
    build_code_tools,
    build_file_tools,
    build_native_shell_tool,
    build_plan_mode_tools,
)

if TYPE_CHECKING:
    from xcode.harness.agent_runtime import ContextualRetrievalState
    from xcode.coding_agent.tools import ShellSpec


def build_project_scoped_registry(
    project_root: Path,
    enabled: set[str],
    contextual_state: ContextualRetrievalState | None,
    shell_spec: ShellSpec,
    cancel_event: threading.Event | None = None,
    env: ExecutionEnv | None = None,
    include_native_shell: bool = False,
    local_shell_skills: tuple[dict[str, str], ...] = (),
) -> tuple[ToolSpec, ...]:
    registry: tuple[ToolSpec, ...] = ()
    registry += build_file_tools(
        project_root, context_state=contextual_state, cancel_event=cancel_event
    )
    registry += build_code_tools(project_root, cancel_event=cancel_event)
    registry += (
        build_bash_tool(
            project_root,
            shell_spec=shell_spec,
            cancel_event=cancel_event,
            env=env,
        ),
    )
    if include_native_shell:
        registry += (
            build_native_shell_tool(
                project_root,
                shell_spec=shell_spec,
                cancel_event=cancel_event,
                env=env,
                skills=local_shell_skills,
            ),
        )
    registry += build_plan_mode_tools(project_root)
    return tuple(t for t in registry if t.group in enabled)
