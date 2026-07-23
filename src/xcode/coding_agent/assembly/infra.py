"""共享基础设施构建。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xcode.harness.config import XcodeRuntimeConfig, resolve_config_path
from xcode.harness.agent_runtime import CancellationToken, ContextualRetrievalState
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.harness.memory import MemoryManager
from xcode.harness.session import SessionHistory


@dataclass(frozen=True)
class SharedInfra:
    contextual_state: ContextualRetrievalState
    cancellation_token: CancellationToken
    compact_controller: CompactController
    compactor: LayeredCompactor
    memory_manager: MemoryManager
    session_history: SessionHistory


def build_shared_infra(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
) -> SharedInfra:
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    memory_manager = MemoryManager(project_root)

    configured_sessions_dir = runtime_config.paths.sessions_dir
    if configured_sessions_dir:
        resolved_sessions_dir = resolve_config_path(
            project_root,
            configured_sessions_dir,
        )
        assert resolved_sessions_dir is not None
        transcript_dir = resolved_sessions_dir
        checkpoint_dir = transcript_dir / "checkpoints"
    else:
        transcript_dir = project_root / ".local" / "sessions"
        checkpoint_dir = project_root / ".xcode" / "checkpoints"

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        checkpoint_dir=checkpoint_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        keep_recent_tokens=runtime_config.agent.keep_recent_tokens,
    )
    session_history = SessionHistory(transcript_dir)
    return SharedInfra(
        contextual_state=contextual_state,
        cancellation_token=cancellation_token,
        compact_controller=compact_controller,
        compactor=compactor,
        memory_manager=memory_manager,
        session_history=session_history,
    )
