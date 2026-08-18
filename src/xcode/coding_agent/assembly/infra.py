"""共享基础设施构建。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xcode.harness.config import XcodeRuntimeConfig, resolve_config_path
from xcode.harness.agent_runtime import CancellationToken, ContextualRetrievalState
from xcode.harness.agent_runtime.compaction import CompactController, LayeredCompactor
from xcode.harness.memory import MemoryManager
from xcode.harness.session import SessionHistory, SessionInbox, SessionStore
from xcode.harness.session.recorder import SessionRecorder


@dataclass(frozen=True)
class SharedInfra:
    contextual_state: ContextualRetrievalState
    cancellation_token: CancellationToken
    compact_controller: CompactController
    compactor: LayeredCompactor
    memory_manager: MemoryManager
    session_history: SessionHistory
    session_inbox: SessionInbox
    session_recorder: SessionRecorder


def build_shared_infra(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
    sessions_dir: Path | None = None,
) -> SharedInfra:
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    memory_manager = MemoryManager(project_root)

    configured_sessions_dir = runtime_config.paths.sessions_dir
    if sessions_dir is not None:
        transcript_dir = sessions_dir.resolve()
    elif configured_sessions_dir:
        resolved_sessions_dir = resolve_config_path(
            project_root,
            configured_sessions_dir,
        )
        assert resolved_sessions_dir is not None
        transcript_dir = resolved_sessions_dir
    else:
        transcript_dir = project_root / ".local" / "sessions"

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        keep_recent_tokens=runtime_config.agent.keep_recent_tokens,
    )
    session_history = SessionHistory(transcript_dir)
    session_recorder = SessionRecorder(
        SessionStore(transcript_dir, project_root=project_root)
    )
    session_inbox = SessionInbox(session_recorder.store)
    return SharedInfra(
        contextual_state=contextual_state,
        cancellation_token=cancellation_token,
        compact_controller=compact_controller,
        compactor=compactor,
        memory_manager=memory_manager,
        session_history=session_history,
        session_inbox=session_inbox,
        session_recorder=session_recorder,
    )
