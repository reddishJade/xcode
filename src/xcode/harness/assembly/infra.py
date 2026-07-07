"""共享基础设施构建。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import XcodeRuntimeConfig, resolve_config_path
from ..agent_runtime import CancellationToken, ContextualRetrievalState
from ..agent_runtime.compaction import CompactController, LayeredCompactor
from ..memory import MemoryManager


@dataclass(frozen=True)
class SharedInfra:
    contextual_state: ContextualRetrievalState
    cancellation_token: CancellationToken
    compact_controller: CompactController
    compactor: LayeredCompactor
    memory_manager: MemoryManager


def build_shared_infra(
    project_root: Path,
    runtime_config: XcodeRuntimeConfig,
) -> SharedInfra:
    contextual_state = ContextualRetrievalState(project_root)
    cancellation_token = CancellationToken()
    compact_controller = CompactController()

    memory_manager = MemoryManager(project_root)

    def _combined_on_compact(summary: str) -> None:
        memory_manager.consolidate(summary)
        memory_manager.record_explicit_references(summary)
        memory_manager.record_llm_references(summary)
        memory_manager.record_compaction_referenced_feedback()

    transcript_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".local" / "sessions"
    )
    checkpoint_dir = (
        resolve_config_path(project_root, runtime_config.paths.sessions_dir)
        if runtime_config.paths.sessions_dir
        else project_root / ".xcode" / "checkpoints"
    )

    compactor = LayeredCompactor(
        transcript_dir=transcript_dir,
        checkpoint_dir=checkpoint_dir,
        max_recent_messages=runtime_config.agent.max_recent_messages,
        keep_recent_tokens=runtime_config.agent.keep_recent_tokens,
        reserve_tokens=runtime_config.agent.reserve_tokens,
        on_compact=_combined_on_compact,
    )
    return SharedInfra(
        contextual_state=contextual_state,
        cancellation_token=cancellation_token,
        compact_controller=compact_controller,
        compactor=compactor,
        memory_manager=memory_manager,
    )
