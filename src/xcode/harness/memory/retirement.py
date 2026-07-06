"""Retire durable memory without destroying its audit trail."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .manager import MemoryLayer, MemoryManager
from .parsing import MemoryTraceEvent


def retire_memory_record(
    manager: MemoryManager,
    memory_id: str,
    *,
    layer: MemoryLayer,
    reason: str,
) -> bool:
    """Remove one active record, archive it, and retain a retirement trace.

    The Markdown record is removed from active retrieval rather than merely
    down-ranked. Its archived copy carries the reason and UTC retirement time,
    while the governance proposal remains the authoritative decision ledger.
    """
    records = manager.read_memory_records(layer=layer)
    target = next((record for record in records if record.memory_id == memory_id), None)
    if target is None:
        return False

    retained_blocks = [
        record.block for record in records if record.memory_id != memory_id
    ]
    _archive_retired_record(manager, target.block, layer=layer, reason=reason)
    manager._write_blocks(retained_blocks, layer)

    lru = manager._read_lru()
    lru.pop(manager._lru_key(layer, target.memory_id), None)
    manager._write_lru(lru)
    manager._emit_trace(
        MemoryTraceEvent(
            type="forgotten",
            memory_id=target.memory_id,
            layer=layer,
            title=target.title,
            source="retirement",
        )
    )
    return True


def _archive_retired_record(
    manager: MemoryManager,
    block: str,
    *,
    layer: MemoryLayer,
    reason: str,
) -> None:
    archive_dir = manager._archive_dir(layer)
    archive_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    archived = (
        block.rstrip()
        + "\n"
        + f"- Retirement-Reason: {reason}\n"
        + f"- Retired-At: {timestamp.isoformat()}\n"
    )
    (archive_dir / f"retired_{stamp}.md").write_text(archived, encoding="utf-8")
