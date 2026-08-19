from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import ValidationError

from .action import _normalize_path_text
from .types import (
    Action,
    BoundaryContext,
    GrantRecord,
    GrantRecordData,
    Target,
)
from .utils import _is_external_path, _is_inside_path


def _lookup_grant_record(
    records: tuple[GrantRecord, ...],
    action: Action,
    target: Target,
    *,
    boundary_context: BoundaryContext | None = None,
) -> GrantRecord | None:
    matching = tuple(
        record
        for record in records
        if _grant_matches_target(
            record,
            action,
            target,
            boundary_context=boundary_context,
        )
    )
    if not matching:
        return None
    return _highest_priority_grant(matching)


def _highest_priority_grant(records: Sequence[GrantRecord]) -> GrantRecord:
    for record in records:
        if record.decision == "deny":
            return record
    return records[0]


def _grant_matches_target(
    record: GrantRecord,
    action: Action,
    target: Target,
    *,
    boundary_context: BoundaryContext | None = None,
) -> bool:
    if record.capability != action.capability:
        return False
    if record.operation != action.operation:
        return False
    if record.target_kind != target.kind:
        return False
    if record.access != target.access:
        return False
    if target.kind != "path":
        from ..rule_matcher import _wildcard_match

        return _wildcard_match(target.value, record.target_pattern, cross_path=True)
    return _path_pattern_matches(
        record.target_pattern,
        target.value,
        boundary_context=boundary_context,
    )


def _path_pattern_matches(
    target_pattern: str,
    candidate: str,
    *,
    boundary_context: BoundaryContext | None = None,
) -> bool:
    pattern = _normalize_target_path(target_pattern, boundary_context=boundary_context)
    normalized_candidate = _normalize_target_path(
        candidate,
        boundary_context=boundary_context,
    )
    if pattern == normalized_candidate:
        return True
    return normalized_candidate.startswith(f"{pattern}/")


def _normalize_target_path(
    path: str,
    *,
    boundary_context: BoundaryContext | None = None,
) -> str:
    normalized = _normalize_path_text(path)
    if boundary_context is None or _is_external_path(normalized):
        return normalized

    root = boundary_context.project_root
    try:
        resolved_root = root.resolve(strict=False)
        candidate = (resolved_root / normalized).resolve(strict=False)
    except (OSError, RuntimeError):
        return normalized
    if not _is_inside_path(candidate, resolved_root):
        return normalized
    return candidate.relative_to(resolved_root).as_posix() or "."


def _grant_record_from_data(value: object) -> GrantRecord | None:
    if not isinstance(value, dict):
        return None
    try:
        return GrantRecord.model_validate(value)
    except ValidationError:
        return None


def _grant_record_to_data(record: GrantRecord) -> GrantRecordData:
    data: GrantRecordData = record.model_dump(exclude_none=True)
    if not record.metadata:
        data.pop("metadata", None)
    else:
        data["metadata"] = dict(record.metadata)
    return data


class InMemoryGrantStore:
    def __init__(
        self,
        records: Iterable[GrantRecord] = (),
        *,
        session_id: str = "",
    ) -> None:
        self._session_id = session_id
        self._records = tuple(records)

    def add(self, record: GrantRecord) -> GrantRecord:
        self._records = tuple(
            existing
            for existing in self._records
            if existing.grant_id != record.grant_id
        ) + (record,)
        return record

    def records(self) -> tuple[GrantRecord, ...]:
        return self._records

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None:
        return _lookup_grant_record(
            self._records,
            action,
            target,
            boundary_context=boundary_context,
        )

    def clear(self) -> None:
        self._records = ()


class SessionGrantStoreManager:
    def __init__(self) -> None:
        self._stores: dict[str, InMemoryGrantStore] = {}

    def get_for_session(self, session_id: str) -> InMemoryGrantStore:
        if session_id not in self._stores:
            self._stores[session_id] = InMemoryGrantStore(session_id=session_id)
        return self._stores[session_id]


class FileGrantStore:
    DEFAULT_RELATIVE_PATH = Path(".xcode") / "approval_grants.json"

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_project_root(cls, project_root: Path) -> FileGrantStore:
        return cls(project_root / cls.DEFAULT_RELATIVE_PATH)

    def add(self, record: GrantRecord) -> GrantRecord:
        updated = tuple(
            existing
            for existing in self.records()
            if existing.grant_id != record.grant_id
        ) + (record,)
        self._write(updated)
        return record

    def records(self) -> tuple[GrantRecord, ...]:
        if not self.path.exists():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ()
        if not isinstance(raw, list):
            return ()
        return tuple(
            record
            for item in raw
            if (record := _grant_record_from_data(item)) is not None
        )

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None:
        return _lookup_grant_record(
            self.records(),
            action,
            target,
            boundary_context=boundary_context,
        )

    def _write(self, records: tuple[GrantRecord, ...]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [_grant_record_to_data(record) for record in records]
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
