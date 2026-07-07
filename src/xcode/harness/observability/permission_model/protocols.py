from __future__ import annotations

from typing import Protocol

from .types import Action, BoundaryContext, Constraint, GrantRecord, Target


class GrantStore(Protocol):
    def add(self, record: GrantRecord) -> GrantRecord: ...

    def records(self) -> tuple[GrantRecord, ...]: ...

    def lookup(
        self,
        action: Action,
        target: Target,
        *,
        boundary_context: BoundaryContext | None = None,
    ) -> GrantRecord | None: ...


class PolicyEvaluator(Protocol):
    def evaluate(self, action: Action) -> tuple[Constraint, ...]: ...
