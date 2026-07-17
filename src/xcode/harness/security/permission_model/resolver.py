from __future__ import annotations

from .types import Constraint, Verdict


class PermissionResolver:
    DEFAULT_SOURCE = "resolver"
    DEFAULT_REASON = "no constraints produced; default allow"

    def resolve(self, constraints: tuple[Constraint, ...]) -> Verdict:
        if not constraints:
            return Verdict(
                decision="allow",
                source=self.DEFAULT_SOURCE,
                reason=self.DEFAULT_REASON,
                winning_constraint=None,
                constraints=constraints,
            )

        winner = self._winning_constraint(constraints)
        return Verdict(
            decision=winner.decision,
            source=winner.source,
            reason=winner.reason,
            winning_constraint=winner,
            constraints=constraints,
            metadata=winner.metadata,
        )

    def _winning_constraint(self, constraints: tuple[Constraint, ...]) -> Constraint:
        explicit_denies = tuple(c for c in constraints if c.decision == "deny")
        if explicit_denies:
            return explicit_denies[0]

        asks = tuple(c for c in constraints if c.decision == "ask")
        if asks:
            return asks[0]

        allows = tuple(c for c in constraints if c.decision == "allow")
        if allows:
            return allows[0]

        raise ValueError("constraint decision must be allow, ask, or deny")
