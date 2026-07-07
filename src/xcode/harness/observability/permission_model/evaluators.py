from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

from .protocols import PolicyEvaluator
from .types import (
    BLOCKED_PATH_PARTS,
    Action,
    BoundaryContext,
    Constraint,
    CREDENTIAL_PATH_PARTS,
    PermissionDecisionV2,
    StaticPermission,
    Target,
)
from .utils import (
    _access_satisfies,
    _is_blocked_workspace_path,
    _is_external_path,
    _is_git_path,
    _is_inside_path,
    _is_sensitive_path,
    _looks_absolute,
    _validate_symlinks_can_resolve,
)

logger = logging.getLogger(__name__)


class ModePolicyEvaluator:
    def __init__(self, decision: PermissionDecisionV2 | None) -> None:
        self._decision: PermissionDecisionV2 | None = decision

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        if self._decision is None:
            return ()
        return (
            Constraint(
                decision=self._decision,
                source="mode",
                reason=f"mode policy returned {self._decision} for {action.tool}",
                operation=action.operation,
            ),
        )


class StaticPolicyEvaluator:
    def __init__(
        self,
        rules: tuple[StaticPermission, ...] = (),
        *,
        global_default: PermissionDecisionV2 | None = None,
        action_input: str | None = None,
    ) -> None:
        self._rules = rules
        self._global_default = global_default
        self._action_input = action_input

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        action_input = self._serialized_action_input(action)
        decision = self._match_rules(action, action_input)
        if decision is not None:
            return self._constraints_for_action(
                action,
                decision,
                f"static permission rule returned {decision} for {action.tool}",
            )
        gd = self._global_default
        if gd is not None:
            return self._constraints_for_action(
                action,
                cast("PermissionDecisionV2", gd),
                f"no static rule matched; global_default={gd}",
            )
        return ()

    def _match_rules(
        self, action: Action, action_input: str
    ) -> PermissionDecisionV2 | None:
        last: PermissionDecisionV2 | None = None
        for rule in self._rules:
            if rule.tool != action.tool and rule.tool != "*":
                continue
            if (
                rule.input_contains is not None
                and rule.input_contains not in action_input
            ):
                continue
            if rule.input_prefix is not None and not action_input.startswith(
                rule.input_prefix
            ):
                continue
            if rule.input_regex is not None and not re.search(
                rule.input_regex, action_input
            ):
                continue
            if rule.target is not None:
                if not any(target.value == rule.target for target in action.targets):
                    continue
            if rule.target_type is not None:
                if not any(
                    target.kind == rule.target_type for target in action.targets
                ):
                    continue
            last = rule.decision
        return last

    def _serialized_action_input(self, action: Action) -> str:
        if self._action_input is not None:
            return self._action_input
        return json.dumps(action.input, ensure_ascii=False, sort_keys=True)

    def _constraints_for_action(
        self, action: Action, decision: PermissionDecisionV2, reason: str
    ) -> tuple[Constraint, ...]:
        if not action.targets:
            return (
                Constraint(
                    decision=decision,
                    source="rule",
                    reason=reason,
                    operation=action.operation,
                ),
            )

        return tuple(
            Constraint(
                decision=decision,
                source="rule",
                reason=reason,
                target_pattern=target.value,
                operation=action.operation,
                access=target.access,
            )
            for target in action.targets
        )


class PathBoundaryPolicyEvaluator:
    CREDENTIAL_PATH_PARTS = CREDENTIAL_PATH_PARTS
    BLOCKED_PATH_PARTS = BLOCKED_PATH_PARTS

    def __init__(self, context: BoundaryContext | None = None) -> None:
        self._context = context

    def evaluate(self, action: Action) -> tuple[Constraint, ...]:
        constraints: list[Constraint] = []
        for target in action.targets:
            if target.kind != "path":
                continue
            constraints.append(self._path_constraint(action, target))
        return tuple(constraints)

    def _path_constraint(self, action: Action, target: Target) -> Constraint:
        path_str = target.value

        if self._context is None:
            if _is_external_path(path_str):
                return Constraint(
                    decision="deny",
                    source="boundary",
                    reason=f"path escapes workspace boundary: {path_str}",
                    target_pattern=path_str,
                    operation=action.operation,
                    access=target.access,
                )
            return self._check_restrictions(path_str, path_str, action, target)

        try:
            resolved = self._resolve_workspace_path(target)
            return self._check_restrictions(resolved, path_str, action, target)
        except _BoundaryEscapeError:
            candidate = self._try_external_directory(target, action)
            if candidate is not None:
                return candidate
            logger.warning(
                "path resolved outside workspace boundary: %s",
                path_str,
            )
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"path outside all approved roots: {path_str}",
                target_pattern=path_str,
                operation=action.operation,
                access=target.access,
            )
        except _BoundaryResolutionError as exc:
            candidate = self._try_external_directory(target, action)
            if candidate is not None:
                return candidate
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"path cannot be resolved safely: {path_str}: {exc}",
                target_pattern=path_str,
                operation=action.operation,
                access=target.access,
            )

    def _try_external_directory(
        self, target: Target, action: Action
    ) -> Constraint | None:
        assert self._context is not None
        raw = target.value
        resolved_root = self._context.project_root.resolve(strict=False)

        try:
            if _looks_absolute(raw):
                candidate = Path(raw).resolve(strict=False)
            else:
                candidate = (resolved_root / raw).resolve(strict=False)
        except (OSError, RuntimeError):
            return None

        for ext in self._context.external_directories:
            if not _is_inside_path(candidate, ext.path):
                continue
            if not _access_satisfies(ext.access, target.access):
                continue
            check = candidate.as_posix()
            return self._check_restrictions(check, raw, action, target)
        return None

    def _check_restrictions(
        self,
        check_path: str,
        original_path: str,
        action: Action,
        target: Target,
    ) -> Constraint:
        if _is_git_path(check_path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"git metadata path is blocked: {original_path}",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        if _is_sensitive_path(check_path, access=target.access):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"sensitive path is blocked: {original_path}",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        if _is_blocked_workspace_path(check_path):
            return Constraint(
                decision="deny",
                source="boundary",
                reason=f"workspace blocked path is denied: {original_path}",
                target_pattern=check_path,
                operation=action.operation,
                access=target.access,
            )

        return Constraint(
            decision="allow",
            source="boundary",
            reason=f"path is allowed: {original_path}",
            target_pattern=check_path,
            operation=action.operation,
            access=target.access,
        )

    def _resolve_workspace_path(self, target: Target) -> str:
        assert self._context is not None
        root = self._context.project_root
        try:
            resolved_root = root.resolve(strict=False)
            source = resolved_root / target.value
            _validate_symlinks_can_resolve(resolved_root, target.value)
            candidate = source.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise _BoundaryResolutionError(str(exc)) from exc

        if not _is_inside_path(candidate, resolved_root):
            raise _BoundaryEscapeError(target.value)

        return candidate.relative_to(resolved_root).as_posix() or "."


StructuredBoundaryPolicyEvaluator = PathBoundaryPolicyEvaluator


class _BoundaryEscapeError(ValueError):
    pass


class _BoundaryResolutionError(ValueError):
    pass


def evaluate_policy_constraints(
    action: Action,
    *,
    execution_decision: PermissionDecisionV2 | None = None,
    static_policy: Any = None,
    action_input: str | None = None,
    boundary_context: BoundaryContext | None = None,
    hook_constraint_providers: tuple[PolicyEvaluator, ...] = (),
) -> tuple[Constraint, ...]:
    rules: tuple[StaticPermission, ...] = ()
    global_default: PermissionDecisionV2 | None = None
    if static_policy is not None:
        rules = static_policy.rules
        gd = static_policy.global_default
        if gd is not None:
            global_default = cast("PermissionDecisionV2", gd)

    evaluators: list[Any] = [
        ModePolicyEvaluator(execution_decision),
        StaticPolicyEvaluator(
            rules,
            global_default=global_default,
            action_input=action_input,
        ),
        PathBoundaryPolicyEvaluator(boundary_context),
    ]
    evaluators.extend(hook_constraint_providers)
    constraints: list[Constraint] = []
    for evaluator in evaluators:
        constraints.extend(evaluator.evaluate(action))
    return tuple(constraints)
