"""安全策略与 ruleset 辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from ..config import ModeRuleRuntimeConfig, SecurityRuntimeConfig, XcodeRuntimeConfig
from ..observability import PermissionDecision, PermissionPolicy, StaticPermission
from ..observability.permission_model import ExternalDirectory, Rule


def _rule_from_runtime_config(rule: ModeRuleRuntimeConfig) -> Rule:
    return Rule(
        action=rule.action,
        effect=rule.effect,
        command=rule.command,
        subcommand=rule.subcommand,
        subcommand_in=set(rule.subcommand_in)
        if rule.subcommand_in is not None
        else None,
        flags_any=set(rule.flags_any) if rule.flags_any is not None else None,
        flags_all=set(rule.flags_all) if rule.flags_all is not None else None,
        resource_pattern=rule.resource_pattern,
    )


def mode_rulesets_from_runtime_config(
    runtime_config: XcodeRuntimeConfig,
) -> dict[str, tuple[Rule, ...]]:
    modes = runtime_config.execution_modes
    result: dict[str, tuple[Rule, ...]] = {}
    for mode_name, ruleset in (
        ("plan", modes.plan),
        ("build", modes.build),
        ("act", modes.act),
    ):
        if ruleset.rules:
            result[mode_name] = tuple(
                _rule_from_runtime_config(rule) for rule in ruleset.rules
            )
    return result


def external_directories_from_security(
    security: SecurityRuntimeConfig,
) -> tuple[ExternalDirectory, ...]:
    dirs: list[ExternalDirectory] = [
        ExternalDirectory(path=Path(ed.path), access=ed.access)
        for ed in security.external_directories
    ]
    home = Path.home()
    for p in (home / ".xcode", home / ".agents"):
        if p.is_dir():
            ext = ExternalDirectory(path=p, access="read")
            if ext not in dirs:
                dirs.append(ext)
    return tuple(dirs)


def permission_policy_from_security(
    security: SecurityRuntimeConfig,
) -> PermissionPolicy | None:
    rules: list[StaticPermission] = []
    for rd in security.rules:
        rules.append(
            StaticPermission(
                tool=rd["tool"],
                decision=rd["decision"],
                target=rd.get("target"),
                target_type=rd.get("target_type"),
                input_contains=rd.get("input_contains"),
                input_prefix=rd.get("input_prefix"),
                input_regex=rd.get("input_regex"),
            )
        )
    global_default: str | None = security.global_default
    if global_default is None and security.resolve_approval_policy() == "always":
        global_default = "ask"
    if not rules and global_default is None:
        return None
    return PermissionPolicy(
        tuple(rules), global_default=cast(PermissionDecision, global_default)
    )
