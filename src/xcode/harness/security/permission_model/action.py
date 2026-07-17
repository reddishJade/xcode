from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .types import Action, PermissionAccess, Target, UnresolvedEffect
from .utils import _looks_absolute

type PathExtractor = Callable[[Mapping[str, object]], tuple[str, ...]]


class ActionExtractor:
    def extract(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        action_profile: tuple[str, str] | None = None,
        path_extractor: PathExtractor | None = None,
    ) -> Action:
        action = self._extract_inner(tool_name, tool_input, path_extractor)
        if action_profile is not None:
            capability_name, _ = action_profile
            action = Action(
                tool=action.tool,
                capability=capability_name,
                operation=action.operation,
                targets=action.targets,
                input=action.input,
                unresolved_effects=action.unresolved_effects,
            )
        return action

    def _extract_inner(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        path_extractor: PathExtractor | None,
    ) -> Action:
        if tool_name == "read_file":
            return self._path_action(tool_name, tool_input, "read", "read_file", "read")
        if tool_name == "write_file":
            return self._path_action(
                tool_name, tool_input, "write", "write_file", "write"
            )
        if tool_name == "edit_file":
            return self._path_action(
                tool_name, tool_input, "edit", "edit_file", "write"
            )
        if tool_name == "apply_patch":
            return self._apply_patch_action(tool_name, tool_input, path_extractor)
        if tool_name == "bash":
            return self._bash_action(tool_name, tool_input)
        if tool_name == "shell":
            return self._shell_action(tool_name, tool_input)
        if tool_name == "delete_file":
            return self._path_action(
                tool_name, tool_input, "write", "delete_file", "write"
            )
        if tool_name in ("grep_search", "glob_files", "find_files", "list_dir"):
            return self._path_action(tool_name, tool_input, "read", tool_name, "read")
        if tool_name == "load_skill":
            return self._load_skill_action(tool_name, tool_input)
        if tool_name.startswith("mcp__"):
            return Action(
                tool=tool_name,
                capability="mcp",
                operation=tool_name,
                targets=(Target(kind="mcp", value=tool_name, access="execute"),),
                input=tool_input,
            )
        return Action(
            tool=tool_name,
            capability="unknown",
            operation=tool_name,
            targets=(),
            input=tool_input,
        )

    def _load_skill_action(
        self, tool_name: str, tool_input: Mapping[str, object]
    ) -> Action:
        name = tool_input.get("name")
        targets: tuple[Target, ...] = ()
        if isinstance(name, str) and name.strip():
            targets = (Target(kind="skill", value=name.strip(), access="read"),)
        return Action(
            tool=tool_name,
            capability="skill",
            operation="load_skill",
            targets=targets,
            input=tool_input,
        )

    def _path_action(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        capability: str,
        operation: str,
        access: PermissionAccess,
    ) -> Action:
        raw_path = tool_input.get("path")
        targets: tuple[Target, ...] = ()
        if isinstance(raw_path, str) and raw_path.strip():
            targets = (
                Target(
                    kind="path",
                    value=_normalize_path_text(raw_path),
                    access=access,
                ),
            )
        return Action(
            tool=tool_name,
            capability=capability,
            operation=operation,
            targets=targets,
            input=tool_input,
        )

    def _apply_patch_action(
        self,
        tool_name: str,
        tool_input: Mapping[str, object],
        path_extractor: PathExtractor | None,
    ) -> Action:
        targets = tuple(
            Target(kind="path", value=_normalize_path_text(path), access="write")
            for path in self._patch_paths(tool_input, path_extractor)
        )
        return Action(
            tool=tool_name,
            capability="patch",
            operation="apply_patch",
            targets=targets,
            input=tool_input,
        )

    def _bash_action(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        command = tool_input.get("command")
        targets: tuple[Target, ...] = ()
        unresolved_effects: tuple[UnresolvedEffect, ...] = ()
        if isinstance(command, str) and command.strip():
            normalized_command = command.strip()
            analysis = self._analyze_command(normalized_command, "posix")
            if analysis.ast_available:
                targets = (
                    Target(
                        kind="command",
                        value=normalized_command,
                        access="execute",
                    ),
                    *analysis.resolved_paths,
                )
                unresolved_effects = analysis.unresolved_effects
            else:
                targets = (
                    Target(
                        kind="command",
                        value=normalized_command,
                        access="execute",
                    ),
                    *self._shell_path_targets(normalized_command),
                )
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=targets,
            input=tool_input,
            unresolved_effects=unresolved_effects,
        )

    def _shell_action(self, tool_name: str, tool_input: Mapping[str, object]) -> Action:
        targets: list[Target] = []
        all_unresolved: list[UnresolvedEffect] = []
        for command in self._shell_commands(tool_input):
            normalized_command = command.strip()
            if not normalized_command:
                continue
            targets.append(
                Target(kind="command", value=normalized_command, access="execute")
            )
            analysis = self._analyze_command(normalized_command, "posix")
            targets.extend(analysis.resolved_paths)
            all_unresolved.extend(analysis.unresolved_effects)
            if not analysis.ast_available:
                targets.extend(self._shell_path_targets(normalized_command))
        return Action(
            tool=tool_name,
            capability="shell",
            operation="run_command",
            targets=tuple(targets),
            input=tool_input,
            unresolved_effects=tuple(all_unresolved),
        )

    def _analyze_command(
        self,
        command: str,
        shell_type: str = "posix",
    ) -> Any:
        try:
            from ..shell_analyzer import analyze_shell_command

            return analyze_shell_command(command, shell_type)
        except ImportError:
            return type(
                "_EmptyAnalysis",
                (),
                {
                    "resolved_paths": (),
                    "unresolved_effects": (),
                    "primary_command": None,
                    "shell_type": shell_type,
                    "parse_error": True,
                    "ast_available": False,
                },
            )()

    def _shell_path_targets(self, command: str) -> tuple[Target, ...]:
        try:
            tokens = shlex.split(command, posix=False)
        except ValueError:
            return ()
        if not tokens:
            return ()

        command_name = Path(tokens[0].strip("\"'")).name.lower()
        path_arguments = _filesystem_command_path_arguments(command_name, tokens[1:])
        access: PermissionAccess = (
            "read" if command_name in _READ_FILESYSTEM_COMMANDS else "write"
        )
        return tuple(
            Target(
                kind="path",
                value=_normalize_path_text(path),
                access=access,
                provenance="shell_literal",
            )
            for path in path_arguments
        )

    def _patch_paths(
        self,
        tool_input: Mapping[str, object],
        path_extractor: PathExtractor | None,
    ) -> tuple[str, ...]:
        if path_extractor is not None:
            extracted = path_extractor(tool_input)
            if extracted:
                return extracted

        raw_path = tool_input.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            return (raw_path,)

        raw_paths = tool_input.get("paths")
        if isinstance(raw_paths, tuple | list):
            return tuple(path for path in raw_paths if isinstance(path, str))

        return ()

    def _shell_commands(self, tool_input: Mapping[str, object]) -> tuple[str, ...]:
        raw_commands = tool_input.get("commands")
        if isinstance(raw_commands, tuple | list):
            return tuple(
                command for command in raw_commands if isinstance(command, str)
            )

        raw_command = tool_input.get("command")
        if isinstance(raw_command, str):
            return (raw_command,)

        return ()


def _normalize_path_text(raw_path: str) -> str:
    path = raw_path.strip()
    if _looks_absolute(path):
        return path.replace("\\", "/")
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        parts.append(part)
    return "/".join(parts) or "."


_READ_FILESYSTEM_COMMANDS = frozenset(
    {
        "cat",
        "dir",
        "get-childitem",
        "get-content",
        "head",
        "less",
        "ls",
        "more",
        "realpath",
        "tail",
    }
)
_WRITE_FILESYSTEM_COMMANDS = frozenset(
    {
        "copy-item",
        "cp",
        "del",
        "move-item",
        "mv",
        "remove-item",
        "rm",
        "set-content",
    }
)


def _filesystem_command_path_arguments(
    command_name: str,
    arguments: Sequence[str],
) -> tuple[str, ...]:
    if command_name not in _READ_FILESYSTEM_COMMANDS | _WRITE_FILESYSTEM_COMMANDS:
        return ()

    paths: list[str] = []
    for argument in arguments:
        cleaned = argument.strip("\"'")
        if not cleaned or cleaned.startswith("-"):
            continue
        if cleaned in {"&&", "||", ";", "|"}:
            break
        paths.append(cleaned)
    return tuple(paths)
