"""在 Agent 生命周期结束后运行控制面 verifier。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from pydantic import ValidationError

from .schema import Task, VerifierResult, VerifierSpec


class VerifierError(RuntimeError):
    """Verifier 基础设施失败，不能计为 Agent 未解决。"""


class VerifierRunner:
    """执行隐藏 verifier，并读取其显式四项结果协议。"""

    def run(
        self,
        *,
        spec: VerifierSpec,
        task: Task,
        workspace: Path,
        changed_paths: tuple[str, ...],
        log_path: Path,
        patch_path: Path | None = None,
    ) -> VerifierResult:
        """验证边界后运行命令；非零退出可判为任务失败而非 runner 失败。"""
        workspace = workspace.resolve()
        hidden_root = Path(spec.hidden_root).resolve()
        self._validate_boundary(workspace, hidden_root)
        if task.verifier_id != spec.verifier_id:
            raise VerifierError("task references a different verifier")
        result_path = hidden_root / spec.result_file
        if result_path.exists():
            result_path.unlink()

        patch_value = str(patch_path.resolve()) if patch_path is not None else "{patch}"
        command = tuple(
            token.replace("{workspace}", str(workspace))
            .replace("{hidden_root}", str(hidden_root))
            .replace("{patch}", patch_value)
            for token in spec.command
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP"}
        }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                command,
                cwd=hidden_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=spec.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise VerifierError(f"verifier execution failed: {error}") from error

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"exit_code={completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}",
            encoding="utf-8",
        )
        if not result_path.is_file():
            raise VerifierError("verifier did not produce its result protocol")
        try:
            payload: Any = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("verifier result must be an object")
            details = payload.get("details", {})
            if not isinstance(details, dict):
                raise TypeError("verifier details must be an object")
            violations = tuple(
                path
                for path in changed_paths
                if not _matches_any_path(path, task.allowed_paths)
                and not _matches_any_path(path, task.ignored_paths)
            )
            result = VerifierResult.model_validate(
                {
                    **payload,
                    "verifier_id": spec.verifier_id,
                    "verifier_version": spec.version,
                    "completed": True,
                    "policy_clean": not violations,
                    "log_artifact": log_path.name,
                    "details": {
                        **details,
                        "exit_code": completed.returncode,
                        "changed_paths": list(changed_paths),
                        "policy_violations": list(violations),
                    },
                }
            )
        except (OSError, ValueError, TypeError, ValidationError) as error:
            raise VerifierError(f"invalid verifier result protocol: {error}") from error
        return result

    @staticmethod
    def _validate_boundary(workspace: Path, hidden_root: Path) -> None:
        """隐藏材料和 Agent 工作区不能互相包含。"""
        if not hidden_root.is_dir():
            raise VerifierError(f"hidden verifier root does not exist: {hidden_root}")
        if hidden_root == workspace or hidden_root in workspace.parents:
            raise VerifierError("hidden verifier root contains the Agent workspace")
        if workspace in hidden_root.parents:
            raise VerifierError("hidden verifier root is inside the Agent workspace")


def _matches_any_path(path: str, patterns: tuple[str, ...]) -> bool:
    """Task 路径项匹配文件本身或目录内任意后代。"""
    if "." in patterns:
        return True
    return any(
        path == pattern or path.startswith(f"{pattern.rstrip('/')}/")
        for pattern in patterns
    )
