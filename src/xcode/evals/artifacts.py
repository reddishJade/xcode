"""可离线重建的 Trial artifact 持久化。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .schema import ArtifactManifest, Task, Trial, TrialResult


class ArtifactError(RuntimeError):
    """Artifact 无法完整或安全持久化。"""


@dataclass(frozen=True)
class TrialArtifactPaths:
    """单 Trial 所有文件的绝对写入路径。"""

    root: Path
    manifest: ArtifactManifest

    def resolve(self, relative: str) -> Path:
        """把 manifest 相对路径解析到本 Trial 根目录。"""
        return self.root / relative


class ArtifactStore:
    """建立固定目录，JSON 文件通过 replace 原子落盘。"""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def begin(
        self,
        *,
        task: Task,
        trial: Trial,
        environment: dict[str, Any],
    ) -> TrialArtifactPaths:
        """运行前保存声明和非秘密环境快照。"""
        _reject_secret_environment(environment)
        trial_root = self._root / trial.experiment_id / trial.trial_id
        if trial_root.exists():
            raise ArtifactError(f"artifact directory already exists: {trial_root}")
        trial_root.mkdir(parents=True)
        manifest = ArtifactManifest(
            task="task.json",
            trial="trial.json",
            trace="trace.jsonl",
            patch="changes.patch",
            stdout="agent.stdout",
            stderr="agent.stderr",
            verifier_log="verifier.log",
            result="result.json",
            environment="environment.json",
            checksums="checksums.json",
        )
        paths = TrialArtifactPaths(trial_root, manifest)
        _write_json(paths.resolve(manifest.task), task)
        _write_json(paths.resolve(manifest.trial), trial)
        _write_json(paths.resolve(manifest.environment), environment)
        paths.resolve(manifest.stdout).touch()
        paths.resolve(manifest.stderr).touch()
        return paths

    def finish(
        self,
        *,
        paths: TrialArtifactPaths,
        patch: str,
        result: TrialResult,
    ) -> None:
        """保存 patch 和最终结果，并验证 manifest 一致。"""
        if result.artifacts != paths.manifest:
            raise ArtifactError("result references a different artifact manifest")
        paths.resolve(paths.manifest.patch).write_text(patch, encoding="utf-8")
        _write_json(paths.resolve(paths.manifest.result), result)
        self.seal(paths.root, paths.manifest)

    def seal(
        self,
        trial_root: Path,
        manifest: ArtifactManifest | None = None,
    ) -> None:
        """为完整 Trial 写入非递归 SHA-256 清单，可用于旧 artifact 补封。"""
        if manifest is None:
            result = self.load_result(trial_root, verify=False)
            manifest = result.artifacts
        checksums = {
            relative: _sha256(trial_root / relative)
            for field, relative in manifest.model_dump().items()
            if field != "checksums"
        }
        _write_json(trial_root / manifest.checksums, checksums)

    def load_result(self, trial_root: Path, *, verify: bool = True) -> TrialResult:
        """只依赖 artifact 文件离线重建结果。"""
        result_path = trial_root / "result.json"
        try:
            result = TrialResult.model_validate_json(
                result_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise ArtifactError(
                f"cannot rebuild result from {result_path}: {error}"
            ) from error
        if verify:
            self.verify(trial_root, result.artifacts)
        return result

    def verify(self, trial_root: Path, manifest: ArtifactManifest) -> None:
        """验证 artifact 文件存在且与封存哈希完全一致。"""
        checksum_path = trial_root / manifest.checksums
        try:
            expected = json.loads(checksum_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ArtifactError(f"cannot read checksum manifest: {error}") from error
        actual = {
            relative: _sha256(trial_root / relative)
            for field, relative in manifest.model_dump().items()
            if field != "checksums"
        }
        if expected != actual:
            raise ArtifactError("artifact checksum verification failed")


def _write_json(path: Path, value: BaseModel | dict[str, Any]) -> None:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    """计算 artifact 内容哈希，缺失文件按持久化错误处理。"""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactError(f"cannot hash artifact {path}: {error}") from error


def _reject_secret_environment(environment: dict[str, Any]) -> None:
    forbidden = ("key", "secret", "token", "password", "credential")
    leaked = [
        key for key in environment if any(word in key.lower() for word in forbidden)
    ]
    if leaked:
        raise ArtifactError(
            f"environment snapshot contains secret-like keys: {sorted(leaked)}"
        )
