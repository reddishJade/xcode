from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class VerifierResult:
    passed: bool
    name: str
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class Verifier(Protocol):
    name: str
    description: str

    async def verify(self, context: dict[str, Any]) -> VerifierResult: ...


@dataclass
class CommandVerifier:
    """运行 shell 命令，用退出码判断是否通过。"""

    name: str
    description: str
    command: str
    expected_exit_code: int = 0
    cwd: str | None = None

    async def verify(self, context: dict[str, Any]) -> VerifierResult:
        cwd = self.cwd or context.get("cwd")
        try:
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            passed = result.returncode == self.expected_exit_code
            return VerifierResult(
                passed=passed,
                name=self.name,
                message=(
                    f"exit code {result.returncode} (expected {self.expected_exit_code})"
                ),
                details={
                    "stdout": result.stdout[:500],
                    "stderr": result.stderr[:500],
                    "returncode": result.returncode,
                },
            )
        except subprocess.TimeoutExpired:
            return VerifierResult(
                passed=False,
                name=self.name,
                message="timed out after 30s",
            )
        except Exception as e:
            return VerifierResult(
                passed=False,
                name=self.name,
                message=str(e),
            )


@dataclass
class FileContentVerifier:
    name: str
    description: str
    path: str
    contains: str | None = None
    not_contains: str | None = None

    async def verify(self, context: dict[str, Any]) -> VerifierResult:
        root = Path(context.get("cwd", "."))
        target = root / self.path
        if not target.exists():
            return VerifierResult(
                passed=False,
                name=self.name,
                message=f"file not found: {self.path}",
            )
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
            if self.contains and self.contains not in text:
                return VerifierResult(
                    passed=False,
                    name=self.name,
                    message=f"expected content not found in {self.path}",
                )
            if self.not_contains and self.not_contains in text:
                return VerifierResult(
                    passed=False,
                    name=self.name,
                    message=f"unexpected content found in {self.path}",
                )
            return VerifierResult(passed=True, name=self.name, message="content check ok")
        except Exception as e:
            return VerifierResult(
                passed=False, name=self.name, message=str(e)
            )


class VerifierRegistry:
    def __init__(self) -> None:
        self._verifiers: dict[str, Verifier] = {}

    def register(self, verifier: Verifier) -> None:
        self._verifiers[verifier.name] = verifier

    def get(self, name: str) -> Verifier | None:
        return self._verifiers.get(name)

    def list(self) -> list[Verifier]:
        return list(self._verifiers.values())

    async def run_all(
        self, context: dict[str, Any] | None = None
    ) -> list[VerifierResult]:
        ctx = context or {}
        results: list[VerifierResult] = []
        for name, verifier in self._verifiers.items():
            try:
                result = await verifier.verify(ctx)
                results.append(result)
            except Exception as e:
                results.append(
                    VerifierResult(
                        passed=False, name=name, message=f"verifier error: {e}"
                    )
                )
        return results

    async def run_named(
        self, names: list[str], context: dict[str, Any] | None = None
    ) -> list[VerifierResult]:
        ctx = context or {}
        results: list[VerifierResult] = []
        for name in names:
            verifier = self._verifiers.get(name)
            if verifier is None:
                results.append(
                    VerifierResult(
                        passed=False, name=name, message=f"unknown verifier: {name}"
                    )
                )
                continue
            try:
                result = await verifier.verify(ctx)
                results.append(result)
            except Exception as e:
                results.append(
                    VerifierResult(
                        passed=False, name=name, message=f"verifier error: {e}"
                    )
                )
        return results

    def to_json(self) -> str:
        verifiers = [
            {"name": v.name, "description": v.description}
            for v in self._verifiers.values()
        ]
        return json.dumps(verifiers, ensure_ascii=False, indent=2)
