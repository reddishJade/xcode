"""SafetyBackstopPolicy 对 shell 命令的 Bucket A/B/C 分类验证。

每条命令的预期分类来自 section 10.2 的三桶设计。
"""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from xcode.harness.observability import ActionExtractor
from xcode.harness.observability.permission_model import (
    SafetyBackstopPolicyEvaluator,
)


@dataclass(frozen=True)
class ClassificationCase:
    name: str
    command: str
    exp_decision: str
    exp_non_bypassable: bool = False


class SafetyBackstopClassificationTests(unittest.TestCase):
    """SafetyBackstopPolicy 的 Bucket A/B/C 分类正确性验证。"""

    maxDiff: int | None = None

    def _eval(self, command: str):
        evaluator = SafetyBackstopPolicyEvaluator()
        action = ActionExtractor().extract("bash", {"command": command})
        return evaluator.evaluate(action)

    def _collect_cases(self) -> tuple[ClassificationCase, ...]:
        return (
            # Bucket A: non-bypassable deny
            ClassificationCase("rm -rf /", "rm -rf /", "deny", True),
            ClassificationCase("cat ~/.ssh/config", "cat ~/.ssh/config", "deny", True),
            ClassificationCase("chmod -R 777 /", "chmod -R 777 /", "deny", True),
            ClassificationCase(
                "dd if=/dev/zero of=/dev/sda",
                "dd if=/dev/zero of=/dev/sda",
                "deny",
                True,
            ),
            ClassificationCase("git credential", "git credential", "deny", True),
            # Bucket B: ask
            ClassificationCase("rm some_file.py", "rm some_file.py", "ask"),
            ClassificationCase("git reset --hard", "git reset --hard", "ask"),
            ClassificationCase(
                "curl https://example.com", "curl https://example.com", "ask"
            ),
            ClassificationCase("git push --force", "git push --force", "ask"),
            ClassificationCase("kill 1234", "kill 1234", "ask"),
            # Bucket C: allow
            ClassificationCase("git status", "git status", "allow"),
            ClassificationCase("ls", "ls", "allow"),
            ClassificationCase("pytest", "pytest", "allow"),
            ClassificationCase("ruff check src/", "ruff check src/", "allow"),
            ClassificationCase("python --version", "python --version", "allow"),
            ClassificationCase("git diff", "git diff", "allow"),
            # Compound commands
            ClassificationCase(
                "git status && rm -rf /", "git status && rm -rf /", "deny"
            ),
            ClassificationCase("ls | grep foo", "ls | grep foo", "allow"),
            ClassificationCase(
                "python --version && pip install django",
                "python --version && pip install django",
                "ask",
            ),
            ClassificationCase(
                "cat ~/.ssh/config && echo done",
                "cat ~/.ssh/config && echo done",
                "deny",
            ),
            # Flag after path
            ClassificationCase("rm / -rf", "rm / -rf", "deny", True),
            ClassificationCase("rm /* -rf", "rm /* -rf", "deny", True),
            # Bare credential dir (no trailing slash)
            ClassificationCase("ls ~/.aws", "ls ~/.aws", "deny", True),
            ClassificationCase("echo ~/.gnupg", "echo ~/.gnupg", "deny", True),
            # Backslash-newline continuation
            ClassificationCase(
                "git status \\\n  && rm -rf /", "git status \\\n  && rm -rf /", "deny"
            ),
            # Unknown/unclassified → default ask
            ClassificationCase(
                "some_custom_tool --flag", "some_custom_tool --flag", "ask"
            ),
        )

    def test_classification(self) -> None:
        for case in self._collect_cases():
            with self.subTest(case.name):
                constraints = self._eval(case.command)

                # SafetyBackstop always produces exactly one constraint
                # (compound commands are split, each segment produces one)
                # The "strictest" constraint wins for compound commands
                if len(constraints) == 1:
                    c = constraints[0]
                    self.assertEqual(c.decision, case.exp_decision, case.name)
                elif len(constraints) > 1:
                    # Compound: take strictest
                    strictest = max(
                        constraints,
                        key=lambda c: {"allow": 0, "ask": 1, "deny": 2}.get(
                            c.decision, 0
                        ),
                    )
                    self.assertEqual(strictest.decision, case.exp_decision, case.name)
                    self.assertEqual(strictest.source, "safety_backstop", case.name)

                if case.exp_non_bypassable:
                    non_bp = [c for c in constraints if c.non_bypassable]
                    self.assertTrue(
                        len(non_bp) > 0, f"expected non-bypassable for {case.name}"
                    )


if __name__ == "__main__":
    unittest.main()
