"""SafetyBackstopPolicy 对 shell 命令的 Bucket A/B/C 分类验证。

每条命令的预期分类来自 section 10.2 的三桶设计。
"""

from __future__ import annotations

from dataclasses import dataclass
from xcode.harness.observability import ActionExtractor
from xcode.harness.observability._safety_backstop import SafetyBackstopPolicyEvaluator
import pytest


@dataclass(frozen=True)
class ClassificationCase:
    name: str
    command: str
    exp_decision: str
    exp_non_bypassable: bool = False


CLASSIFICATION_CASES: tuple[ClassificationCase, ...] = (
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
    ClassificationCase("rm some_file.py", "rm some_file.py", "ask"),
    ClassificationCase("git reset --hard", "git reset --hard", "ask"),
    ClassificationCase("curl https://example.com", "curl https://example.com", "ask"),
    ClassificationCase("git push --force", "git push --force", "ask"),
    ClassificationCase("kill 1234", "kill 1234", "ask"),
    ClassificationCase("git status", "git status", "allow"),
    ClassificationCase("ls", "ls", "allow"),
    ClassificationCase("pytest", "pytest", "allow"),
    ClassificationCase("ruff check src/", "ruff check src/", "allow"),
    ClassificationCase("python --version", "python --version", "allow"),
    ClassificationCase("git diff", "git diff", "allow"),
    ClassificationCase("git status && rm -rf /", "git status && rm -rf /", "deny"),
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
    ClassificationCase("rm / -rf", "rm / -rf", "deny", True),
    ClassificationCase("rm /* -rf", "rm /* -rf", "deny", True),
    ClassificationCase("ls ~/.aws", "ls ~/.aws", "deny", True),
    ClassificationCase("echo ~/.gnupg", "echo ~/.gnupg", "deny", True),
    ClassificationCase(
        "git status \\\n  && rm -rf /", "git status \\\n  && rm -rf /", "deny"
    ),
    ClassificationCase("some_custom_tool --flag", "some_custom_tool --flag", "ask"),
)


class SafetyBackstopClassificationTests:
    """SafetyBackstopPolicy 的 Bucket A/B/C 分类正确性验证。"""

    maxDiff: int | None = None

    def _eval(self, command: str):
        evaluator = SafetyBackstopPolicyEvaluator()
        action = ActionExtractor().extract("bash", {"command": command})
        return evaluator.evaluate(action)

    @pytest.mark.parametrize("case", CLASSIFICATION_CASES, ids=lambda c: c.name)
    def test_classification(self, case: ClassificationCase) -> None:
        constraints = self._eval(case.command)

        if len(constraints) == 1:
            c = constraints[0]
            assert c.decision == case.exp_decision
        elif len(constraints) > 1:
            strictest = max(
                constraints,
                key=lambda c: {"allow": 0, "ask": 1, "deny": 2}.get(c.decision, 0),
            )
            assert strictest.decision == case.exp_decision
            assert strictest.source == "safety_backstop"

        if case.exp_non_bypassable:
            non_bp = [c for c in constraints if c.non_bypassable]
            assert len(non_bp) > 0


if __name__ == "__main__":
    pytest.main()
