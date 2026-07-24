"""Shell 权限分类的行为测试。"""

from xcode.harness.security import (
    ActionExtractor,
    ShellAnalysisPolicyEvaluator,
    analyze_shell_command,
)


def test_read_only_pipeline_exposes_literal_paths_without_approval() -> None:
    analysis = analyze_shell_command("rg needle src tests | head -n 20")

    assert [target.value for target in analysis.resolved_paths] == ["src", "tests"]
    assert analysis.unresolved_effects == ()


def test_unknown_command_requires_approval_without_guessing_side_effects() -> None:
    action = ActionExtractor().extract(
        "bash",
        {"command": "pytest -q"},
        ("shell", "none"),
    )

    constraints = ShellAnalysisPolicyEvaluator().evaluate(action)

    assert action.targets[0].value == "pytest -q"
    assert [constraint.decision for constraint in constraints] == ["ask"]
    assert not any(target.kind == "path" for target in action.targets)


def test_redirection_requires_approval() -> None:
    analysis = analyze_shell_command("rg needle src > result.txt")

    assert [effect.reason for effect in analysis.unresolved_effects] == [
        "wrapper_command"
    ]
    assert analysis.resolved_paths == ()


def test_git_clean_is_denied_even_with_alternate_working_directory() -> None:
    action = ActionExtractor().extract(
        "bash",
        {"command": "git -C /tmp/repo clean -fdx"},
        ("shell", "none"),
    )

    constraints = ShellAnalysisPolicyEvaluator().evaluate(action)

    assert [constraint.decision for constraint in constraints] == ["deny"]
    assert "git clean" in constraints[0].reason


def test_recursive_root_delete_is_denied_but_scoped_delete_requires_approval() -> None:
    root_action = ActionExtractor().extract(
        "bash",
        {"command": "rm -rf /"},
        ("shell", "none"),
    )
    scoped_action = ActionExtractor().extract(
        "bash",
        {"command": "rm -rf build"},
        ("shell", "none"),
    )

    root_constraints = ShellAnalysisPolicyEvaluator().evaluate(root_action)
    scoped_constraints = ShellAnalysisPolicyEvaluator().evaluate(scoped_action)

    assert [constraint.decision for constraint in root_constraints] == ["deny"]
    assert [constraint.decision for constraint in scoped_constraints] == ["ask"]
