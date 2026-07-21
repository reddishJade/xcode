"""交互审批选项测试。"""

from xcode.cli.repl_hitl import hitl_choices


def test_hitl_choices_only_show_allowed_scopes() -> None:
    assert hitl_choices(("once",)) == ("Allow (once)", "Deny")
    assert hitl_choices(("once", "session")) == (
        "Allow (once)",
        "Allow this session",
        "Deny",
    )
