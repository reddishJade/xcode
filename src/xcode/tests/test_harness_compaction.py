"""分层上下文压缩纯函数单元测试。"""

from __future__ import annotations

from xcode.harness.agent_runtime.compaction import (
    context_collapse_clean,
    _compute_recent_count_from_tokens,
    _find_turn_boundary,
    _is_tool_result_message,
    _inactive_branch_id,
    _branch_summary_should_run,
    _protect_identifiers,
    _content_preview,
    _extract_file_ops_from_messages,
    _render_file_tracking,
)


class TestContextCollapseClean:
    def test_extracts_summary_tag(self) -> None:
        content = "<summary>This is the summary</summary>"
        result = context_collapse_clean(content)
        assert "This is the summary" in result
        assert "<summary>" not in result

    def test_removes_analysis_think_tags(self) -> None:
        content = (
            "prefix <analysis>hidden</analysis> middle <think>hidden</think> suffix"
        )
        result = context_collapse_clean(content)
        assert "hidden" not in result
        assert "prefix" in result
        assert "suffix" in result

    def test_empty_input(self) -> None:
        assert context_collapse_clean("") == ""
        assert context_collapse_clean("  ") == ""


class TestComputeRecentCountFromTokens:
    def test_at_least_one(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        result = _compute_recent_count_from_tokens(msgs, 100)
        assert result >= 1

    def test_empty_messages(self) -> None:
        assert _compute_recent_count_from_tokens([], 100) >= 1


def _msg(role: str, content: str | list | None = None) -> dict:
    return {"role": role, "content": content or "text"}


class TestFindTurnBoundary:
    def test_user_is_safe(self) -> None:
        msgs = [_msg("system"), _msg("user"), _msg("assistant")]
        assert _find_turn_boundary(msgs, 1) == 1

    def test_tool_adjusts_backward(self) -> None:
        msgs = [_msg("system"), _msg("user"), _msg("assistant"), _msg("tool")]
        assert _find_turn_boundary(msgs, 3) == 2  # assistant

    def test_clamps_to_min_one(self) -> None:
        assert _find_turn_boundary([_msg("system")], 0) == 1

    def test_clamps_to_max(self) -> None:
        msgs = [_msg("system"), _msg("user")]
        assert _find_turn_boundary(msgs, 5) == 1


class TestIsToolResultMessage:
    def test_tool_role(self) -> None:
        assert _is_tool_result_message({"role": "tool"})

    def test_tool_result_in_content(self) -> None:
        msg = {"role": "user", "content": [{"type": "tool_result"}]}
        assert _is_tool_result_message(msg)

    def test_plain_message(self) -> None:
        assert not _is_tool_result_message({"role": "user"})


class TestInactiveBranchId:
    def test_active_branch_returns_none(self) -> None:
        msg = {
            "metadata": {
                "type": "conversation",
                "branch_id": "b1",
                "active_branch": True,
            }
        }
        assert _inactive_branch_id(msg, "b1") is None

    def test_branch_summary_returns_none(self) -> None:
        msg = {"metadata": {"type": "branch_summary", "branch_id": "b1"}}
        assert _inactive_branch_id(msg, "b1") is None

    def test_inactive_returns_id(self) -> None:
        msg = {"metadata": {"branch_id": "b1"}}
        assert _inactive_branch_id(msg, "active_b") == "b1"


class TestBranchSummaryShouldRun:
    def test_zero_threshold_false(self) -> None:
        assert not _branch_summary_should_run([_msg("user")], 0, 0.5)

    def test_under_threshold_false(self) -> None:
        assert not _branch_summary_should_run([_msg("user", "hi")], 999999, 0.5)


class TestProtectIdentifiers:
    def test_no_frozen(self) -> None:
        assert _protect_identifiers("hello", []) == "hello"

    def test_replaces_with_backtick(self) -> None:
        result = _protect_identifiers("use MyVar in the code", ["MyVar"])
        assert "`MyVar`" in result


class TestContentPreview:
    def test_string(self) -> None:
        preview = _content_preview("hello world " * 50)
        assert len(preview) <= 185  # ~180 + "..."

    def test_list(self) -> None:
        preview = _content_preview([{"type": "text"}, {"type": "tool_result"}])
        assert "text" in preview

    def test_none(self) -> None:
        assert _content_preview(None) == "None"


class TestExtractFileOpsFromMessages:
    def test_read_files(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "read_file",
                        "input": {"path": "src/main.py"},
                    },
                ],
            }
        ]
        reads, modifies = _extract_file_ops_from_messages(msgs)
        assert "src/main.py" in reads
        assert not modifies

    def test_modified_files(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "write_file",
                        "input": {"path": "src/new.py"},
                    },
                ],
            }
        ]
        reads, modifies = _extract_file_ops_from_messages(msgs)
        assert "src/new.py" in modifies

    def test_non_assistant_skipped(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        reads, modifies = _extract_file_ops_from_messages(msgs)
        assert not reads
        assert not modifies

    def test_no_tool_use(self) -> None:
        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]
        reads, modifies = _extract_file_ops_from_messages(msgs)
        assert not reads
        assert not modifies


class TestRenderFileTracking:
    def test_read_only(self) -> None:
        result = _render_file_tracking({"src/a.py"}, set())
        assert "read-files" in result
        assert "modified-files" not in result

    def test_both(self) -> None:
        result = _render_file_tracking({"src/a.py"}, {"src/b.py"})
        assert "read-files" in result
        assert "modified-files" in result

    def test_empty(self) -> None:
        assert _render_file_tracking(set(), set()) == ""
