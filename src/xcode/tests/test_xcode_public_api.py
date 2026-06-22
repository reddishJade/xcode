from __future__ import annotations

import xcode as core
import xcode.coding_agent.tools as tools
import pytest
class XcodePublicApiTests:
    def test_experimental_tools_are_not_reexported_from_tools(self) -> None:
        assert not (hasattr(tools, "WorktreeTaskRunner"))
        assert not (hasattr(tools, "build_worktree_tools"))
        assert hasattr(tools, "build_file_tools")

    def test_eval_is_not_reexported_from_core(self) -> None:
        assert not (hasattr(core, "load_eval_questions"))
        assert not (hasattr(core, "run_end_to_end_eval"))

if __name__ == "__main__":
    pytest.main()
