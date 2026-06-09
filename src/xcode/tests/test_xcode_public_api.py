from __future__ import annotations

import unittest

import xcode as core
import xcode.coding_agent.tools as tools


class XcodePublicApiTests(unittest.TestCase):
    def test_experimental_tools_are_not_reexported_from_tools(self) -> None:
        self.assertFalse(hasattr(tools, "WorktreeTaskRunner"))
        self.assertFalse(hasattr(tools, "build_worktree_tools"))
        self.assertTrue(hasattr(tools, "build_file_tools"))

    def test_eval_is_not_reexported_from_core(self) -> None:
        self.assertFalse(hasattr(core, "load_eval_questions"))
        self.assertFalse(hasattr(core, "run_end_to_end_eval"))


if __name__ == "__main__":
    unittest.main()
