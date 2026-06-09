from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from xcode.harness.skill_loader import SkillLoader, build_skill_loader_tool


class XcodeSkillLoaderTests(unittest.TestCase):
    def test_loader_reads_descriptions_without_full_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "git"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: git\ndescription: Git workflow helpers.\n---\n\nFull workflow.",
                encoding="utf-8",
            )

            loader = SkillLoader(Path(tmp) / "skills")

            self.assertIn("git: Git workflow helpers.", loader.get_descriptions())
            self.assertNotIn("Full workflow", loader.get_descriptions())

    def test_load_skill_returns_full_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "git"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: git\ndescription: Git workflow helpers.\n---\n\nFull workflow.",
                encoding="utf-8",
            )
            loader = SkillLoader(Path(tmp) / "skills")
            tool = build_skill_loader_tool(loader)

            output = tool.handler({"name": "git"})

            self.assertIn('<skill name="git">', output)
            self.assertIn("Full workflow.", output)

    def test_skill_catalog_omits_full_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "git"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: git\n"
                "description: Git workflow helpers.\n"
                "use_when: git workflow\n"
                "dont_use_when: explain git concepts only\n"
                "tools: bash, read_file\n"
                "---\n\n"
                "Full workflow.",
                encoding="utf-8",
            )
            loader = SkillLoader(Path(tmp) / "skills")

            catalog = loader.get_catalog()

            self.assertIn("<skill-catalog>", catalog)
            self.assertIn('<skill name="git"', catalog)
            self.assertIn("Git workflow helpers.", catalog)
            self.assertIn("use_when: git workflow", catalog)
            self.assertIn("dont_use_when: explain git concepts only", catalog)
            self.assertIn("suggested_tools: bash, read_file", catalog)
            self.assertIn('load_skill({"name": "git"})', catalog)
            self.assertNotIn("Full workflow.", catalog)

    def test_loader_reads_trigger_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "review"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: review\n"
                "description: Review code changes.\n"
                "use_when: code review, bug risk\n"
                "risk: low\n"
                "tools: read_file, grep_search\n"
                "---\n\n"
                "Review workflow.",
                encoding="utf-8",
            )

            skill = SkillLoader(Path(tmp) / "skills").skills["review"]

            self.assertEqual(skill.use_when, ("code review", "bug risk"))
            self.assertEqual(skill.tools, ("read_file", "grep_search"))

    def test_skill_loader_scans_catalog_without_reading_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "review"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            skill_path.write_text(
                "---\nname: review\ndescription: Review code.\n---\n\nBody.",
                encoding="utf-8",
            )

            skill = SkillLoader(Path(tmp) / "skills").skills["review"]

            self.assertEqual(skill.name, "review")
            self.assertEqual(skill.path, skill_path)
            self.assertFalse(hasattr(skill, "body"))

    def test_loader_exports_local_shell_skill_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skills" / "csv"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: csv-insights\ndescription: Summarize CSV files.\n---\n\nBody.",
                encoding="utf-8",
            )

            skills = SkillLoader(Path(tmp) / "skills").to_local_shell_skills()

            self.assertEqual(
                skills,
                [
                    {
                        "name": "csv-insights",
                        "description": "Summarize CSV files.",
                        "path": skill_dir.as_posix(),
                    }
                ],
            )

    def test_unknown_skill_reports_error(self) -> None:
        loader = SkillLoader(Path("missing"))
        tool = build_skill_loader_tool(loader)

        self.assertIn("Unknown skill", tool.handler({"name": "missing"}))


if __name__ == "__main__":
    unittest.main()
