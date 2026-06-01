from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path

from ..harness.skills import ToolInput, ToolSpec

"""按需加载 skill 的轻量目录。"""


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path
    use_when: tuple[str, ...] = ()
    dont_use_when: tuple[str, ...] = ()
    risk: str = "low"
    tools: tuple[str, ...] = ()


class SkillLoader:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills = self._scan()

    def get_descriptions(self) -> str:
        if not self.skills:
            return "No skills available."
        return "\n".join(
            f"- {name}: {skill.description}"
            for name, skill in sorted(self.skills.items())
        )

    def get_catalog(self) -> str:
        if not self.skills:
            return "<skill-catalog>No skills available.</skill-catalog>"
        blocks = [
            "<skill-catalog>",
            "These are skill summaries, not full instructions. Call load_skill with a skill name before following a skill.",
        ]
        for name, skill in sorted(self.skills.items()):
            blocks.append(
                f'<skill name="{name}" path="{skill.path.as_posix()}" risk="{skill.risk}">'
            )
            blocks.append(f"description: {skill.description or '(none)'}")
            if skill.use_when:
                blocks.append("use_when: " + "; ".join(skill.use_when))
            if skill.dont_use_when:
                blocks.append("dont_use_when: " + "; ".join(skill.dont_use_when))
            if skill.tools:
                blocks.append("suggested_tools: " + ", ".join(skill.tools))
            blocks.append(f'load: load_skill({{"name": "{name}"}})')
            blocks.append("</skill>")
        blocks.append("</skill-catalog>")
        return "\n".join(blocks)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            return f"Error: Unknown skill '{name}'."
        text = skill.path.read_text(encoding="utf-8")
        _meta, body = _parse_frontmatter(text)
        return f'<skill name="{skill.name}">\n{body.strip()}\n</skill>'

    def _scan(self) -> dict[str, SkillMetadata]:
        if not self.skills_dir.exists():
            return {}
        skills: dict[str, SkillMetadata] = {}
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            meta = _read_frontmatter(path)
            name = str(meta.get("name") or path.parent.name)
            description = str(meta.get("description") or "")
            skills[name] = SkillMetadata(
                name=name,
                description=description,
                path=path,
                use_when=_parse_list(
                    meta.get("use_when", "")
                    or meta.get("use-when", "")
                    or meta.get("Use when", "")
                    or meta.get("triggers", "")
                ),
                dont_use_when=_parse_list(
                    meta.get("dont_use_when", "")
                    or meta.get("don't_use_when", "")
                    or meta.get("dont-use-when", "")
                    or meta.get("Don't use when", "")
                    or meta.get("negative_triggers", "")
                    or meta.get("negative-triggers", "")
                ),
                risk=str(meta.get("risk") or "low"),
                tools=_parse_list(meta.get("tools", "")),
            )
        return skills


def build_skill_loader_tool(loader: SkillLoader) -> ToolSpec:
    def load_skill(data: ToolInput) -> str:
        return loader.get_content(str(data.get("name", "")).strip())

    return ToolSpec(
        name="load_skill",
        description="Load full SKILL.md instructions after choosing a skill from the skill catalog.\n"
        f"Available skills:\n{loader.get_descriptions()}",
        input_hint='JSON: {"name": "skill-name"}',
        handler=load_skill,
        risk="low",
        group="skills",
        read_only=True,
        concurrency_safe=True,
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, parts[2]


def _read_frontmatter(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if first.strip() != "---":
            return meta
        for line in handle:
            if line.strip() == "---":
                break
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip("\"'")
    return meta


def _parse_list(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        return ()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return tuple(item.strip().strip("\"'") for item in text.split(",") if item.strip())
