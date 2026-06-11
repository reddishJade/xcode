from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .token_budget import (
    INSTRUCTION_OPENING_BYTES,
    INSTRUCTION_WARNING_BYTES,
    KEY_INSTRUCTION_SECTIONS,
    MAX_INSTRUCTION_BYTES,
    SECTION_BUDGET_BYTES,
    _utf8_prefix,
    _utf8_size,
)


@dataclass(frozen=True)
class ProjectInstruction:
    name: str
    content_hash: str
    prompt_text: str
    source_bytes: int
    warning: str | None = None


def _project_instruction_sources(project_root: Path) -> tuple[ProjectInstruction, ...]:
    sources = []
    for name in ("AGENTS.md", "CLAUDE.md"):
        path = project_root / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                prompt_text, warning = _prepare_project_instruction(name, text)
                content_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
                sources.append(
                    ProjectInstruction(
                        name=name,
                        content_hash=content_hash,
                        prompt_text=prompt_text,
                        source_bytes=_utf8_size(text),
                        warning=warning,
                    )
                )
    return tuple(sources)


def _render_project_instructions(
    sources: tuple[ProjectInstruction, ...],
) -> str:
    parts = []
    for source in sources:
        header = (
            f'<instruction-source name="{source.name}" '
            f'bytes="{source.source_bytes}" '
            f'prompt_bytes="{_utf8_size(source.prompt_text)}">'
        )
        body_parts = [header]
        if source.warning:
            body_parts.append(
                f"<instruction-warning>{source.warning}</instruction-warning>"
            )
        body_parts.append(source.prompt_text)
        body_parts.append("</instruction-source>")
        parts.append("\n".join(body_parts))
    return "\n\n".join(parts)


def _prepare_project_instruction(name: str, text: str) -> tuple[str, str | None]:
    source_bytes = _utf8_size(text)
    if source_bytes <= INSTRUCTION_WARNING_BYTES:
        return text, None
    if source_bytes <= MAX_INSTRUCTION_BYTES:
        warning = (
            f"{name} is {source_bytes} UTF-8 bytes, above the "
            f"{INSTRUCTION_WARNING_BYTES} byte warning threshold. Full content is "
            "included."
        )
        return text, warning

    condensed = _condense_project_instruction(text, MAX_INSTRUCTION_BYTES)
    warning = (
        f"{name} is {source_bytes} UTF-8 bytes, above the "
        f"{MAX_INSTRUCTION_BYTES} byte hard limit. Content was condensed; opening "
        "context, document index, and key rules are preserved while long examples "
        "and background are omitted. Use read_file before relying on omitted details."
    )
    return condensed, warning


def _condense_project_instruction(text: str, max_bytes: int) -> str:
    parts: list[str] = []
    opening = _utf8_prefix(text, INSTRUCTION_OPENING_BYTES).strip()
    if opening:
        parts.append(opening)

    directory = _extract_directory_lines(text)
    if directory:
        parts.append("## Preserved Directory\n\n" + "\n".join(directory))

    sections = _extract_key_sections(text)
    if sections:
        parts.append("## Preserved Key Rules\n\n" + "\n\n".join(sections))

    parts.append(
        "<instruction-omissions>Long examples, repeated background, and non-key "
        "sections were omitted because this instruction file exceeded the prompt "
        "budget.</instruction-omissions>"
    )
    return _utf8_prefix("\n\n".join(parts), max_bytes).strip()


def _extract_directory_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "docs/" in stripped or "docs\\" in stripped:
            lines.append(stripped)
            continue
        if stripped.startswith("|") and (
            "Document" in stripped or "When to read" in stripped or "---" in stripped
        ):
            lines.append(stripped)
    return _dedupe_preserve_order(lines)


def _extract_key_sections(text: str) -> list[str]:
    sections: list[str] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        if not current_heading or not current_lines:
            return
        heading_key = current_heading.strip("# ").strip().lower()
        if heading_key not in KEY_INSTRUCTION_SECTIONS:
            return
        section = "\n".join(_drop_fenced_blocks(current_lines)).strip()
        if section:
            sections.append(_utf8_prefix(section, SECTION_BUDGET_BYTES).strip())

    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            flush()
            current_heading = line
            current_lines = [line]
            continue
        if current_heading:
            current_lines.append(line)
    flush()
    return sections


def _drop_fenced_blocks(lines: list[str]) -> list[str]:
    result: list[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        result.append(line)
    return result


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        result.append(line)
    return result
