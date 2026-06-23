"""Agent Skills 数据模型。

SkillDef 是技能完整定义，SkillSummary 是轻量摘要，其余为附属类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillDiagnostic:
    """技能加载时的诊断信息。"""

    type: str  # "warning" | "error"
    code: str
    message: str
    path: str | None = None


@dataclass(frozen=True)
class SkillSummary:
    """技能摘要——不含正文的轻量元数据。"""

    name: str
    description: str
    disable_model_invocation: bool = False
    source: str = ""


@dataclass(frozen=True)
class SkillReference:
    """技能引用文件元数据。

    name 是 references/ 内部的规范化 POSIX 相对路径，
    如 "checklist.md" 或 "subdir/guide.md"。
    跳过（hidden/symlink/binary）的文件以 skipped=True 标记。
    """

    name: str
    size: int = 0
    truncated: bool = False
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True)
class SkillResource:
    """技能脚本或资产的相对路径元数据。"""

    path: str
    size: int = 0
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass(frozen=True)
class SkillDef:
    """技能的完整定义，包含懒加载的正文内容和引用元数据。"""

    name: str
    description: str
    file_path: Path
    disable_model_invocation: bool = False
    source: str = ""
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None
    frontmatter: dict[str, object] = field(default_factory=dict)
    content: str | None = None
    references: tuple[SkillReference, ...] = ()
    scripts: tuple[SkillResource, ...] = ()
    assets: tuple[SkillResource, ...] = ()

    def to_summary(self) -> SkillSummary:
        return SkillSummary(
            name=self.name,
            description=self.description,
            disable_model_invocation=self.disable_model_invocation,
            source=self.source,
        )
