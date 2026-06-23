"""技能搜索目录构建与引用/资源文件扫描。

不负责 frontmatter 解析，只做文件系统层面的发现和元数据收集。
"""

from __future__ import annotations

import logging
import os as _os
from pathlib import Path

from xcode.harness.agent_skills.models import SkillReference, SkillResource

logger = logging.getLogger(__name__)

# ── 搜索路径 ──

SOURCE_EXPLICIT = "explicit"
SOURCE_PROJECT = "project"
SOURCE_USER = "user"

_REFERENCE_MAX_BYTES = 50 * 1024


def build_skill_search_dirs(
    project_root: Path | None,
    *,
    trust_project_skills: bool = True,
    skills_dir: Path | None = None,
) -> list[tuple[Path, int]]:
    """构建优先级排序的技能搜索目录列表。

    返回 [(Path, priority), ...]，priority 越小优先级越高。
    项目级技能仅在调用方明确确认工作区可信时加入。
    显式 skills_dir 表示调用方已信任该目录，始终具有最高优先级。
    """
    dirs: list[tuple[Path, int]] = []
    if skills_dir is not None:
        explicit_dir = skills_dir.resolve()
        if not explicit_dir.is_dir():
            logger.warning(
                "Configured skill directory does not exist: %s", explicit_dir
            )
        dirs.append((explicit_dir, 0))
    if project_root is not None and trust_project_skills:
        dirs.append(((project_root / ".xcode" / "skills").resolve(), 1))
        dirs.append(((project_root / ".agents" / "skills").resolve(), 2))
    home = Path.home()
    dirs.append(((home / ".xcode" / "skills").resolve(), 3))
    dirs.append(((home / ".agents" / "skills").resolve(), 4))

    unique_dirs: list[tuple[Path, int]] = []
    seen_paths: set[Path] = set()
    for path, priority in dirs:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        unique_dirs.append((path, priority))
    return unique_dirs


def source_for_priority(priority: int) -> str:
    """返回优先级对应的来源标记。"""
    if priority == 0:
        return SOURCE_EXPLICIT
    if priority <= 2:
        return SOURCE_PROJECT
    return SOURCE_USER


# ── 引用文件扫描 ──


def scan_skill_references(
    skill_root: Path,
) -> tuple[list[SkillReference], dict[str, Path]]:
    """扫描技能根目录下的 references/ 并返回元数据。

    第二个返回值是引用名到解析路径的内部映射，用于懒加载。
    """
    ref_dir = skill_root / "references"
    if not ref_dir.is_dir():
        return [], {}

    references: list[SkillReference] = []
    ref_map: dict[str, Path] = {}
    seen_names: set[str] = set()

    for file_path in _collect_reference_files(ref_dir):
        rel_path = str(file_path.relative_to(ref_dir).as_posix())

        if rel_path in seen_names:
            logger.warning("Duplicate reference name %r; skipping both", rel_path)
            continue
        seen_names.add(rel_path)

        if any(part.startswith(".") for part in rel_path.split("/")):
            references.append(
                SkillReference(name=rel_path, skipped=True, skipped_reason="hidden")
            )
            continue

        if file_path.is_symlink():
            references.append(
                SkillReference(name=rel_path, skipped=True, skipped_reason="symlink")
            )
            continue

        try:
            st = file_path.stat()
        except OSError:
            references.append(
                SkillReference(
                    name=rel_path, skipped=True, skipped_reason="unreadable"
                )
            )
            continue

        if _is_binary_file(file_path):
            references.append(
                SkillReference(
                    name=rel_path,
                    size=st.st_size,
                    skipped=True,
                    skipped_reason="binary",
                )
            )
            continue

        truncated = st.st_size > _REFERENCE_MAX_BYTES
        references.append(
            SkillReference(
                name=rel_path,
                size=st.st_size,
                truncated=truncated,
            )
        )
        ref_map[rel_path] = file_path

    references.sort(key=lambda r: r.name)
    return references, ref_map


def scan_skill_resources(
    skill_root: Path,
    directory_name: str,
) -> tuple[SkillResource, ...]:
    """扫描 scripts/ 或 assets/，仅返回相对路径元数据。"""
    resource_dir = skill_root / directory_name
    if not resource_dir.is_dir():
        return ()

    resources: list[SkillResource] = []
    for file_path in _collect_reference_files(resource_dir):
        relative_path = file_path.relative_to(skill_root).as_posix()
        if any(part.startswith(".") for part in relative_path.split("/")):
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="hidden",
                )
            )
            continue
        if file_path.is_symlink():
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="symlink",
                )
            )
            continue
        try:
            size = file_path.stat().st_size
        except OSError:
            resources.append(
                SkillResource(
                    path=relative_path,
                    skipped=True,
                    skipped_reason="unreadable",
                )
            )
            continue
        resources.append(SkillResource(path=relative_path, size=size))
    return tuple(sorted(resources, key=lambda resource: resource.path))


def load_reference_text(path: Path) -> tuple[str | None, bool]:
    """读取引用文件内容，应用大小预算（截断）。

    返回 (content, truncated)，读取失败时 content 为 None。
    内容以 UTF-8 解码，含替换字符。
    """
    try:
        with open(path, "rb") as f:
            data = f.read(_REFERENCE_MAX_BYTES + 1)
        truncated = len(data) > _REFERENCE_MAX_BYTES
        if truncated:
            data = data[:_REFERENCE_MAX_BYTES]
        text = data.decode("utf-8", errors="replace")
        return text, truncated
    except (OSError, MemoryError):
        return None, False


def _collect_reference_files(ref_dir: Path) -> list[Path]:
    """收集目录下的文件，不跟踪符号链接，跳过隐藏目录。"""
    files: list[Path] = []
    try:
        for dirpath, dirnames, filenames in _os.walk(ref_dir, followlinks=False):
            dir_path = Path(dirpath)
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not d.startswith(".") and not (dir_path / d).is_symlink()
            )
            for f in sorted(filenames):
                files.append(dir_path / f)
    except OSError:
        pass
    return files


def _is_binary_file(path: Path) -> bool:
    """检测文件是否为二进制（含空字节或非 UTF-8 可解码）。"""
    try:
        data = path.read_bytes()[:1024]
        if b"\0" in data:
            return True
        data.decode("utf-8")
        return False
    except (OSError, UnicodeDecodeError):
        return True
