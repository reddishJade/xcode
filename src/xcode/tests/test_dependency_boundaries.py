"""包级依赖方向约束。"""

from __future__ import annotations

import ast
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_FORBIDDEN_TARGETS = {
    "ai": frozenset({"agent", "harness", "coding_agent", "cli", "experimental"}),
    "agent": frozenset({"harness", "coding_agent", "cli", "experimental"}),
    "harness": frozenset({"coding_agent", "cli", "experimental"}),
    "coding_agent": frozenset({"cli", "experimental"}),
}


def _source_module(path: Path) -> tuple[str, ...]:
    relative = path.relative_to(_PACKAGE_ROOT)
    parts = ("xcode", *relative.with_suffix("").parts)
    if path.name == "__init__.py":
        return parts[:-1]
    return parts


def _imported_modules(path: Path) -> tuple[tuple[str, ...], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    source = _source_module(path)
    package = source if path.name == "__init__.py" else source[:-1]
    imported: list[tuple[str, ...]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(tuple(alias.name.split(".")) for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        module = tuple((node.module or "").split(".")) if node.module else ()
        if node.level:
            keep = len(package) - (node.level - 1)
            imported.append((*package[:keep], *module))
        else:
            imported.append(module)
    return tuple(imported)


def test_package_dependencies_follow_layer_direction() -> None:
    """下层包不得引用上层产品或界面包。"""
    violations: list[str] = []
    for source_layer, forbidden in _FORBIDDEN_TARGETS.items():
        for path in sorted((_PACKAGE_ROOT / source_layer).rglob("*.py")):
            for imported in _imported_modules(path):
                if len(imported) < 2 or imported[0] != "xcode":
                    continue
                target_layer = imported[1]
                if target_layer in forbidden:
                    relative = path.relative_to(_PACKAGE_ROOT)
                    violations.append(
                        f"{relative}: {source_layer} -> {'.'.join(imported)}"
                    )
    assert not violations, "依赖方向违规:\n" + "\n".join(violations)
