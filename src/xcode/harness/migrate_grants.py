"""一次性迁移脚本：将旧的 hitl_policy.json 转换为新的 approval_grants.json 格式。

用法：
  uv run python -m xcode.harness.migrate_grants /path/to/project/root

旧格式（hitl_policy.json）：
  [{"tool": "bash", "decision": "allow", "input_contains": "git status"}]

新格式（approval_grants.json）：
  [{"capability": "shell", "operation": "run_command", "target_kind": "command",
    "target_pattern": "git status", "access": "execute", "decision": "allow",
    "scope": "permanent", "grant_id": "..."}]

仅迁移可安全映射的 legacy grants：
- tool="*" 通配规则 → 跳过（无法映射到单一 capability）
- input_prefix 规则 → 跳过（前缀无法可靠归一化）
- 非 read_file/write_file/edit_file/apply_patch 工具 → 跳过（无结构化 target）
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from uuid import uuid4


def migrate_project(project_root: Path) -> None:
    old_path = project_root / ".local" / "hitl_policy.json"
    new_path = project_root / ".local" / "approval_grants.json"

    if not old_path.exists():
        print(f"no hitl_policy.json found at {old_path}")
        return

    old_data = json.loads(old_path.read_text(encoding="utf-8"))
    if not isinstance(old_data, list):
        print(f"invalid hitl_policy.json format at {old_path}")
        return

    new_records: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    from xcode.harness.observability import ActionExtractor

    for item in old_data:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool", "")
        decision = item.get("decision", "")
        input_contains = item.get("input_contains")
        input_prefix = item.get("input_prefix")

        if tool == "*":
            skipped.append({"rule": str(item), "reason": "wildcard tool cannot map"})
            continue
        if input_prefix is not None:
            skipped.append({"rule": str(item), "reason": "input_prefix cannot map"})
            continue
        if tool not in ("read_file", "write_file", "edit_file", "apply_patch"):
            skipped.append({"rule": str(item), "reason": f"non-structured tool {tool}"})
            continue
        if decision not in ("allow", "deny"):
            continue

        action = ActionExtractor().extract(
            tool, {"path": input_contains or ""}
        )
        if not action.targets:
            skipped.append({"rule": str(item), "reason": "no extractable target"})
            continue

        for target in action.targets:
            new_records.append(
                {
                    "capability": action.capability,
                    "operation": action.operation,
                    "target_kind": target.kind,
                    "target_pattern": input_contains or "",
                    "access": target.access,
                    "decision": decision,
                    "scope": "permanent",
                    "grant_id": uuid4().hex,
                }
            )

    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(
        json.dumps(new_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"migrated {len(new_records)} grants to {new_path}")

    if skipped:
        skip_path = project_root / ".local" / "migration_skipped.json"
        skip_path.write_text(
            json.dumps(skipped, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {len(skipped)} unmappable grants written to {skip_path}")
        print("  review manually and re-approve if needed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m xcode.harness.migrate_grants <project_root>")
        sys.exit(1)
    migrate_project(Path(sys.argv[1]))
