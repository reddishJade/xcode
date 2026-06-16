# Context Pipeline Architecture

## 1. Target Architecture

```
                    ┌──────────────────────┐
                    │ ContextCollectorRegistry │
                    │  (ordered collectors)  │
                    └──────┬───────────────┘
                           │ collect()
                           ▼
                    ┌──────────────┐
                    │ ContextBlock[] │
                    └──────┬───────┘
                           │ assemble()
                           ▼
                    ┌──────────────────┐
                    │ DefaultContextAssembler │
                    │  → inject into messages │
                    │  → trim to token budget  │
                    │  → filter expired blocks │
                    └──────────────────┘
```

- `ContextCollectorRegistry` registers and runs an ordered list of `ContextCollector` instances.
- Each collector returns zero or more `ContextBlock` values with a `source`, `priority`, `target`, and `content`.
- `DefaultContextAssembler` merges blocks into the provider message list by target (`SYSTEM` → `SystemMessage`, `USER_CONTEXT` → `UserMessage`) and trims to budget using greedy priority-fill.
- Collectors own their context sources (filesystem, git, message history, task store) and are independently testable.
- No old prompt side channels — no `instructions.py`, no semantic `SkillLoader`, no `PromptTemplate` registry.

## 2. Context Source Map

| Source | Collector | Target | Priority |
|---|---|---|---|
| `INSTRUCTION` | `InstructionCollector` | `SYSTEM` | `CRITICAL` |
| `ACTIVE_DIFF` | `ActiveDiffCollector` | `USER_CONTEXT` | `HIGH` |
| `RECENT_VALIDATION` | `RecentValidationCollector` | `USER_CONTEXT` | `HIGH` |
| `TASK_STATE` | `TaskStateCollector` | `USER_CONTEXT` | `HIGH` |
| `NOTES` | `NotesCollector` | `USER_CONTEXT` | `MEDIUM` |
| `SKILL` | `SkillIndexCollector` | `USER_CONTEXT` | `MEDIUM` |

## 3. Collector Order (Registration Order)

1. `InstructionCollector`
2. `ActiveDiffCollector`
3. `RecentValidationCollector`
4. `TaskStateCollector`
5. `NotesCollector`
6. `SkillIndexCollector`

Collectors run in registration order. Assembly sorts by priority within the budget; higher-priority blocks are retained first when trimming.

## 4. Explicit Non-Goals / Removed Paths

The following paths have been removed and must not be reintroduced:

- `instructions.py` project instruction path
- Semantic `SkillLoader`
- Skill catalog prompt injection
- `load_skill` tool
- `.local/skills/active`
- `enabled.txt` skill activation
- Semantic skill matching
- Task-state volatile prompt injection
- Compaction-to-`TaskStore` duplication

Each context source has exactly one injection path — no fallback or secondary mechanism.

## 5. Skills Policy

**Search paths** (highest priority first):

1. `<project_root>/.xcode/skills/`
2. `<project_root>/.agents/skills/`
3. `~/.xcode/skills/`
4. `~/.agents/skills/`

**Rules**:
- File presence is activation — any `.md`, `.mdx`, or `.txt` file under a search path is included.
- Only `.md`, `.mdx`, and `.txt` files are considered.
- Higher-priority path overrides lower-priority same relative path (deduplication by relative path).
- Bounded traversal: max depth 3, max 50 files, single file ≤ 64 KB, total ≤ 16 KB.
- Path traversal is prevented by `_is_child_path` check on symlink-resolved real paths.

## 6. Compaction Boundary

- `[Compressed]` remains the canonical compacted conversation history prefix.
- Compaction (`LayeredCompactor`) does **not** duplicate content that is already covered by collectors:
  - Project manifest
  - Active diff
  - Skills
  - Notes
  - Task state
- `TaskStateCollector` explicitly excludes `[Compressed]` message content.

## 7. Safety Invariants

- **Single injection path**: each context source has exactly one injection path via its assigned collector.
- **SYSTEM before USER_CONTEXT**: all `SYSTEM`-target blocks are injected before any `USER_CONTEXT` block.
- **Marker integrity**: truncation markers (e.g. `<manifest-truncated>`) appear only on truncation and must be preserved whole; partial markers are never injected.
- **Bounded readers**: every collector caps its output in bytes (4 KB to 32 KB) and its source reads in file count / depth.
- **No placeholder values**: no `ContextBlockSource` enum value exists without a corresponding registered collector — every source is owned.
