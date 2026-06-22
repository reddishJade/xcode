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
- `.local/skills/active`
- `enabled.txt` skill activation
- Semantic skill matching
- Task-state volatile prompt injection
- Compaction-to-`TaskStore` duplication

Each context source has exactly one injection path — no fallback or secondary mechanism.

## 5. Skills Policy

**Search paths** (highest priority first):

0. `<explicit_skills_dir>` — configured via `paths.skills_dir` or `build_app(skills_dir=...)`
1. `<project_root>/.xcode/skills/` (only if `trust_project_skills=true`)
2. `<project_root>/.agents/skills/` (only if `trust_project_skills=true`)
3. `~/.xcode/skills/`
4. `~/.agents/skills/`

**Rules**:
- Skill discovery uses `SKILL.md` frontmatter; any `.md` file under a search path with valid YAML frontmatter is a candidate.
- Only `.md`, `.mdx`, and `.txt` files are considered.
- Higher-priority path overrides lower-priority same relative path (deduplication by relative path).
- Skill file max size: 50 KB (`_REFERENCE_MAX_BYTES`). No fixed depth, file count, or total byte cap.

## 6. Skill Loading

Skill loading follows a two-tier model:

- **`SkillIndexCollector`** (automatic, every turn): injects an `<available-skills>` block with skill name and description only. No body content is loaded.
- **`load_skill` tool** (on-demand, permissioned): loads the full `SKILL.md` body (after frontmatter) for a named skill.
- **`load_skill(name=..., reference=...)`** (on-demand): loads a single reference file from a skill's `references/` directory. Reference content is never auto-injected; the agent must request it explicitly.
- **`references/` metadata**: `load_skill` output includes a `<references>` block listing available reference filenames without their content. The agent can then load individual references via `load_skill`.

Scripts (`scripts/` directory) are ignored. No automatic activation or semantic matching is performed.

## 7. Compaction Boundary

- `[Compressed]` remains the canonical compacted conversation history prefix.
- Compaction (`LayeredCompactor`) does **not** duplicate content that is already covered by collectors:
  - Project manifest
  - Active diff
  - Skills
  - Notes
  - Task state
- `TaskStateCollector` explicitly excludes `[Compressed]` message content.

## 8. Safety Invariants

- **Single injection path**: each context source has exactly one injection path via its assigned collector.
- **SYSTEM before USER_CONTEXT**: all `SYSTEM`-target blocks are injected before any `USER_CONTEXT` block.
- **Marker integrity**: truncation markers (e.g. `<manifest-truncated>`) appear only on truncation and must be preserved whole; partial markers are never injected.
- **Bounded readers**: every collector caps its output in bytes (4 KB to 32 KB) and its source reads in file count / depth.
- **No placeholder values**: no `ContextBlockSource` enum value exists without a corresponding registered collector — every source is owned.
