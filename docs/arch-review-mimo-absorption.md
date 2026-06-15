# MiMo Feature Absorption — Architecture Review Packet

> **Status**: Review  
> **Scope**: Architectural gaps and canonical integration points  
> **Authority**: Decisions below supersede any prior proposal

---

## Blocking Corrections Before Implementation

The following five corrections are mandatory before any implementation begins. They fix security-model, boundary-classification, and path-scoping errors in this document's prior draft.

### 0.1 last-match-wins Is Local to StaticPolicyEvaluator

last-match-wins applies **only** inside `StaticPolicyEvaluator` when resolving a single static-permission rule block against an action. It determines which rule in the block emits the `StaticPolicy` constraint.

It does **not** replace the global `PermissionResolver` ordering. The global ordering remains:

```
non_bypassable_deny > deny > ask > allow
```

SystemSafety (`_is_sensitive_path`, `.env` rules), Boundary (`StructuredBoundaryPolicyEvaluator`), and RuntimeSafety (`SafetyBackstopPolicyEvaluator`) deny constraints **cannot** be overridden by a later static `allow` rule. They are emitted at a higher priority layer and `PermissionResolver` already orders them above `StaticPolicyEvaluator` constraints.

### 0.2 `external_directory` Is a First-Class Boundary Root Set

Do **not** implement `external_directory` as an exception after workspace-escape denial. The check must be:

1. Normalize the target path.
2. Classify it into exactly one of:
   - inside workspace root
   - inside an approved `external_directory`
   - outside all approved roots
3. Only case 3 is workspace-escape → deny.

`external_directory` is a first-class root set in the Boundary axis, not a patch after escape detection.

### 0.3 `.env` Matching Uses Basename Rules

Use `Path(path).name` for basename matching:

- `.env.example` read → **allow**, write → **deny** (it is documentation).
- `.env` and `.env.*` → **deny** by default (consistent with SystemSafety).

Do not split on `/` and iterate parts: the part loop misses `.env.local` because `Path(".env.local").name` is `".env.local"` but the part after splitting `"config/.env.local"` is `"env.local"`, not `".env.local"`.

### 0.4 `load_skill` Remains a Tool; Remove Only Bypass Implementations

The `load_skill` tool name and `ToolSpec` are kept. Only old implementations that bypass `SkillRegistry` are removed. The handler **must** delegate to `SkillRegistry.load()`.

### 0.5 MCP Server Registration Is Config-Gated

MCP is canonical, but only `enabled=true` servers are registered. All MCP tools enter through the same `ToolRegistry`/`ToolGate` path. No unconditional startup of all MCP servers.

---

## 1. Current Architecture Inventory

### 1.1 Canonical Modules and Files

```
src/xcode/
├── main.py                         # CLI entry: arg parse, config discovery, build_app(), run_repl()
│
├── cli/                            # Layer 1: Coding Product UI
│   ├── repl.py                     #   REPL main loop (read-eval-print loop)
│   ├── repl_commands.py            #   Slash command registry (COMMAND_REGISTRY dict)
│   ├── repl_hitl.py                #   HITL approval callback (ReplHITLHandler)
│   ├── repl_sessions.py            #   Session resume, history conversion
│   ├── repl_settings.py            #   /model /effort /thinking /permissions handlers
│   ├── repl_tools.py               #   /tool + !command shortcuts
│   ├── repl_turn_handler.py        #   Event→render decomposition
│   ├── repl_rendering.py           #   Terminal rendering, banner
│   ├── commands.py                 #   CommandRegistry, ReplState, CommandContext
│   ├── completion.py              #   Tab completion (commands, tools, @file)
│   ├── file_refs.py                #   @relative/path → file content injection
│   ├── markdown.py                 #   TerminalMarkdownRenderer
│   └── tool_catalog.py             #   Tool introspection
│
├── coding_agent/                   # Layer 2: Coding Product Tools
│   ├── registry.py                 #   build_project_scoped_registry() — tool factory
│   └── tools/
│       ├── file.py                 #   read_file, write_file, edit_file ToolSpecs
│       ├── file_handlers.py        #   Handler implementations for file tools
│       ├── file_image.py           #   Image file handling
│       ├── file_mutation_queue.py  #   Queued file mutations
│       ├── code_search.py          #   glob_files, grep_search, find_files, ls
│       ├── bash.py                 #   bash tool via ExecutionEnv
│       ├── shell_adapter.py        #   ShellSpec, detect_shell
│       ├── worktree.py             #   Worktree task tools
│       ├── path_utils.py           #   Path resolution helpers
│       ├── edit_diff.py            #   Diff-based editing
│       ├── _constants.py           #   Timeout constants
│       ├── tools_manager.py        #   System tool detection (fd, ripgrep)
│       ├── output_accumulator.py   #   Subprocess output accumulation
│       └── truncate.py             #   Output truncation
│
├── harness/                        # Layer 3: Runtime Infrastructure
│   ├── app.py                      #   XcodeApp dataclass, build_app()
│   ├── assembly.py                 #   resolve_config(), build_shared_infra(), build_agent()
│   │                               #   build_tool_registry()
│   ├── config.py                   #   9 runtime config dataclasses, discover_runtime_config()
│   ├── session.py                  #   SessionStore — JSONL session storage, fork, rewind
│   ├── skills.py                   #   ToolSpec dataclass, build_tool_prompt()
│   ├── execution_env.py            #   ExecutionEnv protocol, SubprocessExecutionEnv
│   ├── task_store.py               #   Task storage + CRUD
│   ├── task_progress.py            #   Progress tracking tools
│   ├── mailbox.py                  #   Agent mailbox
│   ├── daemon.py                   #   HeartbeatDaemon
│   ├── migrate_grants.py           #   Grant migration
│   │
│   ├── observability/
│   │   ├── permissions.py          #   PermissionEngine, PermissionPolicy, PermissionRule
│   │   ├── permission_model.py     #   4-axis model: Action, Target, Constraint, Verdict
│   │   │                          #   + all evaluators + PermissionResolver
│   │   ├── _safety_backstop.py     #   SafetyBackstopPolicyEvaluator
│   │   ├── audit.py                #   AuditRecord, JsonlAuditLogger
│   │   └── hooks.py                #   HookManager, HookRecord
│   │
│   └── agent_runtime/
│       ├── structured.py           #   StructuredAgent — harness adapter for Agent loop
│       ├── config.py               #   GateConfig, AgentRuntimeConfig, build_loop_config()
│       ├── events.py               #   StructuredAgentEvent types + translation
│       ├── result.py               #   RunState, StructuredAgentResult
│       ├── execution_modes.py      #   ExecutionModeState, PlanPolicy, ReviewPolicy, ActPolicy
│       ├── tool_gate.py            #   ToolGate — permission + approval + mode filtering
│       ├── tool_adapter.py         #   ToolSpecAdapter: ToolSpec → AgentTool
│       ├── tool_hooks.py           #   Hook emission helpers
│       ├── tool_audit.py           #   Audit logging
│       ├── history_manager.py      #   HistoryManager — message storage, load, restore
│       ├── message_codec.py        #   messages_from_compacted_dicts()
│       ├── compaction.py           #   CompactController, LayeredCompactor
│       ├── contextual.py           #   ContextualRetrievalState — LRU file/result tracking
│       ├── git_preflight.py        #   build_git_preflight()
│       ├── agent_helpers.py        #   to_dict(), run_coro_sync()
│       ├── async_worker.py         #   IsolatedAsyncWorker
│       ├── subagent.py             #   ManagedSubagentRunner
│       ├── cancellation.py         #   CancellationToken
│       ├── fallback.py             #   _FallbackSwitchingProvider
│       └── prompting/
│           ├── builder.py          #   SystemPromptBuilder — 3-region prompt construction
│           └── identity.py         #   CORE_IDENTITY, TOOL_DISCIPLINE, SEARCH_STRATEGY
│
├── agent/                          # Layer 4: Agent Loop Core
│   ├── agent.py                    #   Agent — thin wrapper
│   ├── agent_loop.py               #   Core loop: compact → model → tool_exec → retry
│   ├── context_assembly.py         #   ContextBlock, DefaultContextAssembler
│   ├── context_collector.py        #   ContextCollectorRegistry + 6 collectors
│   ├── _provider.py                #   call_provider() — orchestrates context + LLM call
│   ├── messages.py                 #   AgentMessage union types
│   ├── events.py                   #   AgentEvent union types
│   ├── types.py                    #   Content block types
│   ├── config.py                   #   AgentContext, AgentLoopConfig, hooks
│   ├── protocols.py                #   AgentTool protocol
│   ├── results.py                  #   AgentLoopMetrics, AgentLoopResult
│   ├── tool_execution.py           #   Tool call execution
│   ├── _tool_scheduling.py         #   Partition by execution mode
│   ├── _tool_validation.py         #   JSON Schema argument validation
│   ├── compaction.py               #   Token estimation, should_compact_token_aware()
│   ├── message_converter.py        #   convert_to_llm() — AgentMessage → provider dict
│   ├── history.py                  #   repair_tool_pairing(), apply_request_hygiene()
│   ├── hooks.py                    #   Hook type aliases
│   └── watchdog.py                 #   Repeated-tool watchdog, idle tool watchdog
│
├── ai/                             # Layer 5: AI Provider
│   ├── registry.py                 #   Model registry
│   ├── model_modes.py              #   ModelMode, parse_model_mode()
│   ├── types.py                    #   Model, Usage, ToolDefinition
│   ├── events.py                   #   Provider stream events
│   ├── cache.py                    #   Cache statistics, tool stabilization
│   └── providers/
│       ├── protocol.py             #   ModelProvider protocol
│       ├── factory.py              #   build_provider_bundle()
│       ├── _registry.py            #   PROVIDER_REGISTRY
│       ├── router.py               #   RouterProvider
│       ├── runtime.py              #   ProviderRuntime
│       ├── metrics.py              #   ProviderMetricsMixin
│       ├── codec.py                #   Schema/delta codec
│       ├── stream_codec.py         #   Stream delta→event codec
│       ├── openai_compat.py        #   OpenAI Chat base
│       └── {openai,deepseek,chatglm,mimo,faux}.py
│
└── experimental/
    ├── mcp.py                      # MCP tool registration, defer_loading, cache
    ├── mcp_client.py               # Raw MCP stdio JSON-RPC client
    ├── memory.py                   # MemoryManager
    ├── memory_parsing.py           # Memory block parsing
    └── plugins.py                  # PluginManager — .local/plugins/*.py
```

### 1.2 Context Assembly Path

**Two-system architecture**, both called from `agent/_provider.py:call_provider()`:

**System A** — `harness/agent_runtime/prompting/builder.py`  
Produces a single system prompt string. Three cache regions:

| Region | Modules | Cache Key |
|---|---|---|
| STABLE | identity, tool_discipline, tools, search_strategy | Registry fingerprint |
| DYNAMIC | environment, cwd | CWD signature |
| VOLATILE | git_preflight, contextual_retrieval, notices | Rebuilt each turn |

Result: `SystemMessage(content=<string>)` at position 0 of the message list.

**System B** — `agent/context_collector.py` + `agent/context_assembly.py`  
Injects structured `ContextBlock`s at SYSTEM or USER_CONTEXT targets.

| Collector | Source | Priority | Target |
|---|---|---|---|
| ProjectManifestCollector | AGENTS.md / CLAUDE.md | CRITICAL | SYSTEM |
| ActiveDiffCollector | git diff --unified=1 | HIGH | USER_CONTEXT |
| RecentValidationCollector | Last failed bash | HIGH | USER_CONTEXT |
| TaskStateCollector | Task store state | HIGH | USER_CONTEXT |
| NotesCollector | .local/notes/ files | MEDIUM | USER_CONTEXT |
| SkillCollector | Skill directories | MEDIUM | USER_CONTEXT |

### 1.3 Permission Path

**Five evaluator layers**, composed via `evaluate_policy_constraints()`:

| Layer | Evaluator | Scope |
|---|---|---|
| 0 | restricted_dirs substring check | All tools via action_input string |
| 1 | ModePolicyEvaluator | Plan/Review policy → deny/allow per tool name |
| 2 | StaticPolicyEvaluator | Config rules: deny_tools, ask_tools, allow_tools |
| 3 | StructuredBoundaryPolicyEvaluator | File tools: workspace escape, sensitive paths, git paths |
| 4 | SafetyBackstopPolicyEvaluator | Shell commands: dangerous/risky/safe classification |

Priority: `non_bypassable_deny > deny > ask > allow`. On `ask` → `compute_shadow_approval_candidate()` checks grant stores → `approval_callback()`.

### 1.4 Session/REPL Path

`cli/repl.py` → `run_repl()`:

1. Create `SessionStore(sessions_dir)` — JSONL files
2. Create `ReplHITLHandler` with `InMemoryGrantStore` + `FileGrantStore`
3. Main loop: `prompt_toolkit` input → slash commands / normal text / !commands
4. Normal text → `store.append()` → expand @file → `_run_agent_turn()`
5. Session tree: `fork()`, `fork_clean()`, `resume()`, `rewind_turns()`

### 1.5 Skills Path

**Two parallel paths** (overlapping):

| Path | Mechanism | When |
|---|---|---|
| SkillCollector | Scans skill dirs → ContextBlock | Every provider request, eager |
| `load_skill` tool | ToolSpec under group "skills" | Agent can call at runtime |

### 1.6 Tool Registry Path

`assembly.py:build_tool_registry()` → flat `tuple[ToolSpec, ...]`. Group filtering via `enabled_groups`. MCP is registered under the experimental `mcp` group.

### 1.7 Legacy/Duplicate Paths

| Duplicate | Severity |
|---|---|
| Two compaction systems (`agent/compaction.py` vs `harness/agent_runtime/compaction.py`) | Harness path is primary; agent path is shadow |
| Two permission paths (cutover flags) | Cutover nearly complete; dead code should be removed |
| Two skill paths (SkillCollector vs `load_skill` tool) | Overlapping; must canonicalize |
| Two compact threshold config fields | Both active |
| `ReviewPolicy` exists as canonical mode policy | Must be removed |

---

## 2. Capability-to-Architecture Mapping

### Axes

| Axis | Meaning |
|---|---|
| Mode | First-class execution mode: plan, build, act |
| Capability | Feature flag / skill group / optional behavior |
| Boundary | File system, external directory, env file access rules |
| Approval | HITL allow/deny/ask with scope (once/session/permanent), per-agent overrides |
| Context | System prompt assembly, instruction injection, file referencing |
| Session | Persistence, fork, resume, rewind, undo, grant isolation |
| Tool Registry | Tool discovery, registration, gating, timeout, enabled flags |
| Skill Registry | Skill discovery (SKILL.md), lazy loading, permission visibility |
| UI | Terminal interactive commands, shortcuts, rendering |

### Mapping

| # | Capability | Axis | Current | Gap |
|---|---|---|---|---|
| 1 | `/undo` | Session | No undo model | Missing undo stack with inverse operations through PermissionPipeline |
| 2 | `@file` fuzzy search | UI/Context | `@file` exists but exact path only | Gap: fuzzy filename matching |
| 3 | Multiline input shortcuts | UI | Single-line prompt only | Gap: no multiline mode |
| 4 | `/new` | Session | `/clear` alias | Supported (must clear session grants) |
| 5 | `/clear` | Session | Full reset + grant clear | Supported (must clear session grants, preserve permanent) |
| 6 | `/sessions` | Session | Interactive picker | Supported |
| 7 | `/resume` | Session | By id/title/last/interactive | Supported |
| 8 | `/continue` | Session | No equivalent | Gap |
| 9 | `--continue` CLI | Session | No CLI flag | Gap |
| 10 | `--session` CLI | Session | No CLI flag | Gap |
| 11 | `--fork` CLI | Session | No CLI flag | Gap |
| 12 | Build Mode | Mode | Does not exist | Must add as third canonical mode |
| 13 | Plan Mode | Mode | Exists | Supported; handoff → build (not act) |
| 14 | allow/ask/deny model | Approval | StaticPermission with input_contains/input_prefix | Partial: needs last-match-wins, input_regex, structured target |
| 15 | Global `*` permission defaults | Approval | `PermissionRule("*", "ask")` only | Partial: not composable with input patterns |
| 16 | Per-tool permission overrides | Approval | Via deny/ask/allow tool lists | Supported after last-match-wins consolidation |
| 17 | Object syntax with input-pattern rules | Approval | input_contains, input_prefix only | Gap: input_regex, structured target patterns |
| 18 | Last-match-wins within pattern block | Approval | `PermissionPolicy.decide()` uses deny > ask > allow priority for static rules | Gap: `StaticPolicyEvaluator` must adopt last-match-wins to resolve its rule block before emitting one constraint; global `PermissionResolver` ordering unchanged |
| 19 | external_directory boundary | Boundary | Single root, absolute paths denied | Gap: no canonical external_directory |
| 20 | .env deny, .env.example allow | Boundary/Safety | .env and .env.* both denied unconditionally | Gap: .env.example read exception |
| 21 | once/session/permanent | Approval | GrantScope exists | Supported |
| 22 | Session-scoped approval grants | Approval | InMemoryGrantStore, manual clear per fork | Partial: needs automatic session isolation |
| 23 | Per-agent permission overrides | Approval | Subagent inherits parent policy | Gap: no separate agent-specific permission config |
| 24 | Skill registry + lazy loading | Skill Registry | Eager full injection, no registry | Gap: must add SkillRegistry + SkillIndexCollector + lazy load_skill |
| 25 | Skill permission controls | Skill Registry | None | Gap: skill loading is a permissioned tool action |
| 26 | Configured instructions array | Context | AGENTS.md only | Gap: ordered instruction sources |
| 27 | AGENTS.md/CLAUDE.md | Context | Supported | Supported |
| 28 | MCP config, enabled, timeout, gating | Tool Registry | Experimental, no per-server enabled/timeout | Gap: must become canonical ToolRegistry citizen |
| 29 | doom_loop guard | Tool Registry | watchdog.py | Supported |
| 30 | glob/grep via ripgrep + .gitignore | Tool Registry | Supported | Supported |
| 31 | todowrite-style task list | Session/UI | task_store.py exists, no UI shortcut | Gap: no /todowrite command |
| 32 | Review as mode | Mode | Exists as ReviewPolicy | **Must be removed.** Review becomes skill/slash command/prompt profile |

---

## 3. Architecture Decisions (Canonical)

Decisions below are authoritative. They supersede any previously proposed design.

---

### 3.1 Skill Design

**SKILL.md + YAML frontmatter** as the metadata index format.

**Required frontmatter:** `name`, `description`

**Optional frontmatter:** `hidden`

**Frontmatter is metadata for progressive loading only — not a replacement for PermissionPipeline.**

**Do NOT require** `risk`, `tools`, `permissions`, or `triggers` at this stage.

**Skill loading is itself a permissioned tool action.** Loading a skill goes through PermissionPipeline as a tool call.

**Backend: `SkillRegistry` is the sole backend** for skill discovery, indexing, permission visibility, and lazy loading. There is exactly one skill backend.

**Injection: `SkillIndexCollector`** injects only available skill summaries (name + description) into context. Not full skill bodies.

**Loading: `load_skill` remains the lazy-load tool** and must use `SkillRegistry`. It is NOT removed.

**Removal:** Remove eager full-skill injection from `SkillCollector`. `SkillCollector` is replaced by `SkillIndexCollector`.

**Summary:**

| Path | Before | After |
|---|---|---|
| Format | Xcode-style files | SKILL.md + YAML frontmatter |
| Frontmatter | Freeform | name (req), description (req), hidden (opt) |
| Discovery | Filesystem walking | SkillRegistry.discover() |
| Injection | Eager full content | SkillIndexCollector → summaries only |
| Loading | Tool or collector | load_skill via SkillRegistry (permissioned) |
| Effect | Two parallel paths | One SkillRegistry backend |

---

### 3.2 Execution Modes

**Three canonical modes exactly:**

| Mode | Purpose | Side Effects |
|---|---|---|
| `plan` | Restricted planning/analysis | No code/file side effects |
| `act` | Default safe development mode | Edits and risky commands require HITL |
| `build` | Edit/acceptEdits-like mode | Ordinary file mutations allowed; high-risk actions through PermissionPipeline |

**Default mode: `act`.** Xcode starts in `act` unless configured otherwise.

**Remove `review` as a canonical execution mode.** `ReviewPolicy` is removed.

**Review behavior is expressed as one of:**
- A code-review skill (SKILL.md)
- A slash command that starts a review task
- A prompt/instruction profile

**Replace `ReviewPolicy` with `BuildPolicy`.**

**Transition rule:** Plan timeout or explicit plan approval → transition to **`build`** (not `review`, not `act`).

**Summary:**

| Component | Before | After |
|---|---|---|
| Modes | plan, act, review | plan, act, build |
| Default | unclear | act |
| Review | First-class mode | Skill/slash command/prompt profile |
| Policy | ReviewPolicy exists | BuildPolicy replaces it |
| Plan handoff | → review or act | → build |
| Plan auto-timeout | → act | → build |

---

### 3.3 `/new` and `/clear`

**Both represent new session / clear context behavior.**

**They MUST:**
- Clear session-scoped grants
- Preserve permanent grants

**Do NOT add `/reset-context`.**

**Do NOT design `/new` to preserve session grants.** Session isolation is critical for security. Permanent grants remain for user convenience.

**Summary:**

| Property | Before | After |
|---|---|---|
| Session grants on /new | ambiguous | cleared |
| Permanent grants on /new | ambiguous | preserved |
| /reset-context | potentially needed | do not add |

---

### 3.4 `/rewind` vs `/undo`

| Command | Scope | Mechanism |
|---|---|---|
| `/rewind` | Conversation/session history only | History line deletion |
| `/undo` | Reversible tool side effects | Inverse mutations through PermissionPipeline |

**Critical restrictions:**
- **Do NOT implement `/undo` as JSONL history deletion.**
- **Do NOT let `/undo` bypass PermissionPipeline.** Every inverse mutation must be routed through the same permission checks as the original call.

**Summary:**

| Concern | Before | After |
|---|---|---|
| `/rewind` target | ambiguous | conversation history only |
| `/undo` target | potentially JSONL deletion | inverse tool mutations |
| Permission bypass | possible for undo | forbidden |

---

### 3.5 MCP (Model Context Protocol)

**MCP is NOT optional or experimental in the target architecture.**

**MCP must become part of the canonical `ToolRegistry`.** There is exactly one tool registry. Do not create a second MCP-specific registry.

**MCP tools must be registered through the same `ToolRegistry` path** as built-in tools.

**Implementation requirements (before remote/OAuth details):**
- Per-server `enabled` flag
- Timeout configuration
- Tool gating (same as built-in tools)
- Source metadata (server name, transport kind)
- Context-budget awareness

**Summary:**

| Property | Before | After |
|---|---|---|
| Status | Optional/experimental | Required, canonical |
| Registry | Experimental group | Canonical ToolRegistry |
| Per-server config | None | enabled, timeout, transport |
| Permission model | None (tool-level only) | Same as built-in tools |

---

### 3.6 Summary Table of All Changes

| Component | Before | After |
|---|---|---|
| **Skill format** | Xcode-style insufficient | SKILL.md + YAML frontmatter |
| **Skill frontmatter** | Freeform | name(req), description(req), hidden(opt) |
| **Skill loading** | Eager full injection | Lazy via load_skill, summaries only |
| **Skill backend** | Parallel collector + tool | Single SkillRegistry |
| **Skill permission** | None | Loading is a permissioned tool action |
| **Modes** | plan, act, review | plan, act, build |
| **Default mode** | unclear | act |
| **Review** | First-class mode | Skill, slash command, or prompt profile |
| **Policy** | ReviewPolicy exists | Replaced by BuildPolicy |
| **Plan handoff** | → review or act | → build |
| **/new / /clear** | Ambiguous grant handling | Clear session grants, preserve permanent |
| **/reset-context** | Potentially exists | Do not add |
| **/rewind** | History rollback | Conversation history only |
| **/undo** | Possibly JSONL deletion | Inverse mutations via PermissionPipeline |
| **MCP** | Optional/experimental | Required, canonical ToolRegistry |

---

## 4. Conflict and Gap Analysis

### 4.1 Root Architectural Gaps

**Gap 1: No Build mode.**

Current modes: plan, act, review. Build mode doesn't exist. `ReviewPolicy` exists as a first-class execution policy; it must be removed and replaced with `BuildPolicy`. Plan→Act auto-transition must become Plan→Build.

**Gap 2: No skill registry, only eager filesystem scanning.**

`SkillCollector` reads every skill file into context on every turn. No registry, no lazy loading, no permissioned loading. The `load_skill` tool exists but as a separate inconsistent path. `SkillCollector` must be replaced by `SkillIndexCollector` (summaries only) backed by `SkillRegistry`.

**Gap 3: Dual context assembly with no canonical merge point.**

System A (prompt builder) → system prompt string. System B (context collector) → context blocks. Called sequentially in `_provider.py`. No single `ContextPlanner`, no dependency graph between modules and collectors.

**Gap 4: No undo model.**

`rewind_turns()` is destructive JSONL rewriting. No undo stack, no inverse operation tracking, no PermissionPipeline routing for undo actions.

**Gap 5: No configured instruction array.**

`ProjectManifestCollector` reads AGENTS.md and CLAUDE.md only. No ordered array of instruction sources with priority.

**Gap 6: Session grants not isolated per session tree branch.**

`session_grant_store.clear()` invoked manually in each fork/switch handler. No automatic grant isolation per session branch.

**Gap 7: Permission static rules lack structured target matching and last-match-wins within the rule block.**

`PermissionRule` has `input_contains` (substring) and `input_prefix` (prefix). The static rule block uses "deny > ask > allow" priority instead of last-match-wins to choose which rule emits the StaticPolicy constraint.

**Gap 8: .env handling is all-or-nothing.**

`_is_sensitive_path()` blocks `.env` and `.env.*` unconditionally. No `.env.example` read exception.

**Gap 9: `external_directory` is not a first-class boundary concept.**

Current enforcement assumes single workspace root. No concept of explicitly allowed external directories.

**Gap 10: MCP is ghettoized in `experimental/`.**

MCP tools registered under `experimental` group. No per-server enabled flag, no timeout config, no sharing of the canonical ToolRegistry gating/policy infrastructure.

**Gap 11: No multiline input or @file fuzzy matching.**

REPL uses single-line prompt. `@file` requires exact relative path.

### 4.2 Duplicate/Multiple Paths to Consolidate

| Path | Consolidation Target | Reason |
|---|---|---|
| `load_skill` tool + SkillCollector | SkillRegistry + load_skill | One backend |
| `agent/compaction.py` | Remove | Harness path is active |
| `_decide_current()` dead code | Remove | Cutover complete |
| ReviewPolicy | Remove | Review is not a mode |
| `ctx.state.approved_plan` on ReplState | Move to ExecutionModeState | Mode state belongs in runtime |
| MCP as experimental group | Canonical ToolRegistry | MCP is not experimental |

### 4.3 What Breaks Current Assumptions

| MiMo Semantics | Current Assumption | Breakage |
|---|---|---|
| last-match-wins within static rule block | `PermissionPolicy.decide()` uses deny > ask > allow priority for static rules | `StaticPolicyEvaluator` must adopt last-match-wins for rule-block resolution; global `PermissionResolver` ordering remains `non_bypassable_deny > deny > ask > allow` |
| `.env.example` read allowed | `.env` and `.env.*` all denied | `_is_sensitive_path()` must add exception |
| external_directory | Single workspace root | Must add multi-root boundary resolver |
| Build as first-class mode | Plan times out to Act | Add Build, remove Review, replace ReviewPolicy |
| Session grants isolated per fork | Manual clear per handler | Auto-scope InMemoryGrantStore to session_id |
| Instruction array with priority | AGENTS.md only | Add configured_instructions to PromptRuntimeConfig |
| Skill loading is permissioned | Free tool call | Must go through PermissionPipeline |
| MCP is canonical | Experimental | Promote to core tool group, share registry infra |

---

## 5. Canonical Target Design

### 5.1 Skill Registry

```python
class SkillRegistry:
    """Sole backend for skill discovery, indexing, permission visibility, lazy loading."""
    _skills: dict[str, SkillDef]

    def discover(self, search_dirs: list[Path]) -> None:
        """Scan directories for SKILL.md files; cache metadata only."""

    def list_summaries(self) -> list[SkillSummary]:
        """Return (name, description, hidden) for all discovered skills."""

    def load(self, skill_name: str) -> SkillDef | None:
        """Lazy-load full skill content. This call is permissioned."""

@dataclass(frozen=True)
class SkillDef:
    name: str
    description: str
    hidden: bool = False
    file_path: Path
    content: str | None = None    # Loaded lazily

@dataclass(frozen=True)
class SkillSummary:
    name: str
    description: str
    hidden: bool
```

**Injection:** `SkillIndexCollector` (replaces `SkillCollector`):
- Calls `SkillRegistry.list_summaries()`
- Injects a compact `<available-skills>` block into USER_CONTEXT
- Does NOT load full skill bodies

**Loading:** `load_skill` tool:
- Calls `SkillRegistry.load(name)`
- Returns full skill content
- Goes through PermissionPipeline before execution

**Config:**
```yaml
skills:
  auto_trigger: true               # Match skills by name/relevance at context time
```

### 5.2 Execution Modes

```python
ExecutionMode = Literal["plan", "build", "act"]

class ExecutionModeState:
    current_mode: ExecutionMode = "act"
    plan_document: str = ""
    build_steps_completed: int = 0
    max_plan_turns: int = 8
    max_build_turns: int = 50

    def set_mode(self, mode: ExecutionMode, plan_document: str | None = None) -> None: ...
    def check_plan_timeout(self) -> bool: ...
    def record_build_step(self) -> None: ...
    def check_build_completion(self) -> bool: ...

class BuildPolicy:
    """Build mode: ordinary file mutations allowed; high-risk through PermissionPipeline."""
    ALLOWED_TOOLS = frozenset({
        "read_file", "write_file", "edit_file", "apply_patch",
        "glob_files", "grep_search", "find_files", "ls", "bash",
        "search_tools", "load_skill",
    })

    def filter_tools(self, tools): ...
    def check_call(self, call) -> PermissionDecision: ...

# Removal: delete ReviewPolicy, delete ReviewCommand, delete REVIEW_BASH_PREFIXES
# PlanPolicy remains unchanged
# ActPolicy.set_mode("plan") → set_mode("build") on timeout
```

**Transition sequence:**
1. `/plan` → mode=plan, tools restricted
2. Plan timeout OR `/build` → mode=build, `plan_document` set from last assistant message
3. Build timeout OR `/act` → mode=act, all tools available
4. `/act` → mode=act (from any mode)
5. Default on startup: mode=act

### 5.3 `/new` and `/clear`

```python
def cmd_clear(cmd: str, ctx: CommandContext) -> bool:
    ctx.store.clear()                     # New JSONL file
    ctx.store.current_grant_store().clear()  # Session grants ONLY
    # Permanent grants (FileGrantStore) NOT touched
    sync_agent_history(ctx.app, ctx.store)
    clear_terminal_display()
    print_startup_banner(ctx.app, ctx.project_root)
    return False
```

No `/reset-context` added. Session grants are scoped to session branch. On fork: parent grants optionally shadow-copied, child store isolated.

### 5.4 `/rewind` vs `/undo`

```python
class UndoStack:
    stack: list[UndoEntry]

    def push(self, inverse: UndoInverse) -> None:
        """Record an inverse operation BEFORE the original tool executes."""

    def undo(self, n: int = 1) -> list[UndoResult]:
        """Pop N entries, route each inverse through PermissionPipeline."""

@dataclass(frozen=True)
class UndoInverse:
    tool: str
    args: dict[str, object]
    inverse_tool: str                   # e.g., "write_file" for "edit_file"
    inverse_args: dict[str, object]     # Previous content, etc.
    description: str

# PermissionPipeline routing:
# undo() → for each inverse:
#     action = ActionExtractor().extract(inverse.inverse_tool, inverse.inverse_args)
#     verdict = evaluate_policy_constraints(action, ...)
#     if verdict.decision == "deny": skip (report)
#     if verdict.decision == "ask": require approval
#     if verdict.decision == "allow": execute
#
# /rewind:
#   Same as current: rewrites JSONL to remove trailing records
#   Only affects conversation history, NOT tool side effects
```

### 5.5 MCP in Canonical ToolRegistry

```python
# Config schema (mcp_config.json):
# {
#   "mcpServers": {
#     "filesystem": {
#       "command": "npx",
#       "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
#       "enabled": true,
#       "timeout": 30000,
#       "transport": "stdio"             # stdio | sse | streamable-http
#     }
#   }
# }

# Implementation:
# - mcp.py promoted from experimental/ to harness/
# - MCP tools registered via build_tool_registry(), not as a separate group
# - build_mcp_tools() called inside build_tool_registry()
# - Only servers with enabled=true are registered
# - All MCP tools enter through the same ToolRegistry/ToolGate path
# - Per-server config: enabled, timeout, transport merged into ToolSpec metadata
# - ToolSpec gains: source_metadata dict (server_name, server_transport, etc.)
# - Context budget: MCP tools contribute to the same tool-description budget
#   as built-in tools
```

### 5.6 Revised Permission Schema

```python
@dataclass(frozen=True)
class StaticPermission:
    tool: str                                           # "*" for global
    decision: Literal["allow", "ask", "deny"]
    target: str | None = None                           # Glob pattern for path/command
    target_type: Literal["path", "command", "mcp", "subagent", None] = None
    input_contains: str | None = None                   # Substring (kept for compatibility)
    input_prefix: str | None = None                     # Prefix (kept for compatibility)
    input_regex: str | None = None                      # Regex on serialized input

# New SecurityRuntimeConfig:
# security:
#   global_default: "ask"
#   rules:
#     - tool: "read_file"
#       target: ".env.example"
#       decision: "allow"
#     - tool: "bash"
#       input_contains: "curl"
#       decision: "ask"
#     - tool: "*"
#       decision: "deny"
#   external_directories:
#     - path: "/home/user/reference"
#       access: "read"

# StaticPolicyEvaluator: last-match-wins within the rule block
# The evaluator receives a list of StaticPermission rules.
# It iterates in rule-declaration order. The last rule whose tool/target/
# input_contains/input_prefix/regex matches the action emits ONE constraint.
# If no rule matches, the global_default applies.
#
# PermissionResolver: global ordering remains unchanged:
#   non_bypassable_deny > deny > ask > allow
#
# This means:
# - SystemSafety (.env deny) → emitted as non_bypassable_deny
# - Boundary (workspace escape) → emitted as non_bypassable_deny (write) or deny
# - RuntimeSafety (doom_loop) → emitted as deny
# - All of these resolve ABOVE any StaticPolicyEvaluator constraint,
#   regardless of rule order.
# - last-match-wins only determines which rule wins WITHIN the static block.
```

### 5.7 `.env.example` Exception

```python
def _is_sensitive_path(path: str, *, access: PermissionAccess = "read") -> bool:
    name = Path(path).name

    # .env.example is documentation — read allowed, write denied.
    if name == ".env.example":
        return access == "write"

    # .env and .env.* are sensitive — always denied.
    if name == ".env" or name.startswith(".env."):
        return True

    parts = tuple(part for part in path.split("/") if part)
    return any(part in StructuredBoundaryPolicyEvaluator.CREDENTIAL_PATH_PARTS for part in parts)
```

### 5.8 Configured Instruction Collector

```python
@dataclass
class InstructionSource:
    type: Literal["file", "inline"]
    path: Path | None = None
    content: str | None = None
    priority: ContextPriority = ContextPriority.CRITICAL

class InstructionCollector:
    """Replaces ProjectManifestCollector. Collects from configured array + fallback files."""
    def collect(self, input: ContextCollectionInput) -> list[ContextBlock]:
        sources = [
            *self._configured_instructions(),
            *self._manifest_files(),        # AGENTS.md / CLAUDE.md (fallback)
        ]
        return [self._resolve(source) for source in sources]

# Config:
# prompt:
#   instructions:
#     - type: file
#       path: AGENTS.md
#     - type: inline
#       content: "No external dependencies without approval."
```

---

## 6. Implementation Sequence

### Step 1: Remove Review as Canonical Mode

**Files**: `harness/agent_runtime/execution_modes.py`, `harness/config.py`, `cli/repl_commands.py`, `cli/repl.py`

**Changes**:
- Delete `ReviewPolicy`, `ReviewCommand`, `REVIEW_BASH_PREFIXES`
- Add `BuildPolicy` (allowed tools: file tools + bash + search + load_skill)
- `ExecutionMode = Literal["plan", "build", "act"]`
- `ExecutionModeState` default: `current_mode = "act"`, `plan_document = ""`
- Add `ExecutionModeState.plan_document: str` for plan→build handoff
- Plan timeout: `set_mode("build")` instead of `set_mode("act")`
- `plan_document` populated with last assistant message content on plan→build transition
- Remove `/review` command, add `/build` command
- `mode_notice()`: handle "build" mode
- `cmd_act()`: remove review→act path, plan→build default

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_execution_modes -v
# Must have zero references to ReviewPolicy
rg "ReviewPolicy|ReviewCommand|REVIEW_BASH_PREFIXES" src/xcode/
# Expected: no matches
```

### Step 2: Build SkillRegistry + SkillIndexCollector

**Files**: New `harness/skills_registry.py`, modify `agent/context_collector.py`, `harness/skills.py`, `coding_agent/registry.py`

**Changes**:
- `SkillDef`, `SkillSummary` dataclasses
- `SkillRegistry` with `discover()`, `list_summaries()`, `load()`
- Replace `SkillCollector` with `SkillIndexCollector`:
  - Calls `SkillRegistry.list_summaries()` → injects `<available-skills>` block
  - Does NOT read full skill files
- `load_skill` tool refactored:
  - Calls `SkillRegistry.load(name)`
  - Permissioned action (goes through PermissionPipeline)
  - Registered under group "skills"
- Remove eager file walking from old `SkillCollector`
- Remove `_walk_skill_files()`, `_build_skill_search_dirs()`, `SKILLS_MAX_BYTES`, etc.

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_skills -v
rg "SkillCollector|SKILLS_MAX_BYTES|_walk_skill_files" src/xcode/
# Expected: no matches (SkillIndexCollector replaces SkillCollector)
```

### Step 3: StaticPermission Rules + last-match-wins Inside StaticPolicyEvaluator

**Files**: `harness/observability/permissions.py`, `harness/observability/permission_model.py`, `harness/config.py`, `harness/assembly.py`

**Changes**:
- `PermissionRule` → `StaticPermission` (add `input_regex`, `target`, `target_type`)
- `StaticPolicyEvaluator` iterates rules in declaration order, last match wins to emit **one** StaticPolicy constraint
- `PermissionResolver` global ordering remains unchanged: `non_bypassable_deny > deny > ask > allow`
- `SecurityRuntimeConfig`: replace `deny_tools/ask_tools/allow_tools` lists with `rules: list[StaticPermission]` + `global_default: str`
- Remove `_decide_current()`, `_decide_cutover()`, `approval_cutover_enabled`, `shell_cutover_enabled` dead code
- `PermissionEngine` uses resolver path only
- Update `_permission_policy_from_security()` to emit `StaticPermission` rules

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_permission_model -v
uv run python -m unittest src.xcode.tests.test_permissions -v
rg "_decide_current|approval_cutover_enabled|shell_cutover_enabled" src/xcode/
# Expected: no matches
```

### Step 4: `.env.example` Read Exception + `external_directory` Boundary

**Files**: `harness/observability/permission_model.py`, `harness/config.py`

**Changes**:
- `ExternalDirectory` dataclass: `path`, `access`
- `BoundaryContext` gains `external_directories: tuple[ExternalDirectory, ...]`
- `StructuredBoundaryPolicyEvaluator._path_constraint()`:
  - Normalize target path
  - Classify into exactly one of: inside workspace root, inside approved external_directory, outside all approved roots
  - Only case 3 is workspace-escape → deny
  - Workspace and external_directory paths: check sensitive-path rules, then allow
- `_is_sensitive_path()`: `.env.example` read is allowed; `.env` and `.env.*` write always denied
- `SecurityRuntimeConfig` gains `external_directories: tuple[dict, ...]`

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_boundary -v
# Test: read_file .env.example → allow
# Test: write_file .env.example → deny
# Test: read_file /home/user/reference/doc.md → allow (if in external_directories)
```

### Step 5: Configured Instruction Collector

**Files**: `agent/context_collector.py` (replace `ProjectManifestCollector`), `harness/config.py`

**Changes**:
- `InstructionSource` dataclass
- `InstructionCollector` class replaces `ProjectManifestCollector`
- `PromptRuntimeConfig` gains `instructions: tuple[dict, ...]`
- Default: AGENTS.md only (backward compatible)
- Priority-based merge across sources
- Register in `ContextCollectorRegistry` replacing `ProjectManifestCollector`

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_context_collector -v
rg "ProjectManifestCollector" src/xcode/
# Expected: no matches
```

### Step 6: Session-Scoped Grant Isolation

**Files**: `harness/session.py`, `harness/observability/permission_model.py`, `cli/repl_commands.py`

**Changes**:
- `GrantStore` multi-session wrapper (keyed by `session_id`)
- `SessionStore` has `grant_store: GrantStore`, exposes `current_grant_store()`
- `fork_into()`: copies parent grants to child session store
- `clear()`: clears current session's grant store (NOT permanent store)
- Remove manual `ctx.session_grant_store.clear()` from `cmd_fork`, `cmd_branch`, `cmd_act`, `cmd_clear`
- `GrantStore.fork_grants(parent_id, child_id)` called by fork

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_session -v
# Test: fork creates isolated grant store
# Test: /clear clears session grants, preserves permanent
```

### Step 7: Undo Stack

**Files**: New `harness/undo.py`, `coding_agent/tools/file_handlers.py`, `cli/repl_commands.py`, `harness/session.py`

**Changes**:
- `UndoStack` class: `push()`, `undo(N)`, `can_undo()`
- `SessionStore` has `undo_stack: UndoStack` scoped to `current_path`
- File tool handlers push `UndoInverse` before mutation:
  - `write_file`: stores previous content (if file existed) or deletion marker
  - `edit_file`: stores original content
- `cmd_undo()`: pops N entries, routes each inverse through `PermissionPipeline`
- PermissionPipeline must route `UndoInverse.inverse_tool` and `inverse_args`
- `fork()` copies parent undo stack (immutable snapshot)

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_undo -v
# Test: write_file then /undo → content restored
# Test: /undo denied if PermissionPipeline blocks inverse
# Test: /undo not implemented as JSONL deletion
```

### Step 8: CLI Flags and `/continue`

**Files**: `main.py`, `cli/repl.py`, `cli/repl_commands.py`

**Changes**:
- CLI: `--continue` (auto-resume last session), `--session <id>`, `--fork <type>`
- `/continue` command: resume latest session + start new turn
- `/undo` slash command: delegates to `UndoStack.undo(N)`

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_repl -v
# Test: --continue resumes last session
# Test: --session <id> resumes specific session
```

### Step 9: MCP Canonicalization

**Files**: Move `experimental/mcp.py` → `harness/mcp.py`, move `experimental/mcp_client.py` → `harness/mcp_client.py`, modify `harness/assembly.py`, `harness/config.py`

**Changes**:
- Promote MCP from `experimental/` to `harness/`
- Per-server config: `enabled` (bool), `timeout` (int ms), `transport` (string)
- `build_mcp_tools()`: filter disabled servers, pass timeout to client
- MCP group moves from experimental set to a top-level enabled group
- ToolSpec gains optional `source_metadata` field (server_name, transport)
- MCP tools contribute to same context budget as built-in tools

**Verification:**
```bash
uv run python -m unittest src.xcode.tests.test_mcp -v
# No references to MCP in experimental/ (except removal commit)
```

### Step 10: Remove All Dead Paths

**Files**: `agent/compaction.py` (delete), `harness/observability/permissions.py` (clean), `harness/skills.py` (clean)

**Changes**:
- Delete `agent/compaction.py`
- Remove stale dead code references in permissions
- Remove old `load_skill` implementation path if it bypasses `SkillRegistry`
- Keep the `load_skill` tool name and `ToolSpec`; handler delegates to `SkillRegistry.load()`
- Confirm no imports break

**Verification:**
```bash
uv run python -m compileall src/xcode
uv run python -m unittest discover src/xcode/tests -v
```

---

## 7. Tests and Verification

### 7.1 Permission Resolver Tests

- `test_last_match_wins_within_static_block`: two rules in StaticPolicyEvaluator, last match emits the constraint
- `test_non_bypassable_deny_final`: SystemSafety non-bypassable deny beats static allow regardless of rule order
- `test_boundary_deny_beats_static_allow`: workspace-escape deny beats later static allow rule
- `test_global_asterisk_default`: `*` rule matched when no specific tool rule
- `test_input_regex_matches`: `input_regex` on `StaticPermission`
- `test_input_pattern_with_target`: path glob matching in static rules
- `test_external_directory_match`: path under external_directory allowed
- `test_env_example_read_allowed`: read_file .env.example → allow
- `test_env_write_always_denied`: write_file .env → deny (SystemSafety non_bypassable_deny beats any static allow)
- `test_env_example_read_static_allow`: read_file .env.example allowed via static rule (boundary says allow)

### 7.2 Path/Boundary Tests

- `.env.example` read allowed, `.env` write denied
- External directory read/write access matching
- External directory not allowing workspace escape

### 7.3 Plan/Build Mode Tests

- `test_plan_to_build_transition`: plan timeout → build (not act)
- `test_build_policy_filter`: build tools exclude network tools
- `test_build_policy_check_call`: write_file allowed, curl denied
- `test_plan_document_handoff`: plan_document carried to build
- `test_build_timeout`: build max turns → act
- `test_default_mode_is_act`: new session starts in act

### 7.4. `.env` Safety Tests

- `test_env_example_read`: boundary tests
- `test_env_write_denied`: write always denied for all .env files

### 7.5 Doom Loop Tests

- `test_repeated_tool_watchdog_triggers`: N+1 same calls blocked
- `test_watchdog_resets_on_different_tool`: different tool resets counter

### 7.6 Session Fork Tests

- `test_fork_grants_isolated`: child grants independent from parent
- `test_fork_grants_copied`: parent grants copied to child on fork
- `test_clear_keeps_permanent_grants`: /clear clears only session grants
- `test_clear_clears_session_grants`: /new clears session grants
- `test_env_example_read_boundary_rule`: read_file .env.example passes boundary (not sensitive for read)

### 7.7 Skill Discovery/Lazy-Load Tests

- `test_registry_discovers_skills`: SKILL.md files found, metadata cached
- `test_skill_lazy_load`: load() reads file content
- `test_skill_summary_injection`: SkillIndexCollector injects summaries, not full bodies
- `test_skill_loading_is_permissioned`: load_skill goes through PermissionPipeline
- `test_skill_hidden_not_listed`: hidden=true → excluded from summaries

### 7.8 Instruction Collector Tests

- `test_configured_instructions_loaded`: inline instructions appear as context block
- `test_instruction_order`: priority ordering preserved
- `test_agents_md_fallback`: no configured instructions → AGENTS.md loaded

### 7.9 Command/REPL Tests

- `test_undo_command`: `/undo` reverses last write_file
- `test_continue_command`: `/continue` resumes latest session, starts turn
- `test_build_command`: `/build` sets mode to build
- `test_cli_continue_flag`: --continue resumes last session
- `test_cli_session_flag`: --session <id> resumes specific session
- `test_cli_fork_flag`: --fork forks from latest session
- `test_no_reset_context`: /reset-context does not exist
- `test_new_preserves_permanent_grants`: /new clears session grants only

### 7.10 MCP Tests

- `test_mcp_per_server_enabled`: disabled server not registered
- `test_mcp_timeout_config`: per-server timeout applied
- `test_mcp_registered_through_tool_registry`: MCP tools share ToolRegistry path
- `test_mcp_not_experimental`: MCP moved out of experimental/

### 7.11 Regression Commands

```bash
# Full suite
uv run python -m unittest discover src/xcode/tests -v

# Targeted
uv run python -m unittest src.xcode.tests.test_permission_model -v
uv run python -m unittest src.xcode.tests.test_execution_modes -v
uv run python -m unittest src.xcode.tests.test_session -v
uv run python -m unittest src.xcode.tests.test_context_collector -v

# Lint + type check
uv run ruff check src/xcode --fix && uv run ruff format src/xcode
uv run pyright src/xcode

# No stale references to removed APIs
rg "ReviewPolicy|_decide_current|approval_cutover|shell_cutover|SkillCollector|ProjectManifestCollector" src/xcode/
```

---

## 8. Safe-to-Commit Criteria

### Per-Step Criteria

```bash
# 1. Static analysis passes
uv run ruff check <modified-files> --fix
uv run ruff format <modified-files>
uv run pyright <modified-files>

# 2. Targeted tests pass
uv run python -m unittest <targeted-test-module> -v

# 3. No stale references to removed APIs
rg "<removed-function-name>" src/xcode/  # Must be zero

# 4. Config backward compatibility
uv run python -c "from xcode.harness.config import discover_runtime_config; print('OK')"

# 5. Full test suite
uv run python -m unittest discover src/xcode/tests -v
```

### Documentation Updates

| Doc | After Step | Change |
|---|---|---|
| `docs/evaluation-guide.md` | All | Update validation commands |
| `docs/code-organization.md` | 2, 5, 9 | Add skills_registry.py, update context_collector, MCP moved |
| `CONFIG.md` | 1, 3, 4 | Modes (remove review), permission schema, external_directory |
| `AGENTS.md` | All | Update agent behavior references |
| `README.md` | 8 | Add CLI flags |
| `experimental/README.md` | 9 | Remove MCP section |

### Stale-Doc Checks

```bash
grep -r "TODO\|FIXME\|cutover\|legacy\|deprecated" src/xcode/<modified-files>
git diff --check -- docs/
```
