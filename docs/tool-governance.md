# Tool Governance: Registration, Exposure, Invocation, Delegation, Execution

## Status

Architecture decision. Not yet implemented.

## Problem

`ToolSpec.group` currently controls unrelated semantics: UI grouping, `/tool` visibility,
completion, primary-agent visibility, subagent exclusion, Plan/Build filtering, and parts
of permission behavior. This conflation causes several concrete failures:

- MCP tools leak into `/tool` tab completion and direct invocation even when they are
  developer-diagnostics tools (`mcp__everything__echo`).
- `/tool` resolves `tool_map[name]` and executes without a separate user-invocability
  check.
- Primary-agent tool availability and subagent inheritance are not derived from a coherent
  policy; subagents exclude all `group == "mcp"` tools via a hard-coded shortcut.
- Plan/Build mode maintain manually curated allowlists that duplicate the concept of
  "which tools belong in planning mode" at the wrong level of abstraction.
- MCP server metadata (name, description, schema) can influence the tool's visibility and
  discoverability in the host, which should be governed by host policy alone.
- `capability == "mcp"` is used as a permission axis, conflating transport/origin with
  the actual action kind (read, write, execute, network).

## Terminology

| Term | Definition |
|------|------------|
| **Registration** | The act of making a tool known to the system by inserting it into the runtime registry. Registration means the system is aware of the tool. It does not imply anything about visibility, invocability, or delegation. |
| **Exposure** | Whether and how a tool appears in the `/tool` command, tab completion, and help text. A tool may be registered but have no UI presence. |
| **User Invocability** | Whether a human may execute the tool directly via `/tool`. |
| **Primary-Agent Invocability** | Whether the coding agent (LLM) sees the tool in its tool schema and is allowed to call it during normal operation. |
| **Subagent Delegation** | Whether a child agent may inherit the tool. Subagent toolset is always a subset of the parent's **delegable capability ceiling**, not the parent's runtime-resolved permission verdict. |
| **Capability Envelope** | The set of tools a given agent is eligible to use, before any invocation-time permission check. Determined by `primary_agent_invocable`, `subagent_policy`, mode constraints, and task-specific grants. Does **not** include runtime four-axis authorization. |
| **Runtime Authorization** | The four-axis permission decision (Mode / Capability / Boundary / Approval) made at invocation time, against the concrete action, target path, and network address of the specific call. This is the existing permission engine and is not redesigned here. |
| **Host-Owned Policy** | A governance descriptor attached to or resolvable for each `RegisteredTool`. It is controlled by the host (Xcode), not by MCP server metadata. |

## Architecture

### Principle: Six Separate Concerns

```text
Registration     → system recognition
Exposure         → UI / discovery
User Invocability → /tool execution
Agent Invocability → primary-agent capability envelope
Subagent Policy  → delegation eligibility ceiling
Runtime Auth     → permission verdict at call time against concrete action/target
```

Each concern has its own governing mechanism. `ToolSpec.group` is retired from all
concerns except UI grouping (display sections in `/tool list`).

### `RegisteredTool` Wraps `ToolSpec` + `ToolSurfacePolicy` + `ToolActionProfile`

`ToolSpec` remains the canonical technical execution definition:

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str              # canonical internal identifier, e.g. "mcp://everything/echo"
    description: str       # LLM-facing description
    input_hint: str        # CLI usage hint
    handler: ActionHandler # sync callable
    schema: dict | None    # JSON Schema for LLM parameters
    read_only: bool
    concurrency_safe: bool
    group: str             # UI grouping only (display sections)
    execution_mode: ToolExecutionMode | None
    counts_as_progress: bool | None
    examples: list[dict]
    prompt_snippet: str | None
    prompt_guidelines: tuple[str, ...]
    builtin: dict | None   # provider-specific metadata
```

Do not introduce a parallel top-level object that duplicates identity, schema, or handler.
All governance metadata lives on the wrapper, not on `ToolSpec`.

```python
@dataclass(frozen=True)
class ToolSurfacePolicy:
    exposure: Literal["root", "grouped", "hidden"]
    user_invocable: bool
    primary_agent_invocable: bool
    subagent_policy: Literal["deny", "explicit_grant", "policy_derived"]

@dataclass(frozen=True)
class ToolOrigin:
    kind: Literal["core", "mcp", "skill"]
    source: str | None       # MCP server name, e.g. "github"

@dataclass(frozen=True)
class ToolActionProfile:
    """Host-controlled action facts for the four-axis permission engine.

    This is not a ToolSurfacePolicy field. It is a separate technical fact
    produced by the host adapter layer, not derived from MCP server metadata
    or tool description text. A tool without a host-provided ToolActionProfile
    must not enter any capability envelope.
    """
    capability: Capability         # read, write, execute, network, credentialed-action
    target_resolver: TargetResolver
    side_effecting: bool
    credentialed: bool

@dataclass(frozen=True)
class ToolSelector:
    selector: str                  # "everything.echo" or "shell.run"

@dataclass(frozen=True)
class RegisteredTool:
    canonical_id: str              # "mcp://everything/echo"
    public_selector: ToolSelector  # "everything.echo" — what users type in /tool
    spec: ToolSpec                 # spec.name retains internal routing name during migration
    surface_policy: ToolSurfacePolicy
    origin: ToolOrigin
    action_profile: ToolActionProfile | None
```

The registry stores `RegisteredTool`, not bare `ToolSpec`. This guarantees that
governance metadata is never accidentally absent or decoupled from the spec.

`ToolActionProfile` is **nullable**. A tool with `action_profile is None`:
- can still be registered (the system knows about it)
- must not enter any primary-agent or subagent capability envelope
- must not be executable via the public `/tool <selector>` path
- is invisible in `/tool` listing and completion unless `exposure` explicitly allows otherwise

This enables the closed default for unknown MCP tools: they exist in the registry
and cache but are fully locked down until the host provides an action profile.

`ToolOrigin.kind` and `ToolOrigin.source` carry transport/source information for
audit and display grouping. They must **not** participate in permission decisions.
The four-axis engine acts on `ToolActionProfile.capability` (read, write, execute,
network, credentialed-action), not on whether the tool arrived via MCP or core.

A tool without a host-provided `ToolActionProfile` must not enter any capability
envelope. This prevents untrusted MCP metadata from determining permission-relevant
facts.

| `ToolSurfacePolicy` Field | Effect |
|---------------------------|--------|
| `exposure` | Controls `/tool` root listing, tab completion, help text |
| `user_invocable` | Controls whether `/tool <name>` executes |
| `primary_agent_invocable` | Controls whether the tool enters the primary-agent capability envelope |
| `subagent_policy` | Controls delegation eligibility ceiling |

Policy is owned by Xcode. MCP server metadata (name, description, inputSchema,
annotations, outputSchema) must not affect `ToolSurfacePolicy` or `ToolActionProfile`.
A server's self-description cannot expand its visibility, invocability, delegation
scope, or action profile.

### Before / After Data Flow

#### Before

```text
mcp_config.json
  → build_mcp_tools()
    → ToolSpec(group="mcp", name="mcp__everything__echo")
      → registry_state.replace_group("mcp", ...)
        ├── /tool:       tool_map[tool.name]       ← no visibility filter
        ├── completion:  tool_names = [t.name...]  ← no visibility filter
        ├── Act mode:    all tools                  ← no invocability filter
        ├── Plan/Build:  manual allowlist           ← duplicates policy
        ├── subagent:    group != "mcp"             ← hard-coded shortcut
        └── permission:  capability="mcp"           ← origin used as capability axis
```

#### After

```text
mcp_config.json
  → build_mcp_tools()
    → RegisteredTool(
        canonical_id="mcp://everything/echo",
        public_selector=ToolSelector("everything.echo"),
        spec=ToolSpec(name="mcp__everything__echo", group="mcp"),
        surface_policy=ToolSurfacePolicy(exposure="grouped", user_invocable=true,
                                         primary_agent_invocable=true,
                                         subagent_policy="policy_derived"),
        origin=ToolOrigin(kind="mcp", source="everything"),
        action_profile=ToolActionProfile(capability="execute", ...))
      → registry_state.replace_group("mcp", ...)
        ├── /tool root:       exposure == "root" && user_invocable && action_profile
        ├── /tool mcp:        exposure == "grouped" && user_invocable && action_profile
        ├── completion:       exposure != "hidden"
        ├── primary agent:    primary_agent_invocable && action_profile_present (capability envelope only)
        ├── Plan/Build/Act:   same capability envelope model (no hand-maintained lists)
        ├── subagent:         parent_capability_envelope ∩ subagent_policy ∩ delegation_context
        └── permission:       ToolActionProfile-driven (read/write/execute/network/...)
                              never origin-derived ("mcp" is not a capability)
```

### Naming Convention

| Scope | Format | Example |
|-------|--------|---------|
| Public selector | `source.tool` | `everything.echo` |
| Canonical internal id | `kind://source/tool` | `mcp://everything/echo` |
| Provider-facing function name | Adapter-generated, provider-valid | varies by LLM provider |
| Handler routing key (stable) | Internal, not in public paths | `mcp__everything__echo` |

- The **public selector** is what users type in `/tool everything.echo` and what appears
  in help text. It uses `source.tool` without a `kind` prefix. `kind` is used only in
  `ToolOrigin` (for audit) and in navigation (`/tool mcp` → `/tool mcp everything`).
  The dot namespace is reserved for MCP selectors and must not collide with core tool
  names. Uniqueness is enforced at registration time (step 1): no two
  `RegisteredTool` may share the same `public_selector`. MCP server
  collision (`everything.echo` vs `github.echo`) is excluded by the
  `source.` prefix — collision only occurs within the same source.
  Core tool selectors (`shell.run`, `file.read`) occupy a flat
  namespace that must not overlap with MCP `source.` names.
- The **canonical internal id** is the stable referent for collision detection, caching,
  and audit records.
- The **provider-facing function name** is generated by the adapter layer (step 6). It
  must satisfy the LLM provider's identifier rules (some providers forbid dots). The
  document must not assume dots work in all provider APIs.
- Raw `mcp__...` names are removed from all public interfaces: `/tool`, completion,
  help, LLM tool definitions, and user-facing documentation. The `ToolSpec.name` field
  retains the internal routing name during migration so that the handler binding is not
  broken by step 1.

### `/tool` Behavior

| Scenario | Behavior |
|----------|----------|
| `/tool` (bare) | Lists `exposure == "root"` tools grouped by `group` |
| `/tool mcp` | Lists MCP servers that have `exposure != "hidden"` tools |
| `/tool mcp <server>` | Lists tools for that server with `exposure != "hidden"` |
| Tab completion at `/tool ` | Completes `exposure != "hidden" && user_invocable && action_profile is not None` tools |
| Tab completion at `/tool mcp ` | Completes server or tool names |
| `/tool <selector>` | Executes only if `exposure != "hidden" && user_invocable == true && action_profile is not None` |
| Raw internal names (`mcp__...`) | Rejected — not recognized as valid selectors |

**Selector resolution requires three conditions**: `exposure != "hidden"`,
`user_invocable == true`, and `action_profile is not None`. A tool missing any
of these cannot be invoked via any user-facing `/tool` selector. This does not
preclude a restricted internal diagnostics path (separate from `/tool`) for hidden
or unclassified tools.

`/tool` listing, completion, help, and direct execution **must be changed together**
as one atomic step. There must be no intermediate state where a tool is hidden from
UI but still executable by users who know its old name.

### Default Policy Rules

Server-level template semantics are defined here at the semantic level.
The concrete schema format is left to the implementation phase (see
Non-Goals).

"Explicitly approved" means the tool is named in host configuration — either by
exact `server.tool` match or by a server-level template that names specific tools.
A server being configured does not approve every tool it exposes. An unmatched
tool from a configured server is not "known" and falls to the closed default.

```text
unmatched server.tool
  → closed host default (hidden, false, false, deny)

tool matched by server-level template
  → server-level defaults for all fields

tool matched by exact server.tool override
  → most-specific wins per field
```

A server-level template applies only to the tools explicitly listed under that
server in configuration. It does not grant default visibility or invocability to
tools added by a future server update. If a server adds a new tool, that tool is
unmatched and gets the closed default until the host configuration is updated.

| Tool Category | `exposure` | `user_invocable` | `primary_agent_invocable` | `subagent_policy` | `action_profile` |
|---|---|---|---|---|---|---|
| Core tools (shell, file, git) | `root` | `true` | `true` | `policy_derived` | `not None` |
| Session tools (todo, progress) | `root` | `true` | `true` | `deny` | `not None` |
| Explicitly approved `server.tool` | Per-tool config | Per-tool config | Per-tool config | Per-tool config | As configured |
| MCP diagnostics (`everything.echo`, `get-env`) | `grouped` | `true` | `false` | `deny` | `not None` |
| Unmatched MCP tool (not in host policy) | `hidden` | `false` | `false` | `deny` | `None` |
| Hidden maintenance (`mcp_tool_search`, deferred fetch) | `hidden` | `false` | `false` | `deny` | `None` |

### Agent Behavior

**Primary agent** capability envelope is assembled from tools where:

```text
primary_agent_invocable == true
  AND action_profile is present
  AND mode_eligibility (Plan / Build / Act)
  AND current capability envelope constraints
```

A tool without a host-provided `ToolActionProfile` cannot enter the envelope,
regardless of `primary_agent_invocable`. This prevents unclassified MCP tools
from reaching the agent.

Plan/Build/Act no longer maintain hand-curated allowlists. Mode constraints are
derived from `primary_agent_invocable` and the existing `ExecutionMode` rules, but
they use the same policy source. The common Plan → Build → Act widening is expressed
as mode-level overrides on the capability envelope, not as separate tool lists.

The capability envelope does **not** include runtime authorization (Boundary, Approval).
Those are resolved at invocation time against the concrete action, target path, and
network address. A tool in the envelope means "the agent may attempt to use this tool",
not "the agent is pre-approved to use this tool".

MCP tools remain typed tools with their schemas. There is no generic `call_mcp(server, tool, args)`
in the primary-agent toolset. A generic call tool may exist as a hidden diagnostics tool
for manual troubleshooting, but it must not be the default invocation path.

**Subagent** capability envelope is derived as:

```text
subagent_envelope =
    parent_capability_envelope
    ∩ subagent_policy_eligibility
    ∩ task_specific_grants
    ∩ delegation_context_constraints
```

- `parent_capability_envelope` is the set of tools the parent agent is eligible to call
  (the capability envelope, **not** the parent's runtime-resolved verdict).
- `subagent_policy == "deny"` excludes the tool unconditionally.
- `subagent_policy == "explicit_grant"` requires an explicit grant for this subagent
  invocation.
- `subagent_policy == "policy_derived"` defers to the parent agent's
  capability envelope **at the time of subagent creation**. The parent's
  envelope is snapshotted, not dynamically inherited. If the parent agent
  transitions from Plan to Act mode after spawning a subagent, the
  subagent's envelope does not expand. This prevents the subagent from
  gaining capabilities that were not evaluated during its creation.
- `delegation_context_constraints` is limited to static or session-level ceilings:
  current mode, task scope, explicit tool grants, subagent type. It must **not**
  include path targets, network addresses, approval state, or any prior invocation's
  permission verdict.
- `task_specific_grants` is injected statically at subagent creation time by
  the parent agent. It is a fixed set of tool selectors (or tool categories)
  determined before the subagent starts. It must **not** be extended during
  the subagent's runtime, and must not carry approval state or invocation
  history from prior operations.
- `group == "mcp"` is not used as an exclusion mechanism.
- Side-effect tools, credentialed tools, and write-capable tools default to
  `explicit_grant` unless a prior permission policy already governs them.

The subagent must independently undergo runtime authorization for each tool call.
A parent's prior approval does not carry over. A parent's one-time approval for a
specific operation does not become the subagent's permanent authorization.

### Runtime

Core and MCP tools converge on a single execution gateway. The gateway is responsible
for:

- Schema validation
- Permission resolution (existing four-axis engine, driven by `ToolActionProfile`)
- Approval (existing callback mechanism)
- Auditing (existing after-hook)
- Timeout / cancellation
- Output size capping
- Normalized error classification

There is no parallel MCP-only permission system. The four-axis engine acts on
`ToolActionProfile.capability` — read, write, execute, network, credentialed-action —
not on whether the tool arrived via MCP or core. `ToolOrigin` is available for
audit and display but must not be a permission axis.

`capability == "mcp"` in the permission model (currently at `permission_model.py:993-1008`)
is removed. MCP is a transport and origin, not a user action capability. The permission
engine derives capability from `ToolActionProfile.capability`, which is host-controlled
and never derived from MCP metadata or tool description text.

MCP output is treated as untrusted tool output. It must not carry authority to alter
policy, configuration, or instructions.

`ToolActionProfile` is the single source of truth for the four-axis permission
engine's capability axis. The permission engine must not read `ToolSpec.description`,
`ToolSpec.builtin["mcp_metadata"]`, MCP annotations, or MCP outputSchema to determine
capability, target resolver, or side-effect classification.

### Migration: Retire `group == "mcp"` Checks

Every site that branches on `tool.group == "mcp"` must be replaced:

| Current site | Replacement |
|---|---|
| `assembly.py:293-298` — subagent exclude `group != "mcp"` | `subagent_policy` filtering |
| `execution_modes.py:75-86` — Plan/Build allowlist | `primary_agent_invocable` + mode constraints |
| `execution_modes.py:95` — Act passes all tools | `primary_agent_invocable` filter |
| `repl_tools.py:72-75` — `/tool list` `[mcp]` suffix | `origin.kind` / `origin.source` replaces string-based `[mcp]` detection. Display format changes from `"tool_name [mcp]"` to `"tool_name (mcp/everything)"`. |
| `repl_tools.py:98` — `/tool` lookup | `exposure != "hidden" && user_invocable && action_profile is not None` |
| `completion.py:80` — tool names from registry | `exposure != "hidden"` filter |
| `permission_model.py:993-1008` — `mcp__` prefix match | Remove. Derive capability from `ToolActionProfile`, not name prefix or `group`. |

Build-time collision detection (`mcp/tools.py:471-499`) stays on the canonical internal id;
it is a transport-layer concern, not a policy concern.

## Test Matrix

| # | Scenario | Expected |
|---|----------|----------|
| 1 | `mcp://everything/echo` is registered | Tool exists in registry as `RegisteredTool` |
| 2 | `/tool` root listing | Does not appear (exposure != "root") |
| 3 | Tab completion at `/tool ` | Does not complete old `mcp__...` names |
| 4 | `/tool mcp everything` | Lists tools for "everything" server |
| 5 | `/tool everything.echo` | Executes (exposure != "hidden" && user_invocable == true && action_profile is not None) |
| 6 | User knows old name `mcp__everything__echo` | Rejected — not a valid selector |
| 7 | Hidden MCP tool on `/tool` root | Does not appear |
| 8 | Hidden MCP tool on tab completion | Does not complete |
| 9 | Hidden MCP tool via `/tool <any selector>` | Rejected — selector resolution requires `exposure != "hidden"` |
| 10 | Primary-agent envelope includes a tool | Requires `primary_agent_invocable == true && mode_eligibility && action_profile present` |
| 11 | Primary-agent envelope excludes tool with `primary_agent_invocable == false` | Excluded regardless of `exposure` |
| 12 | Primary-agent envelope excludes tool without `ToolActionProfile` | Excluded even if `primary_agent_invocable == true` |
| 13 | New unknown MCP tool | `hidden`, `user_invocable=false`, `primary_agent_invocable=false`, subagent deny |
| 14 | Subagent can call a tool | Only if parent envelope has it AND `subagent_policy` allows |
| 15 | Subagent cannot call tool with `subagent_policy == "deny"` | Excluded unconditionally |
| 16 | Subagent does not inherit parent's prior approval | Each call independently authorized |
| 17 | MCP server changes metadata (name/desc/inputSchema) | No change to exposure, invocability, subagent policy, or action profile |
| 18 | Core and MCP call both go through gateway | Same execution pipeline, same audit shapes |
| 19 | MCP output size exceeds cap | Truncated with marker, same as core |
| 20 | MCP tool times out | Cancellation + error, same as core |
| 21 | Permission engine reads `capability == "mcp"` or `mcp__` prefix | Rejected. Capability must be from `ToolActionProfile`. |
| 22 | `group` is used as permission or delegation gate | Rejected. Only `ToolSurfacePolicy` governs these. |
| 23 | Server-level template grants visibility to a tool not named in config | Rejected. Template applies only to explicitly listed tools. |
| 24 | `user_invocable == true` but `action_profile is None` | Rejected by public `/tool` selector — all three conditions required |
| 25 | Subagent created in Plan mode, parent later enters Act | Subagent envelope does not expand — snapshot at creation |

## Non-Goals

- This phase does not redesign the four-axis permission model (Mode / Capability /
  Boundary / Approval). It does remove the `capability == "mcp"` shortcut and replaces
  it with `ToolActionProfile.capability`.
- This phase does not add arbitrary MCP compatibility layers or protocol extensions.
- Raw `mcp__...` names are removed from all public interfaces (see Naming Convention).
- This phase does not introduce risk levels, trust tiers, or an MCP-specific safety
  system parallel to the core execution pipeline.
- This phase does not implement a generic `call_mcp` fallback as the primary agent
  invocation path (a diagnostics-only hidden tool is acceptable).
- This phase does not design the configuration file schema for per-tool policy overrides
  or server-level templates. However, the Default Policy Rules section defines the
  **semantic minimum** that the schema must satisfy: server-level templates apply only
  to explicitly listed tools, and unmatched tools fall to the closed default. The
  concrete schema file format is left to the implementation phase.

## Implementation Sequence

1. **Data structures** — introduce `RegisteredTool`, `ToolSurfacePolicy`, `ToolOrigin`,
   `ToolActionProfile`, canonical internal id scheme, and closed defaults. Registry
   stores `RegisteredTool`. No behavioral change.

2. **Registry filter API and selector resolver** — functions answering "which tools
   are visible to `/tool`", "which are user-invocable", "which go to primary agent",
   "which go to subagent". Selector resolver maps public selector
   (e.g. `everything.echo`) to canonical id. No behavioral change. Full test coverage.

3. **`/tool` list, completion, help, and direct execution** — all switched to query
   `RegisteredTool.surface_policy` atomically via the selector resolver. Hiding a tool
   from UI also blocks direct invocation. Adds `/tool mcp <server>` namespace
   navigation. Old `mcp__...` names rejected as selectors.

4. **Primary agent Plan / Build / Act assembly** — switched from hand-maintained
   allowlists to `primary_agent_invocable` + mode constraints + `ToolActionProfile`
   presence check.

5. **Subagent delegation** — replaced `group != "mcp"` with `subagent_policy` filtering
   and `delegation_context_constraints`. Subagent independently authorized per call.

6. **LLM provider-facing function-name adapter** — generates provider-valid identifiers
   from canonical internal ids. Does not assume dots work in all APIs.

7. **Permission model** — remove `capability == "mcp"` and `mcp__` prefix matching.
   Capability derived from `ToolActionProfile.capability`.

8. **Final audit** — remove all `group == "mcp"` branches. Validate the full test
   matrix. Confirm core and MCP share the same execution gateway.

Timebox step 3 so that the exposure change and the execution gate change ship together.
Do not land them in separate releases. Enforce this with a feature flag
(`use_registered_tool_governance`) that gates both the UI layer and the execution
gateway. The flag is removed after step 3 validation.
