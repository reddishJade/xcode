# Long-horizon memory architecture

## Purpose

Memory exists to let one logical coding task survive bounded model windows and
process restarts. It is not a knowledge-management product and does not assign
behavioral scores to individual notes.

## Storage layers

1. **Transcript** is the lossless history. Session JSONL keeps user messages,
   assistant messages, and tool events.
2. **Session surface** is the disposable model working set. Each rollover
   appends a typed replacement event without rewriting older entries.
3. **Working note** is project-root `NOTE.md`. It contains the current goal,
   confirmed decisions, verification status, unresolved issues, and next action.
4. **Project memory** is `MEMORY.md`. It contains only durable project rules,
   architecture decisions, and verified cross-session facts.
5. **User memory** is `~/.xcode/memory/MEMORY.md`. It contains durable
   cross-project preferences.

The layers have different jobs. Current progress and next actions belong in
`NOTE.md`, never in project or user memory. The transcript remains the source
of truth when a note needs evidence.

## Runtime flow

### Normal turns

The system prompt tells the agent where memory lives and when to use it. It does
not automatically inject search results on every turn. The agent calls the
read-only `search_memory` tool when prior project knowledge may matter.

### Rollover

Xcode uses the provider profile's `context_window` override when present;
otherwise it reads the active model's registered context window. Automatic
rollover begins at 95% or at the output-reserve boundary, whichever comes
first. The old window is closed without a summary. Startup context, activated
skills, and the active user turn form the new working set. The typed event
records the replacement, source entry IDs, a monotonic generation, and a stable
fingerprint.

### Resume

Xcode restores:

```text
latest durable context-window replacement
+ transcript entries appended after the replacement
+ NOTE.md working state
+ budgeted project and user memory
```

The latest final event already contains the structured coding run state. Resume
restores its execution mode and todo list after rebuilding message history.
Older exact evidence remains available through `history` list/search/read/around.

## Invariants

- Markdown is the source of truth for durable memory.
- Surface replacements are isolated by session branch.
- Resume never drops the verbatim transcript tail.
- Memory search is deterministic BM25 over project and user files.
- Writes are explicit and atomically replace the target file.
- Existing governance metadata is ignored; retired legacy records stay excluded.

## Explicit non-goals

Do not add these without evidence from real long-running task failures:

- embeddings or vector databases;
- per-record utility, adoption, success, or failure counters;
- confidence and validity state machines;
- automatic promotion based on inferred model behavior;
- multi-factor reranking;
- online explain/metrics platforms for a local Markdown search;
- automatic retrieval injection on every user turn.

## Stop point

The implemented surface/history cycle is the product boundary:

- rollover never summarizes the previous window;
- malformed or tool-unbalanced replacements are rejected;
- `history list_windows/search/read/around` retrieves exact details older than
  the current working set.

Early background extraction and automatic project-memory promotion are not
planned. They require evidence from real long-running task failures and must
not expand the per-record Memory model.
