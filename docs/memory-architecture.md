# Long-horizon memory architecture

## Purpose

Memory exists to let one logical coding task survive bounded model windows and
process restarts. It is not a knowledge-management product and does not assign
behavioral scores to individual notes.

## Storage layers

1. **Transcript** is the lossless history. Session JSONL keeps user messages,
   assistant messages, and tool events.
2. **Session surface** is the current task state. Each compact cycle appends a
   typed replacement event to the session transcript.
3. **Project memory** is `MEMORY.md`. It contains only durable project rules,
   architecture decisions, and verified cross-session facts.
4. **User memory** is `~/.xcode/memory/MEMORY.md`. It contains durable
   cross-project preferences.

The layers have different jobs. Current progress and next actions belong in the
session surface, never in project or user memory.

## Runtime flow

### Normal turns

The system prompt tells the agent where memory lives and when to use it. It does
not automatically inject search results on every turn. The agent calls the
read-only `search_memory` tool when prior project knowledge may matter.

### Compact

For models with a known context window, Xcode starts a new compact cycle at
roughly 70% utilization instead of waiting for the window reserve boundary.
The compactor keeps a verbatim recent tail and writes its full structured
summary as a new session surface. The event records the complete replacement,
the source entry IDs, a monotonic generation, and a stable fingerprint.

### Resume

Xcode restores:

```text
latest durable surface replacement
+ transcript entries appended after the replacement
+ budgeted project and user memory
```

The latest final event already contains the structured coding run state. Resume
restores its execution mode and todo list after rebuilding message history, so
unfinished work does not depend on the compacted summary mentioning every todo.

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

- compact updates the previous structured state instead of repeatedly
  summarizing it as ordinary conversation;
- malformed or tool-unbalanced replacements are rejected;
- `history search/around` retrieves exact details older than the rebuild
  boundary.

Early background extraction and automatic project-memory promotion are not
planned. They require evidence from real long-running task failures and must
not expand the per-record Memory model.
