# Long-horizon memory architecture

## Purpose

Memory exists to let one logical coding task survive bounded model windows and
process restarts. It is not a knowledge-management product and does not assign
behavioral scores to individual notes.

## Storage layers

1. **Transcript** is the lossless history. Session JSONL keeps user messages,
   assistant messages, and tool events.
2. **Session checkpoint** is the current task state. Each compact cycle writes
   `.xcode/checkpoints/<session-id>/checkpoint.md`.
3. **Project memory** is `MEMORY.md`. It contains only durable project rules,
   architecture decisions, and verified cross-session facts.
4. **User memory** is `~/.xcode/memory/MEMORY.md`. It contains durable
   cross-project preferences.

The layers have different jobs. Current progress and next actions belong in the
checkpoint, never in project or user memory.

## Runtime flow

### Normal turns

The system prompt tells the agent where memory lives and when to use it. It does
not automatically inject search results on every turn. The agent calls the
read-only `search_memory` tool when prior project knowledge may matter.

### Compact

The compactor keeps a verbatim recent tail and writes its full structured
summary to the current session checkpoint. The checkpoint boundary is the real
session message ID supplied by the transcript store.

### Resume

If the checkpoint boundary exists on the selected session branch, Xcode restores:

```text
latest session checkpoint
+ transcript entries from the boundary onward
+ budgeted project and user memory
```

If the checkpoint is absent, corrupt, or belongs to another branch, Xcode falls
back to the complete transcript.

## Invariants

- Markdown is the source of truth for durable memory.
- Checkpoints are isolated by session ID.
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

## Remaining long-horizon work

The current implementation establishes a small correct base. MiMo-style parity
would additionally require:

- early checkpoint extraction before the context is nearly full;
- a raw-history search/around tool for details older than the rebuild boundary;
- an independent single-writer path for promoting repeatedly verified facts
  from session checkpoints into project memory.

These belong to checkpoint/history integration. They must not expand the
per-record Memory model.
