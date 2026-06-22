# Xcode REPL Capability Notes

This sample knowledge base describes the current Xcode REPL workflow. It is
intended for quick eval tasks and examples, so it focuses on behavior that is
visible in `README.md`, `ARCHITECTURE.md`, and the REPL implementation.

## Default Runtime Shape

Xcode is a lightweight coding-agent harness. The default path is:

```text
CLI / REPL -> StructuredAgent -> short prompt -> core tools -> permission/risk/audit -> final answer
```

The core tool set is deliberately small: `read_file`, `write_file`,
`edit_file`, `glob_files`, `grep_search`, guarded `bash`, and
`run_validation`. Optional groups such as `skills`, `subagent`, `worktree`,
`tasks`, and `mcp` are always available by default.

## Planning And Acting

The REPL exposes execution modes as slash commands:

- `/plan` enters a read-only planning mode. It exposes inspection tools such as
  lexical search and file reads, but blocks edits and shell execution.
- `/review` is a middle mode for review and guarded validation.
- `/act` returns to normal execution mode while still applying sandbox,
  permission, risk, audit, HITL, and deny rules.
- `/act --clear` saves the last assistant plan to
  `.local/session_artifacts/plan-{id}.md`, forks into a clean isolated session,
  and injects the approved plan into the next prompt as `<approved-plan>`.

This mode system is enforced by the harness, not merely by prompt wording.

## Direct Tool Debugging

`/tool NAME INPUT` runs a registered tool directly without asking the LLM to
decide the next action. It still goes through the same registry, permission
policy, risk gates, and HITL approval path. `/tool list` shows visible tools by
group, hidden tools that are available through configuration, and available
tool groups.

## Explicit File References

Users can include `@relative/path` anywhere in a normal prompt. The REPL reads
that project-root-relative text file and injects it into the model input as a
temporary `<file-reference>` block. The original user message is preserved
inside `<user-message>`.

File references are sandboxed. They must resolve inside the project root, and
sensitive paths such as `.git`, `.venv`, `.local`, and `.env` are blocked by the
file tooling.

## Git Preflight Context

For coding tasks, the runtime prompt includes a `<git-preflight>` block assembled
before each structured task. It contains the current `git status --short`, the
previous commit summary, and a diff stat when the working tree is dirty. Agents
must treat existing dirty and untracked files as user-owned baseline and read
relevant files before editing them.

The git preflight block is injected by the harness so users do not need to ask
the model to run `git status` before every normal implementation request.

## Markdown Rendering

The REPL renders final assistant answers with `rich.markdown.Markdown` when
`rich` is installed. If `rich` is unavailable, it falls back to printing raw
text. This means normal Markdown such as headings, lists, code fences, and
diff-shaped snippets are useful output formats in the terminal.

## Tab Completion

The REPL uses `prompt_toolkit` completion. Pressing Tab can complete:

- slash commands such as `/plan`, `/review`, `/act`, `/compact`, and `/tool`;
- tool names after `/tool `;
- `@file` references for local project files.

File completion ignores blocked directories such as `.git`, `.venv`, and
`__pycache__`, and limits suggestions to keep the terminal responsive.

## Session And Context Controls

`/compact` requests active context compaction and shrinks large tool results in
the session log. `/sessions` lists recent conversations, `/resume` restores a
conversation, and `--resume` opens the resume picker at startup.

The runtime uses layered compaction: stale `read_file` results can be snipped
while preserving the latest read for each path, large tool results can be
truncated, and restored sessions rebuild active file-read metadata.
