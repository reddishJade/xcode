# Xcode Agent Guide

Entry point for coding agents working in this repository. Keep this file short; read linked documents for details.

## Required Reading

| Document | Purpose |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | General coding behavior, commit format, comment/docstring rules |
| [docs/code-standards.md](docs/code-standards.md) | Detailed code quality, dependency, test, and validation rules |
| [docs/git-workflow.md](docs/git-workflow.md) | Multi-session Git workflow and commit boundaries |
| [docs/code-organization.md](docs/code-organization.md) | Current module mapping and tool group layout |

## Priority

1. Follow the user's latest explicit instruction.
2. Follow this file for repository-specific agent behavior.
3. Follow `CLAUDE.md` for general coding behavior.
4. Follow detailed docs linked above for implementation, validation, and Git workflow.

If instructions conflict, ask for explicit confirmation before proceeding.

## Conversation Style

- Keep answers short, technical, and direct.
- No emojis, no fancy or cheerful filler text.
- When user asks a direct question: answer first, then execute modifications or commands if requested.
- When responding to feedback or analysis: explicitly state agreement or disagreement, then describe what changed.

## Working Rules

- Read complete files before large changes, audits, or edits to files you have not reviewed.
- Prefer small, precise changes that match existing style.
- Do not remove intentional behavior unless the user confirms.
- Do not modify generated files directly; modify the generator.
- Use top-level static imports only. Do not add dynamic imports.
- Treat `src/xcode/experimental/` features as opt-in. Every experimental capability must have an explicit group; `experimental` is only the total enable switch.

## Validation

- After code changes, run formatting, lint, type checks, and tests only for modified files or related functionality.
- Run `ruff format` in write mode before final delivery when Python files are modified.
- Use mocks or fake providers for external services. Do not call real paid APIs in tests.
- For documentation-only changes, run `git diff --check` and targeted tests when the docs describe code behavior.

## Git Safety

- The working tree may contain changes from other sessions.
- Stage only exact paths for the current task.
- Never use `git add -A` or `git add .`.
- Never run `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, or `git commit --no-verify`.
- Before committing, inspect `git status --short` and `git diff --cached --stat`.

## Common Commands

```powershell
# Install editable package
.\.venv\Scripts\python.exe -m pip install -e .

# Targeted tests
uv run python -m unittest src.xcode.tests.test_xcode_app_runtime

# Full tests, only when explicitly needed
uv run python -m unittest discover src\xcode\tests

# Compile check
uv run python -m compileall src

# Python formatting/lint/type checks
uv run ruff format <modified-files>
uv run ruff check <modified-files>
uv run ruff format --check <modified-files>
uv run mypy <modified-files>
```
