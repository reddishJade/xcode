# Xcode Agent Guide

Entry documentation for Xcode Agent. Keep it brief: only for locating information sources. Read linked docs for details.

## Required Reading

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](CLAUDE.md) | Coding behavior rules: think before coding, prefer simple solutions, make precise changes, verify changes |
| [docs/code-organization.md](docs/code-organization.md) | Current module mapping |

## Conversation Style

- Keep answers short, technical, and direct
- No emojis, no fancy or cheerful filler text
- When user asks: answer first, then execute modifications or implementation commands
- When responding to feedback or analysis: explicitly state agreement or disagreement, then describe what changed

## Code Quality

### Reading and Modifying
- For large changes, editing files not fully reviewed, or investigation/audit: read the complete file first. Do not rely on search snippets for large changes
- Do not use `Any` type unless absolutely necessary
- Single-line helper functions with only one call site: inline directly
- Check `site-packages` or virtual environment for external API types; do not guess
- **No dynamic imports** (`importlib.import_module`, `__import__`, etc.). Use top-level static imports only

### Syntax and Compatibility
- Use modern Python syntax (3.10+): type annotations, dataclasses, `match` statements. Do not use deprecated features
- Run `ruff format` on all modified files before committing. This is mandatory
- Never remove or downgrade code to fix type errors from old dependencies — upgrade dependencies instead
- Never maintain backward compatibility unless user requests it
- Before removing seemingly intentional functionality or code, ask first
- Never hardcode key checks — add to config constants to keep them configurable
- Never modify auto-generated code files directly — modify the generator script instead

## Implementation Rules

### After Code Changes
- Run `ruff check` and `ruff format` (check mode only), and `mypy` (if configured). Fix all errors, warnings, and infos before committing
- `ruff format` must be run in write mode on all modified files before the final commit
- Do not run tests unless user requests
- Do not run full suites that include end-to-end tests directly (they require specific environment variables)

### Testing
- After creating or modifying test files: run that test and iterate until passing
- For tests involving external services: use mocks or fake providers. Do not use real service APIs, keys, or paid resources

### Temporary Scripts
- Write to temporary files (e.g., `/tmp`), run, edit if needed, delete when done
- Do not embed multi-line scripts in shell commands

## Dependency and Installation Security

- Treat dependency and lockfile changes as code to be reviewed. Pin external dependencies to exact versions
- Use `pip install --no-deps` or `pip sync --no-deps` to avoid running installation scripts
- When dependency metadata changes, refresh `requirements.txt` or `pyproject.toml` lockfile
- New dependencies with install hooks: require review and explicit addition to allow list. Never add silently

## Git Rules

The current directory may have multiple sessions running simultaneously, each modifying different files. Git commands that operate on unstaged, staged, or untracked files outside your own changes will disrupt other sessions.

### When Committing
- Only commit files modified in the current session
- Explicitly specify paths to stage (`git add <path1> <path2>`); never use `git add -A` / `git add .`
- Run `git status` before committing to confirm only your files are staged

### Never Run
- `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add -A`, `git add .`, `git commit --no-verify`

### When Rebasing Causes Conflicts
- Only resolve conflicts in files you modified
- If conflicts occur in files you didn't modify: abort and ask the user
- Never force push

## Commands

```powershell
# First-time standalone checkout
.\.venv\Scripts\python.exe -m pip install -e .

# Xcode tests
uv run python -m unittest discover src\xcode\tests

# Compile check
uv run python -m compileall src

# Ruff Check
ruff check src/xcode/

# Ruff Format
ruff format src/xcode/

# Mypy Check
mypy src/xcode/
```

## Documentation Index

| Document | Content |
|----------|---------|
| [README](README.md) | Quick start, config examples, CLI/REPL usage |
| [TODO.md](TODO.md) | Remaining advanced Multi-Agent and roadmap work |
| [CONFIG.md](CONFIG.md) | Runtime config keys and defaults |
| [docs/code-organization.md](docs/code-organization.md) | Current module responsibilities |
| [docs/source-review.md](docs/source-review.md) | Source-level capability mapping and known boundaries |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | REPL and single-run workflows |

## User Override

If user instructions conflict with any rule in this document, ask the user for explicit confirmation before proceeding. Only execute after receiving confirmation.