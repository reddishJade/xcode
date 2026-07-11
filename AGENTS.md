# Repository Guidelines

## Project Structure & Architecture

Xcode is a Python 3.12+ coding-agent harness. All package code lives in
`src/xcode/`; the command-line entry point is `src/xcode/main.py`. The layered
design is `ai/` (provider adapters), `agent/` (loop and context), `harness/`
(runtime, sessions, policy, MCP), `coding_agent/` (tools), and `cli/` (REPL and
TUI). Keep new code in the layer that owns the behavior. Tests reside in
`src/xcode/tests/`; documentation and examples are in `docs/` and `examples/`.

## Build, Test, and Development Commands

Install runtime or development dependencies with:

```sh
uv pip install -e .
uv pip install -e ".[dev]"
```

Run the application with `uv run xcode`. Before submitting changes, run:

```sh
uv run ruff check src/ --fix  # lint and apply safe fixes
uv run ruff format src/       # format source files
uv run pyright src/           # type-check the package
uv run pytest src/xcode/tests -q --tb=short
```

## Coding Style & Naming

Use complete type annotations and standard four-space Python indentation.
Ruff enforces formatting with an 88-character line length; do not add `# noqa`
or blanket exception handlers. Prefer small, single-purpose functions and
separate I/O, computation, and presentation. Use `snake_case` for modules,
functions, and variables; `PascalCase` for classes; and `test_<feature>.py` for
test modules. Write comments and docstrings in Simplified Chinese.

## Testing Guidelines

Pytest is configured to discover `test_*.py` under `src/xcode/tests`, with
async tests handled automatically by `pytest-asyncio`. Add automated tests for
pure logic; manually verify provider- or terminal-dependent behavior rather
than mocking external I/O. Run a focused test during development, for example:

```sh
uv run pytest src/xcode/tests/test_tools_file_handlers.py -q --tb=short
```

The `mcp_external` tests require network tooling and are excluded by default.

## Commits & Pull Requests

Use focused commits with imperative Conventional Commit-style subjects, such as
`fix: refine tui input presentation` or `feat: add session export`. Stage only
explicit paths (`git add src/xcode/...`), never `git add .`. In pull requests,
describe the behavioral change, list validation commands, link relevant issues,
and include terminal screenshots when a REPL or TUI change is visible.
