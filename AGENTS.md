# Xcode Agent Guide

Entry point for coding agents working in this repository.

## Scope

This file constrains two things simultaneously:

- **Agent behavior**: how the agent communicates, commits, and validates.
- **Code standards**: what correct Python looks like in this project.

---

## Required Reading

Read on demand, not all upfront.

| Document | Purpose | When to read |
| --- | --- | --- |
| [docs/git-workflow.md](docs/git-workflow.md) | Multi-session Git workflow, commit message format, and commit boundaries | Before every commit |
| [docs/code-organization.md](docs/code-organization.md) | Current module mapping and tool group layout | Before editing source structure |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | Test, lint, type check, compile, and eval workflow commands | Before running validation |
| [docs/source-review.md](docs/source-review.md) | Source-level architecture review and module boundaries | Before large refactors or audits |
| [docs/review-standards.md](docs/review-standards.md) | Review standards for judging code correctness and design | Before code review tasks |

---

## Priority

1. Follow the user's latest explicit instruction.
2. Follow this file for repository-specific agent behavior.
3. Follow other docs linked above for implementation, validation, and Git workflow.

When the user's latest instruction and a rule in this file conflict on the same dimension (e.g. commit granularity, import style), ask for explicit confirmation before proceeding. Otherwise, the user's instruction takes precedence.

---

## Conversation Style

- Keep answers short, technical, and direct.
- No emojis, no fancy or cheerful filler text.
- When the user asks a direct question: answer first, then execute modifications or commands if requested.
- When responding to feedback or analysis: explicitly state agreement or disagreement, then describe what changed.

---

## Python Coding Principles

Every principle below applies to every line of Python written or reviewed in this project.

### 0. Quality Over Convention

- Good code does not need to explain *what* it does — the code itself is the explanation.
- Do not plan in phases. Do it right the first time in one pass: complete, correct, and clean.
- Code must be clear, complete, and extensible — not merely functional.

### 1. Readability Counts

- Name variables, functions, and classes so their purpose is self-evident. Avoid single-letter names outside conventional uses (`i`, `j`, `_`).
- Use whitespace, indentation, and comments deliberately. Comments explain *why*, not *what*.
- Every file, function, and class must have a docstring and complete type annotations.

```python
def calculate_discount(price: float, rate: float) -> float:
    """根据折扣率计算折后价格。"""
    return round(price * (1 - rate), 2)
```

### 2. Explicit Is Better Than Implicit

- Do not pass `*args` / `**kwargs` through layers unless unavoidable.
- Avoid `getattr`, `setattr`, and other dynamic attribute access where a static alternative exists.
- Prefer plain loops and conditionals over clever nested comprehensions.

```python
# 产生新值，无副作用，意图明确
data = [1, 2, 3]
new_data = data + [4]
```

### 3. Flat Is Better Than Nested

- Keep indentation at most 3 levels deep. Extract functions, return early, or use `continue`.
- Replace long `if-elif` chains with dictionaries or `match-case`.

```python
# 提前返回，减少嵌套层级
def process(x):
    if not x:
        return
    for i in x:
        if i <= 0:
            continue
        ...
```

### 4. Do One Thing

- A function should do exactly one thing.
- Do not mix IO, computation, and display in the same function.
- Prefer pure functions — no side effects.

```python
# 职责分离：读取、处理、展示各自独立
def read_file(path: str) -> str: ...
def process(text: str) -> str: ...
def display(text: str) -> None: ...
```

### 5. Use Pythonic Idioms

- `enumerate` instead of `range(len(...))`
- `zip` for parallel iteration
- `in` for membership checks
- `with` for resource management
- `@dataclass` / `NamedTuple` for plain data objects

### 6. Handle Exceptions Strictly, Never Silently

- Catch specific exception types, never bare `except:`.
- Exception messages must carry meaningful context.
- Do not use `try/except` to replace normal control flow.

```python
# 只处理已知异常，其余向上传播
try:
    return int(s)
except ValueError:
    return 0
```

### 7. Enforce PEP 8 Through Tooling

- Format with `ruff`, check with `ruff` / `pyright`.
- Line length ≤ 88 (ruff default).
- Import order: standard library → third-party → local.
- Zero `# noqa` or `# type: ignore` must pass lint. If one is necessary, document the reason inline.

### Checklist

Before marking any change done, verify:

- [ ] Someone can understand the code without reading docs.
- [ ] A function can be safely copied into another project.
- [ ] Changing one behavior requires editing one place.
- [ ] Logic is traceable without a debugger.
- [ ] Type hints and docstrings are complete.
- [ ] Lint passes with zero `# noqa`.

---

## Project Rules

### Syntax and Compatibility

- Use modern Python syntax supported by the project runtime.
- Prefer typed dataclasses and explicit type annotations. Avoid `Any` except at external boundaries or unavoidable dynamic interfaces.
- Use top-level static imports only. Do not use `importlib.import_module`, `__import__`, or similar dynamic import patterns.
- Tests serve the code, not the other way around. Do not preserve awkward production APIs only to keep existing tests unchanged.
- Do not maintain backward compatibility unless the user asks for it.
- Never hardcode key or secret checks; place configurable checks in config constants or policy code.
- Do not modify generated files directly; modify the generator.

### Comments and Docstrings

- Runtime comments, inline notes, and docstrings must use Simplified Chinese.
- Comments must be restrained and factual. Avoid teaching-demo wording, marketing claims, or assertions of production quality.
- Add comments only where they clarify non-obvious constraints or logic.
- Documentation explains architectural decisions, trade-offs, and constraints — why the system is shaped the way it is.

### Dependencies

- Treat dependency and lockfile changes as code changes requiring review.
- Pin external dependencies to exact versions when adding them.
- Use `pip install --no-deps` or `pip sync --no-deps` when installing manually.
- New dependencies with install hooks require explicit review and allow-listing.
- When dependency metadata changes, update `pyproject.toml` and any relevant generated lock or requirements files.

### Temporary Scripts

- Write temporary scripts to a temporary location.
- Delete temporary scripts when done.
- Do not embed multi-line scripts directly in shell commands.
- Prefer project utilities or tests over ad hoc scripts when the project already has a fitting entry point.

### Extension Boundaries

- Do not add an `experimental` package or aggregate enable switch.
- MCP is a core runtime capability and must remain safe when no configuration exists.
- Memory is a formal opt-in capability with its own `memory` group.
- New tools must document group, risk, schema, read-only behavior, and tests.

---

## Git Safety

Allowed operations: `git add <exact-path>`, `git commit`, `git diff`, `git status`, `git log`.

All other operations that rewrite history, discard changes, or move HEAD are forbidden by default. This includes but is not limited to:

- `git add -A` / `git add .`
- `git reset --hard`
- `git checkout .`
- `git clean -fd`
- `git stash`
- `git commit --no-verify`

Before every commit:

```powershell
git status --short
git diff --cached --stat
```

Stage only exact paths for the current task. The working tree may contain changes from other sessions.

---

## Commit Rules

### One Step, One Commit

Each commit must contain exactly one logical change. Do not bundle unrelated modifications. Refactoring, formatting, and feature changes each get their own commit.

### Doc Update Commits

- **Standalone doc changes**: commit separately, never mixed with code changes.
- **Doc changes caused by code changes**: append or merge into the corresponding code commit.

### Refactor Boldly

When architecture is awkward or boundaries are unclear, clean it up directly. Do not preserve bad design out of fear of breaking things.

---

## Validation

Run only for modified files or related functionality.

```powershell
# 静态检查与格式化
uv run ruff check <modified-files> --fix
uv run ruff format <modified-files>
uv run pyright <modified-files>

# 针对性测试
uv run python -m unittest <targeted-test-module>

# 仅修改文档时
git diff --check -- <modified-docs>
```

Run the full test suite only when:

- the user asks for it, or
- the touched code is broad or shared enough to justify it, or
- targeted tests do not cover the modified behavior.

Do not run end-to-end suites that require specific external environment variables unless explicitly requested. Use mocks or fake providers for external services. Do not call real paid APIs in tests.

---

## Common Commands

```powershell
# 安装可编辑包
uv pip install -e .

# 针对性测试
uv run python -m unittest src.xcode.tests.test_xcode_app_runtime

# 完整测试（仅按需运行）
uv run python -m unittest discover src\xcode\tests

# 编译检查
uv run python -m compileall src
```

---

## Working Rules

- Read complete files before large changes, audits, or edits to files not yet reviewed.
- Prefer small, precise changes that match existing style.
- Do not remove intentional behavior unless the user confirms.
- Check local types in the virtual environment or `site-packages` before guessing external APIs.
- Inline single-line helper functions when they have only one call site.

## Debugging Approach

When investigating a bug or unexpected behavior, follow these steps in order:

1. **Understand the architecture** — read relevant module boundaries and data flow before touching any code.
2. **Analyze the root cause** — identify the actual mechanism of failure, not just the symptom.
3. **Design the interface** — decide what the fix should look like at the boundary before writing implementation.
4. **Consider edge cases** — what else could break, what assumptions changed, what input could bypass the fix.
