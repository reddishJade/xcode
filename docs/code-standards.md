# Xcode Code Standards

This document contains detailed implementation rules for coding agents. `AGENTS.md` is the short entry point; this file covers general coding behavior, comment/docstring rules, code quality, dependency, test, and validation rules.

---

## Reading And Editing

- For large changes, audits, or edits to files not fully reviewed, read the complete file first.
- Do not rely only on search snippets for large changes.
- Match existing module boundaries and local style.
- Inline single-line helper functions when they have only one call site.
- Do not use `Any` unless it is necessary at an external boundary or unavoidable dynamic interface.
- Check local types in the virtual environment or `site-packages` before guessing external APIs.
- Use top-level static imports only. Do not use `importlib.import_module`, `__import__`, or similar dynamic import patterns.
- Do not modify generated files directly; modify the generator instead.

---

## Debugging Approach

When investigating a bug or unexpected behavior, follow these steps in order:

1. **Understand the architecture** — read relevant module boundaries and data flow before touching any code.
2. **Analyze the root cause** — identify the actual mechanism of failure, not just the symptom.
3. **Design the interface** — decide what the fix should look like at the boundary before writing implementation.
4. **Consider edge cases** — what else could break, what assumptions changed, what input could bypass the fix.

---

## Syntax And Compatibility

- Use modern Python syntax supported by the project runtime.
- Prefer typed dataclasses and explicit type annotations.
- Tests serve the code, not the other way around. Do not preserve awkward production APIs only to keep existing tests unchanged.
- Do not maintain backward compatibility unless the user asks for it.
- Do not keep compatibility layers for their own sake. When a boundary is wrong, clean it up directly.
- Never remove or downgrade code to fix type errors caused by old dependencies; upgrade dependency metadata instead when that is the intended fix.
- Never hardcode key or secret checks; place configurable checks in config constants or policy code.

---

## Comments And Docstrings

- Runtime comments, inline notes, and docstrings must use Simplified Chinese.
- Comments must be restrained and factual.
- Avoid teaching-demo wording, marketing claims, or assertions of production quality.
- Add comments only where they clarify non-obvious constraints or logic.
- Do not add unnecessary comments. Let the code speak for itself.
- Documentation explains architectural decisions, trade-offs, and constraints — why the system is shaped the way it is.

---

## Dependencies

- Treat dependency and lockfile changes as code changes requiring review.
- Pin external dependencies to exact versions when adding them.
- Use `pip install --no-deps` or `pip sync --no-deps` when installing manually.
- New dependencies with install hooks require explicit review and allow-listing.
- When dependency metadata changes, update `pyproject.toml` and any relevant generated lock or requirements files.

---

## Temporary Scripts

- Write temporary scripts to a temporary location.
- Delete temporary scripts when done.
- Do not embed multi-line scripts directly in shell commands.
- Prefer project utilities or tests over ad hoc scripts when the project already has a fitting entry point.

---

## Validation Scope

After modifying Python files:

```powershell
uv run ruff check <modified-files> --fix
uv run ruff format <modified-files>
uv run pyright <modified-files>
```

After modifying tests:

```powershell
uv run pytest <targeted-test-file> -q --tb=short
```

After modifying documentation only:

```powershell
git diff --check -- <modified-docs>
```

If docs describe runtime behavior, also run the related targeted tests.

Do not run full test suites unless:

- the user asks for it,
- the touched code is broad/shared enough to justify it,
- or targeted tests do not cover the modified behavior.

Do not run end-to-end suites that require specific external environment variables unless explicitly requested.

---

## Extension Boundaries

- Do not add an `experimental` package or aggregate enable switch.
- MCP is part of the core runtime and must remain a no-op when no configuration exists.
- Memory is a formal opt-in capability with its own `memory` group.
- `bm25` is an internal implementation detail of Memory, not a separate group.
- New tools must document group, risk, schema, read-only behavior, and tests.
