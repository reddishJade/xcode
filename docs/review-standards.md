# Xcode Review Standards

This document defines how coding agents should review this repository. It is separate from implementation standards: `AGENTS.md` and code standards guide how to write code; this file guides how to judge code.

## Review Goal

Review code for correctness, maintainability, Pythonic design, clean boundaries, and long-term elegance.

Do not treat “tests pass” as sufficient. Passing tests only prove current behavior under current coverage. A review must also judge whether the code is simple, well-shaped, and worth keeping.

Prefer fewer high-quality findings over many minor comments.

## Review Scope

Check for:

* serious bugs and edge-case failures;
* architecture and module-boundary problems;
* security risks and unsafe defaults;
* non-Pythonic or awkward implementation patterns;
* unnecessary abstraction, compatibility layers, indirection, or defensive code;
* duplicated helpers or concepts;
* APIs shaped for tests rather than real usage;
* tests that overfit implementation details;
* code that works but is not clean, separable, or elegant.

## Core Principles

Code should be Pythonic, explicit, typed, and easy to reason about.

Prefer direct design over compatibility scaffolding. Do not preserve backward compatibility unless there is an explicit project requirement.

Tests serve production code. Do not accept awkward production APIs, leaked internals, or distorted boundaries merely because existing tests expect them.

A clean deletion is often better than preserving unused flexibility.

A small amount of duplication is better than premature abstraction. Abstraction is justified only when it clarifies a real shared concept.

Avoid “just in case” design. Flexibility without a concrete caller is usually noise.

## What To Flag

Flag code when it:

* hides simple logic behind unnecessary helpers, classes, registries, factories, or protocols;
* mixes unrelated responsibilities in one module or class;
* creates compatibility paths that are not required by current users;
* uses dynamic imports, untyped escape hatches, broad exception swallowing, or loose dictionaries where typed structures would be clearer;
* makes tests depend on private implementation details instead of observable behavior;
* changes production design only to make tests easier;
* adds comments that explain obvious code instead of clarifying real constraints;
* introduces dependencies, config, or runtime behavior that is broader than the actual need;
* duplicates concepts under different names;
* keeps dead code, unused extension points, or speculative branches.

## What Not To Flag

Do not report pure personal taste.

Do not nitpick formatting that Ruff or the formatter should handle.

Do not request abstraction merely because two pieces of code look similar. First decide whether they express the same concept.

Do not demand backward compatibility unless the repository explicitly requires it.

Do not treat test failures as automatically meaning production code is wrong; the test may be the thing that should change.

## Finding Format

Each finding should include:

* severity: critical, high, medium, or low;
* file path and symbol;
* the problem;
* why it matters;
* the cleanest recommended direction.

Use direct language. If something exists only for compatibility, test convenience, or accidental complexity, say so.

Prefer this format:

```text
[medium] src/xcode/example.py::ExampleRunner

Problem:
...

Why it matters:
...

Recommended direction:
...
```

## Severity Guide

Use `critical` for issues that can cause data loss, security exposure, command execution risk, or completely broken core behavior.

Use `high` for likely runtime bugs, incorrect behavior in normal use, unsafe architecture, or design that blocks future work.

Use `medium` for maintainability problems, unclear boundaries, bad abstractions, fragile tests, or non-Pythonic design that will likely create bugs later.

Use `low` for local cleanup opportunities that improve clarity but do not currently distort the design.

Do not inflate severity to make a point.

## Review Output Rules

Do not modify files during review.

Do not produce a long list of minor style comments.

Group related findings when they share the same root cause.

Include file paths and concrete examples.

If the code is mostly good, say so and report only meaningful issues.

If a finding depends on an assumption, state the assumption.
