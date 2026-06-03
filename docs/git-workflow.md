# Xcode Git Workflow

This repository may have multiple coding sessions active in the same working tree. Git commands must preserve unrelated user or agent changes.

---

## Before Staging

Inspect the tree:

```powershell
git status --short
git diff --stat
```

Identify the exact files changed for the current task. Treat unrelated modified or untracked files as user-owned unless the user explicitly asks to include them.

---

## Staging

Stage exact paths only:

```powershell
git add path\to\file1 path\to\file2
```

Never use:

```powershell
git add -A
git add .
```

After staging, inspect the staged boundary:

```powershell
git diff --cached --stat
git status --short
```

The staged set must contain only files for the current task.

---

## Commit Message Format

Use the commit format from @AGENTS.md. Commit messages must be written in English.

```text
[type]: [one-line short title describing major feature in lowercase and imperative]

[1-2 paragraph description explaining context, rationale, and impact of changes.]

Key changes:

- [Brief grouped change]
- [Brief grouped change]
```

---

## Commit Command

Use normal commit after exact staging. Follow the format described in [Commit Message Format](#commit-message-format):

```powershell
git commit -m "type: title" -m "body"
```

If using `git commit --only`, place all `-m` flags before the pathspec:

```powershell
git commit --only -m "type: title" -m "body" -- path\to\file
```

> **PowerShell note**: When using `@'...'@` single-quoted here-strings, the `@` delimiters are not passed to the command. However, `@"..."@` double-quoted here-strings expand `$` variables and backtick escapes. Prefer `@'...'@` for multi-line commit messages to avoid accidental variable expansion.

---

## Never Run

Never run these commands unless the user explicitly requests them and confirms the risk:

```powershell
git reset --hard
git checkout .
git clean -fd
git stash
git commit --no-verify
```

Never force push.

---

## Rebase And Conflicts

If rebasing causes conflicts:

- Resolve only files you modified for the current task.
- If conflicts appear in files you did not modify, abort and ask the user.
- Do not use broad checkout/reset commands to resolve conflicts.

---

## After Commit

Confirm the commit boundary:

```powershell
git status --short
git show --stat --oneline --no-renames HEAD
```

Unrelated local changes should remain uncommitted.
