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

Use the repository format from `CLAUDE.md`:

```text
[type]: [one-line short title describing major feature in lowercase and imperative]

[1-2 paragraph description explaining context, rationale, and impact of changes.]

Key changes:

- [Brief grouped change]
- [Brief grouped change]
```

Example:

```text
docs: split agent workflow guidance

Move detailed implementation and Git workflow rules out of AGENTS.md so the agent entry file stays short and points to focused reference documents.

Key changes:

- Add code standards and Git workflow reference docs
- Reduce AGENTS.md to required reading, priority, validation, and safety rules
```

---

## Commit Command

Use normal commit after exact staging:

```powershell
git commit -m "type: title" -m "Body paragraph.

Key changes:

- Change one
- Change two"
```

If using `git commit --only`, place all `-m` flags before the pathspec:

```powershell
git commit --only -m "type: title" -m "body" -- path\to\file
```

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
