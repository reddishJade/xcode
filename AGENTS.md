# Xcode Agent Guide

Entry point for coding agents working in this repository.

## 项目概述

Xcode 是轻量级 Python Agent 运行骨架。四层架构：
`ai/` (Provider) → `agent/` (Loop Core) → `harness/` (Runtime Infra) → `coding_agent/` (Coding Product) → `cli/` (UI)

运行路径：`main.py` → `build_app()` → `StructuredAgent` → `Agent` loop → provider stream → tool execution。

## 常用命令

```powershell
# 安装
uv pip install -e .                          # 运行时
uv pip install -e ".[dev]"                   # 开发依赖

# 静态检查（lint → typecheck）
uv run ruff check src/ --fix
uv run ruff format src/
uv run pyright src/

# 测试
uv run pytest src/xcode/tests/ -q --tb=short            # 全部
uv run pytest src/xcode/tests/test_xcode_file_tools.py -q --tb=short  # 单个

# 编译检查
uv run python -m compileall src

## 代码规范

- Python 3.12+，完整类型注解，ruff 格式化（行宽 88），零 `# noqa` / `# type: ignore`
- 注释和 docstring 使用简体中文
- 函数职责单一，纯函数优先，异常捕获具体类型
- `*args`/`**kwargs` 只在不避免的边界使用
- 禁止动态 `importlib`/`getattr`/`setattr`
- 依赖变更视为代码变更，需要 review
- 不维护向后兼容，除非用户要求

## Git 规则

- 只允许 `git add <exact-path>`，不允许 `git add -A` / `git add .`
- 禁止历史重写操作（reset --hard、checkout .、clean -fd、stash）
- 每个 commit 只含一个逻辑变更
- 提交前检查：`git status --short && git diff --cached --stat`
- commit message 格式：`type: one-line title` + body（英文）

## 测试

- `pytest-asyncio`，`asyncio_mode = "auto"`，位于 `src/xcode/tests/`
- 回归套件（`--suite all`）：`pipeline` + `tool-policy` + `context` + `multi`
- 仅修改文档时：`git diff --check -- <modified-docs>`
- 不运行需要外部环境变量的端到端套件（除非明确要求）
