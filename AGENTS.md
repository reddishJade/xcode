# Xcode Agent Guide

Entry point for coding agents working in this repository. Keep this file short; use the linked docs for detail.

## 项目地图

Xcode 是轻量级 Python Agent 运行骨架，主分层为：
`ai/` (provider) → `agent/` (loop core) → `harness/` (runtime infra) → `coding_agent/` (coding product) → `cli/` (UI)

代码位于 `src/xcode/`。优先按改动目标定位：

- provider / model 接入：`src/xcode/ai/`
- agent loop / state / tool 调度：`src/xcode/agent/`
- runtime / sandbox / harness：`src/xcode/harness/`
- coding agent 产品逻辑：`src/xcode/coding_agent/`
- CLI / app 入口：`src/xcode/cli/`
- 测试：`src/xcode/tests/`

运行路径：`main.py` → `build_app()` → `StructuredAgent` → `Agent` loop → provider stream → tool execution。

入口：

- CLI：`xcode` 或 `python -m xcode`
- 编程式 API：`build_app()`
- eval：`xcode-eval` 或 `python -m xcode.evals.cli`

## 先看哪里

只读取与你当前任务有关的文档：

| 文档 | 何时查看 |
|---|---|
| [docs/code-organization.md](docs/code-organization.md) | 调整源码结构、模块边界、分层职责 |
| [docs/evaluation-guide.md](docs/evaluation-guide.md) | 运行 lint、typecheck、test、eval |
| [docs/git-workflow.md](docs/git-workflow.md) | 准备提交前 |
| [docs/source-review.md](docs/source-review.md) | 做较大改动前的源码审查 |
| [CONFIG.md](CONFIG.md) | 查看运行时配置、开关、环境变量 |

## 常用路径

```powershell
# 安装开发依赖
uv pip install -e ".[dev]"

# 基本静态检查
uv run ruff check src/
uv run pyright src/

# 运行相关测试
uv run pytest src/xcode/tests/ -q --tb=short

# 编译检查
uv run python -m compileall src
```

更多命令、单测示例和 eval 用法见 [docs/evaluation-guide.md](docs/evaluation-guide.md)。

## 编码约定

- Python 3.12+，完整类型注解，使用 ruff 默认格式（行宽 88）
- 注释和 docstring 使用简体中文
- 函数职责单一，优先纯函数，捕获具体异常类型
- 依赖变更视为代码变更，需要一并审查
- 默认不为旧接口保留兼容层，除非任务明确要求

## 仓库约束

- 新工具必须声明 `group`、`risk`、`schema`、`read-only` 和 `concurrency`
- MCP 工具由 `.local/mcp_config.json` 自动发现并注册
- `edit_file` 依赖 read-before-edit 指纹校验

## 安全与边界

- 不运行需要真实 provider、外部凭证或付费资源的流程，除非用户明确要求
- 不提交 secrets、token、凭证文件或包含敏感信息的真实日志
- 不随意修改 `.local/`、本地环境配置或 MCP 配置，除非任务明确涉及这些文件
- 需要联网、改依赖或运行真实 eval 前，先确认任务确实需要

## 提交与测试

- 提交前阅读 [docs/git-workflow.md](docs/git-workflow.md)
- 只修改文档时，优先运行 `git diff --check -- <modified-docs>`
- 不运行依赖外部环境变量的端到端套件，除非用户明确要求
