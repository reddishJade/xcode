# Xcode 代码组织说明

本文描述 Xcode 独立项目根目录的职责划分。当前独立 checkout 使用 `src/` 包布局：根目录包含项目元数据、文档、技能与示例，Python 包、测试和 eval 代码位于 `src/xcode/`。

## 顶层结构

```text
.
├── pyproject.toml          # 独立项目元数据（uv/pip 依赖声明）
├── AGENTS.md               # 开发者与编码 Agent 工作入口说明
├── CLAUDE.md               # 编码风格与实现规范（Surgical edit & Simple first）
├── README.md               # 全局使用手册与功能概览
├── src/xcode/main.py       # CLI/REPL 唯一运行入口（REPL, --prompt, --resume）
├── src/xcode/cli/          # 交互层：prompt_toolkit 终端 UI 适配
│   ├── repl.py             # REPL 主循环、交互式命令与 Session Transcripts 落盘
│   ├── session.py          # SessionStore 模块（支持 clear/rewind/resume/fork/plan 保存）
│   ├── completion.py       # 命令补全（斜杠命令、/tool 名称、@file 引用补全）
│   ├── file_refs.py        # @relative/path 语法解析与显式文件注入
│   └── markdown.py         # 终端富文本渲染器
├── src/xcode/harness/      # 核心运行与托管层
│   ├── app.py              # 统一装配中心：build_real_app()
│   ├── config.py           # 基础配置 dataclass（AgentConfig）
│   ├── event_bus.py        # 事件总线组件
│   ├── session.py          # SessionStore 转发适配器
│   ├── skills.py           # 核心工具规范（ToolSpec）、HITL 执行控制
│   ├── skill_loader.py     # SKILL.md 轻量目录扫描与 load_skill 按需正文加载
│   ├── tool_wrapper.py     # 工具包装与拦截
│   ├── types.py            # 基础运行类型定义
│   ├── agent_runtime/      # Agent 核心状态机与工作流
│   │   ├── provider.py     # ModelProvider 核心协议
│   │   ├── events.py       # TextDelta/ReasoningDelta/ToolCall 等事件流定义
│   │   ├── structured.py   # StructuredAgent（主线 Agent，消费流式事件）
│   │   ├── tool_executor.py# 工具并发执行器，执行只读 Tool Partitioning 分区
│   │   ├── subagent.py     # 子 Agent Runner 调度层
│   │   ├── prompting.py    # SystemPromptBuilder（动静态模块解耦 Prompt 组装）
│   │   ├── contextual.py   # 最近访问文件与工具摘要动态上下文
│   │   ├── compaction.py   # 分层压缩管线（Stale read snip, Micro compact）
│   │   ├── git_preflight.py# Git Preflight 信息采集注入
│   │   └── cancellation.py # CancellationToken 协作取消控制
│   ├── tools/              # 核心内置工具库
│   │   ├── file.py         # 路径沙箱文件读写（支持 read-before-edit 与 hash 校验）
│   │   ├── code_search.py  # glob_files、grep_search 词法级搜索工具
│   │   ├── bash.py         # 安全受控 shell 执行（带 CommandRiskEvaluator 评估）
│   │   ├── shell_adapter.py# Shell 环境探测与适配
│   │   └── operations.py   # 常规工具操作辅助
│   └── observability/      # 可观测性套件
│       ├── audit.py        # 结构化审计日志输出与 Key 敏感词脱敏
│       ├── permissions.py  # PermissionPolicy 权限规则拦截（allow/deny/ask 三态）
│       └── hooks.py        # HookManager 生命周期钩子
├── src/xcode/ai/           # 传输与模型对接层
│   ├── types.py            # 共享的强类型模型接口与响应定义
│   ├── stream.py           # 响应流拦截与事件产生适配器
│   └── providers/          # 各种 LLM 适配器
│       ├── factory.py      # LLM 客户端工厂与 profile/env 装配
│       ├── codec.py        # OpenAI-compatible tool schema 与 streaming delta 解析
│       ├── deepseek.py     # DeepSeek 增强传输与 thinking mode 支持
│       ├── openai.py       # OpenAI Chat completions & Stateful Responses 适配
│       ├── anthropic.py    # Anthropic Messages 适配
│       ├── mimo.py         # MiMo 适配层
│       ├── faux.py         # 单元测试用 Mock provider
│       └── runtime.py      # 客户端执行期控制 (Retry & RateLimit)
├── src/xcode/experimental/ # 扩展功能（默认不包含在 Core Tool 组中，按需 opt-in 启用）
│   ├── mcp.py              # Stdio MCP 客户端与 ToolSpec 动态代理包装
│   ├── tasks.py            # 任务存储、依赖解析、终端 Kanban 视图与 tasks 工具组
│   ├── mailbox.py          # 基于 filelock 与事件日志型的多进程子代理邮箱
│   ├── memory.py           # 基于 H2 契约校验、BM25 召回、元数据重排和 eval 的 MEMORY.md 动态内存管理器
│   ├── worktree.py         # Git Worktree 任务隔离运行沙箱（WorktreeTaskRunner）
│   ├── plugins.py          # Plugin 动态扫描与加载
│   ├── bm25.py             # 共享的纯 Python BM25Okapi 核心算法实现
│   └── speculation.py      # 安全预热事件管理器（不触发副作用的安全预测）
├── skills/                 # 放置 SKILL.md 可装载外部技能库
├── src/xcode/tests/        # 单元测试模块
└── docs/                   # 核心设计文档与审查报告
```

## 运行入口与装配路径

```text
src/xcode/main.py
  -> src/xcode/cli/repl.py (默认交互入口，带 prompt_toolkit 控制)
  -> src/xcode/harness/app.py::build_real_app()
       -> provider bundle (src/xcode/ai/providers 根据 model_profiles 组装)
       -> ToolSpec registry (src/xcode/harness/skills + src/xcode/harness/tools + enabled_groups 过滤组装)
       -> StructuredAgent (src/xcode/harness/agent_runtime 状态机)
```

`build_real_app()` 是唯一装配中心：负责组装工具注册表、配置参数、Agent 运行时，以及可选能力（worktree、audit、Skill 目录加载、MCP）。工具按 `tools.enabled_groups` 条件构造，再把过滤后的可见 registry 传递给 subagent，避免子 Agent 看到或绕过父 Agent 未启用的工具。

## `src/xcode/harness/` 与 `src/xcode/ai/` 模块职责

| 模块 | 作用 | 关键文件 / 目录 |
| --- | --- | --- |
| **Agent 运行时** | 结构化 tool-use 协议编解码、子 Agent 管理、Prompt 构造、分层上下文压缩 | `src/xcode/harness/agent_runtime/*.py` |
| **工具系统** | `ToolSpec` 协议约定、HITL 机制、工具执行过滤与外置技能动态加载 | `src/xcode/harness/skills.py`, `src/xcode/harness/skill_loader.py` |
| **内置工具集合** | 路径沙箱文件读写、统一 Diff 输出、词法正则搜索、受控 bash 执行及适配 | `src/xcode/harness/tools/*.py` |
| **扩展能力** | Git Worktree 沙箱、Stdio MCP 客户端、任务存储、非副作用 Speculation 安全预热 | `src/xcode/experimental/*.py` |
| **I/O 适配层** | 环境变量读取、LLM 客户端工厂、Chat 与 stateful Responses 协议底层对接 | `src/xcode/ai/providers/*.py` |

## 默认工具与扩展工具

默认 `enabled_groups=["core"]`，可见工具组为：

- **文件系统**：`read_file` / `write_file` / `edit_file`
- **代码检索**：`glob_files` / `grep_search`
- **执行**：`bash`

扩展工具组需要显式加入 `tools.enabled_groups` 进行 opt-in 激活：

- `validation`：提供 `run_validation` 白名单验证命令
- `skills`：提供 `load_skill` 手动装载外部技能
- `subagent`：提供 `submit_subagent` / `check_subagent` / `cancel_subagent` 异步多 Agent 协作能力
- `worktree`：提供 `create_worktree_task` / `remove_worktree_task` 物理隔离测试运行
- `tasks`：提供 `create_task` / `update_task` / `list_tasks` / `get_task` 任务依赖关系管理与终端 Kanban 看板工具
- `mcp`：从外部 `mcp_config.json` 指定的 stdio 服务器中，自动扫描、映射并生成的动态工具（前缀统一使用 `mcp__server__tool`）

在 StructuredAgent 的执行逻辑中，只读（`read_only=True`）且并发安全（`concurrency_safe=True`）的工具会分流至 ThreadPoolExecutor 中并行执行，其余写操作和高风险命令则严格保持模型原始顺序串行执行并配合 HITL 人工审计网关。

## 默认本地路径

- **REPL transcripts 记录**：`.local/sessions/` (包含 JSONL 格式的完整交互历史)
- **REPL 会话索引**：`.local/session_index.json` (保存缩略标题与摘要)
- **Plan artifacts 归档**：`.local/session_artifacts/` (用于保存 Plan 模式导出的 approved 计划文件)
- **MCP schema cache**：`.local/mcp_cache.json` (缓存服务器端 Schema 与 Hash)
- **MCP server config**：优先读取 `.local/mcp_config.json`，若不存在则降级到项目根 `mcp_config.json`
- **Skill 技能库目录**：物理路径为 `skills/`

这些本地路径统一支持通过 `xcode.config.json` 覆盖配置。CLI 入口保持简洁，仅保留 `--project-root`、`--config`、`--sessions-dir`、`--resume` 和 `-p/--prompt` 基础控制项。
