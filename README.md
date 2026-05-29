# Xcode Coding Agent Harness Spine

Xcode 是一个轻量、自聚的面向 Python 开发者的 Coding Agent Harness。默认提供 **“短提示词 + 少量核心工具的可靠闭环”** 运行模式：词法级搜索 → 读取文件 → 针对性编辑 → 动态风险评估 → HITL 人工审批网关 → 编译/测试验证 → 敏感词审计脱敏 → 主动上下文分层压缩与重载。

同时，Xcode 具备开箱即用的扩展机制（如 Stdio MCP Gateway、Git Worktree 任务沙箱隔离、Skill 目录渐进加载、Managed Subagents 并发调度等），全部通过配置动态加载。

---

## 核心技术特色

*   **结构化事件流 Agent 状态机 (`StructuredAgent`)**
    *   主循环直接对接 `ContentBlock`（`text` / `tool_use` / `tool_result`）协议，不绑定特定 LLM SDK。
    *   全面流式支持，将底层 LLM 产生的分块转换为强类型事件（`TextDelta`, `ReasoningDelta`, `ToolCallReady` 等），支持推理阶段 (Reasoning Process) 的前置展示且默认不写入历史。
*   **并发工具分区执行 (`Tool Partitioning`)**
    *   在工具执行阶段，通过并发分区算法将工具调用分为并行与串行两类。只读（`read_only`）且并发安全（`concurrency_safe`）的词法检索工具会在独立的线程池（`ThreadPoolExecutor`，默认 4 个 worker）中并行执行，写入与高风险工具则保持串行及严格的执行顺序，防止写入竞态与混乱。
*   **Git Preflight 脏工作区保护与 Diff Stat 注入**
    *   在每轮生成/执行开始前，将当前的 `git status`、上一笔 Commit 摘要和脏代码的 diff stat 统计作为上下文前置注入，确保模型具备完整的工作区环境感知，杜绝静默代码覆盖。
*   **三态 Permission Policy 与受控安全 Bash 执行**
    *   内置 `CommandRiskEvaluator`，根据敏感危险指令（`rm -rf`, `sudo`）及网络连接指令（`pip install`, `curl` 等，支持 ask/deny/allow 策略配置）对 shell 执行进行安全评级。
    *   通过 `ShellAdapter` 自动探测宿主操作系统的首选 Shell（POSIX 优先选择 `$SHELL`；Windows 优先选择 `pwsh > powershell > bash > cmd`），使用 `Popen` 协作控制，在超时或模型取消时终止子进程树。
*   **分层上下文压缩 (`Layered Compaction`) 与自动状态重载**
    *   **Cheap-First 压缩策略**：自动裁剪同一路径中过期的 `read_file` 历史只保留最新版本；针对非只读的超大工具结果执行头尾截断；对旧 `tool_result` 执行微压缩占位；最后才进入 LLM 消息落盘和 System 摘要压缩。
    *   **压缩后状态恢复机制**：会话在压缩或重载后，通过 `restore_read_versions` 自动扫描历史消息，动态重建 `read_file` 的哈希版本校验，保持 Read-Before-Edit 的绝对编辑安全。
*   **流式 Diminishing Returns 收益递减熔断**
    *   在自动续跑（Continuation）阶段，流式监控增量消息。若连续 3 次单轮输出 Token 增量低于 500，将自动抛出熔断异常，彻底杜绝模型因死循环、无限自我纠正或空包回复空耗 Token。
*   **开箱即用的 Stdio MCP Gateway**
    *   提供基于 stdio JSON-RPC 的 MCP 客户端。自动从根目录或 `.local/` 读取 `mcp_config.json`，并将 MCP schema 缓存在 `.local/mcp_cache.json` 中配合配置 Hash 检验防止启动开销。所有动态注册的 MCP 工具（前缀 `mcp__server__tool`）依然经过受控 HITL 和审计链条，确保高度合规。

---

## 目录结构职责

```text
xcode/
├── pyproject.toml          # 独立项目元数据（声明 uv/pip 依赖）
├── AGENTS.md               # 开发者与编码 Agent 开发工作流导读
├── CLAUDE.md               # 编码实现与架构设计约束说明
├── src/xcode/main.py       # CLI/REPL 统一启动入口
├── src/xcode/cli/          # 交互式 prompt_toolkit 终端实现
│   ├── repl.py             # REPL 交互主循环与 Slash 命令行处理器
│   ├── session.py          # SessionStore 模块（Session 落盘、/fork 等）
│   ├── completion.py       # 命令、工具及 @file 引用的自动补全
│   ├── file_refs.py        # @relative/path 文件引用注入语法分析
│   └── markdown.py         # 终端 Markdown & Diff 语法富文本渲染
├── src/xcode/harness/      # 核心运行与托管层
│   ├── app.py              # 唯一装配中心：build_real_app()
│   ├── config.py           # 基础/运行时配置 dataclass (AgentConfig)
│   ├── event_bus.py        # 核心运行事件总线
│   ├── session.py          # SessionStore 转发适配器
│   ├── skills.py           # 工具规范与 HITL 拦截层
│   ├── skill_loader.py     # SKILL.md 轻量目录扫描与按需正文加载
│   ├── tool_wrapper.py     # 工具包装适配器
│   ├── types.py            # 核心数据类型定义
│   ├── agent_runtime/      # 运行时协议、StructuredAgent 循环、Prompt 组装、上下文压缩
│   ├── tools/              # sandboxed file, search, bash 核心内置工具库
│   └── observability/      # 权限策略、审计日志、生命周期 Hooks 注册
├── src/xcode/ai/           # 传输与模型对接层
│   ├── types.py            # 强类型大模型传输接口声明
│   ├── stream.py           # 流式响应拦截适配器
│   └── providers/          # 模型连接适配层 (OpenAI, DeepSeek, Anthropic, MiMo 等)
├── src/xcode/evals/        # Agent 事件流 eval runner、trace 与 grader
├── src/xcode/experimental/ # 实验性扩展组件 (默认不激活，按需 opt-in)
│   ├── mcp.py              # Stdio MCP 传输、Schema 缓存与动态代理
│   ├── tasks.py            # 基于 JSON 文件和 filelock 锁的任务存储
│   ├── mailbox.py          # 基于 filelock 与事件日志型的多进程子代理邮箱
│   ├── memory.py           # 基于 H2 契约校验、BM25 召回、元数据重排和 eval 的 MEMORY.md 内存管理
│   ├── worktree.py         # Git Worktree 多任务沙箱物理隔离
│   ├── plugins.py          # 动态插件与 settings.json 沙箱安全策略
│   ├── bm25.py             # 共享的纯 Python BM25 召回核心算法
│   └── speculation.py      # 安全预热非副作用事件通知
├── skills/                 # 放置 SKILL.md 可装载外部技能库
├── src/xcode/tests/        # 单元测试套件
└── docs/                   # 项目深度设计文档、源码级审查与迁移审计
```

---

## 快速开始

Xcode 目录结构自聚，支持直接作为可独立分发的 editable package 引入项目：

### 1. 环境安装
```powershell
# 推荐使用虚拟环境安装项目依赖
.\.venv\Scripts\python.exe -m pip install -e .
```

### 2. 运行基础验证（无需 API Key）
```powershell
uv run python -m unittest discover src\xcode\tests
```
该指令校验核心装配、工具注册、运行时配置和 StructuredAgent 行为。

### 3. 单次运行真实模型
在系统环境变量中设置 `OPENAI_API_KEY`（或在项目根目录放置 `.env` 文件），通过 `--prompt` 执行一次单步结构化问答：
```powershell
.\.venv\Scripts\python.exe -m xcode.main --prompt "Xcode 中的安全沙箱是如何运行的？"
```

### 4. 运行交互式终端 REPL
直接以无子命令方式启动，即可进入基于 `prompt_toolkit` 的 REPL 界面：
```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

恢复最近会话时使用：
```powershell
.\.venv\Scripts\python.exe -m xcode.main --resume
```

---

## REPL 控制命令指南

在 REPL 交互中，系统提供了一组强大的控制斜杠命令（支持 `Tab` 键补全指令与本地路径）：

*   `@relative/path`：在任意用户输入中加上文件路径引用前缀，系统会自动读取对应的沙箱文件内容以 `<file-reference>` block 形式临时注入本轮 prompt 供 LLM 理解，原始消息不变。
*   `/plan`：切换为 **规划只读模式**。在该模式下，Agent 仅允许执行只读的词法检索和环境探索工具，任何写操作和 bash 命令均会被权限系统直接拦截，用于前期安全的方案设计。
*   `/review`：切换为 **审查验证模式**。在 `/plan` 与 `/act` 之间提供受控的折中，只读工具放行，编辑与测试验证工具要求明确的 HITL 审批。
*   `/act`：切换回 **正常执行模式**。高风险动作恢复 HITL 正常审批。
*   `/act --clear`：将当前在 `/plan` 中达成的计划自动提取并固化保存至 `.local/session_artifacts/plan-{id}.md` 路径中。系统会在 Git 上创建一个干净的并发分叉，并将 `<approved-plan>` 直接装配至新一轮的 prompt 顶部，提供高度纯净的任务实现环境。
*   `/compact`：随时请求手动消息压缩。这会提前收缩上下文，清理冗余的工具历史，保留已读文件的最新 state。
*   `/sessions` & `/resume`：`/sessions` 展示最近会话的短标题与摘要（默认记录于 `.local/session_index.json`）；使用 `/resume <session_id>` 或 `/resume last` 重新恢复对应历史会话。
*   `/tool NAME INPUT`：直通工具调试。不需要 LLM 参与，直接绕过状态机向注册的 ToolSpec 发送参数并接收输出，保留完整 HITL 网关。

---

## 运行时配置架构 (`xcode.config.json`)

系统参数与可选项全部通过项目根目录的 `xcode.config.json` 进行强类型托管。
`python -m xcode.main` 和 `build_real_app(project_root=...)` 会自动读取该文件；
只有需要使用其他配置文件时才传 `--config`。以下是生产推荐配置：

```json
{
  "provider": {
    "model_profiles": {
      "main": {
        "transport": "mimo_chat",
        "chat_model": "mimo-v2.5-pro",
        "base_url": "https://api.xiaomimimo.com/v1",
        "api_key": ""
      },
      "subagent": {
        "chat_model": "mimo-v2.5"
      },
      "fallback": {
        "chat_model": "mimo-v2.5"
      }
    }
  },
  "agent": {
    "max_steps": 20,
    "tool_workers": 4,
    "compact_threshold": 8,
    "compact_token_threshold": 12000,
    "max_recent_messages": 10,
    "watchdog_repeated_tool_limit": 3
  },
  "tools": {
    "enabled_groups": ["core", "mcp"],
    "bash": {
      "network_commands": "ask",
      "shell": "auto"
    }
  },
  "skills": {
    "auto_trigger": true
  },
  "prompt": {
    "modules": [
      "identity",
      "tool_discipline",
      "tools",
      "environment",
      "git_preflight",
      "cwd",
      "instructions",
      "notices"
    ]
  },
  "paths": {
    "sessions_dir": ".local/sessions",
    "skills_dir": "skills"
  },
  "observability": {
    "audit_path": ".local/audit.jsonl"
  }
}
```

---

## 编程式 API 集成与测试

### 1. 简易集成 build_real_app
```python
from pathlib import Path
from xcode.harness.app import build_real_app

# 自动从当前工作区寻找 xcode.config.json 并装配完整应用对象
app = build_real_app(
    project_root=Path.cwd(),
)

# 执行单步任务并获取答案
answer = app.ask("Find all Python test files and summarize their main checks.")
print(answer)
```

如果调用方已经运行在 event loop 中，使用原生异步入口：

```python
answer = await app.aask("Find all Python test files and summarize their main checks.")
```

### 2. 单元测试执行
运行项目下的完整测试：
```powershell
uv run python -m unittest discover src\xcode\tests
```

---

## 设计哲学

1.  **接口与实现彻底解耦 (Interface-First)**
    *   模型传输、上下文估算、审计拦截均采用统一协议与声明式 dataclass 配置。你可以轻松替换 LLM 适配器或自定义安全策略，而完全无需改动 StructuredAgent 运行主线。
2.  **安全默认且深入至进程级 (Safe-by-default)**
    *   Xcode 不仅对路径穿越做正则限制，其路径沙箱完全在 `file.py` 内实现 resolve 双重验证。
    *   Bash 执行并非通过 `shell=True` 直传系统，而是通过 ShellAdapter 分支隔离控制，拦截危险进程链并在外部通过 OS 级 `taskkill` 或进程组 `killpg` 提供确定的超时与终止防挂起设计。
3.  **高度的可测试性 (Hermetic Testing)**
    *   默认路径可以在没有 API 凭据和网络环境的情况下通过 Mock provider 与单元测试验证，覆盖 Harness 主循环、工具注册、权限边界和 Token 压缩边界。
