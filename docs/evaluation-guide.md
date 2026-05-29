# Xcode 评估与验证指南

本指南介绍如何对 Xcode 核心框架、工具系统及交互层进行测试、评估与行为验证。本项目的原则是**“可测试性优先”**，默认路径不依赖真实网络 API 即可完成离线验证。

---

## 1. 离线单元验证

离线验证不再通过 CLI smoke 子命令提供，而是由单元测试覆盖核心装配、工具注册、运行时配置和 StructuredAgent 行为。

### 运行命令
```powershell
uv run python -m unittest discover src\xcode\tests
```

### 验证内容
- **命令行解析与分发**：验证 REPL、`--prompt` 和 `--resume` 主路径。
- **Agent 状态机循环**：检查 `StructuredAgent` 状态流，确保 `tool_use` 到 `tool_result` 轮次递进正常。
- **默认工具加载与隔离**：验证 `core` 工具组（包括文件、搜索、运行校验、Guarded Bash）的装配与访问权限。
- **本地环境拼装**：确保提示词中包含系统环境、git preflight 以及 cwd 等前缀块。

---

## 2. 交互式 REPL 验证 (Interactive REPL Verification)

REPL 是开发和调试的主要界面。它基于 `prompt_toolkit` 构建，提供了多模式切换、命令补全以及会话还原能力。

### 启动 REPL
```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

### 关键指令与交互验证流程
1. **获取帮助**：输入 `/help` 查看所有可用的 REPL 命令。
2. **多模式切换 (Plan / Review / Act)**：
   - 输入 `/plan` 切换为**规划模式**。此时仅暴露只读搜索与文件读取工具。
   - 输入 `/review` 切换为**审查模式**。此时可以运行局部测试、代码审查，但编辑工具和 bash 命令将触发人工确认审批边界。
   - 输入 `/act` 切换为**执行模式**。正常调用所有已启用的工具。
3. **Plan Exit 隔离执行**：
   - 在 `/plan` 模式制定好实施计划后，输入 `/act --clear`。
   - 验证上一轮生成的计划是否已自动落盘至 `.local/session_artifacts/plan-{id}.md`。
   - 验证 REPL 是否开启了干净的子会话，并以 `<approved-plan>` 将计划内容注入下一轮的 system prompt 中，而不在 messages 历史中堆积冗余的探索 transcript。
4. **手动压缩 (Compaction)**：
   - 输入 `/compact` 手动请求下一轮运行前压缩上下文。
   - 验证 stale 历史读取是否被自动剪切，并保持每个路径最新一次 `read_file` 内容。
5. **@file 文件引用**：
   - 输入 `@relative/path/to/file`（支持补全），验证文件内容是否被优雅地组装成 `<file-reference>` 块注入用户提问中。
6. **退出并归档**：
   - 输入 `/exit` 或 `Ctrl+D` 退出。会话标题和摘要将自动落盘至 `.local/session_index.json`，并打印 `Conversation saved: <title>`。
   - 再次启动时，使用 `.\.venv\Scripts\python.exe -m xcode.main --resume` 可恢复之前的会话。

---

## 3. 单元测试套件 (Unit Testing)

单元测试是保障系统稳定和重构安全的基石。项目中包含完整的核心测试与顶层行为测试。

### 运行核心测试
核心单元测试覆盖了 providers、agent_runtime、tools、observability、experimental 等模块的行为规范：
```powershell
uv run python -m unittest discover src\xcode\tests
```

### 运行顶层测试
```powershell
uv run python -m unittest discover src\xcode\tests
```

---

## 4. Agent Eval Pipeline

`src/xcode/evals/` 现在有两条线：

1. **工具 harness smoke**：`src/xcode/evals/eval_harness.py` 直接在临时 sandbox 里执行 `write_file`、`read_file`、`edit_file`、`glob_files`、`grep_search`、`bash`、`run_validation`，验证核心工具链是否能跑通。
2. **Agent event-stream eval**：`EvalRunner` 消费 `XcodeApp.aask_stream()` 的真实事件流，记录 trace，并生成 `report.json` / `report.html`。

Agent eval 不是只验证单个工具函数：

- `EvalTask` 声明 prompt、运行模式、期望答案片段、期望/禁止工具调用和允许的工具错误数。
- `EvalRunner.arun()` 是原生异步入口，消费 `XcodeApp.aask_stream()` 的事件流。
- `EvalRunner.run()` 是同步兼容入口；如果外部已经处在 event loop 中，应使用 `arun()`。
- `TraceRecorder` 将每次 trial 的 `text_delta`、`tool_use`、`tool_result`、`final` 等事件写入 JSONL trace。
- grader 目前覆盖最终事件、运行时错误、答案包含、工具调用约束和工具错误数。
- coding-task eval 可以在 `metadata.evidence.files` 中声明文件证据；runner 会在 trial 前后记录目标文件是否存在、哈希是否变化、指定文本是否出现，并把证据写入 `report.json` 的 `file_evidence` metrics。
- `report.json` 是机器可读汇总，`report.html` 是可直接打开的可视化报告。

### 离线 smoke eval

不需要 API key，默认跑内置的两个离线任务：一个纯回答任务，一个 tool-use 任务。

```powershell
uv run python -m xcode.evals.cli --output-dir .local/eval_runs/offline-smoke
```

输出：

```text
.local/eval_runs/offline-smoke/
├── offline-answer-1.jsonl
├── offline-tool-1.jsonl
├── report.json
└── report.html
```

在浏览器中打开 `report.html` 可以看到每个 trial 的 pass/fail、grader、metrics、trace 路径和最终回答。

### 真实 Agent eval

准备一个 JSONL 任务文件，每行一个 `EvalTask`：

```jsonl
{"id":"explain-tools","prompt":"用一句话说明 Xcode 的核心工具链。","expected_answer_contains":["read_file"],"tags":["smoke"]}
{"id":"no-shell-plan","prompt":"只规划，不要执行 shell。","mode":"plan","disallowed_tool_calls":["bash"],"tags":["policy"]}
```

运行真实 runtime：

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks .local/eval_tasks.jsonl --output-dir .local/eval_runs/real-smoke
```

如果已通过 editable install 安装，也可以使用脚本入口：

```powershell
xcode-eval --real --project-root . --tasks .local/eval_tasks.jsonl --output-dir .local/eval_runs/real-smoke
```

其中 `.local/eval_tasks.jsonl` 是调用方自备任务文件；如果只是想跑仓库内置示例，请使用下一节的 `examples/eval/coding-tasks.jsonl`。

### 最小代码修改 eval

仓库提供了一个小型真实 coding-task 任务包：

```text
examples/eval/coding-tasks.jsonl
examples/eval/fixtures/tiny-calculator/
```

任务会把 fixture 复制到本次 run 的 sandbox，再要求 Agent 给 `calculator.py` 增加 `subtract(left, right)` 并补充 unittest。这样 eval 可以观察真实读文件、编辑文件、运行验证的事件流，同时不修改主仓库。

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks examples/eval/coding-tasks.jsonl --output-dir .local/eval_runs/tiny-coding
```

输出目录会包含：

```text
.local/eval_runs/tiny-coding/
├── sandboxes/tiny-calculator-subtract-1/
├── tiny-calculator-subtract-1.jsonl
├── report.json
└── report.html
```

`report.json` 中每个 trial 的 `metrics.file_evidence` 会记录 `calculator.py` 和 `tests/test_calculator.py` 的存在性、哈希、`contains` 检查结果；对应 grader 会以 `file_exists:*`、`file_changed:*`、`file_contains:*` 形式进入 pass/fail 汇总。

### 针对性验证

运行特定的 eval 单元测试验证：
```powershell
uv run python -m unittest xcode.tests.test_eval_pipeline xcode.tests.test_eval_harness
```

---

## 5. 静态编译检查 (Compilation Check)

在完成任何文档或代码变更后，执行静态编译检查可以确保没有引入语法、拼写或模块导入上的致命错误。

### 运行命令
```powershell
uv run python -m compileall src
```
