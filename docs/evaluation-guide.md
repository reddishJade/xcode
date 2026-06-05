# Xcode 评估与验证指南

本指南记录当前 checkout 的测试、编译和 eval 工作流。所有命令默认从 `D:\WorkSpace\claude\minimal\xcode` 运行。

---

## 1. 单元测试

完整测试：

```powershell
uv run python -m unittest discover src\xcode\tests
```

测试覆盖范围：

- runtime config loading
- app assembly and tool group gating
- `StructuredAgent` loop
- provider codecs and stream adapters
- file/search/bash tools
- permission policy and audit redaction
- REPL/session behavior
- compaction and read-version restoration
- subagent runner
- experimental modules
- eval pipeline

针对性测试示例：

```powershell
uv run python -m unittest src.xcode.tests.test_xcode_app_runtime
uv run python -m unittest src.xcode.tests.test_xcode_mcp_client
uv run python -m unittest src.xcode.tests.test_xcode_mailbox src.xcode.tests.test_xcode_progress
uv run python -m unittest src.xcode.tests.test_eval_pipeline
```

---

## 2. 格式、Lint 和类型检查

修改 Python 文件后，按 `docs/code-standards.md` 的 Python 验证命令对修改文件运行。

修改多个文件时把文件列表显式传入命令。不要默认扩大到全仓库，除非任务需要。

---

## 3. 编译检查

检查包内 Python 文件语法和导入形状：

```powershell
uv run python -m compileall src
```

---

## 4. REPL 验证

启动 REPL：

```powershell
.\.venv\Scripts\python.exe -m xcode.main
```

常用交互验证：

- `/help`：查看 REPL 命令。
- `/plan`：切换只读规划模式。
- `/review`：切换审查模式，写入和 bash 仍走审批边界。
- `/act`：切换执行模式。
- `/act --clear`：保存 plan artifact，并进入干净会话。
- `/compact`：请求下一轮前压缩上下文。
- `/sessions`：列出会话。
- `/resume <id>`：恢复会话。
- `/tool NAME INPUT`：直接调用当前 registry 中的工具。
- `!COMMAND`：通过已注册的 `bash` 工具直接运行 shell 命令。
- `@relative/path`：注入文件引用。

恢复最近会话：

```powershell
.\.venv\Scripts\python.exe -m xcode.main --resume
```

---

## 5. Tool Group 验证

默认工具组只应暴露 core tools。相关测试：

```powershell
uv run python -m unittest src.xcode.tests.test_xcode_app_runtime
```

该测试覆盖：

- 默认不构造 optional/experimental groups。
- 单独启用 group 时只加入对应工具。
- `experimental` 总开关会展开所有 experimental group。
- `mailbox` 和 `progress` group 会注册 Agent 可调用工具。
- subagent 只继承已启用工具。

---

## 6. Agent Eval Pipeline

`src/xcode/evals/` 包含 `EvalRunner`，消费 `XcodeApp.aask_stream()` 的事件流，记录 trace，并输出 `report.json` / `report.html` / `report.csv`。

`EvalTask` 支持：

- `prompt`：任务描述
- `mode`：`act` / `plan` / `review`
- `expected_answer_contains`：预期回答包含文本
- `expected_tool_calls` / `disallowed_tool_calls`：工具调用约束
- `max_tool_errors`：允许的工具错误次数
- `llm_judge_criteria`：LLM-as-judge 评判标准
- `metadata.evidence.files`：文件证据（`exists` / `contains` / `not_contains` / `changed`）

### 快速参考

| 命令 | 用途 |
|------|------|
| `uv run python -m xcode.evals.cli --suite smoke` | 烟雾测试（离线） |
| `uv run python -m xcode.evals.cli --suite tool` | 工具调用测试（离线） |
| `uv run python -m xcode.evals.cli --suite all --trials 3` | 全套 3 轮，测量 pass@k/pass^k |
| `uv run python -m xcode.evals.cli --real --suite coding --trials 3` | 真实 provider 的编码评测 |
| `uv run python -m xcode.evals.cli --real --suite tool` | 真实 provider 的工具调用 |

### 可用套件

| 套件 | 任务数 | 侧重 |
|------|--------|------|
| `smoke` | 1 | 基础 ReAct 循环 |
| `tool` | 3 | 工具调用正确性 |
| `context` | 1 | 上下文读取 |
| `multi` | 1 | 多步工具链 |
| `coding` | 3 | 编码能力（写/改/修） |
| `plan` | 1 | 规划 + 执行 |
| `all` | 10 | 全部 |

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--suite <name>` | 运行预定义套件 |
| `--tasks <path>` | 运行自定义 JSON/JSONL 任务（与 `--suite` 互斥） |
| `--real` | 使用真实 provider（否则用离线静态 provider） |
| `--trials N` | 每任务 N 轮（默认 1），用于 pass@k/pass^k 测量 |
| `--project-root <path>` | 项目根目录（默认当前目录） |
| `--output-dir <path>` | 输出目录（默认 `.local/eval_runs/`） |

### 输出格式

每个 run 输出三个文件：

```text
.local/eval_runs/
├── report.json              # 机器可读（全量结构化数据）
├── report.html              # 可浏览 HTML（含分布表、grader 柱状图）
└── report.csv               # 表格格式（每 trial 一行 + 汇总）
```

每条 trial 的 trace（JSONL）也输出在同目录：

```text
├── write-python-function-1.jsonl
├── write-python-function-2.jsonl
└── write-python-function-3.jsonl
```

### 量化指标

| 指标 | 说明 |
|------|------|
| `pass@k` | N/M + 百分比：k 轮中至少一次正确（探索能力上限） |
| `pass^k` | N/M + 百分比：k 轮全部正确（回归稳定性） |
| `grader_pass_rate` | 所有 grader 的通过率 |
| `per_grader_pass_rate` | 每个 grader 维度（runtime_error、file_exists 等）的通过率 |
| `per_task_grader_rate` | 每个任务的 grader 通过率 |
| `total_llm_calls` | LLM 调用总数 |
| `total_estimated_tokens` | 估计 token 总数 |
| `total_model_ms` | 模型总延迟 |
| `total_tool_calls` | 工具调用总数 |
| `total_tool_errors` | 工具错误总数 |
| `*_distribution` | 各数值指标在 trials 间的分布（min/p50/p95/p99/max/mean） |

### 离线 smoke eval

不需要 API key：

```powershell
uv run python -m xcode.evals.cli --suite smoke
```

### 真实 Agent eval（自定义任务）

使用调用方自备 JSONL：

```jsonl
{"id":"explain-tools","prompt":"用一句话说明 Xcode 的核心工具链。","expected_answer_contains":["read_file"],"tags":["smoke"]}
{"id":"no-shell-plan","prompt":"只规划，不要执行 shell。","mode":"plan","disallowed_tool_calls":["bash"],"tags":["policy"]}
```

运行：

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks .local/eval_tasks.jsonl
```

如果已安装 console script：

```powershell
xcode-eval --real --project-root . --tasks .local/eval_tasks.jsonl
```

### 仓库内置 coding-task eval

示例任务和 fixture：

```text
examples/eval/coding-tasks.jsonl
examples/eval/fixtures/tiny-calculator/
examples/eval/fixtures/buggy-math/
examples/eval/fixtures/string-utils/
```

运行：

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks examples/eval/coding-tasks.jsonl
```

`report.json` 中的 `metrics.file_evidence` 会记录目标文件是否存在、hash 是否变化、指定文本是否出现。`report.csv` 可用于直接导入电子表格分析。

---

## 7. 文档变更验证

只改文档时，至少运行：

```powershell
git diff --check -- README.md CONFIG.md docs\code-organization.md docs\source-review.md docs\evaluation-guide.md
```

如果文档涉及命令、模块路径或工具组，应同时运行相关 targeted tests。例如工具组文档变更：

```powershell
uv run python -m unittest src.xcode.tests.test_xcode_app_runtime
```

涉及 eval 文档变更：

```powershell
uv run python -m unittest src.xcode.tests.test_eval_pipeline
```

---

## 8. 推荐提交前检查

针对代码和文档混合变更：

```powershell
# Run the Python validation commands from docs/code-standards.md on modified Python files.
uv run python -m unittest <targeted-test-modules>
git diff --check -- <modified-files>
```

只提交任务相关文件，保留其他会话或用户已有改动。
