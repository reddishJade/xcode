# Xcode Eval 指南

所有命令默认从仓库根目录运行。

---

## Agent Eval Pipeline

`src/xcode/evals/` 包含 `EvalRunner`，消费 `XcodeApp.aask_stream()` 的事件流，记录 trace，并输出 `report.json` / `report.html` / `report.csv`。

`EvalTask` 支持：

- `prompt`：任务描述
- `mode`：`act` / `plan` / `review`
- `expected_answer_contains`：预期回答包含文本
- `expected_tool_calls` / `disallowed_tool_calls`：工具调用约束
- `max_tool_errors`：允许的工具错误次数
- `llm_judge_criteria`：LLM-as-judge 评判标准
- `metadata.evidence.files`：文件证据（`exists` / `contains` / `not_contains` / `changed`）
- `metadata.validation.commands`：验证命令，按退出码生成 grader
- `metadata.fixture_dir`：真实 provider eval 的 sandbox fixture 来源

内置 HumanEval 与 SWE-bench Lite JSON/JSONL benchmark loader：`src/xcode/evals/benchmarks.py`。通过 `--tasks` 参数加载自定义 JSONL，与内置套件共用 `EvalRunner` 和 grader 体系。

### 快速参考

| 命令 | 用途 |
|------|------|
| `uv run python -m xcode.evals.cli --list-suites` | 列出内置 suite |
| `uv run python -m xcode.evals.cli --show-suite coding-fixture` | 查看 suite 内任务和验证配置 |
| `uv run python -m xcode.evals.cli --suite pipeline` | eval pipeline 回归（离线） |
| `uv run python -m xcode.evals.cli --suite tool-policy` | 工具策略回归（离线） |
| `uv run python -m xcode.evals.cli --suite all --trials 3` | 默认离线回归集合，测量 pass@k/pass^k |
| `uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3` | 真实 provider 的 sandbox coding fixture 评测 |
| `uv run python -m xcode.evals.cli --list-benchmarks` | 列出外部 benchmark adapter 目标 |
| `uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-Mini.jsonl.gz --limit 1` | 公开 benchmark 小子集快速评测 |

### 可用套件

| 套件 | 任务数 | 侧重 |
|------|--------|------|
| `pipeline` | 1 | eval pipeline、事件流、report |
| `tool-policy` | 3 | 工具调用和禁止写入约束 |
| `coding-fixture` | 4 | 真实 provider、小型代码 fixture、validation command |
| `smoke` | 1 | 基础烟雾任务 |
| `tool` | 3 | 基础工具调用任务 |
| `context` | 1 | 上下文读取 |
| `multi` | 1 | 多步工具链 |
| `plan` | 1 | 规划 + 执行 |
| `all` | 6 | 默认离线回归集合 |

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--suite <name>` | 运行预定义套件 |
| `--list-suites` | 列出内置 suite |
| `--show-suite <name>` | 查看 suite 任务 |
| `--list-benchmarks` | 列出外部 benchmark adapter 目标 |
| `--tasks <path>` | 运行 JSON/JSONL 任务文件（与 `--suite` 互斥） |
| `--benchmark <name>` | 从公开或本地 benchmark 数据加载任务 |
| `--benchmark-path <path-or-url>` | benchmark 数据路径或 URL |
| `--limit N` | 限制加载的最大任务数 |
| `--real` | 使用真实 provider（否则用离线静态 provider） |
| `--trials N` | 每任务 N 轮（默认 1），用于 pass@k/pass^k 测量 |
| `--project-root <path>` | 项目根目录（默认当前目录） |
| `--output-dir <path>` | 输出目录（默认 `.local/eval_runs/`） |
| `--allow-project-mutation` | 允许真实 provider eval 使用 `--project-root` 作为写入目录 |

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
├── tiny-calculator-subtract-1.jsonl
├── fix-divide-by-zero-1.jsonl
└── sandboxes/
```

### 量化指标

| 指标 | 说明 |
|------|------|
| `pass@k` | N/M + 百分比：k 轮中至少一次正确（探索能力上限）。使用无偏估计量 `1 - C(n-c,k)/C(n,k)` |
| `pass^k` | N/M + 百分比：k 轮全部正确（回归稳定性） |
| `grader_pass_rate` | 所有 grader 的通过率 |
| `per_grader_pass_rate` | 每个 grader 维度（runtime_error、file_exists 等）的通过率 |
| `per_task_grader_rate` | 每个任务的 grader 通过率 |
| `total_llm_calls` | LLM 调用总数 |
| `total_estimated_tokens` | 估计 token 总数 |
| `total_model_ms` | 模型总延迟 |
| `total_tool_calls` | 工具调用总数 |
| `total_tool_errors` | 工具错误总数 |
| `model_patch` | Git 工作区补丁，供外部 benchmark harness 消费 |
| `*_distribution` | 各数值指标在 trials 间的分布（min/p50/p95/p99/max/mean） |

### 离线 Pipeline Eval

不需要 API key：

```powershell
uv run python -m xcode.evals.cli --suite pipeline
uv run python -m xcode.evals.cli --suite tool-policy
```

### 真实 Provider Coding Fixture

```powershell
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3
```

`coding-fixture` 会把 `examples/eval/fixtures/` 下的小型项目复制到 `.local/eval_runs/sandboxes/`。Agent 的写入发生在 sandbox 中，runner 会执行 `metadata.validation.commands` 并记录 validation 结果。

### 外部 Benchmark Adapter

查看当前 adapter 目标：

```powershell
uv run python -m xcode.evals.cli --list-benchmarks
```

当前 adapter registry 覆盖：

- `evalplus-humaneval`
- `evalplus-mbpp`
- `swebench-lite`
- `swebench-verified`
- `terminal-bench`
- `aider-polyglot`

adapter registry 描述 Xcode 与外部 harness 的职责边界。外部 harness 负责数据集、任务环境和评分；Xcode 负责在任务工作区内生成候选修改或执行终端任务。

EvalPlus loader 会从本地文件或公开 URL 读取 JSON/JSONL 数据并生成 sandbox fixture。每个任务包含 `solution.py`、`tests/test_solution.py` 和 validation command。

```powershell
uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus-Mini.jsonl.gz --trials 1
```

SWE-bench adapter helper 位于 `src/xcode/evals/adapters/swebench.py`。它把 `EvalReport` 和 benchmark tasks 转换为 predictions JSONL 记录，字段包含 `instance_id`、`model_name_or_path` 和 `model_patch`。

### 开发调试 JSONL

`--tasks <path>` 用于开发调试独立 task。JSON/JSONL 内容直接映射到 `EvalTask`。

```jsonl
{"id":"explain-tools","prompt":"用一句话说明 Xcode 的核心工具链。","expected_answer_contains":["read_file"],"tags":["smoke"]}
```

运行：

```powershell
uv run python -m xcode.evals.cli --tasks path/to/tasks.jsonl
```

真实 provider task 需要 `metadata.fixture_dir`，或显式使用 `--allow-project-mutation`。`report.json` 中的 `metrics.file_evidence` 会记录目标文件是否存在、hash 是否变化、指定文本是否出现。`report.csv` 可用于直接导入电子表格分析。
