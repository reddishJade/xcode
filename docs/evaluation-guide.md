# Xcode Eval 指南

所有命令从仓库根目录运行。

---

## Agent Eval Pipeline

`src/xcode/evals/` 包含 `EvalRunner`，消费 `XcodeApp.aask_stream()` 事件流，记录 trace，输出 `report.json` / `report.html` / `report.csv`。

`EvalTask` 支持：
- `prompt`、`mode`（act/plan/build）
- `expected_answer_contains`、`expected_tool_calls`、`disallowed_tool_calls`
- `max_tool_errors`、`llm_judge_criteria`
- `metadata.evidence.files`（exists/contains/not_contains/changed）
- `metadata.validation.commands`（验证命令 grader）
- `metadata.fixture_dir`（真实 provider sandbox）

内置 HumanEval/EvalPlus/MBPP loader：`src/xcode/evals/benchmarks.py`。通过 `--tasks` 加载自定义 JSONL。

---

## 快速参考

| 命令 | 用途 |
|---|---|
| `uv run python -m xcode.evals.cli --list-suites` | 列出内置 suite |
| `uv run python -m xcode.evals.cli --show-suite coding-fixture` | 查看 suite 详情 |
| `uv run python -m xcode.evals.cli --suite pipeline` | eval pipeline 回归（离线） |
| `uv run python -m xcode.evals.cli --suite tool-policy` | 工具策略回归（离线） |
| `uv run python -m xcode.evals.cli --suite all --trials 3` | 默认离线回归集合 |
| `uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3` | 真实 provider sandbox 评测 |
| `uv run python -m xcode.evals.cli --list-benchmarks` | 列出外部 benchmark adapter |
| `uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path <url> --limit 1` | 外部 benchmark 小子集 |

---

## 可用套件

| 套件 | 任务数 | 侧重 |
|---|---|---|
| `pipeline` | 1 | eval pipeline、事件流、report |
| `tool-policy` | 3 | 工具调用和禁止写入约束 |
| `coding-fixture` | 4 | 真实 provider、sandbox fixture、validation command |
| `smoke` | 1 | 基础烟雾任务 |
| `tool` | 3 | 基础工具调用任务 |
| `context` | 1 | 上下文读取 |
| `multi` | 1 | 多步工具链 |
| `plan` | 1 | 规划 + 实现 |
| `all` | 6 | 默认离线回归集合（pipeline + tool-policy + context + multi） |

---

## CLI 参数

| 参数 | 说明 |
|---|---|
| `--suite <name>` | 运行预定义套件 |
| `--list-suites` | 列出内置 suite |
| `--show-suite <name>` | 查看 suite 任务 |
| `--list-benchmarks` | 列出外部 benchmark adapter 目标 |
| `--tasks <path>` | 运行 JSON/JSONL 任务文件 |
| `--benchmark <name>` | 从公开或本地 benchmark 数据加载任务 |
| `--benchmark-path <path-or-url>` | benchmark 数据路径或 URL |
| `--limit N` | 限制加载的最大任务数 |
| `--real` | 使用真实 provider |
| `--trials N` | 每任务 N 轮（默认 1） |
| `--project-root <path>` | 项目根目录 |
| `--output-dir <path>` | 输出目录（默认 `.local/eval_runs/`） |
| `--allow-project-mutation` | 允许 eval 修改项目 |

---

## 输出格式

每个 run 输出三个文件：
```text
.local/eval_runs/
├── report.json     # 全量结构化数据
├── report.html     # 可浏览 HTML
└── report.csv      # 表格格式
```

每条 trial 的 trace（JSONL）输出在同目录，sandbox 在 `sandboxes/` 下。

---

## 量化指标

| 指标 | 说明 |
|---|---|
| `pass@k` | k 轮中至少一次正确（无偏估计量） |
| `pass^k` | k 轮全部正确（回归稳定性） |
| `grader_pass_rate` | 所有 grader 通过率 |
| `per_grader_pass_rate` | 每个 grader 维度通过率 |
| `per_task_grader_rate` | 每任务 grader 通过率 |
| `grader_skipped_count` | 未执行或无法解析的 grader 数量；不计入通过率和 trial 成败 |
| `total_llm_calls` / `total_tool_calls` / `total_model_ms` | 用量统计 |
| `model_patch` | Git 工作区补丁 |

---

## 离线 Pipeline Eval

```powershell
uv run python -m xcode.evals.cli --suite pipeline
uv run python -m xcode.evals.cli --suite tool-policy
```

---

## 真实 Provider Coding Fixture

```powershell
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3
```

`coding-fixture` 把 `examples/eval/fixtures/` 下的小型项目复制到 sandbox。Agent 的写入在 sandbox 中进行。

---

## 外部 Benchmark Adapter

```powershell
uv run python -m xcode.evals.cli --list-benchmarks
```

CLI `--benchmark` 支持：humaneval、swebench-lite、swebench-verified、evalplus-humaneval、evalplus-mbpp、terminal-bench、aider-polyglot。
完整列表以 `--list-benchmarks` 为准。

Benchmark adapter helpers 位于 `src/xcode/evals/adapters/`。

---

## 开发调试 JSONL

```jsonl
{"id":"explain-tools","prompt":"用一句话说明 Xcode 的核心工具链。","expected_answer_contains":["read_file"],"tags":["smoke"]}
```

```powershell
uv run python -m xcode.evals.cli --tasks path/to/tasks.jsonl
```

真实 provider task 需要 `metadata.fixture_dir` 或 `--allow-project-mutation`。

配置 `llm_judge_criteria` 时，runner 通过当前 agent provider 的标准
`stream()` 协议发起独立 judge 请求。judge provider 不可用、调用失败或输出
不可解析时，JSON/HTML/CSV report 会记录 `llm_judge:skipped`，该项不计入
grader 通过率或 trial 成败。
