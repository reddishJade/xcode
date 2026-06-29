# Xcode Eval 指南

所有命令从仓库根目录运行。

---

## Agent Eval Pipeline

`src/xcode/evals/` 包含 `EvalRunner`，消费 `XcodeApp.aask_stream()` 事件流，记录 trace，输出 `run_manifest.json` / `report.json` / `report.html` / `report.csv`。

`EvalTask` 支持：
- `prompt`、`mode`（act/plan/build）
- `expected_answer_contains`、`expected_tool_calls`、`disallowed_tool_calls`
- `max_tool_errors`、`llm_judge_criteria`
- `llm_judge_required`
- `metadata.evidence.files`（exists/contains/not_contains/changed）
- `metadata.validation.commands`（验证命令 grader）
- `metadata.fixture_dir`（真实 provider sandbox）
- `metadata.tool_policy`（顺序、参数、工具结果、结果采用等状态检查）
- `metadata.fault_injection`（离线故障注入脚本场景）
- `version`、`owner`、`capability`、`expected_duration_seconds`、`difficulty`、`run_mode`

说明：
- `expected_tool_calls`、`disallowed_tool_calls`、`max_tool_errors` 默认作为 trajectory / tool-policy 诊断 grader，不默认决定 `trial.success`
- 文件证据、验证命令、最终状态和 required judge 属于 outcome grader，会决定 `trial.success`

内置 HumanEval/EvalPlus/MBPP loader：`src/xcode/evals/benchmarks.py`。通过 `--tasks` 加载自定义 JSONL。

---

## 快速参考

| 命令 | 用途 |
|---|---|
| `uv run python -m xcode.evals.cli --list-suites` | 列出内置 suite |
| `uv run python -m xcode.evals.cli --show-suite coding-fixture` | 查看 suite 详情 |
| `uv run python -m xcode.evals.cli --suite pipeline` | eval pipeline 回归（离线） |
| `uv run python -m xcode.evals.cli --suite tool-policy` | 工具策略回归（离线） |
| `uv run python -m xcode.evals.cli --suite fault-injection` | 错误恢复 / 降级能力回归（离线） |
| `uv run python -m xcode.evals.cli --suite regression` | 默认稳定回归门禁集合 |
| `uv run python -m xcode.evals.cli --suite capability` | 能力边界探索集合 |
| `uv run python -m xcode.evals.cli --suite all --trials 3` | 默认离线回归集合 |
| `uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3` | 真实 provider sandbox 评测 |
| `uv run pytest -o addopts='' -m mcp_external src/xcode/tests/test_xcode_mcp_official_server.py -q` | 官方 MCP Everything server 外部回归 |
| `uv run python -m xcode.evals.cli --list-benchmarks` | 列出外部 benchmark adapter |
| `uv run python -m xcode.evals.cli --real --benchmark evalplus-humaneval --benchmark-path <url> --limit 1` | 外部 benchmark 小子集 |
| `uv run python -m xcode.evals.cli --suite regression --baseline .local/eval_runs/<baseline>` | 与 baseline 做回归对照 |

---

## 可用套件

| 套件 | 任务数 | 侧重 |
|---|---|---|
| `pipeline` | 1 | eval pipeline、事件流、report |
| `tool-policy` | 3 | 工具调用和禁止写入约束 |
| `fault-injection` | 3 | 命令失败重试、错误路径恢复、provider 中断降级 |
| `coding-fixture` | 4 | 真实 provider、sandbox fixture、validation command |
| `smoke` | 1 | 基础烟雾任务 |
| `tool` | 3 | 基础工具调用任务 |
| `context` | 1 | 上下文读取 |
| `multi` | 1 | 多步工具链 |
| `plan` | 1 | 规划 + 实现 |
| `regression` | 9 | 稳定回归门禁集合 |
| `capability` | 9 | 能力边界与真实编码探索 |
| `all` | 9 | 默认离线回归集合（pipeline + tool-policy + context + multi + fault-injection） |

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
| `--baseline <report-or-run-dir>` | 与 baseline report 比较 |
| `--fail-on-regression` | baseline 存在 regression 时返回非零 |
| `--max-p95-model-ms-growth N` | baseline 对比的 p95 model latency 增长预算 |
| `--max-avg-token-growth N` | baseline 对比的平均 token 增长预算 |
| `--max-avg-tool-calls-growth N` | baseline 对比的平均工具调用增长预算 |
| `--require-grader-pass <name>` | 指定 grader 在 candidate 中必须保持 100% 通过率 |
| `--limit N` | 限制加载的最大任务数 |
| `--real` | 使用真实 provider |
| `--trials N` | 每任务 N 轮（默认 1） |
| `--project-root <path>` | 项目根目录 |
| `--output-dir <path>` | 输出目录（默认 `.local/eval_runs/`） |
| `--allow-project-mutation` | 允许 eval 修改项目 |

---

## 输出格式

每个 run 输出四个主文件；当使用 `--baseline` 时还会额外输出 `baseline_diff.json`：
```text
.local/eval_runs/
├── run_manifest.json  # 轻量运行清单
├── report.json     # 全量结构化数据
├── report.html     # 可浏览 HTML
└── report.csv      # 表格格式
```

每条 trial 的 trace（JSONL）输出在同目录，sandbox 在 `sandboxes/` 下。
运行历史会在输出目录的父目录维护：
- `run_index.jsonl`
- `trend_summary.json`

---

## 量化指标

| 指标 | 说明 |
|---|---|
| `pass@k` | k 轮中至少一次正确（无偏估计量） |
| `pass^k` | k 轮全部正确（回归稳定性） |
| `grader_pass_rate` | 所有 grader 通过率 |
| `per_grader_pass_rate` | 每个 grader 维度通过率 |
| `per_task_grader_rate` | 每任务 grader 通过率 |
| `failure_category_pass_rate` | failure category 维度通过率，用于失败分类对照 |
| `grader_skipped_count` | 未执行或无法解析的 grader 数量；不计入通过率和 trial 成败 |
| `total_llm_calls` / `total_tool_calls` / `total_model_ms` | 用量统计 |
| `first_expected_tool_step` | 首次进入目标工具链的步数 |
| `repeated_tool_call_count` | 同名工具重复调用次数 |
| `unexpected_tool_call_count` / `unexpected_tool_call_rate` | 偏离声明工具链的调用数量与比例 |
| `model_patch` | Git 工作区补丁 |

---

## 离线 Pipeline Eval

```powershell
uv run python -m xcode.evals.cli --suite pipeline
uv run python -m xcode.evals.cli --suite tool-policy
uv run python -m xcode.evals.cli --suite fault-injection
```

---

## 真实 Provider Coding Fixture

```powershell
uv run python -m xcode.evals.cli --real --suite coding-fixture --trials 3
```

`coding-fixture` 把 `examples/eval/fixtures/` 下的小型项目复制到 sandbox。Agent 的写入在 sandbox 中进行。

## 官方 MCP Server 回归

默认 pytest 排除 `mcp_external`，避免网络和 npm 成为离线测试前提。显式运行：

```powershell
uv run pytest -o addopts='' -m mcp_external src/xcode/tests/test_xcode_mcp_official_server.py -q
```

该测试固定官方 `@modelcontextprotocol/server-everything@2026.1.26`，覆盖 stdio
发现、普通调用、structured content、progress notification 和关闭生命周期。
`tools/listChanged` 的可观察刷新行为由离线 SDK adapter conformance 测试覆盖；
Everything server 没有提供用于动态修改 tool catalog 的公开测试工具。

---

## 外部 Benchmark Adapter

```powershell
uv run python -m xcode.evals.cli --list-benchmarks
```

`--benchmark` 只接受当前 `integrated` registry 项：
- `humaneval`
- `swebench-lite`
- `evalplus-humaneval`
- `evalplus-mbpp`

`--list-benchmarks` 会同时列出三类目标：
- `integrated`：可直接通过 CLI `--benchmark` 运行
- `export-only`：只提供导出或桥接能力，不能直接运行
- `catalog-only`：仅作为目录声明，不接通本地 harness

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
`stream()` 协议发起独立 judge 请求。
- `llm_judge_required = false` 时，judge provider 不可用、调用失败或输出不可解析会记录 `llm_judge:skipped`，不影响 `trial.success`
- `llm_judge_required = true` 时，上述情况会记录失败的 `llm_judge:required` grader，并使 `trial.success = false`

`metadata.tool_policy` 目前支持的最小离线声明：
- `ordered_tools`
- `argument_contains`
- `result_contains`
- `answer_contains_from_tool`

这些 grader 默认是 diagnostic，不覆盖 outcome；若需要禁止副作用，应配合
`metadata.evidence.files` 或验证命令做 outcome 检查。
