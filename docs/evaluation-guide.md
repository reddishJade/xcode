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
uv run python -m unittest src.xcode.tests.test_eval_pipeline src.xcode.tests.test_eval_harness
```

---

## 2. 格式、Lint 和类型检查

修改 Python 文件后，按 AGENTS 规则对修改文件运行：

```powershell
uv run ruff format path\to\file.py
uv run ruff check path\to\file.py
uv run ruff format --check path\to\file.py
uv run mypy path\to\file.py
```

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

`src/xcode/evals/` 包含两类 eval：

1. `eval_harness.py`：直接验证 core tools 的 smoke harness。
2. `EvalRunner`：消费 `XcodeApp.aask_stream()` 的事件流，记录 trace，并输出 `report.json` / `report.html`。

`EvalTask` 支持：

- prompt
- mode
- expected answer contains
- required / disallowed tool calls
- allowed tool error count
- metadata file evidence

### 离线 smoke eval

不需要 API key：

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

### 真实 Agent eval

使用调用方自备 JSONL 任务：

```jsonl
{"id":"explain-tools","prompt":"用一句话说明 Xcode 的核心工具链。","expected_answer_contains":["read_file"],"tags":["smoke"]}
{"id":"no-shell-plan","prompt":"只规划，不要执行 shell。","mode":"plan","disallowed_tool_calls":["bash"],"tags":["policy"]}
```

运行：

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks .local/eval_tasks.jsonl --output-dir .local/eval_runs/real-smoke
```

如果已安装 console script：

```powershell
xcode-eval --real --project-root . --tasks .local/eval_tasks.jsonl --output-dir .local/eval_runs/real-smoke
```

### 仓库内置 coding-task eval

示例任务和 fixture：

```text
examples/eval/coding-tasks.jsonl
examples/eval/fixtures/tiny-calculator/
```

运行：

```powershell
uv run python -m xcode.evals.cli --real --project-root . --tasks examples/eval/coding-tasks.jsonl --output-dir .local/eval_runs/tiny-coding
```

输出目录包含 sandbox、trace、`report.json` 和 `report.html`：

```text
.local/eval_runs/tiny-coding/
├── sandboxes/
├── tiny-calculator-subtract-1.jsonl
├── report.json
└── report.html
```

`report.json` 中的 `metrics.file_evidence` 会记录目标文件是否存在、hash 是否变化、指定文本是否出现。

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
uv run python -m unittest src.xcode.tests.test_eval_pipeline src.xcode.tests.test_eval_harness
```

---

## 8. 推荐提交前检查

针对代码和文档混合变更：

```powershell
uv run ruff format <modified-python-files>
uv run ruff check <modified-python-files>
uv run ruff format --check <modified-python-files>
uv run mypy <modified-python-files>
uv run python -m unittest <targeted-test-modules>
git diff --check -- <modified-files>
```

只提交任务相关文件，保留其他会话或用户已有改动。
