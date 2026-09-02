# Benchmarks

## Long-horizon context-window rollover benchmark

This benchmark measures one controlled question: how much summary-free context
rollover reduces cumulative input tokens, cost, and interrupted long sessions
without lowering test-defined task success or state retention.

### Experimental groups

- **Baseline** sets runtime context rollover to `None`. It keeps full logical
  history and restores full history after a simulated process restart.
- **Xcode** opens a fresh active context at the declared boundary, appends a
  typed `context_window_reset` event, and restores the durable surface. No
  summary request is made.

All other runtime configuration is shared. Request hygiene remains enabled in
both groups, so the ablation isolates fresh-window rollover instead of
mixing it with transport sanitation. Web tools and subagents are prohibited by
a shared benchmark instruction. Automatic and model-initiated rollover are
disabled; the runner alone applies each task's declared boundary. Both groups
run in Build mode because the
non-interactive benchmark has no HITL approval callback; workspace writes and
verification commands therefore execute automatically under the same safety
boundaries. Repeat order alternates to reduce time-order bias.

Every copied fixture is initialized as a fresh, benchmark-owned Git repository
with a deterministic initial commit. The commit is identical across paired
runs, `git diff HEAD` therefore compares against the task fixture instead of a
parent checkout, and runtime artifacts such as `.benchmark/` and Python caches
are excluded from status output. Any fixture-provided `.git` metadata is not
copied.

Session transcript state lives in a sibling runtime directory rather
than inside the task workspace. The runner can still resume from it, while the
Agent's workspace-scoped tools cannot treat internal benchmark state as task
evidence. `--keep-workspaces` preserves both locations for debugging; normal
runs remove them after writing the raw record.

Provider `UsageUpdate` events are recorded for every agent request. A run is
excluded from token and cost aggregation if any provider request omits usage.
Known-model cost uses the price snapshot in
`src/xcode/ai/models.py` and is stored in every raw result.

### Run the paired example

Use the same explicit model configuration and temperature for every group:

```sh
uv run python -m benchmarks.runners.run_ablation \
  benchmarks/tasks/long_horizon/parser_recovery/task.json \
  --config xcode.config.json \
  --temperature 0 \
  --repeat 3 \
  --max-pair-attempts 2 \
  --require-complete-usage
```

This command makes real API calls. Raw JSON, `summary.json`, and `report.md` are
written below `benchmark-results/long_horizon/<timestamp>/`. Add
`--keep-workspaces` when a failed run needs manual inspection.

`--max-pair-attempts` retries both baseline and Xcode in fresh workspaces when
a transient provider error leaves usage incomplete. Every attempt remains in
raw JSON; the report selects the first complete pair and lists excluded
attempts with their reasons. The default is one attempt to avoid unexpected API
cost. `--require-complete-usage` (alias `--fail-on-incomplete`) writes the report
and then exits with status 2 if any selected pair still lacks complete usage.

Interactive terminals show an overall run bar plus the current task's turn,
model request, tool, context-window reset, restart, and verification status. During a
model call, the status distinguishes waiting for the first event from active
reasoning, answer streaming, tool calls, usage, and finalization. It also shows
the request number, elapsed time, time since the last event, and event count,
with a one-second heartbeat even when the provider is silent. Redirected output
and CI receive timestamped lines with model heartbeats limited to one every 30
seconds. Use `--no-progress` only when another process is supervising the
command.

Run groups separately when required:

```sh
uv run python -m benchmarks.runners.run_baseline TASK.json --repeat 3
uv run python -m benchmarks.runners.run_xcode TASK.json --repeat 3
```

Regenerate a report from existing raw records:

```sh
uv run python -m benchmarks.reports.generate_report \
  benchmark-results/long_horizon/RUN_DIR \
  --output-dir benchmark-results/long_horizon/RUN_DIR
```

### Task contract

Each `task.json` declares:

- an isolated fixture workspace;
- ordered user turns, including explicit rollover/restart boundaries;
- one test command that determines `task_success`;
- deterministic state facts such as changed/unchanged file hashes, forbidden
  paths, required text, and post-resume commands;
- fallback recent-message and recent-token budgets used when no active user
  turn can be identified.

The included parser task is a wiring example, not enough evidence for a resume
claim. A resume run counts only when `surface_resumes` is nonzero. Before
publishing results, add 20–30 tasks with at least 10 turns, run multiple repeats,
and inspect per-task pairs instead of reporting only a pooled mean.

### Reported metrics

- `input_tokens_total` and `peak_input_tokens` come from provider usage;
- `pre_rollover_input_tokens` covers turns before the declared rollover;
- `post_rollover_input_tokens` starts at the declared fresh-window turn;
- `post_resume_input_tokens` starts on the turn after the restart boundary;
- `task_success` comes only from the verification process exit code;
- `state_retention` is the fraction of deterministic facts that pass;
- `context_overflow` is detected from provider/runtime context-limit errors;
- `long_session_completed` requires tests, all turns, normal termination, and
  no context overflow;
- `repeated_read_calls` counts repeated reads of the same path as a diagnostic,
  not a success criterion.

Each metric uses its own paired cohort. Total Token and cost require complete
usage for the whole baseline/Xcode pair, while post-rollover metrics remain
eligible when missing usage occurred only before that phase. Correctness and
state-retention metrics include the selected attempt regardless of usage
completeness. Reports show the cohort size on every row.

Do not quote percentages from the example until enough paired task runs have
completed and `usage_complete` is true for the included samples.

## Tool scheduling benchmark

This deterministic ablation measures the production tool scheduler without a
model request. Both groups replay identical tool-call batches against isolated
workspace copies:

- **Serial** sets `AgentLoopConfig.tool_execution` to `sequential`;
- **Xcode** uses production parallel partitioning, the configured worker cap,
  and each tool's `parallel` or `sequential` side-effect classification.

The included workloads read 5, 10, and 20 distinct files. A mixed workload
adds real file writes between parallel-read batches to verify that writes never
overlap another operation. Every operation performs real local file I/O and a
declared controlled delay. The delay provides a reproducible I/O wait window;
it must not be described as model latency or end-to-end Agent latency.

Run the complete paired benchmark yourself:

```sh
uv run python -m benchmarks.runners.run_tool_scheduling \
  benchmarks/tasks/parallel_reads \
  --repeat 10 \
  --warmup 1
```

The command does not call a model or API. It alternates Serial/Xcode order,
shows live progress, writes every measured run immediately, and then produces
`summary.json` and `report.md` below
`benchmark-results/tool_scheduling/<timestamp>/`. Override the production-style
worker cap with `--workers N`, preserve isolated copies with
`--keep-workspaces`, or disable progress with `--no-progress`.

Regenerate a report from raw records:

```sh
uv run python -m benchmarks.reports.generate_tool_scheduling_report \
  benchmark-results/tool_scheduling/RUN_DIR
```

Performance cohorts require one successful Serial/Xcode pair, identical call
counts and result order, zero unsafe write overlap, matching tool-output hashes,
and matching final workspace hashes. Invalid pairs remain in raw results, are
listed under `excluded_pairs`, and make the runner exit with status 2 after the
report is written.

The primary report is per workload and includes tool-stage P50/P95 latency,
paired latency reduction, median speedup, and observed maximum concurrency.
Safety metrics include output/workspace equivalence and write isolation. Since
the benchmark deliberately excludes provider time, use its numbers for the
tool-execution stage only; a separate model-driven task suite is required for
an end-to-end Agent latency claim.

Run the worker-count sweep without overwriting earlier results:

```sh
./benchmarks/scripts/run_tool_worker_sweep.sh
```

It benchmarks `1`, `2`, `4`, `8`, and `16` workers in separate directories.
Optional positional arguments set `repeat`, `warmup`, and the output root:

```sh
./benchmarks/scripts/run_tool_worker_sweep.sh \
  10 1 benchmark-results/tool_scheduling/my-worker-sweep
```
