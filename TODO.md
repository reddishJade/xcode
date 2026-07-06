# Xcode TODO

> 来自全层设计回顾中识别的问题和待办。
> 每个条目标注了问题层级、严重程度、当前代码位置（如果明确）。

---

## 优先级说明

| 级 | 含义 |
|----|------|
| **P0** | 设计缺陷，影响正确性或安全性 |
| **P1** | 严重问题，当前行为可接受但必须改 |
| **P2** | 中等重要 |
| **P3** | 值得改进的设计细节 |

---

## P0 — 设计缺陷

### Memory 系统缺少 LLM 介入——纯规则提取无法判断"什么值得记住"

`consolidate_structured()` 从压缩摘要的 Key Decisions 节提取列表项填模板，写入 MEMORY.md。没有 LLM 参与判断"这个决策值得记录吗"、"用什么样的表述最适合未来检索"。检索只有 BM25，没有语义理解。没有反馈闭环验证注入的记忆是否被模型使用。

- **位置**：`harness/memory/manager.py` → `consolidate_structured()`, `_decision_to_memory_block()`
- **检索**：`harness/memory/manager.py` → `search_memory_records()` → BM25 + 重排
- **建议方向**：
  - 写入端：small LLM 判断"这个决策是否值得记录"，结构化输出（scope / related_files / confidence）
  - 检索端：语义 embedding + BM25 混合检索
  - 消费端：注入后在 message 中标记来源，compaction 时检测"注入的记忆是否被引用"，反馈给 MemoryManager 做 utility 更新

---

## P1 — 严重问题（行为可接受，但必须改）

### HITL 缺少超时和预览——Deny 应携带 suggestion

- 弹窗没有超时——用户离开终端 agent 永远等下去
- 预览只有 `brief_input`（一行摘要），不显示文件 diff、命令预期效果、多工具执行计划
- Deny 只有 reason，没有 suggestion 引导模型替代操作

- **位置**：`cli/repl_hitl.py` → `ReplHITLHandler`
- **建议方向**：
  - 添加可配置超时（默认 300s）→ 超时自动 deny
  - `BeforeToolCallResult` 添加 `suggestion` 字段，携带"不能做 X，请做 Y 替代"
  - 预览层展示 edit_file diff、bash 命令解析、多工具的上下文

---

### Compaction `summarize_fn` 未接入——摘要生成是纯规则的

`LayeredCompactor` 创建时没有传入 `summarize_fn`，`summarize_messages()` 走 `_fallback_structured_summary()` 纯规则分支。不调用 LLM，token 成本为零，但摘要质量受限：只提取消息角色、工具名、预定义节关键词，没有语义理解。

与 Memory 的 `consolidate_structured()` 问题同源——LLM 介入的接口预留了但没落地。

- **位置**：`harness/assembly.py` → `build_shared_infra()` → `LayeredCompactor()` 构造
- **建议方向**：
  - 接入 `summarize_fn`：复用 provider 或使用独立的 cheap model（如 deepseek-chat）生成结构化摘要
  - `summarize_messages()` 中已有 LLM 路径的完整骨架（`if summarize_fn: raw = summarize_fn(older)`），只需传入即可
  - 需评估：每次 compaction 的 LLM 调用 token 成本 vs 摘要质量提升的收益

---

### `/plan` `/build` `/act` 模式重构——build 应中间态

当前问题：
- build 和 act 在权限层面完全一样（`BuildPolicy.check_call()` 和 `ActPolicy.check_call()` 都返回 `"allow"`），区别仅在于工具可见性集合
- plan 使用硬编码工具名集合（`PLAN_TOOL_NAMES`），新增 MCP 工具需要手动加入
- SafetyBackstop 的 Bucket C 粒度太粗（`git` 整体放行，不识别 `--force`），不能直接作为 build 默认放行的依据

目标状态：
- **build** = 中间态：write_file/edit_file 默认 allow（无 HITL），shell 命令基于结构化分析分层（Bucket C allow / Bucket B ask / Bucket A deny），MCP 工具不可见
- **plan** = 权限驱动的工具可见性（借鉴 OpenCode 方案，用 deny/edit:deny 推导工具可见性，而非硬编码集合）
- **act** = 全部工具可见，走完整权限系统

前置依赖：SafetyBackstop 需先完成结构化 flag 识别（`--force` 等），build 的 check_call 才能消费 Bucket verdict。

- **位置**：`harness/agent_runtime/execution_modes.py` → `BuildPolicy`, `PlanPolicy`, `ActPolicy`
- **位置**：`harness/observability/_safety_backstop.py` → 结构化 flag 识别

---

## P2 — 中等重要

### 重复看门狗 + 空闲看门狗可能互相干扰

当工具连续失败时（如磁盘满→write_file 持续失败），重复看门狗（3 次）在空闲看门狗（4 次）之前触发，但终止信息说"重复工具调用"而不是"工具持续失败"——掩盖了根因。

- **位置**：`agent/watchdog.py` → `update_repeated_tool_watchdog()`, `update_idle_tool_watchdog()`
- **建议方向**：重复看门狗只在"工具成功但模型还在重复调用"时触发。如果连续相同的调用每次都失败，归给空闲看门狗。

---

### Shell 命令 grant matching 是精确匹配而非模式匹配

`_grant_matches_target()` 对 shell 命令使用 `record.target_pattern == target.value`（精确字符串）。而 ShellAnalyzer 已经产出了结构化表示（command、subcommand、flags、args、targets），但没有被 GrantStore 消费。

- **位置**：`harness/observability/permission_model.py` → `_grant_matches_target()`
- **建议方向**：基于 ShellAnalyzer 的结构化产出做模式匹配。用户批准 `git add *` 就覆盖所有 `git add <path>` 调用，但 `git push --force` 不匹配，需重新走策略。

---

### SafetyBackstop 未识别危险 flag——git 整体放行

`git` 被整体归为 Bucket C（allow），但 `git push --force`、`git reset --hard` 等风险操作没有被独立识别。

- **位置**：`harness/observability/_safety_backstop.py` → `BUCKET_C_ALLOW_COMMANDS`
- **建议方向**：在 ShellAnalyzer 解析出 flag 后，SafetyBackstop 对 `--force`、`--hard`、`--destroy` 等危险 flag 在任何命令上触发 ask 或 deny。这和上一条共享"ShellAnalyzer 结构化产出 → 消费者"的基础设施。

---

### fork_type 语义未落地——explore 和 verify 行为无区别

Fork 类型在 `session_index.json` 中记录了 `fork_type` 字段，但没有任何运行时逻辑根据 `fork_type` 改变行为。explore 和 verify 都复制完整 transcript，行为一致。

- **位置**：`harness/session.py` → `FORK_TYPES`, `_fork_base()`
- **建议方向**：要么注入运行时语义（explore 可写、verify 只读），要么移除未使用的 fork_type 值。

---

## P3 — 值得改进的设计细节

### `/act --clear` 语义不明确

是 mode switch 解耦后的残余接口。mode switch 不再重置状态，但 `--clear` 作为独立 flag 保留下来，语义既不 mode switch 也不全量重置。

- **位置**：`cli/repl_commands.py` → 搜索 `--clear`
- **建议**：移除或重命名为语义更清晰的操作。

---

### `watchdog_repeated_tool_skip` 豁免列表配置安全风险未评估

当前为空。设计意图是允许低风险只读工具（如 `list_dir`、`search_tools`）不受重复检测限制。但需要评估：如果某个工具被加入豁免列表，而权限系统对同一工具有独立的 deny/allow 规则，两者的优先级和交互可能导致意外的行为（如权限系统 deny 了该工具，但豁免列表让它继续重复调用）。

**风险**：可能引入过度设计的双层决策引擎。接入前需要先定义：权限 deny 是否可以 override 看门狗豁免？

- **位置**：`agent/config.py` → `AgentLoopConfig.watchdog_repeated_tool_skip`
- **建议**：先明确看门狗豁免和权限系统的交互规则，再启用。

---

### execution_mode 无任何工具显式设置

当前所有工具的 execution_mode 通过 `read_only + concurrency_safe → parallel` 派生。没有工具主动声明 `execution_mode="parallel"`。

- **位置**：`coding_agent/tools/file.py`、`bash.py`、`grep_search.py` 等→ ToolSpec.execution_mode
- **建议**：可选——显式声明可以让设计意图更清晰，但派生规则已足够，非紧迫。

---

### blinker HookManager 同步阻塞

`Signal.send()` 是同步阻塞的。如果 post_tool 订阅者做了 IO（如写入远程日志），它会阻塞工具执行流。

- **位置**：`harness/observability/hooks.py` → `HookManager.emit()`
- **建议**：引入异步队列或线程池解耦。

---

### `non_bypassable` 存在但缺少运行时行为差异

`non_bypassable` 当前在两个层面被活跃使用：

**生产者**：
- `SafetyBackstopPolicyEvaluator` → Bucket A 全部标记 `non_bypassable=True`（rm -rf /、.ssh 凭据、dd 写设备、apt 等）
- `PathBoundaryPolicyEvaluator` → 路径解析失败的 write/execute、`.git` metadata 写操作 → `non_bypassable=True`

**消费者**：
- `PermissionResolver._winning_constraint()` → `non_bypassable=True` 的 deny **优先于**普通 deny 被选中
- `PermissionEngine._decide_resolver()` → 在结果 metadata 中标记 `{"non_bypassable": True}`，告知调用方此拒绝不可绕过

**问题**：在运行时行为层面，non_bypassable deny 和普通 deny 没有实际差异——当前没有任何路径能把普通 deny 升为 allow。如果将来引入"管理员可 override 普通 deny"的机制，non_bypassable 就有区分意义了。在此之前它是一个语义标记，不是功能开关。

- **位置**：`harness/observability/_safety_backstop.py`（生产者）、`harness/observability/permission_model.py` → `Constraint`、`PermissionResolver`（消费者）
- **建议**：明确文档说明当前 non_bypassable 是保守预留——有语义标记无行为差异。

---
