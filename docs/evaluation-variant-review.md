# Eval Variant 审计

本文件记录阶段 3 配对实验中 `full` 与 `minimal` 的真实执行差异。Variant 名称本身不构成
对照；Trial artifact 中的 capabilities 和完整脱敏 runtime 快照必须与这里一致。

## 保持不变的控制条件

- 同一 Task、初始 Git revision、隐藏 verifier 和允许修改路径；
- 同一 `xcode.config.json` 主 provider、模型、thinking 参数和 API endpoint；
- 同一 wall time、模型调用、工具调用和 token 预算；
- 同一 build 执行模式、bubblewrap 文件边界、非交互权限反馈和外部搜索 deny；
- 同一核心 coding tools、项目指令、context assembly、session、MCP 和 memory 装配；
- 同一 repetition 序号各自创建全新 workspace 与 session。

## 执行差异

| 能力 | `full` | `minimal` | 可执行证据 |
|---|---|---|---|
| compaction | 开启 | 关闭 | minimal 在 app 启动后清除 compactor 与 controller；不是只提高阈值。 |
| provider fallback | 开启 | 关闭 | minimal 的有效 provider profiles 删除 `fallback`，主 profile 不变。 |
| parallel tools | 最多使用配置 worker 数 | 单 worker | minimal 固定 `tool_workers=1`。 |
| request hygiene | 开启 | 关闭 | minimal 的正式 `RequestHygieneConfig.enabled=false`。 |
| repeated-tool watchdog | 开启 | 关闭 | minimal 固定 `watchdog_repeated_tool_limit=0`。 |
| contextual retrieval prompt | 开启 | 关闭 | minimal 从 prompt modules 删除 `git_preflight` 与 `contextual_retrieval`。 |

`minimal` 仍能读取、定位、编辑、执行测试并接收权限反馈，不是故意破坏的 Agent。当前任务
工作区没有外部 MCP 配置或既有跨 session memory，因此这两项不会伪装成 Variant 差异；
它们留给阶段 4 的真实压力任务独立消融。

阶段 4 首个单能力 Variant 为 `no-compaction`：它保留 provider fallback、并行工具、
request hygiene、repeated-tool watchdog 和 contextual retrieval，仅移除 compactor 与
compact controller。该 Variant 用于长上下文真实任务的 paired ablation，不与 `minimal`
的总体对照结果混合。

## 配对和报告规则

`full` 是 candidate，`minimal` 是 control。`harness_gain` 只在双方 Trial 都有效的严格
task/repetition pair 上计算：

```text
harness_gain = mean(success(full) - success(minimal))
```

报告同时保存 declared、observed、valid、invalid 和 missing pair 数，以及 candidate wins、
control wins、ties、input token/tool call/wall-time 差。任一方无效的 pair 不进入 gain，但
双方已经消耗的资源仍进入 Variant 总成本和已观察 pair 的成本差。
