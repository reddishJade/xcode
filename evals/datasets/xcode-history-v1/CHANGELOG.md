# xcode-history 数据变更记录

## xcode-history-v1（2026-07-15）

- 首次冻结三个经过人工审核的历史修复任务。
- 每个任务均确认父提交失败、修复提交通过，并在只应用生产 patch 的恢复工作区完成
  隐藏 verifier 参考重放。
- 排除公开的旧 Eval fixtures：它们是玩具项目，且验证材料对 Agent 可见。
- 排除纯 TUI 展示修复：当前无稳定、低成本且独立的终端视觉 oracle。
- 排除全仓历史测试作为统一 regression verifier：首个候选运行超过 300 秒；改为逐任务
  声明稳定回归切片，并在 `known_limitations` 保留覆盖缺口。
- 首次真实预运行在 20 steps 消耗 222,567 input tokens，暴露原 150,000 上限与声明的
  30-call 预算不一致。该 Trial 因超预算被排除且未形成基线；在首个有效基线前统一校准
  预算。第二次 30-step 权限诊断运行消耗 672,201 input tokens，据此最终冻结为
  30 model calls / 900,000 input tokens，并保留所有无效 Trial artifact 作为依据。
- 权限策略修正后的 30-call Trial 消耗 917,508 input tokens，证明 900,000 上限仍会
  把正常达到 step limit 的运行错误分类为预算无效。首个有效基线前再次统一校准为
  40 model calls / 1,600,000 input tokens / 40,000 output tokens；该上限用于完成阶段 1
  闭环，后续效率实验必须完整计入其高成本，不能把它当作推荐预算。
- `thinking` Task 明确状态 API 使用既有 `off` 词汇并禁止修改测试。这是用户可观察
  契约和允许路径的澄清，不包含参考 patch、符号名或隐藏 verifier 实现。
- 阶段 2 扩展加入 session reset 授权生命周期任务，覆盖临时授权、永久授权和非空历史
  恢复的差异。同一外置 verifier 在父版本行为 2 failed / 回归 2 passed，在修复版本和
  只应用允许生产 patch 的参考工作区均为行为 2 passed / 回归 2 passed。
- 阶段 2 加入 non-critical external observer 后台调度任务。同一外置 verifier 在父版本
  行为 2 failed / 回归 2 passed，在修复版本和只应用允许生产 patch 的参考工作区均为
  行为 2 passed / 回归 2 passed；旧同步完成断言因语义被有意改变而不作为回归项。
- 阶段 2 加入 MCP unknown exact override 诊断任务。同一外置 verifier 在父版本行为
  1 failed / 回归 2 passed，在修复版本和只应用允许生产 patch 的参考工作区均为行为
  1 passed / 回归 2 passed。
