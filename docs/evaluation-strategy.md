# Xcode Eval 战略

## 文档职责

本文定义 Xcode Eval 的长期目标、评测边界和测量契约，是 Eval 设计与实现的
最高层依据。阶段安排与当前进度记录在
[evaluation-roadmap.md](evaluation-roadmap.md)。

当阶段实现、旧代码或外部 benchmark 与本文冲突时，应先修正实现或路线；只有
Xcode 的产品定位发生变化时，才修改本文。

## 核心定义

Xcode Eval 评测的对象是 **coding-agent harness**，不是底层模型，也不是 Eval
代码自身。

Xcode 为模型提供理解和解决真实软件工程问题所需的“手脚”：上下文组织、工具、
执行环境、权限反馈、会话、压缩、错误恢复、MCP、记忆以及任务推进策略。Eval 要
回答的是：

> 在模型、任务和预算保持可比的条件下，Xcode 能否让模型更可靠、更高效地解决
> 真实问题；具体是哪项 harness 能力产生了多少增益，又付出了多少成本？

一次有效评测必须让真实 Xcode 在真实工作区中运行，实际读取和修改源码、执行
工具并留下可验证的结果。Agent 的最终回答、内部判断和自行执行的测试只属于运行
轨迹，不能单独作为成功证据。

## Eval 与 Test 的边界

Test 验证 Xcode 的实现是否符合代码契约，例如：

- pipeline 是否按顺序产生事件；
- CLI 参数、序列化和报告渲染是否正确；
- grader、指标公式和 telemetry collector 是否正确计算；
- sandbox、权限、工具和 provider adapter 是否符合接口约定；
- fake provider 场景是否覆盖预期分支。

Eval 衡量真实 Xcode 解决问题的表现，例如：

- 是否产出了通过独立验证的实现；
- 是否破坏原有行为或越过任务约束；
- 多次运行是否稳定；
- 工具、token、时间和模型调用是否有效转化为成功结果；
- 关闭某项 harness 能力后，成功率、稳定性或成本如何变化；
- 面对长上下文、工具失败、权限限制和外部能力不可用时能否恢复。

任何只证明 Eval pipeline 能运行的场景都属于 Test，不得进入 Xcode 能力得分。
Eval 基础设施本身仍必须由 pytest 覆盖，但这些测试结果不得被包装成 Eval 分数。

## 长期目标

Xcode Eval 的终局不是单一排行榜，而是一套可复现的实验系统，能够持续回答以下
问题：

1. Xcode 在目标任务分布上能解决多少真实问题？
2. 相比最小 harness 或上一稳定版本，Xcode 带来了多少净增益？
3. 上下文、工具、恢复、权限、会话、MCP 和记忆等能力分别贡献了什么？
4. 每次成功消耗多少 token、模型调用、工具调用、时间和费用？
5. 结果是否稳定，失败来自 Agent、环境、grader 还是任务数据？
6. 一次代码变更改善了什么，又让什么发生了退化？
7. 结论能否由保存的任务、配置、源码版本、trace、patch 和 verifier 证据重放？

## Harness 边界与能力地图

“harness”在不同语境中可能同时指模型外的软件运行骨架，以及项目团队围绕 coding
agent 建立的规则、脚本和质量反馈。Xcode Eval 必须固定边界，避免把环境质量误算
成 Xcode 自身能力：

- **模型**不属于 Xcode harness，是受控的实验变量；
- **Xcode builder harness** 是主要评测对象，包括 prompt 与上下文装配、工具暴露与
  执行、agent loop、状态、压缩、权限、恢复、MCP、memory 和 delegation；
- **项目 outer harness** 包括 AGENTS.md、skills、linters、测试、架构规则和项目专用
  脚本，是 Task 环境的一部分；
- **Eval harness** 负责数据、隔离、调度、采集、验证和聚合，不属于被评分对象。

项目 outer harness 会影响任务可解决性。Task 必须记录可用的 guides 与 sensors，
配对 Trial 必须保持它们一致。只有当实验显式研究 Xcode 发现、选择或使用这些能力
的方式时，才能把差值归因到 Xcode。

Xcode builder harness 的长期测量面包括：

1. **实时工作区理解**：仓库状态、项目指令、源码和变更是否被正确发现与更新；
2. **上下文装配与效率**：信息选择、稳定前缀、缓存、裁剪、去重和压缩；
3. **工具与执行环境**：工具描述、参数验证、文件编辑、shell、sandbox 和结果反馈；
4. **循环与控制**：继续、停止、预算、验证反馈、错误恢复和防止无效重复；
5. **持久状态**：完整 trace、working memory、session 恢复和跨任务边界；
6. **扩展与编排**：skills、MCP、subagent、并行工作和上下文隔离；
7. **安全与约束**：权限反馈、路径边界、风险控制和受限条件下的替代路径。

能力面用于解释“Xcode 提供了什么手脚”，不能直接成为分数。每项能力最终仍要通过
真实任务结果、成本或鲁棒性证明价值。

任务结果还应按被调节的代码目标分层：

- **行为正确性**：软件是否实现用户需要的功能；
- **可维护性**：实现是否避免明显结构退化、重复和不可接受的复杂度；
- **架构适应度**：实现是否保持模块边界、性能、安全和可观测性等架构约束。

这三类目标的可验证程度不同。行为正确性通常最难拥有完整 oracle，不得因为测试
通过就夸大结论；可维护性和架构约束更容易使用确定性工具，但仍不能替代功能结果。

## 基本实验单位

### Task

Task 描述待解决的问题，不包含解法。一个可评分 Task 至少包含：

- 不可变的任务标识和数据版本；
- 可恢复的初始工作区及其精确版本；
- 提供给 Agent 的自然语言问题；
- Agent 可见和不可见材料的明确边界；
- 独立 verifier 及成功条件；
- 允许修改的范围和资源预算；
- 任务来源、许可证和已知限制。

### Variant

Variant 描述被评测的 harness 配置。首要 Variant 是完整 Xcode；后续通过最小
harness、功能开关或上一版本形成对照。Variant 必须记录所有可能影响结果的运行
配置，不能只保存一个展示名称。

### Trial

Trial 是一个 Task、Variant、模型配置、预算和重复序号的真实运行。每个 Trial
必须使用隔离且可恢复的工作区，不得复用其他 Trial 的未声明状态。

### Experiment

Experiment 是一组可比较 Trial。配对实验应在同一任务集合、同一模型、同一预算
和相同重复策略下比较 Variant。模型自身无法从评测中消失，但可以通过控制变量和
配对差值降低其对 harness 归因的干扰。

## 任务数据战略

任务集本身是 Eval 的核心产品，不是 runner 的样例文件。Xcode 按以下来源逐步
建立任务组合：

1. **历史修复任务**：从 Xcode 和许可兼容项目的修复提交中恢复修复前版本，使用
   issue 或提交语义形成问题描述，使用修复后新增的回归测试作为隐藏证据。
2. **真实开发任务**：在解法产生前保存真实用户请求、初始源码和最终人工验收
   结果，经过脱敏和许可确认后进入数据集。
3. **外部 benchmark**：接入 SWE-bench、Terminal-Bench 等拥有独立官方 verifier
   的任务，用于外部可比性和更广的任务分布。
4. **受控合成任务**：只用于覆盖真实数据暂时缺少的特定 harness 压力，不承担
   总体能力结论，并必须标记生成方法和局限。

历史任务提取不能机械地把所有 commit 变成任务。候选任务必须经过数据审核：问题
能够脱离最终 patch 被理解，父版本可运行，验证器能独立判分，且任务不是纯格式化、
机械重命名或依赖已失效外部环境。

任务数据必须版本化。任务修订、grader 修复、排除项和污染风险都要进入变更记录，
避免不同数据版本的分数被直接比较。

## 独立验证契约

Verifier 必须运行在 Agent 不可访问的边界之外。隐藏测试、参考 patch、判分命令和
参考输出不得出现在 Agent 工作区、prompt、工具结果或可读取的环境变量中。

一次 Trial 的结果至少拆分为：

- `valid_trial`：环境、Agent 和 verifier 完成了可判定运行；
- `resolved`：目标问题的独立验证通过；
- `regression_free`：原有行为没有发生不可接受的回归；
- `policy_clean`：没有修改禁止材料、绕过验证或越过任务约束。

任务成功定义为上述条件全部成立。Agent 自己运行测试成功，只能作为 trace 中的
行为证据；最终判定必须由运行结束后的独立 verifier 重新执行。

基础设施错误必须与能力失败分离，至少区分：

- Agent 正常结束但未解决问题；
- Agent、provider 或工具运行失败；
- 超出声明预算；
- verifier 或环境失败；
- 任务无效或数据损坏。

无效 Trial 不得悄悄进入成功率分母；排除原因和排除数量必须随报告公开。

Verifier 优先使用确定性、可重复、低成本的 computational sensors，例如隐藏测试、
类型检查、lint、结构规则、性能阈值和差分执行。需要语义判断时可以增加人工审核或
LLM judge 等 inferential sensors，但必须单独报告其模型、rubric、重复一致性和成本，
不得用不稳定的语义分数覆盖确定性失败。

提供给 Agent 的 guides 与 Agent 可见的验证反馈属于 harness 输入；最终 verifier
属于评测边界。两者可以检查相同性质，但必须独立执行，避免 Agent 自报成功。

## 核心量化维度

### 结果

- 任务成功率、`pass@k` 和 `pass^k`；
- 目标验证、回归验证和约束验证的分项通过率；
- 多阶段或任务链的阶段成功率与完整链成功率；
- 按任务来源、能力要求和难度划分的结果。

### Harness 增益

核心结论来自配对 Variant，而不是孤立绝对分数：

```text
harness_gain = success_rate(candidate) - success_rate(control)
```

除完整 Xcode 与最小 harness 的总体对照外，逐步加入 compaction、工具并行、错误
恢复、权限反馈、session、MCP 和 memory 等消融实验。消融必须保持任务、模型和预算
一致，并报告置信区间或重复运行分布，不能用单次成败归因。

### 效率

- token、模型调用、工具调用、墙钟时间和可获得时的费用；
- 首次有效动作、首次有效修改和首次验证成功的时间；
- 重复、失败、无效和未被结果采用的工具调用；
- `tokens_per_success`、`tool_calls_per_success`、`time_per_success` 和
  `cost_per_success`。

成功单位成本必须将失败 Trial 的消耗计入总成本，避免通过丢弃昂贵失败美化结果。

报告应展示结果与投入的联合分布或效率前沿，而不是分别给出成功率与 token 总量。
更高预算未必带来更好结果；当额外模型思考、上下文或工具调用不再转化为成功时，
应识别投入甜点区和被支配的 Variant。

### 鲁棒性

- 工具、provider、MCP 和验证命令失败后的恢复率；
- 上下文压力及压缩前后的任务完成率；
- 权限被拒绝或资源受限后的替代路径成功率；
- 重复运行方差、超时率和破坏性修改率；
- 持久 session 中前序任务对后续任务的帮助与污染。

内部计数器只有在能够解释结果、成本或失败时才进入报告。指标数量不是目标，能够
支持决策才是目标。

## 可复现性与证据

每个 Experiment 必须保存：

- Xcode commit、工作区初始版本和脏状态检查；
- Task 数据版本、选中任务及排除项；
- Variant、模型、provider、采样参数和预算；
- 精确运行命令和必要的非秘密环境描述；
- 每个 Trial 的 trace、patch、终止原因、资源用量和错误分类；
- verifier 版本、命令、日志和分项结果；
- 聚合公式和生成报告所用的原始结构化数据。

密钥和敏感源码不得写入 artifact。报告必须能够从原始 Trial artifact 离线重建，
展示层不能成为唯一结果来源。

## 外部 Benchmark 的位置

外部 benchmark 通过 adapter 接入统一 Trial 和 Artifact 协议，但保留其官方任务、
环境和 verifier 语义。Xcode 不重新解释官方分数，也不把生成 prediction 当作完成
评测。

不同 benchmark 的绝对分数不强行汇总为一个总分。它们用于覆盖不同任务分布；
Xcode 自身的 harness 归因仍依赖受控的配对 Variant 实验。

公开 benchmark 还存在任务污染、任务形态狭窄和与真实产品工作流不一致的问题。
因此它们只提供外部坐标；Xcode 仍需维护覆盖自身产品面的数据集，包括多文件修改、
重构、迁移、项目指令、终端、MCP、长上下文和多轮任务。公开数据与产品数据必须
分开报告。

模型可能对训练时常见的 prompt、工具名称和编辑协议形成 harness 适配。单一模型上的
结论只能说明该模型与 Xcode 的组合表现。长期实验应在多个模型系列上重复关键
Variant，并区分“普遍 harness 增益”和“模型特定适配”，但不得用更换模型代替同模型
配对实验。

## 设计依据与外部参考

以下材料用于校准术语和路线，不作为未经验证的实现规范：

- [Harness engineering for coding agent users](https://martinfowler.com/articles/harness-engineering.html)：
  guides/sensors、feedforward/feedback、computational/inferential controls，以及行为、
  可维护性和架构适应度的区分；
- [The Anatomy of an Agent Harness](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness)：
  `Agent = Model + Harness`、文件系统、执行环境、状态、工具和编排边界；
- [Components of a Coding Agent](https://magazine.sebastianraschka.com/p/components-of-a-coding-agent)：
  实时仓库上下文、prompt/cache、工具、上下文缩减、session memory 和 bounded
  subagents；
- [The Coding Harness Behind GitHub Copilot in VS Code](https://code.visualstudio.com/blogs/2026/05/15/agent-harnesses-github-copilot-vscode)：
  产品专用任务集、可复现容器、解决率与 token 效率联合分析，以及 harness 变更的
  合并前评测；
- [What Is an Eval Harness](https://deepeval.com/blog/what-is-an-eval-harness)：
  dataset、运行采集和 metric suite 的基本闭环。Xcode 在此基础上额外要求真实源码
  工作区、独立 verifier、配对 Variant 和可重放 artifact。

## 防偏航规则

每个 Eval 变更都必须回答：

1. 它测量了哪项 harness 能力或为哪项真实测量提供基础？
2. 它的成功证据是否来自独立 verifier？
3. 它是 Eval 能力场景，还是应由 pytest 承担的基础设施测试？
4. 它产生的指标能支持什么工程决策？
5. 它是否保持 Task、Trial 和 Experiment 的可复现性？

如果无法回答，应停止扩展 runner 或指标，先补齐任务、验证或实验设计。阶段完成
不能修改本文定义的终局；新的实现工作必须在路线文档中找到归属，或先显式更新
路线与理由。
