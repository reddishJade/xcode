# Xcode Eval 路线图

## 文档职责

本文把 [evaluation-strategy.md](evaluation-strategy.md) 的长期目标拆成连续阶段，
记录每一阶段的交付物、验收门槛和后续方向。它不是一次性第一版计划，而是 Eval
从零到长期实验系统的实施路线。

路线可以根据证据调整，但不得绕过阶段验收门槛，也不得把 Test 重新包装成 Eval。
调整路线时应记录修改原因、已有证据和对终局的影响。

## 当前基线

- `eval-legacy` 保存旧 Eval 实现并跟踪远程 `origin/eval`；它只作为代码和历史
  参考，不再承载新设计。
- 新 `eval` 从当前 `dev` 创建，使用现行 Xcode 架构重新建设 Eval。
- 旧实现中的离线 pipeline、fake provider 和 tool-policy 场景属于测试候选，不是
  新 Eval 的能力分数。
- 旧 task、sandbox、adapter、trace、patch、report 和 baseline 代码必须逐项审核，
  不整体迁移。

## 目标系统形态

```text
Task Source
    -> versioned Task Dataset
    -> isolated Workspace
    -> real Xcode + real Model + Variant
    -> Trial Artifacts
    -> isolated Verifier
    -> Trial Result
    -> paired Experiment
    -> metrics, comparison and report
```

目标模块按职责划分，具体文件可随实现演进：

```text
src/xcode/evals/
├── schema.py       # Task、Variant、Trial、Result 和 Artifact 契约
├── task_sources/   # Git 历史、真实任务和外部 benchmark 数据入口
├── workspace.py    # 可重复、隔离的任务工作区
├── executor.py     # 通过正式装配路径运行真实 Xcode
├── verifier.py     # Agent 边界外的独立验证
├── artifacts.py    # trace、patch、配置、日志和结果持久化
├── experiment.py   # 重复试验、配对 Variant 和调度
├── metrics.py      # 结果、增益、效率和鲁棒性聚合
├── reporting.py    # 可重建的结构化和展示报告
└── cli.py          # 薄命令入口
```

## 阶段 0：冻结旧实现并确立契约

### 目标

停止在旧假设上继续增加功能，以战略文档和最小领域模型约束后续建设。

### 交付物

- `eval-legacy` 与新 `eval` 的分支边界；
- Eval 战略与本路线图；
- Task、Variant、Trial、VerifierResult、TrialResult 和错误分类的设计草案；
- 旧 Eval 模块的保留、重写、迁入 pytest 或废弃清单。

### 验收门槛

- 团队能够明确解释 Eval 与 Test 的边界；
- 每个旧模块都有迁移决定和理由；
- 新 schema 不依赖 fake provider 或旧 pipeline 事件；
- 尚未拥有独立 verifier 的场景不会被标记为能力 Eval。

## 阶段 1：首个真实、可判分闭环

### 目标

证明真实 Xcode 可以在隔离工作区解决一个真实来源任务，并由不可见 verifier 独立
判分，完整保存可重放证据。

### 交付物

- Git 历史任务候选扫描与人工审核流程；
- 3 至 5 个高质量、版本化的历史修复任务；
- 初始工作区恢复、Agent 可见边界和隐藏 verifier；
- 通过正式 `build_app()` 路径启动的真实 Xcode executor；
- patch、trace、配置、预算、终止原因和 verifier 日志 artifact；
- 单 Trial 的结构化结果，不以 HTML 或排行榜作为优先目标。

### 验收门槛

- Agent 实际读取、修改并验证真实源码；
- verifier 在 Agent 结束后从独立边界重新判分；
- Agent 无法读取隐藏测试、参考 patch 和 verifier 命令；
- 相同 Task 可以从干净初始状态重复运行；
- 环境失败、grader 失败和未解决任务能够被准确区分；
- pytest 覆盖基础设施契约，但 pytest 结果不进入 Eval 分数。

## 阶段 2：可重复的第一版 Eval

### 目标

把单任务闭环扩展为能够形成第一份可信基线的最小任务集和重复实验系统。

### 交付物

- 至少 10 个经过审核的真实任务，覆盖读取、定位、编辑、执行和回归验证；
- 每任务多次 Trial、明确预算和失败分类；
- `success_rate`、`pass@k`、`pass^k`、有效 Trial 数和排除项；
- token、模型调用、工具调用和墙钟时间原始数据；
- 解决率与 token、时间等投入的联合分布及初始效率前沿；
- JSON/JSONL 原始结果与可由其离线重建的摘要；
- 固定 Task 数据版本、Xcode commit 和模型配置的基线运行。

### 验收门槛

- 报告中的每个成功和失败都能追溯到 artifact；
- 重跑不会复用未声明的工作区或 session 状态；
- 失败 Trial 的资源消耗不会从效率统计中消失；
- 任务集不是只覆盖 Eval pipeline 或玩具函数生成；
- 第一版结论明确限定模型、数据版本和预算，不宣称普适能力。

## 阶段 3：Harness 归因与配对实验

### 目标

从“Xcode 得了多少分”推进到“Xcode 的 harness 带来了多少可归因增益”。

### 交付物

- `full` 与 `minimal` 两个明确且可审计的 Variant；
- 同任务、同模型、同预算、同重复策略的配对调度；
- `harness_gain`、成功单位成本和运行分布；
- Variant 配置快照和配对完整性检查；
- 对模型随机性和小样本不确定性的明确展示。

### 验收门槛

- Variant 差异局限于声明的 harness 能力；
- 不使用不同 prompt、预算或任务为某个 Variant 制造优势；
- 单次偶然成功不被解释为 harness 增益；
- 报告同时展示绝对表现、配对差值、样本数和无效 Trial；
- `minimal` 仍是可运行的真实对照，而不是故意破坏的 Agent。

## 阶段 4：能力消融与鲁棒性

### 目标

量化 Xcode 各项 harness 能力在何种任务和压力下产生价值。

### 交付物

- compaction、工具并行、错误恢复、权限反馈、session、MCP 和 memory 的独立
  Variant；
- 长上下文、工具故障、权限受限、provider 中断和外部能力不可用的真实压力任务；
- 恢复率、污染率、破坏性修改率和退化幅度；
- 能力与任务标签之间的分层结果。

### 验收门槛

- 每个压力场景仍要求解决真实问题，而不是检查某个事件是否出现；
- 故障注入改变环境条件，不直接向 Agent 泄露预期轨迹；
- 内部指标能够与任务结果或成本建立解释关系；
- 无显著收益或产生负收益的能力同样如实报告；
- 权限和安全消融不削弱 verifier 隔离或宿主机安全边界。

## 阶段 5：任务规模与外部可比性

### 目标

扩展任务分布，降低只对 Xcode 自身历史过拟合的风险，并获得外部可比较结果。

### 交付物

- 持续维护的真实开发任务采集和脱敏流程；
- 许可兼容的第三方历史修复任务；
- SWE-bench、Terminal-Bench 等官方 benchmark adapter；
- 数据污染、任务重复、环境失效和版本迁移审计；
- 在多个模型系列上复跑关键 Variant 的兼容性矩阵；
- 按任务来源分别报告的结果。

### 验收门槛

- 外部 benchmark 使用官方 verifier 和官方评分语义；
- prediction 生成与官方评分明确分离；
- 不同 benchmark 不被武断合成一个失去语义的总分；
- 数据许可证、来源和排除原因完整可查；
- 任务集包含足够差异，结论不依赖单一仓库或单一任务模式。
- 模型特定适配与跨模型稳定的 harness 增益分别报告。

## 阶段 6：持续决策系统

### 目标

让 Eval 成为 Xcode 架构决策和发布判断的长期证据，而不是偶尔运行的演示。

### 交付物

- 稳定基线、候选版本对比和趋势历史；
- 按风险分层的快速、标准和完整实验档位；
- 成功率、稳定性、成本和鲁棒性的回归预算；
- 可重放的失败案例库与任务数据修订记录；
- 定期校准、人工审核和任务退役机制。

### 验收门槛

- 发布结论引用精确 Experiment 和 artifact，而不是截图或单次运行；
- 回归门槛同时考虑效果与成本，不鼓励无限增加模型预算；
- 快速档位只用于反馈速度，不冒充完整能力结论；
- 历史趋势能区分代码变化、模型变化、数据变化和环境变化；
- Eval 失败能转化为可定位的产品问题或明确的数据问题。

## 贯穿所有阶段的工作流

每一项实现按以下顺序推进：

1. 在战略中确认它属于 Xcode harness 的测量范围；
2. 在本路线图中定位阶段、交付物和验收门槛；
3. 先定义任务证据、对照条件和预期决策，再设计指标；
4. 实现最小真实路径，并用 pytest 验证基础设施契约；
5. 运行真实 Trial，检查 artifact 和失败分类；
6. 只有结果可复现后才扩展报表、并发和任务规模；
7. 将新证据、限制和路线调整落盘。

## 近期实施队列

阶段 0、1、2 已达到文档门槛。阶段 3 的内部配对实验仍暂缓，但已在外部任务上做
`full/minimal` 归因 smoke；阶段 4、6 保持全局可见但不提前堆叠实现：

1. 暂停内部 `full/minimal` 配对运行，保留已冻结的 Variant 契约和实现；
2. 把 20/40/60 等预算档提升为版本化 Experiment 变量，不再依赖临时复制 dataset
   文件来改变预算；
3. 在外部任务上完成模型可解性校准：已完成 Requests 的 20/40 次预算阶梯，
   并完成 Astropy、Django 的 40 次预算重复；
4. 用公开字段建立外部 benchmark adapter，坚持 Agent 生成 prediction、官方 scorer
   独立判分的边界；
5. 阶段 5 外部门槛已达到第一道门槛：3 个不同仓库任务、每任务至少 2 次有效重复，
   公开预算档、无效 Trial 和可解性分层；下一步扩展模型矩阵，并继续配对 Variant 重复。

## 阶段证据与限制

### 阶段 0（已通过，2026-07-15）

- 分支：`eval` 从 `dev` 的 `3c2c17a` 创建；`eval-legacy` 保留旧实现并跟踪
  `origin/eval`。
- 契约：`src/xcode/evals/schema.py` 定义不依赖 fake provider 或旧 pipeline 事件的
  Task、Variant、Trial、VerifierSpec、VerifierResult、TrialResult、错误分类与 artifact
  索引；Task 只含不透明 verifier id，隐藏命令不进入 Agent 可见对象。
- 审计：[evaluation-legacy-audit.md](evaluation-legacy-audit.md) 对旧模块逐项记录保留、
  重写、迁入 pytest 或废弃决定，并明确旧场景均不得直接计能力分。
- Test 边界：`test_evals_schema.py` 只验证基础设施契约，不产生能力分数。
- 限制：尚无通过审核的真实 Task、隔离 workspace、真实 executor 或独立 verifier
  Trial，因此阶段 1 尚未通过，也没有 Xcode 能力结论。

### 阶段 1（已通过，2026-07-15）

- 历史证据：`337b989` 候选已完成父版本失败、修复版本通过的外置 verifier 差分重放，
  详细命令边界与结果见 [evaluation-task-review.md](evaluation-task-review.md)。
- 基础设施：`GitWorkspaceFactory` 使用精确 Git commit 创建无 `.git` 的独占工作区并
  保存初始内容；`RealXcodeExecutor` 调用正式 `build_app()`、强制墙钟/模型调用预算并
  保存逐事件 trace；`VerifierRunner` 只在 Agent 结束后运行独立目录命令并要求显式
  四项结果协议。
- 真实模型证据：Experiment `phase1-baseline-thinking-20260715a` 使用正式 Xcode 与
  `deepseek-v4-flash`。repetition 0 正常完成并由隐藏 verifier 判定
  `resolved=true`、`regression_free=true`、`policy_clean=true`；消耗 27 次模型调用、
  752,706 input tokens、15,086 output tokens、42 次工具调用和 182.09 秒。Agent patch
  只修改两个允许的生产文件，隐藏行为 3 passed、独立回归 10 passed。
- 重复恢复证据：repetition 1 从同一父提交创建不同 workspace，不复用 session，形成
  另一条有效 Trial；它在 40 steps 后未解决，verifier 正确记录
  `resolved=false`、`regression_free=true`、`policy_clean=true`，没有把能力失败分类成
  环境或 grader 失败。两次运行均无 approval gate 或缺失 `uv`。
- 隔离证据：宿主 `bwrap` 负向探测在 mount namespace 内查询控制仓库私有 verifier
  路径得到 `False`。隔离 worker 使用 `--clearenv`，只挂载 Trial workspace、当前
  `src/xcode`、Python runtime 和专用空输出目录；不挂载源 Git 仓库、Task 数据集、
  artifact 或 `evals/private`。
- Eval 权限策略：Trial 固定使用独立 build policy，清除普通运行配置中的交互式
  `ask`；shell 分析器残余 ask 仅在 bubblewrap 内一次性放行，显式联网 deny 仍优先。
  虚拟环境与 `uv` 只读挂载，pytest、Ruff、uv 缓存均写入沙盒 `/tmp`。
- Artifact：每次 Trial 保存 Task、Trial、脱敏环境、trace、patch、stdout/stderr、
  verifier 日志和结构化结果，并以 `checksums.json` 封存 SHA-256；`ArtifactStore` 可在
  无模型、无 verifier 的情况下离线重建并校验完整性。真实证据保存在被 Git 忽略的
  `eval-results/artifacts/phase1-baseline-thinking-20260715a/`。
- 任务集进度：已固化 3 个真实 Git 历史修复 Task，分别覆盖 provider fallback 恢复、
  工具错误/watchdog 协作、thinking/model 请求控制；三者均完成父失败、修复通过的
  外置差分和只应用生产 patch 的参考重放。数量达到阶段 1 的 3 至 5 个要求下限。
- 结论边界：阶段 1 只证明真实闭环、隔离、判分、重复恢复和证据保存成立；1/2 的
  单任务观察不是稳定能力基线，不得外推为总体成功率。三个任务均来自单一仓库，回归
  verifier 使用声明的稳定切片而非完整历史套件。阶段 2 仍需至少 10 个任务和多次 Trial。

### 阶段 2（已通过，2026-07-15）

- Git 复现基线：阶段 0–1 实现提交为 `a45dc01`，覆盖率忽略规则提交为 `4d5ecac`；后者
  工作树干净时运行的 `phase1-clean-thinking-20260715a` 精确记录
  `harness_revision=4d5ecac7e6ce30aed8c2ef88a5d408f5b53801cc` 和 `xcode_dirty=false`。
  该 Trial 为有效能力失败（2/3 隐藏行为通过、回归与 policy 通过），证明 Git 身份与
  能力结果分离保存。
- 任务扩展：新增 session reset 授权生命周期、external observer 后台调度、MCP 未知
  override 诊断和 parallel tool worker limit Task，均完成父失败/修复通过及只应用生产
  patch 的参考重放；随后加入 snapshot unindexable path、MCP lazy connection recovery
  与 provider transport consistency Task 并完成同样三重重放；当前为 10/10 个审核任务，
  阶段 2 的语料门槛已满足。
- Experiment 基础设施：已增加严格的 task/Variant/repetition 配对展开、恢复时完整 Trial
  跳过与不完整目录隔离；聚合明确从成功率分母排除无效 Trial，但将其资源消耗计入单位
  成本，并计算 `success_rate`、`pass@k`、`pass^k`、分项通过率和初始效率前沿。
- 离线证据：Experiment 声明、逐 Trial JSONL、结构化 summary、Markdown 报告和控制文件
  checksum 均从已封存 Trial artifact 重建；报告过程不读取 provider 或重新运行 Agent。
- 固定基线：Experiment `phase2-baseline-full-deepseek-v4-flash-20260715a` 在干净提交
  `76391bee05324a9522345c2d91d8302ecdbc41e8` 上，使用 `xcode-history-v1` 的 10 个任务、
  `xcode.config.json` 配置的 `openai_chat/deepseek-v4-flash` 和 `full` Variant，每任务
  两次独立 Trial；20/20 均形成封存 artifact，workspace 与 session 不复用。
- 基线结果：17 个有效 Trial 中 2 成功，`success_rate=11.76%`；3 个 Trial 因
  `budget_exceeded` 排除。具备两次有效结果的 8 个任务上，`pass@2=12.5%`、
  `pass^2=12.5%`。目标验证通过率为 11.76%，回归验证通过率为 100%，policy clean
  通过率为 58.82%。成功集中在 parallel tool worker limit Task 的两次重复。
- 基线成本：所有 20 个 Trial（含失败和排除项）累计 22,624,847 input tokens、
  512,583 output tokens、739 次模型调用、1,093 次工具调用和 6,384.58 秒；每次成功
  分摊 11,312,423.5 input tokens、546.5 次工具调用和 3,192.29 秒。provider 未返回
  USD 成本，因此不估算 `cost_per_success`。
- 可复现证据：20 个 Trial 的环境快照一致记录 clean revision、数据版本、模型和 Variant；
  离线命令重新校验所有 Trial checksum 后重建同一 summary/report。去除 trace 和源码的
  固定摘要保存在
  `evals/baselines/phase2-baseline-full-deepseek-v4-flash-20260715a.json`，并记录原始控制
  artifact 的 SHA-256；完整原始证据仍在被 Git 忽略的 `eval-results/artifacts/`。
- 结论边界：该结果只描述这一模型、完整 Xcode、单仓库历史任务集和每任务两次重复；
  单 Variant 的效率前沿只是绝对坐标，不能解释为 harness gain。阶段 3 必须在相同任务、
  模型、预算和重复策略下加入可运行的 `minimal` 配对对照。

### 阶段 3 外部配对 smoke（进行中，2026-07-16）

- 配对契约：Django 与 Requests 各完成 1 个 `external-40` repetition 的 `full/minimal`
  配对；两边使用同一任务、模型、预算和 repetition，两个 Variant 均形成有效 Trial，
  官方 verifier 均 `resolved=true`、`regression_free=true`、`policy_clean=true`。
- 配对结果：Django 配对的 `harness_gain=0`，full/minimal 都成功，full 比 minimal 多用
  83.82 秒；Requests 配对同样 `harness_gain=0`，full 比 minimal 多用 25.42 秒。两组
  都是成功平局，不能解释为 harness 无增益，当前只能说明在这两个任务的一次重复上没有
  可观测成功率差异。
- Astropy 配对限制：full Agent 已生成 patch，独立手工 scorer 判定该 patch resolved，
  但隔离 worker 在退出阶段超过 `external-40` wall budget，未形成合法 Trial；该运行按
  `agent_failure`/执行器收尾问题排除，不进入配对分数。此前 Astropy 的两次单 Variant
  full Trial 仍保持有效记录。
- 可靠性修复：`BubblewrapExecutor` 现在把 `subprocess.TimeoutExpired` 转换成
  `IsolationError`，由 TrialRunner 封存为无效 Trial，而不是留下半成品目录；对应契约测试
  已加入 `test_evals_isolation.py`。修复后的真实 Trial 仍需继续验证。
- 阶段边界：当前只有 2 个有效外部配对、每对 1 次重复，且仅覆盖一个模型和 `full`/`minimal`；
  阶段 3 仍未通过，需扩展到三任务并增加重复后才能报告稳定 harness gain。

### 阶段 5（外部 smoke，外部门槛 A 已通过，完整阶段未通过，2026-07-16）

- 外部入口：`swebench-lite-smoke-v1` 固化公开任务字段、仓库 commit、许可证和
  `problem_statement`；Agent 不可见隐藏 patch、测试和 verifier 命令。prediction 生成
  与官方评分分离，独立 verifier 使用 `swebench==3.0.11` 的 Docker scorer。
- 真实运行：`psf__requests-2148` 有 3 次有效 `full` Trial（20 次预算 1 次、40 次预算
  2 次）；`astropy__astropy-12907` 和 `django__django-10914` 各有 2 次有效 40 次预算
  Trial。7 次 Trial 均保存 trace、patch、verifier 日志和 checksum。
- 外部判定：7 次有效 Trial 中 6 次 `resolved=true` 且 `regression_free=true`，外部 smoke
  成功率为 6/7（85.71%）；唯一失败是 Requests 的 20 次预算 Trial。仅看统一的
  `external-40` profile，3 个不同仓库任务共 6/6 成功。该结果满足本路线临时设定的
  “3 个任务、每任务 2 次有效重复”外部门槛 A，但仍只覆盖 `deepseek-v4-flash/full`，
  不能解释 harness gain 或跨模型稳定性。
- 环境限制：首次冷启动因镜像构建超过单 Trial verifier 预算而无效；缓存镜像后的本轮
  官方 scorer 执行约 223 秒完成。两类结果均保留为环境与任务证据，不混入成功率。
- 框架对照：同一任务、模型、预算和官方 verifier 下，FirstCoder `36318b8` 也生成了
  等价的功能 patch；它在第 20 次模型调用达到自身 `provider_call_limit`，因仓库缺少
  `firstcoder.agent.tool_settlement` 还需临时兼容层才能启动，因此不计为有效 FirstCoder
  Trial。将其留下的 patch 单独送入同一官方 scorer 后，仍是 10 个 `FAIL_TO_PASS` 通过、
  同一个 `PASS_TO_PASS` 回归失败。该单任务交叉证据更支持修复方案/模型输出存在回归，
  但不能替代多任务、多重复的框架归因实验。
- 预算阶梯：同一 Xcode、模型、任务和 verifier 将模型调用上限从 20 提高到 40 后，
  Experiment `phase5-swebench-lite-requests-2148-budget40-20260716a` 形成有效 Trial，
  40 次模型调用、43 次工具调用、858,435 input tokens、21,765 output tokens、292.26 秒；
  官方判定 `resolved=true`、`regression_free=true`、`policy_clean=true`。40 次版本在
  两个具体读取路径分别转换 `socket.error`，并通过回归测试；因此当前证据把 20 次档的
  失败归为预算不足的优先假设，暂不据此更换模型。该结论仍只覆盖单一外部任务。
- 预算重复：Requests 的第二次 40 次预算 Trial（`phase5-swebench-lite-requests-2148-budget40-20260716b`）
  使用正式 `external-40` profile，22 次模型调用、27 次工具调用、302,303 input tokens、
  104.65 秒，官方判定 `resolved=true`、`regression_free=true`。因此 Requests 在 40 次档
  当前为 2/2 成功，但仍只是单一任务证据。
- 任务扩展：Astropy 首次运行因私有 verifier 尚未配置而记为 `verifier_failure`，不计入有效率；
  预热官方镜像后重跑 `phase5-swebench-lite-astropy-12907-budget40-20260716c`，35 次模型调用、
  41 次工具调用、716,322 input tokens、329.34 秒，官方判定 `resolved=true`、
  `regression_free=true`。这表明首次失败是评测环境准备问题，不是模型能力判定。
- Django 任务（`django__django-10914`）的两次 40 次预算 Trial 分别消耗 983,625/749,941
  input tokens 和 204.34/219.91 秒，均由官方 scorer 判定 resolved 且无回归；它补充了与
  Requests、Astropy 不同仓库和文件权限语义的外部任务分布。
- 跨框架 40 次对照：FirstCoder `36318b8` 在同一模型、任务和 40 次调用上限下正常生成
  patch，但官方 scorer 判定 `resolved=false`；10 个 `FAIL_TO_PASS` 通过，
  `test_redirect_with_wrong_gzipped_header` 与 `test_stream_timeout` 两个既有测试失败。
  FirstCoder 当前 commit 缺少 `firstcoder.agent.tool_settlement`，运行使用了仅限本次进程的
  兼容层，因此该结果是 patch-level 交叉证据，不计入正式有效 Trial。它提示 harness
  差异可能影响模型是否能在更高预算下收敛，但需要无兼容层的可运行对照和更多任务才能
  形成 Harness gain 结论。

## 路线变更记录

路线发生实质调整时，在此追加日期、决定、证据和影响。不要覆盖过去的方向，使后续
维护者能够理解为什么改变。

| 日期 | 决定 | 证据与影响 |
|---|---|---|
| 2026-07-15 | 冻结旧 Eval，从当前 `dev` 新建 `eval` | 旧分支落后主线且混合了 pipeline test 与能力评测；新路线以真实运行、独立 verifier 和 harness 量化为核心。 |
| 2026-07-15 | 阶段 0 通过，进入阶段 1 | 新 schema 强制区分 Agent 可见 Task、隐藏 VerifierSpec、有效 Trial 与基础设施错误；旧模块逐项审计完成。当前没有真实能力分数。 |
| 2026-07-15 | Eval 使用独立非交互 build policy | 真实 trace 证明普通 act/build 静态规则和 shell unresolved-effect 会产生无审批机制的 ask；Eval 改由 bubblewrap 承担宿主文件边界，显式网络 deny 保留，沙盒内残余 ask 自动单次放行。 |
| 2026-07-15 | 阶段 1 通过，进入阶段 2 | 同一真实历史任务产生 1 次独立 verifier 成功和 1 次有效能力失败；两次均从精确父提交恢复独立 workspace，并完整保存哈希封存 artifact。该证据只证明闭环，不作为稳定基线。 |
| 2026-07-15 | 阶段 2 通过，进入阶段 3 | 10 个审核任务各完成两次真实 Trial；17 个有效结果中 2 成功，3 个预算排除，所有成本均进入聚合。固定摘要和原始 artifact 哈希已版本化；结论限定于 `deepseek-v4-flash` 与 `full` Variant。 |
| 2026-07-15 | 冻结阶段 3 `full/minimal` Variant v1 | minimal 保留模型、任务、预算、安全边界、核心工具和项目指令，只关闭 compaction、fallback、工具并发、request hygiene、重复工具 watchdog 与 contextual retrieval prompt；配对指标仅使用双方有效的相同 task/repetition。 |
| 2026-07-16 | 暂缓阶段 3 内部配对，先验证阶段 5 外部边界 | 内部配对 smoke 暂停；新增 SWE-bench Lite 公开字段 adapter 与官方 Docker scorer。`psf__requests-2148` 的 1 次有效 Trial 因既有测试回归而未解决，证明判分链路成立但不足以通过阶段门槛。 |
| 2026-07-16 | 增加模型可解性与预算校准门槛 | 同一外部任务在 20 次调用下失败、40 次调用下由 Xcode 成功；FirstCoder 40 次对照仍有回归但受缺失模块兼容层影响。后续先固定预算档、扩展任务和重复，再判断模型替换或 harness 增益。 |
| 2026-07-16 | 外部 40 次预算校准继续有效 | Requests 第二次 40 次 Trial 成功；Astropy 在补齐私有 verifier、预热官方镜像后 40 次 Trial 成功。当前外部证据为 Requests 2/2、Astropy 1/1，阶段 5 仍缺第三任务及足够重复。 |
| 2026-07-16 | 外部门槛 A 达到，暂不宣称阶段 5 完成 | Django 40 次预算完成 2/2，Astropy 补齐第二次后为 2/2；3 个仓库共 7 次有效 Trial、6 次成功。后续必须扩展至少一个模型系列并进行同任务同模型同预算的 `full/minimal` 配对，才能完成归因与阶段 5 全部要求。 |
| 2026-07-16 | 外部配对 smoke 启动，阶段 3 仍未通过 | Django、Requests 各完成一次有效 `full/minimal` 配对且均成功平局；Astropy 配对因隔离 worker 收尾超时无效。新增 TimeoutExpired 封存修复和测试，后续继续验证三任务多重复。 |
