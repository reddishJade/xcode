# TODO

## 当前执行顺序

`xcode` 当前阶段的主线按以下顺序推进：

1. `Memory`：先把 memory 从静态检索库补成轻量、可评测的经验辅助层
2. `Eval`：再把 eval 从可运行 pipeline 补成可信的回归与对照子系统
3. `MCP`：补齐 coding agent 必要能力，保持与主流 MCP 工作流的基础兼容

`Tasks`、`Progress`、`Daemon`、`Config` 和 `Experimental State` 当前视为
支持性事项：仅在它们直接服务上述三条主线，或修复明确正确性问题时继续推进。

## Memory — 从静态检索库演进为轻量、可评测的经验辅助层

当前基线：项目级 `MEMORY.md` 与用户级 `~/.xcode/memory/MEMORY.md` 保存
结构化 H2 记忆块；写入路径包括 `/memory add` 和 compaction consolidation；
读取路径包括每轮 BM25 主动召回、隔离的 `<memory>` 上下文注入和只读
`search_memory` 工具。Xcode 中 memory 的定位是基础设施，不是独立产品；目标是
让 agent 记住项目级事实、用户偏好和已验证的近期经验，而不是在仓库里内嵌一套
完整的数据治理系统。

### MEM.1 建立轻量评测与 trace 基线
- 保留统一 memory trace：至少覆盖 accepted / rejected、retrieved、injected、
  tool searched、used、superseded 和 forgotten；允许保留 `candidate_created`
  作为写入尝试事件，但不再围绕它扩展审查管线
- 记录 memory id、层级、检索分数、进入上下文的 token 数、延迟和拒绝原因；不得
  在 trace 中复制未脱敏的完整用户级记忆
- 将检索评测接入现有 eval report，至少维持 Recall@k、MRR、无关注入率、过期或
  冲突记忆召回率、检索延迟和 token 成本
- 保留同一 coding fixture 的 memory on / off 对照，测量任务成功率、工具调用数
  和负迁移；不能只用“预期标题是否出现在 top-k”作为 memory 有效性的结论
- 评测覆盖优先级：精确错误消息、代码符号、路径、中文查询、语义改写、多轮事实
  更新和不应召回的干扰记忆

### MEM.2 保留稳定身份与最小可用元数据
- 保留稳定 `memory_id`，并区分 `semantic`、`episodic` 和 `preference` 三类主用
  记忆；`procedural` 只作为显式人工写入类型保留，不再在 Xcode 内自动晋升
- 保留结构化字段：scope、source session、相关文件或 symbol、created /
  modified、confidence、status、validity；这些字段服务于检索门控和人类 review
- 继续保持 Markdown 为正式界面；内部索引不得破坏项目级和用户级分层
- 提供一次性兼容读取或迁移，将现有 H2 块转换为带稳定 id 的记录；不长期维护
  两套并行写入协议
- evidence 字段允许保留为可选补充信息，但 Xcode 不再围绕它实现重型 gate、
  provenance 审批或自动 verified 治理

### MEM.3 收敛到可用的检索门控与遗忘策略
- 自动 `<memory>` 注入和 `search_memory` 工具使用同一检索入口，并都更新
  `last_used_at`
- 保留最低相关性和置信度门槛，允许返回零条结果；不得固定把前三条弱相关记录
  注入每轮上下文
- query 除用户文本外，应可使用当前文件、symbol、错误消息、任务阶段和涉及模块，
  且这些运行时信号必须作为结构化检索上下文传入，不拼成隐藏 prompt
- 注入短小 memory packet，仅包含 id、类型、结论、适用范围、证据摘要和来源；
  完整记录由 `search_memory` 按需获取
- 遗忘策略继续综合 freshness、使用频率、验证状态和成功/失败反馈；重点是可解释、
  够用和不挡路，而不是构建复杂的生命周期编排

### MEM.4 保留反馈驱动的轻量降权
- 记录某条记忆是否被注入、被 agent 明确引用或采用，以及任务最终成功、失败或被
  用户纠正；不得把“被召回”自动视为“有帮助”
- 根据结果维护 utility、success count 和 failure count，支持降权、review、
  supersede 和恢复，不直接用一次失败物理删除记录
- 增加 stale memory、错误经验传播、分布变化和错误相似度匹配测试，确保经验复用
  不会稳定复制旧错误

### Memory Backlog（仅在真实需求出现后恢复）
- 不在 Xcode 内继续实现 candidate / quarantine / promote 审查管线
- 不在 Xcode 内继续实现 episodic → procedural → skill 自动晋升链
- 不为 memory 单独建设重型 evidence gate、来源审计或长期 provenance 治理
- 不在没有评测缺口证明之前引入 embedding / vector store / graph database /
  LLM reranker

### Memory 暂不实现
- 不立即引入 graph database：只有真实 multi-hop、时序关系或跨记录冲突评测显示
  平面记录无法满足需求时，再评估显式关系图
- 不立即把 memory 主存储迁移到 SQLite 或外部 vector database：先完成稳定数据
  模型和评测；仅在索引重建、查询延迟、并发写入或事务一致性出现实证瓶颈后迁移
- 不保存完整 chain-of-thought、未经筛选的全量聊天或所有工具输出；长期记忆只保存
  完成任务所需的结论、证据、边界和可复用经验
- 不允许 memory 自动修改项目规范、用户偏好或高置信度架构事实而没有可追踪证据
  和 supersede 记录
- 不以“接入 embedding、知识图谱或某个 memory 框架”本身作为完成标准；完成标准
  是 coding task 增益、低负迁移、可解释检索和可治理生命周期

---

## Eval — 从可运行 pipeline 演进为可信的回归与对照子系统

当前基线：`EvalRunner` 已支持多 task、多 trial、结构化事件 trace、确定性 grader、
文件证据、validation command、LLM-as-judge、fixture 目录隔离、`pass@k` /
`pass^k` 和 JSON / HTML / CSV report；已提供 HumanEval、EvalPlus、MBPP loader
及 SWE-bench patch 导出。当前离线 suite 主要验证 eval 基础设施而非真实 agent
能力，task metadata 缺少强 schema，run 缺少完整可复现清单，也尚未形成
baseline / candidate 对照、统计置信度、能力切片和持续回归闭环。

### E.1 修正 eval 基线与能力声明
- 修复默认 `all` suite：离线 provider 必须能表达任务声明的完整多步工具轨迹；
  当前 `multi-read-grep` 只生成第一个预期工具调用，不能把 runner fixture 的限制
  计作 agent 能力失败
- 将 benchmark 状态明确区分为 `integrated`、`export-only` 和 `catalog-only`；
  `--list-benchmarks`、文档和 CLI choices 必须使用同一 registry，不能把
  Terminal-Bench、SWE-bench Verified、Aider Polyglot 等未接通 harness 的目标
  展示成可直接运行
- capability suite 与 regression suite 分离：capability 用于发现能力边界，允许
  较低通过率；regression 必须稳定接近全绿并作为代码变更门禁
- 为内置 task 增加稳定版本、owner、能力维度、预期时长、难度和适用运行模式；
  suite 不再只是硬编码 task tuple 的匿名集合
- 增加 eval 自检，验证 task id 唯一、fixture 存在、validation command 合法、
  grader 引用有效，以及真实写入任务具有隔离环境

### E.2 轻量 run 记录与 artifact
- 每次运行输出轻量 `run_manifest.json`，至少记录模型 / provider、AgentConfig、
  suite / task 标识、开始时间和 schema version；优先服务“这次比上次好还是差”
- trace 和 report 使用稳定 schema version，并提供一次性迁移或兼容读取；不长期
  维护多个无版本输出协议
- 记录实际可获得的 input / output token、墙钟时间、模型与工具延迟、退出原因；
  无法获得的字段显式标记 unavailable，不用估算值冒充真实用量
- fixture 目录复制继续作为快速本地基线；需要依赖、服务或系统权限的任务继续标注
  为受限环境，不把目录复制描述成安全沙箱
- commit hash、dirty state、工具目录 hash、runtime hash、dataset digest、OS、
  随机种子等字段按真实需求逐项补充，不先写成 Xcode 的主线要求

### E.3 强类型 task、grader 与 outcome 模型
- 将 `EvalTask.metadata` 中的 evidence、validation、fixture、benchmark 等隐式协议
  提升为强类型 schema；JSON / JSONL 加载必须拒绝未知字段和错误类型，并输出
  精确字段路径
- grader 使用稳定 id，并返回 `score`、`passed`、`required`、`weight`、
  evidence 和 failure category；trial 不再只能用全部布尔 grader 的简单 AND
- 区分 outcome grader 与 trajectory grader：测试、编译、目标状态等决定任务
  是否完成；工具策略、路径质量和效率用于诊断，不应默认覆盖真实 outcome
- 确定性代码 grader 优先；LLM judge 只评判无法可靠程序化判断的质量维度
- LLM judge 使用固定模型和结构化输出，保留失败重试和成本上限；不为 Xcode 先做
  重型标注校准工作流
- judge unavailable、parse failure 和 timeout 继续显式记录，但 required judge
  不得以 skipped 方式让 trial 静默通过

### E.4 Baseline 对照与回归门禁
- CLI 支持 `--baseline <report-or-run-dir>`，按 task、grader、能力切片比较
  candidate 与 baseline 的成功率、成本、延迟、工具调用和失败类型
- 保留最小可用的 baseline diff：优先解决“有没有回归、回归在哪个 task / grader”
- 对同一 task 尽量使用固定环境和稳定输入；多随机种子、配对运行和复杂统计在出现
  真实决策歧义前不做成主线要求
- 输出 regression、improvement、unchanged 和 incomparable task 列表，并能下钻
  到 trace、patch、grader evidence 和 validation 输出
- 增加可配置 CI gate，例如 regression suite 不得下降、关键 grader 必须全过、
  p95 延迟和平均 token / 成本增长不得超过预算
- 保存历史 run 索引和趋势数据，但 report 文件继续是可移植的正式产物；不要求
  先部署数据库或外部观测平台才能运行本地 eval

### E.6 Agent trajectory、探索与恢复能力评测
- 为错误恢复增加 fault-injection suite，覆盖命令失败、测试失败、缺失依赖、
  错误路径、工具超时、provider 中断和上下文冲突，测量诊断、重试和降级能力
- trajectory 指标只保留对 Xcode 当前有诊断价值的最小集合，例如目标文件 / symbol
  Recall@k、首次有效定位步数、无关文件访问率和重复工具调用
- 增加 tool policy 的状态结果检查：不能只看是否调用某个工具，还要检查参数、
  调用顺序、工具结果是否被采用以及禁止副作用是否实际发生
- 将 memory on / off、retrieval 策略、prompt 模块、工具组和 subagent 策略作为
  一等实验变量，支持同任务 A/B，对比 outcome、负迁移、成本和 trajectory
- 失败分类至少区分理解、检索、规划、工具选择、工具执行、代码修改、验证、停止
  条件和环境故障，避免所有失败最终只表现为 `trial.success = false`

### Eval Backlog（仅在真实需求出现后恢复）
- 不先建设研究型 run manifest：commit hash、dirty state、工具目录 hash、runtime
  config hash、dataset digest、OS、随机种子等字段按需加
- 不先建设 train / dev / held-out 数据集治理、许可证台账、污染审查和失效登记流
- 不先定义通用 benchmark adapter 生命周期，也不把 Xcode 变成外部 benchmark 平台
- SWE-bench、Terminal-Bench 等外部环境继续保持 catalog 或 export-only 心态；
  只有当 Xcode 真正需要它们驱动发布决策时，才接官方 harness 和长时程环境

### Eval 暂不实现
- 不先建设复杂 Web dashboard、远程服务或多租户平台；先让 run artifact、
  baseline diff 和 CI gate 可靠
- 不以增加 benchmark 数量作为完成标准；未接通官方环境和 outcome grader 的
  benchmark 只保留 catalog 或 export 状态
- 不依赖单一 LLM judge 作为主要正确性信号，也不保存或评估私有 chain-of-thought
- 不为所有任务规定唯一黄金工具轨迹；允许不同有效策略，只处罚可证明的错误、
  无效副作用和显著资源浪费
- 不立即引入结果数据库、分布式调度或大规模并发执行；出现历史查询、运行时长或
  吞吐瓶颈后再设计持久存储和 worker 调度
- 不将公开 benchmark 分数直接等同于真实用户价值；完成标准是可复现的能力提升、
  低回归、合理成本和对 xcode 实际任务的稳定增益

---

## MCP — Coding Agent 必要能力

状态：M.1-M.5 已完成。真实官方 server 回归通过 `mcp_external` marker 与默认
离线 pytest 分离；协议范围仍遵循下方“暂不实现”约束。

当前基线：使用官方 Python SDK 连接本地 stdio server，支持初始化协商、
`tools/list`、`tools/call`、分页、`tools/listChanged`、schema cache、延迟加载、
超时取消和有限重连。后续只补 coding agent 的实际运行缺口，不追求完整 MCP
协议覆盖。

### M.1 暴露 workspace roots
- 通过 SDK 的 roots callback 向 server 暴露当前项目根目录
- 仅包含宿主权限边界允许读取的 workspace；不得把用户主目录或任意磁盘根目录
  默认暴露给 server
- workspace 发生明确切换时发送 roots changed notification
- 添加 roots 请求、空 roots 和越界目录过滤的协议测试

### M.2 MCP 运行时状态与手动重载
- 增加只读状态接口，展示 server 的 configured / deferred / connected /
  failed 状态、协商协议版本、server identity、工具数量和脱敏后的最近错误
- 支持手动重新读取 `.local/mcp_config.json` 并重建对应动态工具；配置变化不要求
  自动文件监听
- 重载必须关闭被删除或被替换的 stdio session，清理失效 schema cache，且不得
  影响未变化 server
- 为 REPL 增加最小 `/mcp status` 和 `/mcp reload` 入口，不扩展为通用 MCP
  管理控制台

### M.3 长工具调用的进度与取消
- 为 `tools/call` 传递 progress token，并消费 SDK progress notification
- 将进度映射为现有结构化事件或诊断状态，避免长时间调用只能等待最终结果
- 用户取消、agent run 取消和 timeout 使用同一取消路径；确认 server 不响应取消
  时仍能回收子进程
- 不为 MCP 单独建立第二套事件总线或后台任务系统

### M.4 工具元数据覆盖
- 实现 `.local/mcp_config.json` 中当前被跳过的 `overrides`
- 仅允许宿主侧覆盖 `risk`、`read_only`、`concurrency_safe`、启用状态和展示说明
- server annotations 只作为提示；权限和并发属性最终由宿主配置与保守默认值决定
- 配置按 `server + tool` 精确匹配，未知工具名给出诊断，不静默忽略

### M.5 真实 server 兼容回归
- 固定一个官方 `modelcontextprotocol/servers` 示例作为 stdio conformance smoke
  test，覆盖发现、调用、structured content、list changed 和关闭生命周期
- 单元测试继续围绕 SDK adapter 的可观察行为，不断言 SDK 私有实现细节
- 外部 server 测试与默认离线 pytest 分离，避免网络或 npm 环境成为本地测试前提
- 记录启动失败、协议不兼容、schema 非法、调用超时和异常退出的稳定诊断格式

### MCP 暂不实现
- Streamable HTTP / SSE、OAuth 和远程 server 凭据管理：出现必须使用的真实
  coding server 后再做
- resources / prompts：当前工具调用已覆盖 coding agent 主路径；先不增加第二套
  发现、注入和权限语义
- sampling / elicitation：会形成 server 反向驱动模型或用户交互的新信任边界
- 通用连接池、自动故障转移、周期健康轮询和 MCP marketplace：当前规模没有收益
- 为追求协议覆盖而透传所有 notification：只消费会影响工具目录、进度、取消和
  workspace 边界的事件

---

## Tasks — 剩余一致性问题

### T.1 `advance_task` 乐观锁
- 为 `advance_task` schema 增加 `expected_version`
- 完成主任务前检查当前版本，冲突时返回与 `update_task` 一致的可重试诊断
- 下游依赖解除仍在同一目录锁内完成，不为每个下游任务要求调用方提供版本

### T.2 `blocked_by` 显式清空
- `update_task(blocked_by=[])` 应清除已有依赖；当前 truthy 判断无法区分“未传入”
  与“显式传空数组”
- handler 使用键存在性判断，并保留 schema 的 `additionalProperties: false`

---

## Progress — 默认输出路径

### P.1 移除函数级 `claude-progress.txt` 回退
- `paths.progress_summary` 默认设为 `.local/progress_summary.md`，并补充到
  `CONFIG.md`
- `save_progress()` 不再自行选择 Claude Code 专用文件名；路径由装配层传入
- 直接调用底层函数时若没有路径，使用项目级 `.local/progress_summary.md`

---

## Daemon — 持久自定义任务语义

### D.1 明确恢复边界
- 当前 `.local/daemon_tasks.json` 只能恢复自定义任务名称，不能恢复进程内
  callable；文档和状态接口必须明确显示 `callable pending`
- 只有出现真实跨进程自定义任务需求时，才设计可序列化的任务类型注册表；不持久化
  Python callable，也不引入动态代码加载

---

## Config — 合并后强类型校验

### C.1 使用 Pydantic 校验最终配置
- 保留当前全局、项目、本地配置的 raw dict 深度合并语义，在所有来源合并及环境
  变量覆盖后执行一次统一模型校验
- 使用 Pydantic 模型替代 `_config_from_dict()` 的手写递归转换，拒绝未知字段、
  错误标量类型、非法 enum 和错误的嵌套结构
- 错误信息必须包含完整字段路径和配置来源提示，避免只暴露底层 validation error
- 不引入 `pydantic-settings` 或第二套配置发现机制；Pydantic 只负责最终数据模型
  与验证
- 迁移时保持现有显式键覆盖、profile 继承、hook source 标注和环境变量优先级

---

## Experimental State — SQLite 迁移评估门槛

### S.1 达到规模或并发瓶颈后再统一存储
- 当前继续使用 JSON / JSONL + `filelock`；不立即迁移
- 当出现以下任一真实证据时，评估使用标准库 `sqlite3` 统一 tasks、mailbox 和
  orchestration：
  - mailbox 或 task 扫描、清理、索引维护产生可测量性能瓶颈
  - 多进程写入冲突或原子更新逻辑继续扩张
  - ACK、lease、版本和查询索引需要跨多个旁路文件保持事务一致性
  - 状态迁移与损坏恢复成本明显高于单文件存储的可读性收益
- 评估时先设计最小 schema、事务边界、WAL/锁策略和现有文件的一次性迁移；
  session transcript 与 MCP schema cache 不在首轮迁移范围
- 没有 benchmark、故障案例或真实并发需求时，不为“统一技术栈”引入数据库
