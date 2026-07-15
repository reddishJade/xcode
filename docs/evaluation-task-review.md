# Git 历史任务审核

本清单用于阶段 1 的历史修复任务入集。候选提交不是 Task；只有全部必选项有证据后，
才能写入版本化数据集。

## 必选检查

1. 来源仓库、父提交、修复提交、许可证和上游问题语义可追溯。
2. prompt 能描述用户可观察的问题，但不泄露最终 patch、隐藏测试或实现步骤。
3. 父提交可在隔离环境安装和运行，且不依赖已失效的外部服务。
4. 修复提交包含或能够导出确定性的回归证据；隐藏 verifier 在父提交失败、在修复提交
   通过。
5. 隐藏测试、参考 patch、verifier 命令与输出均可置于 Agent 工作区之外。
6. 任务要求真实读取、定位、修改和验证源码，不是格式化、机械重命名、玩具函数或
   Eval pipeline 自测。
7. 允许路径、禁止路径、原有回归检查和 policy 检查可确定性执行。
8. 预计能在统一 Trial 预算内运行；依赖、平台和已知 oracle 缺口已经记录。
9. 人工审核确认问题描述不依赖查看最终解法，且成功结果具有工程决策价值。

## 首轮候选

| 候选修复 | 父版本问题与价值 | 当前结论 | 待补证据 |
|---|---|---|---|
| `337b989` preserve fallback wrapper across `set_model` | 热切换主模型会静默丢失 fallback 容灾层；覆盖 provider 恢复与运行时配置能力。 | 已入 `xcode-history-v1` | 阶段 2 基线重复运行。 |
| `da58a39` avoid repeated watchdog masking idle watchdog | 连续工具错误被错误归因为重复调用；覆盖循环控制、工具失败反馈和恢复诊断。 | 已入 `xcode-history-v1` | 阶段 2 基线重复运行。 |
| `6c1a27f` make `/thinking off` disable reasoning | 配置的 effort 覆盖关闭指令；覆盖模型控制与请求装配。 | 已入并完成真实闭环 | 1/2 Trial 成功仅用于阶段 1 闭环；阶段 2 需更多重复。 |
| `797bce1` reset session grants on new sessions | 新会话错误继承临时授权，永久授权与恢复中的活动会话又不能被一并清空；覆盖权限与 session 生命周期。 | 已入 `xcode-history-v1` | 阶段 2 真实模型 Trial。 |
| `dd296ea` dispatch external observer hooks in background | 非关键 observer 同步阻塞事件发射且异常可中断 Agent；覆盖主循环延迟、观测与故障隔离。 | 已入 `xcode-history-v1` | 阶段 2 真实模型 Trial。 |
| `994bc24` diagnose unknown MCP overrides | 拼写错误或过期的精确工具 override 被静默忽略；覆盖 MCP 配置反馈与外部能力可诊断性。 | 已入 `xcode-history-v1` | 阶段 2 真实模型 Trial。 |
| `5691546` enforce parallel tool worker limits | 配置的 worker 上限未进入 core loop，parallel batch 无界启动 handler；覆盖工具并行与资源控制。 | 已入 `xcode-history-v1` | 阶段 2 真实模型 Trial。 |
| `9fcee80` skip unindexable snapshot paths | 单个 Windows reserved/unindexable 路径会让整轮 hidden Git tree snapshot 失败；覆盖 session snapshot 与文件系统恢复。 | 已入 `xcode-history-v1` | 阶段 2 真实模型 Trial。 |

八个候选均已入集并完成父失败/修复通过的隔离重放；提交说明、参考 patch 和原隐藏测试
不会进入 Agent 工作区。`6c1a27f` 已由真实模型运行两次，产生一次成功和一次有效能力
失败；这个小样本只证明阶段 1 闭环，不构成阶段 2 基线。

## 差分重放证据

### `337b989`（2026-07-15）

- 父提交：`7441803cdb3ccd98cb3937ba2f7b83e1c07f5e05`。
- 修复提交：`337b9891e4392e3f947f8b809758730b8b7e6b9a`。
- 恢复方式：分别使用 `git archive` 写入独立 `/tmp` 工作区；工作区不含 `.git`。
- verifier 边界：从修复提交提取的回归测试放在第三个、非工作区目录，通过
  `PYTHONPATH=<workspace>/src` 对两个版本运行完全相同的命令。
- 父版本结果：2 failed / 2 passed；失败分别证明 wrapper 被裸 provider 替换、旧
  fallback 计数未重置。
- 修复版本结果：4 passed。
- 结论边界：这只证明候选具有确定性 oracle，尚未证明真实 Xcode 能解决任务；原测试
  含实现提示，不能原样给 Agent，后续隐藏包还必须增加允许路径和全套回归检查。

后续将原测试改写为不向 Agent 暴露的行为 oracle，并在只应用生产 patch 的父版本
工作区完成参考重放：隐藏行为测试 1 passed，provider/model 稳定回归切片 17 passed。
完整历史套件超过 300 秒 verifier 预算，故 `regression_free` 仅覆盖已声明切片；这项
oracle 缺口保留在 Task 的 `known_limitations`，不能扩大解释为全仓库无回归。

### `da58a39`（2026-07-15）

- 父提交：`d17a8db80fcbd7c521c9dc67f1c5314981d1d06d`。
- 同一外置回归文件：父版本 5 failed / 6 passed，修复版本 11 passed。
- 入集后的隐藏 oracle 覆盖全错误重复不触发重复 watchdog、成功重复仍受限、错误结果
  清除旧成功计数；只应用生产 patch 的参考重放为 3 passed，父版本稳定回归为 6 passed。
- 工程价值：测量循环控制是否保留工具失败信号，而非只检查某个 watchdog 事件出现。

### `6c1a27f`（2026-07-15）

- 父提交：`c0d4426f5bf8590c29530866f7b52423e331cc47`。
- 同一外置回归文件：父版本 1 failed / 3 passed，修复版本 4 passed。
- 隐藏 oracle 同时检查显式 off 覆盖已配置 effort、启用路径保持 effort、应用状态明确
  显示 off；只应用生产 patch 的参考重放为 3 passed，稳定 provider/model 回归为
  10 passed。
- 工程价值：测量 Xcode 模型控制是否真正改变 provider 请求，不把 CLI 文案或 Agent
  自报状态当作成功。

### `797bce1`（2026-07-15）

- 父提交：`7ec0db84d759f0a5e9436198450674492c89b85e`。
- 隐藏 oracle 分别验证 `clear_history()` 清除临时授权但保留永久授权，以及
  `load_history([])` 开始新授权会话而非空历史恢复保持活动授权。
- 同一外置 verifier：父版本行为 2 failed / 稳定回归 2 passed；修复版本行为 2 passed /
  回归 2 passed；只向父版本应用两个允许生产文件的 patch 后同样为 2 passed / 2 passed。
- 完整历史 structured-agent 测试超过 90 秒，因此回归切片限定为父版本已有的 history
  restore 与 provider conversation reset 两项测试；覆盖缺口已写入 Task。
- 工程价值：测量 session 与权限反馈的生命周期是否一致，避免新任务静默继承旧任务
  的临时授权。

### `dd296ea`（2026-07-15）

- 父提交：`eb4c6ec50ab1472d86b962c89effbcd8f5186257`。
- 隐藏 oracle 通过 assembly 的真实 hook 装配路径，用线程屏障验证非关键 observer 尚未
  完成时事件发射线程已经返回，并验证 observer 异常不会回传给发射者。
- 同一外置 verifier：父版本行为 2 failed / 稳定回归 2 passed；修复版本和只应用两个
  允许生产文件 patch 的参考工作区均为行为 2 passed / 回归 2 passed。
- 旧 `test_hook_manager_wires_all_non_pre_events` 假设 emit 返回时外部命令已经完成，与新
  语义直接冲突，因此不伪装成不变回归；隐藏 oracle 使用 `drain_background()` 验证最终
  完成，回归切片保留外部执行脱敏和 blocking pre-tool deny 行为。
- 工程价值：判断观测扩展是否给 coding loop 引入尾延迟或故障耦合，不以线程/队列事件
  本身作为能力结果。

### `994bc24`（2026-07-15）

- 父提交：`169edcc84bb0b54e1c63e325decee3bee59527aa`。
- 隐藏 oracle 通过正式 `build_mcp_tools()` 路径提供本地 discovered catalog，验证未知
  精确 override 产生稳定 warning、通配符不被误报且有效工具仍注册并继承默认元数据。
- 同一外置 verifier：父版本行为 1 failed / 稳定 MCP 注册回归 2 passed；修复版本和
  只应用 `src/xcode/harness/mcp/tools.py` patch 的参考工作区均为行为 1 passed / 回归
  2 passed。
- 工程价值：测量 MCP 配置错误是否形成 Agent/用户可定位反馈，避免把外部能力缺失误判
  为模型无能；grader 使用本地 client double 保持确定性，不把 fake MCP 算作能力分。

### `5691546`（2026-07-15）

- 父提交：`b5cb3dcfb41ecfc1cd2264de5007cb232f36cd78`。
- 隐藏 oracle 用受控异步执行替身测量 6-call parallel batch 的最大活跃数严格等于配置 2，
  并通过正式 `build_loop_config()` 验证公开 `AgentConfig.tool_workers` 传播到 core loop。
- 同一外置 verifier：父版本行为 2 failed / 稳定回归 2 passed；修复版本和只应用三个
  允许生产文件 patch 的参考工作区均为行为 2 passed / 回归 2 passed；修复行为测试连续
  5 次通过。
- 历史 direct-execute 测试在父环境产生不退出线程，因此不进入 grader；回归切片保留
  execution-mode metadata 与 sequential barrier 分区，实际 bounded execution/result count
  由隐藏 oracle 覆盖。
- 工程价值：判断并行工具吞吐是否以可控资源上限实现，为后续工具并行 Variant 的效果/
  成本归因提供真实任务，而不是把并发计数器自身当作能力分。

### `9fcee80`（2026-07-15）

- 父提交：`1fd099d0eb0641a0b208a001921be40549fc0d4e`。
- 隐藏 oracle 注入一个路径级 `git add` 失败，验证有效路径仍被逐个处理、tree id 正常产生、
  失败路径进入 skipped diagnostics，并验证常见 `nul` reserved-name 形式预先排除。
- 同一外置 verifier：父版本行为 2 failed / 正常 snapshot 回归 1 passed；修复版本和只应用
  `src/xcode/harness/snapshot.py` patch 的参考工作区均为行为 2 passed / 回归 1 passed。
- 正常回归在真实临时 Git tree 中验证普通源码进入 snapshot、`.env` secret 被排除；失败
  注入用于跨平台确定性，不把 Linux 上可创建 `nul` 文件的偶然行为当作 Windows oracle。
- 工程价值：测量 session snapshot 是否能容忍局部文件系统异常，避免一次特殊路径破坏
  整轮 undo/restore 证据。
