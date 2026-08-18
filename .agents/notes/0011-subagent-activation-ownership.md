# Subagent activation has direct-parent ownership
Status: Implemented
Date: 2026-08-18

## Context

durable child session 解决了身份、回放和 continuation，但 session identity 本身不
代表当前进程正在运行它。若没有单独的 activation 语义，冷恢复可能静默获得新增
工具，任意当前 session 都可能 interrupt 或 release child，父 app 关闭时也可能先
销毁共享资源而留下仍在运行的 child。

## Decision

每次 child 在进程内物化都创建新的 `activation_id`，并在 child log 追加
`subagent/activation` 的 materialized/released 事件。run 事件同时保存
`activation_id`，因此 durable session identity、activation epoch 和单次 run 是三个
不同层级。

`SubagentSessionManager` 保存以直属父 session ID 分组的 live ownership。只有当前
direct parent 可以 continuation、interrupt 或 release；release 仅允许 idle
activation，且永不删除 child session。one-shot settle 后自动 release，continuable
child 可显式 release，之后再从日志冷物化。

descriptor 冻结 child 初次发布的工具名。冷物化时可用 registry 必须与该集合取
交集：工具被产品删除时可以收缩，产品后来增加工具时 child 不能自动扩权。child
gate 使用独立 session correlation，并从明确绑定的父 gate 派生；child 不读取环境中
其他 session 的隐式权限。

child cancellation token 同时观察局部 interrupt 与父 cancellation。父 app 关闭时，
subagent closer 排在其他共享资源之前：先 cancel 所有 live child，在有界时间内等待
turn settle，再按逆物化顺序 release。若 child 未能 settle，关闭失败且不得继续销毁
后续共享资源。

旧的无 activation ID 事件、跨 session 控制和冷恢复时直接采用最新工具集合的路径
直接删除，不提供兼容解析或适配层。

## Consequences

- child session 可跨 activation 延续，但每段进程内执行都有可审计 epoch。
- continuation 继承身份和已发布能力上限，不等于继承未来权限。
- interrupt 只停止当前 turn；durable session 和尚未 claim 的 inbox 继续存在。
- parent teardown 明确拥有 child-first 清理责任，不会静默产生孤儿 activation。
- child registry 不含 delegation 工具，因此当前最大 delegation depth 明确为一层。

## Validation

- session recorder 测试覆盖 activation 和 run ID 的严格 schema。
- direct-parent 测试拒绝切换父 session 后控制已有 child。
- cold activation 测试证明产品新增工具不会进入既有 child composition。
- cancellation 测试证明 child reset 不会遮蔽父取消。
- close 测试证明 activation 被 release，而 durable child session 保留且 manager
  不再接受新 turn。
- Ruff、Pyright 与完整测试套件验证单一生命周期路径。

## Supersedes

None
