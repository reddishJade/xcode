# Subagents 子代理架构

在处理长程复杂任务（如大型代码重构、全库安全扫描、多模块独立开发）时，单一会话容易出现上下文混乱。Xcode 实现了**基于独立 Durable Session 的子代理（Subagent）委托架构**。

---

## 1. 架构核心不变量

* **独立 Durable Session**：子代理并不是父会话中的临时对象，每一个子代理实例拥有独立的 Session ID、事实账本（JSONL）、消息历史与 Context 预算；
* **谱系与生命周期追踪**：父会话通过 `subagent_run` 事件精准记录子代理的 Batch ID、Run ID、启动/完成/失败状态与总结摘要；
* **权限域继承与不可扩张**：子代理继承父会话的权限边界，且无法越权获取父会话未授权的能力。子代理不包含二次委托工具，最大委托深度严格限制为一层。

---

## 2. 运行模式：One-shot 与 Continuable

Xcode 支持两种子代理执行模式：

### 2.1 One-shot 模式（并行单回合任务）
适合可以独立完成的勘察、搜索或单元测试验证任务：
* 父 Agent 可并行启动多个 One-shot 子代理；
* 子代理执行完成后自动汇总结果返回给父 Agent，并自动释放（Release）进程资源。

### 2.2 Continuable 模式（多轮对话协作）
适合需要多步骤交互、反复推敲的复杂子任务：
* 创建具有持久化生命周期的子代理；
* 父 Agent 通过 `subagent_continue` 按子代理 Session ID 进行多轮 FIFO 消息传递；
* 即使系统重启，也可从磁盘账本中“冷恢复”子代理上下文继续对话。

---

## 3. 工具配置与共享

默认情况下，子代理仅拥有核心代码工具。如需为子代理开放额外工具（例如 `todowrite`），可在 `xcode.config.json` 中配置：

```json
{
  "tools": {
    "subagent_extra_tools": ["todowrite", "websearch"]
  }
}
```

---

## 4. 优雅关闭与资源回收 (Child-first Teardown)

当退出应用或中断任务时，Xcode 采用 **Child-first 逆序清理策略**：
1. 向所有活跃的子代理发送取消信号，并等待进行中的操作平稳收敛（Settle）；
2. 逆序释放子代理的 Activation 句柄；
3. 确保所有子代理的账本落盘完成后，再关闭父会话的 MCP 等共享网络与进程资源。

---

← **上一篇**：[Skills 技能系统 (skills.md)](skills.md) | **下一篇**：[长期记忆系统 (memory.md)](memory.md) →

