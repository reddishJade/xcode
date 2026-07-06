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

## P3 — 值得改进的设计细节

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
