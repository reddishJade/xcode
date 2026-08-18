# 测试策略

## 质量标准

测试对象是组装后的 Xcode，而不只是单个函数。每项行为至少在最接近其风险的
层级验证；跨层不变量必须有契约测试。

## 测试层级

### 纯逻辑测试

用于 parser、codec、权限规则、路径计算、事件投影和渲染摘要。测试应确定、
快速，不访问外部 provider 或终端。

### 组件契约测试

验证相邻层之间的完整数据：

- ToolOutput -> AgentToolResult -> ToolResultMessage；
- runtime event -> session codec -> replay；
- compaction replacement -> current surface -> restart；
- provider 实际请求 -> `provider_request` envelope；
- composition generation -> run snapshot -> `provider_request.composition_id`；
- subagent lifecycle -> parent session ledger；
- child descriptor/index lineage -> 独立 session surface -> cold continuation；
- render intent -> CLI/TUI projection。

### 真实组合测试

`test_app_composition.py` 使用真实 `build_app()`、真实 registry、真实 session
存储和 replay。只在网络边界替换 provider，不能替换 app builder、loader、
recorder 或 replayer。该测试必须覆盖：

1. 最小 app 成功组装；
2. 第一轮请求与回答落盘；
3. envelope 与 provider 实际输入逐字段一致；
4. 新 app 从相同 session 恢复；
5. 第二轮 provider 能看到第一轮上下文。

### 外部依赖验证

真实 provider、终端 UI、MCP server 和平台相关 shell 行为按需手工验证。
不能用 HTTP 200、mock loader 或另一个服务实例代替用户实际运行路径。

## 必跑命令

```sh
uv run ruff check src/
uv run pyright src/
uv run pytest src/xcode/tests -q --tb=short
```

修改局部行为时先运行聚焦测试，提交前运行完整套件。Pyright 的既有 warning
可以单独治理，但新增代码不得增加 error。

## 变更要求

- 新 session event 必须同时有编码、持久化和回放测试；
- 新模型输入必须出现在 `provider_request` envelope；
- 新工具呈现必须使用类型化 intent，并覆盖两个宿主的共享投影；
- 新组合参数必须由 `build_app()` 的真实测试覆盖；
- composition 输入必须测试发布后隔离，运行时替换必须产生新 generation；
- 删除或更改接口时，同一提交直接更新所有调用方与测试，不添加兼容分支；
- 文件和终端依赖使用窄本地协议注入，不能引入远程执行假设。

## 事故回归

每份正式复盘至少产生一个能防止复发的自动化测试，或明确记录为何只能手工
验证。测试名称应描述被保护的不变量，而不只复述具体 bug。
