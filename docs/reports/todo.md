🔴 P0 — 阻塞性 / 生产问题
问题	文件	说明
except _BoundaryEscapeError: pass	permission_model.py:792	静默吞下路径边界逃逸异常——安全相关，应至少加 log
except Exception: pass (x2)	worktree.py:356,363	过于宽泛，静默吞 git 命令异常
_tool_result_text 死函数	result.py:178	与 tool_hooks.py:50 重复，无人调用
tests/conftest.py 与 _helpers.py 重复	tests/	_LogCapture/assert_logs/assert_no_logs 两份几乎相同的实现
🟠 P1 — 过度设计 (Over-engineering)
问题	文件	说明
双重事件类型系统	agent/events.py + harness/agent_runtime/events.py	AgentEvent → StructuredAgentEvent 翻译层 135 行，13 个分支；加新事件需改两套类型commit
双重结果模型	result.py + results.py	StructuredAgentResult 与 AgentLoopResult 重复 3 个 stopped_by_* 计算属性
StructuredAgent 500 行包装 Agent	structured.py	包装 + 编排 + 状态 + 结果翻译，四职责合一
ToolSpecAdapter 双向适配器	tool_adapter.py:35-192	8 个属性委托 + 反向适配器 (用 asyncio.run() 反模式)
AuditRecord 构造重复	tool_gate.py:366 + tool_audit.py:16	~90 行近乎相同的 audit 记录构造逻辑
app_contract.py 三个 Protocol	cli/app_contract.py	三个 Protocol 组合定义 ReplApp，但只有 XcodeApp 一个实现
17 个 Protocol 只对应一个实现	遍布 10+ 文件	ContextAssembler、ExecutionEnv、MailboxTransport、MarkdownRenderer、FileOperations、GrepOperations、LsOperations、CancellationSignal、ContextCollector、PolicyEvaluator、ModelProfileProto 等
提示词三 Builder 缓存	prompting/builder.py	StableRegionBuilder + DynamicRegionBuilder + VolatileRegionBuilder 各自独立缓存
13 个配置 dataclass	config.py	ToolsRuntimeConfig (1字段)、SkillsRuntimeConfig (1字段)、DaemonRuntimeConfig (2字段) 等，多数仅用于分拆
Dual Signal 总线 (blinker)	observability/hooks.py	register + subscribe 双重信号注册，6 种事件类型 + HookEvent Literal + 6 个事件 dataclass
StreamCodec 6 个 Protocol	ai/providers/stream_codec.py	6 个 structural protocol 描述 OpenAI 响应形状，可用 TypedDict 代替
4 个服务分组 dataclass	assembly.py:69-115	SharedServices、OptInServices、ResolvedConfig、SharedInfra，每个仅用于一处
类型变量 T 未使用	file_mutation_queue.py:7	TypeVar("T") 定义后从未做类型参数化
✅ P3 — Wrapper / 中间产物 / 可简化（已完成）
问题	处理
HistoryManager 薄包装	移除包装类，StructuredAgent 直接维护消息列表；反序列化归入 message_codec.py
_tool_scheduling.py / _tool_validation.py	内联到 tool_execution.py 并删除两个单函数模块
policy_for_mode 每次创建新实例	改为复用三个无状态策略单例
cancellation.py 包装 threading.Event	CancellationToken 直接继承 threading.Event，保留取消原因语义
REPL 主循环 4-5 层嵌套	提取输入和快照轮次辅助函数，主循环改为线性分派
build_providers 纯委托函数	删除转发函数，build_app 直接调用 build_provider_bundle()
🟣 P4 — Non-xcode / 超出本地 CLI 范围
问题	文件	说明
harness/task_store.py + task_progress.py + orchestration_store.py	~1300 行	完整任务编排系统（依赖拓扑、lease、kanban、CRUD），远超本地 coding agent 需求
harness/mailbox.py 跨进程/跨机器传输	540 行	声明 跨进程/跨机器 Agent 通信，但有且仅有本地文件实现
harness/daemon.py 后台心跳守护进程	380 行	周期检查 git dirty/mailbox/task，含自愈重启
Git Worktree 沙箱	coding_agent/tools/worktree.py	高度专业化的 git 功能，测试文件 342 行
ChatGLM 和 MiMo provider	ai/providers/chatglm.py:148行, mimo.py:50行	生产环境中不会使用的第三方云 provider
faux provider 暴露在公开 API	ai/providers/__init__.py	测试 mock 和 faux_text/faux_tool_call 等辅助函数在生产 API 表面
🟢 P5 — 架构约束 / 已知但可接受
问题	说明
agent/ 导入 harness/？	不违反当前分层——agent/ 导入 harness/ 的反向依赖是已知折衷
函数体内延迟导入	assembly.py、repl.py、repl_commands.py 多处——用于避免循环导入，有理由但难看
harness/observability/__init__.py 导出 46 个符号	追踪依赖困难，但这是性能优化（一次 import）
harness/skills_registry.py 1000 行	单体文件包含 discovery/index/load/render，功能耦合度高
