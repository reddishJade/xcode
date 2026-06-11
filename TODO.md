# src/xcode/ai 重构清单

## 高

- [x] 1. 模板方法模式断裂：`_stream_sync` 基类实现是死代码（`openai_compat.py:75`）
- [x] 2. `_record_usage` 三份重复代码（`deepseek.py:195` / `chatglm.py:144` / `mimo.py:65`）
- [x] 3. `ChatGLMProvider.stream()` 绕过基类消息归一化（`chatglm.py:59`）

## 中

- [x] 4. `cache.py` 混合两个无关概念：缓存统计 + 工具 Schema 规范化
- [x] 5. `_build_thinking_params` 调用不一致（OpenAI 不调用）
- [x] 6. API key 存储为实例属性（安全风险，死存储，`openai_compat.py:53`）
- [x] 7. `ChatGLMProvider._clean_reasoning_content` 无条件深拷贝（`chatglm.py:136`）

## 低

- [x] 8. `ChatGLMProvider.__init__` 冗余存储 `self.base_url`
- [x] 9. 各子类 `_stream_sync` 签名不一致
- [x] 10. `factory.py` 函数体内 `import` 规避循环导入（`factory.py:165`）

---

# src/xcode/agent 重构清单

## P0

- [ ] 1. `agent_loop.py`（834 行）过大：混合 provider/watchdog/event/helper 逻辑，需拆出 `_provider.py` 并将 watchdog 更新函数移入 `watchdog.py`
- [ ] 2. `messages.py` 领域类型（dataclass）与 LLM 格式转换逻辑混在一起，`convert_to_llm` 及辅助函数应抽到 `message_converter.py`

## P1

- [ ] 3. `config.py`（208 行）职责过多：Context types / Hook type aliases / Metrics / Result / AgentLoopConfig 挤在一处，建议抽出 `hooks.py` 和 `results.py`
- [ ] 4. `watchdog.py` 硬编码工具名列表（`is_file_mutation_tool` / `is_file_read_tool`），应改为可配置
- [ ] 5. `__init__.py` 为空，无公共 API 导出，调用方需深层导入
- [ ] 6. `tool_execution.py`（470 行）边界过大，可抽出参数校验和分批逻辑

## P2

- [ ] 7. `agent.py` 用 `dataclasses.replace` 隐式修改 config，建议改为显式 Builder 模式
- [ ] 8. `compaction.py` 的 `estimate_tokens_simple` 用 `len//4` 过于粗糙

---

# src/xcode/harness 重构清单

## 高

- [ ] 1. `StructuredAgent.__init__` 参数过多（17 个），建议归并为配置对象
- [ ] 2. `build_tool_registry` 过长（114 行），嵌套 + 内联闭包，建议拆为独立工厂函数
- [ ] 3. `build_loop_config` 过长（119 行）且创建 5 个闭包，建议提取为命名函数或策略对象
- [ ] 4. 封装泄露：`structured.py` 访问 `_FallbackSwitchingProvider` 私有属性；`subagent.py` 访问 `runner._jobs`；`task_store.py` 调用 `store._locked()`，建议加正式公开接口

## 中

- [ ] 5. 消息反序列化重复：`history_manager.py` 与 `message_codec.py` 的 `_message_from_dict`/`_tool_call_from_dict` 几乎相同，应合并
- [ ] 6. `session.py` 中 `fork_into` 与 `fork_clean_into` 大量重复，应提取 `_fork_base`
- [ ] 7. `skill_loader.py` 中 `_parse_frontmatter` 与 `_read_frontmatter` 重复实现
- [ ] 8. `skill_loader.py:8` 奇怪的相对导入 `from ..harness.skills`，应改为 `from ..skills`
- [ ] 9. `TaskProgress` 是纯静态方法的类，应改为模块级函数
- [ ] 10. 生产代码混入测试代码：`skills.py:134` 的 `run_tool`、`execution_env.py:130` 的 `MockExecutionEnv` 应移至 `tests/`

## 低

- [ ] 11. `skills.py` 重复的模块 docstring（行 1-4 与行 26-30）
- [ ] 12. 多处函数体内延迟导入（`filelock`、`asyncio`、`os` 等），建议提到模块顶部
- [ ] 13. `events.py:174` 的长 `isinstance` 链（9 分支），可用字典分发替代
- [ ] 14. `hooks.py:142` 的长 `if-elif` 链（7 分支），可用字典分发替代
- [ ] 15. `tool_events.py` 仅 11 行，只有一个 `ToolResult` dataclass，建议内联
- [ ] 16. `config.py` 中 `_resolve_value` 对 `Union` 类型处理过于简单，复杂联合可能选错

---

# src/xcode/coding_agent 重构清单

## 高

- [ ] 1. 拆分 740 行的 `file.py`：建议拆为 `file_tools.py`（工具构造+Schema）、`file_handlers.py`（读写编辑逻辑）、`file_image.py`（图片处理）
- [ ] 2. 提取共享常量：`DANGEROUS_PATTERNS`、`HIGH_RISK_WRITE_COMMANDS`、超时常量从 `bash.py` 提取到独立 `_constants.py`，消除 `bash.py` 与 `native_shell.py` 之间的隐式跨模块依赖

## 中

- [ ] 3. 合并两个风险评估器：`bash.py:_bash_risk_evaluator` 与 `native_shell.py:_native_shell_risk_evaluator` 逻辑重复，应共享同一套评估规则
- [ ] 4. 简化 `bash.py` 的 `OutputAccumulator` 使用：`_render_bash_output` 创建累加器仅为了追加两段文本，直接用 `truncate_tail` 即可

## 低

- [ ] 5. 合并 `truncate_head`/`truncate_tail`：两者共享大量逻辑，可加 `direction` 参数合并
- [ ] 6. 拆分 `worktree.py:remove` 方法（~70 行）：脏检查、upstream 检测、cherry-pick 检查等应提取为独立方法
- [ ] 7. 处理 `coding_agent/__init__.py` 为空：建议导出 `registry.py` 的公共 API，或用注释标记为 namespace package
- [ ] 8. 记录 `_prepare_edit_arguments` 的 LLM 防御性设计边界：明确哪些 LLM 输出格式偏差触发 fallback，以及移除条件

---

# src/xcode/cli 重构清单

## 高

- [ ] 1. `_ReplTurnRenderer` 过重（`repl.py:332-540`，208 行）：混合 reasoning 预览、答案流式、tool call 跟踪、tool result 记录、tool update 进度、tool group 聚合等多种职责，应拆分为 `ReasoningRenderer`、`ToolCallRenderer`、`AnswerStreamRenderer` 等独立类或提取到单独模块
- [ ] 2. `run_setup_wizard()` 过长（`setup_wizard.py:106-261`，155 行）：覆盖 provider 选择、API key 输入、base URL、model 选择、transport 映射、thinking/effort 配置、确认、合并、写入等 10+ 步骤，每一步应提取为独立函数
- [ ] 3. `ModelControlApp` 协议重复定义：`app_contract.py:28-40` 与 `repl_settings.py:15-27` 签名相同，`repl_settings.py` 应导入而非重定义

## 中

- [ ] 4. `_run_agent_turn()` 参数过多（7 个），应引入 `AgentTurnContext` dataclass
- [ ] 5. `deep_merge()` 定义在 `run_setup_wizard()` 内部（`setup_wizard.py:233-240`），应提为模块级函数
- [ ] 6. `_console` 模块级全局变量（`repl_tools.py:42`），应改为显式参数或输出抽象
- [ ] 7. `CommandContext.app` 类型为 `object`（`commands.py`），导致 `cmd_*` handler 中大量 `getattr()` 调用，应使用 `app_contract.py` 中的 Protocol

## 低

- [ ] 8. `repl_commands.py` 中所有 `cmd_*` 函数缺少 docstring
- [ ] 9. `cmd_sessions()` 内联 `import questionary`，与其他文件不一致
- [ ] 10. `handle_permissions()` 无子命令匹配时隐式 fallthrough 到 `list_permissions()`，缺显式 `else`
- [ ] 11. `_ReplTurnRenderer`、`LiveMarkdownStream`、`LiveReasoningPreview` 缺少 docstring
