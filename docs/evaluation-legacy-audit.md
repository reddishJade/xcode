# 旧 Eval 逐模块审计

本表审计 `eval-legacy` 分支的 `src/xcode/evals/` 与 Eval fixtures。旧分支仅供审计；
任何“选择性迁移”都要求重新满足
[评测战略](evaluation-strategy.md) 的真实运行与独立 verifier 契约。

| 旧模块 | 决定 | 理由与后续归属 |
|---|---|---|
| `schema.py` | 重写 | 旧 Task 把答案片段、预期工具轨迹和 validation 命令放在同一对象，且 TrialResult 以 Agent answer/grader 为核心。新 schema 拆分 Agent 可见 Task 和控制面 VerifierSpec。 |
| `tasks.py` | 废弃，少量候选进入 pytest | smoke、tool policy、memory event 和自修改 Eval 代码场景主要验证 pipeline；三个 fixture 是人工玩具项目且验证材料 Agent 可见，不计能力分。 |
| `runner.py` | 重写 | `_StaticProvider` 与离线事件流只可作为基础设施测试；真实 executor 必须走现行 `build_app()` 并记录真实 provider 配置。 |
| `sandbox.py` | 选择性重写 | 临时目录、复制与 patch 思路可参考，但必须支持 Git 精确版本恢复、每 Trial 独立状态和 verifier 边界。 |
| `graders.py` | 废弃能力判分，公式可迁入 pytest | answer/tool-call/evidence 自报不能判定任务成功；独立 verifier 的确定性结果将取代它。纯聚合公式可选择性重写并单测。 |
| `tracing.py` | 选择性重写 | trace 采集有审计价值，但旧 pipeline 事件不是新 schema 依赖；后续从正式 Xcode runtime 事件生成只读 artifact。 |
| `reporting.py` | 延后选择性重写 | 旧报告混合 fake/offline 与 real 分数。阶段 2 只从原始 Trial artifact 离线重建摘要后才引入展示层。 |
| `validation.py` | 迁入 pytest 候选 | 命令解析和超时属于基础设施 Test；Agent 工作区内 validation 不等于隐藏 verifier。 |
| `benchmarks.py` | 延后重写 | prediction 生成不是官方评分；阶段 5 adapter 必须保留官方 task、环境、verifier 和评分语义。 |
| `adapters/registry.py` | 延后重写 | 注册机制可参考，需等待统一 Task/Artifact 协议稳定。 |
| `adapters/swebench.py` | 延后重写 | 只有调用官方 verifier 并保存其证据才算外部 benchmark Eval。 |
| `cli.py` | 重写 | 旧入口暴露 offline/fake suite 和 pipeline 分数；新 CLI 只能是 workspace、executor、experiment 的薄入口。 |
| `examples/eval/fixtures/*` | 保留在旧分支；不迁移能力集 | buggy-math、string-utils、tiny-calculator 是公开验证的玩具项目，可用于基础设施 Test，不满足真实来源与隐藏验证门槛。 |
| `docs/evaluation-guide.md` | 仅历史参考 | 描述旧命令与旧分数，不能作为新系统操作指南。 |

## 阶段 0 结论

- 旧 Eval 没有任何场景可直接标为新能力 Eval。
- 新领域模型不导入 provider、agent pipeline、旧 grader 或报告事件。
- 阶段 1 首个任务必须来自可审计 Git 历史，隐藏材料保存在 Agent 工作区之外；在此之前不发布能力分数。
