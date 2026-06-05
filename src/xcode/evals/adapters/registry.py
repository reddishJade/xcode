from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkAdapterSpec:
    """外部 benchmark adapter 的声明信息。"""

    name: str
    display_name: str
    purpose: str
    harness: str
    xcode_role: str
    upstream_url: str


BENCHMARK_ADAPTERS: dict[str, BenchmarkAdapterSpec] = {
    "swebench-lite": BenchmarkAdapterSpec(
        name="swebench-lite",
        display_name="SWE-bench Lite",
        purpose="仓库级缺陷修复能力评测。",
        harness="SWE-bench 官方 harness",
        xcode_role="生成候选补丁，交由 harness 检出仓库、运行测试并评分。",
        upstream_url="https://www.swebench.com/SWE-bench/",
    ),
    "swebench-verified": BenchmarkAdapterSpec(
        name="swebench-verified",
        display_name="SWE-bench Verified",
        purpose="专家验证实例上的仓库级缺陷修复评测。",
        harness="SWE-bench 官方 harness",
        xcode_role="生成候选补丁，交由 harness 检出仓库、运行测试并评分。",
        upstream_url="https://www.swebench.com/SWE-bench/",
    ),
    "terminal-bench": BenchmarkAdapterSpec(
        name="terminal-bench",
        display_name="Terminal-Bench",
        purpose="终端任务、shell 调试和工程操作能力评测。",
        harness="Terminal-Bench harness",
        xcode_role="作为终端 agent 执行任务，评分由 harness 产出。",
        upstream_url="https://terminalbench.lol/",
    ),
    "aider-polyglot": BenchmarkAdapterSpec(
        name="aider-polyglot",
        display_name="Aider Polyglot",
        purpose="多语言代码编辑能力评测。",
        harness="Aider Polyglot benchmark",
        xcode_role="在任务工作区内编辑代码，评分由语言测试命令产出。",
        upstream_url="https://aider.chat/docs/leaderboards/",
    ),
}
