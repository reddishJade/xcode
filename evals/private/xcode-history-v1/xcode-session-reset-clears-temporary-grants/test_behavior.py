"""Agent 不可见的新会话授权生命周期行为 oracle。"""

from xcode.agent.messages import UserMessage
from xcode.harness.agent_runtime import StructuredAgent
from xcode.harness.agent_runtime.config import GateConfig
from xcode.harness.observability import GrantRecord, InMemoryGrantStore
from xcode.tests.fixtures import FakeProvider


def _grant(*, grant_id: str, scope: str) -> GrantRecord:
    return GrantRecord(
        capability="read",
        operation="read_file",
        target_kind="path",
        target_pattern="src/main.py",
        access="read",
        decision="allow",
        scope=scope,
        grant_id=grant_id,
    )


def _agent(
    session_store: InMemoryGrantStore,
    permanent_store: InMemoryGrantStore,
) -> StructuredAgent:
    return StructuredAgent(
        provider=FakeProvider([]),
        registry=(),
        gate=GateConfig(
            session_grant_store=session_store,
            permanent_grant_store=permanent_store,
        ),
    )


def test_clear_history_clears_only_session_grants() -> None:
    session_store = InMemoryGrantStore()
    permanent_store = InMemoryGrantStore()
    temporary = _grant(grant_id="temporary", scope="session")
    permanent = _grant(grant_id="permanent", scope="permanent")
    session_store.add(temporary)
    permanent_store.add(permanent)
    agent = _agent(session_store, permanent_store)

    agent.clear_history()

    assert session_store.records() == ()
    assert permanent_store.records() == (permanent,)


def test_only_empty_loaded_history_starts_a_new_grant_session() -> None:
    session_store = InMemoryGrantStore()
    temporary = _grant(grant_id="temporary", scope="session")
    session_store.add(temporary)
    agent = _agent(session_store, InMemoryGrantStore())

    agent.load_history([UserMessage(content="resume")])
    assert session_store.records() == (temporary,)

    agent.load_history([])
    assert session_store.records() == ()
