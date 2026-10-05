"""Provider failover at the ACP harness boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from omnigent.inner import acp_harness
from omnigent.inner.acp_executor import AcpExecutor
from omnigent.inner.acp_provider_failover import (
    AcpProviderAttempt,
    AcpProviderFailoverSupervisor,
    parse_provider_attempts,
)
from omnigent.inner.executor import (
    CompactionStarted,
    Executor,
    ExecutorError,
    ReasoningChunk,
    SubAgentStarted,
    TextChunk,
    ToolCallRequest,
    TurnComplete,
)


class _ScriptedExecutor(Executor):
    def __init__(
        self,
        events: list[object],
        lifecycle: list[str],
        name: str,
        *,
        close_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.lifecycle = lifecycle
        self.name = name
        self.close_error = close_error
        self._tool_executor = None
        self._policy_evaluator = None
        self._elicitation_handler = None
        self._elicitation_choice_handler = None
        self._side_effect_observer = None

    async def run_turn(self, messages, tools, system_prompt, config=None) -> AsyncIterator[object]:
        self.lifecycle.append(f"run:{self.name}:{messages[-1]['content']}")
        for event in self.events:
            yield event

    async def close(self) -> None:
        self.lifecycle.append(f"close:{self.name}")
        if self.close_error is not None:
            raise self.close_error


def _supervisor(
    scripts: list[list[object]],
    lifecycle: list[str],
    *,
    close_error: Exception | None = None,
) -> AcpProviderFailoverSupervisor:
    attempts = (
        AcpProviderAttempt(name="primary", env={"LINGXIAO_MODEL_ATTEMPT": "0"}),
        AcpProviderAttempt(name="fallback", env={"LINGXIAO_MODEL_ATTEMPT": "1"}),
    )

    def factory(attempt: AcpProviderAttempt) -> Executor:
        index = len([item for item in lifecycle if item.startswith("create:")])
        lifecycle.append(f"create:{attempt.name}")
        error = close_error if index == 0 else None
        return _ScriptedExecutor(scripts[index], lifecycle, attempt.name, close_error=error)

    return AcpProviderFailoverSupervisor(attempts, factory)


async def _collect(supervisor: Executor) -> list[object]:
    return [
        event
        async for event in supervisor.run_turn(
            [{"role": "user", "content": "same turn"}], [], "system"
        )
    ]


def test_attempt_json_accepts_only_named_string_env_overrides() -> None:
    attempts = parse_provider_attempts(
        '[{"name":"primary","env":{"LINGXIAO_MODEL_ATTEMPT":"0"},"model":"fallback/model"}]'
    )
    assert attempts == (
        AcpProviderAttempt(
            name="primary",
            env={"LINGXIAO_MODEL_ATTEMPT": "0"},
            model="fallback/model",
        ),
    )

    with pytest.raises(ValueError, match="non-secret selector"):
        parse_provider_attempts('[{"name":"bad","env":{"SECRET":1}}]')
    with pytest.raises(ValueError, match="non-secret selector"):
        parse_provider_attempts('[{"name":"bad","env":{"OPENAI_API_KEY":"leak"}}]')


def test_plain_authorization_errors_are_not_assumed_to_be_provider_failures() -> None:
    from omnigent.inner.acp_provider_failover import is_recoverable_provider_error

    assert not is_recoverable_provider_error(
        ExecutorError(message="workspace operation forbidden", retryable=True)
    )
    assert not is_recoverable_provider_error(ExecutorError(message="unauthorized", retryable=True))


@pytest.mark.asyncio
async def test_provider_failure_before_output_closes_then_replays_same_turn() -> None:
    lifecycle: list[str] = []
    supervisor = _supervisor(
        [
            [ExecutorError(message="provider returned HTTP 429 rate limit", retryable=True)],
            [TextChunk(text="ok"), TurnComplete(response="ok")],
        ],
        lifecycle,
    )

    events = await _collect(supervisor)

    assert [type(event) for event in events] == [TextChunk, TurnComplete]
    assert lifecycle == [
        "create:primary",
        "run:primary:same turn",
        "close:primary",
        "create:fallback",
        "run:fallback:same turn",
    ]


@pytest.mark.asyncio
async def test_structured_provider_timeout_switches_but_generic_acp_timeout_does_not() -> None:
    switching_lifecycle: list[str] = []
    provider_timeout = ExecutorError(message="provider upstream timed out", retryable=True)
    switched = await _collect(
        _supervisor([[provider_timeout], [TurnComplete(response="ok")]], switching_lifecycle)
    )
    assert switched == [TurnComplete(response="ok")]
    assert switching_lifecycle == [
        "create:primary",
        "run:primary:same turn",
        "close:primary",
        "create:fallback",
        "run:fallback:same turn",
    ]

    generic_lifecycle: list[str] = []
    generic_timeout = ExecutorError(message="Timeout waiting for ACP response", retryable=True)
    not_switched = await _collect(
        _supervisor([[generic_timeout], [TurnComplete(response="wrong")]], generic_lifecycle)
    )
    assert not_switched == [generic_timeout]
    assert generic_lifecycle == ["create:primary", "run:primary:same turn"]


@pytest.mark.asyncio
async def test_provider_auth_failure_switches_even_when_same_provider_is_not_retryable() -> None:
    lifecycle: list[str] = []
    auth_error = ExecutorError(message="provider authentication failed: HTTP 401", retryable=False)

    events = await _collect(_supervisor([[auth_error], [TurnComplete(response="ok")]], lifecycle))

    assert events == [TurnComplete(response="ok")]
    assert "create:fallback" in lifecycle


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "event"),
    [
        ("text", TextChunk(text="partial")),
        ("reasoning", ReasoningChunk(delta="thought", event_type="reasoning_text")),
        ("tool", ToolCallRequest(name="bash", args={})),
        ("subagent", SubAgentStarted(child_key="child", title="child")),
        ("compaction", CompactionStarted()),
    ],
)
async def test_any_visible_or_side_effect_boundary_disables_failover(
    boundary: str, event: object
) -> None:
    lifecycle: list[str] = []
    error = ExecutorError(message="provider HTTP 503", retryable=True)
    supervisor = _supervisor([[event, error], [TurnComplete(response="wrong")]], lifecycle)

    events = await _collect(supervisor)

    assert events == [event, error], boundary
    assert lifecycle == ["create:primary", "run:primary:same turn"]


@pytest.mark.asyncio
async def test_elicitation_callback_disables_failover_even_without_an_event() -> None:
    lifecycle: list[str] = []
    supervisor = _supervisor(
        [
            [ExecutorError(message="provider HTTP 503", retryable=True)],
            [TurnComplete(response="wrong")],
        ],
        lifecycle,
    )

    async def elicit(_name, _args):
        return True

    supervisor._elicitation_handler = elicit
    original_factory = supervisor._factory

    def factory(attempt):
        executor = original_factory(attempt)
        original_run = executor.run_turn

        async def run(*args, **kwargs):
            assert executor._elicitation_handler is not None
            await executor._elicitation_handler("bash", {})
            async for item in original_run(*args, **kwargs):
                yield item

        executor.run_turn = run
        return executor

    supervisor._factory = factory
    events = await _collect(supervisor)

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert lifecycle == ["create:primary", "run:primary:same turn"]


@pytest.mark.asyncio
async def test_native_acp_side_effect_callback_disables_failover() -> None:
    lifecycle: list[str] = []
    supervisor = _supervisor(
        [
            [ExecutorError(message="provider HTTP 503", retryable=True)],
            [TurnComplete(response="wrong")],
        ],
        lifecycle,
    )
    original_factory = supervisor._factory

    def factory(attempt):
        executor = original_factory(attempt)
        original_run = executor.run_turn

        async def run(*args, **kwargs):
            assert executor._side_effect_observer is not None
            executor._side_effect_observer()
            async for item in original_run(*args, **kwargs):
                yield item

        executor.run_turn = run
        return executor

    supervisor._factory = factory
    events = await _collect(supervisor)

    assert len(events) == 1 and isinstance(events[0], ExecutorError)
    assert lifecycle == ["create:primary", "run:primary:same turn"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "context length exceeded (HTTP 429)",
        "MCP tool execution failed with HTTP 503",
        "invalid request: HTTP 500 in sandbox",
    ],
)
async def test_non_provider_errors_never_switch(message: str) -> None:
    lifecycle: list[str] = []
    error = ExecutorError(message=message, retryable=True)
    events = await _collect(_supervisor([[error], [TurnComplete(response="wrong")]], lifecycle))
    assert events == [error]
    assert lifecycle == ["create:primary", "run:primary:same turn"]


@pytest.mark.asyncio
async def test_close_failure_returns_original_error_without_starting_fallback() -> None:
    lifecycle: list[str] = []
    error = ExecutorError(message="provider quota exhausted", retryable=True)
    events = await _collect(
        _supervisor(
            [[error], [TurnComplete(response="wrong")]],
            lifecycle,
            close_error=RuntimeError("process tree still alive"),
        )
    )
    assert events == [error]
    assert lifecycle == ["create:primary", "run:primary:same turn", "close:primary"]


@pytest.mark.asyncio
async def test_all_attempts_fail_emits_only_the_last_error() -> None:
    lifecycle: list[str] = []
    first = ExecutorError(message="provider HTTP 429", retryable=True)
    last = ExecutorError(message="provider authentication failed: HTTP 401", retryable=True)
    events = await _collect(_supervisor([[first], [last]], lifecycle))
    assert events == [last]


def test_harness_without_attempts_keeps_plain_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_ACP_COMMAND", "wrapper acp")
    monkeypatch.delenv("HARNESS_ACP_PROVIDER_ATTEMPTS", raising=False)

    executor = acp_harness._build_harness_executor()

    assert type(executor) is AcpExecutor
    assert executor._config.spawn_env_overrides == {}


def test_harness_single_attempt_keeps_plain_executor_with_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_ACP_COMMAND", "wrapper acp")
    monkeypatch.setenv(
        "HARNESS_ACP_PROVIDER_ATTEMPTS",
        '[{"name":"primary","env":{"LINGXIAO_ACP_PROVIDER_ATTEMPT":"primary"}}]',
    )

    executor = acp_harness._build_harness_executor()

    assert type(executor) is AcpExecutor
    assert executor._config.spawn_env_overrides == {"LINGXIAO_ACP_PROVIDER_ATTEMPT": "primary"}


def test_harness_multiple_attempts_builds_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_ACP_COMMAND", "wrapper acp")
    monkeypatch.setenv(
        "HARNESS_ACP_PROVIDER_ATTEMPTS",
        """[
            {"name":"primary","env":{"LINGXIAO_ACP_PROVIDER_ATTEMPT":"primary"}},
            {"name":"fallback","env":{"LINGXIAO_ACP_PROVIDER_ATTEMPT":"fallback"}}
        ]""",
    )

    executor = acp_harness._build_harness_executor()

    assert isinstance(executor, AcpProviderFailoverSupervisor)


def test_harness_attempt_model_overrides_primary_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_ACP_COMMAND", "wrapper acp")
    monkeypatch.setenv("HARNESS_ACP_MODEL", "lingxiao/primary")
    monkeypatch.setenv(
        "HARNESS_ACP_PROVIDER_ATTEMPTS",
        """[
            {"name":"primary","env":{"LINGXIAO_MODEL_ATTEMPT":"0"},"model":"lingxiao/primary"},
            {"name":"fallback","env":{"LINGXIAO_MODEL_ATTEMPT":"1"},"model":"lingxiao/fallback"}
        ]""",
    )

    supervisor = acp_harness._build_harness_executor()
    assert isinstance(supervisor, AcpProviderFailoverSupervisor)
    supervisor._attempt_index = 1
    fallback = supervisor._build_active()

    assert isinstance(fallback, AcpExecutor)
    assert fallback._config.model == "lingxiao/fallback"
