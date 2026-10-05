"""Conservative provider failover for the generic ACP harness.

The supervisor retries the same Omnigent turn only before the wrapped ACP
executor has exposed output or crossed a side-effect boundary.  It intentionally
does not belong to the generic :mod:`executor` contract: provider attempts and
ACP subprocess replacement are properties of this harness alone.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from omnigent.inner.executor import (
    CompactionComplete,
    CompactionStarted,
    EnqueuedContent,
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    ReasoningChunk,
    SubAgentCompleted,
    SubAgentStarted,
    SubAgentToolCall,
    TextChunk,
    ToolCallComplete,
    ToolCallRequest,
    ToolSpec,
)

logger = logging.getLogger(__name__)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RECOVERABLE_STATUS = re.compile(r"\b(?:401|403|429|5\d{2})\b")
_PROVIDER_MARKERS = (
    "provider",
    "rate limit",
    "rate-limit",
    "quota exceeded",
    "quota exhausted",
    "usage quota",
    "available accounts exhausted",
    "authentication failed",
    "unauthorized",
    "forbidden",
)
_DIRECT_RECOVERABLE_MARKERS = (
    "quota exceeded",
    "quota exhausted",
    "usage quota",
    "available accounts exhausted",
    "authentication failed",
    "unauthorized",
    "forbidden",
)
_PROVIDER_ERROR_CODES = {
    "authentication_error",
    "authorization_error",
    "rate_limit_exceeded",
    "quota_exceeded",
    "provider_timeout",
    "provider_unavailable",
}
_NON_PROVIDER_MARKERS = (
    "context length",
    "context window",
    "content policy",
    "invalid request",
    "invalid parameter",
    "tool error",
    "tool execution",
    "tool validation",
    "mcp ",
    "sandbox",
    "credential broker",
)
_BOUNDARY_EVENTS = (
    TextChunk,
    ReasoningChunk,
    ToolCallRequest,
    ToolCallComplete,
    SubAgentStarted,
    SubAgentCompleted,
    SubAgentToolCall,
    CompactionStarted,
    CompactionComplete,
)


@dataclass(frozen=True)
class AcpProviderAttempt:
    """One wrapper restart choice containing only non-secret selectors."""

    name: str
    env: Mapping[str, str]


def parse_provider_attempts(raw: str | None) -> tuple[AcpProviderAttempt, ...]:
    """Parse ``HARNESS_ACP_PROVIDER_ATTEMPTS``.

    Values are copied into the ACP subprocess environment, so this contract is
    deliberately limited to named string overrides.  Callers must put only
    non-sensitive selectors here; provider credentials remain in the wrapper's
    own credential channel.
    """
    if raw is None or not raw.strip():
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("ACP provider attempts must be valid JSON") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("ACP provider attempts must be a non-empty JSON array")

    attempts: list[AcpProviderAttempt] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != {"name", "env"}:
            raise ValueError("each ACP provider attempt requires only name and env")
        name = item.get("name")
        env = item.get("env")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("ACP provider attempt names must be non-empty strings")
        if not isinstance(env, dict) or not env:
            raise ValueError("ACP provider attempts require string environment overrides")
        if any(
            not isinstance(key, str)
            or _ENV_NAME.fullmatch(key) is None
            or not isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError("ACP provider attempts require string environment overrides")
        attempts.append(AcpProviderAttempt(name=name.strip(), env=dict(env)))
    return tuple(attempts)


def is_recoverable_provider_error(error: ExecutorError) -> bool:
    """Return whether a fresh provider can clearly repair ``error``."""
    message = error.message.casefold()
    if any(marker in message for marker in _NON_PROVIDER_MARKERS):
        return False
    code = (error.code or "").casefold()
    if code in _PROVIDER_ERROR_CODES or code.startswith("provider_5"):
        return True
    if any(marker in message for marker in _DIRECT_RECOVERABLE_MARKERS):
        return True
    has_provider_marker = any(marker in message for marker in _PROVIDER_MARKERS)
    if _RECOVERABLE_STATUS.search(message):
        return has_provider_marker
    return has_provider_marker and any(
        marker in message
        for marker in ("timeout", "timed out", "connection error", "connection reset")
    )


class AcpProviderFailoverSupervisor(Executor):
    """Own a sequence of mutually exclusive ACP provider attempts."""

    def __init__(
        self,
        attempts: tuple[AcpProviderAttempt, ...],
        factory: Callable[[AcpProviderAttempt], Executor],
    ) -> None:
        if len(attempts) < 2:
            raise ValueError("provider failover supervisor requires at least two attempts")
        self._attempts = attempts
        self._factory = factory
        self._attempt_index = 0
        self._active: Executor | None = None
        self._tool_executor: Any = None
        self._policy_evaluator: Any = None
        self._elicitation_handler: Any = None
        self._elicitation_choice_handler: Any = None
        self._side_effect_observed = False

    def _build_active(self) -> Executor:
        executor = self._factory(self._attempts[self._attempt_index])
        self._bind_bridges(executor)
        self._active = executor
        return executor

    def _bind_bridges(self, executor: Executor) -> None:
        executor._tool_executor = self._wrap_boundary_callback(self._tool_executor)  # type: ignore[attr-defined]
        executor._policy_evaluator = self._policy_evaluator  # type: ignore[attr-defined]
        executor._elicitation_handler = self._wrap_boundary_callback(  # type: ignore[attr-defined]
            self._elicitation_handler
        )
        executor._elicitation_choice_handler = self._wrap_boundary_callback(  # type: ignore[attr-defined]
            self._elicitation_choice_handler
        )

    def _wrap_boundary_callback(self, callback: Any) -> Any:
        if callback is None:
            return None

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._side_effect_observed = True
            return await callback(*args, **kwargs)

        return wrapped

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        self._side_effect_observed = False
        while True:
            executor = self._active or self._build_active()
            candidate_error: ExecutorError | None = None
            async for event in executor.run_turn(messages, tools, system_prompt, config):
                if isinstance(event, _BOUNDARY_EVENTS):
                    self._side_effect_observed = True
                if isinstance(event, ExecutorError):
                    candidate_error = event
                    break
                yield event

            if candidate_error is None:
                return
            has_fallback = self._attempt_index + 1 < len(self._attempts)
            if (
                self._side_effect_observed
                or not has_fallback
                or not is_recoverable_provider_error(candidate_error)
            ):
                yield candidate_error
                return

            try:
                strict_close = getattr(executor, "close_for_failover", None)
                if strict_close is not None:
                    await strict_close()
                else:
                    await executor.close()
            except Exception as exc:  # noqa: BLE001 - refusing overlap is fail-closed
                logger.error(
                    "ACP provider attempt %s could not be reaped; refusing failover: %s",
                    self._attempts[self._attempt_index].name,
                    exc,
                )
                yield candidate_error
                return
            self._active = None
            self._attempt_index += 1

    def handles_tools_internally(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def max_context_tokens(self) -> int | None:
        return self._active.max_context_tokens() if self._active is not None else None

    async def interrupt_session(self, session_key: str) -> bool:
        return (
            await self._active.interrupt_session(session_key)
            if self._active is not None
            else False
        )

    async def enqueue_session_message(
        self, session_key: str, content: EnqueuedContent
    ) -> bool:
        return (
            await self._active.enqueue_session_message(session_key, content)
            if self._active is not None
            else False
        )

    def supports_live_message_queue(self) -> bool:
        return (
            self._active.supports_live_message_queue() if self._active is not None else False
        )

    def supports_tool_boundary_interrupt(self) -> bool:
        return (
            self._active.supports_tool_boundary_interrupt()
            if self._active is not None
            else False
        )

    async def close_session(self, session_key: str) -> None:
        if self._active is not None:
            await self._active.close_session(session_key)

    async def close(self) -> None:
        if self._active is not None:
            active, self._active = self._active, None
            await active.close()
