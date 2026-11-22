"""Typed handoff identity shared by original native executors and SSE responses."""

from __future__ import annotations

from typing_extensions import TypedDict


class NativeInputReceipt(TypedDict):
    """A submitted input, not a model completion; response identity is opaque."""

    input_stable_id: str
    terminal_response_id: str
