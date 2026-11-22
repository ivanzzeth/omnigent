"""``tool_groups:`` on the runner side: the native relay and the dispatch gate.

Native harnesses see only the relay surface, and every other harness is held
to the spec's registered surface at dispatch, so a disabled group must vanish
from both — not just from ``ToolManager``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omnigent.native.deployment_policy import NativeDeploymentPolicy
from omnigent.runner import tool_dispatch
from omnigent.runner.mcp_manager import RunnerMcpManager
from omnigent.runner.tool_dispatch import (
    _ungranted_tool_reason,
    build_native_relay_tool_schemas,
)
from omnigent.spec import load
from omnigent.spec.types import AgentSpec, ToolGroupsConfig
from omnigent.tools.manager import ToolManager

_ALL_OFF = ToolGroupsConfig(
    browser=False,
    scheduled_tasks=False,
    comments=False,
    policies=False,
    agent_discovery=False,
)


def _spec(groups: ToolGroupsConfig) -> AgentSpec:
    """A minimal spec with the given tool groups."""
    return AgentSpec(spec_version=1, skills_filter="none", tool_groups=groups)


def _registered(spec: AgentSpec) -> set[str]:
    """Names ``ToolManager`` registers for *spec*."""
    return {s["function"]["name"] for s in ToolManager(spec).get_tool_schemas()}


def _gated() -> frozenset[str]:
    """Tools the five groups control, derived from registration itself."""
    gated = frozenset(_registered(_spec(ToolGroupsConfig())) - _registered(_spec(_ALL_OFF)))
    assert len(gated) == 16, sorted(gated)
    return gated


def _relay_names(spec: AgentSpec) -> set[str]:
    """Names the native relay advertises for *spec*."""
    return {schema["name"] for schema in build_native_relay_tool_schemas(spec)}


def test_native_relay_carries_every_group_by_default() -> None:
    """Without a ``tool_groups:`` block the relay advertises all 16 gated tools."""
    missing = _gated() - _relay_names(_spec(ToolGroupsConfig()))
    assert not missing, f"relay dropped {sorted(missing)} with every group on"


def test_native_relay_drops_disabled_groups() -> None:
    """
    With every group off the relay advertises none of the gated tools but
    keeps the keyless session reads and ``sys_cancel_task``.
    """
    names = _relay_names(_spec(_ALL_OFF))
    assert not (_gated() & names), f"relay still advertises {sorted(_gated() & names)}"
    assert {"sys_session_list", "sys_session_get_info", "sys_cancel_task"} <= names


def test_dispatch_refuses_a_disabled_group_tool() -> None:
    """A model that calls a disabled tool anyway is refused, not executed."""
    default_spec = _spec(ToolGroupsConfig())
    trimmed_spec = _spec(_ALL_OFF)
    for name in sorted(_gated()):
        assert _ungranted_tool_reason(name, default_spec) is None, name
        reason = _ungranted_tool_reason(name, trimmed_spec)
        assert reason is not None and "not enabled for this agent" in reason, name


@pytest.fixture
def restricted_native_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentSpec:
    """Original parser, sanitized shape of the real two-MCP organization bundle."""
    home = tmp_path / "private-home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    (tmp_path / "config.yaml").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "restricted-native-surface",
                "executor": {
                    "type": "omnigent",
                    "config": {
                        "harness": "codex-native",
                        "native_runtime_selector": "omnigent:codex",
                        "native_provider_key": "codex",
                        "native_deployment_policy": asdict(NativeDeploymentPolicy()),
                        "native_deployment_policy_version": 1,
                    },
                },
                "async": False,
                "spawn": False,
                "tool_groups": asdict(_ALL_OFF),
                "tools": {
                    "lingxiao": {"type": "mcp", "url": "https://core.example.test/current-run"},
                    "lingxiao_evidence_files": {
                        "type": "mcp",
                        "transport": "stdio",
                        "command": "/operator/verified/lx",
                        "args": ["mcp", "evidence-files", "--run-bound", "--file-root", "."],
                        "tools": ["issue_evidence_upload"],
                    },
                },
            }
        )
    )
    return load(tmp_path, expand_env=False, enforce_handler_allowlist=True)


def test_restricted_native_relay_respects_original_spec_selection(
    restricted_native_spec: AgentSpec,
) -> None:
    assert restricted_native_spec.os_env is None
    assert len(restricted_native_spec.mcp_servers) == 2
    names = _relay_names(restricted_native_spec)
    assert "sys_os_shell" not in names
    assert not (
        {"sys_session_create", "sys_call_async", "sys_timer_set", "sys_session_share"} & names
    )


@pytest.mark.asyncio
async def test_restricted_native_dispatch_refuses_before_os_handler(
    restricted_native_spec: AgentSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = AsyncMock(return_value=json.dumps({"sentinel_executed": True}))
    monkeypatch.setattr(tool_dispatch, "_execute_os_env_tool", handler)
    output = json.loads(
        await tool_dispatch.execute_tool(
            tool_name="sys_os_shell",
            arguments=json.dumps({"command": "fixture sentinel"}),
            agent_spec=restricted_native_spec,
            effective_harness="codex-native",
        )
    )
    handler.assert_not_awaited()
    assert "not enabled for this agent" in output["error"]


@pytest.mark.asyncio
async def test_restricted_native_selected_mcp_still_uses_original_manager(
    restricted_native_spec: AgentSpec,
) -> None:
    manager = AsyncMock(spec=RunnerMcpManager)
    manager.call_tool.return_value = "selected evidence MCP result"
    args = {"file": "owned.png"}
    output = await tool_dispatch.execute_tool(
        tool_name="lingxiao_evidence_files__issue_evidence_upload",
        arguments=json.dumps(args),
        agent_spec=restricted_native_spec,
        effective_harness="codex-native",
        mcp_manager=manager,
    )
    assert output == "selected evidence MCP result"
    manager.call_tool.assert_awaited_once_with(
        restricted_native_spec, "lingxiao_evidence_files__issue_evidence_upload", args
    )
