"""The private native HOME and original managed relay root stay separate."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from omnigent.harnesses.claude_native import bridge as shared_bridge
from omnigent.harnesses.codex_native import app_server, bridge


def _relay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "relay-home/.omnigent/codex-native"
    directory = root / "instance"
    directory.mkdir(parents=True, mode=0o700)
    monkeypatch.setattr(bridge, "_BRIDGE_ROOT", root)
    bridge.write_mcp_bridge_config(directory)
    return directory


def test_managed_home_matches_both_configs_without_changing_native_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _relay(tmp_path, monkeypatch)
    native_home = directory / "codex-home/deployment-home"
    monkeypatch.setenv("HOME", str(native_home))
    expected = {"HOME": str(tmp_path / "relay-home")}
    rendered = app_server._codex_mcp_server_config_section(directory, required=True)
    assert tomllib.loads(rendered)["mcp_servers"]["omnigent"]["env"] == expected
    overrides = bridge.codex_mcp_config_overrides(directory, required=True)
    assert f"mcp_servers.omnigent.env.HOME={json.dumps(expected['HOME'])}" in overrides
    assert os.environ["HOME"] == str(native_home)
    assert (
        "env"
        not in tomllib.loads(app_server._codex_mcp_server_config_section(directory))[
            "mcp_servers"
        ]["omnigent"]
    )
    assert not any(".env." in value for value in bridge.codex_mcp_config_overrides(directory))
    config = json.loads((directory / "bridge.json").read_text())
    assert set(config) == {"token"}
    tools, close = shared_bridge._build_tools(config)
    assert tools == {}
    close()


@pytest.mark.parametrize("invalid", ["outside", "test-root", "workspace", "symlink"])
def test_managed_home_rejects_untrusted_roots_and_nonrelay_configs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    directory = _relay(tmp_path, monkeypatch)
    if invalid == "outside":
        directory = tmp_path / "outside"
        directory.mkdir()
    elif invalid == "test-root":
        monkeypatch.setattr(bridge, "_BRIDGE_ROOT", directory.parent.parent)
    elif invalid == "workspace":
        (directory / "bridge.json").write_text(json.dumps({"token": "test", "workspace": "/"}))
    else:
        config = directory / "bridge.json"
        config.rename(directory / "original.json")
        config.symlink_to(directory / "original.json")
    with pytest.raises(ValueError, match="managed Codex MCP"):
        bridge.codex_managed_mcp_env(directory)


async def test_official_initialize_uses_explicit_relay_home_not_private_native_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _relay(tmp_path, monkeypatch)
    private = tmp_path / "private-native-home"
    private.mkdir()
    env = {"HOME": str(private), "TMPDIR": str(tmp_path), "PATH": os.defpath}
    args = [
        "-I",
        "-m",
        "omnigent.harnesses.claude_native.bridge",
        "serve-mcp",
        "--bridge-dir",
        str(directory),
    ]

    async def initialize(child_env: dict[str, str], stderr: Path) -> None:
        params = StdioServerParameters(command=sys.executable, args=args, env=child_env)
        with stderr.open("w") as errors:
            async with asyncio.timeout(10):
                async with stdio_client(params, errlog=errors) as (read, write):
                    async with ClientSession(read, write) as client:
                        result = await client.initialize()
                        assert result.serverInfo.name

    # Original root rejection is real subprocess stderr, not a mocked hash.
    with pytest.raises(BaseExceptionGroup):
        await initialize(env, tmp_path / "before-stderr.log")
    assert "not under an allowed bridge root" in (tmp_path / "before-stderr.log").read_text()
    managed = cast(
        "dict[str, str]",
        tomllib.loads(app_server._codex_mcp_server_config_section(directory, required=True))[
            "mcp_servers"
        ]["omnigent"]["env"],
    )
    await initialize({**env, **managed}, tmp_path / "after-stderr.log")
    assert (tmp_path / "after-stderr.log").read_text() == ""
    assert env["HOME"] == str(private)
