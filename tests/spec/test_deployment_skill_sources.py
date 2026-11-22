"""Deployment skill catalogs and execution share explicit bundle/repo roots."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnigent.native.deployment_policy import NativeDeploymentPolicy, deployment_policy_from_spec
from omnigent.runner.tool_dispatch import build_native_relay_tool_schemas, execute_tool
from omnigent.spec.parser import _discover_skills, discover_host_skills
from omnigent.spec.skill_sources import resolve_session_skills
from omnigent.spec.types import AgentSpec
from omnigent.tools.base import ToolContext
from omnigent.tools.manager import ToolManager
from tests.spec.test_skill_sources import _write_skill


def test_restricted_catalog_load_read_and_menu_share_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, parent, bundle = (tmp_path / name for name in ("home", "parent", "bundle"))
    repo = parent / "repo"
    for root, dotdir, name in (
        (home, ".claude", "personal-claude"),
        (home, ".agents", "personal-agents"),
        (parent, ".agents", "parent-only"),
        (repo, ".agents", "repo-only"),
    ):
        _write_skill(root / dotdir / "skills", name)
    _write_skill(bundle / "skills", "selected")
    _write_skill(repo / ".agents" / "skills", "selected")
    selected = bundle / "skills" / "selected"
    (selected / "references").mkdir()
    (selected / "references" / "good.md").write_text("selected reference")
    (bundle / "agent.json").write_text("not a skill resource")
    (selected / "outside.md").symlink_to(bundle / "agent.json")
    (repo / ".claude").symlink_to(home / ".claude", target_is_directory=True)
    (repo / ".agents" / "skills" / "escape").symlink_to(
        home / ".claude" / "skills" / "personal-claude", target_is_directory=True
    )
    escape_file = repo / ".agents" / "skills" / "escape-file"
    escape_file.mkdir()
    (escape_file / "SKILL.md").symlink_to(
        home / ".agents" / "skills" / "personal-agents" / "SKILL.md"
    )
    monkeypatch.setenv("HOME", str(home))
    spec = AgentSpec(spec_version=1, skills=_discover_skills(bundle / "skills"))
    spec.executor.config.update(
        native_deployment_policy=asdict(NativeDeploymentPolicy()),
        native_deployment_policy_version=1,
    )
    # Trap reads/scans, not merely the advertised names. These are synthetic dirs.
    original_read, original_iter = Path.read_text, Path.iterdir
    forbidden = (home.resolve(), (parent / ".agents").resolve())

    def guarded_read(path: Path, *args, **kwargs):
        assert not any(path.resolve().is_relative_to(root) for root in forbidden)
        return original_read(path, *args, **kwargs)

    def guarded_iter(path: Path):
        assert not any(path.resolve().is_relative_to(root) for root in forbidden)
        return original_iter(path)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr(Path, "iterdir", guarded_iter)
    manager = ToolManager(spec, workdir=bundle, skill_workspace=repo, os_env_schema_only=True)
    try:
        description = next(
            s["function"]["description"]
            for s in manager.get_tool_schemas()
            if s["function"]["name"] == "load_skill"
        )
        assert "repo-only" in description and "selected" in description
        assert all(
            name not in description
            for name in ("personal-claude", "personal-agents", "parent-only", "escape")
        )
        menu = resolve_session_skills(spec, (repo,), bundle)
        assert {s.name for s in menu} == {"selected", "repo-only"}
        assert next(s for s in menu if s.name == "selected").skill_dir == selected
        relay = build_native_relay_tool_schemas(spec, bundle_dir=bundle, workspace=repo)
        assert next(s["description"] for s in relay if s["name"] == "load_skill") == description
        context = ToolContext(task_id="", agent_id="")
        assert (
            manager.call_tool(
                "read_skill_file", '{"skill_name":"selected","path":"references/good.md"}', context
            )
            == "selected reference"
        )
        for resource in ("../../agent.json", "outside.md", "/etc/passwd"):
            assert manager.call_tool(
                "read_skill_file", '{"skill_name":"selected","path":"' + resource + '"}', context
            ).startswith("Error:")
        assert "outside.md" not in manager.call_tool("load_skill", '{"name":"selected"}', context)
        assert manager.call_tool("load_skill", '{"name":"personal-claude"}', context).startswith(
            "Error:"
        )
        # Re-check containment at use time after an advertised path is replaced.
        repo_skill = repo / ".agents" / "skills" / "repo-only"
        (repo_skill / "SKILL.md").unlink()
        repo_skill.rmdir()
        repo_skill.symlink_to(
            home / ".claude" / "skills" / "personal-claude", target_is_directory=True
        )
        assert manager.call_tool("load_skill", '{"name":"repo-only"}', context).startswith(
            "Error:"
        )
        assert manager.call_tool(
            "read_skill_file", '{"skill_name":"repo-only","path":"SKILL.md"}', context
        ).startswith("Error:")
    finally:
        manager.shutdown()


async def test_actual_dispatch_uses_repo_and_bundle_not_ambient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, bundle, home = (tmp_path / name for name in ("repo", "bundle", "home"))
    _write_skill(repo / ".agents" / "skills", "repo-only")
    _write_skill(bundle / "skills", "selected")
    _write_skill(home / ".claude" / "skills", "personal-only")
    (bundle / "skills" / "selected" / "note.md").write_text("reference body")
    monkeypatch.setenv("HOME", str(home))
    spec = AgentSpec(spec_version=1, skills=_discover_skills(bundle / "skills"))
    spec.executor.config.update(
        native_deployment_policy=asdict(NativeDeploymentPolicy()),
        native_deployment_policy_version=1,
    )
    args = {"agent_spec": spec, "runner_workspace": repo, "local_tool_workdir": bundle}
    assert "body" in await execute_tool(
        tool_name="load_skill", arguments='{"name":"repo-only"}', **args
    )
    assert (
        await execute_tool(
            tool_name="read_skill_file",
            arguments='{"skill_name":"selected","path":"note.md"}',
            **args,
        )
        == "reference body"
    )
    assert "not found" in await execute_tool(
        tool_name="load_skill", arguments='{"name":"personal-only"}', **args
    )
    # No workspace/bundle is available at some metadata stages: StoredSpec
    # content remains, but cwd/Home and filesystem references are not authority.
    menu = resolve_session_skills(spec, (), None)
    assert [s.name for s in menu] == ["selected"] and menu[0].skill_dir is None


def test_legacy_scanner_keeps_ancestor_and_home_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, parent = tmp_path / "home", tmp_path / "parent"
    repo = parent / "repo"
    repo.mkdir(parents=True)
    _write_skill(home / ".claude" / "skills", "personal-only")
    _write_skill(parent / ".agents" / "skills", "parent-only")
    monkeypatch.setenv("HOME", str(home))
    assert {s.name for s in discover_host_skills(repo, "all")} == {"personal-only", "parent-only"}


def test_missing_legacy_config_is_not_a_malformed_typed_deployment() -> None:
    legacy = SimpleNamespace(executor=SimpleNamespace(harness_kind="codex-native"))
    assert deployment_policy_from_spec(legacy) is None
    spec = AgentSpec(spec_version=1)
    for invalid in (None, [], False):
        spec.executor.config = invalid
        with pytest.raises(ValueError, match="executor config"):
            deployment_policy_from_spec(spec)
    spec.executor.config = {
        "native_deployment_policy": asdict(NativeDeploymentPolicy()),
        "native_deployment_policy_version": True,
    }
    with pytest.raises(ValueError, match="version 1"):
        deployment_policy_from_spec(spec)
