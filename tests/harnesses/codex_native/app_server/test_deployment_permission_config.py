"""Required deployments use native profile requests and the user reviewer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omnigent.harnesses.codex_native import app_server
from omnigent.native.deployment_policy import NativeDeploymentPolicy


@pytest.mark.parametrize("required", [False, True])
def test_provider_permission_config_preserves_private_home_and_legacy(
    tmp_path: Path, required: bool
) -> None:
    policy = NativeDeploymentPolicy() if required else None
    server = app_server.build_codex_native_server(
        socket_path=tmp_path / "server.sock",
        codex_home=tmp_path / "codex-home",
        cwd=tmp_path,
        model=None,
        profile=None,
        bridge_dir=tmp_path / "bridge",
        codex_path="/not-launched/codex",
        deployment_policy=policy,
    )
    if required:
        assert server.config_overrides == list(app_server._CODEX_DEPLOYMENT_PERMISSION_OVERRIDES)
        assert server.env["HOME"] == str(tmp_path / "codex-home/deployment-home")
        assert server.env["USERPROFILE"] == server.env["HOME"]
    else:
        assert server.config_overrides == []


def test_explicit_user_reviewer_is_not_normalized_into_auto_review() -> None:
    args = ["-c", 'approvals_reviewer="user"']
    assert app_server.normalize_codex_permission_launch_args(args) == args
    assert app_server.normalize_codex_permission_launch_args([]) == [
        "-c",
        'approvals_reviewer="auto_review"',
    ]


@pytest.mark.parametrize("thread_id", [None, "same-thread"])
@pytest.mark.parametrize("required", [False, True])
def test_fresh_and_modern_resume_tui_permission_config(
    required: bool, thread_id: str | None
) -> None:
    policy = NativeDeploymentPolicy() if required else None
    config = app_server._CODEX_DEPLOYMENT_PERMISSION_OVERRIDES if required else ()
    args = app_server.build_codex_remote_args(
        codex_args=(),
        thread_id=thread_id,
        remote_url="ws://127.0.0.1:1",
        config_overrides=config,
        codex_cli_version=(0, 166, 0),
        deployment_policy=policy,
    )
    if required:
        assert 'approvals_reviewer="auto_review"' not in args
        assert "features.request_permissions_tool=true" in args
        if thread_id is None:
            assert 'approvals_reviewer="user"' in args
            assert 'approval_policy="on-request"' in args
            assert 'sandbox_mode="workspace-write"' in args
        else:
            # Typed permissions are applied by original preload, not forbidden
            # permission arguments on a modern remote-resume terminal.
            assert 'approvals_reviewer="user"' not in args
    elif thread_id is None:
        assert 'approvals_reviewer="auto_review"' in args
        assert "features.request_permissions_tool=true" not in args


@pytest.mark.parametrize("required", [False, True])
async def test_original_preload_sends_same_thread_typed_reviewer_and_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, required: bool
) -> None:
    client = AsyncMock(spec=app_server.CodexAppServerClient)
    monkeypatch.setattr(app_server, "client_for_transport", lambda *a, **k: client)
    await app_server.preload_codex_thread_for_resume(
        "ws://127.0.0.1:1",
        "same-thread",
        cwd=tmp_path,
        deployment_policy=NativeDeploymentPolicy() if required else None,
    )
    params = {
        "threadId": "same-thread",
        "excludeTurns": True,
        "approvalsReviewer": "user" if required else "auto_review",
    }
    if required:
        params.update(
            approvalPolicy="on-request",
            sandbox="workspace-write",
            config={"features.request_permissions_tool": True},
        )
    client.request.assert_awaited_once_with("thread/resume", params)
    client.close.assert_awaited_once()


def test_required_client_args_still_cannot_override_provider_review() -> None:
    with pytest.raises(ValueError, match="restricted native deployment"):
        app_server.validate_codex_deployment_args(["-c", 'approvals_reviewer="auto_review"'])
