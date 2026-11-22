"""Optional deployment constraints for original native-provider launches.

Absent policy keeps the existing interactive/user-session behavior. This is
configuration, never an authentication or tool-approval authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnigent.runner.native.orchestration import ResolvedSpec
    from omnigent.spec.types import AgentSpec


@dataclass(frozen=True, kw_only=True)
class NativeDeploymentPolicy:
    own_auth_only: bool = True
    trust_project: bool = False
    trust_hooks: bool = False
    allow_project_config: bool = False
    personal_state: bool = False
    host_skills: bool = False
    allow_bypass: bool = False


def deployment_policy_from_spec(
    spec: AgentSpec | ResolvedSpec | None,
) -> NativeDeploymentPolicy | None:
    """Read exact boolean constraints from the already resolved immutable spec."""
    if spec is None:
        return None
    from omnigent.runner.native.orchestration import ResolvedSpec

    resolved = spec.spec if isinstance(spec, ResolvedSpec) else spec
    raw = resolved.executor.config.get("native_deployment_policy")
    if raw is None:
        return None
    if (
        type(resolved.executor.config.get("native_deployment_policy_version")) is not int
        or resolved.executor.config["native_deployment_policy_version"] != 1
    ):
        raise ValueError("native deployment policy version 1 is required")
    names = {item.name for item in fields(NativeDeploymentPolicy)}
    if (
        not isinstance(raw, dict)
        or set(raw) - names
        or any(type(value) is not bool for value in raw.values())
    ):
        raise ValueError("invalid native deployment policy: exact known boolean fields required")
    return NativeDeploymentPolicy(**raw)


def _without_skill_runtime_locations(spec: AgentSpec) -> AgentSpec:
    """Keep every declaration while excluding parser-derived extraction handles.

    Skill/reference file identity remains owned by the immutable bundle digest;
    this profile identity does not replace that resource-byte verification.
    """
    return replace(
        spec,
        skills=[replace(skill, skill_dir=None) for skill in spec.skills],
        sub_agents=[_without_skill_runtime_locations(child) for child in spec.sub_agents],
    )


def deployment_spec_identity(spec: AgentSpec | ResolvedSpec | None) -> str:
    """Fingerprint the complete persisted spec, including explicit policy version.

    Bundle persistence and its request hash remain owned by the existing server
    contract; this digest only rejects reuse of a different private profile.
    """
    from omnigent.runner.native.orchestration import ResolvedSpec

    if spec is None or deployment_policy_from_spec(spec) is None:
        raise ValueError("restricted deployment requires the persisted agent spec")
    resolved = spec.spec if isinstance(spec, ResolvedSpec) else spec
    return hashlib.sha256(
        json.dumps(
            asdict(_without_skill_runtime_locations(resolved)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
