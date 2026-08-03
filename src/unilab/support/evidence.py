"""Small, stable support declarations for the generated support matrix.

The product package consumes this semantic declaration only.  Phase gates,
benchmark dumps, and repository-history checks live in ``tooling/acceptance``
and are intentionally not imported from runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

SUPPORT_EVIDENCE_PATH = Path("conf/support/evidence.yaml")


class DeclaredEvidenceLevel(str, Enum):
    TESTED = "tested"
    BENCHMARKED = "benchmarked"
    RECOMMENDED = "recommended"


@dataclass(frozen=True)
class SupportCombination:
    entrypoint_id: str
    task_slug: str
    env_name: str
    backend: str
    execution_profile: str | None
    evidence_level: DeclaredEvidenceLevel

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.entrypoint_id, self.task_slug, self.backend)


@dataclass(frozen=True)
class SupportEvidenceManifest:
    schema_version: int
    combinations: tuple[SupportCombination, ...]
    source: Path


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping")
    return value


def load_support_evidence(path: Path) -> SupportEvidenceManifest:
    try:
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    except Exception as exc:  # pragma: no cover - OmegaConf exception types vary
        raise ValueError(f"cannot load support evidence {path}: {exc}") from exc
    root = _mapping(raw, path="support evidence")
    if root.get("schema_version") != 1:
        raise ValueError("support evidence schema_version must be 1")
    combinations: list[SupportCombination] = []
    for index, value in enumerate(root.get("combinations", [])):
        item = _mapping(value, path=f"combinations[{index}]")
        try:
            level = DeclaredEvidenceLevel(str(item["evidence_level"]))
            combinations.append(
                SupportCombination(
                    entrypoint_id=str(item["entrypoint_id"]),
                    task_slug=str(item["task_slug"]),
                    env_name=str(item["env_name"]),
                    backend=str(item["backend"]),
                    execution_profile=(
                        None
                        if item.get("execution_profile") is None
                        else str(item["execution_profile"])
                    ),
                    evidence_level=level,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid support evidence combination {index}: {exc}") from exc
    return SupportEvidenceManifest(
        schema_version=1,
        combinations=tuple(combinations),
        source=path,
    )
