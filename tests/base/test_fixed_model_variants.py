from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from omegaconf import OmegaConf
from unisim.dr.types import DomainRandomizationCapabilities

from unilab.base.base import EnvCfg
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.entity import EntityCfg
from unilab.base.scene import SceneCfg
from unilab.base.variants import (
    FixedModelVariantAssignmentCfg,
    FixedModelVariantCatalogCfg,
    FixedModelVariantCfg,
    build_fixed_variant_plan,
    materialize_fixed_model_variants,
    prepare_fixed_model_variants,
    require_fixed_model_variant_support,
    validate_fixed_model_variant_materialization,
)
from unilab.envs import manager_based_rl_env
from unilab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg, make_manager_based_rl_env


def _catalog(
    mode: Literal["round_robin", "explicit"] = "round_robin",
    names: tuple[str, ...] = (),
) -> FixedModelVariantCatalogCfg:
    return FixedModelVariantCatalogCfg(
        variants=(
            FixedModelVariantCfg("tool_a", "tools/a.xml"),
            FixedModelVariantCfg("tool_b", "tools/b.xml"),
        ),
        assignment=FixedModelVariantAssignmentCfg(mode=mode, explicit_variant_names=names),
    )


def test_hydra_materializes_typed_catalog_and_assignment() -> None:
    cfg = EnvCfg()

    apply_cfg_overrides(
        cfg,
        OmegaConf.create(
            {
                "fixed_model_variants": {
                    "variants": [
                        {"name": "tool_a", "source_model_file": "tools/a.xml"},
                        {"name": "tool_b", "source_model_file": "tools/b.xml"},
                    ],
                    "assignment": {"mode": "explicit", "explicit_variant_names": ["tool_b"]},
                }
            }
        ),
    )
    cfg.validate()

    assert cfg.fixed_model_variants is not None
    assert isinstance(cfg.fixed_model_variants, FixedModelVariantCatalogCfg)
    assert cfg.fixed_model_variants.variants == (
        FixedModelVariantCfg("tool_a", "tools/a.xml"),
        FixedModelVariantCfg("tool_b", "tools/b.xml"),
    )
    assert cfg.fixed_model_variants.assignment == FixedModelVariantAssignmentCfg(
        mode="explicit", explicit_variant_names=("tool_b",)
    )


def test_round_robin_materialization_is_final_and_immutable() -> None:
    materialization = materialize_fixed_model_variants(_catalog(), num_envs=5)

    assert materialization.variant_names == ("tool_a", "tool_b")
    np.testing.assert_array_equal(
        materialization.model_assignments, np.array([0, 1, 0, 1, 0], dtype=np.int32)
    )
    assert not materialization.model_assignments.flags.writeable
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        materialization.model_assignments[0] = 1
    copied = materialization.copy_model_assignments()
    assert copied.flags.writeable
    np.testing.assert_array_equal(copied, materialization.model_assignments)


def test_explicit_assignment_uses_names_not_engine_objects() -> None:
    materialization = materialize_fixed_model_variants(
        _catalog(mode="explicit", names=("tool_b", "tool_a", "tool_b")), num_envs=3
    )

    np.testing.assert_array_equal(
        materialization.model_assignments, np.array([1, 0, 1], dtype=np.int32)
    )
    validate_fixed_model_variant_materialization(materialization, num_envs=3)


@pytest.mark.parametrize(
    ("names", "num_envs", "match"),
    [
        (("tool_a",), 2, "exactly num_envs"),
        (("tool_a", "missing", "tool_b"), 3, "unknown variants"),
    ],
)
def test_explicit_assignment_fail_closed(names: tuple[str, ...], num_envs: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        materialize_fixed_model_variants(_catalog(mode="explicit", names=names), num_envs=num_envs)


def test_catalog_rejects_duplicate_and_empty_sources() -> None:
    with pytest.raises(ValueError, match="duplicate 'tool_a'"):
        FixedModelVariantCatalogCfg(
            variants=(
                FixedModelVariantCfg("tool_a", "a.xml"),
                FixedModelVariantCfg("tool_a", "b.xml"),
            )
        )
    with pytest.raises(ValueError, match="must not be empty"):
        FixedModelVariantCatalogCfg()
    with pytest.raises(ValueError, match="source_model_file must be a non-empty string"):
        FixedModelVariantCatalogCfg(variants=(FixedModelVariantCfg("tool_a", " "),))


def test_final_assignment_validation_rejects_a_writable_array() -> None:
    materialization = materialize_fixed_model_variants(_catalog(), num_envs=2)
    writable = np.arange(2, dtype=np.int32)
    object.__setattr__(materialization, "model_assignments", writable)

    with pytest.raises(ValueError, match="must be read-only"):
        validate_fixed_model_variant_materialization(materialization, num_envs=2)


def test_fixed_variant_support_fails_closed_on_one_capability_contract() -> None:
    catalog = _catalog()

    with pytest.raises(NotImplementedError, match="does not support"):
        require_fixed_model_variant_support(DomainRandomizationCapabilities())

    materialization = prepare_fixed_model_variants(
        catalog,
        2,
        DomainRandomizationCapabilities(supports_fixed_variants=True),
    )
    validate_fixed_model_variant_materialization(materialization, num_envs=2)


def test_fixed_variant_materialization_builds_unisim_plan() -> None:
    materialization = materialize_fixed_model_variants(_catalog(), num_envs=2)
    plan = build_fixed_variant_plan(materialization)

    assert plan.layout.value == "same_layout"
    np.testing.assert_array_equal(plan.assignment, materialization.model_assignments)
    assert tuple(variant.model_file for variant in plan.variants) == (
        "tools/a.xml",
        "tools/b.xml",
    )
    assert not plan.assignment.flags.writeable


def test_variant_owner_module_does_not_reference_engine_internals() -> None:
    repo_root = Path(__file__).parents[2]
    source = (repo_root / "src" / "unilab" / "base" / "variants.py").read_text(encoding="utf-8")
    manager_source = (repo_root / "src" / "unilab" / "envs" / "manager_based_rl_env.py").read_text(
        encoding="utf-8"
    )

    assert "mjbatch" not in source
    assert "MjSpec" not in source
    assert "mujoco" not in source
    assert "mjbatch" not in manager_source
    assert "MjSpec" not in manager_source


def test_manager_env_fails_closed_before_variant_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnsupportedBackend:
        backend_type = "test"
        num_envs = 2
        cleaned = False

        def get_dr_capabilities(self) -> DomainRandomizationCapabilities:
            return DomainRandomizationCapabilities()

        def cleanup_scene_assets(self) -> None:
            self.cleaned = True

    backend = UnsupportedBackend()
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            model_file="scene.xml",
            entities={"robot": EntityCfg(root_body_name="base")},
        ),
        max_episode_seconds=1.0,
        fixed_model_variants=_catalog(),
    )
    monkeypatch.setattr(manager_based_rl_env, "env_backend_kwargs", lambda _cfg: {})
    monkeypatch.setattr(
        manager_based_rl_env,
        "create_backend",
        lambda *_args, **_kwargs: backend,
    )

    def _fail_if_env_is_constructed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsupported variants must fail before env construction")

    monkeypatch.setattr(manager_based_rl_env, "ManagerBasedRlEnv", _fail_if_env_is_constructed)

    with pytest.raises(NotImplementedError, match="does not support"):
        make_manager_based_rl_env(cfg, num_envs=2, backend_type="mujoco")

    assert backend.cleaned is True
    assert cfg.scene is not None
    assert cfg.scene.fixed_variant_plan is not None
    np.testing.assert_array_equal(
        cfg.scene.fixed_variant_plan.assignment, np.array([0, 1], dtype=np.int32)
    )
