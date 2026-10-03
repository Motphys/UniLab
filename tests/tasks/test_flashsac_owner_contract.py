from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.base.config_materialization import apply_cfg_overrides
from unilab.base.variants import FixedModelVariantCatalogCfg, FixedModelVariantCfg
from unilab.envs import ManagerBasedRlEnvCfg
from unilab.tasks.motion_tracking.common.manager_terms import TensorMotionCommandCfg
from unilab.tasks.motion_tracking.g1 import flashsac_owner_contract as module

ROOT_DIR = Path(__file__).parents[2]
CONF_DIR = ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"


def _materialize_task(task: str, *, algo: str = "flashsac") -> ManagerBasedRlEnvCfg:
    GlobalHydra.instance().clear()
    config_dir = CONF_DIR if algo == "flashsac" else ROOT_DIR / "src/unilab/conf/sac"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        composed = compose("config", overrides=[f"task={task}"])
    override = BackendAdapter(
        composed, root_dir=ROOT_DIR, algo_name=algo
    ).build_task_env_cfg_override()
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(cfg, override)
    return cfg


def test_flashsac_owner_fingerprint_accepts_canonical_backends() -> None:
    mujoco = _materialize_task("g1_motion_tracking/mujoco")
    mjwarp = _materialize_task("g1_motion_tracking/mjwarp")
    assert module._torch_g1_flashsac_owner_identity(mujoco) == (
        module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11
    )
    assert module._torch_g1_flashsac_owner_identity(mjwarp) == (
        module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V16
    )


def test_flashsac_motrix_owner_is_out_of_tensor_manager_scope() -> None:
    registry.ensure_registries()
    assert "motrix" not in registry._envs["G1MotionTrackingSAC"].env_factory_dict


def test_reusable_tensor_runtime_accepts_second_g1_manager_owner() -> None:
    cfg = _materialize_task("g1_motion_tracking/mjwarp_tensor", algo="sac")
    assert module._torch_g1_flashsac_owner_identity(cfg) == module._TORCH_G1_SAC_OWNER_IDENTITY_V2


def test_reusable_tensor_runtime_accepts_second_manager_based_task() -> None:
    cfg = _materialize_task("g1_flip_tracking/mjwarp_tensor", algo="sac")
    assert cfg.commands["motion"].sampling_mode == "mixed"
    assert isinstance(cfg.commands["motion"], TensorMotionCommandCfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.observations["actor"].terms["base_lin_vel"].noise, "n_min", -0.2),
        lambda cfg: setattr(cfg.rewards["motion_reward_pack"], "root_pos_std", 0.4),
        lambda cfg: setattr(cfg.actions["joint_pos"], "clip", {"joint": (0.0, 1.0)}),
        lambda cfg: setattr(cfg.commands["motion"].params, "adaptive_alpha", 0.1),
    ],
)
def test_flashsac_owner_fingerprint_fails_closed(mutate) -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    mutate(cfg)
    with pytest.raises(ValueError, match="canonical owner contract"):
        module._validate_torch_g1_flashsac_owner_contract(cfg)


def test_fused_motion_reward_pack_owner_identity_is_canonical() -> None:
    """The fused Manager reward is the canonical FlashSAC semantic owner."""
    for backend in ("mujoco", "mjwarp", "newton"):
        cfg = _materialize_task(f"g1_motion_tracking/{backend}")
        identity = module._torch_g1_flashsac_owner_identity(cfg)
        expected = {
            "mujoco": module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11,
            "genesis": module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11,
            "newton": module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V11,
            "mjwarp": module._TORCH_G1_FLASHSAC_OWNER_IDENTITY_V16,
        }[backend]
        assert identity == expected


def test_flashsac_owner_rejects_fixed_model_variants_before_backend_creation() -> None:
    cfg = _materialize_task("g1_motion_tracking/mujoco")
    cfg.fixed_model_variants = FixedModelVariantCatalogCfg(
        variants=(FixedModelVariantCfg(name="variant", source_model_file="variant.xml"),)
    )
    with pytest.raises(ValueError, match="does not support fixed model variants"):
        module._validate_torch_g1_flashsac_owner_contract(cfg)
