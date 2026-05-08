from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.training.seed import (
    apply_training_seed,
    derive_worker_seed,
    resolve_training_seed,
)

_ROOT_DIR = Path(__file__).resolve().parents[2]
_CONF_DIR = _ROOT_DIR / "conf"


def _compose(config_dir: str, overrides: list[str] | None = None):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR / config_dir), version_base="1.3"):
        return compose("config", overrides=overrides or [])


def test_resolve_training_seed_prefers_algo_seed_over_legacy_training_seed():
    cfg = OmegaConf.create({"algo": {"seed": 7}, "training": {"seed": 99}})

    seed_info = resolve_training_seed(cfg)

    assert seed_info.configured_seed == 7
    assert seed_info.configured_seed_source == "algo.seed"
    assert seed_info.effective_seed == 7


def test_apply_training_seed_controls_python_numpy_and_torch_rng():
    apply_training_seed(123, torch_runtime=True, cuda=True)
    first = (random.random(), np.random.rand(), torch.rand(3))

    apply_training_seed(123, torch_runtime=True, cuda=True)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert second[0] == first[0]
    assert second[1] == first[1]
    assert torch.equal(second[2], first[2])


def test_apply_training_seed_rejects_negative_seed():
    with pytest.raises(ValueError, match="non-negative"):
        apply_training_seed(-1)


def test_derive_worker_seed_is_deterministic_and_distinct_from_base_seed():
    assert derive_worker_seed(10, worker_index=0) == 11
    assert derive_worker_seed(10, worker_index=3) == 14
    assert derive_worker_seed(None, worker_index=3) is None


@pytest.mark.parametrize(
    ("config_dir", "overrides"),
    [
        ("ppo", ["task=go1_joystick_flat/mujoco"]),
        ("ppo", ["task=go1_joystick_flat/mujoco", "algo.seed=41"]),
        ("ppo", ["task=sharpa_inhand/mujoco_hora"]),
        ("appo", ["task=go1_joystick_flat/mujoco"]),
        ("appo", ["task=sharpa_inhand/mujoco_hora"]),
        ("offpolicy", ["algo=sac", "task=sac/g1_walk_flat/mujoco"]),
        ("offpolicy", ["algo=td3", "task=td3/g1_walk_flat/mujoco"]),
    ],
)
def test_owner_configs_resolve_algorithm_seed_contract(config_dir: str, overrides: list[str]):
    cfg = _compose(config_dir, overrides)

    seed_info = resolve_training_seed(cfg)

    assert seed_info.configured_seed == int(cfg.algo.seed)
    assert seed_info.configured_seed_source == "algo.seed"
    assert seed_info.effective_seed == int(cfg.algo.seed)


def test_mlx_config_keeps_training_seed_as_legacy_fallback_only():
    mlx_cfg = OmegaConf.load(_CONF_DIR / "ppo" / "config_mlx.yaml")
    mlx_cfg.algo.seed = 17
    mlx_cfg.training.seed = 29

    seed_info = resolve_training_seed(mlx_cfg)

    assert seed_info.configured_seed == 17
    assert seed_info.configured_seed_source == "algo.seed"
