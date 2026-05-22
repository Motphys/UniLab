from __future__ import annotations

from omegaconf import OmegaConf

from unilab.training.metrics import stable_config_hash


def test_stable_config_hash_is_order_independent():
    cfg_a = OmegaConf.create({"algo": {"seed": 1, "num_envs": 2}, "training": {"logger": "tensorboard"}})
    cfg_b = OmegaConf.create({"training": {"logger": "tensorboard"}, "algo": {"num_envs": 2, "seed": 1}})

    assert stable_config_hash(cfg_a) == stable_config_hash(cfg_b)
