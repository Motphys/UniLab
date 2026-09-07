"""Go2's research SuperDex owner preserves policy I/O and reset semantics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.base import registry
from unilab.base.config_adapter import BackendAdapter
from unilab.utils.sim2sim import DENYLIST, CrossBackendIncompatibleError, resolve_sim2sim_config

ROOT = Path(__file__).resolve().parents[2]


def _owner(backend: str) -> tuple[Any, dict[str, Any]]:
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "src/unilab/conf/ppo"), version_base="1.3"):
        owner = compose("config", overrides=[f"task=go2_joystick_flat/{backend}"])
    return owner, BackendAdapter(owner, root_dir=ROOT).build_task_env_cfg_override()


def test_go2_superdex_preserves_mujoco_policy_contract() -> None:
    source, _ = _owner("mujoco")
    target, _ = _owner("superdex")
    for field in DENYLIST:
        assert OmegaConf.select(source, field) == OmegaConf.select(target, field), field
    assert target.env.events.pd_gains is None
    assert target.env.events.reset_root_state_uniform is not None
    assert target.env.superdex_allow_contact_approximation is True
    assert target.env.sim_dt == source.env.sim_dt
    assert target.env.ctrl_dt == source.env.ctrl_dt


def test_go2_superdex_native_rollout_and_free_root_reset() -> None:
    pytest.importorskip("superdex.physics")
    if not (ROOT / "src/unilab/assets/robots/go2/assets/base_0.obj").is_file():
        pytest.skip("native Go2 rollout requires the registered Go2 robot assets")
    _, override = _owner("superdex")
    env = registry.make(
        "Go2JoystickFlat", sim_backend="superdex", num_envs=2, env_cfg_override=override
    )
    try:
        state = env.init_state()
        assert env.action_space.shape == (12,)
        assert env.obs_groups_spec == {"obs": 49, "critic": 52}
        for _ in range(32):
            state = env.step(np.zeros((2, 12), dtype=np.float32))
            assert np.isfinite(state.reward).all()
            assert all(np.isfinite(value).all() for value in state.obs.values())
        before = env.scene["robot"].data.root_link_pos_w.copy()
        obs, _ = env.reset(env_ids=np.array([0], dtype=np.int32))
        assert obs["obs"].shape == (1, 49)
        np.testing.assert_array_equal(env.scene["robot"].data.root_link_pos_w[1], before[1])
        np.testing.assert_allclose(
            np.linalg.norm(env.scene["robot"].data.root_link_quat_w, axis=-1), 1.0, atol=1e-5
        )
    finally:
        env.close()


def test_go2_superdex_executes_mujoco_checkpoint(tmp_path: Path) -> None:
    checkpoint_value = os.environ.get("UNILAB_SUPERDEX_GO2_CHECKPOINT")
    if not checkpoint_value:
        pytest.skip("set UNILAB_SUPERDEX_GO2_CHECKPOINT to a MuJoCo Go2 PPO checkpoint")
    pytest.importorskip("superdex.physics")
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from uni_rl.algos.rsl_rl import RslRlVecEnvWrapper, get_policy_obs_dims, normalize_ppo_train_cfg

    from unilab.training.run import algo_config_dict
    from unilab.visualization.interactive_playback import (
        RslRlPlaybackConfig,
        create_rsl_rl_playback_session,
        infer_checkpoint_actor_input_dim,
        make_sim2sim_preflight,
    )

    checkpoint = Path(checkpoint_value).resolve(strict=True)
    owner, overrides = _owner("superdex")
    # Validate source policy-I/O before any environment is constructed.
    resolve_sim2sim_config(str(checkpoint.parent), owner, algo_name="ppo", strict=True)
    incompatible = OmegaConf.create(OmegaConf.to_container(owner, resolve=True))
    incompatible.env.actions.joint_pos.scale = 0.5
    with pytest.raises(CrossBackendIncompatibleError, match="env.actions"):
        resolve_sim2sim_config(str(checkpoint.parent), incompatible, algo_name="ppo", strict=True)

    env = registry.make(
        "Go2JoystickFlat", sim_backend="superdex", num_envs=2, env_cfg_override=overrides
    )
    try:
        session, _, loaded = create_rsl_rl_playback_session(
            playback_cfg=RslRlPlaybackConfig(
                task="Go2JoystickFlat",
                load_run=str(checkpoint),
                checkpoint=None,
                action_mode="policy",
                policy_obs_mode="flat",
                algo_log_name="rsl_rl_ppo",
                log_root=str(tmp_path),
                num_envs=2,
            ),
            env_factory=lambda n: env,
            algo_config=algo_config_dict(owner),
            root_dir=ROOT,
            device="cpu",
            checkpoint_resolver=lambda *_: str(checkpoint),
            checkpoint_input_dim_reader=infer_checkpoint_actor_input_dim,
            entrypoint_log_root=lambda *args, **kwargs: tmp_path,
            wrapper_cls=RslRlVecEnvWrapper,
            runner_cls=OnPolicyRunner,
            policy_obs_dims_getter=get_policy_obs_dims,
            train_cfg_normalizer=normalize_ppo_train_cfg,
            sim2sim_preflight=make_sim2sim_preflight(owner, algo_name="ppo"),
            guard_algo_name="ppo",
        )
        assert loaded == str(checkpoint) and session.policy is not None
        session.reset()
        with torch.inference_mode():
            for _ in range(64):
                obs = session.step_once()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert np.isfinite(env.state.reward).all()
        assert session.step_count == 64
    finally:
        env.close()
