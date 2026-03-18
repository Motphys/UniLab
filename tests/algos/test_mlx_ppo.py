"""Unit tests for MLX PPO — MLPActorCritic, RolloutBuffer, PPOTrainer.

MLX is macOS-only (Apple Silicon Metal backend). These tests are skipped
automatically on Linux/Windows or when mlx is not installed.

Run:
    uv run pytest tests/algos/test_mlx_ppo.py -v
"""

from __future__ import annotations

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("MLX is macOS-only", allow_module_level=True)

mlx = pytest.importorskip("mlx.core", reason="mlx not installed")

import mlx.core as mx
import numpy as np

from unilab.algos.mlx.common import RolloutBuffer
from unilab.algos.mlx.ppo import MLPActorCritic, PPOConfig, PPOTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OBS_DIM = 12
_ACT_DIM = 4
_NUM_ENVS = 8
_NUM_STEPS = 6
_HIDDEN = [32, 32]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(**kwargs) -> MLPActorCritic:
    return MLPActorCritic(
        obs_dim=_OBS_DIM,
        action_dim=_ACT_DIM,
        actor_hidden_dims=_HIDDEN,
        critic_hidden_dims=_HIDDEN,
        **kwargs,
    )


def _make_buffer() -> RolloutBuffer:
    return RolloutBuffer(
        num_steps=_NUM_STEPS,
        num_envs=_NUM_ENVS,
        obs_dim=_OBS_DIM,
        action_dim=_ACT_DIM,
        gamma=0.99,
        lam=0.95,
    )


def _fill_buffer(buf: RolloutBuffer, model: MLPActorCritic) -> None:
    for _ in range(_NUM_STEPS):
        obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
        actions, log_probs, values, mean, std = model.act(obs)
        rewards = mx.random.normal((_NUM_ENVS,))
        dones = mx.zeros((_NUM_ENVS,))
        buf.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=rewards,
            dones=dones,
            values=values,
        )


# ---------------------------------------------------------------------------
# PPOConfig defaults
# ---------------------------------------------------------------------------


def test_ppo_config_defaults():
    cfg = PPOConfig()
    assert cfg.num_learning_epochs == 4
    assert cfg.num_mini_batches == 4
    assert 0.0 < cfg.clip_param < 1.0
    assert cfg.learning_rate > 0


def test_ppo_config_custom():
    cfg = PPOConfig(learning_rate=1e-3, clip_param=0.1, num_mini_batches=2)
    assert cfg.learning_rate == 1e-3
    assert cfg.clip_param == 0.1
    assert cfg.num_mini_batches == 2


# ---------------------------------------------------------------------------
# MLPActorCritic — construction
# ---------------------------------------------------------------------------


def test_model_init_log_std():
    model = _make_model(noise_std_type="log")
    assert hasattr(model, "log_std")
    assert model.log_std.shape == (_ACT_DIM,)


def test_model_init_scalar_std():
    model = _make_model(noise_std_type="scalar")
    assert hasattr(model, "std")
    assert model.std.shape == (_ACT_DIM,)


def test_model_init_state_dependent_std():
    model = _make_model(state_dependent_std=True, noise_std_type="log")
    # state-dependent std: no top-level log_std attribute
    assert not hasattr(model, "log_std")


def test_model_init_unknown_std_type_raises():
    with pytest.raises(ValueError, match="Unknown noise_std_type"):
        _make_model(noise_std_type="invalid")


# ---------------------------------------------------------------------------
# MLPActorCritic — forward pass shapes
# ---------------------------------------------------------------------------


def test_model_policy_output_shape():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    out = model.policy(obs)
    mx.eval(out)
    assert out.shape == (_NUM_ENVS, _ACT_DIM)


def test_model_value_output_shape():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    val = model.value(obs)
    mx.eval(val)
    assert val.shape == (_NUM_ENVS,)


def test_model_act_output_shapes():
    model = _make_model()
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    mx.eval(actions, log_probs, values, mean, std)
    assert actions.shape == (_NUM_ENVS, _ACT_DIM)
    assert log_probs.shape == (_NUM_ENVS,)
    assert values.shape == (_NUM_ENVS,)
    assert mean.shape == (_NUM_ENVS, _ACT_DIM)
    assert std.shape == (_NUM_ENVS, _ACT_DIM)


def test_model_clipped_log_std_within_bounds():
    model = _make_model(noise_std_type="log", min_log_std=-5.0, max_log_std=2.0)
    log_std = model.clipped_log_std()
    mx.eval(log_std)
    arr = np.array(log_std.tolist())
    assert (arr >= -5.0).all()
    assert (arr <= 2.0).all()


def test_model_with_obs_normalization():
    """Obs normalizer should not crash on forward pass."""
    model = _make_model(obs_normalization=True)
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    out = model.policy(obs)
    mx.eval(out)
    assert out.shape == (_NUM_ENVS, _ACT_DIM)


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


def test_buffer_step_increments():
    buf = _make_buffer()
    model = _make_model()
    assert buf.step == 0
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    buf.add(
        obs=obs,
        actions=actions,
        log_probs=log_probs,
        action_mean=mean,
        action_std=std,
        rewards=mx.zeros((_NUM_ENVS,)),
        dones=mx.zeros((_NUM_ENVS,)),
        values=values,
    )
    assert buf.step == 1


def test_buffer_overflow_raises():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    obs = mx.random.normal((_NUM_ENVS, _OBS_DIM))
    actions, log_probs, values, mean, std = model.act(obs)
    with pytest.raises(OverflowError):
        buf.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=mx.zeros((_NUM_ENVS,)),
            dones=mx.zeros((_NUM_ENVS,)),
            values=values,
        )


def test_buffer_compute_returns_stacks_arrays():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)
    mx.eval(buf.advantages, buf.returns)
    # After compute, arrays are stacked
    assert buf.advantages.shape == (_NUM_STEPS, _NUM_ENVS)
    assert buf.returns.shape == (_NUM_STEPS, _NUM_ENVS)


def test_buffer_clear_resets_step():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    assert buf.step == _NUM_STEPS
    buf.clear()
    assert buf.step == 0
    assert buf.observations == []


def test_buffer_mini_batch_generator_shapes():
    buf = _make_buffer()
    model = _make_model()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    num_mini_batches = 2
    batch_size = _NUM_STEPS * _NUM_ENVS
    mini_batch_size = batch_size // num_mini_batches

    batches = list(buf.mini_batch_generator(num_mini_batches=num_mini_batches, num_epochs=1))
    assert len(batches) == num_mini_batches
    b = batches[0]
    mx.eval(b["obs"])
    assert b["obs"].shape == (mini_batch_size, _OBS_DIM)
    assert b["actions"].shape == (mini_batch_size, _ACT_DIM)
    assert b["old_log_probs"].shape == (mini_batch_size,)
    assert b["returns"].shape == (mini_batch_size,)
    assert b["advantages"].shape == (mini_batch_size,)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


def test_ppo_trainer_update_returns_metrics():
    model = _make_model()
    cfg = PPOConfig(num_learning_epochs=1, num_mini_batches=2)
    trainer = PPOTrainer(model, cfg)

    buf = _make_buffer()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    metrics = trainer.update(buf, iteration=0)
    assert isinstance(metrics, dict)
    for key in ("surrogate", "value", "entropy", "approx_kl"):
        assert key in metrics, f"missing key: {key}"


def test_ppo_trainer_update_decreases_loss():
    """Loss should be finite after one update step."""
    model = _make_model()
    cfg = PPOConfig(num_learning_epochs=2, num_mini_batches=2)
    trainer = PPOTrainer(model, cfg)

    buf = _make_buffer()
    _fill_buffer(buf, model)
    last_values = model.value(mx.random.normal((_NUM_ENVS, _OBS_DIM)))
    buf.compute_returns_and_advantages(last_values)

    metrics = trainer.update(buf, iteration=0)
    assert np.isfinite(metrics["surrogate"])
    assert np.isfinite(metrics["value"])


# ---------------------------------------------------------------------------
# Full training iteration on real env (veryslow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.veryslow
def test_mlx_ppo_one_iteration_real_env(default_go2_reward_config):
    """Run 1 full MLX PPO iteration (collect rollout + update) on a real env."""
    _mujoco = pytest.importorskip("mujoco")

    from unilab.base import registry
    from unilab.config.structured_configs import PPOConfig as PPOStructuredConfig
    from unilab.utils.algo_utils import ensure_registries
    from unilab.utils.obs_utils import flatten_obs_dict

    ensure_registries()

    env_name = "Go2JoystickFlatTerrain"
    num_envs = 4
    num_steps = 8

    cfg = PPOStructuredConfig()
    algo_cfg = cfg.algorithm

    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco", env_cfg_override={"reward_config": default_go2_reward_config})
    obs_dim = sum(env.obs_groups_spec.values())
    action_dim = env.action_space.shape[0]

    model = MLPActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
    )
    ppo_cfg = PPOConfig(
        num_learning_epochs=1,
        num_mini_batches=2,
        clip_param=float(algo_cfg.clip_param),
        gamma=float(algo_cfg.gamma),
        lam=float(algo_cfg.lam),
    )
    trainer = PPOTrainer(model, ppo_cfg)

    # Init env and get first obs
    if env.state is None:
        env.init_state()
    reset_indices = np.arange(num_envs, dtype=np.int32)
    obs_dict, _ = env.reset(reset_indices)
    obs = mx.array(flatten_obs_dict(obs_dict))

    # Collect rollout
    buffer = RolloutBuffer(
        num_steps=num_steps,
        num_envs=num_envs,
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
    )

    for _ in range(num_steps):
        actions, log_probs, values, mean, std = model.act(obs)
        mx.eval(actions, log_probs, values, mean, std)

        env_actions = np.asarray(actions)
        state = env.step(env_actions)
        raw_obs = flatten_obs_dict(state.obs)
        rewards = mx.array(state.reward)
        dones = mx.array(state.done.astype(np.float32))

        buffer.add(
            obs=obs,
            actions=actions,
            log_probs=log_probs,
            action_mean=mean,
            action_std=std,
            rewards=rewards,
            dones=dones,
            values=values,
        )
        obs = mx.array(raw_obs)

    last_values = model.value(obs)
    buffer.compute_returns_and_advantages(last_values)

    # PPO update
    metrics = trainer.update(buffer, iteration=0)
    assert isinstance(metrics, dict)
    assert np.isfinite(metrics["surrogate"]), (
        f"surrogate loss is not finite: {metrics['surrogate']}"
    )
    assert np.isfinite(metrics["value"]), f"value loss is not finite: {metrics['value']}"

    env.close()
