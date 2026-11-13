"""Policy-device transfers at the public RSL-RL adapter boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unilab.rl.adapters.rsl_rl.vec_env import RslRlVecEnvAdapter


@pytest.fixture(params=["cpu", "mps"])
def policy_device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable")
    return request.param


class _CpuEnv:
    def __init__(self, dtype: torch.dtype = torch.float64):
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.cfg = SimpleNamespace(max_episode_seconds=10.0, ctrl_dt=0.02)
        self.observation_space = SimpleNamespace(shape=(2,))
        self.action_space = SimpleNamespace(shape=(1,))
        self.obs_groups_spec = {"obs": 2, "critic": 1}
        self.state = SimpleNamespace(
            obs={
                "obs": torch.tensor([[1.25, -2.5], [3.75, 4.5]], dtype=dtype),
                "critic": torch.tensor([[0.5], [0.75]], dtype=dtype),
            },
            reward=torch.tensor([0.25, 0.5], dtype=torch.float64),
            terminated=torch.tensor([True, False]),
            truncated=torch.tensor([False, True]),
            info={},
        )
        self.last_action = None

    def reset(self):
        return self.state.obs, {}

    def step(self, actions):
        self.last_action = actions
        return self.state

    def close(self):
        pass


def test_float64_cpu_observations_and_rewards_reach_policy_device(policy_device):
    env = _CpuEnv()
    wrapped = RslRlVecEnvAdapter(env, device=policy_device)

    reset_obs, _ = wrapped.reset()
    for key, original in (("actor", env.state.obs["obs"]), ("critic", env.state.obs["critic"])):
        assert reset_obs[key].device.type == policy_device
        assert reset_obs[key].dtype == torch.float32
        torch.testing.assert_close(reset_obs[key].cpu(), original.float())
    torch.testing.assert_close(reset_obs["policy"].cpu(), env.state.obs["obs"].float())
    torch.testing.assert_close(wrapped.get_observations()["actor"].cpu(), reset_obs["actor"].cpu())
    torch.testing.assert_close(
        wrapped.get_privileged_observations().cpu(), env.state.obs["critic"].float()
    )

    _, rewards, dones, info = wrapped.step(torch.zeros((2, 1), device=policy_device))
    assert rewards.device.type == policy_device
    assert rewards.dtype == torch.float32
    torch.testing.assert_close(rewards.cpu(), env.state.reward.float())
    assert dones.dtype == torch.bool
    torch.testing.assert_close(dones.cpu(), torch.tensor([True, True]))
    assert info["time_outs"].dtype == torch.bool
    torch.testing.assert_close(info["time_outs"].cpu(), env.state.truncated)
    assert env.last_action.device.type == "cpu"
    assert env.last_action.dtype == torch.float32
    # Conversion must not mutate the environment's CPU float64 buffers.
    assert env.state.obs["obs"].dtype == env.state.reward.dtype == torch.float64


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_integer_and_boolean_observation_dtypes_are_preserved(policy_device, dtype):
    env = _CpuEnv(dtype=dtype)
    if dtype == torch.int64:
        # A float32 detour would irreversibly round this integer.
        env.state.obs["obs"][0, 0] = 2**53 + 1
    wrapped = RslRlVecEnvAdapter(env, device=policy_device)
    obs = wrapped.get_observations()
    assert obs["actor"].dtype == dtype
    assert obs["actor"].device.type == policy_device
    torch.testing.assert_close(obs["actor"].cpu(), env.state.obs["obs"])
