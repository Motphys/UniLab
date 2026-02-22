"""Off-policy Ray rollout worker for TD3 and SAC.

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy (with exploration noise).  Designed to run on CPU while the
learner trains on MPS/GPU.
"""

import ray
import torch
import numpy as np
import pkgutil
import importlib
from tensordict import TensorDict

from rsl_rl.utils import resolve_callable


# Ensure all environment modules are imported so they are registered
def ensure_registries():
    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception as e:
                    print(f"[OffPolicyWorker] Warning: failed to import {name}: {e}")
    except ImportError as e:
        print(f"[OffPolicyWorker] Warning: failed to import unilab.envs.locomotion: {e}")


ensure_registries()

from unilab.envs import registry  # noqa: E402


@ray.remote
class OffPolicyWorker:
    """Ray actor that collects transitions for off-policy algorithms."""

    def __init__(self, env_name, env_cfg_overrides, device="cpu", exploration_noise=0.1):
        self.device = device
        self.env_name = env_name
        self.exploration_noise = exploration_noise

        # Ensure registries are populated in this process
        ensure_registries()

        # Debug: print registered envs
        registered = registry.list_registered_envs()
        print(f"[OffPolicyWorker] Registered envs: {list(registered.keys())}")

        num_envs = env_cfg_overrides.pop("num_envs", 1)
        self.env = registry.make(env_name, num_envs=num_envs)
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]

        self.actor = None

        # Reset env
        all_indices = np.arange(self.num_envs)
        _, obs, _ = self.env.reset(all_indices)
        self.current_obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

        # Episode metrics
        self.episode_sums = {
            "reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "length": torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
        }
        self.episode_metrics = {}
        self.step_log_entries = []

    def init_policy(self, policy_cfg):
        """Initialize worker policy architecture on CPU."""
        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        obs_groups = policy_cfg.get("obs_groups", {"default": ["policy"]})
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        actor_cfg = policy_cfg["actor"].copy()
        cls_name = actor_cfg.pop("class_name")
        actor_class = resolve_callable(cls_name)

        self.actor = actor_class(
            td_example, obs_groups, "actor", self.num_actions, **actor_cfg
        ).to(self.device)
        self.actor.eval()

    def set_weights(self, weights):
        """Update local policy weights."""
        if self.actor is None:
            raise RuntimeError("Policy not initialized!")
        if "actor_state_dict" in weights:
            self.actor.load_state_dict(weights["actor_state_dict"])
        else:
            self.actor.load_state_dict(weights)

    def set_exploration_noise(self, noise):
        """Update exploration noise level."""
        self.exploration_noise = noise

    @torch.inference_mode()
    def sample(self, num_steps, stochastic=False):
        """Collect num_steps transitions.

        Args:
            num_steps: Number of environment steps to collect per environment.
            stochastic: If True, sample from stochastic policy (SAC).
                        If False, use deterministic + Gaussian noise (TD3).

        Returns:
            dict with pre-stacked buffers + metrics.
        """
        if self.actor is None:
            raise RuntimeError("Policy not initialized!")

        N = self.num_envs
        T = num_steps
        dev = self.device

        # Pre-allocate buffers
        obs_buf = torch.zeros((T, N, self.obs_dim), device=dev, dtype=torch.float32)
        act_buf = torch.zeros((T, N, self.num_actions), device=dev, dtype=torch.float32)
        rew_buf = torch.zeros((T, N), device=dev, dtype=torch.float32)
        next_obs_buf = torch.zeros((T, N, self.obs_dim), device=dev, dtype=torch.float32)
        dones_buf = torch.zeros((T, N), device=dev, dtype=torch.float32)

        obs_td = TensorDict({"policy": self.current_obs}, batch_size=N, device=dev)

        for t in range(T):
            obs_buf[t] = self.current_obs
            obs_td["policy"] = self.current_obs

            if stochastic:
                actions = self.actor(obs_td, stochastic_output=True)
            else:
                actions = self.actor(obs_td)
                noise = torch.randn_like(actions) * self.exploration_noise
                actions = (actions + noise).clamp(-1.0, 1.0)

            act_buf[t] = actions

            # Step environment — unilab env returns a state object
            state = self.env.step(actions.cpu().numpy())
            next_obs = state.obs
            rew = state.reward
            terminated = state.terminated
            truncated = state.truncated
            infos = state.info
            dones = np.logical_or(terminated, truncated)

            rew_buf[t] = torch.as_tensor(rew, device=dev, dtype=torch.float32)
            next_obs_t = torch.as_tensor(next_obs, device=dev, dtype=torch.float32)
            next_obs_buf[t] = next_obs_t
            dones_buf[t] = torch.as_tensor(dones, device=dev, dtype=torch.float32)

            self.current_obs = next_obs_t

            # Metrics
            self.episode_sums["reward"] += rew_buf[t]
            self.episode_sums["length"] += 1

            if isinstance(infos, dict) and "log" in infos:
                self.step_log_entries.append(infos["log"])

            done_indices = torch.nonzero(dones_buf[t]).squeeze(-1)
            if len(done_indices) > 0:
                for key in ["reward", "length"]:
                    val_tensor = self.episode_sums[key]
                    metric_name = "episode_returns" if key == "reward" else "episode_lengths"
                    if metric_name not in self.episode_metrics:
                        self.episode_metrics[metric_name] = []
                    self.episode_metrics[metric_name].extend(val_tensor[done_indices].cpu().tolist())
                    val_tensor[done_indices] = 0.0

        ret = {
            "observations": obs_buf,        # [T, N, D]
            "actions": act_buf,              # [T, N, A]
            "rewards": rew_buf,              # [T, N]
            "next_observations": next_obs_buf,  # [T, N, D]
            "dones": dones_buf,              # [T, N]
            "metrics": self.episode_metrics.copy(),
            "step_logs": self.step_log_entries.copy(),
        }

        self.episode_metrics = {}
        self.step_log_entries = []
        return ret

    def close(self):
        if hasattr(self, "env"):
            self.env.close()
