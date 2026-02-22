import ray
import torch
import numpy as np
import time
from unilab.envs import registry
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable
from rsl_rl.models import MLPModel
import pkgutil
import importlib


# Ensure all environment modules are imported so they are registered
def ensure_registries():
    # Try importing unilab.envs.locomotion and walking
    try:
        import unilab.envs.locomotion

        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


ensure_registries()


class RslRlVecEnvWrapper:
    """Minimal wrapper to make unilab env compatible with rsl_rl policy."""

    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.num_actions = env.action_space.shape[0]

    def get_observations(self):
        # Convert numpy obs → torch → TensorDict
        # Assuming single 'policy' group for now
        obs_tensor = torch.as_tensor(self.env.state.obs, device=self.device, dtype=torch.float32)
        return TensorDict({"policy": obs_tensor}, batch_size=self.num_envs, device=self.device)


@ray.remote
class RolloutWorker:
    def __init__(self, env_name, env_cfg_overrides, device="cpu"):
        self.device = device
        self.env_name = env_name

        print(f"Worker initializing environment: {env_name}")
        # Force CPU device for env to save GPU memory
        # But we might need policy on GPU or CPU.
        # If policy on CPU -> slow inference?
        # Usually policy inference on CPU is fine for small batches.
        # But if we have GPU, we can put policy on GPU:0 (shared) or CPU.
        # Ray workers usually default to CPU unless num_gpus specified.
        # Let's keep policy on CPU for "CPU-physics" paradigm effectively.

        # self.env = registry.make(env_name, num_envs=env_cfg_overrides.get("num_envs", 1), ... )
        # Wait, env_cfg_overrides might not have num_envs
        num_envs = env_cfg_overrides.pop("num_envs", 1)
        self.env = registry.make(env_name, num_envs=num_envs, **env_cfg_overrides)
        self.num_envs = self.env.num_envs
        self.num_actions = self.env.action_space.shape[0]

        self.actor = None
        self.wrapper = RslRlVecEnvWrapper(self.env, device=device)

        # Reset
        all_indices = np.arange(self.num_envs)
        # Go2WalkTaskMj returns (physics_state, obs, info)
        _, obs, _ = self.env.reset(all_indices)
        self.current_obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

        # Episode Metrics
        self.episode_sums = {
            "reward": torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
            "length": torch.zeros(self.num_envs, dtype=torch.int32, device=self.device),
        }

        self.episode_metrics = {}
        # Per-step log entries (matching rsl_rl's extras["log"] / extras["episode"])
        self.step_log_entries = []

    def init_policy(self, policy_cfg):
        """Initialize worker policy architecture on CPU."""
        # Create dummy observation for initialization
        obs_dim = self.env.observation_space.shape[0]
        obs_example = torch.zeros((self.num_envs, obs_dim), device=self.device)

        # Construct TensorDict example as MLPModel expects
        # We need check obs_groups structure
        obs_groups = policy_cfg.get("obs_groups", {"default": ["policy"]})
        # If default is ["policy"], we wrap example in {"policy": ...}

        # Check if obs_groups uses just 'policy' key or maps to env obs
        # locomotion_params.py: "obs_groups": {"default": ["policy"]}
        # So MLPModel will look for obs["policy"]
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        actor_cfg = policy_cfg["actor"].copy()
        cls_name = actor_cfg.pop("class_name")
        actor_class = resolve_callable(cls_name)

        self.actor = actor_class(td_example, obs_groups, "actor", self.num_actions, **actor_cfg).to(self.device)
        self.actor.eval()

    def set_weights(self, weights):
        """Update local policy weights."""
        if self.actor is None:
            raise RuntimeError("Policy not initialized!")

        # weights: {"actor_state_dict": ...} or direct state_dict
        if "actor_state_dict" in weights:
            # Load only actor weights
            # Be careful about strict loading if keys differ
            # rsl_rl saves actor.state_dict(), so keys should match
            self.actor.load_state_dict(weights["actor_state_dict"])
        else:
            # Fallback
            self.actor.load_state_dict(weights)

    @torch.inference_mode()
    def sample(self, num_steps):
        """Collect num_steps transitions with pre-allocated buffers."""
        if self.actor is None:
            raise RuntimeError("Policy not initialized!")

        N = self.num_envs
        T = num_steps
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.num_actions
        dev = self.device

        # Pre-allocate storage tensors (avoids per-step allocation + final torch.stack)
        obs_buf = torch.zeros((T, N, obs_dim), device=dev, dtype=torch.float32)
        act_buf = torch.zeros((T, N, act_dim), device=dev, dtype=torch.float32)
        rew_buf = torch.zeros((T, N), device=dev, dtype=torch.float32)
        dones_buf = torch.zeros((T, N), device=dev, dtype=torch.bool)
        truncated_buf = torch.zeros((T, N), device=dev, dtype=torch.bool)
        logprob_buf = torch.zeros((T, N), device=dev, dtype=torch.float32)
        mu_buf = torch.zeros((T, N, act_dim), device=dev, dtype=torch.float32)
        sigma_buf = torch.zeros((T, N, act_dim), device=dev, dtype=torch.float32)

        # Reusable TensorDict for actor inference (avoid per-step creation)
        obs_td = TensorDict({"policy": self.current_obs}, batch_size=N, device=dev)

        for t in range(T):
            # 1. Store obs & run actor
            obs_buf[t] = self.current_obs
            obs_td["policy"] = self.current_obs  # In-place update

            actions = self.actor(obs_td, stochastic_output=True)
            logprob_buf[t] = self.actor.get_output_log_prob(actions)
            mu_buf[t] = self.actor.output_mean
            sigma_buf[t] = self.actor.output_std
            act_buf[t] = actions

            # 2. Step environment
            state = self.env.step(actions.cpu().numpy())

            next_obs = state.obs
            rew = state.reward
            terminated = state.terminated
            truncated_np = state.truncated
            infos = state.info
            dones = np.logical_or(terminated, truncated_np)

            # 3. Write directly into pre-allocated buffers
            rew_buf[t] = torch.as_tensor(rew, device=dev, dtype=torch.float32)
            dones_buf[t] = torch.as_tensor(dones, device=dev, dtype=torch.bool)
            truncated_buf[t] = torch.as_tensor(truncated_np, device=dev, dtype=torch.bool)

            # 4. Update obs
            self.current_obs = torch.as_tensor(next_obs, device=dev, dtype=torch.float32)

            # 5. Metrics accumulation
            self.episode_sums["reward"] += rew_buf[t]
            self.episode_sums["length"] += 1

            # Per-step log entries (rsl_rl extras["log"])
            if isinstance(infos, dict) and "log" in infos:
                self.step_log_entries.append(infos["log"])

            # Collect completed episodes
            done_indices = torch.nonzero(dones_buf[t]).squeeze(-1)
            if len(done_indices) > 0:
                for key in ["reward", "length"]:
                    val_tensor = self.episode_sums[key]
                    metric_name = "episode_returns" if key == "reward" else "episode_lengths"

                    if metric_name not in self.episode_metrics:
                        self.episode_metrics[metric_name] = []
                    self.episode_metrics[metric_name].extend(val_tensor[done_indices].cpu().tolist())
                    val_tensor[done_indices] = 0.0

        # Return pre-stacked buffers (no torch.stack needed)
        ret_storage = {
            "observations": obs_buf,
            "actions": act_buf,
            "rewards": rew_buf,
            "dones": dones_buf,
            "truncated": truncated_buf,
            "actions_log_prob": logprob_buf,
            "mu": mu_buf,
            "sigma": sigma_buf,
            "last_obs": self.current_obs.clone(),
        }

        metrics = self.episode_metrics.copy()
        step_logs = self.step_log_entries.copy()
        self.episode_metrics = {}
        self.step_log_entries = []

        ret_storage["metrics"] = metrics
        ret_storage["step_logs"] = step_logs

        return ret_storage

    def get_metrics(self):
        """Return gathered metrics."""
        return {}

    def close(self):
        if hasattr(self, "env"):
            self.env.close()
