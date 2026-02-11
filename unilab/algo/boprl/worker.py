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

        print(f"DEBUG WORKER {self.env_name}: Actor Config: {actor_cfg}")
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
        """Collect num_steps transitions."""
        if self.actor is None:
            raise RuntimeError("Policy not initialized!")

        storage = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],  # Not computed here (unless we add critic)
            "actions_log_prob": [],
            "returns": [],  # Computed on learner? Or here with critic?
            # For strict PPO, we need values/returns.
            # If we don't have critic on worker, we send raw trajectories to learner.
            # Learner computes values? But learner needs state for value.
            # So we send states.
        }

        # Important: detailed PPO needs GAE, which needs Value(s_t).
        # We need Critic on worker to compute Value(s_t), OR we send s_t to learner.
        # Sending s_t (obs) to learner is fine. Learner can compute V(s_t) with its critic.
        # But we need V(s_{t+1}) for GAE.
        # So Learner needs (s_t, r_t, d_t, s_{t+1}).

        # Baseline approach: Worker collects (s_t, a_t, r_t, d_t).
        # Learner computes V(s_t) and GAE.
        # Advantage of computing V on worker: parallelism.
        # Disadvantage: need to sync critic weights to worker.
        # Let's sync critic too? It matches "Async PPO" better.
        # But for minimal bandwidth, just sync actor (policy) and compute global critic on GPU.
        # GPU is fast at batch inference. CPU worker is slow at neural net inference.
        # So "CPU Physics, GPU Learning" implies minimal NN on CPU.
        # Thus: Only Actor on CPU (for action selection).
        # Critic on GPU (for Value & Update).

        # Pre-allocate storage lists (minor opt)
        obs_list = []
        act_list = []
        rew_list = []
        dones_list = []
        log_prob_list = []
        truncated_list = []

        # TensorDict reuse? No, better to recreate wrapper or slice.
        # But for speed, just wrap current_obs.

        for _ in range(num_steps):
            # 1. Obs
            # obs_td = TensorDict({"policy": self.current_obs}, batch_size=self.num_envs, device=self.device)
            # Optimization: construct TensorDict once? No, current_obs changes.
            # But TensorDict overhead is small.

            with torch.inference_mode():
                obs_td = TensorDict({"policy": self.current_obs}, batch_size=self.num_envs, device=self.device)

                # 2. Action
                # MLPModel.forward(stochastic_output=True) samples automatically
                actions = self.actor(obs_td, stochastic_output=True)
                # It also updates self.distribution internally so we can get log_prob
                log_prob = self.actor.get_output_log_prob(actions)

            # 3. Step
            actions_np = actions.cpu().numpy()

            state = self.env.step(actions_np)

            # Unpack state
            next_obs = state.obs
            rew = state.reward
            terminated = state.terminated
            truncated = state.truncated
            infos = state.info

            dones = np.logical_or(terminated, truncated)

            # Store (keep purely tensor)
            obs_list.append(obs_td)
            act_list.append(actions)
            rew_list.append(torch.as_tensor(rew, device=self.device, dtype=torch.float32))
            dones_list.append(torch.as_tensor(dones, device=self.device, dtype=torch.bool))
            log_prob_list.append(log_prob)
            # Store truncated (time-out) for bootstrap correction
            truncated_list.append(torch.as_tensor(truncated, device=self.device, dtype=torch.bool))

            # Update obs
            self.current_obs = torch.as_tensor(next_obs, device=self.device, dtype=torch.float32)

            # Metrics Accumulation
            rew_tensor = torch.as_tensor(rew, device=self.device, dtype=torch.float32)
            dones_tensor = torch.as_tensor(dones, device=self.device, dtype=torch.bool)

            # 1. Manual total
            self.episode_sums["reward"] += rew_tensor
            self.episode_sums["length"] += 1

            # 2. Detailed components
            if isinstance(infos, dict) and "reward_components" in infos:
                for key, val in infos["reward_components"].items():
                    # val is expected to be array of shape (num_envs,)
                    val_tensor = torch.as_tensor(val, device=self.device, dtype=torch.float32)
                    if key not in self.episode_sums:
                        self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
                    self.episode_sums[key] += val_tensor

            # Collect completed episodes
            done_indices = torch.nonzero(dones_tensor).squeeze(-1)
            if len(done_indices) > 0:
                for key, val_tensor in self.episode_sums.items():
                    # Map names for compatibility with runner
                    metric_name = key
                    if key == "reward":
                        metric_name = "episode_returns"
                    if key == "length":
                        metric_name = "episode_lengths"

                    if metric_name not in self.episode_metrics:
                        self.episode_metrics[metric_name] = []
                    self.episode_metrics[metric_name].extend(val_tensor[done_indices].cpu().tolist())

                    # Reset
                    val_tensor[done_indices] = 0.0

        # Stack
        # Ensure we return expected keys for Learner

        # Learner expects: observations, actions, rewards, dones, actions_log_prob
        # observations must be [T, N, D] (or TensorDict)

        ret_storage = {
            "observations": torch.stack([td["policy"] for td in obs_list]),
            "actions": torch.stack(act_list),
            "rewards": torch.stack(rew_list),
            "dones": torch.stack(dones_list),
            "truncated": torch.stack(truncated_list),
            "actions_log_prob": torch.stack(log_prob_list),
            "last_obs": self.current_obs.clone(),  # s_{T+1} for GAE bootstrap
        }

        # Prepare metrics to return
        metrics = self.episode_metrics.copy()

        # Reset completed
        self.episode_metrics = {}

        ret_storage["metrics"] = metrics

        return ret_storage

    def get_metrics(self):
        """Return gathered metrics."""
        # TODO: Implement proper metric aggregation
        return {}

    def close(self):
        if hasattr(self, "env"):
            self.env.close()
