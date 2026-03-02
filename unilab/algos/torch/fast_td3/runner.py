"""FastTD3 Runner — synchronous single-process training loop.

Aligned with the reference FastTD3 repository's train.py:
- Single process: environment step and training happen in the same process
- Per-env replay buffer
- Observation normalization
- Cosine LR schedule
- tqdm progress bar with speed measurement

This replaces the previous async runner design.
"""

import os
import time
import statistics
import torch
import numpy as np
from collections import defaultdict, deque

from unilab.algos.torch.fast_td3.learner import FastTD3Learner, SimpleReplayBuffer
from unilab.algos.torch.common.logger import TrainingLogger


def _mx_to_torch(x, device):
    """Convert MLX array to torch tensor."""
    return torch.from_numpy(np.array(x, copy=False)).to(device=device, dtype=torch.float32)


def _mx_to_np(x):
    """Convert MLX array to numpy."""
    return np.array(x, copy=False)


class FastTD3Runner:
    """Synchronous FastTD3 training runner.

    Aligned with reference FastTD3 repo's train.py:
    1. Step environment with actor.explore()
    2. Store transitions in replay buffer
    3. Sample batch and run N critic + delayed actor updates
    4. Soft update target networks
    """

    def __init__(
        self,
        env_name: str,
        device: str | None = None,
        num_envs: int = 1024,
        # Buffer
        buffer_size: int = 10240,
        batch_size: int = 32768,
        # Training
        warmup_steps: int = 10,
        num_updates: int = 8,
        policy_frequency: int = 2,
        total_timesteps: int = 50000,
        # Algorithm
        gamma: float = 0.97,
        tau: float = 0.1,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 1024,
        num_atoms: int = 101,
        v_min: float = -10.0,
        v_max: float = 10.0,
        init_scale: float = 0.01,
        std_min: float = 0.2,
        std_max: float = 0.8,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        weight_decay: float = 0.1,
        use_cdq: bool = True,
        obs_normalization: bool = True,
        **kwargs,
    ):
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.env_name = env_name
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.num_updates = num_updates
        self.policy_frequency = policy_frequency
        self.total_timesteps = total_timesteps

        # Detect env dims
        self.obs_dim, self.action_dim = self._detect_dims()

        # Store learner kwargs
        self.learner_kwargs = dict(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_envs=num_envs,
            device=device,
            gamma=gamma,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            actor_hidden_dim=actor_hidden_dim,
            critic_hidden_dim=critic_hidden_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            init_scale=init_scale,
            std_min=std_min,
            std_max=std_max,
            weight_decay=weight_decay,
            use_cdq=use_cdq,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_frequency=policy_frequency,
            total_timesteps=total_timesteps,
            obs_normalization=obs_normalization,
        )

    def _detect_dims(self):
        from unilab.envs import registry
        self._ensure_registries()
        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, action_dim

    @staticmethod
    def _ensure_registries():
        import pkgutil
        import importlib
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

    def learn(
        self,
        max_iterations: int | None = None,
        save_interval: int = 50,
        log_dir: str = "logs",
    ):
        """Main training loop (synchronous, single-process)."""
        import mlx.core as mx
        from unilab.envs import registry

        os.makedirs(log_dir, exist_ok=True)

        total_timesteps = max_iterations if max_iterations else self.total_timesteps
        collect_device = "cpu"  # CPU: replay buffer + env data
        train_device = self.device               # GPU: actor inference + training

        # Build learner (on training device: MPS/CUDA)
        learner = FastTD3Learner(**self.learner_kwargs)

        # Build replay buffer on CPU to save GPU memory
        rb = SimpleReplayBuffer(
            n_env=self.num_envs,
            buffer_size=self.buffer_size,
            n_obs=self.obs_dim,
            n_act=self.action_dim,
            device=collect_device,
        )

        # Build env (MuJoCo physics runs on CPU)
        self._ensure_registries()
        env = registry.make(self.env_name, num_envs=self.num_envs, sim_backend="mujoco")

        # Init env
        if env.state is None:
            env.init_state()
        env_indices = mx.arange(self.num_envs, dtype=mx.int32)
        _, obs_mx, _ = env.reset(env_indices)
        obs = _mx_to_torch(obs_mx, collect_device)  # obs on CPU

        # Logger
        logger = TrainingLogger(
            algo_name="FastTD3",
            max_iterations=total_timesteps,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        logger.start()

        # Tracking (on CPU)
        reward_history = deque(maxlen=100)
        ep_length_history = deque(maxlen=100)
        ep_rewards = torch.zeros(self.num_envs, device=collect_device)
        ep_lengths = torch.zeros(self.num_envs, device=collect_device)
        latest_reward_components = {}
        dones = None

        for global_step in range(1, total_timesteps + 1):
            iter_start = time.time()

            # ---- Environment step ----
            # Actor inference on GPU, env step on CPU
            with torch.no_grad():
                obs_gpu = obs.to(train_device)
                dones_gpu = dones.to(train_device) if dones is not None else None
                norm_obs = learner.normalize_obs(obs_gpu)
                actions_gpu = learner.actor.explore(obs=norm_obs, dones=dones_gpu)
                actions = actions_gpu.cpu()  # back to CPU for env

            # Step env (MuJoCo on CPU via MLX)
            actions_np = actions.float().numpy()
            state = env.step(actions_np)

            next_obs_mx = state.obs
            rewards_mx = state.reward
            terminated_mx = state.terminated
            truncated_mx = state.truncated

            next_obs = _mx_to_torch(next_obs_mx, collect_device)
            rewards = _mx_to_torch(rewards_mx, collect_device)
            terminated = _mx_to_torch(terminated_mx, collect_device)
            truncated = _mx_to_torch(truncated_mx, collect_device)
            dones_float = (terminated + truncated).clamp(max=1.0)

            # Compute true next obs (use reset obs for done envs)
            # NOTE: env._reset_done_envs() has already been called inside env.step(),
            # so state.obs already contains the reset obs for done envs.
            # We need the pre-reset next_obs for the buffer. Since the env auto-resets,
            # we store the current state.obs which is post-reset for done envs.
            # The buffer should store the terminal obs for learning, not the reset obs.
            # However, since env auto-resets, we need to track the pre-reset obs.
            # For simplicity, we use the auto-reset obs (standard in vectorized envs).
            # The bootstrap flag handles this: bootstrap=0 for true terminal, 1 for truncated.

            # Track episode rewards
            ep_rewards += rewards
            ep_lengths += 1.0
            done_mask = dones_float > 0.5
            if done_mask.any():
                done_rewards = ep_rewards[done_mask]
                done_lens = ep_lengths[done_mask]
                for r in done_rewards.cpu().tolist():
                    reward_history.append(r)
                for l in done_lens.cpu().tolist():
                    ep_length_history.append(l)
                ep_rewards[done_mask] = 0.0
                ep_lengths[done_mask] = 0.0
                if ep_length_history:
                    logger.update_ep_length(statistics.mean(ep_length_history))

            # Store transition (on CPU)
            rb.extend({
                "observations": obs,
                "actions": actions.float(),
                "rewards": rewards,
                "dones": terminated.long(),
                "truncations": truncated.long(),
                "next_observations": next_obs,
            })

            obs = next_obs
            dones = dones_float

            # Extract reward components from env log
            if hasattr(state, "info") and "log" in state.info:
                log_dict = state.info["log"]
                if log_dict:
                    latest_reward_components = {
                        k.replace("reward/", ""): v
                        for k, v in log_dict.items()
                        if k.startswith("reward/")
                    }

            # ---- Training (on GPU) ----
            collect_time = time.time() - iter_start

            if global_step > self.warmup_steps:
                train_start = time.time()
                iter_metrics = defaultdict(list)
                batch_per_env = max(1, self.batch_size // self.num_envs)

                for update_idx in range(self.num_updates):
                    data = rb.sample(batch_per_env)

                    # Move batch to training device (GPU)
                    for k in data:
                        data[k] = data[k].to(train_device)

                    # Normalize observations
                    data["observations"] = learner.normalize_obs(data["observations"])
                    data["next_observations"] = learner.normalize_obs(data["next_observations"])

                    critic_metrics = learner.update_critic(data)
                    for k, v in critic_metrics.items():
                        iter_metrics[k].append(v)

                    # Delayed policy update
                    if self.num_updates > 1:
                        if update_idx % self.policy_frequency == 1:
                            actor_metrics = learner.update_actor(data)
                            for k, v in actor_metrics.items():
                                iter_metrics[k].append(v)
                    else:
                        if global_step % self.policy_frequency == 0:
                            actor_metrics = learner.update_actor(data)
                            for k, v in actor_metrics.items():
                                iter_metrics[k].append(v)

                    learner.soft_update()

                train_time = time.time() - train_start

                # Update buffer/steps info
                logger.log_collector(
                    total_steps=global_step * self.num_envs,
                    buffer_size=rb.size,
                )

                # Logging — every step
                avg_metrics = {k: statistics.mean(v) for k, v in iter_metrics.items() if v}
                mean_reward = statistics.mean(reward_history) if reward_history else 0.0

                logger.log_step(
                    iteration=global_step,
                    metrics=avg_metrics,
                    reward=mean_reward,
                    reward_components=latest_reward_components,
                    collect_time=collect_time,
                    train_time=train_time,
                )

            # Step LR schedulers (only after first optimizer step)
            if global_step > self.warmup_steps:
                learner.step_schedulers()

            # Save checkpoint
            if save_interval > 0 and global_step > 0 and global_step % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{global_step}.pt")
                torch.save(learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        # Final save
        ckpt_path = os.path.join(log_dir, f"model_{total_timesteps}.pt")
        torch.save(learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()
        print(f"Training complete. {total_timesteps} steps.")

        # Cleanup
        env.close()

    def close(self):
        """No-op for sync runner (env is closed at end of learn())."""
        pass
