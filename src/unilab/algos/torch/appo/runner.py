"""APPO Runner — Asynchronous PPO with native multiprocessing.

Pipeline:
  1. Collector subprocess publishes rollout payloads → RolloutRingBuffer
  2. Learner reads rollouts, computes V-trace corrected updates
  3. Weights synced back to collector via SharedWeightSync
"""

import multiprocessing as mp
import os
import sys
import time
from collections import deque
from copy import deepcopy
from typing import Any

import torch
from rsl_rl.utils import resolve_callable

from unilab.algos.torch.appo.learner import APPOLearner
from unilab.algos.torch.appo.staging import RolloutStagingPool
from unilab.algos.torch.appo.worker import appo_collector_fn
from unilab.ipc import AsyncRunner, RolloutRingBuffer, SharedWeightSync
from unilab.logging import OffPolicyLogger
from unilab.training.seed import apply_training_seed, derive_worker_seed


class APPORunner(AsyncRunner):
    """APPO async runner using shared memory."""

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        device: str | None = None,
        collector_device: str | None = None,
        sim_backend: str = "mujoco",
        num_envs: int = 1024,
        steps_per_env: int = 24,
        num_workers: int = 1,  # kept for API compat, but only 1 collector used
        replay_queue_size: int = 3,
        seed: int | None = None,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides=env_cfg_overrides,
            rl_cfg=rl_cfg,
            device=device,
            collector_device=collector_device,
            sim_backend=sim_backend,
            num_envs=num_envs,
        )

        self.steps_per_env = steps_per_env
        self.replay_queue_size = replay_queue_size
        self.staging_pool_size = replay_queue_size
        self.seed = seed
        if self.staging_pool_size < 1:
            raise ValueError("APPO staging pool size must be >= 1")

        # Resolve dims
        self._resolve_dims()

    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dims(self):
        self.obs_dim, self.action_dim = self._detect_dims()

        # Update rl_cfg so internal RSL-RL networks get correct observation dimension
        if "obs_groups" not in self.rl_cfg:
            self.rl_cfg["obs_groups"] = {
                "actor": {"policy": self.obs_dim},
                "critic": {"policy": self.critic_input_dim},
            }
        else:
            actor_group = self.rl_cfg["obs_groups"].get(
                "actor", self.rl_cfg["obs_groups"].get("policy", {})
            )
            if isinstance(actor_group, dict) and "policy" in actor_group:
                actor_group["policy"] = self.obs_dim
            critic_group = self.rl_cfg["obs_groups"].get("critic")
            if critic_group is None:
                self.rl_cfg["obs_groups"]["critic"] = {"policy": self.critic_input_dim}
            elif isinstance(critic_group, dict) and "policy" in critic_group:
                critic_group["policy"] = self.critic_input_dim

    def _detect_dims(self):
        """Create a tiny env to read obs/action dims, then close it."""
        from unilab.base import registry
        from unilab.base.observations import get_critic_base_dim, get_obs_dims
        from unilab.base.registry import ensure_registries

        ensure_registries()

        apply_training_seed(self.seed, torch_runtime=True, cuda=True)
        env = registry.make(
            self.env_name,
            num_envs=1,
            sim_backend=self.sim_backend,
            env_cfg_override=self.env_cfg_overrides if self.env_cfg_overrides else None,
        )
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        self.critic_dim = critic_dim
        self.critic_input_dim = get_critic_base_dim(env.obs_groups_spec)
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]
        env.close()

        return obs_dim, action_dim

    def _build_learner(self):
        cfg = dict(self.rl_cfg)

        import torch
        from tensordict import TensorDict

        apply_training_seed(self.seed, torch_runtime=True, cuda=True)
        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        critic_obs_dim = self.critic_input_dim
        critic_obs_example = torch.zeros((self.num_envs, critic_obs_dim), device=self.device)
        critic_td_example = TensorDict({"policy": critic_obs_example}, batch_size=self.num_envs)

        # Build actor (stochastic MLPModel — distribution_cfg carries GaussianDistribution)
        # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
        # mutate the shared rl_cfg that gets sent to the collector subprocess.
        actor_cfg = deepcopy(cfg.get("actor", {}))
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        actor = actor_cls(td_example, cfg["obs_groups"], "actor", self.action_dim, **actor_cfg)

        # Build critic (deterministic MLPModel, no distribution).
        critic_cfg: dict[str, Any] = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
        critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)  # critic is deterministic
        critic = critic_cls(critic_td_example, cfg["obs_groups"], "critic", 1, **critic_cfg)

        # Extract algorithm hyperparams from rl_cfg["algorithm"] (or top-level)
        algo_cfg = cfg.get("algorithm", cfg)
        learner = APPOLearner(
            actor=actor,
            critic=critic,
            device=self.device,
            num_learning_epochs=algo_cfg.get("num_learning_epochs", 5),
            num_mini_batches=algo_cfg.get("num_mini_batches", 4),
            clip_param=algo_cfg.get("clip_param", 0.2),
            gamma=algo_cfg.get("gamma", 0.99),
            lam=algo_cfg.get("lam", 0.95),
            value_loss_coef=algo_cfg.get("value_loss_coef", 1.0),
            entropy_coef=algo_cfg.get("entropy_coef", 0.01),
            learning_rate=algo_cfg.get("learning_rate", 1e-3),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            use_clipped_value_loss=algo_cfg.get("use_clipped_value_loss", True),
            schedule=algo_cfg.get("schedule", "fixed"),
            desired_kl=algo_cfg.get("desired_kl", 0.01),
            optimizer=algo_cfg.get("optimizer", "adam"),
            tau=algo_cfg.get("tau", 1.0),
            target_update_freq=algo_cfg.get("target_update_freq", 1),
            vtrace_clip_rho=algo_cfg.get("vtrace_clip_rho", 1.0),
            vtrace_clip_c=algo_cfg.get("vtrace_clip_c", 1.0),
        )
        return learner

    def _collector_fn(self, stop_event, **kwargs):
        appo_collector_fn(stop_event=stop_event, **kwargs)

    def learn(
        self,
        max_iterations: int = 1500,
        save_interval: int = 50,
        log_dir: str = "logs",
        logger_type: str = "tensorboard",
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        train_start_wall = time.time()
        best_mean_reward = float("-inf")
        last_mean_reward = 0.0
        ckpt_path: str | None = None
        iteration = 0

        learner = self._build_learner()

        # Create shared rollout IPC ring buffer; learner-side tensor lifetime is
        # owned by the bounded staging pool below.
        rollout_ring_buffer = RolloutRingBuffer(
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            critic_dim=self.critic_dim,
            num_slots=4,
            create=True,
        )
        self._shared_resources.append(rollout_ring_buffer)

        # Create weight sync for collector-side actor and critic bootstrap values.
        actor_weight_sync = SharedWeightSync.from_state_dict(
            learner.actor.state_dict(), create=True
        )
        critic_weight_sync = SharedWeightSync.from_state_dict(
            learner.critic.state_dict(), create=True
        )
        self._shared_resources.extend([actor_weight_sync, critic_weight_sync])

        actor_weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }
        critic_weight_param_shapes = {
            name: p.shape for name, p in learner.critic.state_dict().items()
        }

        metrics_queue: mp.Queue = mp.get_context("spawn").Queue(maxsize=100)

        # Start collector
        collector_kwargs = {
            "env_name": self.env_name,
            "rl_cfg": self.rl_cfg,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "shm_rollout_ring_buffer_name": rollout_ring_buffer.name,
            "sync_primitives": (
                rollout_ring_buffer._write_ptr,
                rollout_ring_buffer._read_ptr,
            ),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "critic_dim": self.critic_dim,
            "actor_weight_sync_name": actor_weight_sync.name,
            "actor_weight_param_shapes": actor_weight_param_shapes,
            "critic_weight_sync_name": critic_weight_sync.name,
            "critic_weight_param_shapes": critic_weight_param_shapes,
            "metrics_queue": metrics_queue,
            "collector_device": self.collector_device,
            "sim_backend": self.sim_backend,
            "env_cfg_override": self.env_cfg_overrides if self.env_cfg_overrides else None,
            "seed": derive_worker_seed(self.seed, worker_index=0),
        }
        self._start_collector(
            target_fn=appo_collector_fn,
            kwargs={"stop_event": self._stop_event, **collector_kwargs},
        )

        env_steps_per_sync = self.steps_per_env * self.num_envs

        logger = OffPolicyLogger(
            algo_name="APPO",
            max_iterations=max_iterations,
            num_envs=self.num_envs,
            env_name=self.env_name,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            log_dir=log_dir,
            log_backend=logger_type,
        )
        logger.set_collection_sync(True, env_steps_per_sync)
        logger.start()
        logger.log_status(
            f"Waiting for first rollout... "
            f"(staging_pool={self.staging_pool_size}, "
            f"epochs={learner.num_learning_epochs})"
        )

        reward_history: deque = deque(maxlen=200)
        latest_reward_components: dict = {}

        staging_pool = RolloutStagingPool(
            capacity=self.staging_pool_size,
            num_envs=self.num_envs,
            slot_shapes=rollout_ring_buffer.slot_shapes,
            device=self.device,
        )

        for iteration in range(1, max_iterations + 1):
            # Drain collector metrics while waiting for next rollout
            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            wait_start = time.time()

            data_ready = rollout_ring_buffer.wait_for_data(timeout=60.0)
            if not data_ready:
                # Check if the collector subprocess died — fail fast instead of
                # burning through remaining iterations with 60s timeouts each.
                if not self._check_collector_alive():
                    self._drain_metrics(
                        metrics_queue, reward_history, latest_reward_components, logger
                    )
                    raise RuntimeError(
                        "APPO collector process died before producing data. "
                        "Check stderr for [APPO WORKER CRASH] messages."
                    )
                logger.log_status(
                    f"[yellow]Warning: Timeout waiting for data at iteration {iteration}[/]"
                )
                continue

            available_on_arrive = rollout_ring_buffer.available()
            wait_time = time.time() - wait_start

            # Drain ALL available slots into the staging pool in one pass.
            # This keeps the GPU busy: if the collector produced 3 rollouts while
            # the learner was training, we consume all 3 immediately rather than
            # processing them one-per-iteration.
            num_new = rollout_ring_buffer.available()
            learner_incremental_h2d_time = 0.0
            for _ in range(num_new):
                h2d_start = time.perf_counter()
                staging_pool.stage_numpy_views(rollout_ring_buffer.read_numpy_views())
                learner_incremental_h2d_time += time.perf_counter() - h2d_start
                rollout_ring_buffer.advance_read()

            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)

            combined = staging_pool.batch()

            train_start = time.time()
            learner.process_batch(combined)
            metrics = learner.update(combined)
            train_time = time.time() - train_start
            weight_sync_start = time.perf_counter()
            actor_weight_sync.write_weights(learner.actor.state_dict())
            critic_weight_sync.write_weights(learner.critic.state_dict())
            weight_sync_time = time.perf_counter() - weight_sync_start

            metrics["staging_pool_len"] = float(staging_pool.active_count)
            metrics["staging_pool_capacity"] = float(staging_pool.capacity)
            metrics["available_on_arrive"] = float(available_on_arrive)
            metrics["rollouts_read"] = float(num_new)

            logger.update_staging_pool(staging_pool.active_count, staging_pool.capacity)

            mean_reward = (
                sum(list(reward_history)[-50:]) / max(len(list(reward_history)[-50:]), 1)
                if reward_history
                else 0.0
            )
            last_mean_reward = float(mean_reward)
            best_mean_reward = max(best_mean_reward, last_mean_reward)

            logger.log_step(
                iteration=iteration,
                metrics=metrics,
                reward=mean_reward,
                reward_components=latest_reward_components,
                train_time=train_time,
                wait_time=wait_time,
                learner_incremental_h2d_time=learner_incremental_h2d_time,
                weight_sync_time=weight_sync_time,
                extra_info={
                    "throughput_steps": num_new * env_steps_per_sync,
                },
            )

            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()
        summary = {
            "status": "completed",
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": last_mean_reward if reward_history else None,
            "best_mean_reward": best_mean_reward if reward_history else None,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "training_wall_time_sec": time.time() - train_start_wall,
        }
        self.last_run_summary = summary

    def _check_collector_alive(self) -> bool:
        if self._collector_process is not None and not self._collector_process.is_alive():
            return False
        return True

    @staticmethod
    def _drain_metrics(queue, reward_history, reward_components, logger):
        """Drain all pending messages from the collector metrics queue.

        Mirrors OffPolicyRunner._drain_metrics so APPO has the same
        logger update coverage (ep_length, done rates, collector timing).
        """
        while not queue.empty():
            try:
                m = queue.get_nowait()
                if "error" in m:
                    logger.log_status(f"[red]Collector ERROR: {m['error']}[/]")
                    raise RuntimeError(f"Collector process failed: {m['error']}")

                if "mean_ep_reward" in m:
                    reward_history.append(m["mean_ep_reward"])

                if "reward_components" in m:
                    reward_components.clear()
                    reward_components.update(m["reward_components"])

                if "mean_ep_length" in m:
                    logger.update_ep_length(m["mean_ep_length"])

                if "collector_timing_ms" in m:
                    logger.update_collector_timing(m["collector_timing_ms"])

                if "timeout_rate" in m or "terminated_rate" in m:
                    logger.update_done_rates(
                        timeout_rate=float(m.get("timeout_rate", 0.0)),
                        terminated_rate=float(m.get("terminated_rate", 0.0)),
                    )

                if "total_steps" in m:
                    logger.log_collector(
                        m["total_steps"],
                        0,  # APPO uses shared memory, not a separate buffer
                        m.get("mean_ep_reward", 0.0),
                    )

            except Exception as e:
                print(f"[APPORunner] metrics drain error: {e}", file=sys.stderr)
                break
