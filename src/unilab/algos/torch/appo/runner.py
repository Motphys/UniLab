"""APPO Runner — Asynchronous PPO with native multiprocessing.

Pipeline:
  1. Collector subprocess collects on-policy rollouts → SharedOnPolicyStorage
  2. Learner reads rollouts, computes V-trace corrected updates
  3. Weights synced back to collector via SharedWeightSync
"""

import multiprocessing as mp
import os
import sys
import time
from collections import deque
from copy import deepcopy

import torch
from rsl_rl.utils import resolve_callable

from unilab.algos.torch.appo.learner import APPOLearner
from unilab.algos.torch.appo.worker import appo_collector_fn
from unilab.ipc import AsyncRunner, SharedOnPolicyStorage, SharedWeightSync
from unilab.utils.offpolicy_logger import OffPolicyLogger
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4, is_rsl_rl_v5


class APPORunner(AsyncRunner):
    """APPO async runner using shared memory."""

    def __init__(
        self,
        env_name: str,
        env_cfg_overrides: dict,
        rl_cfg: dict,
        device: str | None = None,
        collector_device: str | None = None,
        num_envs: int = 1024,
        steps_per_env: int = 24,
        num_workers: int = 1,  # kept for API compat, but only 1 collector used
        replay_queue_size: int = 3,
    ):
        super().__init__(
            env_name=env_name,
            env_cfg_overrides=env_cfg_overrides,
            rl_cfg=rl_cfg,
            device=device,
            collector_device=collector_device,
            num_envs=num_envs,
        )

        self.steps_per_env = steps_per_env
        self.replay_queue_size = replay_queue_size

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
            self.rl_cfg["obs_groups"] = {"actor": {"policy": self.obs_dim}}
        else:
            actor_group = self.rl_cfg["obs_groups"].get(
                "actor", self.rl_cfg["obs_groups"].get("policy", {})
            )
            if isinstance(actor_group, dict) and "policy" in actor_group:
                actor_group["policy"] = self.obs_dim

    def _detect_dims(self):
        """Create a tiny env to read obs/action dims, then close it."""
        from unilab.base import registry
        from unilab.utils.algo_utils import ensure_registries

        ensure_registries()

        env = registry.make(self.env_name, num_envs=1, sim_backend="mujoco")
        obs_dim = sum(env.obs_groups_spec.values())
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]
        env.close()

        return obs_dim, action_dim

    def _build_learner(self):
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v5():
            pass  # appo_config is already v5-compatible (actor/critic format)
        elif is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        import torch
        from tensordict import TensorDict

        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        td_example = TensorDict({"policy": obs_example}, batch_size=self.num_envs)

        # Build actor (stochastic MLPModel — distribution_cfg carries GaussianDistribution)
        # deepcopy so MLPModel.__init__'s distribution_cfg.pop("class_name") doesn't
        # mutate the shared rl_cfg that gets sent to the collector subprocess.
        actor_cfg = deepcopy(cfg.get("actor", {}))
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        actor = actor_cls(td_example, cfg["obs_groups"], "actor", self.action_dim, **actor_cfg)

        # Build critic (deterministic MLPModel, no distribution)
        critic_cfg = deepcopy(cfg.get("critic", cfg.get("actor", {})))
        critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)  # critic is deterministic
        critic = critic_cls(td_example, cfg["obs_groups"], "actor", 1, **critic_cfg)

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
    ):
        os.makedirs(log_dir, exist_ok=True)

        learner = self._build_learner()

        # Create shared storage (4-slot ring buffer for IPC; replay queue lives in learner)
        shared_storage = SharedOnPolicyStorage(
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            num_slots=4,
            create=True,
        )
        self._shared_resources.append(shared_storage)

        # Create weight sync
        weight_sync = SharedWeightSync.from_state_dict(learner.actor.state_dict(), create=True)
        self._shared_resources.append(weight_sync)

        weight_param_shapes = {name: p.shape for name, p in learner.actor.state_dict().items()}

        metrics_queue: mp.Queue = mp.get_context("spawn").Queue(maxsize=100)

        # Start collector
        collector_kwargs = {
            "env_name": self.env_name,
            "rl_cfg": self.rl_cfg,
            "num_envs": self.num_envs,
            "steps_per_env": self.steps_per_env,
            "shm_storage_name": shared_storage.name,
            "sync_primitives": (
                shared_storage._write_ptr,
                shared_storage._read_ptr,
            ),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "weight_sync_name": weight_sync.name,
            "weight_param_shapes": weight_param_shapes,
            "metrics_queue": metrics_queue,
            "collector_device": self.collector_device,
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
            f"(replay_queue={self.replay_queue_size}, "
            f"epochs={learner.num_learning_epochs})"
        )

        reward_history: deque = deque(maxlen=200)
        latest_reward_components: dict = {}

        # Replay queue: stores the last replay_queue_size preprocessed rollouts in
        # learner-process memory.  Each entry is a dict of GPU tensors with shape
        # [T, N, *] (time-major) ready for process_batch / update.
        replay_queue: deque[dict] = deque(maxlen=self.replay_queue_size)

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()

            # Drain collector metrics while waiting for next rollout
            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            wait_start = time.time()

            data_ready = shared_storage.wait_for_data(timeout=60.0)
            if not data_ready:
                logger.log_status(
                    f"[yellow]Warning: Timeout waiting for data at iteration {iteration}[/]"
                )
                continue

            available_on_arrive = shared_storage.available()
            wait_time = time.time() - wait_start

            # Drain ALL available slots into the replay queue in one pass.
            # This keeps the GPU busy: if the collector produced 3 rollouts while
            # the learner was training, we consume all 3 immediately rather than
            # processing them one-per-iteration.
            num_new = shared_storage.available()
            for _ in range(num_new):
                raw = shared_storage.read_torch(self.device)
                shared_storage.advance_read()

                # Preprocess: storage is [N, T, *] (env-major); learner expects [T, N, *]
                rollout: dict = {}
                for k, v in raw.items():
                    if k != "last_obs" and v.ndim >= 2:
                        rollout[k] = v.transpose(0, 1)
                    else:
                        rollout[k] = v
                if "obs" in rollout:
                    rollout["observations"] = rollout.pop("obs")
                if "log_probs" in rollout:
                    rollout["actions_log_prob"] = rollout.pop("log_probs")
                replay_queue.append(rollout)

            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            collect_time = time.time() - iter_start

            # Concatenate all rollouts in the replay queue along the env dimension.
            # Each rollout keeps its own behavior_log_probs so V-trace IS ratios are
            # computed independently per env-column — correct for any staleness level.
            combined: dict = {
                k: torch.cat([r[k] for r in replay_queue], dim=0 if k == "last_obs" else 1)
                for k in replay_queue[0]
            }

            train_start = time.time()
            learner.process_batch(combined)
            metrics = learner.update(combined)
            weight_sync.write_weights(learner.actor.state_dict())
            train_time = time.time() - train_start

            metrics["replay_queue_len"] = float(len(replay_queue))
            metrics["available_on_arrive"] = float(available_on_arrive)

            logger.update_replay_queue(len(replay_queue), self.replay_queue_size)

            mean_reward = (
                sum(list(reward_history)[-50:]) / max(len(list(reward_history)[-50:]), 1)
                if reward_history
                else 0.0
            )

            logger.log_step(
                iteration=iteration,
                metrics=metrics,
                reward=mean_reward,
                reward_components=latest_reward_components,
                collect_time=collect_time,
                train_time=train_time,
                wait_time=wait_time,
            )

            if save_interval > 0 and iteration % save_interval == 0:
                ckpt_path = os.path.join(log_dir, f"model_{iteration}.pt")
                torch.save(learner.get_state_dict(), ckpt_path)
                logger.log_save(ckpt_path)

        ckpt_path = os.path.join(log_dir, f"model_{max_iterations}.pt")
        torch.save(learner.get_state_dict(), ckpt_path)
        logger.log_save(ckpt_path)
        logger.finish()

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
