"""HORA-owned APPO runner."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from rsl_rl.utils import resolve_callable

from unilab.algos.torch.appo.runner import APPORunner
from unilab.algos.torch.hora.appo_learner import HoraAPPOLearner
from unilab.algos.torch.hora.appo_worker import hora_appo_collector_fn
from unilab.algos.torch.hora.rsl_rl_compat import (
    convert_config_v3_to_v4,
    is_rsl_rl_v4,
    is_rsl_rl_v5,
)
from unilab.base.observations import get_critic_base_dim, get_obs_dims
from unilab.base.registry import ensure_registries
from unilab.ipc import SharedOnPolicyStorage, SharedWeightSync
from unilab.logging import OffPolicyLogger


class HoraAPPORunner(APPORunner):
    """APPO runner variant that preserves grouped HORA observations."""

    def __init__(self, *args, **kwargs):
        self.priv_info_dim = 0
        super().__init__(*args, **kwargs)

    def _resolve_dims(self):
        self.obs_dim, self.action_dim = self._detect_dims()

        obs_groups = self.rl_cfg.setdefault("obs_groups", {})
        actor_group = obs_groups.setdefault("actor", {})
        critic_group = obs_groups.setdefault("critic", {})
        if isinstance(actor_group, dict):
            actor_group["actor"] = self.obs_dim
            actor_group["priv_info"] = self.priv_info_dim
        if isinstance(critic_group, dict):
            critic_group["actor"] = self.obs_dim
            critic_group["priv_info"] = self.priv_info_dim

    def _detect_dims(self):
        from unilab.base import registry

        ensure_registries()

        env = registry.make(
            self.env_name,
            num_envs=self._detect_dim_probe_num_envs(),
            sim_backend=self.sim_backend,
            env_cfg_override=self.env_cfg_overrides if self.env_cfg_overrides else None,
        )
        obs_dim, critic_dim = get_obs_dims(env.obs_groups_spec)
        self.critic_dim = critic_dim
        self.critic_input_dim = get_critic_base_dim(env.obs_groups_spec)
        if env.state is None:
            env.init_state()
        info = env.state.info if env.state is not None else {}
        priv_info = info.get("critic_info") if isinstance(info, dict) else None
        if isinstance(priv_info, np.ndarray) and priv_info.ndim == 2:
            self.priv_info_dim = int(priv_info.shape[1])
        elif critic_dim > obs_dim:
            self.priv_info_dim = int(critic_dim - obs_dim)
        if self.priv_info_dim <= 0:
            env.close()
            raise ValueError("HORA APPO requires a positive privileged-info dimension.")
        assert env.action_space.shape is not None
        action_dim = env.action_space.shape[0]
        env.close()
        return obs_dim, action_dim

    def _detect_dim_probe_num_envs(self) -> int:
        scale_list = None
        if isinstance(self.env_cfg_overrides, dict):
            domain_rand = self.env_cfg_overrides.get("domain_rand")
            if isinstance(domain_rand, dict):
                scale_list = domain_rand.get("scale_list")
            if scale_list is None:
                scale_list = self.env_cfg_overrides.get("scale_list")
        if isinstance(scale_list, (list, tuple)):
            return max(1, len(scale_list))
        return 1

    def _build_learner(self):
        cfg = dict(self.rl_cfg)
        if is_rsl_rl_v5():
            pass
        elif is_rsl_rl_v4():
            cfg = convert_config_v3_to_v4(cfg)

        from tensordict import TensorDict

        obs_example = torch.zeros((self.num_envs, self.obs_dim), device=self.device)
        priv_info_example = torch.zeros((self.num_envs, self.priv_info_dim), device=self.device)
        td_example = TensorDict(
            {"actor": obs_example, "priv_info": priv_info_example},
            batch_size=self.num_envs,
            device=self.device,
        )

        actor_cfg = deepcopy(cfg.get("actor", {}))
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        actor_cfg.pop("num_actions", None)
        actor = actor_cls(td_example, cfg["obs_groups"], "actor", self.action_dim, **actor_cfg)

        critic_cfg: dict[str, Any] = deepcopy(cfg.get("critic") or cfg.get("actor") or {})
        critic_cls = resolve_callable(critic_cfg.pop("class_name", "rsl_rl.models.MLPModel"))
        critic_cfg.pop("num_actions", None)
        critic_cfg.pop("distribution_cfg", None)
        critic = critic_cls(td_example, cfg["obs_groups"], "critic", 1, **critic_cfg)

        algo_cfg = cfg.get("algorithm", cfg)
        return HoraAPPOLearner(
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

    def _collector_fn(self, stop_event, **kwargs):
        hora_appo_collector_fn(stop_event=stop_event, **kwargs)

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

        shared_storage = SharedOnPolicyStorage(
            num_envs=self.num_envs,
            num_steps=self.steps_per_env,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            critic_dim=self.critic_dim,
            priv_info_dim=self.priv_info_dim,
            num_slots=4,
            create=True,
        )
        self._shared_resources.append(shared_storage)

        actor_weight_sync = SharedWeightSync.from_state_dict(
            learner.actor.state_dict(), create=True
        )
        critic_weight_sync = SharedWeightSync.from_state_dict(
            learner.critic.state_dict(),
            create=True,
        )
        self._shared_resources.extend([actor_weight_sync, critic_weight_sync])

        actor_weight_param_shapes = {
            name: p.shape for name, p in learner.actor.state_dict().items()
        }
        critic_weight_param_shapes = {
            name: p.shape for name, p in learner.critic.state_dict().items()
        }

        metrics_queue: mp.Queue = mp.get_context("spawn").Queue(maxsize=100)
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
            "critic_dim": self.critic_dim,
            "priv_info_dim": self.priv_info_dim,
            "actor_weight_sync_name": actor_weight_sync.name,
            "actor_weight_param_shapes": actor_weight_param_shapes,
            "critic_weight_sync_name": critic_weight_sync.name,
            "critic_weight_param_shapes": critic_weight_param_shapes,
            "metrics_queue": metrics_queue,
            "collector_device": self.collector_device,
            "sim_backend": self.sim_backend,
            "env_cfg_override": self.env_cfg_overrides if self.env_cfg_overrides else None,
        }
        self._start_collector(
            target_fn=hora_appo_collector_fn,
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
        replay_queue: deque[dict] = deque(maxlen=self.replay_queue_size)

        for iteration in range(1, max_iterations + 1):
            iter_start = time.time()
            self._drain_metrics(metrics_queue, reward_history, latest_reward_components, logger)
            wait_start = time.time()

            data_ready = shared_storage.wait_for_data(timeout=60.0)
            if not data_ready:
                if not self._check_collector_alive():
                    self._drain_metrics(
                        metrics_queue,
                        reward_history,
                        latest_reward_components,
                        logger,
                    )
                    raise RuntimeError(
                        "APPO collector process died before producing data. "
                        "Check stderr for [HORA APPO WORKER CRASH] messages."
                    )
                logger.log_status(
                    f"[yellow]Warning: Timeout waiting for data at iteration {iteration}[/]"
                )
                continue

            available_on_arrive = shared_storage.available()
            wait_time = time.time() - wait_start

            num_new = shared_storage.available()
            for _ in range(num_new):
                raw = shared_storage.read_torch(self.device)
                shared_storage.advance_read()

                rollout: dict = {}
                for k, v in raw.items():
                    if k not in ("last_obs", "last_critic", "last_priv_info") and v.ndim >= 2:
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

            combined: dict = {}
            for k in replay_queue[0]:
                if k in ("last_obs", "last_critic", "last_priv_info"):
                    combined[k] = torch.cat([r[k] for r in replay_queue], dim=0)
                else:
                    combined[k] = torch.cat([r[k] for r in replay_queue], dim=1)

            train_start = time.time()
            learner.process_batch(combined)
            metrics = learner.update(combined)
            actor_weight_sync.write_weights(learner.actor.state_dict())
            critic_weight_sync.write_weights(learner.critic.state_dict())
            train_time = time.time() - train_start

            metrics["replay_queue_len"] = float(len(replay_queue))
            metrics["available_on_arrive"] = float(available_on_arrive)
            logger.update_replay_queue(len(replay_queue), self.replay_queue_size)

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
        self.last_run_summary = {
            "status": "completed",
            "completed_iterations": iteration,
            "total_env_steps": int(logger._total_steps),
            "final_mean_reward": last_mean_reward if reward_history else None,
            "best_mean_reward": best_mean_reward if reward_history else None,
            "mean_episode_length": float(logger._mean_ep_length),
            "last_checkpoint": ckpt_path,
            "training_wall_time_sec": time.time() - train_start_wall,
        }
