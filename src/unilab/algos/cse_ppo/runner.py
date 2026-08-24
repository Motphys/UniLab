# SPDX-License-Identifier: BSD-3-Clause
"""On-policy runner for CSE-PPO."""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Callable, cast

import torch

from .actor_critic import CSEActorCritic
from .algorithm import CSEPPO


class _CSELogger:
    def __init__(self) -> None:
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.tot_timesteps = 0


class CSEOnPolicyRunner:
    """Train and serve CSE-PPO against the ManagerBased VecEnv wrapper."""

    def __init__(
        self, env: Any, train_cfg: dict[str, Any], log_dir: str | None = None, device: str = "cpu"
    ) -> None:
        self.env, self.device, self.log_dir = env, device, log_dir
        self.current_learning_iteration = 0
        self.logger = _CSELogger()
        cfg = dict(train_cfg)
        one_step = int(cfg["num_one_step_obs"])
        actor_history = int(cfg.get("num_actor_history", 1))
        num_actor_obs = one_step * actor_history
        num_critic_obs = int(getattr(env, "num_privileged_obs", None) or env.num_obs)
        policy_cfg, estimator_cfg, algorithm_cfg = (
            dict(cfg.get(name) or {}) for name in ("policy", "estimator", "algorithm")
        )
        self.actor_critic = CSEActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=one_step,
            num_actions=int(env.num_actions),
            actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [512, 256, 128]),
            critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            activation=str(policy_cfg.get("activation", "elu")),
            init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
            estimator=estimator_cfg,
        ).to(device)
        self.alg = CSEPPO(self.actor_critic, device=device, **algorithm_cfg)
        self.num_steps_per_env = int(cfg.get("num_steps_per_env", 24))
        self.save_interval = int(cfg.get("save_interval", 100))
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [int(env.num_actions)],
        )
        self._ep_returns = torch.zeros(env.num_envs, device=device)
        self._ep_lengths = torch.zeros(env.num_envs, device=device)
        self._writer: Any = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                pass

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = True) -> None:
        obs_td, _ = self.env.reset()
        obs, critic_obs = (
            obs_td["actor"].to(self.device),
            obs_td.get("critic", obs_td["actor"]).to(self.device),
        )
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.alg.train_mode()
        start = self.current_learning_iteration
        for iteration in range(start, start + int(num_learning_iterations)):
            infos: dict[str, Any] = {}
            timing_accum = {
                "step_core_ms": 0.0,
                "backend_physics_ms": 0.0,
                "update_state_ms": 0.0,
                "apply_action_ms": 0.0,
                "reset_done_ms": 0.0,
                "env_step_total_ms": 0.0,
            }
            collect_start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs_td, rewards, dones, infos = self.env.step(actions)
                    next_obs, next_critic_obs = (
                        obs_td["actor"].to(self.device),
                        obs_td.get("critic", obs_td["actor"]).to(self.device),
                    )
                    self._ep_returns += rewards.to(self.device)
                    self._ep_lengths += 1
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    if done_ids.numel():
                        self.logger.rewbuffer.extend(self._ep_returns[done_ids].tolist())
                        self.logger.lenbuffer.extend(self._ep_lengths[done_ids].tolist())
                        self._ep_returns[done_ids] = 0
                        self._ep_lengths[done_ids] = 0
                    self.alg.process_env_step(obs_td, rewards, dones, infos)
                    obs, critic_obs = next_obs, next_critic_obs
                    step_timing = infos.get("timing")
                    if step_timing:
                        for key in timing_accum:
                            timing_accum[key] += float(step_timing.get(key, 0.0))
                self.alg.compute_returns(critic_obs)
            collection_time = time.time() - collect_start
            learn_start = time.time()
            value_loss, surrogate_loss, estimation_loss = self.alg.update()
            self.current_learning_iteration = iteration + 1
            self.logger.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            learn_time = time.time() - learn_start
            elapsed = collection_time + learn_time
            num_steps = self.num_steps_per_env * self.env.num_envs
            stats = {
                "value_loss": value_loss,
                "surrogate_loss": surrogate_loss,
                "estimation_loss": estimation_loss,
                "learning_rate": self.alg.learning_rate,
                "mean_noise_std": float(self.actor_critic.std.mean().detach()),
                "collection_time": collection_time,
                "learn_time": learn_time,
                "fps": int(num_steps / max(elapsed, 1e-9)),
                "wf_physics": timing_accum["backend_physics_ms"],
                "wf_slow_path": max(
                    0.0, timing_accum["step_core_ms"] - timing_accum["backend_physics_ms"]
                ),
                "wf_update_state": timing_accum["update_state_ms"],
                "wf_apply_action": timing_accum["apply_action_ms"],
                "wf_reset_done": timing_accum["reset_done_ms"],
                "wf_env_misc": max(
                    0.0,
                    timing_accum["env_step_total_ms"]
                    - timing_accum["step_core_ms"]
                    - timing_accum["update_state_ms"]
                    - timing_accum["apply_action_ms"]
                    - timing_accum["reset_done_ms"],
                ),
                "wf_learn": learn_time * 1000.0,
            }
            self._print_iter(
                self.current_learning_iteration,
                start + int(num_learning_iterations),
                stats,
                elapsed,
                infos,
            )
            if self._writer is not None:
                step = self.current_learning_iteration
                for key, value in (
                    ("Loss/value", value_loss),
                    ("Loss/surrogate", surrogate_loss),
                    ("Loss/estimation", estimation_loss),
                    ("Loss/learning_rate", self.alg.learning_rate),
                    ("Policy/mean_noise_std", float(self.actor_critic.std.mean())),
                    ("Perf/learning_time", time.time() - learn_start),
                ):
                    self._writer.add_scalar(key, value, step)
                for key, value in (infos.get("log") or {}).items():
                    self._writer.add_scalar(key, value, step)
            if (
                self.log_dir is not None
                and self.current_learning_iteration % self.save_interval == 0
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iteration": self.current_learning_iteration,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(checkpoint["actor_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "iteration" in checkpoint:
            self.current_learning_iteration = int(checkpoint["iteration"])

    def get_inference_policy(self, device: str | None = None) -> Callable[..., Any]:
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return cast(Callable[..., Any], self.actor_critic.act_inference)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        original_device = next(self.actor_critic.parameters()).device
        ac = self.actor_critic.cpu().eval()
        one_step = ac.num_one_step_obs

        class PolicyExport(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.estimator, self.actor_mlp = ac.estimator, ac.actor

            def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
                latent = self.estimator.get_latent(obs_history)
                return self.actor_mlp(torch.cat((obs_history[:, -one_step:], latent), dim=-1))

        os.makedirs(path, exist_ok=True)
        with torch.inference_mode():
            traced = torch.jit.trace(PolicyExport(), (torch.zeros(1, ac.num_actor_obs),))
        traced.save(os.path.join(path, filename))
        self.actor_critic.to(original_device)

    def _print_iter(
        self,
        it: int,
        tot: int,
        stats: dict[str, float],
        elapsed: float,
        infos: dict[str, Any],
    ) -> None:
        """Print the established CSE-PPO iteration summary."""
        sep = "-" * 80
        mean_rew = (
            sum(self.logger.rewbuffer) / len(self.logger.rewbuffer)
            if self.logger.rewbuffer
            else 0.0
        )
        mean_len = (
            sum(self.logger.lenbuffer) / len(self.logger.lenbuffer)
            if self.logger.lenbuffer
            else 0.0
        )
        time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        eta = elapsed / it * (tot - it) if it > 0 else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(sep)
        print(f"{'Iteration':>40}: {it}/{tot}")
        print(f"{'Computation (fps)':>40}: {int(stats['fps'])} steps/s")
        print(f"{'Mean value loss':>40}: {stats['value_loss']:.4f}")
        print(f"{'Mean surrogate loss':>40}: {stats['surrogate_loss']:.4f}")
        print(f"{'Mean estimation loss':>40}: {stats['estimation_loss']:.4f}")
        print(f"{'Learning rate':>40}: {stats['learning_rate']:.2e}")
        print(f"{'Mean action noise std':>40}: {stats['mean_noise_std']:.3f}")
        if mean_rew:
            print(f"{'Mean episode reward':>40}: {mean_rew:.4f}")
        if mean_len:
            print(f"{'Mean episode length':>40}: {mean_len:.1f}")
        for key, value in sorted((infos.get("log") or {}).items()):
            print(f"{key:>40}: {value:.4f}")
        print(f"{'Total timesteps':>40}: {self.logger.tot_timesteps}")
        print(f"{'Iteration time':>40}: {stats['collection_time'] + stats['learn_time']:.2f}s")
        if "wf_physics" in stats:
            total_ms = (stats["collection_time"] + stats["learn_time"]) * 1000.0
            waterfall = [
                ("physics", stats["wf_physics"]),
                ("learn", stats["wf_learn"]),
                ("update_state", stats["wf_update_state"]),
                ("apply_action", stats["wf_apply_action"]),
                ("reset_done", stats["wf_reset_done"]),
                ("env_misc(nan_guard/reset)", stats["wf_env_misc"]),
                ("slow_path", stats["wf_slow_path"]),
            ]
            print(f"{'Iter waterfall (ms/iter, %)':>40}: total={total_ms:.0f}ms")
            for name, value in sorted(waterfall, key=lambda item: -item[1]):
                print(f"{name:>40}: {value:7.1f} ms  ({100.0 * value / max(1e-9, total_ms):4.1f}%)")
        print(f"{'Time elapsed':>40}: {time_str}")
        print(f"{'ETA':>40}: {eta_str}")
        print(sep)
