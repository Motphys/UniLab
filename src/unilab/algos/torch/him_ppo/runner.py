# SPDX-License-Identifier: BSD-3-Clause
#
# HIM-PPO OnPolicy Runner for UniLab.

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Callable, cast

import torch
from tensordict import TensorDict

from unilab.algos.torch.him_ppo.actor_critic import HIMActorCritic
from unilab.algos.torch.him_ppo.algorithm import HIMPPO


class _HIMLogger:
    """Minimal logger compatible with train_rsl_rl.py summary extraction."""

    def __init__(self) -> None:
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.tot_timesteps: int = 0


class HIMOnPolicyRunner:
    """On-policy training runner for HIM-PPO.

    Provides the same external interface as rsl_rl's ``OnPolicyRunner`` so that
    ``train_him_ppo.py`` can share most of its logic with ``train_rsl_rl.py``.
    """

    def __init__(
        self,
        env: Any,
        train_cfg: dict[str, Any],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.current_learning_iteration: int = 0
        self.logger = _HIMLogger()

        # ── Parse config ────────────────────────────────────────────────────
        cfg: dict[str, Any] = dict(train_cfg)

        num_one_step_obs = int(cfg["num_one_step_obs"])
        num_actor_history = int(cfg.get("num_actor_history", 1))
        num_actor_obs = num_actor_history * num_one_step_obs
        num_critic_obs = int(getattr(env, "num_privileged_obs", None) or env.num_obs)
        num_actions = int(env.num_actions)

        policy_cfg: dict[str, Any] = dict(cfg.get("policy") or {})
        estimator_cfg: dict[str, Any] = dict(cfg.get("estimator") or {})
        algo_cfg: dict[str, Any] = dict(cfg.get("algorithm") or {})

        # ── Build actor-critic ───────────────────────────────────────────────
        self.actor_critic = HIMActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_one_step_obs=num_one_step_obs,
            num_actions=num_actions,
            actor_hidden_dims=list(policy_cfg.get("actor_hidden_dims", [512, 256, 128])),
            critic_hidden_dims=list(policy_cfg.get("critic_hidden_dims", [512, 256, 128])),
            activation=str(policy_cfg.get("activation", "elu")),
            init_noise_std=float(policy_cfg.get("init_noise_std", 1.0)),
            estimator=estimator_cfg,
        ).to(device)

        # ── Build algorithm ──────────────────────────────────────────────────
        self.alg = HIMPPO(self.actor_critic, device=device, **algo_cfg)

        self.num_steps_per_env = int(cfg.get("num_steps_per_env", 24))
        self.save_interval = int(cfg.get("save_interval", 100))

        # ── Init storage ─────────────────────────────────────────────────────
        self.alg.init_storage(
            env.num_envs,
            self.num_steps_per_env,
            [num_actor_obs],
            [num_critic_obs],
            [num_actions],
        )

        # Per-env episode tracking (independent of env wrapper's counters)
        self._ep_returns = torch.zeros(env.num_envs, device=device)
        self._ep_lengths = torch.zeros(env.num_envs, device=device)

        # Tensorboard writer (optional)
        self._writer: Any = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                pass

    # ── Public interface ─────────────────────────────────────────────────────

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = True,
    ) -> None:
        obs_td, _ = self.env.reset()
        obs = obs_td["actor"].to(self.device)
        critic_obs = obs_td.get("critic", obs).to(self.device)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        self.alg.train_mode()
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        start_time = time.time()

        for it in range(start_iter, tot_iter):
            infos: dict[str, Any] = {}
            # ── Rollout collection ───────────────────────────────────────────
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs_td, rewards, dones, infos = self.env.step(actions)

                    next_obs = obs_td["actor"].to(self.device)
                    next_critic_obs = obs_td.get("critic", next_obs).to(self.device)

                    # Track episode stats before env wrapper resets counters
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    self._ep_returns += rewards.to(self.device)
                    self._ep_lengths += 1
                    if len(done_ids) > 0:
                        for idx in done_ids:
                            self.logger.rewbuffer.append(float(self._ep_returns[idx]))
                            self.logger.lenbuffer.append(float(self._ep_lengths[idx]))
                        self._ep_returns[done_ids] = 0.0
                        self._ep_lengths[done_ids] = 0.0

                    self.alg.process_env_step(obs_td, rewards, dones, infos)
                    obs = next_obs
                    critic_obs = next_critic_obs

                self.alg.compute_returns(critic_obs)

            # ── Update ───────────────────────────────────────────────────────
            value_loss, surrogate_loss, estimation_loss, swap_loss = self.alg.update()

            self.current_learning_iteration = it + 1
            self.logger.tot_timesteps += self.num_steps_per_env * self.env.num_envs

            # ── Logging ──────────────────────────────────────────────────────
            elapsed = time.time() - start_time
            self._print_iter(
                it + 1,
                tot_iter,
                value_loss,
                surrogate_loss,
                estimation_loss,
                swap_loss,
                elapsed,
                infos,
            )
            if self._writer is not None:
                global_step = self.current_learning_iteration
                self._writer.add_scalar("train/value_loss", value_loss, global_step)
                self._writer.add_scalar("train/surrogate_loss", surrogate_loss, global_step)
                self._writer.add_scalar("train/estimation_loss", estimation_loss, global_step)
                self._writer.add_scalar("train/swap_loss", swap_loss, global_step)
                for k, v in (infos.get("log") or {}).items():
                    self._writer.add_scalar(k, v, global_step)

            # ── Checkpoint ───────────────────────────────────────────────────
            if (
                self.log_dir is not None
                and self.current_learning_iteration % self.save_interval == 0
            ):
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

        # Final checkpoint
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
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(ckpt["actor_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "iteration" in ckpt:
            self.current_learning_iteration = int(ckpt["iteration"])

    def get_inference_policy(self, device: str | None = None) -> Callable[..., Any]:
        self.actor_critic.eval()
        if device is not None:
            self.actor_critic.to(device)
        return cast(Callable[..., Any], self.actor_critic.act_inference)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx") -> None:
        """Export the end-to-end HIM-PPO policy (estimator + actor) to ONNX.

        Input:  obs_history  (1, H_a * num_one_step_obs)  — flattened actor obs history
        Output: actions      (1, num_actions)
        """

        class _PolicyExport(torch.nn.Module):
            def __init__(self, ac: Any) -> None:
                super().__init__()
                self.ac = ac

            def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
                return self.ac.act_inference(obs_history)

        orig_device = next(self.actor_critic.parameters()).device
        model = _PolicyExport(self.actor_critic).cpu().eval()
        dummy = torch.zeros(1, self.actor_critic.num_actor_obs)
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy,),
                save_path,
                export_params=True,
                opset_version=18,
                input_names=["obs_history"],
                output_names=["actions"],
                dynamic_axes={"obs_history": {0: "batch_size"}, "actions": {0: "batch_size"}},
            )
        self.actor_critic.to(orig_device)
        print(f"Exported HIM-PPO policy to {save_path}")

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the end-to-end HIM-PPO policy (estimator + actor) via TorchScript trace.

        Wraps estimator + actor as a plain module (without HIMActorCritic's properties)
        so that torch.jit.trace can introspect it without hitting the distribution assert.
        """
        orig_device = next(self.actor_critic.parameters()).device
        ac = self.actor_critic.cpu().eval()
        num_one_step_obs = ac.num_one_step_obs

        class _PolicyExport(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.estimator = ac.estimator
                self.actor_mlp = ac.actor

            def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
                vel, latent = self.estimator.get_latent(obs_history)
                actor_input = torch.cat((obs_history[:, :num_one_step_obs], vel, latent), dim=-1)
                return self.actor_mlp(actor_input)

        model = _PolicyExport().eval()
        dummy = torch.zeros(1, ac.num_actor_obs)
        with torch.inference_mode():
            traced = cast(torch.jit.ScriptModule, torch.jit.trace(model, (dummy,)))
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)
        traced.save(save_path)
        self.actor_critic.to(orig_device)
        print(f"Exported HIM-PPO policy (JIT) to {save_path}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _print_iter(
        self,
        it: int,
        tot: int,
        value_loss: float,
        surrogate_loss: float,
        estimation_loss: float,
        swap_loss: float,
        elapsed: float,
        infos: dict,
    ) -> None:
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
        print(f"{'Mean value loss':>40}: {value_loss:.4f}")
        print(f"{'Mean surrogate loss':>40}: {surrogate_loss:.4f}")
        print(f"{'Mean estimation loss':>40}: {estimation_loss:.4f}")
        print(f"{'Mean swap loss':>40}: {swap_loss:.4f}")
        if mean_rew:
            print(f"{'Mean episode reward':>40}: {mean_rew:.4f}")
        if mean_len:
            print(f"{'Mean episode length':>40}: {mean_len:.1f}")
        for k, v in (infos.get("log") or {}).items():
            print(f"{k:>40}: {v:.4f}")
        print(f"{'Time elapsed':>40}: {time_str}")
        print(f"{'ETA':>40}: {eta_str}")
        print(sep)
