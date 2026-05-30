from __future__ import annotations

import math
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.hora.models import (
    HoraActorModel,
    HoraCoreOutput,
    HoraSharedActorCritic,
    ProprioAdaptTConv,
)
from unilab.algos.torch.hora.sac_models import HoraSACActor


class HoraSACDistillShared(nn.Module):
    """SAC-teacher-compatible HORA stage-2 shared actor."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        priv_info_dim: int,
        hidden_dim: int = 512,
        priv_info_embed_dim: int = 9,
        priv_mlp_hidden_dims: list[int] | tuple[int, ...] = (256, 128, 9),
        use_layer_norm: bool = True,
        proprio_hist_len: int = 30,
        proprio_frame_dim: int | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        teacher = HoraSACActor(
            obs_dim=obs_dim,
            priv_info_dim=priv_info_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=priv_mlp_hidden_dims,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.priv_info_embed_dim = int(priv_info_embed_dim)
        self.proprio_hist_len = int(proprio_hist_len)
        self.proprio_frame_dim = (
            int(proprio_frame_dim) if proprio_frame_dim is not None else self.obs_dim // 3
        )
        self.obs_normalizer = nn.Identity()
        self.priv_encoder = teacher.priv_encoder
        self.priv_projection = teacher.priv_projection
        self.actor_trunk = teacher.actor_trunk
        self.action_mean_head = teacher.action_mean_head
        self.adapt_tconv = ProprioAdaptTConv(self.proprio_frame_dim, self.priv_info_embed_dim)

    def load_teacher_actor_state_dict(self, actor_state: dict[str, torch.Tensor]) -> None:
        own_state = self.state_dict()
        teacher_state = {
            key: value
            for key, value in actor_state.items()
            if key in own_state and not key.startswith("adapt_tconv.")
        }
        missing = sorted(
            key
            for key in own_state
            if not key.startswith("adapt_tconv.") and key not in teacher_state
        )
        if missing:
            raise ValueError(f"HORA-SAC teacher checkpoint is missing actor keys: {missing}")
        self.load_state_dict(teacher_state, strict=False)

    def encode_privileged_info(self, priv_info: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.priv_projection(self.priv_encoder(priv_info)))

    def encode_proprio_history(self, proprio_hist: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.adapt_tconv(proprio_hist))

    def _zero_privileged_latent(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, self.priv_info_embed_dim), device=device, dtype=dtype)

    def policy_mean(
        self,
        obs: TensorDict,
        *,
        prefer_student: bool,
        require_privileged_target: bool = True,
    ) -> tuple[torch.Tensor, HoraCoreOutput]:
        actor_obs = obs["actor"]
        priv_info = obs.get("priv_info")
        if priv_info is None:
            if require_privileged_target or not prefer_student:
                raise ValueError("priv_info is required for HORA-SAC distillation")
            privileged_target = self._zero_privileged_latent(
                actor_obs.shape[0],
                actor_obs.device,
                actor_obs.dtype,
            )
        else:
            privileged_target = self.encode_privileged_info(priv_info)

        if prefer_student:
            proprio_hist = obs.get("proprio_hist")
            if proprio_hist is None:
                raise ValueError("proprio_hist is required for HORA-SAC student inference")
            privileged_latent = self.encode_proprio_history(proprio_hist)
        else:
            privileged_latent = privileged_target

        trunk_latent = self.actor_trunk(torch.cat([actor_obs, privileged_latent], dim=-1))
        mean = self.action_mean_head(trunk_latent)
        return (
            mean,
            HoraCoreOutput(
                policy_obs=actor_obs,
                trunk_latent=trunk_latent,
                privileged_latent=privileged_latent,
                privileged_target=privileged_target,
            ),
        )


class HoraSACDistillActor(nn.Module):
    """Stage-2 actor wrapper for HORA-SAC teachers."""

    is_recurrent: bool = False

    def __init__(self, shared: HoraSACDistillShared) -> None:
        super().__init__()
        self.shared = shared
        self.prefer_student = True

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del masks, hidden_state, stochastic_output
        mean, _ = self.shared.policy_mean(
            obs,
            prefer_student=self.prefer_student,
            require_privileged_target=False,
        )
        return torch.tanh(mean)

    def load_sac_teacher_actor_state_dict(self, actor_state: dict[str, torch.Tensor]) -> None:
        self.shared.load_teacher_actor_state_dict(actor_state)


DistillActor = HoraActorModel | HoraSACDistillActor


@dataclass
class HoraDistillStats:
    agent_steps: int = 0
    best_reward: float = float("-inf")
    mean_reward: float = float("nan")
    mean_episode_length: float = float("nan")


def build_student_actor_and_normalizer(
    env,
    cfg: DictConfig,
    *,
    device: torch.device,
) -> tuple[DistillActor, EmpiricalNormalization]:
    actor_obs = env.get_observations()
    actor_dim = int(actor_obs["actor"].shape[-1])
    priv_info_dim = int(actor_obs["priv_info"].shape[-1])
    proprio_hist_shape = actor_obs["proprio_hist"].shape[1:]

    model_cfg = OmegaConf.to_container(cfg.algo.model, resolve=True)
    assert isinstance(model_cfg, dict)
    if model_cfg.get("teacher_arch") == "hora_sac":
        shared_sac = HoraSACDistillShared(
            obs_dim=actor_dim,
            action_dim=int(env.num_actions),
            priv_info_dim=priv_info_dim,
            hidden_dim=int(model_cfg.get("actor_hidden_dim", 512)),
            priv_info_embed_dim=int(model_cfg.get("priv_info_embed_dim", priv_info_dim)),
            priv_mlp_hidden_dims=model_cfg.get("priv_mlp_hidden_dims", [256, 128, 9]),
            use_layer_norm=bool(model_cfg.get("use_layer_norm", True)),
            proprio_hist_len=int(proprio_hist_shape[0]),
            proprio_frame_dim=int(proprio_hist_shape[1]),
            device=device,
        ).to(device)
        actor = cast(HoraSACDistillActor, HoraSACDistillActor(shared_sac).to(device))
        hist_normalizer = EmpiricalNormalization(proprio_hist_shape, device=device)
        return actor, hist_normalizer

    shared = HoraSharedActorCritic(
        obs_dim=actor_dim,
        action_dim=int(env.num_actions),
        priv_info_dim=priv_info_dim,
        actor_hidden_dims=model_cfg.get("hidden_dims", [512, 256, 128]),
        activation=model_cfg.get("activation", "elu"),
        obs_normalization=model_cfg.get("obs_normalization", True),
        distribution_cfg=model_cfg.get("distribution_cfg", {"init_std": 1.0, "std_type": "scalar"}),
        priv_info_embed_dim=model_cfg.get("priv_info_embed_dim", priv_info_dim),
        priv_mlp_hidden_dims=model_cfg.get("priv_mlp_hidden_dims", [256, 128, 8]),
        use_student_encoder=True,
        proprio_hist_len=int(proprio_hist_shape[0]),
        proprio_frame_dim=int(proprio_hist_shape[1]),
    ).to(device)
    actor = cast(
        HoraActorModel,
        HoraActorModel(
            actor_obs,
            {"actor": ["actor"], "critic": ["actor"]},
            "actor",
            int(env.num_actions),
            shared_model=shared,
            use_student_encoder=True,
        ).to(device),
    )
    hist_normalizer = EmpiricalNormalization(proprio_hist_shape, device=device)
    return actor, hist_normalizer


def load_teacher_actor_weights(
    actor: nn.Module,
    teacher_checkpoint: str | Path,
    *,
    teacher_algo_family: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    if str(teacher_algo_family) == "sac":
        actor_state = checkpoint.get("actor")
        if actor_state is None:
            raise ValueError(
                "Checkpoint does not contain the expected HORA-SAC actor weights. "
                f"checkpoint={teacher_checkpoint}"
            )
        load_sac = getattr(actor, "load_sac_teacher_actor_state_dict", None)
        if load_sac is None:
            raise ValueError("Selected distillation actor does not support HORA-SAC weights.")
        load_sac(actor_state)
        return

    actor_state_key = {
        "ppo": "actor_state_dict",
        "appo": "actor",
    }.get(str(teacher_algo_family))
    if actor_state_key is None:
        raise ValueError(
            "Unsupported HORA teacher algorithm family for distillation: "
            f"{teacher_algo_family!r}. Expected one of ['ppo', 'appo', 'sac']."
        )
    actor_state = checkpoint.get(actor_state_key)
    if actor_state is None:
        raise ValueError(
            "Checkpoint does not contain the expected teacher actor weights. "
            f"algo_family={teacher_algo_family!r} expected_key={actor_state_key!r} "
            f"checkpoint={teacher_checkpoint}"
        )
    actor.load_state_dict(actor_state, strict=False)


def load_distilled_checkpoint(
    actor: nn.Module,
    hist_normalizer: EmpiricalNormalization,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {checkpoint_path}")
    actor.load_state_dict(model_state, strict=True)

    history_normalizer = checkpoint.get("history_normalizer")
    if history_normalizer is not None:
        hist_normalizer.load_state_dict(history_normalizer)
    return cast(dict[str, Any], checkpoint)


class HoraDistillationTrainer:
    """Stage-2 HORA latent distillation trainer."""

    def __init__(
        self,
        env,
        cfg: DictConfig,
        *,
        device: str,
        log_dir: str | Path,
        teacher_checkpoint: str | Path,
        teacher_algo_family: str,
        teacher_metadata: dict[str, Any] | None = None,
        distill_runtime_cfg: DictConfig,
        logger,
    ) -> None:
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.logger = logger
        self.teacher_checkpoint = Path(teacher_checkpoint)
        self.teacher_algo_family = str(teacher_algo_family)
        self.teacher_metadata = dict(teacher_metadata or {})
        self.distill_runtime_cfg = OmegaConf.to_container(distill_runtime_cfg, resolve=True)
        self.actor, self.hist_normalizer = build_student_actor_and_normalizer(
            env,
            cfg,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(
            self._trainable_parameters(), lr=float(cfg.algo.learning_rate)
        )
        self.stats = HoraDistillStats()
        self._reward_buffer: deque[float] = deque(maxlen=100)
        self._episode_length_buffer: deque[float] = deque(maxlen=100)
        self._step_reward = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        self._step_length = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        self._tb_writer = self._build_tensorboard_writer()
        self._load_teacher_checkpoint()

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        for name, param in self.actor.named_parameters():
            requires_grad = "adapt_tconv" in name
            param.requires_grad = requires_grad
            if requires_grad:
                params.append(param)
        return params

    def _load_teacher_checkpoint(self) -> None:
        load_teacher_actor_weights(
            self.actor,
            self.teacher_checkpoint,
            teacher_algo_family=self.teacher_algo_family,
            device=self.device,
        )
        self.actor.train()
        self.actor.shared.obs_normalizer.eval()

    def _build_tensorboard_writer(self) -> Any | None:
        """Create the stage-2 TensorBoard writer when the config requests it.

        Args:
            None.

        Returns:
            Summary writer rooted at ``<log_dir>/tb``, or ``None`` when scalar
            backend logging is disabled or TensorBoard is unavailable.
        """
        logger_type = str(OmegaConf.select(self.cfg, "training.logger", default="tensorboard"))
        if logger_type.lower() != "tensorboard":
            return None

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.logger.warning(
                "tensorboard is not installed; disabling HORA distillation TensorBoard logging."
            )
            return None

        tb_dir = self.log_dir / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("TensorBoard: %s", tb_dir)
        return SummaryWriter(log_dir=str(tb_dir))

    def _add_scalar_if_finite(self, tag: str, value: float, *, step: int) -> None:
        """Write a scalar only when a TensorBoard writer exists and the value is finite.

        Args:
            tag: TensorBoard metric name.
            value: Scalar value to record.
            step: Global step associated with the scalar.

        Returns:
            None. Invalid or disabled values are skipped silently so early NaNs
            from unfinished episodes do not pollute the event stream.
        """
        if self._tb_writer is None or not math.isfinite(value):
            return
        self._tb_writer.add_scalar(tag, value, step)

    def _log_tensorboard_step(self, *, loss: float, elapsed: float) -> None:
        """Record the latest distillation scalars to TensorBoard.

        Args:
            loss: Latest latent-distillation loss.
            elapsed: Wall-clock training time since the run started.

        Returns:
            None. Metrics are written at the current agent-step count.
        """
        if self._tb_writer is None:
            return

        step = self.stats.agent_steps
        self._add_scalar_if_finite("train/loss", loss, step=step)
        self._add_scalar_if_finite("reward/mean", self.stats.mean_reward, step=step)
        self._add_scalar_if_finite("reward/best", self.stats.best_reward, step=step)
        self._add_scalar_if_finite(
            "episode/length",
            self.stats.mean_episode_length,
            step=step,
        )
        self._add_scalar_if_finite("perf/fps", step / max(elapsed, 1e-6), step=step)
        self._add_scalar_if_finite("perf/training_time_sec", elapsed, step=step)
        self._tb_writer.flush()

    def _normalize_student_obs(self, obs_td) -> dict[str, torch.Tensor]:
        actor_obs = obs_td["actor"].to(self.device)
        proprio_hist = obs_td["proprio_hist"].to(self.device)
        return {
            "actor": actor_obs,
            "priv_info": obs_td["priv_info"].to(self.device),
            "proprio_hist": self.hist_normalizer(proprio_hist),
        }

    @staticmethod
    def _next_interval_boundary(current_steps: int, interval_steps: int) -> int | None:
        """Return the next positive save boundary after the current step count.

        Args:
            current_steps: Number of agent steps already completed.
            interval_steps: Positive interval in agent steps between saves.

        Returns:
            The next interval boundary, or ``None`` when periodic saving is disabled.
        """
        if interval_steps <= 0:
            return None
        return ((current_steps // interval_steps) + 1) * interval_steps

    def train(self) -> None:
        obs_td, _ = self.env.reset()
        max_agent_steps = int(self.cfg.algo.max_agent_steps)
        save_interval = int(self.cfg.algo.save_interval_steps)
        log_interval = int(self.cfg.algo.log_interval_steps)
        next_log_steps = self._next_interval_boundary(self.stats.agent_steps, log_interval)
        next_save_steps = self._next_interval_boundary(self.stats.agent_steps, save_interval)
        start_time = time.time()
        last_loss = float("nan")

        try:
            while self.stats.agent_steps < max_agent_steps:
                norm_obs = self._normalize_student_obs(obs_td)
                obs_batch = {
                    key: value.detach() if key == "actor" else value
                    for key, value in norm_obs.items()
                }
                td = TensorDict(obs_batch, batch_size=obs_td.batch_size, device=self.device)
                _, core_output = self.actor.shared.policy_mean(td, prefer_student=True)
                loss = torch.mean(
                    (core_output.privileged_latent - core_output.privileged_target.detach()) ** 2
                )
                last_loss = float(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    actions = self.actor(td, stochastic_output=False).clamp_(-1.0, 1.0)
                obs_td, rewards, dones, infos = self.env.step(actions)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                self.stats.agent_steps += int(self.env.num_envs)

                self._step_reward += rewards
                self._step_length += 1
                done_idx = torch.nonzero(dones, as_tuple=False).flatten()
                if len(done_idx) > 0:
                    completed_rewards = self._step_reward[done_idx]
                    completed_lengths = self._step_length[done_idx]
                    done_mean_reward = float(torch.mean(completed_rewards).item())
                    self._reward_buffer.extend(completed_rewards.detach().cpu().numpy().tolist())
                    self._episode_length_buffer.extend(
                        completed_lengths.detach().cpu().numpy().tolist()
                    )
                    self.stats.mean_reward = float(statistics.mean(self._reward_buffer))
                    self.stats.mean_episode_length = float(
                        statistics.mean(self._episode_length_buffer)
                    )
                    self.stats.best_reward = max(self.stats.best_reward, done_mean_reward)
                    self._step_reward[done_idx] = 0.0
                    self._step_length[done_idx] = 0.0

                if next_log_steps is not None and self.stats.agent_steps >= next_log_steps:
                    elapsed = max(time.time() - start_time, 1e-6)
                    self.logger.info(
                        "agent_steps=%d loss=%.6f mean_reward=%.4f best_reward=%.4f "
                        "mean_episode_length=%.2f training_time=%.2fs fps=%.1f",
                        self.stats.agent_steps,
                        last_loss,
                        self.stats.mean_reward,
                        self.stats.best_reward,
                        self.stats.mean_episode_length,
                        elapsed,
                        self.stats.agent_steps / elapsed,
                    )
                    self._log_tensorboard_step(loss=last_loss, elapsed=elapsed)
                    next_log_steps = self._next_interval_boundary(
                        self.stats.agent_steps, log_interval
                    )

                if next_save_steps is not None and self.stats.agent_steps >= next_save_steps:
                    self.save(self.log_dir / f"hora_stage2_{self.stats.agent_steps}.pt")
                    next_save_steps = self._next_interval_boundary(
                        self.stats.agent_steps, save_interval
                    )

            self.save(self.log_dir / "hora_stage2_last.pt")
            total_elapsed = max(time.time() - start_time, 1e-6)
            self.logger.info(
                "training_complete agent_steps=%d mean_reward=%.4f best_reward=%.4f "
                "mean_episode_length=%.2f training_time=%.2fs",
                self.stats.agent_steps,
                self.stats.mean_reward,
                self.stats.best_reward,
                self.stats.mean_episode_length,
                total_elapsed,
            )
            self._log_tensorboard_step(loss=last_loss, elapsed=total_elapsed)
        finally:
            if self._tb_writer is not None:
                self._tb_writer.close()
                self._tb_writer = None

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state_dict": self.actor.state_dict(),
                "history_normalizer": self.hist_normalizer.state_dict(),
                "agent_steps": self.stats.agent_steps,
                "teacher_checkpoint": str(self.teacher_checkpoint),
                "teacher_algo_family": self.teacher_algo_family,
                "teacher_metadata": self.teacher_metadata,
                "distill_runtime_cfg": self.distill_runtime_cfg,
            },
            Path(path),
        )
