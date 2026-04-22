from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.hora.models import HoraActorModel, HoraSharedActorCritic


@dataclass
class HoraDistillStats:
    agent_steps: int = 0
    best_reward: float = float("-inf")
    mean_reward: float = float("nan")


def build_student_actor_and_normalizer(
    env,
    cfg: DictConfig,
    *,
    device: torch.device,
) -> tuple[HoraActorModel, EmpiricalNormalization]:
    actor_obs = env.get_observations()
    actor_dim = int(actor_obs["actor"].shape[-1])
    priv_info_dim = int(actor_obs["priv_info"].shape[-1])
    proprio_hist_shape = actor_obs["proprio_hist"].shape[1:]

    model_cfg = OmegaConf.to_container(cfg.algo.model, resolve=True)
    assert isinstance(model_cfg, dict)
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
    actor = HoraActorModel(
        actor_obs,
        {"actor": ["actor"], "critic": ["actor"]},
        "actor",
        int(env.num_actions),
        shared_model=shared,
        use_student_encoder=True,
    ).to(device)
    hist_normalizer = EmpiricalNormalization(proprio_hist_shape, device=device)
    return actor, hist_normalizer


def load_teacher_actor_weights(
    actor: HoraActorModel,
    teacher_checkpoint: str | Path,
    *,
    teacher_algo_family: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
    actor_state_key = {
        "ppo": "actor_state_dict",
        "appo": "actor",
    }.get(str(teacher_algo_family))
    if actor_state_key is None:
        raise ValueError(
            "Unsupported HORA teacher algorithm family for distillation: "
            f"{teacher_algo_family!r}. Expected one of ['ppo', 'appo']."
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
    actor: HoraActorModel,
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
    return checkpoint


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
        self.distill_runtime_cfg = OmegaConf.to_container(distill_runtime_cfg, resolve=True)
        self.actor, self.hist_normalizer = build_student_actor_and_normalizer(
            env,
            cfg,
            device=self.device,
        )
        self.optimizer = torch.optim.Adam(self._trainable_parameters(), lr=float(cfg.algo.learning_rate))
        self.stats = HoraDistillStats()
        self._reward_buffer: deque[float] = deque(maxlen=100)
        self._step_reward = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
        self._step_length = torch.zeros((env.num_envs,), dtype=torch.float32, device=self.device)
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

        while self.stats.agent_steps < max_agent_steps:
            norm_obs = self._normalize_student_obs(obs_td)
            obs_batch = {
                key: value.detach() if key == "actor" else value
                for key, value in norm_obs.items()
            }
            td = TensorDict(obs_batch, batch_size=obs_td.batch_size, device=self.device)
            _, core_output = self.actor.shared.policy_mean(td, prefer_student=True)
            loss = torch.mean((core_output.privileged_latent - core_output.privileged_target.detach()) ** 2)

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
                done_mean_reward = float(torch.mean(completed_rewards).item())
                self._reward_buffer.extend(completed_rewards.detach().cpu().numpy().tolist())
                self.stats.mean_reward = float(statistics.mean(self._reward_buffer))
                self.stats.best_reward = max(self.stats.best_reward, done_mean_reward)
                self._step_reward[done_idx] = 0.0
                self._step_length[done_idx] = 0.0

            if next_log_steps is not None and self.stats.agent_steps >= next_log_steps:
                elapsed = max(time.time() - start_time, 1e-6)
                self.logger.info(
                    "agent_steps=%d loss=%.6f mean_reward=%.4f best_reward=%.4f fps=%.1f",
                    self.stats.agent_steps,
                    float(loss.item()),
                    self.stats.mean_reward,
                    self.stats.best_reward,
                    self.stats.agent_steps / elapsed,
                )
                next_log_steps = self._next_interval_boundary(self.stats.agent_steps, log_interval)

            if next_save_steps is not None and self.stats.agent_steps >= next_save_steps:
                self.save(self.log_dir / f"hora_stage2_{self.stats.agent_steps}.pt")
                next_save_steps = self._next_interval_boundary(self.stats.agent_steps, save_interval)

        self.save(self.log_dir / "hora_stage2_last.pt")

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model_state_dict": self.actor.state_dict(),
                "history_normalizer": self.hist_normalizer.state_dict(),
                "agent_steps": self.stats.agent_steps,
                "teacher_checkpoint": str(self.teacher_checkpoint),
                "distill_runtime_cfg": self.distill_runtime_cfg,
            },
            Path(path),
        )
