from typing import Dict, Literal

import torch
import torch.nn.functional as F
from torch import autograd
from torch.amp import autocast
from torch.utils._pytree import tree_map

from ..base import BaseConfig
from ..fb.agent import FBAgent, FBAgentTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from ..pytree_utils import tree_get_batch_size
from .model import FBcprModel, FBcprModelConfig


class FBcprAgentTrainConfig(FBAgentTrainConfig):
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    scale_reg_mode: Literal["fb", "ratio"] = "fb"
    reg_denom_floor: float = 1.0

    expert_asm_ratio: float = 0

    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0

    weight_decay_discriminator: float = 0.0


class FBcprAgentConfig(BaseConfig):
    name: Literal["FBcprAgent"] = "FBcprAgent"
    model: FBcprModelConfig = FBcprModelConfig()
    train: FBcprAgentTrainConfig = FBcprAgentTrainConfig()
    cudagraphs: bool = False
    compile: bool = False

    def build(self, obs_space, action_dim):
        return FBcprAgent(obs_space, action_dim, self)

    @property
    def object_class(self):
        return FBcprAgent


class FBcprAgent(FBAgent):
    config_class = FBcprAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg

        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), (
            "Batch size should be divisable by seq_length"
        )
        del seq_length, batch_size

        self._model: FBcprModel = self.cfg.model.build(obs_space, action_dim)

        self._model.to(self.device)
        self.setup_training()
        self.setup_compile()
        self.env_idx_with_expert_rollout = None

    @classmethod
    def supported_evaluations(cls):
        return ["reward", "tracking"]

    @property
    def optimizer_dict(self):
        optimizers = super().optimizer_dict
        optimizers.update(
            {
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            }
        )
        return optimizers

    def setup_training(self) -> None:
        super().setup_training()

        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(
            x for x in self._model._target_critic.parameters()
        )

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    @torch.no_grad()
    def sample_mixed_z(
        self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs
    ):
        with autocast(
            device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp
        ):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
            p_goal = self.cfg.train.train_goal_ratio
            p_expert_asm = self.cfg.train.expert_asm_ratio
            prob = torch.tensor(
                [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
                dtype=torch.float32,
                device=self.device,
            )
            mix_idxs = torch.multinomial(
                prob, num_samples=self.cfg.train.batch_size, replacement=True
            ).reshape(-1, 1)

            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            train_goal = tree_map(lambda x: x[perm], train_goal)
            goals = self._model._backward_map(train_goal)
            goals = self._model.project_z(goals)
            z = torch.where(mix_idxs == 0, goals, z)

            perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
            z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
    ):

        with autocast(
            device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp
        ):
            B_expert = self._model._backward_map(next_obs).detach()
            B_expert = B_expert.view(
                self.cfg.train.batch_size // self.cfg.model.seq_length,
                self.cfg.model.seq_length,
                B_expert.shape[-1],
            )
            z_expert = B_expert.mean(dim=1)
            z_expert = self._model.project_z(z_expert)
            z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        grad_penalty = (
            self.cfg.train.grad_penalty_discriminator
            if self.cfg.train.grad_penalty_discriminator > 0
            else None
        )
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = (
                torch.rand((self.cfg.train.batch_size, 1), device=self.device)
                <= self.cfg.train.relabel_ratio
            )
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = (
            self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None
        )

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor | dict[str, torch.Tensor],
        real_z: torch.Tensor,
        fake_obs: torch.Tensor | dict[str, torch.Tensor],
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(real_obs)
        alpha = torch.rand(batch_size, 1, device=real_z.device)

        interpolated_obs = {}
        interpolated_obs_list = []
        if isinstance(real_obs, torch.Tensor):
            if real_obs.shape != fake_obs.shape:
                raise ValueError(f"Shape mismatch: {real_obs.shape} vs {fake_obs.shape}")
            interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
            interpolated_obs_list.append(interpolated_obs)
        else:
            for key in real_obs.keys():
                real_obs_tensor = real_obs[key]
                fake_obs_tensor = fake_obs[key]
                if isinstance(real_obs_tensor, torch.Tensor):
                    interpolated_obs[key] = (
                        alpha * real_obs_tensor + (1 - alpha) * fake_obs_tensor
                    ).requires_grad_(True)
                    interpolated_obs_list.append(interpolated_obs[key])
                else:
                    raise ValueError(f"Unsupported type for key {key}: {type(real_obs_tensor)}")

        interpolated_z = alpha * real_z + (1 - alpha) * fake_z
        interpolated_z = interpolated_z.requires_grad_(True)

        d_interpolates = self._model._discriminator.compute_logits(interpolated_obs, interpolated_z)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_obs_list + [interpolated_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )

        gradients = [g for g in gradients if g is not None]
        cat_gradients = torch.cat(gradients, dim=1)
        gradient_penalty = ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(
            device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp
        ):
            expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
            unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)

            expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
            unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
            loss = torch.mean(expert_loss + unlabeled_loss)

            if grad_penalty is not None:
                wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
                loss += grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        with autocast(
            device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp
        ):
            num_parallel = self.cfg.model.archi.critic.num_parallel

            with torch.no_grad():
                reward = self._model._discriminator.compute_reward(obs=obs, z=z)
                dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
                next_action = dist.sample(clip=self.cfg.train.stddev_clip)
                next_Qs = self._model._target_critic(next_obs, z, next_action)
                Q_mean, Q_unc, next_V = self.get_targets_uncertainty(
                    next_Qs, self.cfg.train.critic_pessimism_penalty
                )
                target_Q = reward + discount * next_V
                expanded_targets = target_Q.expand(num_parallel, -1, -1)

            Qs = self._model._critic(obs, z, action)
            critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        with autocast(
            device_type=self.device, dtype=self._model.amp_dtype, enabled=self.cfg.model.amp
        ):
            dist = self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)

            Qs_discriminator = self._model._critic(obs, z, action)
            _, _, Q_discriminator = self.get_targets_uncertainty(
                Qs_discriminator, self.cfg.train.actor_pessimism_penalty
            )

            Fs = self._model._forward_map(obs, z, action)
            Qs_fb = (Fs * z).sum(-1)
            _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            if self.cfg.train.scale_reg and self.cfg.train.scale_reg_mode == "ratio":
                denom = (
                    Q_discriminator.abs().mean().detach().clamp_min(self.cfg.train.reg_denom_floor)
                )
                w_disc = weight / denom
                actor_loss = (
                    -Q_discriminator.mean() * self.cfg.train.reg_coeff * w_disc - Q_fb.mean()
                )
            else:
                w_disc = weight
                actor_loss = (
                    -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()
                )

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            if self.cfg.train.scale_reg and self.cfg.train.scale_reg_mode == "ratio":
                _qd = Q_discriminator.abs().mean()
                _r_disc = (
                    self.cfg.train.reg_coeff * (_qd / _qd.clamp_min(self.cfg.train.reg_denom_floor))
                ).detach()
            else:
                _r_disc = self.cfg.train.reg_coeff * Q_discriminator.abs().mean()
            _w_diag = (
                w_disc
                if isinstance(w_disc, torch.Tensor)
                else torch.tensor(w_disc, device=Q_fb.device)
            )
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
                "r_disc": _r_disc.detach(),
                "w_disc": _w_diag.detach(),
                "fb_share": (1.0 / (1.0 + _r_disc)).detach(),
            }
        return output_metrics
