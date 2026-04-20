from __future__ import annotations

import torch
import torch.optim as optim
from itertools import chain
from typing import Any

from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups, resolve_optimizer

from unilab.algos.torch.hora.models import HoraActorModel, HoraCriticModel, HoraSharedActorCritic
from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO


class HoraPPO(FinalObservationAwarePPO):
    """PPO variant that constructs a shared HORA actor-critic backbone."""

    def __init__(
        self,
        actor: HoraActorModel,
        critic: HoraCriticModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        if rnd_cfg:
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            from rsl_rl.extensions import RandomNetworkDistillation

            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        if symmetry_cfg is not None:
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            from rsl_rl.utils import resolve_callable

            symmetry_cfg["data_augmentation_func"] = resolve_callable(
                symmetry_cfg["data_augmentation_func"]
            )
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.optimizer = resolve_optimizer(optimizer)(
            self._unique_trainable_parameters(),
            lr=learning_rate,
        )
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def _unique_trainable_parameters(self) -> list[torch.nn.Parameter]:
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for param in chain(self.actor.parameters(), self.critic.parameters()):
            ident = id(param)
            if ident in seen:
                continue
            seen.add(ident)
            params.append(param)
        return params

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])
        actor_cfg.pop("class_name", None)
        critic_cfg.pop("class_name", None)

        proprio_hist_len = int(actor_cfg.pop("proprio_hist_len", obs["proprio_hist"].shape[1]))
        proprio_frame_dim = int(actor_cfg.pop("proprio_frame_dim", obs["proprio_hist"].shape[-1]))

        shared_model = HoraSharedActorCritic(
            obs_dim=int(obs["actor"].shape[-1]),
            action_dim=int(env.num_actions),
            priv_info_dim=int(obs["priv_info"].shape[-1]),
            actor_hidden_dims=actor_cfg.pop("hidden_dims", (512, 256, 128)),
            activation=actor_cfg.pop("activation", "elu"),
            obs_normalization=bool(actor_cfg.pop("obs_normalization", False)),
            distribution_cfg=actor_cfg.pop("distribution_cfg", None),
            priv_info_embed_dim=int(actor_cfg.pop("priv_info_embed_dim", obs["priv_info"].shape[-1])),
            priv_mlp_hidden_dims=actor_cfg.pop("priv_mlp_hidden_dims", (256, 128, 8)),
            use_student_encoder=bool(actor_cfg.pop("use_student_encoder", False)),
            proprio_hist_len=proprio_hist_len,
            proprio_frame_dim=proprio_frame_dim,
        ).to(device)

        actor = HoraActorModel(
            obs,
            cfg["obs_groups"],
            "actor",
            env.num_actions,
            shared_model=shared_model,
            **actor_cfg,
        ).to(device)
        critic = HoraCriticModel(
            obs,
            cfg["obs_groups"],
            "critic",
            1,
            shared_model=shared_model,
            **critic_cfg,
        ).to(device)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        algorithm_cfg = dict(cfg["algorithm"])
        algorithm_cfg.pop("class_name", None)
        return HoraPPO(actor, critic, storage, device=device, **algorithm_cfg, multi_gpu_cfg=cfg["multi_gpu"])

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        self.actor.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        timeouts = extras.get("time_outs")
        timeout_bootstrap_obs = extras.get("time_out_bootstrap_obs")
        if isinstance(timeouts, torch.Tensor):
            timeout_mask = timeouts.to(self.device).float()
            can_bootstrap = (
                timeout_bootstrap_obs is not None
                and isinstance(timeout_bootstrap_obs, TensorDict)
                and "priv_info" in timeout_bootstrap_obs
                and torch.count_nonzero(timeout_mask) > 0
            )
            if can_bootstrap:
                bootstrap_obs = timeout_bootstrap_obs.to(self.device)
                bootstrap_values = self.critic(bootstrap_obs).detach()
                self.transition.rewards += self.gamma * torch.squeeze(
                    bootstrap_values * timeout_mask.unsqueeze(1), 1
                )
            else:
                transition_values = self.transition.values
                assert transition_values is not None
                self.transition.rewards += self.gamma * torch.squeeze(
                    transition_values * timeout_mask.unsqueeze(1), 1
                )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
