from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class RMAConfig:
    obs_dim: int = 49
    critic_dim: int = 52
    privileged_dim: int = 2  # sliding friction coefficient, added base mass (kg)
    history_length: int = 50
    latent_dim: int = 8
    action_dim: int = 12
    hidden: tuple = (256, 128)


def mlp(widths, final=None):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.ELU())
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


class PrivilegedEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Fixed scales serialized in checkpoint. No unexported running statistics.
        self.register_buffer("scale", torch.tensor([1.0, 5.0]))
        if cfg.privileged_dim != 2:
            raise ValueError("This contract requires friction and added base mass")
        self.net = mlp([2, 64, 64, cfg.latent_dim], nn.Tanh())

    def forward(self, privileged):
        return self.net(privileged / self.scale)


class Actor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mlp([cfg.obs_dim + cfg.latent_dim, *cfg.hidden, cfg.action_dim])

    def forward(self, obs, z):
        # Pre-tanh Gaussian mean; squashing is explicit in training and deployment.
        return self.net(torch.cat((obs, z), -1))


class ActorCriticRMA(nn.Module):
    def __init__(self, cfg=RMAConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = PrivilegedEncoder(cfg)
        self.actor = Actor(cfg)
        self.critic = mlp([cfg.critic_dim + cfg.privileged_dim, *cfg.hidden, 1])
        self.log_std = nn.Parameter(torch.full((cfg.action_dim,), -0.7))

    def value(self, critic_obs, privileged):
        return self.critic(torch.cat((critic_obs, privileged / self.encoder.scale), -1)).squeeze(-1)

    def distribution(self, obs, privileged):
        mean = self.actor(obs, self.encoder(privileged))
        return Normal(mean, self.log_std.clamp(-5, 1).exp())

    @staticmethod
    def log_prob(dist, raw):
        # Numerically stable log(1 - tanh(raw)^2).
        correction = 2 * (0.6931471805599453 - raw - nn.functional.softplus(-2 * raw))
        return (dist.log_prob(raw) - correction).sum(-1)

    def sample(self, obs, privileged):
        dist = self.distribution(obs, privileged)
        raw = dist.sample()
        return raw.tanh(), raw, self.log_prob(dist, raw)

    def metadata(self):
        return asdict(self.cfg)
