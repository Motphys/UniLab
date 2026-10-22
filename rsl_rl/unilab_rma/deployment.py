from torch import nn

from .actor_critic_rma import Actor
from .adaptation_module import AdaptationModule


class DeploymentPolicy(nn.Module):
    """Only student adaptation and base actor. No teacher encoder or critic."""

    def __init__(self, cfg):
        super().__init__()
        self.adaptation = AdaptationModule(cfg)
        self.actor = Actor(cfg)

    def forward(self, observation, history):
        return self.actor(observation, self.adaptation(history)).tanh()
