from torch import nn

from .actor_critic_rma import Actor
from .adaptation_module import AdaptationModule


def go2_control_layout():
    """Current task contract, checked against runtime; not stored in old weights."""
    return {
        "action_joint_order": ["FL", "FR", "RL", "RR"],
        "scene_actuator_order": ["FR", "FL", "RR", "RL"],
        "action_to_actuator_indices": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        "control_layout_source": (
            "Current Go2 YAML and Entity.find_joints_by_actuator_names natural joint order; "
            "verified in the original 1.3.3 runtime. Historical checkpoints do not store "
            "this layout; confirm against a trace from the matching scene before deployment."
        ),
    }


class DeploymentPolicy(nn.Module):
    """Only student adaptation and base actor. No teacher encoder or critic."""

    def __init__(self, cfg):
        super().__init__()
        self.adaptation = AdaptationModule(cfg)
        self.actor = Actor(cfg)

    def forward(self, observation, history):
        return self.actor(observation, self.adaptation(history)).tanh()
