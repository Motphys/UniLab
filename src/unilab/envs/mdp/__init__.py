"""Community-style built-in MDP terms for UniLab's NumPy manager runtime."""

from unilab.envs.mdp.actions import JointPositionAction as JointPositionAction
from unilab.envs.mdp.actions import JointPositionActionCfg as JointPositionActionCfg
from unilab.envs.mdp.commands import UniformVelocityCommand as UniformVelocityCommand
from unilab.envs.mdp.commands import UniformVelocityCommandCfg as UniformVelocityCommandCfg
from unilab.envs.mdp.events import apply_body_impulse as apply_body_impulse
from unilab.envs.mdp.events import dof_armature as dof_armature
from unilab.envs.mdp.events import geom_friction as geom_friction
from unilab.envs.mdp.events import joint_armature as joint_armature
from unilab.envs.mdp.events import pd_gains as pd_gains
from unilab.envs.mdp.events import push_by_setting_velocity as push_by_setting_velocity
from unilab.envs.mdp.events import randomize_encoder_bias as randomize_encoder_bias
from unilab.envs.mdp.events import (
    randomize_physics_scene_gravity as randomize_physics_scene_gravity,
)
from unilab.envs.mdp.events import randomize_rigid_body_com as randomize_rigid_body_com
from unilab.envs.mdp.events import randomize_rigid_body_mass as randomize_rigid_body_mass
from unilab.envs.mdp.events import reset_root_state_uniform as reset_root_state_uniform
from unilab.envs.mdp.events import reset_scene_to_default as reset_scene_to_default
from unilab.envs.mdp.events import resolve_env_ids as resolve_env_ids
from unilab.envs.mdp.observations import base_ang_vel as base_ang_vel
from unilab.envs.mdp.observations import base_lin_vel as base_lin_vel
from unilab.envs.mdp.observations import builtin_sensor as builtin_sensor
from unilab.envs.mdp.observations import generated_commands as generated_commands
from unilab.envs.mdp.observations import joint_pos_rel as joint_pos_rel
from unilab.envs.mdp.observations import joint_vel_rel as joint_vel_rel
from unilab.envs.mdp.observations import last_action as last_action
from unilab.envs.mdp.observations import projected_gravity as projected_gravity
from unilab.envs.mdp.observations import (
    projected_gravity_from_sensor as projected_gravity_from_sensor,
)
from unilab.envs.mdp.rewards import action_acc_l2 as action_acc_l2
from unilab.envs.mdp.rewards import action_rate_l2 as action_rate_l2
from unilab.envs.mdp.rewards import (
    body_angular_velocity_penalty as body_angular_velocity_penalty,
)
from unilab.envs.mdp.rewards import flat_orientation_l2 as flat_orientation_l2
from unilab.envs.mdp.rewards import is_alive as is_alive
from unilab.envs.mdp.rewards import is_terminated as is_terminated
from unilab.envs.mdp.rewards import joint_vel_l2 as joint_vel_l2
from unilab.envs.mdp.rewards import track_angular_velocity as track_angular_velocity
from unilab.envs.mdp.rewards import track_linear_velocity as track_linear_velocity
from unilab.envs.mdp.terminations import bad_orientation as bad_orientation
from unilab.envs.mdp.terminations import (
    root_height_below_minimum as root_height_below_minimum,
)
from unilab.envs.mdp.terminations import time_out as time_out

__all__ = [
    "JointPositionAction",
    "JointPositionActionCfg",
    "UniformVelocityCommand",
    "UniformVelocityCommandCfg",
    "action_acc_l2",
    "action_rate_l2",
    "apply_body_impulse",
    "base_ang_vel",
    "base_lin_vel",
    "builtin_sensor",
    "bad_orientation",
    "body_angular_velocity_penalty",
    "dof_armature",
    "flat_orientation_l2",
    "geom_friction",
    "generated_commands",
    "joint_pos_rel",
    "joint_armature",
    "joint_vel_rel",
    "joint_vel_l2",
    "last_action",
    "pd_gains",
    "push_by_setting_velocity",
    "randomize_encoder_bias",
    "randomize_physics_scene_gravity",
    "randomize_rigid_body_com",
    "randomize_rigid_body_mass",
    "is_alive",
    "is_terminated",
    "projected_gravity",
    "projected_gravity_from_sensor",
    "reset_root_state_uniform",
    "reset_scene_to_default",
    "resolve_env_ids",
    "root_height_below_minimum",
    "time_out",
    "track_angular_velocity",
    "track_linear_velocity",
]
