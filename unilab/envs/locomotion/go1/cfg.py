from dataclasses import dataclass, field, asdict

from unilab import ROOT_PATH
from unilab.envs import registry
from unilab.envs.base import EnvCfg

model_file = (ROOT_PATH / "models" / "robots" / "locomotion" / "go1" / "scene_mjx_feetonly_flat_terrain.xml").as_posix()

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale: float = 0.5
    Kp: float = 35.0
    Kd: float = 0.5


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.278]


@dataclass
class Commands:
    vel_limit = [
        [-1.5, -0.8, -1.2],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [ 1.5,  0.8,  1.2],  # max
    ]


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            # Tracking
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            # Base
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "orientation": -5.0,
            # "base_height": -5.0,
            # Other
            "dof_pos_limits": -1.0,
            "pose": 0.5,
            "termination": -1.0,
            "stand_still": -1.0,
            # Regularization
            "torques": -0.0002,
            "action_rate": -0.01,
            "energy": -0.001,
            # Feet
            "feet_clearance": -2.0,
            "feet_height": -0.2,
            "feet_slip": -0.1,
            "feet_air_time": 0.1,
        }
    )

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1

@dataclass
class Asset:
    body_name = "trunk"
    foot_name = "foot"
    ground = "floor"

@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class RslPolicyCfg:
    init_noise_std: float = 1.0
    actor_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    class_name: str = "ActorCritic"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True


@dataclass
class RslAlgorithmCfg:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    class_name: str = "PPO"


@dataclass
class RslRunnerCfg:
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 50
    experiment_name: str = "go1_walk_rsl"
    run_name: str = "test_run"
    resume: bool = False
    load_run: int = -1
    checkpoint: int = -1
    resume_path: str = None


@dataclass
class RslTrainCfg:
    policy: RslPolicyCfg = field(default_factory=RslPolicyCfg)
    algorithm: RslAlgorithmCfg = field(default_factory=RslAlgorithmCfg)
    runner: RslRunnerCfg = field(default_factory=RslRunnerCfg)
    obs_groups: dict = field(default_factory=lambda: {"policy": ["policy"]})

@registry.envcfg("go1-flat-terrain-walk")
@dataclass
class Go1WalkNpEnvCfg(EnvCfg):
    train_cfg: RslTrainCfg = field(default_factory=RslTrainCfg)
    max_episode_seconds: float = 20.0
    model_file: str = model_file
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    # normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.004
    ctrl_dt: float = 0.02
