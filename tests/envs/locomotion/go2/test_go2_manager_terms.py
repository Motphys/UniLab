"""Go2 owner reward-pack tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from unisim.backend.base import BackendSensorView

from unilab.envs.manager_based_rl_env import ManagerBasedRlEnv
from unilab.tasks.locomotion.go2.manager_terms import Go2WalkRewardPack, Go2WalkRewardPackCfg


class _Scene:
    def __init__(self) -> None:
        self.values = {
            "gyro": np.array([[0.1, -0.2, 0.4], [-0.3, 0.2, -0.1]], dtype=np.float32),
            "FL_foot_contact": np.array([[0.2], [0.0]], dtype=np.float32),
            "FR_foot_contact": np.array([[0.0, 0.9, 0.9], [0.2, 0.0, 0.0]], dtype=np.float32),
            "RL_foot_contact": np.array([[0.0], [0.2]], dtype=np.float32),
            "RR_foot_contact": np.array([[0.2, 0.0, 0.0], [0.0, 0.9, 0.9]], dtype=np.float32),
            "FL_pos": np.array([[0, 0, 0.1], [0, 0, 0.0]], dtype=np.float32),
            "FR_pos": np.array([[0, 0, 0.2], [0, 0, 0.1]], dtype=np.float32),
            "RL_pos": np.array([[0, 0, 0.1], [0, 0, 0.2]], dtype=np.float32),
            "RR_pos": np.array([[0, 0, 0.0], [0, 0, 0.1]], dtype=np.float32),
        }
        tensors = {name: torch.as_tensor(value) for name, value in self.values.items()}
        self._tensor_read_plan = SimpleNamespace(
            device=torch.device("cpu"),
            sensor_names={"robot": tuple(self.values)},
            sensor_tensor_views=lambda entity, names: SimpleNamespace(
                values={name: tensors[name] for name in names}
            ),
            body_tensor_view=lambda entity, names: SimpleNamespace(
                pos_w=torch.tensor([[[1.0, 2.0, 0.4]], [[-1.0, 0.5, 0.2]]]),
                quat_w=torch.tensor(
                    [[[0.70710678, 0.0, 0.0, -0.70710678]], [[1.0, 0.0, 0.0, 0.0]]]
                ),
                lin_vel_w=torch.tensor([[[0.0, 2.0, 3.0]], [[0.3, -0.2, 0.5]]]),
            ),
            joint_tensor_view=lambda entity: SimpleNamespace(
                joint_pos=torch.tensor([[0.2, -0.1, 0.5], [-0.4, 0.3, 0.1]])
            ),
        )
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                default_joint_pos_torch=lambda device: torch.tensor(
                    [[0.1, -0.2, 0.5], [-0.1, 0.1, 0.0]]
                )
            )
        )

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot

    def bind_sensor_data(self, names) -> BackendSensorView:
        names = tuple(names)
        dimensions = tuple(self.values[name].shape[1] for name in names)

        def read() -> np.ndarray:
            return np.concatenate([self.values[name] for name in names], axis=1)

        view = BackendSensorView("fake", names, dimensions, 2, read)
        view.read()
        return view


def _env() -> ManagerBasedRlEnv:
    command = torch.tensor([[0.5, -0.2, 0.3], [-0.1, 0.4, -0.2]], dtype=torch.float32)
    return cast(
        ManagerBasedRlEnv,
        SimpleNamespace(
            num_envs=2,
            device=torch.device("cpu"),
            step_dt=0.02,
            common_step_counter=5,
            scene=_Scene(),
            command_manager=SimpleNamespace(get_command=lambda name: command),
            action_manager=SimpleNamespace(
                action=torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32),
                prev_action=torch.tensor([[0.1, 0.1], [0.0, -0.2]], dtype=torch.float32),
            ),
        ),
    )


def test_go2_walk_reward_pack_matches_component_equations() -> None:
    env = _env()
    cfg = Go2WalkRewardPackCfg(func=Go2WalkRewardPack, weight=0.0)
    pack = Go2WalkRewardPack(cfg, env)
    actual = pack(cast(Any, env))
    root_velocity_b = torch.tensor([[-2.0, 0.0, 3.0], [0.3, -0.2, 0.5]])
    command = torch.tensor([[0.5, -0.2, 0.3], [-0.1, 0.4, -0.2]])
    gyro = torch.tensor([[0.1, -0.2, 0.4], [-0.3, 0.2, -0.1]])
    phase = torch.tensor([0.0, 0.5, 0.5, 0.0])
    contact = torch.tensor([[True, False, False, True], [False, True, True, False]])
    heights = torch.tensor([[0.1, 0.2, 0.1, 0.0], [0.0, 0.1, 0.2, 0.1]])
    expected = torch.stack(
        (
            torch.exp(-(command[:, :2] - root_velocity_b[:, :2]).square().sum(dim=1) / 0.25),
            torch.exp(-(command[:, 2] - gyro[:, 2]).square() / 0.25),
            root_velocity_b[:, 2].square(),
            gyro[:, :2].square().sum(dim=1),
            torch.tensor([0.01, 0.01]),
            torch.tensor([0.05, 0.45]),
            torch.tensor([0.2, 0.6]),
            (contact == (phase < 0.6)).float().mean(dim=1),
            (torch.exp(-(heights - 0.1).square() / 0.01) * (phase >= 0.6).float()).mean(dim=1),
        ),
        dim=1,
    )
    torch.testing.assert_close(actual, expected)
