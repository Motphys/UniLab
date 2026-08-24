from __future__ import annotations

import numpy as np

from unilab.tasks.locomotion.a2arm.state import A2ArmTeleopCommand
from unilab.visualization.a2arm_pos_force import (
    TeleopState,
    make_key_callback,
)


def test_teleop_state_clamps_velocity_and_spherical_goal() -> None:
    teleop = TeleopState(
        velocity_low=np.asarray([-0.5, -0.4, -0.8]),
        velocity_high=np.asarray([1.0, 0.4, 0.8]),
        sphere_low=np.asarray([0.2, -1.0, -1.2]),
        sphere_high=np.asarray([0.6, 1.0, 1.2]),
        ee_init=np.asarray([0.4, 0.3, 0.0]),
    )

    teleop.nudge_velocity(0, 100.0)
    teleop.nudge_velocity(1, -100.0)
    teleop.nudge_sphere(2, 100.0)

    assert teleop.velocity.tolist() == [1.0, -0.4, 0.0]
    assert teleop.ee_sphere.tolist() == [0.4, 0.3, 1.2]


def test_teleop_force_episode_ramps_holds_and_releases() -> None:
    teleop = TeleopState(
        velocity_low=np.full(3, -1.0),
        velocity_high=np.full(3, 1.0),
        sphere_low=np.asarray([0.2, -1.0, -1.0]),
        sphere_high=np.asarray([0.6, 1.0, 1.0]),
        ee_init=np.asarray([0.4, 0.3, 0.0]),
        ee_ramp=2,
        ee_hold=1,
        base_ramp=2,
        base_hold=1,
        impulse_ee_n=10.0,
        impulse_base_n=20.0,
    )

    teleop.push_ee(0, 1.0)
    values = []
    for _ in range(7):
        teleop.advance_forces()
        values.append(float(teleop.ee_force[0]))
    assert values == [0.0, 5.0, 10.0, 10.0, 5.0, 0.0, 0.0]

    teleop.toggle_hold()
    teleop.push_base(1, -1.0)
    for _ in range(4):
        teleop.advance_forces()
    assert teleop.base_force[1] == -20.0
    teleop.toggle_hold()
    teleop.advance_forces()
    teleop.advance_forces()
    assert abs(float(teleop.base_force[1])) < 20.0
    teleop.clear_forces()
    assert np.all(teleop.ee_force == 0.0)
    assert np.all(teleop.base_force == 0.0)


def test_teleop_command_is_typed_and_batch_ready() -> None:
    teleop = TeleopState(
        velocity_low=np.full(3, -1.0),
        velocity_high=np.full(3, 1.0),
        sphere_low=np.asarray([0.2, -1.0, -1.0]),
        sphere_high=np.asarray([0.6, 1.0, 1.0]),
        ee_init=np.asarray([0.4, 0.3, 0.0]),
    )
    teleop.nudge_velocity(0, 0.1)
    command = teleop.as_command()
    assert isinstance(command, A2ArmTeleopCommand)
    np.testing.assert_allclose(command.velocity, [[0.1, 0.0, 0.0]])
    np.testing.assert_allclose(command.ee_sphere, [[0.4, 0.3, 0.0]])
    assert command.ee_force.shape == (1, 3)


def test_key_callback_preserves_pause_reset_and_force_bindings() -> None:
    teleop = TeleopState(
        velocity_low=np.full(3, -1.0),
        velocity_high=np.full(3, 1.0),
        sphere_low=np.asarray([0.2, -1.0, -1.0]),
        sphere_high=np.asarray([0.6, 1.0, 1.0]),
        ee_init=np.asarray([0.4, 0.3, 0.0]),
    )
    events: list[str] = []
    callback = make_key_callback(
        teleop,
        on_pause=lambda: events.append("pause"),
        on_reset=lambda: events.append("reset"),
        on_toggle_range=lambda: events.append("range"),
    )

    callback(ord("w"))
    callback(ord("g"))
    callback(ord(" "))
    callback(259)
    callback(262)
    callback(ord("f"))

    assert teleop.velocity[0] == 0.1
    assert events == ["range", "pause", "reset"]
    assert np.all(teleop.ee_force == 0.0)
