from __future__ import annotations

import numpy as np

from unilab.tasks.locomotion.a2arm.actions import A2ArmPdActionCfg


def test_pd_action_config_preserves_17_dof_control_contract() -> None:
    cfg = A2ArmPdActionCfg(entity_name="robot")
    assert len(cfg.actuator_names) == 17
    assert cfg.kp[-5:] == (90.0, 120.0, 70.0, 30.0, 30.0)
    assert cfg.kd[-5:] == (5.5, 10.5, 5.5, 1.0, 1.0)
    np.testing.assert_allclose(cfg.torque_limits[-5:], [30.0, 30.0, 30.0, 10.0, 10.0])
