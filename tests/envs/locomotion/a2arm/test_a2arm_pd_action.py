from __future__ import annotations

import numpy as np
import pytest

from unilab.tasks.locomotion.a2arm.actions import A2ArmPdActionCfg


def test_pd_action_config_preserves_17_dof_control_contract() -> None:
    cfg = A2ArmPdActionCfg(entity_name="robot")
    assert len(cfg.actuator_names) == 17
    assert cfg.kp[-5:] == (90.0, 120.0, 70.0, 30.0, 30.0)
    assert cfg.kd[-5:] == (5.5, 10.5, 5.5, 1.0, 1.0)
    np.testing.assert_allclose(cfg.torque_limits[-5:], [30.0, 30.0, 30.0, 10.0, 10.0])


def test_pd_action_requires_state_feedback_on_every_physics_substep() -> None:
    from unilab.tasks.locomotion.a2arm.actions import A2ArmPdAction

    assert A2ArmPdAction.requires_substep_state_feedback is True


@pytest.mark.parametrize("field", ["action_scale", "kp", "kd", "torque_limits", "motor_strength"])
def test_pd_action_rejects_partial_parameter_vectors(field: str) -> None:
    from unilab.tasks.locomotion.a2arm.actions import A2ArmPdAction

    cfg = A2ArmPdActionCfg(entity_name="robot", **{field: (1.0,)})
    with pytest.raises(ValueError, match=f"{field}.*17"):
        A2ArmPdAction._validate_cfg(cfg)
