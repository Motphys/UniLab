"""Check nonzero joint targets against actual actuator storage without stepping physics."""

import argparse
import json

import torch
from train_rma import build_environment
from unilab_rma import RMAConfig
from unilab_rma.evaluation_diagnostics import FORWARD_TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    raw, env = build_environment(2, RMAConfig(), "cpu", 123, task=FORWARD_TASK)
    try:
        env.reset()
        robot = raw.scene["robot"]
        joints = list(robot.joint_names)
        actuator_joints = list(raw.backend.get_actuator_joint_names())
        indices = [actuator_joints.index(name) for name in joints]
        control = robot.data.control_buffer
        if not isinstance(control, torch.Tensor) or len(joints) != 12:
            raise ValueError("Expected the 12-joint Torch Go2 control plane")
        errors, equivalence = [], []
        for after_reset in (False, True):
            if after_reset:
                raw.reset(torch.tensor([0], device=raw.device))
            saved = control.clone()
            try:
                for column in range(12):
                    target = torch.zeros((2, 12), dtype=control.dtype, device=control.device)
                    target[0, column] = 0.1
                    target[1, column] = -0.1
                    robot.set_full_width_joint_position_target(target)
                    actual = control.clone()
                    errors.append((actual[:, indices] - target).abs().max().item())
                    robot.set_joint_position_target(target)
                    equivalence.append((actual - control).abs().max().item())
            finally:
                control.copy_(saved)
        report = dict(
            cases=len(errors),
            num_envs=2,
            partial_reset_checked=True,
            max_abs_mapping_error_rad=max(errors),
            max_abs_fast_vs_ordinary_error_rad=max(equivalence),
            physics_steps=0,
        )
        print(json.dumps(report, indent=2))
        if max(errors + equivalence) > 1e-6:
            parser.exit(1, "Mapping check FAILED; do not start a new training run.\n")
    finally:
        raw.close()


if __name__ == "__main__":
    main()
