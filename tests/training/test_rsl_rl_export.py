"""Executable validation for RSL-RL policy export artifacts."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from unilab.training.entrypoints import EntrypointContractError
from unilab.training.rsl_rl import validate_rsl_rl_policy_exports


class _Policy:
    obs_dim = 4

    def __init__(self, *, weight_scale: float = 1.0) -> None:
        self._module = torch.nn.Linear(self.obs_dim, 2)
        with torch.no_grad():
            self._module.weight.copy_(
                weight_scale
                * torch.tensor(
                    ((0.5, -0.25, 0.75, 0.1), (-0.4, 0.2, 0.3, -0.6)),
                    dtype=torch.float32,
                )
            )
            self._module.bias.copy_(torch.tensor((0.1, -0.2), dtype=torch.float32))

    def as_jit(self) -> torch.nn.Module:
        return deepcopy(self._module)


def _export(policy: _Policy, root: Path) -> dict[str, Path]:
    sample = torch.zeros((1, policy.obs_dim), dtype=torch.float32)
    jit_path = root / "policy.pt"
    onnx_path = root / "policy.onnx"
    module = policy.as_jit().eval()
    torch.jit.script(module).save(str(jit_path))
    torch.onnx.export(
        module,
        sample,
        onnx_path,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
    )
    return {"onnx": onnx_path, "jit": jit_path}


def test_rsl_rl_export_validation_reloads_jit_and_onnx(tmp_path: Path) -> None:
    policy = _Policy()
    artifacts = _export(policy, tmp_path)

    assert validate_rsl_rl_policy_exports(policy=policy, artifacts=artifacts) == (
        artifacts["onnx"],
        artifacts["jit"],
    )


def test_rsl_rl_export_validation_rejects_numerically_foreign_jit(tmp_path: Path) -> None:
    policy = _Policy()
    artifacts = _export(policy, tmp_path)
    torch.jit.script(_Policy(weight_scale=2.0).as_jit()).save(str(artifacts["jit"]))

    with pytest.raises(EntrypointContractError, match="TorchScript.*parity"):
        validate_rsl_rl_policy_exports(policy=policy, artifacts={"jit": artifacts["jit"]})


def test_rsl_rl_export_validation_rejects_numerically_foreign_onnx(tmp_path: Path) -> None:
    policy = _Policy()
    artifacts = _export(_Policy(weight_scale=2.0), tmp_path)

    with pytest.raises(EntrypointContractError, match="ONNX.*parity"):
        validate_rsl_rl_policy_exports(policy=policy, artifacts={"onnx": artifacts["onnx"]})
