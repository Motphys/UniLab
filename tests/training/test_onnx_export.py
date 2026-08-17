"""Tests for the shared ONNX export / verification helpers."""

from __future__ import annotations

import onnx
import torch

from unilab.training.onnx_export import export_policy_onnx, verify_policy_onnx


class _TinyActor(torch.nn.Module):
    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


class _PrivInfoActor(torch.nn.Module):
    def __init__(self, obs_dim: int = 4, priv_info_dim: int = 3, action_dim: int = 2) -> None:
        super().__init__()
        self.obs_proj = torch.nn.Linear(obs_dim, action_dim)
        self.priv_proj = torch.nn.Linear(priv_info_dim, action_dim)

    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        return self.obs_proj(obs) + self.priv_proj(priv_info)


class _TupleOutputActor(torch.nn.Module):
    def __init__(self, obs_dim: int = 4, action_dim: int = 2) -> None:
        super().__init__()
        self.mlp = torch.nn.Linear(obs_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.mlp(obs)
        return action, action


def test_export_policy_onnx_writes_expected_graph(tmp_path, capsys):
    torch.manual_seed(0)
    module = _TinyActor()
    onnx_path = str(tmp_path / "policy.onnx")

    export_policy_onnx(module, onnx_path, (torch.randn(1, 4),), input_names=["obs"])

    assert f"Exported actor ONNX to {onnx_path}" in capsys.readouterr().out
    model = onnx.load(onnx_path)
    assert [graph_input.name for graph_input in model.graph.input] == ["obs"]
    assert [graph_output.name for graph_output in model.graph.output] == ["action"]
    opsets = {opset.version for opset in model.opset_import if opset.domain in ("", "ai.onnx")}
    assert opsets == {17}


def test_verify_policy_onnx_matches_pytorch(tmp_path, capsys):
    torch.manual_seed(0)
    module = _TinyActor()
    onnx_path = str(tmp_path / "policy.onnx")
    export_policy_onnx(module, onnx_path, (torch.randn(1, 4),), input_names=["obs"])
    capsys.readouterr()

    max_diff, mean_diff = verify_policy_onnx(
        module, onnx_path, (torch.randn(1, 4),), input_names=["obs"]
    )

    out = capsys.readouterr().out
    assert isinstance(max_diff, float)
    assert isinstance(mean_diff, float)
    assert max_diff <= 1e-4
    assert 0.0 <= mean_diff <= max_diff
    assert "ONNX export verified OK." in out
    assert "WARNING" not in out


def test_verify_policy_onnx_warns_on_divergence(tmp_path, capsys):
    torch.manual_seed(0)
    exported = _TinyActor()
    onnx_path = str(tmp_path / "policy.onnx")
    export_policy_onnx(exported, onnx_path, (torch.randn(1, 4),), input_names=["obs"])
    capsys.readouterr()

    diverged = _TinyActor()
    with torch.no_grad():
        for param in diverged.parameters():
            param.add_(torch.randn_like(param) * 10.0)

    max_diff, _ = verify_policy_onnx(diverged, onnx_path, (torch.randn(1, 4),), input_names=["obs"])

    assert max_diff > 1e-4
    assert "WARNING: ONNX output diverges from PyTorch!" in capsys.readouterr().out


def test_verify_policy_onnx_supports_extra_named_inputs(tmp_path):
    torch.manual_seed(0)
    module = _PrivInfoActor()
    onnx_path = str(tmp_path / "policy.onnx")
    export_policy_onnx(
        module,
        onnx_path,
        (torch.randn(1, 4), torch.zeros(1, 3)),
        input_names=["obs", "priv_info"],
    )
    model = onnx.load(onnx_path)
    assert [graph_input.name for graph_input in model.graph.input] == ["obs", "priv_info"]

    max_diff, _ = verify_policy_onnx(
        module,
        onnx_path,
        (torch.randn(1, 4), torch.zeros(1, 3)),
        input_names=["obs", "priv_info"],
    )
    assert max_diff <= 1e-4


def test_verify_policy_onnx_compares_first_element_of_tuple_output(tmp_path, capsys):
    torch.manual_seed(0)
    module = _TupleOutputActor()
    onnx_path = str(tmp_path / "policy.onnx")
    export_policy_onnx(
        module,
        onnx_path,
        (torch.randn(1, 4),),
        input_names=["obs"],
        output_names=["action", "aux"],
    )

    max_diff, _ = verify_policy_onnx(module, onnx_path, (torch.randn(1, 4),), input_names=["obs"])
    assert max_diff <= 1e-4
    assert "ONNX export verified OK." in capsys.readouterr().out
