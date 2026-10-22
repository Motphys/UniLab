"""Wire precision, stable storage, and rank-consistent loss flag regressions."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from unilab.rl.runtime.distributed.sync import DpParameterSync


@pytest.mark.parametrize(
    ("option", "count", "dtype", "wire_dtype"),
    [
        ("float32", 1024, torch.float32, torch.float32),
        ("bfloat16", 1024, torch.float32, torch.bfloat16),
        ("bfloat16", 1023, torch.float32, torch.float32),
        ("bfloat16", 1024, torch.float64, torch.float64),
    ],
)
def test_wire_precision_mean_and_stable_storage(option, count, dtype, wire_dtype):
    sync = DpParameterSync(
        world_size=3,
        rank=0,
        rendezvous_path="unused",
        backend="gloo",
        gradient_wire_dtype=option,
    )
    value = torch.linspace(-0.1234567, 0.9876543, count, dtype=dtype)
    parameter = torch.nn.Parameter(torch.zeros_like(value))
    parameter.grad = value.clone()
    pointer = parameter.grad.data_ptr()
    wire_pointers = []

    def reduce(packed, op):
        assert packed.dtype == wire_dtype
        wire_pointers.append(packed.data_ptr())
        packed.add_(value.to(wire_dtype)).add_(value.to(wire_dtype))

    expected = value.to(wire_dtype)
    expected = expected.add(expected).add(value.to(wire_dtype))
    expected = expected.to(dtype).mul(1.0 / 3) if wire_dtype != dtype else expected.div(3)
    with patch("unilab.rl.runtime.distributed.sync.dist.all_reduce", reduce):
        for _ in range(2):
            parameter.grad.copy_(value)
            sync.allreduce_gradients([parameter])
            assert torch.equal(parameter.grad, expected)
    assert parameter.grad.data_ptr() == pointer
    assert wire_pointers[0] == wire_pointers[1]


def test_default_and_invalid_wire_precision():
    assert (
        DpParameterSync(world_size=2, rank=0, rendezvous_path="unused").gradient_wire_dtype
        == "float32"
    )
    with pytest.raises(ValueError, match="gradient_wire_dtype"):
        DpParameterSync(
            world_size=2, rank=0, rendezvous_path="unused", gradient_wire_dtype="float16"
        )


@pytest.mark.parametrize("count", [2, 1025])
def test_remote_loss_flag_remains_positive(count):
    sync = DpParameterSync(
        world_size=8,
        rank=0,
        rendezvous_path="unused",
        backend="gloo",
        gradient_wire_dtype="bfloat16",
    )
    parameter = torch.nn.Parameter(torch.ones(count))
    parameter.grad = torch.zeros(count)

    def reduce(packed, op):
        # A remote rank reports a nonfinite loss with finite derivatives.
        packed[-1] = 1

    with patch("unilab.rl.runtime.distributed.sync.dist.all_reduce", reduce):
        sync.allreduce_gradients([parameter])
    assert parameter.grad[-1].item() == 0.125
