"""Real-CUDA producer/physics/consumer stream-contract tests."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch
from tests.training.test_device_transition_abi import _control_batch, _fixture

from unilab.base.backend import (
    BufferView,
    ControlBatch,
    DeviceBufferContractError,
    DeviceBufferLease,
    DeviceCompletion,
    DeviceTensorView,
    RowSelection,
)

pytestmark = pytest.mark.slow


def test_missing_completion_event_is_detected() -> None:
    """The backend must reject an action with no producer event, not sync it."""

    fixture = _fixture(128)
    control, _ = _control_batch(fixture, completion=None)
    with pytest.raises(DeviceBufferContractError, match="no producer completion"):
        fixture.backend.step_batch(fixture.plan, control, nsteps=1)


def test_explicit_producer_and_consumer_stream_handoff_has_no_global_sync() -> None:
    """Physics waits for the policy event and exposes a consumer event in turn."""

    fixture = _fixture(128)
    producer_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device="cuda:0"))
    action = torch.empty((128, 29), dtype=torch.float32, device="cuda:0")
    action_lease = DeviceBufferLease("stream-test-policy")
    with torch.cuda.stream(producer_stream):
        action.fill_(0.05)
        # Keep the producer visibly asynchronous relative to the caller; the
        # test is about event ordering, not a host-side timing estimate.
        torch.cuda._sleep(5_000_000)
        producer = DeviceCompletion.record(
            placement=fixture.placement,
            owner_id="stream-test-policy",
            epoch=action_lease.epoch,
            stream=producer_stream,
        )
    action_view = DeviceTensorView(
        tensor_handle=action,
        contract=fixture.control_contract,
        lease=action_lease,
        completion=producer,
    )
    control = ControlBatch(
        plan=fixture.plan,
        rows=RowSelection.all(128),
        buffer=BufferView(
            handle=action_view,
            shape=(128, 29),
            contract=fixture.control_contract,
        ),
    )

    # A global synchronization is a contract violation.  The backend is
    # allowed only event waits on its dedicated stream.
    with patch("torch.cuda.synchronize", side_effect=AssertionError("global sync is forbidden")):
        result = fixture.backend.step_batch(fixture.plan, control, nsteps=1)

    completion = result.diagnostics.completion_event
    assert completion is not None
    backend_completion = completion.handle
    assert isinstance(backend_completion, DeviceCompletion)

    consumer_stream = cast(torch.cuda.Stream, torch.cuda.Stream(device="cuda:0"))
    state_view = result.terminal_state.buffer("root.position").handle
    assert isinstance(state_view, DeviceTensorView)
    received = torch.empty((128, 3), dtype=torch.float32, device="cuda:0")
    with torch.cuda.stream(consumer_stream):
        backend_completion.wait(consumer_stream)
        received.copy_(state_view.torch(), non_blocking=True)
    # Test-only stream synchronization observes completion; production code
    # receives the event and continues asynchronously.
    consumer_stream.synchronize()
    assert torch.isfinite(received).all().item()
