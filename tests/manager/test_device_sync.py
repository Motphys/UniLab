from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import torch

from unilab.manager.device_runtime import DeviceManagedRuntime
from unilab.manager.device_sync import DeviceRuntimeSynchronization


def test_device_runtime_sync_primitives_are_created_once_and_reused() -> None:
    task_stream = MagicMock(name="task_stream")
    producer_stream = MagicMock(name="producer_stream")
    events = [MagicMock(name=f"event_{index}") for index in range(5)]

    with (
        patch("unilab.manager.device_sync.torch.cuda.Stream", return_value=task_stream) as stream,
        patch("unilab.manager.device_sync.torch.cuda.Event", side_effect=events) as event,
        patch(
            "unilab.manager.device_sync.torch.cuda.current_stream",
            return_value=producer_stream,
        ),
        patch("unilab.manager.device_sync.torch.cuda.stream", return_value=nullcontext()),
        patch.object(torch.Tensor, "record_stream", autospec=True) as record_stream,
    ):
        synchronization = DeviceRuntimeSynchronization.create(torch.device("cuda:0"))
        identities = tuple(
            id(value)
            for value in (
                synchronization.task_stream,
                synchronization.control_event,
                synchronization.reset_event,
                synchronization.output_event,
                synchronization.episode_length_input_event,
                synchronization.cold_init_event,
            )
        )
        synchronization.publish_cold_initialization()

        first = torch.arange(4, dtype=torch.int64)
        second = torch.arange(4, dtype=torch.int64) + 4
        target = torch.empty((4,), dtype=torch.int64)
        synchronization.copy_episode_lengths(values=first, target=target)
        synchronization.copy_episode_lengths(values=second, target=target)

        assert (
            tuple(
                id(value)
                for value in (
                    synchronization.task_stream,
                    synchronization.control_event,
                    synchronization.reset_event,
                    synchronization.output_event,
                    synchronization.episode_length_input_event,
                    synchronization.cold_init_event,
                )
            )
            == identities
        )
        stream.assert_called_once_with(device=torch.device("cuda:0"))
        assert event.call_count == 5

    synchronization.cold_init_event.record.assert_called_once_with(producer_stream)
    synchronization.task_stream.wait_event.assert_any_call(synchronization.cold_init_event)
    assert synchronization.episode_length_input_event.record.call_count == 2
    assert synchronization.task_stream.wait_event.call_count == 3
    assert record_stream.call_count == 2
    assert all(call.args == (task_stream,) for call in record_stream.call_args_list)
    torch.testing.assert_close(target, second)


def test_episode_length_setter_delegates_to_sync_owner() -> None:
    runtime = object.__new__(DeviceManagedRuntime)
    runtime._device = torch.device("cpu")
    runtime._num_envs = 4
    runtime._episode_steps = torch.empty((4,), dtype=torch.int64)
    runtime._synchronization = MagicMock(spec=DeviceRuntimeSynchronization)
    values = torch.arange(4, dtype=torch.int64)

    runtime.set_episode_length_buffer(values)

    runtime._synchronization.copy_episode_lengths.assert_called_once_with(
        values=values,
        target=runtime._episode_steps,
    )
