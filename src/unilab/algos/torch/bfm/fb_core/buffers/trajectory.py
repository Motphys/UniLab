import dataclasses
import typing
from collections.abc import Mapping
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils._pytree import tree_map

from .transition import DictBuffer, _to_torch

INT_CLASSES_TYPING = Union[int, np.integer]
INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)


def _is_int(index):
    if isinstance(index, INT_CLASSES):
        return True
    if isinstance(index, (np.ndarray, torch.Tensor)):
        return index.ndim == 0
    return False


def tree_concat(list_of_pytree_of_tensors, dim=0):

    tds = tuple(
        map(lambda x: TensorDict.from_pytree(x, auto_batch_size=True), list_of_pytree_of_tensors)
    )
    concatenated = torch.cat(tds, dim=dim)

    if isinstance(concatenated, TensorDict):
        return concatenated.to_pytree()

    return concatenated


class TrajectoryDictBuffer:
    def __init__(
        self,
        episodes: List[dict],
        device: str = "cpu",
        seq_length: int = 1,
        output_key_t: List[str] = ["observation"],
        output_key_tp1: List[str] = ["observation"],
        end_key: Tuple[str] | str = "done",
        motion_id_key: Tuple[str] | str = "motion_id",
    ) -> None:
        self._is_full = True
        self.output_key_t = output_key_t
        self.output_key_tp1 = output_key_tp1
        self.seq_length = seq_length
        self.device = device
        self.end_key = end_key
        self.motion_id_key = motion_id_key

        def add_new_data(data, storage):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    if k not in storage:
                        storage[k] = {}
                    add_new_data(v, storage=storage[k])
                else:
                    if k not in storage:
                        storage[k] = []
                    storage[k].append(_to_torch(v, device=self.device))

        def concat_dict(storage):
            for k, v in storage.items():
                if isinstance(v, Mapping):
                    concat_dict(storage[k])
                else:
                    storage[k] = torch.cat(v, axis=0)

        self.storage = {}
        set_key(self.storage, self.end_key, [])
        self.motion_ids = []
        for ep in episodes:
            _k = [k for k, v in ep.items() if not isinstance(v, Mapping)][0]

            add_new_data(ep, self.storage)

            if not key_exists(ep, self.end_key):
                episode_start = torch.zeros(
                    (ep[_k].shape[0], 1), dtype=torch.bool, device=self.device
                )
                episode_start[-1] = True
                get_key(self.storage, self.end_key).append(episode_start)
            self.motion_ids.append(get_key(ep, self.motion_id_key)[0].item())

        concat_dict(self.storage)
        assert get_key(self.storage, self.end_key).dtype == torch.bool
        done = get_key(self.storage, self.end_key)

        self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
            done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=None
        )

        self.priorities = torch.ones(
            len(self.lengths), device=self.device, dtype=torch.float32
        ) / len(self.lengths)
        self._get_idxs = get_idxs

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )
        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        idxs = self._get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=self.lengths[traj_idx],
            start_idx=self.start_idx[traj_idx],
            storage_length=self.capacity,
            priorities=self.priorities[traj_idx],
        )
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = tree_map(lambda x: x[idxs], self.storage[k])

        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            output["next"][k] = tree_map(lambda x: x[idxs], self.storage[k])
        return output

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor) -> None:

        assert len(priorities) == len(self.priorities)
        self.priorities[idxs] = priorities
        self.priorities = self.priorities / torch.sum(self.priorities)

    @property
    def capacity(self):
        return get_key(self.storage, self.end_key).shape[0]

    def __len__(self) -> int:
        return self.capacity

    def empty(self) -> bool:
        return len(self) == 0


@dataclasses.dataclass(kw_only=True)
class TrajectoryDictBufferMultiDim(DictBuffer):
    n_dim: int = 1
    seq_length: int = 1
    output_key_t: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    output_key_tp1: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    end_key: Tuple[str] | str = "done"

    def __post_init__(self) -> None:
        self.storage = None
        self._idx = 0
        self._is_full = False
        self._recompute_start_stop = True
        self._get_idxs = get_idxs
        assert self.n_dim == 1 or self.n_dim == 2, (
            "n_dim must be either 1 or 2 for TrajectoryDictBufferMultiDim"
        )

    def _ndim(self) -> int:
        return self.n_dim

    def size(self):
        return len(self) * self.storage[self.end_key].shape[1]

    @torch.no_grad
    def extend(self, data: Dict) -> None:

        super().extend(data)
        self._recompute_start_stop = True

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if self._recompute_start_stop:
            done = get_key(self.storage, self.end_key)
            self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
                done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=self._idx - 1
            )
            self._recompute_start_stop = False

        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        num_slices = batch_size // seq_length
        traj_idx = self.lengths >= (seq_length + offset)
        if not traj_idx.any():
            raise ValueError(
                f"No trajectories with length >= {seq_length + offset} to sample {num_slices} slices."
            )
        idxs = self._get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=self.lengths[traj_idx],
            start_idx=self.start_idx[traj_idx],
            storage_length=self.capacity,
            priorities=None,
        )
        idxs = idxs.to(torch.long).unbind(-1)
        for k in self.output_key_t:
            output[k] = tree_map(lambda x: x[idxs], self.storage[k])

        idxs = ((idxs[0] + 1) % self.capacity, *idxs[1:])
        for k in self.output_key_tp1:
            output["next"][k] = tree_map(lambda x: x[idxs], self.storage[k])
        return output

    def get_full_buffer(self) -> Dict:

        if self._recompute_start_stop:
            done = get_key(self.storage, self.end_key)
            self.start_idx, self.stop_idx, self.lengths = find_start_stop_traj(
                done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=self._idx - 1
            )
            self._recompute_start_stop = False

        output, offset = {}, 0
        if len(self.output_key_tp1) > 0:
            output["next"] = {}
            offset = 1
        for start, length in zip(self.start_idx, self.lengths):
            idxs = (start[0] + torch.arange(length - offset, device=self.device)) % self.capacity
            for k in self.output_key_t:
                if k not in output:
                    output[k] = []
                output[k].append(tree_map(lambda x: x[idxs, start[1]], self.storage[k]))
            idxs = (
                start[0] + 1 + torch.arange(length - offset, device=self.device)
            ) % self.capacity
            for k in self.output_key_tp1:
                if k not in output["next"]:
                    output["next"][k] = []
                output["next"][k].append(tree_map(lambda x: x[idxs, start[1]], self.storage[k]))
        for k in self.output_key_t:
            output[k] = tree_concat(output[k])
        for k in self.output_key_tp1:
            output["next"][k] = tree_concat(output["next"][k])
        return output


def get_idxs(
    seq_length,
    num_slices,
    lengths,
    start_idx,
    storage_length,
    priorities: torch.Tensor | None = None,
):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    end_point = lengths[traj_idx] - seq_length - 1
    relative_starts = (
        (torch.rand(num_slices, device=lengths.device) * end_point).floor().to(start_idx.dtype)
    )
    starts = torch.cat(
        [
            (start_idx[traj_idx, 0] + relative_starts).unsqueeze(1),
            start_idx[traj_idx, 1:],
        ],
        1,
    )
    idxs = _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)
    return idxs


def find_start_stop_traj(end: torch.Tensor, at_capacity: bool = True, cursor=None | int):
    length = end.shape[0]
    if not at_capacity:
        end = torch.index_fill(
            end,
            index=torch.tensor(length - 1, device=end.device, dtype=torch.long),
            dim=0,
            value=1,
        )
    else:
        if cursor is not None:
            if not _is_int(cursor):
                raise ValueError("cursor must be an int")
            end = torch.index_fill(
                end,
                index=torch.tensor(cursor, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if not end.any(0).all():
            mask = ~end.any(0, True)
            mask = torch.cat([torch.zeros_like(end[:-1]), mask])
            end = torch.masked_fill(mask, end, 1)
    return _end_to_start_stop(length=length, end=end)


def _end_to_start_stop(end, length):

    stop_idx = end.transpose(0, -1).nonzero()
    stop_idx[:, [0, -1]] = stop_idx[:, [-1, 0]].clone()

    start_idx = stop_idx.clone()
    start_idx[:, 0] += 1
    start_idx[:, 0] %= end.shape[0]

    if start_idx.shape[0] > 1:
        start_idx_mask = (start_idx[1:, 1:] == start_idx[:-1, 1:]).all(-1)
        m1 = torch.cat([torch.zeros_like(start_idx_mask[:1]), start_idx_mask])
        m2 = torch.cat([start_idx_mask, torch.zeros_like(start_idx_mask[:1])])
        start_idx_replace = torch.empty_like(start_idx)
        start_idx_replace[m1] = start_idx[m2]
        start_idx_replace[~m1] = start_idx[~m2]
        start_idx = start_idx_replace
    else:
        pass
    lengths = stop_idx[:, 0] - start_idx[:, 0] + 1
    lengths[lengths <= 0] = lengths[lengths <= 0] + length
    return start_idx, stop_idx, lengths


def _tensor_slices_from_startend(seq_length, starts, storage_length):
    arange = torch.arange(seq_length, device=starts.device, dtype=starts.dtype)
    ndims = starts.shape[-1] - 1 if (starts.ndim - 1) else 0
    if ndims:
        arange_reshaped = torch.empty(
            list(arange.shape) + [ndims + 1], device=starts.device, dtype=starts.dtype
        )
        arange_reshaped[..., 0] = arange
        arange_reshaped[..., 1:] = 0
    else:
        arange_reshaped = arange.unsqueeze(-1)
    arange_expanded = arange_reshaped.expand([starts.shape[0]] + list(arange_reshaped.shape))
    n_missing_dims = arange_expanded.dim() - starts.dim()
    start_expanded = starts[(slice(None),) + (None,) * n_missing_dims].expand_as(arange_expanded)
    result = (start_expanded + arange_expanded).flatten(0, 1)
    result[:, 0] = result[:, 0] % storage_length
    return result


def key_exists(data, key):
    if isinstance(key, str):
        return key in data.keys()
    else:
        if len(key) == 1:
            return key[0] in data.keys()
        if key[0] not in data.keys():
            return False
        return key_exists(data[key[0]], key[1:])


def set_key(data, key, value):
    if isinstance(key, str):
        data[key] = value
    elif len(key) == 1:
        data[key[0]] = value
    else:
        set_key(data, key[1:], value)


def get_key(data, key):
    if isinstance(key, str):
        return data[key]
    elif len(key) == 1:
        return data[key[0]]
    else:
        return get_key(data, key[1:])
