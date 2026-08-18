from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import numbers
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch

Device = Union[str, torch.device]


@functools.singledispatch
def _to_torch(value: Any, device: Device | None = None) -> Any:
    raise Exception(
        f"No known conversion for type ({type(value)}) to PyTorch registered. Report as issue on github."
    )


@_to_torch.register(numbers.Number)
@_to_torch.register(np.ndarray)
def _np_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    tensor = torch.tensor(value)
    if device:
        return tensor.to(device=device)
    return tensor


@_to_torch.register(torch.Tensor)
def _torch_to_torch(value: np.ndarray, device: Device | None = None) -> torch.Tensor:
    tensor = value.clone().detach()
    if device:
        return tensor.to(device=device)
    return tensor


@dataclasses.dataclass(kw_only=True)
class DictBuffer:
    capacity: int
    device: str = "cpu"
    nested_key_separator: str = "-"

    def __post_init__(self) -> None:
        self.storage = None
        self._idx = 0
        self._is_full = False

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def size(self):
        return len(self)

    def empty(self) -> bool:
        return len(self) == 0

    def _ndim(self) -> int:
        return 1

    @torch.no_grad
    def extend(self, data: Dict) -> None:
        if len(data) == 0:
            return
        if self.storage is None:
            self.storage = {}
            initialize_storage(data, self.storage, self.capacity, self.device, n_dim=self._ndim())
            self._idx = 0
            self._is_full = False

            self._non_nested_key = [
                k for k, v in self.storage.items() if not isinstance(v, Mapping)
            ][0]

        def add_new_data(data, storage, expected_dim: int):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    add_new_data(v, storage=storage[k], expected_dim=expected_dim)
                else:
                    if v.ndim <= self._ndim():
                        raise RuntimeError(
                            f"Expected input values to have at least {self._ndim() + 1} dimensions, but got {v.shape}. Did you forget to add the batch dimension?"
                        )
                    if v.shape[0] != expected_dim:
                        raise ValueError("We expect all keys to have the same dimension")
                    end = self._idx + v.shape[0]
                    if end >= self.capacity:
                        diff = self.capacity - self._idx

                        storage[k][self._idx :] = _to_torch(v[:diff], device=self.device)

                        if v[diff:].shape[0] > self.capacity:
                            raise ValueError(
                                "The amount of data to put into buffer was way bigger than capacity. We do not currently handle this. Try extending in smaller batches / increase capacity."
                            )
                        storage[k][: v.shape[0] - diff] = _to_torch(v[diff:], device=self.device)
                        self._is_full = True
                    else:
                        storage[k][self._idx : end] = _to_torch(v, device=self.device)

        data_dim = data[self._non_nested_key].shape[0]
        add_new_data(data, self.storage, expected_dim=data_dim)
        self._idx = (self._idx + data_dim) % self.capacity

    @torch.no_grad
    def sample(self, batch_size) -> Dict[str, torch.Tensor]:
        self.ind = torch.randint(0, len(self), (batch_size,))
        return extract_values(self.storage, self.ind)

    def get_full_buffer(self) -> Dict:
        if self._is_full:
            return self.storage
        else:
            return extract_values(self.storage, torch.arange(0, len(self)))

    def save(self, folder: str | Path, nested_key_separator: str | None = None) -> None:
        if not self.empty():
            import h5py

            nested_key_separator = (
                self.nested_key_separator if nested_key_separator is None else nested_key_separator
            )
            folder = Path(folder)
            folder.mkdir(exist_ok=True, parents=True)
            hf = h5py.File(str(folder / "buffer.hdf5"), "w")

            def save_field(data, prefix: str = "", nested_key: str = "-"):
                for k, v in data.items():
                    if nested_key != "" and nested_key in k:
                        raise ValueError(
                            f"For storing, we tried to use '{nested_key}' as a nested key, but it also appeared in key {k}. Change your dataset key names or the nested key separator"
                        )
                    if isinstance(v, Mapping):
                        save_field(v, prefix=f"{prefix}{k}{nested_key}")
                    else:
                        hf.create_dataset(
                            f"{prefix}{k}", data=v[: len(self)].cpu().detach().numpy()
                        )

            save_field(self.storage, nested_key=nested_key_separator)
            hf.close()

            with (folder / "config.json").open("w+") as f:
                m_dict = dataclasses.asdict(self)
                m_dict["_idx"] = self._idx
                m_dict["_is_full"] = self._is_full
                m_dict["__target__"] = (
                    f"{inspect.getmodule(self).__name__}.{self.__class__.__name__}"
                )
                json.dump(m_dict, f, indent=4)

    def load_hdf5(self, h5_file: str | Path, nested_key_separator: str | None = None) -> None:
        import h5py

        nested_key_separator = nested_key_separator or self.nested_key_separator
        hf = h5py.File(str(h5_file), "r")
        storage = {}
        storage_size_in_h5 = None
        for k, v in hf.items():
            splits = k.split(nested_key_separator)
            current_dict = storage
            for i in range(len(splits) - 1):
                key = splits[i]
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]

            storage_tensor = _to_torch(v[:], device=self.device)
            if storage_tensor.shape[0] < self.capacity:
                storage_tensor = torch.cat(
                    [
                        storage_tensor,
                        torch.zeros(
                            (self.capacity - storage_tensor.shape[0], *storage_tensor.shape[1:]),
                            device=self.device,
                            dtype=storage_tensor.dtype,
                        ),
                    ]
                )
            assert storage_tensor.shape[0] == self.capacity, (
                f"Data shape {storage_tensor.shape} is larger than buffer capacity {self.capacity}. Seems like the saved buffer is corrupted"
            )
            assert storage_tensor.ndim > self._ndim(), (
                f"Data shape {storage_tensor.shape} has less than {self._ndim()} dimensions. Seems like the saved buffer is corrupted"
            )
            current_dict[splits[-1]] = storage_tensor

            if storage_size_in_h5 is None:
                storage_size_in_h5 = v.shape[0]
            else:
                assert storage_size_in_h5 == v.shape[0], (
                    f"Batch dimension 0 is {v.shape[0]} for hdf5 entry {k} and it is different from the first shape {storage_size_in_h5}. Seems like the saved buffer is corrupted"
                )
        hf.close()

        self.storage = storage
        self._non_nested_key = [k for k, v in self.storage.items() if not isinstance(v, Mapping)][0]

        if self._idx is None:
            self._idx = storage_size_in_h5 % self.capacity
            self._is_full = storage_size_in_h5 == self.capacity

    @classmethod
    def load(cls, path: str, device: str | None = None) -> DictBuffer:
        path = Path(path)
        with (path / "config.json").open() as f:
            loaded_config = json.load(f)
        if "__target__" in loaded_config:
            del loaded_config["__target__"]
        if device is not None:
            loaded_config["device"] = device

        _idx = None
        _is_full = None
        if "_idx" in loaded_config:
            _idx = loaded_config.pop("_idx")
            _is_full = loaded_config.pop("_is_full")
        else:
            warnings.warn(
                "Loading a buffer without _idx and _is_full. Instead we assume _idx is current length of the buffer. Newly saved buffers have this fixed"
            )
        buffer = cls(**loaded_config)
        buffer._idx = _idx
        buffer._is_full = _is_full
        buffer.load_hdf5(path / "buffer.hdf5")
        return buffer


def extract_values(d: Dict, idxs: List | torch.Tensor | np.ndarray) -> Dict:
    result = {}
    for k, v in d.items():
        if isinstance(v, Mapping):
            result[k] = extract_values(v, idxs)
        else:
            result[k] = v[idxs]
    return result


def initialize_storage(
    data: Dict, storage: Dict, capacity: int, device: Device, n_dim: int = 1
) -> None:
    def recursive_initialize(d, s):
        for k, v in d.items():
            if isinstance(v, Mapping):
                s[k] = {}
                recursive_initialize(v, s[k])
            else:
                assert v.ndim >= n_dim, (
                    f"Expected at least {n_dim} dimensions for key {k}, got {v.ndim}"
                )

                s[k] = torch.zeros(
                    (capacity, *v.shape[1:])
                    if len(v.shape) > n_dim
                    else (capacity, *v.shape[1:], 1),
                    device=device,
                    dtype=dtype_numpytotorch(v.dtype),
                )

    recursive_initialize(data, storage)


def dtype_numpytotorch(np_dtype: Any) -> torch.dtype:
    if isinstance(np_dtype, torch.dtype):
        return np_dtype
    if np_dtype == np.float16:
        return torch.float16
    elif np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float64
    elif np_dtype == np.int16:
        return torch.int16
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    elif np_dtype == np.dtype(bool):
        return torch.bool
    elif np_dtype == np.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unknown type {np_dtype}")


def dtype_numpytotorch_lower_precision(np_dtype: Any) -> torch.dtype:

    th_dtype = dtype_numpytotorch(np_dtype)
    if th_dtype == torch.float64:
        return torch.float32
    return th_dtype
