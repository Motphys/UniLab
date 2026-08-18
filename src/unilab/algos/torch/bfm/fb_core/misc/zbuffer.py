from typing import Union

import numpy as np
import torch


class ZBuffer:
    def __init__(
        self,
        capacity: int,
        dim: int,
        device: Union[torch.device, str],
        dtype: torch.dtype = torch.float32,
    ):
        self._storage = torch.zeros((capacity, dim), device=device, dtype=dtype)
        self._idx = 0
        self._is_full = False
        self.capacity = capacity
        self.device = device

    def __len__(self) -> int:
        return self.capacity if self._is_full else self._idx

    def empty(self) -> bool:
        return self._idx == 0 and not self._is_full

    def add(self, data: torch.Tensor) -> None:
        if self._idx + data.shape[0] >= self.capacity:
            diff = self.capacity - self._idx
            self._storage[self._idx : self._idx + data.shape[0]] = data[:diff]
            self._storage[: data.shape[0] - diff] = data[diff:]
            self._is_full = True
        else:
            self._storage[self._idx : self._idx + data.shape[0]] = data
        self._idx = (self._idx + data.shape[0]) % self.capacity

    def sample(self, num, device=None) -> torch.Tensor:
        idx = np.random.randint(0, len(self), size=num)
        return self._storage[idx].clone().to(device if device is not None else self.device)
