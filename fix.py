import numpy as np
from typing import Union, Any, Optional, Sequence

class NumPyRewardTerm:
    def __init__(self, name: str = "reward", value: Union[float, np.ndarray, Any] = 0.0,
                 channels: int = 1, dim: Optional[int] = None):
        self._name = name
        self._channels = channels
        
        # Convert raw input to numpy array for consistent indexing
        self._data = np.asarray(value)
        
        # Handle 'No Sensor' migration logic: Normalize dimensionality
        # If channels=1, often data arrives as 1D, but logic might expect 2D
        if self._channels == 1:
            if self._data.ndim == 0: # Scalar
                self._data = np.expand_dims(self._data, 0)
            elif self._data.ndim == 1 and self._channels == 1:
                self._shape = self._data.shape
            else:
                self._shape = self._data.shape if self._data.ndim > 1 else (1,)
        else:
            self._shape = self._data.shape if self._data.ndim > 0 else (1,)

        # Support legacy 'dim' attribute if passed
        self._dim = dim if dim is not None else self._shape[0] if self._shape[0] > 0 else 1

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def channels(self):
        return self._channels

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def __float__(self):
        # Essential for direct math operations (e.g., `reward + 1.0`)
        return float(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data) if self._data.ndim == 1 else self._data.shape[0]

    @property
    def value(self):
        return self._data

    def _get_channel(self, idx):
        if self._channels == 1 and self._data.ndim == 2:
            return self._data[:, idx]
        elif self._channels == 1 and self._data.ndim == 1:
            return self._data[idx]
        else:
            return self._data

    def __repr__(self):
        return f"{self._name}({self._data})"

    def __getattr__(self, key):
        # Fallback to numpy attributes for 'No Sensor' migration
        try:
            return getattr(self._data, key)
        except AttributeError:
            return getattr(self, f"_{key}")

    @classmethod
    def from_list(cls, name, lst, **kwargs):
        return cls(name, np.asarray(lst), **kwargs)

    @classmethod
    def from_scalar(cls, name, scalar, **kwargs):
        return cls(name, np.asarray(scalar), **kwargs)