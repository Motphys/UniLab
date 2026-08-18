"""Focused tests for the cold-path named sensor view contract."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from unilab.base.backend import BackendSensorView
from unilab.base.backend.base import SimBackend


class _SensorBackend:
    backend_type = "fake"
    num_envs = 2

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.values = {
            "contact_a": np.asarray([[1.0], [0.0]], dtype=np.float32),
            "contact_b": np.asarray([[0.0], [1.0]], dtype=np.float32),
        }

    def get_sensor_data(self, name: str) -> np.ndarray:
        self.calls["single"] += 1
        return self.values[name]

    def get_sensor_data_batch(self, names: tuple[str, ...]) -> np.ndarray:
        self.calls["batch"] += 1
        return np.concatenate([self.values[name] for name in names], axis=1)

    def _bind_sensor_data_reader(self, names: tuple[str, ...]):
        batch_reader = self.get_sensor_data_batch
        return lambda: batch_reader(names)


def test_bind_sensor_data_validates_once_and_reads_through_formal_view() -> None:
    backend = _SensorBackend()

    view = SimBackend.bind_sensor_data(backend, ("contact_a", "contact_b"))  # type: ignore[arg-type]

    assert isinstance(view, BackendSensorView)
    assert view.names == ("contact_a", "contact_b")
    assert view.dimensions == (1, 1)
    assert view.width == 2
    assert backend.calls == {"single": 2, "batch": 1}

    np.testing.assert_array_equal(view.read(), [[1.0, 0.0], [0.0, 1.0]])
    assert backend.calls == {"single": 2, "batch": 2}


@pytest.mark.parametrize(
    "names, error, match",
    [
        ("contact_a", TypeError, "sequence of strings"),
        ((), ValueError, "at least one name"),
        (("",), ValueError, "non-empty strings"),
        (("contact_a", "contact_a"), ValueError, "unique"),
    ],
)
def test_bind_sensor_data_rejects_invalid_name_requests(
    names: str | tuple[str, ...], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        SimBackend.bind_sensor_data(_SensorBackend(), names)  # type: ignore[arg-type]


def test_bind_sensor_data_fails_closed_on_unknown_sensor() -> None:
    with pytest.raises(KeyError, match="cannot bind sensor 'missing'"):
        SimBackend.bind_sensor_data(_SensorBackend(), ("missing",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value, error, match",
    [
        (np.zeros((1, 2), dtype=np.float32), ValueError, "expected leading dimension 2"),
        (np.full((2, 1), np.nan, dtype=np.float32), ValueError, "NaN or Inf"),
    ],
)
def test_backend_sensor_view_rejects_invalid_runtime_data(
    value: np.ndarray, error: type[Exception], match: str
) -> None:
    backend = _SensorBackend()
    backend.values["contact_a"] = value

    if value.shape[0] != backend.num_envs:
        with pytest.raises(error, match=match):
            SimBackend.bind_sensor_data(backend, ("contact_a",))  # type: ignore[arg-type]
        return

    with pytest.raises(error, match=match):
        SimBackend.bind_sensor_data(backend, ("contact_a",))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value, error, match",
    [
        (np.zeros((2, 2), dtype=np.float32), ValueError, r"expected \(2, 1\)"),
        (np.full((2, 1), np.inf, dtype=np.float32), ValueError, "NaN or Inf"),
        (np.full((2, 1), "invalid", dtype=object), TypeError, "non-numeric dtype"),
    ],
)
def test_backend_sensor_view_detects_runtime_contract_drift(
    value: np.ndarray, error: type[Exception], match: str
) -> None:
    backend = _SensorBackend()
    view = SimBackend.bind_sensor_data(backend, ("contact_a",))  # type: ignore[arg-type]

    backend.values["contact_a"] = value

    with pytest.raises(error, match=match):
        view.read()


def test_backend_sensor_view_metadata_is_immutable() -> None:
    view = BackendSensorView(
        backend_type="fake",
        names=("sensor",),
        dimensions=(1,),
        num_envs=1,
        _reader=lambda: np.zeros((1, 1), dtype=np.float32),
    )

    with pytest.raises(AttributeError):
        view.names = ("other",)  # type: ignore[misc]
