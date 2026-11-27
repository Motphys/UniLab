"""Tests for the shared viser playback/rendering helper functions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from unilab.visualization.viser_utils import (
    cylinder_transform,
    nominal_fps,
    scaled_vector_endpoint,
    set_handle_property,
)


def test_nominal_fps_ignores_short_exact_endpoint_interval() -> None:
    """Verify the nominal rate reflects configured cadence despite an exact final endpoint."""
    timestamps = np.asarray([0.0, 0.02, 0.04, 0.051])

    assert nominal_fps(timestamps) == pytest.approx(50.0)
    assert nominal_fps([0.0]) is None


def test_cylinder_transform_spans_contact_tip_and_anchor() -> None:
    """Verify the transform aligns a unit cylinder with both endpoints."""
    midpoint, quaternion, scale = cylinder_transform(
        np.asarray([0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.2])
    )

    np.testing.assert_allclose(midpoint, [0.0, 0.0, 0.1])
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])
    assert scale == (1.0, 1.0, pytest.approx(0.2))


def test_cylinder_transform_handles_opposing_and_degenerate_directions() -> None:
    """Verify the transform stays finite for anti-parallel and zero-length spans."""
    midpoint, quaternion, scale = cylinder_transform(
        np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(midpoint, [0.0, 0.0, 0.5])
    np.testing.assert_allclose(quaternion, [0.0, 1.0, 0.0, 0.0])
    assert scale == (1.0, 1.0, pytest.approx(1.0))

    midpoint, quaternion, scale = cylinder_transform(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0])
    )
    np.testing.assert_allclose(midpoint, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])
    assert scale == (1.0, 1.0, 1.0e-9)


def test_scaled_vector_endpoint_caps_display_length() -> None:
    """Verify the display endpoint scales the vector and honors the length cap."""
    endpoint = scaled_vector_endpoint(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([3.0, 0.0, 0.0]),
        scale=0.5,
        maximum_length=1.0,
    )
    np.testing.assert_allclose(endpoint, [2.0, 1.0, 1.0])

    endpoint = scaled_vector_endpoint(
        np.asarray([1.0, 1.0, 1.0]),
        np.asarray([3.0, 4.0, 0.0]),
        scale=1.0,
        maximum_length=2.5,
    )
    np.testing.assert_allclose(endpoint, [2.5, 3.0, 1.0])


def test_set_handle_property_skips_unchanged_values() -> None:
    """Verify numpy-aware comparison avoids redundant handle assignments."""
    handle = SimpleNamespace(value=1.0, array=np.asarray([1.0, 2.0]))

    set_handle_property(handle, "value", np.float64(1.0))
    set_handle_property(handle, "array", np.asarray([1.0, 2.0]))
    assert handle.value == 1.0

    set_handle_property(handle, "value", 2.0)
    set_handle_property(handle, "array", np.asarray([1.0, 3.0]))
    assert handle.value == 2.0
    np.testing.assert_allclose(handle.array, [1.0, 3.0])
