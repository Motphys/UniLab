# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), observation/buffer/noise tests.
# Modified by UniLab for NumPy and env-owned RNG; Apache-2.0.

from __future__ import annotations

import numpy as np
import pytest

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._buffers import CircularBuffer, DelayBuffer
from unilab.managers._noise import (
    ConstantNoiseCfg,
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    UniformNoiseCfg,
)

from .conftest import FakeEnv


def test_circular_buffer_history_backfill_lag_and_partial_reset() -> None:
    buffer = CircularBuffer(max_len=3, batch_size=2)
    first = np.array([[1.0], [10.0]], dtype=np.float32)
    buffer.append(first)
    np.testing.assert_array_equal(buffer.buffer[:, :, 0], [[1, 1, 1], [10, 10, 10]])
    buffer.append(np.array([[2.0], [20.0]], dtype=np.float32))
    buffer.append(np.array([[3.0], [30.0]], dtype=np.float32))
    np.testing.assert_array_equal(buffer[np.array([0, 2])][:, 0], [3, 10])

    buffer.reset([1])
    buffer.append(np.array([[4.0], [99.0]], dtype=np.float32))
    np.testing.assert_array_equal(buffer.buffer[0, :, 0], [2, 3, 4])
    np.testing.assert_array_equal(buffer.buffer[1, :, 0], [99, 99, 99])


def test_circular_buffer_rejects_invalid_usage() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        CircularBuffer(max_len=0, batch_size=2)
    buffer = CircularBuffer(max_len=2, batch_size=2)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = buffer.buffer
    with pytest.raises(ValueError, match="batch size"):
        buffer.append(np.zeros((3, 1)))


def test_delay_buffer_constant_delay_and_partial_backfill() -> None:
    buffer = DelayBuffer(min_lag=2, max_lag=2, batch_size=2)
    outputs = []
    for value in (1.0, 2.0, 3.0, 4.0):
        buffer.append(np.full((2, 1), value, dtype=np.float32))
        outputs.append(buffer.compute().copy())
    np.testing.assert_array_equal(np.stack(outputs)[:, 0, 0], [1, 1, 1, 2])

    buffer.reset(np.array([1]))
    buffer.backfill(np.array([[8.0], [9.0]], dtype=np.float32), np.array([1]))
    np.testing.assert_array_equal(buffer.peek()[1], [9.0])


def test_delay_rng_is_reproducible_and_required() -> None:
    def draw(seed: int) -> list[np.ndarray]:
        buffer = DelayBuffer(
            min_lag=0,
            max_lag=3,
            batch_size=8,
            generator=np.random.default_rng(seed),
        )
        values = []
        for step in range(5):
            buffer.append(np.full((8, 1), step, dtype=np.float32))
            buffer.compute()
            values.append(buffer.current_lags.copy())
        return values

    for left, right in zip(draw(123), draw(123), strict=True):
        np.testing.assert_array_equal(left, right)

    missing_rng = DelayBuffer(min_lag=0, max_lag=2, batch_size=2)
    missing_rng.append(np.zeros((2, 1)))
    with pytest.raises(ValueError, match="env-owned"):
        missing_rng.compute()


def test_noise_configs_use_supplied_generator() -> None:
    data = np.ones((4, 3), dtype=np.float32)
    uniform = UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    first = uniform.apply(data, rng=np.random.default_rng(9))
    second = uniform.apply(data, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.float32
    with pytest.raises(ValueError, match="env-owned"):
        uniform.apply(data)

    gaussian = GaussianNoiseCfg(mean=0.0, std=0.1)
    assert gaussian.apply(data, rng=np.random.default_rng(2)).shape == data.shape
    np.testing.assert_array_equal(ConstantNoiseCfg(bias=2.0, operation="abs").apply(data), 2.0)


@pytest.mark.parametrize("operation", ["add", "scale", "abs"])
def test_uniform_noise_inplace_matches_reference_expression(operation: str) -> None:
    data = np.arange(24, dtype=np.float32).reshape(8, 3)
    n_min = np.asarray([-0.2, -0.1, -0.05], dtype=np.float32)
    n_max = np.asarray([0.3, 0.4, 0.5], dtype=np.float32)
    cfg = UniformNoiseCfg(n_min=tuple(n_min), n_max=tuple(n_max), operation=operation)

    reference_rng = np.random.default_rng(1702)
    unit = reference_rng.random(data.shape).astype(data.dtype, copy=False)
    noise = unit * (n_max - n_min) + n_min
    if operation == "add":
        expected = data + noise
    elif operation == "scale":
        expected = data * noise
    else:
        expected = noise

    actual = cfg.apply(data, rng=np.random.default_rng(1702))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data, np.arange(24, dtype=np.float32).reshape(8, 3))


def test_additive_bias_noise_supports_scalar_terms() -> None:
    from unilab.managers._noise import NoiseModelWithAdditiveBias

    cfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0),
        bias_noise_cfg=ConstantNoiseCfg(bias=0.5),
    )
    model = NoiseModelWithAdditiveBias(cfg, num_envs=4, rng=np.random.default_rng(2))
    result = model(np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(result, 1.5)
    assert result.shape == (4,)


def test_observation_groups_pipeline_order_and_history(fake_env: FakeEnv) -> None:
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "state": ObservationTermCfg(
                    func=lambda env: env.obs,
                    clip=(-1.0, 4.0),
                    scale=2.0,
                    history_length=2,
                ),
                "bias": ObservationTermCfg(
                    func=lambda env: np.ones((env.num_envs, 1), dtype=np.float32)
                ),
            }
        ),
        "dict_group": ObservationGroupCfg(
            terms={"state": ObservationTermCfg(func=lambda env: env.obs)},
            concatenate_terms=False,
        ),
    }
    manager = ObservationManager(cfg, fake_env)
    first = manager.compute(update_history=True)
    expected_first = np.clip(fake_env.obs, -1, 4) * 2
    np.testing.assert_array_equal(first["policy"][:, :4], np.tile(expected_first, (1, 2)))
    assert list(first["dict_group"]) == ["state"]
    assert manager.group_obs_dim["policy"] == (5,)

    fake_env.obs = fake_env.obs + 10
    second = manager.compute(update_history=True)["policy"]
    expected_second = np.clip(fake_env.obs, -1, 4) * 2
    np.testing.assert_array_equal(second[:, :2], expected_first)
    np.testing.assert_array_equal(second[:, 2:4], expected_second)
    assert manager.get_active_iterable_terms(0)[0][0] == "policy-state"


def test_concatenated_result_owns_each_result_and_protects_term_buffers(
    fake_env: FakeEnv,
) -> None:
    source = fake_env.obs
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "source": ObservationTermCfg(func=lambda env: env.obs),
                    "constant": ObservationTermCfg(
                        func=lambda env: np.ones((env.num_envs, 1), dtype=np.float32)
                    ),
                }
            )
        },
        fake_env,
    )

    first = manager.compute(update_history=True)["policy"]
    assert isinstance(first, np.ndarray)
    first_address = first.ctypes.data
    expected_first = np.concatenate((source.copy(), np.ones((fake_env.num_envs, 1))), axis=1)
    np.testing.assert_array_equal(first, expected_first)

    fake_env.obs += 100.0
    second = manager.compute(update_history=True)["policy"]
    assert isinstance(second, np.ndarray)
    assert second.ctypes.data != first_address
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second[:, :2], fake_env.obs)
    assert not np.shares_memory(second, fake_env.obs)


def test_concatenated_nan_sanitize_does_not_mutate_term_owned_input(
    fake_env: FakeEnv,
) -> None:
    source = fake_env.obs.copy()
    source[0, 0] = np.nan
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: source)},
                nan_policy="sanitize",
                nan_check_per_term=False,
            )
        },
        fake_env,
    )

    result = manager.compute(update_history=True)["policy"]
    assert isinstance(result, np.ndarray)
    assert np.isfinite(result).all()
    assert np.isnan(source[0, 0])


def test_concatenated_nan_error_still_identifies_offending_term(fake_env: FakeEnv) -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = np.ones((env.num_envs, 1), dtype=np.float32)
        result[2, 0] = np.nan
        return result

    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "finite": ObservationTermCfg(func=lambda env: env.obs),
                    "invalid": ObservationTermCfg(func=invalid),
                }
            )
        },
        fake_env,
    )

    with pytest.raises(
        ValueError,
        match=r"NaN detected.*'policy/invalid'.*environments: \[2\]",
    ):
        manager.compute(update_history=True)


def test_observation_noise_model_delay_and_seed_reproducibility() -> None:
    cfg = {
        "policy": ObservationGroupCfg(
            terms={
                "state": ObservationTermCfg(
                    func=lambda env: env.obs,
                    noise=NoiseModelWithAdditiveBiasCfg(
                        noise_cfg=GaussianNoiseCfg(std=0.1),
                        bias_noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                    ),
                    delay_min_lag=1,
                    delay_max_lag=1,
                )
            },
            enable_corruption=True,
        )
    }
    left = ObservationManager(cfg, FakeEnv(seed=11)).compute(update_history=True)
    right = ObservationManager(cfg, FakeEnv(seed=11)).compute(update_history=True)
    np.testing.assert_array_equal(left["policy"], right["policy"])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_observation_default_finite_policy_fails_closed(fake_env: FakeEnv, bad: float) -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.copy()
        result[1, 0] = bad
        return result

    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"bad": ObservationTermCfg(func=invalid)})},
        fake_env,
    )
    with pytest.raises(ValueError, match="ObservationManager term 'policy/bad'"):
        manager.compute()


@pytest.mark.parametrize(
    ("invalid_values", "invalid_kind"),
    [
        ((np.nan, 0.0), "NaN"),
        ((np.inf, 0.0), "Inf"),
        ((np.nan, np.inf), "NaN/Inf"),
    ],
)
def test_observation_finite_error_keeps_kind_term_and_env_diagnostics(
    fake_env: FakeEnv,
    invalid_values: tuple[float, float],
    invalid_kind: str,
) -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.copy()
        result[2] = invalid_values
        return result

    manager = ObservationManager(
        {"policy": ObservationGroupCfg(terms={"bad": ObservationTermCfg(func=invalid)})},
        fake_env,
    )
    match = rf"{invalid_kind} detected.*'policy/bad'.*environments: \[2\]"
    with pytest.raises(ValueError, match=match):
        manager.compute()


def test_observation_finite_warn_sanitizes_and_disabled_preserves(
    fake_env: FakeEnv, capsys: pytest.CaptureFixture[str]
) -> None:
    def invalid(env: FakeEnv) -> np.ndarray:
        result = env.obs.copy()
        result[1, 0] = np.nan
        return result

    warn = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"bad": ObservationTermCfg(func=invalid)}, nan_policy="warn"
            )
        },
        fake_env,
    )
    warned = warn.compute()["policy"]
    assert np.isfinite(warned).all()
    warning = capsys.readouterr().out
    assert "policy/bad" in warning
    assert "envs: [1]" in warning

    disabled = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"bad": ObservationTermCfg(func=invalid)}, nan_policy="disabled"
            )
        },
        fake_env,
    )
    assert np.isnan(disabled.compute()["policy"][1, 0])


def test_observation_explicit_sanitize_and_shape_error(fake_env: FakeEnv) -> None:
    sanitize = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "bad": ObservationTermCfg(func=lambda env: np.full((env.num_envs, 1), np.nan))
                },
                nan_policy="sanitize",
            )
        },
        fake_env,
    )
    np.testing.assert_array_equal(sanitize.compute()["policy"], 0.0)

    with pytest.raises(ValueError, match="num_envs"):
        ObservationManager(
            {
                "policy": ObservationGroupCfg(
                    terms={"bad": ObservationTermCfg(func=lambda env: np.zeros((2, 1)))}
                )
            },
            fake_env,
        )
