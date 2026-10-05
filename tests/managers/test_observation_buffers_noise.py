# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), observation/buffer/noise tests.
# Modified by UniLab for NumPy and env-owned RNG; Apache-2.0.

from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._buffers import CircularBuffer, DelayBuffer
from unilab.managers._noise import (
    ConstantNoiseCfg,
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    SegmentwiseUniformNoiseCfg,
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
        outputs.append(buffer.compute().clone())
    np.testing.assert_array_equal(np.stack(outputs)[:, 0, 0], [1, 1, 1, 2])

    buffer.reset(np.array([1]))
    buffer.backfill(np.array([[8.0], [9.0]], dtype=np.float32), np.array([1]))
    np.testing.assert_array_equal(buffer.peek()[1], [9.0])


def test_delay_rng_is_reproducible_and_required() -> None:
    def draw(seed: int) -> list[torch.Tensor]:
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
            values.append(buffer.current_lags.clone())
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
    # Float32 data draws directly in float32 (issue #1350 fast path).
    unit = reference_rng.random(data.shape, dtype=data.dtype)
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


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (
            ConstantNoiseCfg(bias=(0.1, -0.2), operation="add"),
            lambda data, noise: data + torch.tensor((0.1, -0.2)),
        ),
        (
            ConstantNoiseCfg(bias=2.0, operation="scale"),
            lambda data, noise: data * 2.0,
        ),
        (
            ConstantNoiseCfg(bias=(0.3, -0.1), operation="abs"),
            lambda data, noise: torch.tensor((0.3, -0.1)).expand_as(data),
        ),
    ],
)
def test_constant_tensor_noise_stays_on_device_without_rng(
    cfg: ConstantNoiseCfg, expected, fake_env: FakeEnv
) -> None:
    device = fake_env.device
    data = torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2)
    result = cfg.apply(data)

    assert isinstance(result, torch.Tensor)
    assert result.device == device
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected(data, None))


@pytest.mark.parametrize("operation", ["add", "scale", "abs"])
def test_uniform_tensor_noise_preserves_rng_stream_and_device(operation: str) -> None:
    device = torch.device("cpu")
    data = torch.ones((8, 3))
    cfg = UniformNoiseCfg(n_min=(-0.2, -0.1, -0.05), n_max=(0.3, 0.4, 0.5), operation=operation)

    host_rng = np.random.default_rng(1702)
    tensor_rng = np.random.default_rng(1702)
    host_result = cfg.apply(data.numpy(), rng=host_rng)
    tensor_result = cfg.apply(data, rng=tensor_rng)

    assert isinstance(tensor_result, torch.Tensor)
    assert tensor_result.device == device
    np.testing.assert_array_equal(tensor_result.numpy(), host_result)
    assert host_rng.bit_generator.state == tensor_rng.bit_generator.state


@pytest.mark.parametrize("operation", ["add", "scale", "abs"])
def test_segmentwise_tensor_noise_bounds_each_final_column(operation: str) -> None:
    cfg = SegmentwiseUniformNoiseCfg(
        ranges=((-0.1, 0.1), (-0.2, 0.2), (0.0, 0.0)),
        operation=operation,
    )
    data = torch.ones((8, 3), dtype=torch.float32)
    host_rng = np.random.default_rng(1811)
    tensor_rng = np.random.default_rng(1811)

    host_result = cfg.apply(data.numpy(), rng=host_rng)
    tensor_result = cfg.apply(data, rng=tensor_rng)

    assert tensor_result.shape == data.shape
    np.testing.assert_array_equal(tensor_result.numpy(), host_result)
    if operation == "abs":
        assert bool((host_result[:, 0] >= -0.1).all()) and bool((host_result[:, 0] < 0.1).all())
        assert bool((host_result[:, 1] >= -0.2).all()) and bool((host_result[:, 1] < 0.2).all())
    elif operation == "scale":
        assert bool((host_result[:, 0] >= -0.1).all()) and bool((host_result[:, 0] < 0.1).all())
        assert bool((host_result[:, 1] >= -0.2).all()) and bool((host_result[:, 1] < 0.2).all())
    else:
        assert bool((host_result[:, 0] >= 0.9).all()) and bool((host_result[:, 0] < 1.1).all())
        assert bool((host_result[:, 1] >= 0.8).all()) and bool((host_result[:, 1] < 1.2).all())
    if operation == "scale":
        np.testing.assert_array_equal(host_result[:, 2], np.zeros(8))
    elif operation == "abs":
        np.testing.assert_array_equal(host_result[:, 2], np.zeros(8))
    else:
        np.testing.assert_array_equal(host_result[:, 2], np.ones(8))


def test_gaussian_tensor_noise_preserves_rng_stream_and_device() -> None:
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    cfg = GaussianNoiseCfg(mean=(0.1, -0.1, 0.0), std=(0.2, 0.3, 0.4))
    host_rng = np.random.default_rng(31)
    tensor_rng = np.random.default_rng(31)

    host_result = cfg.apply(data.numpy(), rng=host_rng)
    tensor_result = cfg.apply(data, rng=tensor_rng)

    assert isinstance(tensor_result, torch.Tensor)
    np.testing.assert_array_equal(tensor_result.numpy(), host_result)
    assert host_rng.bit_generator.state == tensor_rng.bit_generator.state


def test_gaussian_noise_clamps_standard_normal_draw() -> None:
    host_cfg = GaussianNoiseCfg(std=0.1, clamp=3.0)
    torch_cfg = GaussianNoiseCfg(std=0.1, clamp=3.0)
    host_data = np.full((128, 4), 1.0, dtype=np.float32)
    tensor_data = torch.full((128, 4), 1.0)

    host_result = host_cfg.apply(host_data, rng=np.random.default_rng(71))
    tensor_result = torch_cfg.apply(tensor_data, torch_rng=torch.Generator().manual_seed(72))

    assert isinstance(tensor_result, torch.Tensor)
    assert float(np.max(np.abs(host_result - host_data))) <= 0.3 + 1e-8
    assert float((tensor_result - tensor_data).abs().max()) <= 0.3 + 1e-8


def test_additive_bias_noise_supports_scalar_terms() -> None:
    from unilab.managers._noise import NoiseModelWithAdditiveBias

    cfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0),
        bias_noise_cfg=ConstantNoiseCfg(bias=0.5),
    )
    model = NoiseModelWithAdditiveBias(
        cfg,
        num_envs=4,
        rng=np.random.default_rng(2),
        device=torch.device("cpu"),
    )
    result = model(torch.ones(4, dtype=torch.float32))
    torch.testing.assert_close(result, torch.full((4,), 1.5))
    assert result.shape == (4,)


def test_additive_bias_noise_model_uses_manager_torch_generator(fake_env: FakeEnv) -> None:
    from unilab.managers._noise import NoiseModelWithAdditiveBias

    device = fake_env.device
    cfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(std=0.1),
        bias_noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    )
    data = torch.ones((4, 3), dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    model = NoiseModelWithAdditiveBias(
        cfg,
        num_envs=4,
        torch_rng=generator,
        device=device,
    )

    result = model(data)
    assert isinstance(result, torch.Tensor)
    assert result.device == device
    assert isinstance(model._bias, torch.Tensor)
    assert model._bias.device == device


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


def test_non_temporal_tensor_terms_stay_on_device_for_clip_scale_concat(
    fake_env: FakeEnv,
) -> None:
    device = fake_env.device
    source = torch.arange(fake_env.num_envs * 2, dtype=torch.float32, device=device).reshape(
        fake_env.num_envs, 2
    )
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=lambda env: source,
                        clip=(-1.0, 2.0),
                        scale=(2.0, 3.0),
                    ),
                    "constant": ObservationTermCfg(
                        func=lambda env: torch.ones((env.num_envs, 1), device=device)
                    ),
                }
            )
        },
        fake_env,
    )

    result = manager.compute(update_history=True)["policy"]

    assert isinstance(result, torch.Tensor)
    assert result.device == device
    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result,
        torch.cat(
            (
                source.clamp(-1.0, 2.0) * torch.tensor([2.0, 3.0], device=device),
                torch.ones((fake_env.num_envs, 1), device=device),
            ),
            dim=1,
        ),
    )
    assert source.isfinite().all()


def test_tensor_delay_and_history_stay_on_device_without_host_detour(
    fake_env: FakeEnv,
) -> None:
    device = fake_env.device
    source = torch.arange(fake_env.num_envs * 2, dtype=torch.float32, device=device).reshape(
        fake_env.num_envs, 2
    )
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "delayed": ObservationTermCfg(
                        func=lambda env: source, delay_min_lag=1, delay_max_lag=1
                    ),
                    "history": ObservationTermCfg(func=lambda env: source, history_length=2),
                }
            )
        },
        fake_env,
    )

    first = manager.compute(update_history=True)["policy"]
    assert isinstance(first, torch.Tensor)
    assert first.device == device

    history_buffer = manager._group_obs_term_history_buffer["policy"]["history"]
    delay_buffer = manager._group_obs_term_delay_buffer["policy"]["delayed"]
    assert history_buffer.buffer.device == device
    assert delay_buffer.peek().device == device

    source = source + 10
    second = manager.compute(update_history=True)["policy"]
    assert second.device == device
    torch.testing.assert_close(second[:, :2], first[:, :2])
    torch.testing.assert_close(
        second[:, 2:4],
        first[:, 2:4],
    )
    torch.testing.assert_close(second[:, 4:6], source)


def test_tensor_noise_stays_on_device_without_observation_host_detour(
    fake_env: FakeEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = fake_env.device
    source = torch.arange(fake_env.num_envs * 2, dtype=torch.float32, device=device).reshape(
        fake_env.num_envs, 2
    )
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=lambda env: source,
                        noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
                    ),
                    "biased": ObservationTermCfg(
                        func=lambda env: source,
                        noise=NoiseModelWithAdditiveBiasCfg(
                            noise_cfg=GaussianNoiseCfg(std=0.1),
                            bias_noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
                        ),
                    ),
                }
            )
        },
        fake_env,
    )

    def fail_cpu(self: torch.Tensor):
        raise AssertionError("observation noise must not copy observations to host")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)
    result = manager.compute(update_history=True)["policy"]

    assert isinstance(result, torch.Tensor)
    assert result.device == device
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_tensor_terms_are_row_scoped_on_reset(fake_env: FakeEnv) -> None:
    device = fake_env.device
    source = torch.arange(fake_env.num_envs * 2, dtype=torch.float32, device=device).reshape(
        fake_env.num_envs, 2
    )
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: source)}
            )
        },
        fake_env,
    )

    rows = manager.compute(update_history=True, env_ids=torch.tensor([1], dtype=torch.int64))[
        "policy"
    ]

    assert isinstance(rows, torch.Tensor)
    assert rows.device == device
    torch.testing.assert_close(rows, source[[1]])


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
    assert isinstance(first, torch.Tensor)
    assert first.dtype == torch.float32
    first_address = first.data_ptr()
    expected_first = torch.cat((source.clone(), torch.ones((fake_env.num_envs, 1))), dim=1)
    np.testing.assert_array_equal(first, expected_first)

    fake_env.obs += 100.0
    second = manager.compute(update_history=True)["policy"]
    assert isinstance(second, torch.Tensor)
    assert second.data_ptr() != first_address
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_array_equal(second[:, :2], fake_env.obs)
    assert not np.shares_memory(second, fake_env.obs)


def test_concatenated_nan_sanitize_does_not_mutate_term_owned_input(
    fake_env: FakeEnv,
) -> None:
    source = fake_env.obs.clone()
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
    assert isinstance(result, torch.Tensor)
    assert torch.isfinite(result).all()
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
        result = env.obs.clone()
        result[2] = torch.tensor(invalid_values, dtype=result.dtype)
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
        result = env.obs.clone()
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
    assert torch.isfinite(warned).all()
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
    assert torch.isnan(disabled.compute()["policy"][1, 0])


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


def test_identical_terms_share_raw_compute_across_groups() -> None:
    """Issue #1351: terms with identical func+params compute once per compute().

    The critic group reuses the raw (pre-noise) output of the policy group's
    identical term; the policy group still applies its own noise on top.
    """
    calls = {"n": 0}

    def counting(env: FakeEnv, scale: float = 1.0) -> np.ndarray:
        calls["n"] += 1
        return env.obs * scale

    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=counting,
                        params={"scale": 2.0},
                        noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
                    )
                },
                enable_corruption=True,
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=counting, params={"scale": 2.0})}
            ),
        },
        FakeEnv(seed=3),
    )
    env = manager._env
    calls["n"] = 0
    out = manager.compute(update_history=True)
    assert calls["n"] == 1
    np.testing.assert_array_equal(out["critic"], env.obs * 2.0)
    noise = out["policy"] - out["critic"]
    assert torch.abs(noise).max() <= 0.1 + 1e-6
    assert torch.abs(noise).max() > 0.0

    # The cache is per compute() call, not across calls.
    manager.compute(update_history=True)
    assert calls["n"] == 2


def test_class_terms_are_never_shared_across_groups() -> None:
    """Class-based (possibly stateful) terms must keep per-group calls."""

    class StatefulTerm:
        def __call__(self, env: FakeEnv) -> np.ndarray:
            return env.obs.clone()

        def reset(self, env_ids: np.ndarray | None = None) -> None:
            del env_ids

    shared = StatefulTerm()
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=shared)}),
            "critic": ObservationGroupCfg(terms={"state": ObservationTermCfg(func=shared)}),
        },
        FakeEnv(seed=5),
    )
    assert manager._group_obs_term_share["policy"] == {}
    assert manager._group_obs_term_share["critic"] == {}


def test_list_param_terms_share_raw_compute_across_groups() -> None:
    calls = {"n": 0}

    def windowed(env: FakeEnv, future_steps: list[int]) -> np.ndarray:
        calls["n"] += 1
        return np.tile(env.obs[:, :1], (1, len(future_steps)))

    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={"ref": ObservationTermCfg(func=windowed, params={"future_steps": [0, 1, 2]})}
            ),
            "critic": ObservationGroupCfg(
                terms={"ref": ObservationTermCfg(func=windowed, params={"future_steps": [0, 1, 2]})}
            ),
        },
        FakeEnv(seed=11),
    )
    calls["n"] = 0
    output = manager.compute(update_history=True)

    assert calls["n"] == 1
    np.testing.assert_array_equal(output["policy"], output["critic"])


def test_observation_manager_publishes_step_phase_attribution() -> None:
    manager = ObservationManager(
        {
            "policy": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=lambda env: env.obs,
                        noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
                    )
                },
                enable_corruption=True,
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=lambda env: env.obs)}
            ),
        },
        FakeEnv(seed=3),
    )
    manager.compute(update_history=True)

    expected = {
        "update_state_observation_term_dispatch_ms",
        "update_state_observation_validation_ms",
        "update_state_observation_noise_ms",
        "update_state_observation_transform_ms",
        "update_state_observation_temporal_ms",
        "update_state_observation_concatenation_ms",
        "update_state_observation_boundary_ms",
        "update_state_observation_manager_residual_ms",
    }
    assert set(manager.last_step_timing_ms) == expected
    assert all(value >= 0.0 for value in manager.last_step_timing_ms.values())

    manager.last_step_timing_ms.update({"stale": 1.0})
    manager.compute(update_history=True)
    assert set(manager.last_step_timing_ms) == expected

    child_keys = expected - {"update_state_observation_manager_residual_ms"}
    assert manager.last_step_timing_ms["update_state_observation_manager_residual_ms"] <= sum(
        manager.last_step_timing_ms[key] for key in child_keys
    )
