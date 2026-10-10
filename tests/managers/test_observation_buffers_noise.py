# Derived from mujocolab/mjlab v1.6.0 (0fb8a681), observation/buffer/noise tests.
# Modified by UniLab for Torch-only buffers/noise; Apache-2.0.

from __future__ import annotations

import numpy as np
import pytest
import torch

from unilab.managers import ObservationGroupCfg, ObservationManager, ObservationTermCfg
from unilab.managers._buffers import CircularBuffer, DelayBuffer
from unilab.managers._noise import (
    ConstantNoiseCfg,
    GaussianNoiseCfg,
    NoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
    SegmentwiseUniformNoiseCfg,
    UniformNoiseCfg,
)

from .conftest import FakeEnv


def test_circular_buffer_history_backfill_lag_and_partial_reset() -> None:
    buffer = CircularBuffer(max_len=3, batch_size=2)
    first = torch.tensor([[1.0], [10.0]], dtype=torch.float32)
    buffer.append(first)
    torch.testing.assert_close(
        buffer.buffer[:, :, 0], torch.tensor([[1, 1, 1], [10, 10, 10]], dtype=buffer.buffer.dtype)
    )
    buffer.append(torch.tensor([[2.0], [20.0]], dtype=torch.float32))
    buffer.append(torch.tensor([[3.0], [30.0]], dtype=torch.float32))
    torch.testing.assert_close(buffer[torch.tensor([0, 2])][:, 0], torch.tensor([3.0, 10.0]))

    buffer.reset([1])
    buffer.append(torch.tensor([[4.0], [99.0]], dtype=torch.float32))
    torch.testing.assert_close(
        buffer.buffer[0, :, 0], torch.tensor([2.0, 3.0, 4.0], dtype=buffer.buffer.dtype)
    )
    torch.testing.assert_close(
        buffer.buffer[1, :, 0], torch.tensor([99.0, 99.0, 99.0], dtype=buffer.buffer.dtype)
    )


def test_circular_buffer_rejects_invalid_usage() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        CircularBuffer(max_len=0, batch_size=2)
    buffer = CircularBuffer(max_len=2, batch_size=2)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = buffer.buffer
    with pytest.raises(ValueError, match="batch size"):
        buffer.append(torch.zeros((3, 1)))
    with pytest.raises(TypeError, match="torch.Tensor"):
        buffer.append(np.zeros((2, 1)))


def test_delay_buffer_constant_delay_and_partial_backfill() -> None:
    buffer = DelayBuffer(min_lag=2, max_lag=2, batch_size=2)
    outputs = []
    for value in (1.0, 2.0, 3.0, 4.0):
        buffer.append(torch.full((2, 1), value, dtype=torch.float32))
        outputs.append(buffer.compute().clone())
    torch.testing.assert_close(torch.stack(outputs)[:, 0, 0], torch.tensor([1.0, 1.0, 1.0, 2.0]))

    buffer.reset(torch.tensor([1]))
    buffer.backfill(torch.tensor([[8.0], [9.0]], dtype=torch.float32), torch.tensor([1]))
    torch.testing.assert_close(buffer.peek()[1], torch.tensor([9.0]))


def test_delay_rng_is_reproducible_and_required() -> None:
    def draw(seed: int) -> list[torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        buffer = DelayBuffer(min_lag=0, max_lag=3, batch_size=8, torch_generator=generator)
        values = []
        for step in range(5):
            buffer.append(torch.full((8, 1), step, dtype=torch.float32))
            buffer.compute()
            values.append(buffer.current_lags.clone())
        return values

    for left, right in zip(draw(123), draw(123), strict=True):
        torch.testing.assert_close(left, right)

    missing_rng = DelayBuffer(min_lag=0, max_lag=2, batch_size=2)
    missing_rng.append(torch.zeros((2, 1)))
    with pytest.raises(ValueError, match="env-owned Torch generator"):
        missing_rng.compute()


def test_noise_configs_use_supplied_generator() -> None:
    data = torch.ones((4, 3), dtype=torch.float32)
    uniform = UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    first_rng = torch.Generator().manual_seed(9)
    second_rng = torch.Generator().manual_seed(9)
    first = uniform.apply(data, torch_rng=first_rng)
    second = uniform.apply(data, torch_rng=second_rng)
    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float32
    with pytest.raises(ValueError, match="env-owned"):
        uniform.apply(data)

    gaussian = GaussianNoiseCfg(mean=0.0, std=0.1)
    assert gaussian.apply(data, torch_rng=torch.Generator().manual_seed(2)).shape == data.shape
    torch.testing.assert_close(
        ConstantNoiseCfg(bias=2.0, operation="abs").apply(data), torch.full_like(data, 2.0)
    )


@pytest.mark.parametrize(
    "cfg",
    [
        ConstantNoiseCfg(bias=2.0),
        UniformNoiseCfg(n_min=-0.2, n_max=0.2),
        SegmentwiseUniformNoiseCfg(ranges=((-0.2, 0.2),)),
        GaussianNoiseCfg(std=0.1),
    ],
)
def test_noise_configs_reject_numpy_carriers(cfg: NoiseCfg) -> None:
    generator = torch.Generator().manual_seed(7)

    with pytest.raises(TypeError, match="torch.Tensor"):
        cfg.apply(np.zeros((4, 1), dtype=np.float32), torch_rng=generator)


@pytest.mark.parametrize("operation", ["add", "scale", "abs"])
def test_uniform_noise_inplace_matches_reference_expression(operation: str) -> None:
    data = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    n_min = torch.tensor([-0.2, -0.1, -0.05], dtype=torch.float32)
    n_max = torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32)
    cfg = UniformNoiseCfg(n_min=tuple(n_min), n_max=tuple(n_max), operation=operation)

    reference_rng = torch.Generator().manual_seed(1702)
    unit = torch.rand(tuple(data.shape), dtype=data.dtype, generator=reference_rng)
    noise = unit * (n_max - n_min) + n_min
    if operation == "add":
        expected = data + noise
    elif operation == "scale":
        expected = data * noise
    else:
        expected = noise

    actual = cfg.apply(data, torch_rng=torch.Generator().manual_seed(1702))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(data, torch.arange(24, dtype=torch.float32).reshape(8, 3))


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
def test_uniform_tensor_noise_uses_manager_rng_stream_and_device(operation: str) -> None:
    device = torch.device("cpu")
    data = torch.ones((8, 3))
    cfg = UniformNoiseCfg(n_min=(-0.2, -0.1, -0.05), n_max=(0.3, 0.4, 0.5), operation=operation)

    first_rng = torch.Generator().manual_seed(1702)
    second_rng = torch.Generator().manual_seed(1702)
    first = cfg.apply(data, torch_rng=first_rng)
    second = cfg.apply(data, torch_rng=second_rng)

    assert isinstance(first, torch.Tensor)
    assert first.device == device
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("operation", ["add", "scale", "abs"])
def test_segmentwise_tensor_noise_bounds_each_final_column(operation: str) -> None:
    cfg = SegmentwiseUniformNoiseCfg(
        ranges=((-0.1, 0.1), (-0.2, 0.2), (0.0, 0.0)),
        operation=operation,
    )
    data = torch.ones((8, 3), dtype=torch.float32)
    generator = torch.Generator().manual_seed(1811)
    tensor_result = cfg.apply(data, torch_rng=generator)

    assert tensor_result.shape == data.shape
    if operation == "abs":
        assert bool((tensor_result[:, 0] >= -0.1).all()) and bool((tensor_result[:, 0] < 0.1).all())
        assert bool((tensor_result[:, 1] >= -0.2).all()) and bool((tensor_result[:, 1] < 0.2).all())
    elif operation == "scale":
        assert bool((tensor_result[:, 0] >= -0.1).all()) and bool((tensor_result[:, 0] < 0.1).all())
        assert bool((tensor_result[:, 1] >= -0.2).all()) and bool((tensor_result[:, 1] < 0.2).all())
    else:
        assert bool((tensor_result[:, 0] >= 0.9).all()) and bool((tensor_result[:, 0] < 1.1).all())
        assert bool((tensor_result[:, 1] >= 0.8).all()) and bool((tensor_result[:, 1] < 1.2).all())
    if operation == "scale":
        torch.testing.assert_close(tensor_result[:, 2], torch.zeros(8))
    elif operation == "abs":
        torch.testing.assert_close(tensor_result[:, 2], torch.zeros(8))
    else:
        torch.testing.assert_close(tensor_result[:, 2], torch.ones(8))


def test_gaussian_tensor_noise_uses_manager_rng_stream_and_device() -> None:
    data = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    cfg = GaussianNoiseCfg(mean=(0.1, -0.1, 0.0), std=(0.2, 0.3, 0.4))
    first_rng = torch.Generator().manual_seed(31)
    second_rng = torch.Generator().manual_seed(31)
    first = cfg.apply(data, torch_rng=first_rng)
    second = cfg.apply(data, torch_rng=second_rng)

    assert isinstance(first, torch.Tensor)
    torch.testing.assert_close(first, second)


def test_gaussian_noise_clamps_standard_normal_draw() -> None:
    torch_cfg = GaussianNoiseCfg(std=0.1, clamp=3.0)
    tensor_data = torch.full((128, 4), 1.0)

    tensor_result = torch_cfg.apply(tensor_data, torch_rng=torch.Generator().manual_seed(72))

    assert isinstance(tensor_result, torch.Tensor)
    assert float((tensor_result - tensor_data).abs().max()) <= 0.3 + 1e-8


@pytest.mark.parametrize("clamp", [0.0, -3.0, float("inf"), True])
def test_gaussian_noise_rejects_invalid_clamp(clamp) -> None:
    with pytest.raises((TypeError, ValueError), match="clamp"):
        GaussianNoiseCfg(std=0.1, clamp=clamp)


def test_additive_bias_noise_supports_scalar_terms() -> None:
    from unilab.managers._noise import NoiseModelWithAdditiveBias

    cfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=ConstantNoiseCfg(bias=0.0),
        bias_noise_cfg=ConstantNoiseCfg(bias=0.5),
    )
    model = NoiseModelWithAdditiveBias(
        cfg,
        num_envs=4,
        torch_rng=torch.Generator().manual_seed(2),
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
                    func=lambda env: torch.ones((env.num_envs, 1), dtype=torch.float32)
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
    expected_first = fake_env.obs.clamp(-1, 4) * 2
    torch.testing.assert_close(first["policy"][:, :4], expected_first.repeat(1, 2))
    assert list(first["dict_group"]) == ["state"]
    assert manager.group_obs_dim["policy"] == (5,)

    fake_env.obs = fake_env.obs + 10
    second = manager.compute(update_history=True)["policy"]
    expected_second = fake_env.obs.clamp(-1, 4) * 2
    torch.testing.assert_close(second[:, :2], expected_first)
    torch.testing.assert_close(second[:, 2:4], expected_second)
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
                        func=lambda env: torch.ones((env.num_envs, 1), dtype=torch.float32)
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
    torch.testing.assert_close(first, expected_first)

    fake_env.obs += 100.0
    second = manager.compute(update_history=True)["policy"]
    assert isinstance(second, torch.Tensor)
    assert second.data_ptr() != first_address
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second[:, :2], fake_env.obs)
    assert second.data_ptr() != fake_env.obs.data_ptr()


def test_concatenated_nan_sanitize_does_not_mutate_term_owned_input(
    fake_env: FakeEnv,
) -> None:
    source = fake_env.obs.clone()
    source[0, 0] = torch.nan
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
    assert bool(torch.isnan(source[0, 0]))


def test_concatenated_nan_error_still_identifies_offending_term(fake_env: FakeEnv) -> None:
    def invalid(env: FakeEnv) -> torch.Tensor:
        result = torch.ones((env.num_envs, 1), dtype=torch.float32)
        result[2, 0] = torch.nan
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
    def invalid(env: FakeEnv) -> torch.Tensor:
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
    def invalid(env: FakeEnv) -> torch.Tensor:
        result = env.obs.clone()
        result[1, 0] = torch.nan
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
                    "bad": ObservationTermCfg(
                        func=lambda env: torch.full((env.num_envs, 1), torch.nan)
                    )
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
                    terms={
                        "bad": ObservationTermCfg(
                            func=lambda env: torch.zeros((2, 1), dtype=torch.float32)
                        )
                    }
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

    def counting(env: FakeEnv, scale: float = 1.0) -> torch.Tensor:
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
        def __call__(self, env: FakeEnv) -> torch.Tensor:
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
        return env.obs[:, :1].repeat(1, len(future_steps))

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
