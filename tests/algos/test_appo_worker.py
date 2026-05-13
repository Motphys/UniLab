from __future__ import annotations

import numpy as np
import torch

from unilab.algos.torch.appo.worker import compute_timeout_bootstrap_correction


class _FakeCritic:
    def __call__(self, obs):
        policy = obs["policy"]
        return policy.sum(dim=1, keepdim=True)


def test_compute_timeout_bootstrap_correction_uses_final_observation_value():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=np.array([True, False]),
        final_obs=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
        final_critic=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(correction, np.array([2.5, 0.0], dtype=np.float32))


def test_compute_timeout_bootstrap_correction_prefers_explicit_final_critic():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=np.array([True, False]),
        final_obs=np.array([[2.0, 3.0], [9.0, 9.0]], dtype=np.float32),
        final_critic=np.array([[11.0, 13.0], [0.0, 0.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(correction, np.array([12.0, 0.0], dtype=np.float32))
