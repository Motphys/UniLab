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
        final_privileged=np.array([[5.0], [1.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(correction, np.array([5.0, 0.0], dtype=np.float32))
