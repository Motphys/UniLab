from __future__ import annotations

import queue

import numpy as np

from unilab.algos.torch.appo.worker import (
    compute_timeout_bootstrap_correction,
    put_latest_metrics,
)


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


def test_put_latest_metrics_replaces_stale_item_when_queue_is_full(capsys):
    metrics_queue = queue.Queue(maxsize=1)
    metrics_queue.put_nowait({"total_steps": 1})

    put_latest_metrics(metrics_queue, {"total_steps": 2}, worker_name="APPOWorker")

    assert metrics_queue.get_nowait() == {"total_steps": 2}
    captured = capsys.readouterr()
    assert captured.err == ""
