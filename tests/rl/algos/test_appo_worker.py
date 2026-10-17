from __future__ import annotations

import queue

import torch

from unilab.rl.algos.appo.worker import (
    advance_rollout_observations,
    compute_timeout_bootstrap_correction,
    put_latest_metrics,
    timeout_bootstrap_mask,
    write_rollout_bootstrap_observations,
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
        timeout_mask=torch.tensor([True, False]),
        final_obs=torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float32),
        final_critic=torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float32),
    )

    torch.testing.assert_close(correction, torch.tensor([2.5, 0.0], dtype=torch.float32))


def test_advance_rollout_observations_updates_actor_and_critic_each_step():
    actor_obs, critic_obs = advance_rollout_observations(
        next_obs=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        next_critic=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        collector_device="cpu",
    )

    torch.testing.assert_close(actor_obs, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(critic_obs, torch.tensor([[3.0, 4.0]]))


def test_compute_timeout_bootstrap_correction_prefers_explicit_final_critic():
    correction = compute_timeout_bootstrap_correction(
        critic=_FakeCritic(),
        collector_device="cpu",
        gamma=0.5,
        timeout_mask=torch.tensor([True, False]),
        final_obs=torch.tensor([[2.0, 3.0], [9.0, 9.0]], dtype=torch.float32),
        final_critic=torch.tensor([[11.0, 13.0], [0.0, 0.0]], dtype=torch.float32),
    )

    torch.testing.assert_close(correction, torch.tensor([12.0, 0.0], dtype=torch.float32))


def test_timeout_bootstrap_mask_excludes_terminated_timeouts():
    mask = timeout_bootstrap_mask(
        terminated=torch.tensor([False, False, True]),
        truncated=torch.tensor([True, False, True]),
    )

    torch.testing.assert_close(mask, torch.tensor([True, False, False]))


def test_put_latest_metrics_replaces_stale_item_when_queue_is_full(capsys):
    metrics_queue = queue.Queue(maxsize=1)
    metrics_queue.put_nowait({"total_steps": 1})

    put_latest_metrics(metrics_queue, {"total_steps": 2}, worker_name="APPOWorker")

    assert metrics_queue.get_nowait() == {"total_steps": 2}
    captured = capsys.readouterr()
    assert captured.err == ""


def test_rollout_bootstrap_observations_are_published_before_signal():
    write_buf = {
        "last_obs": torch.zeros(2, 3),
        "last_critic": torch.zeros(2, 4),
    }

    write_rollout_bootstrap_observations(
        write_buf,
        last_obs=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        last_critic=torch.ones(2, 4),
    )

    torch.testing.assert_close(
        write_buf["last_obs"],
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    torch.testing.assert_close(write_buf["last_critic"], torch.ones(2, 4))
