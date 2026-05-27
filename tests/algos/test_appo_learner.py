from __future__ import annotations

from typing import Any

import torch

from unilab.algos.torch.appo.learner import APPOLearner


class _FakeModule:
    class MLP(torch.nn.Module):
        def forward(self, x):
            return x

    mlp = MLP()


def test_appo_learner_compile_targets_actor_critic_and_target(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn, **kwargs):
        calls.append((getattr(fn, "__qualname__", type(fn).__name__), kwargs))
        return fn

    learner = object.__new__(APPOLearner)
    learner._device_type = "cuda"
    learner.actor = _FakeModule()
    learner.critic = _FakeModule()
    learner.target_actor = _FakeModule()
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert len(calls) == 3
    assert all(name.endswith("MLP.forward") for name, _ in calls)
    assert all(kwargs == {"options": {"triton.cudagraphs": False}} for _, kwargs in calls)
