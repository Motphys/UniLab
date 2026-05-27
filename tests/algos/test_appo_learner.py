from __future__ import annotations

from typing import Any

import torch

from unilab.algos.torch.appo.learner import APPOLearner


def test_appo_learner_compile_targets_minibatch_loss(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn, **kwargs):
        calls.append((getattr(fn, "__qualname__", type(fn).__name__), kwargs))
        return fn

    learner = object.__new__(APPOLearner)
    learner._device_type = "cuda"
    learner._minibatch_loss_fn = learner._minibatch_loss_tensors
    monkeypatch.setattr(torch, "compile", fake_compile)

    learner._compile_training_methods()

    assert calls == [
        (
            "APPOLearner._minibatch_loss_tensors",
            {"mode": "reduce-overhead", "fullgraph": False},
        )
    ]
    assert learner._minibatch_loss_fn == learner._minibatch_loss_tensors
