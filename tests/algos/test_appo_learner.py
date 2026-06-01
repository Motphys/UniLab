from __future__ import annotations

from typing import Any

import torch

from unilab.algos.torch.appo.learner import APPOLearner
from unilab.algos.torch.hora.appo_learner import HoraAPPOLearner


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


def test_hora_appo_learner_compile_uses_same_minibatch_loss_hook(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_compile(fn, **kwargs):
        calls.append((getattr(fn, "__qualname__", type(fn).__name__), kwargs))
        return fn

    learner = object.__new__(HoraAPPOLearner)
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
