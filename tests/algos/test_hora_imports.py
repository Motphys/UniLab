from __future__ import annotations

import importlib
import sys

import pytest
from rsl_rl.utils import resolve_callable


def test_hora_package_import_keeps_appo_lazy() -> None:
    sys.modules.pop("unilab.algos.torch.hora", None)
    sys.modules.pop("unilab.algos.torch.hora.appo", None)

    importlib.import_module("unilab.algos.torch.hora")

    assert "unilab.algos.torch.hora.appo" not in sys.modules


def test_resolve_callable_loads_hora_ppo_from_package_export() -> None:
    resolved = resolve_callable("unilab.algos.torch.hora:HoraPPO")

    from unilab.algos.torch.hora.ppo import HoraPPO

    assert resolved is HoraPPO


def test_rsl_rl_runtime_resolver_loads_hora_wrapper_from_owner_marker() -> None:
    from unilab.algos.torch.hora.rsl_rl import HoraRslRlVecEnvWrapper
    from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    runtime = resolve_rsl_rl_ppo_runtime(
        {
            "runtime_impl": "hora_ppo",
            "runtime_resolver": "unilab.algos.torch.hora.rsl_rl:resolve_hora_ppo_runtime",
        },
        default_wrapper_cls=RslRlVecEnvWrapper,
    )

    assert runtime.wrapper_cls is HoraRslRlVecEnvWrapper


def test_rsl_rl_runtime_resolver_rejects_unresolved_custom_runtime() -> None:
    from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime
    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    with pytest.raises(ValueError, match="runtime_impl='hora_ppo'.*runtime_resolver"):
        resolve_rsl_rl_ppo_runtime(
            {"runtime_impl": "hora_ppo"},
            default_wrapper_cls=RslRlVecEnvWrapper,
        )
