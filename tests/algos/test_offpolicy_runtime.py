from __future__ import annotations

import pytest


def test_offpolicy_runtime_defaults_to_standard_sac_overrides() -> None:
    from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

    runtime = OffPolicyRuntime()

    assert runtime.learner_cls is None
    assert runtime.algo_type is None
    assert runtime.supports_symmetry is True
    assert runtime.build_model_kwargs(obs_dim=4, critic_obs_dim=6) == {}


def test_hora_sac_runtime_builds_privileged_actor_kwargs() -> None:
    from unilab.algos.torch.hora.sac import resolve_hora_sac_runtime

    runtime = resolve_hora_sac_runtime(
        {
            "runtime_impl": "hora_sac",
            "actor": {
                "priv_info_embed_dim": 7,
                "priv_mlp_hidden_dims": [16, 7],
            },
        }
    )

    assert runtime is not None
    assert runtime.algo_type == "hora_sac"
    assert runtime.supports_symmetry is False
    assert runtime.build_model_kwargs(obs_dim=5, critic_obs_dim=8) == {
        "priv_info_dim": 3,
        "priv_info_embed_dim": 7,
        "priv_mlp_hidden_dims": (16, 7),
    }


def test_hora_sac_runtime_requires_critic_tail() -> None:
    from unilab.algos.torch.hora.sac import resolve_hora_sac_runtime

    runtime = resolve_hora_sac_runtime({"runtime_impl": "hora_sac"})

    assert runtime is not None
    with pytest.raises(ValueError, match="privileged tail"):
        runtime.build_model_kwargs(obs_dim=5, critic_obs_dim=5)


def test_offpolicy_runtime_rejects_marker_without_resolver() -> None:
    from unilab.algos.torch.offpolicy.runtime import resolve_custom_offpolicy_runtime

    with pytest.raises(ValueError, match="runtime_impl='hora_sac'.*runtime_resolver"):
        resolve_custom_offpolicy_runtime({"runtime_impl": "hora_sac"})
