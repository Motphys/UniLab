"""Owner-declared training entrypoint contracts fail closed before runtime construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from unilab.training.entrypoints import (
    ENTRYPOINT_CONTRACT_FINGERPRINT,
    EntrypointContractError,
    EntrypointDisposition,
    EntrypointRoute,
    guarded_policy_load,
    policy_load_target,
    preflight_policy_source,
    require_entrypoint_route,
    require_policy_load_contracts,
    resolve_entrypoint_contract,
    resolve_ppo_operation,
)
from unilab.training.rsl_rl import (
    infer_rsl_rl_checkpoint_actor_input_dim,
    validate_rsl_rl_checkpoint,
)
from unilab.training.sim2sim import CrossBackendIncompatibleError

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg(*, backend: str = "mujoco") -> DictConfig:
    return OmegaConf.create(
        {
            "training": {
                "task_name": "FixtureTask",
                "sim_backend": backend,
                "execution_profile": "host_numpy",
                "operation": "auto",
                "play_only": False,
            },
            "algo": {
                "runtime_impl": "fixture_runtime",
                "runtime_resolver": "fixture.resolve_runtime",
            },
            "entrypoints": {
                "fingerprint": ENTRYPOINT_CONTRACT_FINGERPRINT,
                "renderer_backend": backend,
                "export_formats": ["onnx", "jit"],
                "routes": {route.value: "native" for route in EntrypointRoute},
                "diagnostics": {},
            },
        }
    )


def _mjwarp_owner() -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "conf" / "ppo"),
        version_base="1.3",
    ):
        return compose("config", overrides=["task=g1_walk_flat/mjwarp"])


def test_mjwarp_owner_route_matrix_is_explicit_and_fails_unsupported_early() -> None:
    cfg = _mjwarp_owner()

    for route in (
        EntrypointRoute.TRAIN,
        EntrypointRoute.EXPORT,
        EntrypointRoute.CHECKPOINT_SAVE,
        EntrypointRoute.CHECKPOINT_LOAD,
        EntrypointRoute.RESUME,
    ):
        contract = require_entrypoint_route(resolve_entrypoint_contract(cfg, route))
        assert contract.disposition is EntrypointDisposition.NATIVE
        assert contract.identity.backend == "mjwarp"
        assert contract.identity.execution_profile == "device_resident"

    for route in (EntrypointRoute.PLAY, EntrypointRoute.VISUALIZE):
        contract = resolve_entrypoint_contract(cfg, route)
        assert contract.disposition is EntrypointDisposition.UNSUPPORTED
        assert contract.renderer_backend is None
        with pytest.raises(EntrypointContractError, match="no mjwarp renderer|no native playback"):
            require_entrypoint_route(contract, renderer_backend="mujoco")


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("training.sim_backend", "mujoco", "backend"),
        ("training.execution_profile", "host_numpy", "execution_profile"),
        ("algo.runtime_impl", "rsl_rl_default", "runtime_impl"),
        ("algo.runtime_resolver", "fixture.resolve", "runtime_resolver"),
    ],
)
def test_mjwarp_owner_pins_backend_profile_and_runtime_identity(
    path: str,
    value: str,
    message: str,
) -> None:
    cfg = _mjwarp_owner()
    OmegaConf.update(cfg, path, value)

    with pytest.raises(EntrypointContractError, match=message):
        resolve_entrypoint_contract(cfg, EntrypointRoute.TRAIN)


def test_native_and_explicit_adapter_renderer_identity_cannot_be_mixed() -> None:
    native_cfg = _cfg(backend="mjwarp")
    native_cfg.entrypoints.renderer_backend = "mujoco"
    with pytest.raises(EntrypointContractError, match="explicit_adapter"):
        resolve_entrypoint_contract(native_cfg, EntrypointRoute.VISUALIZE)

    adapter_cfg = _cfg(backend="mjwarp")
    adapter_cfg.entrypoints.renderer_backend = "mujoco"
    adapter_cfg.entrypoints.routes.visualize = {
        "disposition": "explicit_adapter",
        "adapter_backend": "mujoco",
    }
    contract = resolve_entrypoint_contract(adapter_cfg, EntrypointRoute.VISUALIZE)
    assert contract.disposition is EntrypointDisposition.EXPLICIT_ADAPTER
    assert contract.identity.backend == "mjwarp"
    assert contract.renderer_backend == "mujoco"
    assert require_entrypoint_route(contract, renderer_backend="mujoco") is contract
    with pytest.raises(EntrypointContractError, match="not 'motrix'"):
        require_entrypoint_route(contract, renderer_backend="motrix")


def test_unsupported_renderer_route_must_not_retain_backend_identity() -> None:
    cfg = _cfg(backend="mjwarp")
    cfg.entrypoints.routes.play = "unsupported"

    with pytest.raises(EntrypointContractError, match="renderer_backend=null"):
        resolve_entrypoint_contract(cfg, EntrypointRoute.PLAY)


def test_policy_operation_also_requires_shared_checkpoint_load_route() -> None:
    cfg = _cfg()
    cfg.entrypoints.routes.checkpoint_load = "unsupported"
    cfg.entrypoints.diagnostics.checkpoint_load = "checkpoint loading disabled by owner"

    assert (
        require_entrypoint_route(resolve_entrypoint_contract(cfg, EntrypointRoute.EXPORT)).route
        is EntrypointRoute.EXPORT
    )
    with pytest.raises(EntrypointContractError, match="checkpoint loading disabled by owner"):
        require_policy_load_contracts(cfg, EntrypointRoute.EXPORT)

    with pytest.raises(EntrypointContractError, match="not a policy-load operation"):
        require_policy_load_contracts(cfg, EntrypointRoute.TRAIN)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: setattr(cfg.entrypoints, "fingerprint", "stale-v0"), "fingerprint"),
        (lambda cfg: setattr(cfg.entrypoints.routes, "resume", "fallback"), "unknown disposition"),
        (lambda cfg: setattr(cfg.entrypoints, "export_formats", ["onnx", "onnx"]), "duplicates"),
        (lambda cfg: setattr(cfg.entrypoints, "export_formats", ["torchscript"]), "unsupported"),
        (lambda cfg: setattr(cfg.entrypoints, "export_formats", []), "at least one"),
    ],
)
def test_malformed_owner_schema_fails_closed(mutation, message: str) -> None:
    cfg = _cfg()
    mutation(cfg)
    route = (
        EntrypointRoute.RESUME
        if "resume" in str(message) or "disposition" in message
        else EntrypointRoute.EXPORT
    )

    with pytest.raises(EntrypointContractError, match=message):
        resolve_entrypoint_contract(cfg, route)


def test_ppo_operation_selector_rejects_ambiguous_or_unknown_modes() -> None:
    cfg = _cfg()
    assert resolve_ppo_operation(cfg) is EntrypointRoute.TRAIN

    cfg.training.play_only = True
    assert resolve_ppo_operation(cfg) is EntrypointRoute.PLAY

    cfg.training.operation = "export"
    with pytest.raises(EntrypointContractError, match="conflicts"):
        resolve_ppo_operation(cfg)

    cfg.training.play_only = False
    cfg.training.operation = "evaluate"
    with pytest.raises(EntrypointContractError, match="auto, train, play, or export"):
        resolve_ppo_operation(cfg)


def test_guarded_policy_load_wraps_only_dimension_failures(tmp_path: Path) -> None:
    cfg = _cfg()
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {"entrypoints": {"fingerprint": ENTRYPOINT_CONTRACT_FINGERPRINT}},
                "contract_snapshot": {},
            }
        ),
        encoding="utf-8",
    )
    contract = resolve_entrypoint_contract(cfg, EntrypointRoute.CHECKPOINT_LOAD)
    target = policy_load_target(
        managed_policy_abi=None,
        observation_dim=17,
        action_dim=6,
    )

    with pytest.raises(CrossBackendIncompatibleError, match="env policy obs dim: 17"):
        with guarded_policy_load(
            contract=contract,
            source_run_dir=tmp_path,
            target_cfg=cfg,
            target=target,
            algo_name="ppo",
            strict=True,
        ):
            raise RuntimeError("size mismatch for actor input")

    with pytest.raises(RuntimeError, match="corrupt archive"):
        with guarded_policy_load(
            contract=contract,
            source_run_dir=tmp_path,
            target_cfg=cfg,
            target=target,
            algo_name="ppo",
            strict=True,
        ):
            raise RuntimeError("corrupt archive")


def test_policy_source_metadata_is_required_before_config_preflight(tmp_path: Path) -> None:
    cfg = _cfg()

    with pytest.raises(EntrypointContractError, match="metadata is missing"):
        preflight_policy_source(
            source_run_dir=tmp_path,
            target_cfg=cfg,
            algo_name="ppo",
            strict=True,
        )

    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "config": {"entrypoints": {"fingerprint": "stale-v0"}},
                "contract_snapshot": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EntrypointContractError, match="fingerprint"):
        preflight_policy_source(
            source_run_dir=tmp_path,
            target_cfg=cfg,
            algo_name="ppo",
            strict=True,
        )


@pytest.mark.parametrize("name,value", [("observation_dim", 0), ("action_dim", True)])
def test_policy_load_target_rejects_invalid_dimensions(name: str, value: object) -> None:
    kwargs = {"managed_policy_abi": None, "observation_dim": 4, "action_dim": 2}
    kwargs[name] = value
    with pytest.raises(EntrypointContractError, match=name):
        policy_load_target(**kwargs)


def test_rsl_rl_checkpoint_validation_and_actor_dimension_are_shared(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_3.pt"
    torch.save(
        {"actor_state_dict": {"actor.mlp.0.weight": torch.zeros((32, 17))}},
        checkpoint,
    )

    payload = validate_rsl_rl_checkpoint(checkpoint)

    assert "actor_state_dict" in payload
    assert infer_rsl_rl_checkpoint_actor_input_dim(checkpoint) == 17


@pytest.mark.parametrize("payload", [[], {}, {"actor_state_dict": None}])
def test_rsl_rl_checkpoint_validation_rejects_malformed_envelopes(
    tmp_path: Path,
    payload: object,
) -> None:
    checkpoint = tmp_path / "malformed.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(EntrypointContractError, match="rsl-rl"):
        validate_rsl_rl_checkpoint(checkpoint)


def test_rsl_rl_checkpoint_validation_wraps_parse_errors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malformed.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")

    with pytest.raises(EntrypointContractError, match="could not be parsed"):
        infer_rsl_rl_checkpoint_actor_input_dim(checkpoint)
