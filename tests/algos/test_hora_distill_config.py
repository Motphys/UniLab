"""Contract tests for HORA distill teacher-owner Hydra composition.

The legacy reference implementation below freezes the pre-refactor manual
loader (``OmegaConf.load`` + a loop over direct ``defaults`` entries) and the
Python-held teacher -> student hyperparameter mapping. The parity tests pin
the refactor to identical numerics while the loader moves to standard Hydra
``initialize_config_dir + compose`` and the mapping moves into
``conf/hora_distill/student_model/*.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from unilab.algos.torch.hora import distill_config

_REPO_ROOT = Path(__file__).resolve().parents[2]

TEACHER_CASES = [
    ("ppo", "sharpa_inhand/mujoco_hora"),
    # The APPO HORA owner composes through /task/sharpa_inhand/mujoco, which
    # exercises an owner whose own defaults chain into another owner file.
    ("appo", "sharpa_inhand/mujoco_hora"),
    ("sac", "sac/sharpa_inhand/mujoco_hora"),
]


# ---------------------------------------------------------------------------
# Legacy reference implementation (frozen copy of the pre-refactor behavior)
# ---------------------------------------------------------------------------


def _legacy_teacher_cfg(algo_family: str, task: str) -> Any:
    if algo_family == "sac":
        owner_path = _REPO_ROOT / "conf" / "offpolicy" / "task" / f"{task}.yaml"
        defaults_base = _REPO_ROOT / "conf" / "offpolicy"
        algo_defaults_path = _REPO_ROOT / "conf" / "offpolicy" / "algo" / "sac.yaml"
    else:
        owner_path = _REPO_ROOT / "conf" / algo_family / "task" / f"{task}.yaml"
        defaults_base = _REPO_ROOT / "conf" / algo_family
        algo_defaults_path = None
    merged_cfg = OmegaConf.create()
    if algo_defaults_path is not None:
        merged_cfg = OmegaConf.merge(
            merged_cfg,
            OmegaConf.create({"algo": OmegaConf.load(algo_defaults_path)}),
        )
    owner_cfg = OmegaConf.load(owner_path)
    for default_entry in owner_cfg.get("defaults", []):
        if not isinstance(default_entry, str) or default_entry == "_self_":
            continue
        include_path = defaults_base / f"{default_entry.lstrip('/')}.yaml"
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.load(include_path))
    return OmegaConf.merge(merged_cfg, owner_cfg)


def _legacy_model_cfg(algo_family: str, task: str) -> dict[str, Any]:
    teacher_cfg = _legacy_teacher_cfg(algo_family, task)
    if algo_family == "sac":
        actor_cfg = OmegaConf.to_container(
            OmegaConf.select(teacher_cfg, "algo.actor"), resolve=True
        )
        if not isinstance(actor_cfg, dict):
            actor_cfg = {}
        return {
            "teacher_arch": "hora_sac",
            "actor_hidden_dim": OmegaConf.select(teacher_cfg, "algo.actor_hidden_dim", default=512),
            "use_layer_norm": OmegaConf.select(teacher_cfg, "algo.use_layer_norm", default=True),
            "priv_info_embed_dim": actor_cfg.get("priv_info_embed_dim", 9),
            "priv_mlp_hidden_dims": actor_cfg.get("priv_mlp_hidden_dims", [256, 128, 9]),
        }
    actor_cfg = OmegaConf.to_container(OmegaConf.select(teacher_cfg, "algo.actor"), resolve=True)
    actor_cfg = dict(actor_cfg) if isinstance(actor_cfg, dict) else {}
    actor_cfg.pop("class_name", None)
    distribution_cfg = actor_cfg.get("distribution_cfg")
    if isinstance(distribution_cfg, dict):
        distribution_cfg = {
            key: value for key, value in distribution_cfg.items() if key != "class_name"
        }
    return {
        "hidden_dims": actor_cfg.get("hidden_dims"),
        "activation": actor_cfg.get("activation"),
        "obs_normalization": actor_cfg.get("obs_normalization"),
        "priv_info_embed_dim": actor_cfg.get("priv_info_embed_dim"),
        "priv_mlp_hidden_dims": actor_cfg.get("priv_mlp_hidden_dims"),
        "distribution_cfg": distribution_cfg,
    }


def _assert_subset(legacy: Any, new: Any, path: str) -> None:
    """Every leaf the legacy loader produced must be identical in the new one."""
    if isinstance(legacy, dict) and isinstance(new, dict):
        for key, value in legacy.items():
            assert key in new, f"missing key {path}.{key}"
            _assert_subset(value, new[key], f"{path}.{key}")
        return
    assert legacy == new, f"value mismatch at {path}: legacy={legacy!r} new={new!r}"


# ---------------------------------------------------------------------------
# Parity: new Hydra compose path vs frozen legacy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("algo_family", "task"), TEACHER_CASES)
def test_teacher_default_cfg_matches_legacy_manual_loader(algo_family: str, task: str) -> None:
    cfg = OmegaConf.create({"teacher": {"algo_family": algo_family, "task": task}})

    new_cfg = OmegaConf.to_container(
        distill_config.teacher_default_cfg(cfg, root_dir=_REPO_ROOT), resolve=True
    )
    legacy_cfg = OmegaConf.to_container(_legacy_teacher_cfg(algo_family, task), resolve=True)

    assert new_cfg["algo"]["model"] == _legacy_model_cfg(algo_family, task)
    for section in ("training", "reward", "env"):
        _assert_subset(legacy_cfg.get(section), new_cfg.get(section), section)


# ---------------------------------------------------------------------------
# Compose capabilities the manual loader could not express
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_teacher_owner_config_supports_nested_defaults_packages_and_interpolation(
    tmp_path: Path,
) -> None:
    conf_dir = tmp_path / "conf" / "ppo"
    _write(
        conf_dir / "config.yaml",
        "defaults:\n  - _self_\n  - task: group/owner\n  - extra: packed\n\nroot_scalar: 3\n",
    )
    # Package directive: file content lands under `extra`, not at the root.
    _write(conf_dir / "extra" / "packed.yaml", "# @package extra\ninner: ${root_scalar}\n")
    _write(
        conf_dir / "task" / "group" / "owner.yaml",
        "# @package _global_\ndefaults:\n  - group/mid\n  - _self_\n\nleaf: owner\n",
    )
    # Nested defaults: an included group file with its own defaults list.
    _write(
        conf_dir / "task" / "group" / "mid.yaml",
        "# @package _global_\ndefaults:\n  - nested_leaf\n  - _self_\n\nmid_value: 5\n",
    )
    _write(
        conf_dir / "task" / "group" / "nested_leaf.yaml",
        "# @package _global_\nnested_value: 42\n",
    )

    cfg = distill_config.load_teacher_owner_config("ppo", "group/owner", root_dir=tmp_path)

    assert cfg.nested_value == 42
    assert cfg.mid_value == 5
    assert cfg.leaf == "owner"
    assert cfg.extra.inner == 3


def test_hora_sac_mapping_keeps_yaml_fallbacks_for_missing_teacher_fields(tmp_path: Path) -> None:
    teacher_cfg = OmegaConf.create({"algo": {"runtime_impl": "hora_sac", "actor": {}}})

    model_cfg = distill_config._student_model_defaults("hora_sac", teacher_cfg, root=_REPO_ROOT)

    assert model_cfg == {
        "teacher_arch": "hora_sac",
        "actor_hidden_dim": 512,
        "use_layer_norm": True,
        "priv_info_embed_dim": 9,
        "priv_mlp_hidden_dims": [256, 128, 9],
    }


def test_hora_actor_mapping_strips_distribution_class_name() -> None:
    teacher_cfg = OmegaConf.create(
        {
            "algo": {
                "actor": {
                    "class_name": "unilab.algos.torch.hora:HoraActorModel",
                    "hidden_dims": [64, 32],
                    "activation": "relu",
                    "obs_normalization": False,
                    "priv_info_embed_dim": 4,
                    "priv_mlp_hidden_dims": [16, 4],
                    "distribution_cfg": {
                        "class_name": "GaussianDistribution",
                        "init_std": 0.5,
                        "std_type": "scalar",
                    },
                }
            }
        }
    )

    model_cfg = distill_config._student_model_defaults("hora_actor", teacher_cfg, root=_REPO_ROOT)

    assert model_cfg == {
        "hidden_dims": [64, 32],
        "activation": "relu",
        "obs_normalization": False,
        "priv_info_embed_dim": 4,
        "priv_mlp_hidden_dims": [16, 4],
        "distribution_cfg": {"init_std": 0.5, "std_type": "scalar"},
    }


def test_hora_actor_mapping_fails_closed_when_teacher_field_is_missing() -> None:
    teacher_cfg = OmegaConf.create(
        {
            "algo": {
                "actor": {
                    "class_name": "unilab.algos.torch.hora:HoraActorModel",
                    "activation": "elu",
                    "obs_normalization": True,
                    "priv_info_embed_dim": 9,
                    "priv_mlp_hidden_dims": [256, 128, 9],
                    "distribution_cfg": {"init_std": 1.0, "std_type": "scalar"},
                }
            }
        }
    )

    with pytest.raises(InterpolationResolutionError):
        distill_config._student_model_defaults("hora_actor", teacher_cfg, root=_REPO_ROOT)
