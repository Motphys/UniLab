"""Contract tests for HORA distill teacher-owner Hydra composition."""

from __future__ import annotations

from pathlib import Path
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from unilab.algos.hora import distill_config

# distill_config's root_dir parameter expects the directory containing "conf";
# after the packaging move that is the unilab package directory.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "unilab"


# ---------------------------------------------------------------------------
# Hydra composition capabilities used by teacher-owner configs
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

    model_cfg = distill_config._student_model_defaults("hora_sac", teacher_cfg, root=_PACKAGE_ROOT)

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
                    "class_name": "unilab.algos.hora:HoraActorModel",
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

    model_cfg = distill_config._student_model_defaults(
        "hora_actor", teacher_cfg, root=_PACKAGE_ROOT
    )

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
                    "class_name": "unilab.algos.hora:HoraActorModel",
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
        distill_config._student_model_defaults("hora_actor", teacher_cfg, root=_PACKAGE_ROOT)
