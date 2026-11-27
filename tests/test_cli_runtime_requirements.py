from __future__ import annotations

import pytest

from unilab import cli


def test_check_runtime_requirements_requires_mujoco_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mujoco" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


def test_check_runtime_requirements_mujoco_needs_mjbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Plain `mujoco` can arrive via other extras (e.g. superdex); the MuJoCo
    # physics backend is only usable with the mjbatch batch engine.
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "mjbatch" else object())

    with pytest.raises(SystemExit, match="sim=mujoco requires the MuJoCo extra"):
        cli._check_runtime_requirements("ppo", "mujoco")


@pytest.mark.parametrize("sim", ["isaacgym", "isaacsim"])
def test_isaac_sims_fail_closed_when_external_worker_is_unavailable(
    sim: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(f"unisim.backend.{sim}.dependencies.{sim}_runtime_available", lambda: False)

    with pytest.raises(SystemExit, match=f"sim={sim} requires the external"):
        cli._check_runtime_requirements("sac", sim)


def test_check_runtime_requirements_requires_newton_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "newton" else object())

    with pytest.raises(SystemExit, match="sim=newton requires the newton extra"):
        cli._check_runtime_requirements("sac", "newton")


def test_check_runtime_requirements_requires_motrix_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "motrixsim" else object())

    with pytest.raises(SystemExit, match="sim=motrix requires the motrix extra"):
        cli._check_runtime_requirements("sac", "motrix")


def test_check_runtime_requirements_requires_mtxwarp_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "unisim.backend.motrix_gpu.dependencies.motrix_gpu_available", lambda: False
    )

    with pytest.raises(SystemExit, match="sim=mtxwarp requires the mtxwarp extra"):
        cli._check_runtime_requirements("sac", "mtxwarp")


def test_check_runtime_requirements_requires_drake_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "find_spec", lambda name: None if name == "drake_uni" else object())

    with pytest.raises(SystemExit, match="sim=drake requires the Drake extra"):
        cli._check_runtime_requirements("ppo", "drake")


def test_check_runtime_requirements_rejects_unusable_drake_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_find_spec(name: str):
        return object() if name == "drake_uni" else None

    def unavailable() -> tuple[bool, ImportError | None]:
        return False, ImportError("native extension unavailable")

    monkeypatch.setattr(cli, "find_spec", fake_find_spec)
    monkeypatch.setattr("unisim.backend.drake.backend.ensure_drake_batch_available", unavailable)

    with pytest.raises(SystemExit, match="could not load the DrakeUni batch extension"):
        cli._check_runtime_requirements("ppo", "drake")


def test_cli_backend_choices_are_tensor_manager_scope() -> None:
    assert cli.SUPPORTED_SIMS == (
        "mujoco",
        "mjwarp",
        "genesis",
        "newton",
        "motrix",
        "mtxwarp",
        "superdex",
        "drake",
        "isaacsim",
        "isaacgym",
    )
