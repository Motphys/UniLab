"""G1 motion tracking on the fixed-shape (masked / CUDA-graph) autoreset path."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unilab.base.reset_selection import supports_reset_selection
from unilab.envs.mdp.actions import JointPositionAction
from unilab.tasks.motion_tracking.common import manager_terms as motion_terms

ROOT_DIR = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "term_cls",
    [
        motion_terms.MotionCommand,
        motion_terms.MotionJointPositionAction,
        motion_terms.MotionRelativeBodyPositionErrorExp,
        motion_terms.MotionRelativeBodyOrientationErrorExp,
        motion_terms.MotionGlobalBodyLinearVelocityErrorExp,
        motion_terms.MotionGlobalBodyAngularVelocityErrorExp,
        motion_terms.UndesiredBodyContacts,
        motion_terms.BadAnchorPosZOnly,
        motion_terms.BadMotionBodyPosZOnly,
        JointPositionAction,
    ],
)
def test_g1_motion_terms_declare_reset_selection(term_cls: type) -> None:
    assert supports_reset_selection(term_cls)


def _make_env(num_envs: int, **flags):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from unilab.base.config_adapter import EnvConfigAdapter
    from unilab.base.registry import apply_cfg_overrides
    from unilab.envs import ManagerBasedRlEnvCfg, make_manager_based_rl_env

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(ROOT_DIR / "src" / "unilab" / "conf" / "flashsac"), version_base="1.3"
    ):
        owner = compose(
            "config",
            overrides=[
                "task=g1_motion_tracking/mjwarp",
                "hydra.run.dir=.",
                "hydra.output_subdir=null",
            ],
        )
    cfg = ManagerBasedRlEnvCfg()
    apply_cfg_overrides(
        cfg,
        EnvConfigAdapter(
            owner, root_dir=ROOT_DIR, algo_name="flashsac"
        ).build_task_env_cfg_override(),
    )
    for name, value in flags.items():
        setattr(cfg, name, value)
    return make_manager_based_rl_env(cfg, num_envs=num_envs, backend_type="mjwarp")


@pytest.mark.slow
def test_g1_motion_tracking_captures_and_replays_autoreset_without_host_sync() -> None:
    pytest.importorskip("mujoco_warp")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph capture requires a CUDA device")
    env = _make_env(
        64,
        mjwarp_defer_tensor_validation=True,
        masked_autoreset=True,
        persistent_step_buffers=True,
        autoreset_cuda_graph=True,
        # Poll explicitly below so every step is checked for host syncs.
        finite_validation_interval=10**6,
    )
    assert env.reset_selection_blockers() == []
    env.seed(1)
    env.init_state()
    generator = torch.Generator(device=env.device).manual_seed(0)
    dim = env.action_manager.total_action_dim

    def action() -> torch.Tensor:
        return torch.randn((env.num_envs, dim), generator=generator, device=env.device) * 0.3

    for _ in range(4):  # warmup + capture
        env.step(action())
    graph = env.autoreset_graph
    assert graph is not None and graph.captured
    done_rows = torch.zeros((), dtype=torch.int64, device=env.device)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(40):
            state = env.step(action())
            done_rows += (state.terminated | state.truncated).sum()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert graph.replays >= 40
    assert int(done_rows) > 0, "the captured reset was not exercised"
    assert env.poll_finite_validation(force=True)
    for value in state.obs.values():
        assert bool(torch.isfinite(value).all())
