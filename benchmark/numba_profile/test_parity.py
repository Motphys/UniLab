"""Consistency: the numba kernel must equal the numpy oracle within tolerance.

Run standalone (``python -m benchmark.numba_profile.test_parity``) or under
pytest.  We compare, over a fresh synthetic batch:

  * total reward (per env),
  * termination mask (exact — booleans),
  * every active per-term reward-mean in the log dict,
  * each device function's ``.py_func`` vs its numpy sibling (pillar 1: the
    scalar source really is the same math).

Tolerance is ``rtol=1e-4`` on float32 reductions: ``fastmath``/FMA reorders
float ops, so bit-exactness is neither expected nor required (issue #665 asks
for "逐 bit 或在容差内一致").
"""

from __future__ import annotations

import numpy as np

from . import numba_terms as T
from . import numpy_reference as ref
from . import spec
from .numba_fused import update_state
from .state import make_batch

RTOL, ATOL = 1e-4, 1e-5


def _check(name, a, b, rtol=RTOL, atol=ATOL):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    print(f"  [{'OK ' if ok else 'BAD'}] {name:<34} max|Δ|={max_abs:.3e}")
    assert ok, f"{name} mismatch: max|Δ|={max_abs:.3e}"


def test_device_functions_match_numpy():
    """Each ``.py_func`` device fn reproduces its vectorised numpy sibling."""
    b = make_batch(64, seed=1)
    a = spec.ANCHOR_BODY_IDX
    print("device-function parity (.py_func vs numpy):")

    def per_env(fn):
        return np.array([fn(i) for i in range(b.num_envs)])

    _check(
        "motion_global_root_pos",
        per_env(
            lambda i: T.motion_global_root_pos_i.py_func(
                b.motion_body_pos_w, b.robot_body_pos_w, a, spec.REWARD_SPEC.std_root_pos, i
            )
        ),
        ref.motion_global_root_pos(b),
    )
    _check(
        "motion_body_ori",
        per_env(
            lambda i: T.motion_body_ori_i.py_func(
                b.ref_body_quat_relative_w,
                b.robot_body_quat_w,
                spec.N_BODY,
                spec.REWARD_SPEC.std_body_ori,
                i,
            )
        ),
        ref.motion_body_ori(b),
    )
    _check(
        "motion_body_ang_vel",
        per_env(
            lambda i: T.motion_body_ang_vel_i.py_func(
                b.motion_body_ang_vel_w,
                b.robot_body_ang_vel_w,
                spec.N_BODY,
                spec.REWARD_SPEC.std_body_ang_vel,
                i,
            )
        ),
        ref.motion_body_ang_vel(b),
    )
    _check(
        "joint_limit",
        per_env(
            lambda i: T.joint_limit_i.py_func(
                b.dof_pos, b.joint_lower, b.joint_upper, spec.NUM_ACTION, i
            )
        ),
        ref.joint_limit(b),
    )


def test_update_state_parity():
    """Full fused kernel vs numpy oracle: reward, terminations, per-term log."""
    for n in (512, 8192):
        b = make_batch(n, seed=7)
        r_np, term_np, log_np = ref.update_state(b)
        r_nb, term_nb, log_nb = update_state(b, force="numba", num_threads=8)
        print(f"\nupdate_state parity @ num_envs={n}:")
        _check("reward", r_nb, r_np)
        assert np.array_equal(term_nb, term_np), (
            f"termination mismatch: {int((term_nb != term_np).sum())} envs differ"
        )
        print(f"  [OK ] terminations exact           ({int(term_np.sum())} done)")
        assert log_np.keys() == log_nb.keys(), (log_np.keys(), log_nb.keys())
        for k in log_np:
            _check(f"log {k}", log_nb[k], log_np[k], rtol=1e-3, atol=1e-6)


def main():
    print("=" * 66)
    print("NUMBA vs NUMPY PARITY — g1_motion_tracking update_state")
    print("=" * 66)
    test_device_functions_match_numpy()
    test_update_state_parity()
    print("\nALL PARITY CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
