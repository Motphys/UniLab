"""Cross-side obs alignment test for the G1 WBT Obs deploy chain.

This is the load-bearing test that ensures the train -> export -> deploy
pipeline produces byte-identical actor obs at every step.  Three independent
implementations are exercised against the SAME inputs:

    1. Training side  — tracking_obs.py's _push_obs_history /
                        _fill_obs_history + actor obs assembly in
                        _build_actor_obs (replicated below in numpy).
    2. Schema side    — sim_prototype.ObsAssembler driven by deploy_config.yaml.
    3. Deploy side    — observation_manager.h::ObservationTermCfg semantics
                        replicated in Python (oldest-first deque per term,
                        group-by-term flatten, use_gym_history=false).

If any pair diverges, redeployment will silently fail at runtime — better to
catch it here than at the FSM transition with a robot on a rig.

The test is hermetic: it does NOT spin up MuJoCo, does NOT load motion clips,
and does NOT depend on training infra. It synthesizes fixed random inputs,
runs the assembly on both sides, and asserts bit-for-bit equality.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_PROTOTYPE = REPO_ROOT / "scripts" / "deploy" / "sim_prototype.py"


def _load_sim_prototype():
    """Import sim_prototype as a module (it's under scripts/, not src/)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("sim_prototype", SIM_PROTOTYPE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_prototype"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Reference deque (mirrors deploy ObservationTermCfg::reset / add / get).
# ---------------------------------------------------------------------------

class _DeployTerm:
    """Bit-for-bit copy of ObservationTermCfg buffer semantics.

    reset(obs): add(obs) H times (== fill).
    add(obs): push at back; if buffer > H, pop_front.
    get(): concat deque front (oldest) to back (newest).
    """

    def __init__(self, dim: int, history_length: int) -> None:
        self.dim = dim
        self.H = history_length
        self.buf: deque[np.ndarray] = deque()

    def reset(self, val: np.ndarray) -> None:
        v = np.asarray(val, dtype=np.float32).reshape(self.dim)
        for _ in range(self.H):
            self.add(v)

    def add(self, val: np.ndarray) -> None:
        v = np.asarray(val, dtype=np.float32).reshape(self.dim)
        self.buf.append(v.copy())
        while len(self.buf) > self.H:
            self.buf.popleft()

    def get(self) -> np.ndarray:
        return np.concatenate([np.asarray(v) for v in self.buf]).astype(np.float32)


def _deploy_compute_group(layout, current_segments_by_name):
    """Mirror ObservationManager::compute_group with use_gym_history=False."""
    terms = {}
    for seg in layout:
        terms[seg["name"]] = _DeployTerm(int(seg["dim"]), int(seg.get("history_length", 1)))
    # Match deploy reset(): fill once with the FIRST step's value.
    for seg in layout:
        terms[seg["name"]].reset(current_segments_by_name[0][seg["name"]])
    # First "step" inside compute_group calls term.add() once more BEFORE get();
    # that final add corresponds to the current frame after the reset fill.
    out_per_step = []
    for step_idx, segments in enumerate(current_segments_by_name):
        for seg in layout:
            terms[seg["name"]].add(segments[seg["name"]])
        # Concat each term's full history oldest-first, then across terms.
        out = np.concatenate(
            [terms[seg["name"]].get() for seg in layout], axis=0
        ).astype(np.float32)
        out_per_step.append(out)
    return out_per_step


# ---------------------------------------------------------------------------
# Schema fixture — mirrors what export_deploy_config.py writes for the
# deploy profile (H=5, both zero flags ON).
# ---------------------------------------------------------------------------

@pytest.fixture
def deploy_cfg():
    n = 29
    H = 5
    obs_layout = [
        {"name": "command_joint_pos",   "dim": n, "history_length": 1},
        {"name": "command_joint_vel",   "dim": n, "history_length": 1},
        {"name": "motion_anchor_ori_b", "dim": 6, "history_length": 1},
        {"name": "gyro",                "dim": 3, "history_length": H},
        {"name": "joint_pos_rel",       "dim": n, "history_length": H},
        {"name": "dof_vel",             "dim": n, "history_length": H},
        {"name": "last_actions",        "dim": n, "history_length": H},
    ]
    total = sum(s["dim"] * s["history_length"] for s in obs_layout)
    return {
        "obs_dim": total,
        "use_gym_history": False,
        "action_dim": n,
        "obs_layout": obs_layout,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _random_segments(rng, num_action=29):
    return {
        "command_joint_pos":   rng.normal(size=num_action).astype(np.float32),
        "command_joint_vel":   rng.normal(size=num_action).astype(np.float32),
        "motion_anchor_ori_b": rng.normal(size=6).astype(np.float32),
        "gyro":                rng.normal(size=3).astype(np.float32),
        "joint_pos_rel":       rng.normal(size=num_action).astype(np.float32),
        "dof_vel":             rng.normal(size=num_action).astype(np.float32),
        "last_actions":        rng.normal(size=num_action).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObsDim:
    def test_deploy_profile_dim_is_514(self, deploy_cfg):
        assert deploy_cfg["obs_dim"] == 514

    def test_layout_sum_matches_obs_dim(self, deploy_cfg):
        total = sum(
            s["dim"] * s.get("history_length", 1) for s in deploy_cfg["obs_layout"]
        )
        assert total == deploy_cfg["obs_dim"]


class TestSchemaAssemblerVsDeploy:
    """sim_prototype.ObsAssembler vs the deploy-side ObservationTermCfg."""

    def test_first_step_matches_deploy_reset_then_add(self, deploy_cfg, rng):
        sp = _load_sim_prototype()
        assembler = sp.ObsAssembler(deploy_cfg)

        seg0 = _random_segments(rng)
        prototype = assembler.step(seg0)

        deploy = _deploy_compute_group(deploy_cfg["obs_layout"], [seg0])[0]

        np.testing.assert_array_equal(prototype, deploy)

    def test_multi_step_buffer_eviction_matches_deploy(self, deploy_cfg, rng):
        sp = _load_sim_prototype()
        assembler = sp.ObsAssembler(deploy_cfg)

        all_segments = [_random_segments(rng) for _ in range(20)]
        prototype_seq = [assembler.step(s) for s in all_segments]
        deploy_seq = _deploy_compute_group(deploy_cfg["obs_layout"], all_segments)

        for k, (p, d) in enumerate(zip(prototype_seq, deploy_seq)):
            np.testing.assert_array_equal(
                p, d, err_msg=f"sim_prototype <-> deploy mismatch at step {k}"
            )

    def test_history_terms_carry_oldest_first(self, deploy_cfg, rng):
        """Spot-check the gyro history block manually: at step k>=H, the 5*3
        gyro slot of obs should be [gyro_{k-H+1}, gyro_{k-H+2}, ..., gyro_k]
        flattened, NOT the reverse."""
        sp = _load_sim_prototype()
        assembler = sp.ObsAssembler(deploy_cfg)

        gyros = [np.array([float(k), 0.0, 0.0], dtype=np.float32) for k in range(10)]
        for k in range(10):
            seg = _random_segments(rng)
            seg["gyro"] = gyros[k]
            obs = assembler.step(seg)

        # Find the gyro block. layout order: cmd_jp, cmd_jv, anchor_ori, gyro,...
        # offset = 29 + 29 + 6 = 64
        offset = 29 + 29 + 6
        gyro_block = obs[offset:offset + 3 * 5].reshape(5, 3)
        # Newest at the END (idx 4) = gyros[9]; oldest = gyros[5].
        expected = np.stack(gyros[5:10])
        np.testing.assert_array_equal(gyro_block, expected)


class TestTrainingAssemblerVsDeploy:
    """Replicate training-side history maintenance (np ring buffer) and compare.

    This replicates the exact buffer logic from tracking_obs.py:
        * _fill_obs_history: buf[:, :] = val[:, None, :]
        * _push_obs_history: buf[:, :-1] = buf[:, 1:]; buf[:, -1] = val
    and the actor obs assembly: refs first, then per-term history blocks
    (gyro, joint_pos_rel, dof_vel, last_actions) flattened (n_env, H*dim).
    """

    @staticmethod
    def _training_actor_obs(history_buf, current_refs, hist_components, num_envs):
        # refs (single-step)
        parts = [
            current_refs["command_joint_pos"],
            current_refs["command_joint_vel"],
            current_refs["motion_anchor_ori_b"],
        ]
        # proprio history (oldest-first per term, then concat across terms)
        for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
            buf = history_buf[key]  # (num_envs, H, D)
            parts.append(buf.reshape(num_envs, -1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def test_training_path_matches_deploy(self, deploy_cfg, rng):
        n_env = 1
        n = deploy_cfg["action_dim"]
        H = 5

        # Initial buffer of zeros (matches tracking_obs.py allocation).
        buf = {
            "gyro":          np.zeros((n_env, H, 3),  dtype=np.float32),
            "joint_pos_rel": np.zeros((n_env, H, n),  dtype=np.float32),
            "dof_vel":       np.zeros((n_env, H, n),  dtype=np.float32),
            "last_actions":  np.zeros((n_env, H, n),  dtype=np.float32),
        }

        all_segments = [_random_segments(rng) for _ in range(15)]
        deploy_seq = _deploy_compute_group(deploy_cfg["obs_layout"], all_segments)

        # Step 0: reset (matches tracking_obs.py is_reset=True path).
        s0 = all_segments[0]
        for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
            buf[key][:, :, :] = s0[key][None, None, :]
        refs0 = {
            "command_joint_pos":   s0["command_joint_pos"][None, :],
            "command_joint_vel":   s0["command_joint_vel"][None, :],
            "motion_anchor_ori_b": s0["motion_anchor_ori_b"][None, :],
        }
        train_obs0 = self._training_actor_obs(buf, refs0, s0, n_env)
        np.testing.assert_array_equal(train_obs0[0], deploy_seq[0])

        # Steps 1..N-1: push (matches tracking_obs.py is_reset=False path).
        for k in range(1, len(all_segments)):
            sk = all_segments[k]
            for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
                buf[key][:, :-1] = buf[key][:, 1:]
                buf[key][:, -1] = sk[key][None, :]
            refs = {
                "command_joint_pos":   sk["command_joint_pos"][None, :],
                "command_joint_vel":   sk["command_joint_vel"][None, :],
                "motion_anchor_ori_b": sk["motion_anchor_ori_b"][None, :],
            }
            train_obs = self._training_actor_obs(buf, refs, sk, n_env)
            np.testing.assert_array_equal(
                train_obs[0], deploy_seq[k],
                err_msg=f"training <-> deploy mismatch at step {k}"
            )


class TestBackCompat:
    """H=1 ('no history') must reproduce the pre-history 154-d path bit-exact."""

    @pytest.fixture
    def legacy_cfg(self):
        n = 29
        obs_layout = [
            {"name": "command_joint_pos",   "dim": n, "history_length": 1},
            {"name": "command_joint_vel",   "dim": n, "history_length": 1},
            {"name": "motion_anchor_ori_b", "dim": 6, "history_length": 1},
            {"name": "gyro",                "dim": 3, "history_length": 1},
            {"name": "joint_pos_rel",       "dim": n, "history_length": 1},
            {"name": "dof_vel",             "dim": n, "history_length": 1},
            {"name": "last_actions",        "dim": n, "history_length": 1},
        ]
        return {
            "obs_dim": 154,
            "use_gym_history": False,
            "action_dim": n,
            "obs_layout": obs_layout,
        }

    def test_legacy_obs_dim_154(self, legacy_cfg):
        assert legacy_cfg["obs_dim"] == 154

    def test_legacy_matches_simple_concat(self, legacy_cfg, rng):
        sp = _load_sim_prototype()
        assembler = sp.ObsAssembler(legacy_cfg)

        for _ in range(5):
            seg = _random_segments(rng)
            obs = assembler.step(seg)
            expected = np.concatenate([seg[s["name"]] for s in legacy_cfg["obs_layout"]])
            np.testing.assert_array_equal(obs, expected)
