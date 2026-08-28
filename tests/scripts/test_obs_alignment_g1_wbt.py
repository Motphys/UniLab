from __future__ import annotations

from collections import deque

import numpy as np
import pytest


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
    """Assemble per step: each term's full history oldest-first, then concat."""
    terms = {}
    for seg in layout:
        terms[seg["name"]] = _DeployTerm(int(seg["dim"]), int(seg.get("history_length", 1)))
    for seg in layout:
        terms[seg["name"]].reset(current_segments_by_name[0][seg["name"]])
    out_per_step = []
    for segments in current_segments_by_name:
        for seg in layout:
            terms[seg["name"]].add(segments[seg["name"]])
        out = np.concatenate([terms[seg["name"]].get() for seg in layout], axis=0).astype(
            np.float32
        )
        out_per_step.append(out)
    return out_per_step


@pytest.fixture
def deploy_cfg():
    n = 29
    H = 5
    obs_layout = [
        {"name": "command_joint_pos", "dim": n, "history_length": 1},
        {"name": "command_joint_vel", "dim": n, "history_length": 1},
        {"name": "motion_anchor_ori_b", "dim": 6, "history_length": 1},
        {"name": "gyro", "dim": 3, "history_length": H},
        {"name": "joint_pos_rel", "dim": n, "history_length": H},
        {"name": "dof_vel", "dim": n, "history_length": H},
        {"name": "last_actions", "dim": n, "history_length": H},
    ]
    total = sum(s["dim"] * s["history_length"] for s in obs_layout)
    return {
        "obs_dim": total,
        "action_dim": n,
        "obs_layout": obs_layout,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _random_segments(rng, num_action=29):
    return {
        "command_joint_pos": rng.normal(size=num_action).astype(np.float32),
        "command_joint_vel": rng.normal(size=num_action).astype(np.float32),
        "motion_anchor_ori_b": rng.normal(size=6).astype(np.float32),
        "gyro": rng.normal(size=3).astype(np.float32),
        "joint_pos_rel": rng.normal(size=num_action).astype(np.float32),
        "dof_vel": rng.normal(size=num_action).astype(np.float32),
        "last_actions": rng.normal(size=num_action).astype(np.float32),
    }


class TestObsDim:
    def test_deploy_profile_dim_is_514(self, deploy_cfg):
        assert deploy_cfg["obs_dim"] == 514

    def test_layout_sum_matches_obs_dim(self, deploy_cfg):
        total = sum(s["dim"] * s.get("history_length", 1) for s in deploy_cfg["obs_layout"])
        assert total == deploy_cfg["obs_dim"]


class TestHistoryOrdering:
    """History blocks must read oldest-first, not newest-first."""

    def test_history_terms_carry_oldest_first(self, deploy_cfg, rng):
        """Spot-check the gyro history block manually: at step k>=H, the 5*3
        gyro slot of obs should be [gyro_{k-H+1}, gyro_{k-H+2}, ..., gyro_k]
        flattened, NOT the reverse."""
        layout = deploy_cfg["obs_layout"]

        gyros = [np.array([float(k), 0.0, 0.0], dtype=np.float32) for k in range(10)]
        all_segments = []
        for k in range(10):
            seg = _random_segments(rng)
            seg["gyro"] = gyros[k]
            all_segments.append(seg)

        obs = _deploy_compute_group(layout, all_segments)[-1]

        offset = 29 + 29 + 6
        gyro_block = obs[offset : offset + 3 * 5].reshape(5, 3)
        expected = np.stack(gyros[5:10])
        np.testing.assert_array_equal(gyro_block, expected)


class TestTrainingAssemblerVsDeploy:
    """Replicate training-side ObservationManager history and compare.

    The manager resets each term history by filling it with the current sample,
    then shifts oldest-first and appends each new value. Concatenation is
    term-major, matching the deploy-side observation layout.
    """

    @staticmethod
    def _training_actor_obs(history_buf, current_refs, hist_components, num_envs):
        parts = [
            current_refs["command_joint_pos"],
            current_refs["command_joint_vel"],
            current_refs["motion_anchor_ori_b"],
        ]
        for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
            buf = history_buf[key]
            parts.append(buf.reshape(num_envs, -1))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def test_training_path_matches_deploy(self, deploy_cfg, rng):
        n_env = 1
        n = deploy_cfg["action_dim"]
        H = 5

        buf = {
            "gyro": np.zeros((n_env, H, 3), dtype=np.float32),
            "joint_pos_rel": np.zeros((n_env, H, n), dtype=np.float32),
            "dof_vel": np.zeros((n_env, H, n), dtype=np.float32),
            "last_actions": np.zeros((n_env, H, n), dtype=np.float32),
        }

        all_segments = [_random_segments(rng) for _ in range(15)]
        deploy_seq = _deploy_compute_group(deploy_cfg["obs_layout"], all_segments)

        s0 = all_segments[0]
        for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
            buf[key][:, :, :] = s0[key][None, None, :]
        refs0 = {
            "command_joint_pos": s0["command_joint_pos"][None, :],
            "command_joint_vel": s0["command_joint_vel"][None, :],
            "motion_anchor_ori_b": s0["motion_anchor_ori_b"][None, :],
        }
        train_obs0 = self._training_actor_obs(buf, refs0, s0, n_env)
        np.testing.assert_array_equal(train_obs0[0], deploy_seq[0])

        for k in range(1, len(all_segments)):
            sk = all_segments[k]
            for key in ("gyro", "joint_pos_rel", "dof_vel", "last_actions"):
                buf[key][:, :-1] = buf[key][:, 1:]
                buf[key][:, -1] = sk[key][None, :]
            refs = {
                "command_joint_pos": sk["command_joint_pos"][None, :],
                "command_joint_vel": sk["command_joint_vel"][None, :],
                "motion_anchor_ori_b": sk["motion_anchor_ori_b"][None, :],
            }
            train_obs = self._training_actor_obs(buf, refs, sk, n_env)
            np.testing.assert_array_equal(
                train_obs[0], deploy_seq[k], err_msg=f"training <-> deploy mismatch at step {k}"
            )


class TestBackCompat:
    """H=1 ('no history') must reproduce the pre-history 154-d path bit-exact."""

    @pytest.fixture
    def legacy_layout(self):
        n = 29
        return [
            {"name": "command_joint_pos", "dim": n, "history_length": 1},
            {"name": "command_joint_vel", "dim": n, "history_length": 1},
            {"name": "motion_anchor_ori_b", "dim": 6, "history_length": 1},
            {"name": "gyro", "dim": 3, "history_length": 1},
            {"name": "joint_pos_rel", "dim": n, "history_length": 1},
            {"name": "dof_vel", "dim": n, "history_length": 1},
            {"name": "last_actions", "dim": n, "history_length": 1},
        ]

    def test_legacy_obs_dim_154(self, legacy_layout):
        total = sum(s["dim"] * s["history_length"] for s in legacy_layout)
        assert total == 154

    def test_legacy_matches_simple_concat(self, legacy_layout, rng):
        """With H=1 every term holds exactly the current value, so the
        assembled vector is a plain concat in layout order."""
        all_segments = [_random_segments(rng) for _ in range(5)]
        assembled = _deploy_compute_group(legacy_layout, all_segments)

        for seg, obs in zip(all_segments, assembled):
            expected = np.concatenate([seg[s["name"]] for s in legacy_layout])
            np.testing.assert_array_equal(obs, expected)
