from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np

from .fb_core.fb_cpr_aux.agent import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig
from .fb_core.fb_cpr_aux.model import FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from .fb_core.nn_filters import DictInputFilterConfig
from .fb_core.nn_models import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from .fb_core.normalizers import BatchNormNormalizerConfig, ObsNormalizerConfig

OBS_KEYS: tuple[str, ...] = ("state", "privileged_state", "last_action", "history_actor")
FORWARD_KEYS = ["state", "privileged_state", "last_action", "history_actor"]
ACTOR_KEYS = ["state", "last_action", "history_actor"]
BACKWARD_KEYS = ["state", "privileged_state"]
DISC_KEYS = ["state", "privileged_state"]

_B_MODE = os.environ.get("UNILAB_BFM_B_INPUT", "default")
if _B_MODE == "novel":
    OBS_KEYS = OBS_KEYS + ("state_novel",)
    BACKWARD_KEYS = ["state_novel", "privileged_state"]
elif _B_MODE == "state_only":
    BACKWARD_KEYS = ["state"]
elif _B_MODE == "novel_only":
    OBS_KEYS = OBS_KEYS + ("state_novel",)
    BACKWARD_KEYS = ["state_novel"]
elif _B_MODE != "default":
    raise ValueError(
        f"UNILAB_BFM_B_INPUT={_B_MODE!r}; expected one of "
        "'default', 'novel', 'state_only', 'novel_only'"
    )

NONWRIST_23 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25]

DEFAULT_AUX_REWARDS = [
    "penalty_torques",
    "penalty_action_rate",
    "limits_dof_pos",
    "limits_torque",
    "penalty_undesired_contact",
    "penalty_feet_ori",
    "penalty_ankle_roll",
    "penalty_slippage",
]

DEFAULT_AUX_SCALING = {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
}


def _bn(momentum: float = 0.01) -> BatchNormNormalizerConfig:
    return BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=momentum)


def _filter(keys: list[str]) -> DictInputFilterConfig:
    return DictInputFilterConfig(name="DictInputFilterConfig", key=keys)


def build_bfm_agent_config(algo_cfg: dict[str, Any]) -> FBcprAuxAgentConfig:

    m = algo_cfg["model"]
    a = algo_cfg["algorithm"]
    device = str(algo_cfg.get("buffer_device", "cuda"))

    def forward(keys: list[str]) -> ForwardArchiConfig:
        return ForwardArchiConfig(
            name="ForwardArchi",
            hidden_dim=m["hidden_dim"],
            model="residual",
            hidden_layers=m["hidden_layers"],
            embedding_layers=m["embedding_layers"],
            num_parallel=m["num_parallel"],
            ensemble_mode="batch",
            input_filter=_filter(keys),
        )

    archi = FBcprAuxModelArchiConfig(
        name="FBcprAuxModelArchiConfig",
        z_dim=m["z_dim"],
        norm_z=m["norm_z"],
        f=forward(FORWARD_KEYS),
        b=BackwardArchiConfig(
            name="BackwardArchi",
            hidden_dim=m["backward_hidden_dim"],
            hidden_layers=m["backward_hidden_layers"],
            norm=True,
            input_filter=_filter(BACKWARD_KEYS),
        ),
        actor=ActorArchiConfig(
            name="actor",
            model="residual",
            hidden_dim=m["hidden_dim"],
            hidden_layers=m["hidden_layers"],
            embedding_layers=m["embedding_layers"],
            input_filter=_filter(ACTOR_KEYS),
        ),
        critic=forward(FORWARD_KEYS),
        discriminator=DiscriminatorArchiConfig(
            name="DiscriminatorArchi",
            hidden_dim=m["discriminator_hidden_dim"],
            hidden_layers=m["discriminator_hidden_layers"],
            input_filter=_filter(DISC_KEYS),
        ),
        aux_critic=forward(FORWARD_KEYS),
    )

    model = FBcprAuxModelConfig(
        name="FBcprAuxModel",
        device=device,
        archi=archi,
        obs_normalizer=ObsNormalizerConfig(
            name="ObsNormalizerConfig",
            normalizers={k: _bn() for k in OBS_KEYS},
            allow_mismatching_keys=True,
        ),
        inference_batch_size=500000,
        seq_length=m["seq_length"],
        actor_std=m["actor_std"],
        amp=False,
        norm_aux_reward=RewardNormalizerConfig(
            name="RewardNormalizer", translate=False, scale=True
        ),
    )

    train = FBcprAuxAgentTrainConfig(
        name="FBcprAuxAgentTrainConfig",
        lr_f=a["lr_f"],
        lr_b=a["lr_b"],
        lr_actor=a["lr_actor"],
        lr_critic=a["lr_critic"],
        lr_discriminator=a["lr_discriminator"],
        lr_aux_critic=a["lr_aux_critic"],
        weight_decay=0.0,
        clip_grad_norm=0.0,
        fb_target_tau=a["fb_target_tau"],
        critic_target_tau=a["critic_target_tau"],
        ortho_coef=a["ortho_coef"],
        train_goal_ratio=a["train_goal_ratio"],
        fb_pessimism_penalty=a["fb_pessimism_penalty"],
        actor_pessimism_penalty=a["actor_pessimism_penalty"],
        critic_pessimism_penalty=a["critic_pessimism_penalty"],
        aux_critic_pessimism_penalty=a["aux_critic_pessimism_penalty"],
        stddev_clip=a["stddev_clip"],
        q_loss_coef=0.0,
        batch_size=a["batch_size"],
        discount=a["discount"],
        use_mix_rollout=True,
        update_z_every_step=a["update_z_every_step"],
        z_buffer_size=a["z_buffer_size"],
        rollout_expert_trajectories=a["rollout_expert_trajectories"],
        rollout_expert_trajectories_length=a["rollout_expert_trajectories_length"],
        rollout_expert_trajectories_percentage=a["rollout_expert_trajectories_percentage"],
        expert_asm_ratio=a["expert_asm_ratio"],
        relabel_ratio=a["relabel_ratio"],
        grad_penalty_discriminator=a["grad_penalty_discriminator"],
        weight_decay_discriminator=0.0,
        reg_coeff=a["reg_coeff"],
        reg_coeff_aux=a["reg_coeff_aux"],
        scale_reg=a["scale_reg"],
        scale_reg_mode=a["scale_reg_mode"],
        reg_denom_floor=a["reg_denom_floor"],
    )

    return FBcprAuxAgentConfig(
        name="FBcprAuxAgent",
        model=model,
        train=train,
        aux_rewards=DEFAULT_AUX_REWARDS,
        aux_rewards_scaling=DEFAULT_AUX_SCALING,
        cudagraphs=m["cudagraphs"],
        compile=m["compile"],
    )


class BFMRunner:
    def __init__(
        self, algo_cfg: dict[str, Any], train_cfg: dict[str, Any], expert_cfg: dict[str, Any]
    ):
        self.algo_cfg = algo_cfg
        self.train_cfg = train_cfg
        self.expert_cfg = dict(expert_cfg) if expert_cfg else {}
        self._resolve_expert_data_paths()
        self.agent_cfg = build_bfm_agent_config(algo_cfg)
        self.agent = None
        self.env = None

    def _resolve_expert_data_paths(self) -> None:

        pkl = self.expert_cfg.get("lafan_pkl")
        if not pkl:
            return
        try:
            from unilab.assets.hub import resolve_motion_files

            resolved = resolve_motion_files(pkl)
            if resolved != pkl:
                print(f"[bfm] expert.lafan_pkl resolved: {pkl} -> {resolved}")
            self.expert_cfg["lafan_pkl"] = resolved
        except Exception as exc:
            print(f"[bfm] could not resolve expert.lafan_pkl ({pkl!r}): {exc}")

    def _build_env(self, num_envs: int | None = None):
        from unilab.base.registry import make

        ov = {}
        if (self.expert_cfg or {}).get("lafan_pkl"):
            ov["lafan_pkl"] = self.expert_cfg["lafan_pkl"]
            if self.expert_cfg.get("max_motions") is not None:
                ov["max_motions"] = self.expert_cfg["max_motions"]
        return make(
            self.train_cfg["task_name"],
            sim_backend=self.train_cfg.get("sim_backend", "mujoco"),
            num_envs=int(num_envs) if num_envs is not None else self.algo_cfg["num_envs"],
            env_cfg_override=ov or None,
        )

    def _build_agent(self, obs_space: gym.Space, action_dim: int):
        return self.agent_cfg.build(obs_space=obs_space, action_dim=action_dim)

    def _obs_space(self, spec: dict[str, int]) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {k: gym.spaces.Box(-1e6, 1e6, (d,), np.float32) for k, d in spec.items()}
        )

    def _setup_logger(self):

        log_dir = (
            self.train_cfg.get("log_dir") or f"logs/bfm/{self.train_cfg.get('task_name', 'G1Bfm')}"
        )
        self._log_dir = log_dir
        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)
            print(f"[bfm] tensorboard: tensorboard --logdir {log_dir}")
        except Exception as exc:
            print(f"[bfm] tensorboard unavailable: {exc}")
        self._wandb = None
        if self.train_cfg.get("logger") == "wandb":
            try:
                import wandb

                wandb.init(
                    project=self.train_cfg.get("wandb_project", "unilab"),
                    group=self.train_cfg.get("wandb_group"),
                    config=self.algo_cfg,
                    dir=log_dir,
                )
                self._wandb = wandb
            except Exception as exc:
                print(f"[bfm] wandb unavailable: {exc}")
        return writer

    def _save_checkpoint(self, step: int, replay=None) -> None:

        import json
        from pathlib import Path

        ckpt = Path(self._log_dir) / "checkpoint"
        ckpt.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(ckpt))
        with (ckpt / "train_status.json").open("w") as f:
            json.dump({"time": int(step)}, f, indent=4)

        with (Path(self._log_dir) / "config.json").open("w") as f:
            json.dump(
                {"algo": self.algo_cfg, "training": self.train_cfg, "expert": self.expert_cfg},
                f,
                indent=4,
                default=str,
            )
        if replay is not None and self.algo_cfg.get("checkpoint_buffer", True):
            try:
                replay["train"].save(ckpt / "buffers" / "train")
            except Exception as exc:
                print(f"[bfm] buffer save skipped ({exc})")
        print(f"[bfm] checkpoint saved @ step {step} -> {ckpt}")

    def _log(self, writer, metrics: dict, step: int) -> None:
        if writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(k, float(v), step)
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def _build_eval_refs(self, env, n_motions: int | None = None):

        from pathlib import Path as _Path

        import numpy as np

        from unilab.envs.locomotion.g1.bfm import EXTEND_BODIES

        from .lafan_loader import (
            _ang_vel_from_quat,
            _make_model_and_ids,
            _motion_fields,
            _motion_frames,
            _resample_to_ctrl_rate,
            _xyzw2wxyz,
        )

        n_motions = int(self.algo_cfg.get("eval_n_motions", n_motions or 16))
        ctrl_dt = float(env._cfg.ctrl_dt)
        ctrl_fps = 1.0 / ctrl_dt
        lafan = (self.expert_cfg or {}).get("lafan_pkl")

        try:
            from unilab.base.registry import make

            self._eval_env = make(
                self.train_cfg["task_name"],
                sim_backend=self.train_cfg.get("sim_backend", "mujoco"),
                num_envs=1,
            )
            self._eval_env.init_state()
        except Exception as exc:
            print(f"[bfm] eval env unavailable ({exc}); EMD eval disabled")
            self._eval_env = None
        if not lafan or not _Path(lafan).exists():
            print("[bfm] no lafan pkl for eval refs; EMD eval disabled")
            return [], None

        import joblib

        body_names, foot_names = tuple(env._cfg.body_names), tuple(env._cfg.foot_names)
        b2i = {nm: int(env._backend.get_body_ids([nm])[0]) for nm in body_names}
        body_idx = np.array([b2i[n] for n in body_names], dtype=np.int64)
        model, data, mj_ids, foot_local, extends = _make_model_and_ids(
            env._cfg.scene.model_file, body_names, foot_names, EXTEND_BODIES
        )
        motions = joblib.load(lafan)
        keys = list(motions.keys())[:n_motions]
        refs = []
        for key in keys:
            rt, rr, dof, fps = _motion_fields(motions[key])

            frames = _motion_frames(
                model,
                data,
                mj_ids,
                foot_local,
                extends,
                rt,
                rr,
                dof,
                fps,
                env._default_dof,
                env._num_action,
                ctrl_dt,
            )
            obs_seq = {k: np.stack([f[k] for f in frames]).astype(np.float32) for k in frames[0]}

            rt_r, rr_r, dof_r = _resample_to_ctrl_rate(rt, rr, dof, fps, ctrl_dt)

            qpos0 = np.concatenate([rt_r[0], _xyzw2wxyz(rr_r[0]), dof_r[0]]).astype(np.float32)
            from .bfm_obs import quat_rotate_inverse

            root_lin0 = (rt_r[1] - rt_r[0]) * ctrl_fps if len(rt_r) > 1 else np.zeros(3, np.float32)

            root_ang0 = quat_rotate_inverse(rr_r[0:1], _ang_vel_from_quat(rr_r, ctrl_fps)[0:1])[0]
            dof_v0 = (
                (dof_r[1] - dof_r[0]) * ctrl_fps
                if len(dof_r) > 1
                else np.zeros(env._num_action, np.float32)
            )
            qvel0 = np.concatenate([root_lin0, root_ang0, dof_v0]).astype(np.float32)
            refs.append(
                {
                    "obs_seq": obs_seq,
                    "ref_dof": dof_r.astype(np.float32),
                    "qpos0": qpos0,
                    "qvel0": qvel0,
                    "T": dof_r.shape[0],
                }
            )
        if refs:
            print(
                f"[bfm] eval refs: {len(refs)} lafan motions @ {ctrl_fps:.0f}Hz, ~{refs[0]['T']} frames each"
            )
        return refs, body_idx

    @staticmethod
    def _emd_ot(a: "np.ndarray", b: "np.ndarray") -> float:

        import ot

        Xn = (a**2).sum(1)[:, None]
        Yn = (b**2).sum(1)[None, :]
        cost = np.sqrt(np.clip(Xn + Yn - 2.0 * a @ b.T, 0.0, None))
        wa = np.ones(a.shape[0]) / a.shape[0]
        wb = np.ones(b.shape[0]) / b.shape[0]
        return float(ot.emd2(wa, wb, cost, numItermax=100000))

    def _compute_priorities(self, per_motion, device):

        import torch

        c = self.algo_cfg
        ids = [int(x["motion_id"]) for x in per_motion]
        pr = torch.clamp(
            torch.tensor([x["emd"] for x in per_motion], dtype=torch.float32, device=device),
            min=float(c.get("prioritization_min_val", 0.5)),
            max=float(c.get("prioritization_max_val", 2.0)),
        ) * float(c.get("prioritization_scale", 2.0))
        mode = str(c.get("prioritization_mode", "exp"))
        if mode == "exp":
            pr = 2**pr
        elif mode == "lin":
            pass
        elif mode == "bin":
            bins = torch.floor(pr)
            for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                mask = bins == i
                n = int(mask.sum().item())
                if n > 0:
                    pr[mask] = 1.0 / n
        else:
            raise ValueError(f"Unsupported prioritization_mode {mode}")
        return ids, pr

    def _eval_tracking_emd(self, agent, refs, body_idx) -> dict[str, float]:

        import torch

        device = str(self.algo_cfg.get("buffer_device", "cuda"))
        eenv = getattr(self, "_eval_env", None)
        if not refs or eenv is None:
            return {"emd": float("nan")}
        scale = eenv._cfg.control_config.action_scale
        substeps = eenv._cfg.sim_substeps
        emds, emds_state23, emds_dof29, mpjpes = [], [], [], []

        vel_dists, accel_dists, proximities, successes = [], [], [], []

        root_h_errs, upright_fracs = [], []

        import os as _os

        _legacy_z = _os.environ.get("BFM_EVAL_Z_LEGACY") == "1"
        _shift = 0 if _legacy_z else 1

        _perdim = _os.environ.get("BFM_EVAL_PERDIM") == "1"
        _perdim_abs = np.zeros(29, np.float64)
        _perdim_n = 0

        _step0 = _os.environ.get("BFM_EVAL_STEP0") == "1"
        _step0_abs = np.zeros(29, np.float64)
        _step0_n = 0
        _step0_curve = []
        with torch.no_grad():
            for r in refs:
                obs_t = {
                    k: torch.as_tensor(r["obs_seq"][k], dtype=torch.float32, device=device)
                    for k in r["obs_seq"]
                }

                if _legacy_z:
                    z_seq = agent._model.tracking_inference(obs_t)
                else:
                    obs_next = {k: v[1:] for k, v in obs_t.items()}
                    z_seq = agent._model.project_z(agent._model.backward_map(obs_next))
                Tref = z_seq.shape[0]
                eenv._backend.set_state(np.array([0]), r["qpos0"][None], r["qvel0"][None])
                eenv._obs_builder.reset_envs(np.array([0]))
                eenv._last_action[:] = 0.0
                if _step0:
                    _reset_dof = np.asarray(eenv._backend.get_dof_pos()[0], np.float32)
                    _step0_abs += np.abs(_reset_dof - r["ref_dof"][0])
                    _step0_n += 1
                ach_dof, ach_state, ach_rooth = [], [], []
                for t in range(Tref):
                    obs = eenv._compute_obs(push=True)
                    o = {
                        k: torch.as_tensor(obs[k], dtype=torch.float32, device=device) for k in obs
                    }
                    zt = z_seq[t % Tref].unsqueeze(0)
                    a = agent.act(o, zt, mean=True).cpu().numpy().astype(np.float32)
                    a_norm = np.clip(
                        a * eenv.NORMALIZE_ACTION_TO, -eenv.ACTION_CLIP, eenv.ACTION_CLIP
                    )

                    eenv._last_action = a_norm
                    eenv._backend.step(
                        a_norm * scale * eenv._act_rescale + eenv._default_dof, substeps
                    )

                    ach_dof.append(np.array(eenv._backend.get_dof_pos()[0], np.float32))
                    ach_state.append(np.asarray(obs["state"][0], np.float32))

                    ach_rooth.append(float(eenv._backend.get_physics_state()[0, 3]))
                ach_dof = np.stack(ach_dof)
                ach_state = np.stack(ach_state)

                ref_dof = r["ref_dof"][_shift : _shift + ach_dof.shape[0]]
                ref_state = r["obs_seq"]["state"][_shift : _shift + ach_state.shape[0]]

                emds.append(self._emd_ot(ach_state[:, NONWRIST_23], ref_state[:, NONWRIST_23]))

                emds_state23.append(self._emd_ot(ach_state[:, :23], ref_state[:, :23]))

                emds_dof29.append(self._emd_ot(ach_dof, ref_dof))
                n = min(ach_dof.shape[0], ref_dof.shape[0])
                mpjpes.append(
                    float(np.linalg.norm(ach_dof[:n] - ref_dof[:n], axis=-1).mean() * 1000.0)
                )

                _a, _b = ach_dof[:n], ref_dof[:n]
                if n >= 2:
                    vel_dists.append(
                        float(
                            np.linalg.norm((_a[1:] - _a[:-1]) - (_b[1:] - _b[:-1]), axis=-1).mean()
                            * 1000.0
                        )
                    )
                if n >= 3:
                    accel_dists.append(
                        float(
                            np.linalg.norm(
                                (_a[:-2] - 2.0 * _a[1:-1] + _a[2:])
                                - (_b[:-2] - 2.0 * _b[1:-1] + _b[2:]),
                                axis=-1,
                            ).mean()
                            * 100.0
                        )
                    )

                _d = np.linalg.norm(_a - _b, axis=-1)
                _inb = _d <= 2.0
                _outb = _d > 4.0
                proximities.append(float((_inb + ((4.0 - _d) / 2.0) * (~_inb) * (~_outb)).mean()))

                successes.append(float(_inb.min()))

                _ref_rh = r["obs_seq"]["privileged_state"][_shift : _shift + len(ach_rooth), 0]
                _ach_rh = np.asarray(ach_rooth, np.float32)
                _m = min(len(_ach_rh), len(_ref_rh))
                if _m > 0:
                    root_h_errs.append(float(np.abs(_ach_rh[:_m] - _ref_rh[:_m]).mean()))
                    upright_fracs.append(float((_ach_rh[:_m] > 0.7 * _ref_rh[:_m]).mean()))
                if _perdim:
                    _perdim_abs += np.abs(ach_dof[:n] - ref_dof[:n]).mean(axis=0)
                    _perdim_n += 1
                if _step0:
                    perc = np.abs(ach_dof[:n] - ref_dof[:n]).mean(axis=1)
                    _step0_curve.append(perc[:20])
        if _perdim and _perdim_n:
            pd = _perdim_abs / _perdim_n
            _JN = [
                "L_hip_p",
                "L_hip_r",
                "L_hip_y",
                "L_knee",
                "L_ank_p",
                "L_ank_r",
                "R_hip_p",
                "R_hip_r",
                "R_hip_y",
                "R_knee",
                "R_ank_p",
                "R_ank_r",
                "waist_y",
                "waist_r",
                "waist_p",
                "L_sho_p",
                "L_sho_r",
                "L_sho_y",
                "L_elbow",
                "L_wri_r",
                "L_wri_p",
                "L_wri_y",
                "R_sho_p",
                "R_sho_r",
                "R_sho_y",
                "R_elbow",
                "R_wri_r",
                "R_wri_p",
                "R_wri_y",
            ]
            print(f"[bfm][PERDIM] over {_perdim_n} motions, per-dof mean|achieved-ref| (rad):")
            for i in range(29):
                print(f"    dim{i:2d} {_JN[i]:8s} = {pd[i]:.4f}")
            la, ra = pd[15:22], pd[22:29]
            print(
                f"[bfm][PERDIM] LEFT-arm (15-21) mean={la.mean():.4f}  RIGHT-arm (22-28) mean={ra.mean():.4f}  R/L={ra.mean() / max(la.mean(), 1e-9):.2f}x"
            )
            print(
                "[bfm][PERDIM] per-joint L-vs-R: "
                + ", ".join(f"{_JN[15 + k][2:]}={la[k]:.3f}/{ra[k]:.3f}" for k in range(7))
            )
        if _step0 and _step0_n:
            s0 = _step0_abs / _step0_n
            _JN2 = [
                "L_hip_p",
                "L_hip_r",
                "L_hip_y",
                "L_knee",
                "L_ank_p",
                "L_ank_r",
                "R_hip_p",
                "R_hip_r",
                "R_hip_y",
                "R_knee",
                "R_ank_p",
                "R_ank_r",
                "waist_y",
                "waist_r",
                "waist_p",
                "L_sho_p",
                "L_sho_r",
                "L_sho_y",
                "L_elbow",
                "L_wri_r",
                "L_wri_p",
                "L_wri_y",
                "R_sho_p",
                "R_sho_r",
                "R_sho_y",
                "R_elbow",
                "R_wri_r",
                "R_wri_p",
                "R_wri_y",
            ]
            print(
                f"[bfm][STEP0] reset-pose dof error vs ref_dof[0], over {_step0_n} motions (rad):"
            )
            print(
                f"[bfm][STEP0]   mean over 29 dof = {s0.mean():.5f}  max = {s0.max():.5f} @ {_JN2[int(s0.argmax())]}"
            )
            print(
                "[bfm][STEP0]   per-dof: " + ", ".join(f"{_JN2[i]}={s0[i]:.3f}" for i in range(29))
            )

            _cur = np.stack([c for c in _step0_curve if len(c) >= 20])
            if len(_cur):
                m = _cur.mean(axis=0)
                print(
                    "[bfm][STEP0]   rollout mean|err| per-t (t=0..19): "
                    + " ".join(f"{v:.3f}" for v in m)
                )
        return {
            "emd": float(np.mean(emds_dof29)),
            "emd_state23": float(np.mean(emds_state23)),
            "emd_nonwrist23": float(np.mean(emds)),
            "emd_dof29": float(np.mean(emds_dof29)),
            "mpjpe_l": float(np.mean(mpjpes)),
            "distance": float(np.mean(mpjpes)) / 1000.0,
            "vel_dist": float(np.mean(vel_dists)) if vel_dists else float("nan"),
            "accel_dist": float(np.mean(accel_dists)) if accel_dists else float("nan"),
            "proximity": float(np.mean(proximities)) if proximities else float("nan"),
            "success": float(np.mean(successes)) if successes else float("nan"),
            "root_h_err": float(np.mean(root_h_errs)) if root_h_errs else float("nan"),
            "upright_frac": float(np.mean(upright_fracs)) if upright_fracs else float("nan"),
            "emd_over_dist": (
                float(np.mean(emds_dof29)) / (float(np.mean(mpjpes)) / 1000.0)
                if np.mean(mpjpes) > 0
                else float("nan")
            ),
            "per_motion": [{"motion_id": j, "emd": float(e)} for j, e in enumerate(emds_dof29)],
        }

    def learn(self, max_iterations: int | None = None) -> None:

        import torch

        from unilab.training.seed import apply_training_seed

        _seed = self.algo_cfg.get("seed")
        if _seed is not None:
            apply_training_seed(int(_seed), torch_runtime=True, cuda=True)
            print(f"[bfm] seed applied: {int(_seed)} (torch/cuda/numpy/random)")

        device = str(self.algo_cfg.get("buffer_device", "cuda"))
        env = self._build_env()

        if bool(self.algo_cfg.get("add_obs_noise", True)) and hasattr(env, "_add_obs_noise"):
            env._add_obs_noise = True
            print("[bfm] obs noise ENABLED on training collector (BFM-Zero noise_scales)")
        num_envs = int(getattr(env, "num_envs", self.algo_cfg["num_envs"]))
        n_act = int(env._num_action)
        spec = env.obs_groups_spec
        agent = self._build_agent(self._obs_space(spec), n_act)
        self.agent, self.env = agent, env

        base_step = 0
        resume = self.train_cfg.get("resume") or self.train_cfg.get("load_run")
        if resume and str(resume) not in ("", "-1", "None"):
            from pathlib import Path

            ck = Path(resume)
            ck = ck if ck.name == "checkpoint" else ck / "checkpoint"
            if ck.exists():
                import json as _json

                import safetensors.torch as _st

                _st.load_model(
                    agent._model,
                    str(ck / "model" / "model.safetensors"),
                    device=device,
                    strict=False,
                )
                opt = torch.load(
                    str(ck / "optimizers.pth"), weights_only=False, map_location=device
                )
                for k, v in opt.items():
                    if hasattr(agent, k):
                        getattr(agent, k).load_state_dict(v)
                try:
                    base_step = int(_json.load(open(ck / "train_status.json"))["time"])
                except Exception:
                    base_step = 0
                print(f"[bfm] RESUMED from {ck} @ step {base_step}")
            else:
                print(f"[bfm] resume path {ck} not found; training from scratch")

        from .fb_core.buffers.trajectory import TrajectoryDictBufferMultiDim

        cap = max(1, int(self.algo_cfg["buffer_size"]) // num_envs)
        seq_len = int(self.algo_cfg["model"]["seq_length"])

        train_buf = TrajectoryDictBufferMultiDim(
            capacity=cap,
            device=device,
            n_dim=2,
            end_key="truncated",
            seq_length=seq_len,
            output_key_t=[
                "observation",
                "action",
                "z",
                "terminated",
                "truncated",
                "step_count",
                "reward",
                "aux_rewards",
            ],
            output_key_tp1=["observation", "terminated"],
        )
        replay = {"train": train_buf}

        from pathlib import Path as _Path

        self._use_real_expert = False
        ecap = max(1024, cap)
        lafan_pkl = (self.expert_cfg or {}).get("lafan_pkl")
        motion_glob = (self.expert_cfg or {}).get("motion_glob")
        max_motions = (self.expert_cfg or {}).get("max_motions")
        expert_buf = None
        if lafan_pkl and _Path(lafan_pkl).exists():
            try:
                from unilab.envs.locomotion.g1.bfm import EXTEND_BODIES

                from .lafan_loader import build_expert_trajectory_buffer

                expert_buf = build_expert_trajectory_buffer(
                    pkl_path=lafan_pkl,
                    body_names=tuple(env._cfg.body_names),
                    foot_names=tuple(env._cfg.foot_names),
                    extends_cfg=EXTEND_BODIES,
                    mj_xml=env._cfg.scene.model_file,
                    default_dof=env._default_dof,
                    num_dof=n_act,
                    device=device,
                    seq_length=seq_len,
                    max_motions=max_motions,
                    ctrl_dt=float(env._cfg.ctrl_dt),
                )
                self._use_real_expert = len(expert_buf) > 0
                print(f"[bfm] expert TRAJECTORY buffer (lafan): {len(expert_buf)} motions/episodes")
            except Exception as exc:
                print(f"[bfm] lafan pkl load failed ({exc}); trying NPZ")
                expert_buf = None
        if expert_buf is None and motion_glob:
            try:
                from .expert_data import build_expert_buffer

                b2i = {nm: int(env._backend.get_body_ids([nm])[0]) for nm in env._cfg.body_names}
                from unilab.envs.locomotion.g1.bfm import EXTEND_BODIES

                expert_buf = build_expert_buffer(
                    motion_glob=motion_glob,
                    body_names=tuple(env._cfg.body_names),
                    foot_names=tuple(env._cfg.foot_names),
                    body_name_to_idx=b2i,
                    default_dof=env._default_dof,
                    num_dof=n_act,
                    device=device,
                    capacity=ecap,
                    extends_cfg=EXTEND_BODIES,
                )
                self._use_real_expert = len(expert_buf) > 0
                print(f"[bfm] expert buffer (motion NPZ): {len(expert_buf)} transitions")
            except Exception as exc:
                print(f"[bfm] real expert unavailable ({exc}); using rollout stand-in")
                expert_buf = None
        if not self._use_real_expert:
            raise RuntimeError(
                "[bfm] no expert motion data available — FB-CPR-Aux cannot train on an empty "
                "expert buffer (the discriminator / tracking-z path samples it every update).\n"
                f"  expert.lafan_pkl  = {lafan_pkl!r} (not found / not resolvable)\n"
                f"  expert.motion_glob = {motion_glob!r} (matched no NPZ)\n"
                "The default config pulls 'motions/g1/lafan_29dof_10s-clipped.pkl' from the HF "
                "motions repo 'unilabsim/unilab-motions'; ensure network / HF access, or point "
                "UNILAB_BFM_LAFAN_PKL / UNILAB_BFM_MOTIONS at a local source. See conf/bfm/config.yaml."
            )
        replay["expert_slicer"] = expert_buf

        seed_iters = max(1, int(self.algo_cfg["num_seed_steps"]) // num_envs)
        update_every = max(1, int(self.algo_cfg["update_agent_every"]) // num_envs)
        eval_every = max(1, int(self.algo_cfg["eval_every_steps"]) // num_envs)
        ckpt_every = max(1, int(self.algo_cfg["checkpoint_every_steps"]) // num_envs)
        max_iters = max_iterations or (int(self.algo_cfg["num_env_steps"]) // num_envs)

        writer = self._setup_logger()
        eval_refs, eval_body_idx = self._build_eval_refs(env)

        if (
            bool(self.algo_cfg.get("prioritization", False))
            and self._use_real_expert
            and hasattr(expert_buf, "motion_ids")
            and len(eval_refs) != len(expert_buf.motion_ids)
        ):
            raise ValueError(
                f"prioritization=true requires eval_n_motions ({len(eval_refs)}) == expert motion "
                f"count ({len(expert_buf.motion_ids)}). Set algo.eval_n_motions="
                f"{len(expert_buf.motion_ids)} (or expert.max_motions to match), or set "
                f"algo.prioritization=false."
            )

        def to_t(obs):
            return {
                k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()
            }

        state = env.init_state()
        context = None
        step_count = torch.zeros((num_envs, 1), dtype=torch.long, device=device)
        rng = np.random.RandomState(int(self.algo_cfg.get("seed", 0)))
        act_low, act_high = env.action_space.low, env.action_space.high

        print(
            f"[bfm] start: num_envs={num_envs} seed_iters={seed_iters} max_iters={max_iters} device={device}"
        )
        for it in range(max_iters):
            context = agent.maybe_update_rollout_context(
                z=context, step_count=step_count, replay_buffer=replay
            )
            if it < seed_iters:
                action_np = rng.uniform(act_low, act_high, (num_envs, n_act)).astype(np.float32)
            else:
                with torch.no_grad():
                    action_t = agent.act(to_t(state.obs), context, mean=False)
                action_np = action_t.detach().cpu().numpy().astype(np.float32)

            obs = state.obs
            next_state = env.step(action_np)
            term = np.asarray(next_state.terminated).reshape(-1, 1)
            trunc = np.asarray(next_state.truncated).reshape(-1, 1)
            zc = (
                context.detach()
                if context is not None
                else torch.zeros((num_envs, self.algo_cfg["model"]["z_dim"]), device=device)
            )

            def _td(x):
                return x.unsqueeze(0)

            data = {
                "observation": {k: _td(v) for k, v in to_t(obs).items()},
                "action": _td(torch.as_tensor(action_np, device=device)),
                "z": _td(zc),
                "step_count": _td(step_count.clone()),
                "reward": _td(torch.zeros((num_envs, 1), device=device)),
                "terminated": _td(torch.as_tensor(term, dtype=torch.bool, device=device)),
                "truncated": _td(torch.as_tensor(trunc, dtype=torch.bool, device=device)),
                "aux_rewards": {
                    k: _td(
                        torch.as_tensor(
                            np.asarray(v).reshape(-1, 1), dtype=torch.float32, device=device
                        )
                    )
                    for k, v in next_state.info.get("aux_rewards", {}).items()
                },
            }
            train_buf.extend(data)

            step_count = step_count + 1
            done = torch.as_tensor((term | trunc), dtype=torch.bool, device=device)
            step_count[done.reshape(-1)] = 0
            state = next_state

            if len(train_buf) > 0 and it >= seed_iters and (it % update_every == 0):
                for _ in range(int(self.algo_cfg["num_agent_updates"])):
                    metrics = agent.update(replay, it)
                self._log(
                    writer,
                    {f"train/{k}": float(v) for k, v in metrics.items()},
                    base_step + it * num_envs,
                )
                _STDOUT_KEYS = {"Q_fb", "fb_share", "r_disc", "r_aux", "w_disc"}
                to_print = {
                    k: round(float(v), 4)
                    for k, v in metrics.items()
                    if "loss" in k or k in _STDOUT_KEYS
                }
                print(f"[bfm] it={it} {to_print}")

            if it >= seed_iters and it % eval_every == 0:
                m = self._eval_tracking_emd(agent, eval_refs, eval_body_idx)
                self._log(
                    writer,
                    {
                        f"eval/humanoidverse_tracking_eval/{k}": v
                        for k, v in m.items()
                        if isinstance(v, (int, float))
                    },
                    base_step + it * num_envs,
                )
                print(
                    f"[bfm] EVAL it={it} emd={m['emd']:.4f} "
                    f"emd_dof29={m.get('emd_dof29', float('nan')):.4f} "
                    f"mpjpe_l={m.get('mpjpe_l', float('nan')):.1f} "
                    f"dist={m.get('distance', float('nan')):.4f} "
                    f"vel={m.get('vel_dist', float('nan')):.1f} "
                    f"accel={m.get('accel_dist', float('nan')):.1f} "
                    f"prox={m.get('proximity', float('nan')):.4f} "
                    f"succ={m.get('success', float('nan')):.4f} "
                    f"e/d={m.get('emd_over_dist', float('nan')):.4f} "
                    f"rh_err={m.get('root_h_err', float('nan')):.3f} "
                    f"up={m.get('upright_frac', float('nan')):.3f}"
                )

                if (
                    bool(self.algo_cfg.get("prioritization", False))
                    and self._use_real_expert
                    and hasattr(expert_buf, "motion_ids")
                ):
                    motion_ids_buf = list(expert_buf.motion_ids)
                    assert len(m["per_motion"]) == len(motion_ids_buf), (
                        f"prioritization needs FULL-SET eval: {len(m['per_motion'])} eval motions vs "
                        f"{len(motion_ids_buf)} expert motions — set algo.eval_n_motions="
                        f"{len(motion_ids_buf)}"
                    )
                    index_in_buffer = {int(mid): i for i, mid in enumerate(motion_ids_buf)}
                    ids, pr = self._compute_priorities(m["per_motion"], device)
                    idxs = torch.tensor(
                        [index_in_buffer[i] for i in ids], dtype=torch.long, device=device
                    )

                    expert_buf.update_priorities(
                        priorities=pr.to(expert_buf.device), idxs=idxs.to(expert_buf.device)
                    )

                    mb = getattr(env, "_motion_bank", None)
                    if mb is not None:
                        assert mb.num_motions == len(motion_ids_buf), (
                            mb.num_motions,
                            len(motion_ids_buf),
                        )
                        w = np.zeros(mb.num_motions, np.float32)
                        w[idxs.cpu().numpy()] = pr.detach().cpu().numpy()
                        mb.set_sampling_weights(w)
                    print(
                        f"[bfm] prioritization updated: p in [{float(pr.min()):.2f}, {float(pr.max()):.2f}]"
                    )

            if it > 0 and it % ckpt_every == 0:
                self._save_checkpoint(base_step + it * num_envs, replay)

        self._save_checkpoint(base_step + max_iters * num_envs, replay)
        if writer is not None:
            writer.close()
        print("[bfm] training loop finished")

    def play(self) -> None:

        import json as _json
        from pathlib import Path

        import safetensors.torch as _st
        import torch

        device = str(self.algo_cfg.get("buffer_device", "cuda"))

        env = self._build_env(num_envs=1)
        if hasattr(env, "_add_obs_noise"):
            env._add_obs_noise = False
        env.init_state()
        agent = self._build_agent(self._obs_space(env.obs_groups_spec), int(env._num_action))
        self.agent, self.env = agent, env

        src = (
            self.train_cfg.get("resume")
            or self.train_cfg.get("load_run")
            or self.algo_cfg.get("load_run")
            or self.train_cfg.get("log_dir")
        )
        if not src or str(src) in ("", "-1", "None"):
            raise ValueError(
                "play(): set training.resume=<run_dir> (or training.log_dir) to the run to evaluate"
            )
        ck = Path(src)
        ck = ck if ck.name == "checkpoint" else ck / "checkpoint"
        model_path = ck / "model" / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"play(): no checkpoint at {model_path}")
        _load_ret = _st.load_model(agent._model, str(model_path), device=device, strict=False)
        try:
            _missing, _unexpected = _load_ret
            print(
                f"[bfm] load_model strict=False: missing={len(_missing)} "
                f"unexpected={len(_unexpected)}"
            )
            if _missing:
                print("  missing sample:", list(_missing)[:25])
            if _unexpected:
                print("  unexpected sample:", list(_unexpected)[:25])
        except Exception as _e:
            print(f"[bfm] load_model return introspection failed: {_e}")
        step = 0
        try:
            step = int(_json.load(open(ck / "train_status.json"))["time"])
        except Exception:
            pass
        print(f"[bfm] EVAL-ONLY: loaded {model_path} @ step {step}")

        eval_refs, eval_body_idx = self._build_eval_refs(env)
        m = self._eval_tracking_emd(agent, eval_refs, eval_body_idx)
        print(
            f"[bfm] EVAL-ONLY @ step {step}: "
            f"emd(state[:23])={m['emd']:.4f}  "
            f"emd_nonwrist23={m.get('emd_nonwrist23', float('nan')):.4f}  "
            f"emd_dof29={m.get('emd_dof29', float('nan')):.4f}  "
            f"mpjpe_l={m.get('mpjpe_l', float('nan')):.1f}  "
            f"(emd == UFO/BFM-Zero eval/emd == state[:, :23]; "
            f"n_motions={self.algo_cfg.get('eval_n_motions', 16)})"
        )
        return m
