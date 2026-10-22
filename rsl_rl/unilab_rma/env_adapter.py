"""UniLab manual-reset adapter: retain terminal obs before resetting histories."""

import torch


class UniLabRMAEnv:
    def __init__(self, env, cfg, device="cpu"):
        self.env, self.cfg, self.device = env, cfg, torch.device(device)
        if env._autoreset:
            raise ValueError("RMA adapter requires env.auto_reset=false")
        self.num_envs = env.num_envs

    def read(self, observations):
        if not hasattr(self.env, "_rma_label_valid") or not self.env._rma_label_valid.all():
            raise RuntimeError("RMA reset events must populate actual friction and payload labels")
        # ManagerBasedRlEnv exports policy as 'obs' and drops other manager
        # groups except the explicitly configured critic group.
        policy_key = "obs" if "obs" in observations else "policy"
        if policy_key not in observations or "critic" not in observations:
            raise KeyError(
                f"RMA requires public obs/critic tensors; got {list(observations)}. "
                "Set critic_observation_group=critic in the RMA overlay."
            )
        if "privileged" in observations:
            privileged = observations["privileged"]
        else:
            # These are the SAME reset samples written to the physics backend.
            # Clone immediately: partial resets will overwrite these label rows.
            privileged = self.env._rma_privileged
        result = tuple(
            torch.as_tensor(value, device=self.device, dtype=torch.float32).clone()
            for value in (observations[policy_key], observations["critic"], privileged)
        )
        for tensor, width in zip(
            result, (self.cfg.obs_dim, self.cfg.critic_dim, self.cfg.privileged_dim)
        ):
            if tensor.shape != (self.num_envs, width) or not torch.isfinite(tensor).all():
                raise ValueError(
                    f"RMA observation mismatch: {tensor.shape}, expected width {width}"
                )
        return result

    def reset(self):
        obs, _ = self.env.reset()
        return self.read(obs)

    def step(self, action):
        state = self.env.step(action.to(self.env.device))
        terminal = self.read(state.obs)
        reward = state.reward.to(self.device).clone()
        terminated = state.terminated.to(self.device).bool().clone()
        truncated = state.truncated.to(self.device).bool().clone()
        done = terminated | truncated
        if done.any():
            self.env.reset(done.nonzero(as_tuple=False).flatten().to(self.env.device))
        current = self.read(self.env.get_observations())
        return current, reward, terminated, truncated, terminal
