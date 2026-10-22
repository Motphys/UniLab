"""Small explicit PPO implementation, independent of external RSL-RL internals."""

import torch


def gae(rewards, values, next_values, terminated, truncated, gamma=0.99, lam=0.95):
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        # Timeout bootstraps from PRE-reset state, but never crosses episode boundary.
        delta = rewards[t] + gamma * (~terminated[t]).float() * next_values[t] - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])).float() * carry
        advantage[t] = carry
    return advantage, advantage + values


class PPO:
    def __init__(self, model, lr=3e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, batch, epochs=5, minibatch=256, clip=0.2):
        adv = batch["advantages"]
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        losses = []
        for _ in range(epochs):
            for ids in torch.randperm(len(adv), device=adv.device).split(minibatch):
                dist = self.model.distribution(batch["obs"][ids], batch["priv"][ids])
                logp = self.model.log_prob(dist, batch["raw"][ids])
                ratio = (logp - batch["logp"][ids]).exp()
                surrogate = torch.minimum(
                    ratio * adv[ids], ratio.clamp(1 - clip, 1 + clip) * adv[ids]
                )
                value = self.model.value(batch["critic"][ids], batch["priv"][ids])
                value_loss = (value - batch["returns"][ids]).square().mean()
                # Sample estimate of squashed-policy entropy; base Gaussian entropy is not used.
                raw_entropy = dist.rsample()
                entropy = -self.model.log_prob(dist, raw_entropy).mean()
                loss = -surrogate.mean() + 0.5 * value_loss - 0.001 * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(loss.item())
        return sum(losses) / len(losses)
