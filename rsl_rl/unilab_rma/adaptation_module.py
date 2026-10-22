import torch
from torch import nn

from .actor_critic_rma import mlp


class AdaptationModule(nn.Module):
    """Time-major history: oldest first, current observation last; zero padding."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.net = mlp([cfg.history_length * cfg.obs_dim, 256, 128, cfg.latent_dim], nn.Tanh())

    def forward(self, history):
        return self.net(history.flatten(1))


class HistoryBuffer:
    def __init__(self, n, cfg, device="cpu"):
        self.data = torch.zeros(n, cfg.history_length, cfg.obs_dim, device=device)

    def push(self, obs, done=None):
        if done is not None:
            self.data[done.bool()] = 0
        self.data[:, :-1] = self.data[:, 1:].clone()
        self.data[:, -1] = obs
        return self.data


class AdaptationTrainer:
    """DAgger-style latent labels on STUDENT-visited states, aggregated FIFO data.

    Teacher and actor are fixed in phase 2; only adaptation parameters update.
    """

    def __init__(self, teacher, module, capacity=8192, lr=1e-4, seed=0):
        self.teacher = teacher.eval().requires_grad_(False)
        self.module = module
        self.optimizer = torch.optim.Adam(module.parameters(), lr=lr)
        cfg = module.cfg
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.histories = torch.empty(capacity, cfg.history_length, cfg.obs_dim)
        self.labels = torch.empty(capacity, cfg.latent_dim)
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def add(self, history, privileged):
        labels = self.teacher.encoder(privileged)
        h, z = history.detach().cpu(), labels.detach().cpu()
        if len(h) > self.capacity:
            h, z = h[-self.capacity :], z[-self.capacity :]
        for start in range(0, len(h), self.capacity):
            count = min(len(h) - start, self.capacity)
            indices = (torch.arange(count) + self.cursor) % self.capacity
            self.histories[indices] = h[start : start + count]
            self.labels[indices] = z[start : start + count]
            self.cursor = (self.cursor + count) % self.capacity
            self.size = min(self.size + count, self.capacity)

    def update(self, batch_size=256, updates=4):
        if not self.size:
            raise RuntimeError("Collect student rollouts first")
        device = next(self.module.parameters()).device
        losses = []
        for _ in range(updates):
            indices = torch.randint(
                self.size, (min(batch_size, self.size),), generator=self.generator
            )
            prediction = self.module(self.histories[indices].to(device))
            loss = nn.functional.mse_loss(prediction, self.labels[indices].to(device))
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)
