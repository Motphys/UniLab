from dataclasses import asdict
from pathlib import Path

import torch

from .adaptation_module import AdaptationTrainer, HistoryBuffer
from .ppo import PPO, gae


def checkpoint(path, model, adaptation, iteration, optimizer=None, phase="teacher"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "unilab-rma-v1",
            "config": asdict(model.cfg),
            "teacher": model.state_dict(),
            "adaptation": adaptation.state_dict(),
            "iteration": iteration,
            "phase": phase,
            "optimizer": None if optimizer is None else optimizer.state_dict(),
        },
        path,
    )


def train_teacher(env, model, adaptation, iterations, path, horizon=24, logger=None):
    ppo = PPO(model)
    obs, critic, priv = env.reset()
    if logger is not None:
        from .training_log import EpisodeLog

        episodes = EpisodeLog(env.num_envs)
    for iteration in range(iterations):
        data = {
            k: []
            for k in (
                "obs",
                "critic",
                "priv",
                "raw",
                "logp",
                "values",
                "rewards",
                "next_values",
                "terminated",
                "truncated",
            )
        }
        for _ in range(horizon):
            with torch.no_grad():
                action, raw, logp = model.sample(obs, priv)
                value = model.value(critic, priv)
                following, reward, term, trunc, terminal = env.step(action)
                next_value = model.value(terminal[1], terminal[2])
            entries = dict(
                obs=obs,
                critic=critic,
                priv=priv,
                raw=raw,
                logp=logp,
                values=value,
                rewards=reward,
                next_values=next_value,
                terminated=term,
                truncated=trunc,
            )
            for key, tensor in entries.items():
                data[key].append(tensor.detach().clone())
            obs, critic, priv = following
        batch = {key: torch.stack(value) for key, value in data.items()}
        batch["advantages"], batch["returns"] = gae(
            batch["rewards"],
            batch["values"],
            batch["next_values"],
            batch["terminated"],
            batch["truncated"],
        )
        loss = ppo.update({k: v.flatten(0, 1) for k, v in batch.items()})
        print(
            f"teacher iteration={iteration} loss={loss:.5f} mean_step_reward={batch['rewards'].mean():.5f}",
            flush=True,
        )
        if logger is not None:
            logger.write(
                iteration + 1,
                (iteration + 1) * horizon * env.num_envs,
                loss=loss,
                mean_step_reward=batch["rewards"].mean().item(),
                **episodes.consume(batch["rewards"], batch["terminated"] | batch["truncated"]),
            )
        if (iteration + 1) % 20 == 0 or iteration + 1 == iterations:
            checkpoint(path, model, adaptation, iteration + 1, ppo.optimizer)


def train_student(env, teacher, adaptation, iterations, path, horizon=24, logger=None):
    trainer = AdaptationTrainer(teacher, adaptation)
    obs, _, priv = env.reset()
    history = HistoryBuffer(env.num_envs, teacher.cfg, obs.device)
    history.push(obs)
    episode_returns = torch.zeros(env.num_envs, device=obs.device)
    episode_lengths = torch.zeros(env.num_envs, device=obs.device)
    for iteration in range(iterations):
        completed_returns, completed_lengths, rewards = [], [], []
        for _ in range(horizon):
            trainer.add(history.data, priv)
            with torch.no_grad():
                # Students collect their OWN state distribution, not teacher rollouts.
                action = teacher.actor(obs, adaptation(history.data)).tanh()
                (obs, _, priv), reward, term, trunc, _ = env.step(action)
            done = term | trunc
            episode_returns += reward
            episode_lengths += 1
            rewards.append(reward.mean().item())
            completed_returns.extend(episode_returns[done].tolist())
            completed_lengths.extend(episode_lengths[done].tolist())
            episode_returns[done] = 0
            episode_lengths[done] = 0
            history.push(obs, done)
        loss = trainer.update()
        episodic = (
            f" episode_return={sum(completed_returns) / len(completed_returns):.5f}"
            f" episode_length={sum(completed_lengths) / len(completed_lengths):.1f}"
            if completed_returns
            else " completed_episodes=0"
        )
        print(
            f"student iteration={iteration} latent_mse={loss:.6f}"
            f" mean_step_reward={sum(rewards) / len(rewards):.5f}" + episodic,
            flush=True,
        )
        if logger is not None:
            logger.write(
                iteration + 1,
                (iteration + 1) * horizon * env.num_envs,
                latent_mse=loss,
                mean_step_reward=sum(rewards) / len(rewards),
                completed_episodes=len(completed_returns),
                mean_episode_return=(
                    sum(completed_returns) / len(completed_returns) if completed_returns else None
                ),
                mean_episode_length_steps=(
                    sum(completed_lengths) / len(completed_lengths) if completed_lengths else None
                ),
            )
        if (iteration + 1) % 20 == 0 or iteration + 1 == iterations:
            checkpoint(path, teacher, adaptation, iteration + 1, trainer.optimizer, phase="student")
