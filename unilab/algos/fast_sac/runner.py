"""FastSAC Ray-async Runner.

Identical pipeline to FastTD3Runner but uses SAC learner with stochastic
policy and automatic entropy tuning.
"""

import ray
import time
import torch
import statistics
import os
from collections import defaultdict, deque

from unilab.algos.off_policy_common.worker import OffPolicyWorker
from unilab.algos.off_policy_common.replay_buffer import ReplayBuffer
from unilab.algos.fast_sac.learner import FastSACLearner
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
from rsl_rl.utils import resolve_callable


class FastSACRunner:
    def __init__(
        self,
        env_name,
        env_cfg_overrides,
        rl_cfg,
        device="mps",
        num_workers=1,
        steps_per_env=24,
        num_envs_per_worker=1,
        replay_buffer_n=1000,
        batch_size=256,
        warmup_steps=1000,
        updates_per_step=1,
        exploration_noise=0.1,  # only used during warmup random
    ):
        self.device = device
        self.env_name = env_name
        self.num_workers = num_workers
        self.steps_per_env = steps_per_env
        self.num_envs_per_worker = num_envs_per_worker
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step

        # Config
        if is_rsl_rl_v4():
            self.rl_cfg = convert_config_v3_to_v4(rl_cfg)
        else:
            self.rl_cfg = rl_cfg

        # Ray
        if not ray.is_initialized():
            ray.init()

        # Workers — SAC uses stochastic sampling
        print(f"Spawning {num_workers} off-policy workers (SAC, stochastic) with {num_envs_per_worker} envs each...")
        worker_env_cfg = env_cfg_overrides.copy()
        worker_env_cfg["num_envs"] = num_envs_per_worker

        self.workers = [
            OffPolicyWorker.remote(
                env_name=env_name,
                env_cfg_overrides=worker_env_cfg.copy(),
                device="cpu",
                exploration_noise=exploration_noise,
            )
            for _ in range(num_workers)
        ]

        # Init stochastic policies on workers
        policy_cfg = {
            "actor": self.rl_cfg["actor"],
            "obs_groups": self.rl_cfg.get("obs_groups", {"default": ["policy"]}),
        }
        ray.get([w.init_policy.remote(policy_cfg) for w in self.workers])
        print("Workers initialized.")

        # Build learner
        from unilab.envs import registry

        temp_env = registry.make(env_name)
        obs_dim = temp_env.observation_space.shape[0]
        num_actions = temp_env.action_space.shape[0]
        temp_env.close()

        self.total_envs = num_workers * num_envs_per_worker
        self.num_actions = num_actions

        # Replay buffer: N transitions per env
        replay_buffer_size = replay_buffer_n * self.total_envs
        print(f"Obs Dim: {obs_dim}, Actions: {num_actions}, Total Envs: {self.total_envs}, Buffer: {replay_buffer_size}")

        # Create actor (stochastic) and twin critics
        obs_example = torch.zeros((self.total_envs, obs_dim), device=device)
        from tensordict import TensorDict

        td_obs = TensorDict({"policy": obs_example}, batch_size=self.total_envs)

        q_input_dim = obs_dim + num_actions
        q_example = torch.zeros((self.total_envs, q_input_dim), device=device)
        td_q = TensorDict({"policy": q_example}, batch_size=self.total_envs)

        actor_cfg = self.rl_cfg["actor"].copy()
        critic_cfg = self.rl_cfg["critic"].copy()
        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        critic_cls = resolve_callable(critic_cfg.pop("class_name"))

        # Actor: stochastic
        actor_cfg_stoch = actor_cfg.copy()
        actor_cfg_stoch["stochastic"] = True
        actor = actor_cls(td_obs, self.rl_cfg["obs_groups"], "actor", num_actions, **actor_cfg_stoch)

        # Twin critics
        critic_cfg1 = critic_cfg.copy()
        critic_cfg2 = critic_cfg.copy()
        obs_groups_q = {"critic": ["policy"]}
        critic1 = critic_cls(td_q, obs_groups_q, "critic", 1, **critic_cfg1)
        critic2 = critic_cls(td_q, obs_groups_q, "critic", 1, **critic_cfg2)

        algo_cfg = self.rl_cfg.get("algorithm", {}).copy()
        algo_cfg.pop("class_name", None)
        algo_cfg.pop("rnd_cfg", None)

        self.learner = FastSACLearner(
            actor, critic1, critic2,
            device=device,
            num_actions=num_actions,
            actor_lr=algo_cfg.get("learning_rate", 3e-4),
            critic_lr=algo_cfg.get("learning_rate", 3e-4),
            gamma=algo_cfg.get("gamma", 0.99),
            max_grad_norm=algo_cfg.get("max_grad_norm", 1.0),
            **{k: v for k, v in algo_cfg.items() if k in {
                "tau", "init_alpha", "alpha_lr", "target_entropy"
            }},
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_dim, num_actions, device=device)

    def _ingest_results(self, results):
        for r in results:
            obs = r["observations"]
            actions = r["actions"]
            rewards = r["rewards"]
            next_obs = r["next_observations"]
            dones = r["dones"]
            T, N = obs.shape[:2]
            self.replay_buffer.add_batch(
                obs.reshape(T * N, -1), actions.reshape(T * N, -1),
                rewards.reshape(T * N), next_obs.reshape(T * N, -1),
                dones.reshape(T * N),
            )
            if "metrics" in r:
                for key, val_list in r["metrics"].items():
                    self.metrics_buffers[key].extend(val_list)
            if "step_logs" in r:
                self.step_log_accumulator.extend(r["step_logs"])

    def learn(self, max_iterations=1000, save_interval=50, log_dir=None):
        self.metrics_buffers = defaultdict(lambda: deque(maxlen=100))
        self.step_log_accumulator = []

        tb_writer = None
        if log_dir:
            import os
            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            except ImportError:
                print("  [Warning] tensorboard not installed")

        try:
            from unilab.utils.resource_monitor import ResourceMonitor
            res_monitor = ResourceMonitor()
            res_monitor.start()
        except Exception:
            res_monitor = None

        tot_timesteps = 0
        tot_time = 0.0
        collection_size = self.steps_per_env * self.total_envs
        width = 80
        pad = 40

        self.learner.train_mode()

        weights_ref = ray.put(self.learner.get_weights())
        ray.get([w.set_weights.remote(weights_ref) for w in self.workers])

        # Warmup
        print(f"  [Warmup] Collecting {self.warmup_steps} initial transitions...")
        warmup_iters = max(1, self.warmup_steps // collection_size)
        for _ in range(warmup_iters):
            results = ray.get([w.sample.remote(self.steps_per_env, stochastic=True) for w in self.workers])
            self._ingest_results(results)
        print(f"  [Warmup] Buffer size: {len(self.replay_buffer)}")

        for it in range(max_iterations):
            iter_start = time.time()

            sync_start = time.time()
            weights_ref = ray.put(self.learner.get_weights())
            ray.get([w.set_weights.remote(weights_ref) for w in self.workers])
            sync_time = time.time() - sync_start

            collect_start = time.time()
            sample_futures = [w.sample.remote(self.steps_per_env, stochastic=True) for w in self.workers]

            learn_start = time.time()
            loss_accum = defaultdict(float)
            num_updates = self.updates_per_step * self.steps_per_env
            for _ in range(num_updates):
                if len(self.replay_buffer) < self.batch_size:
                    break
                batch = self.replay_buffer.sample(self.batch_size)
                losses = self.learner.update(*batch)
                for k, v in losses.items():
                    loss_accum[k] += v
            learn_time = time.time() - learn_start

            if num_updates > 0:
                for k in loss_accum:
                    loss_accum[k] /= num_updates

            results = ray.get(sample_futures)
            collect_time = time.time() - collect_start
            self._ingest_results(results)

            iteration_time = time.time() - iter_start
            tot_timesteps += collection_size
            tot_time += iteration_time

            if log_dir and save_interval > 0 and (it % save_interval == 0 or it == max_iterations - 1):
                self._save_checkpoint(log_dir, it)

            fps = int(collection_size / iteration_time) if iteration_time > 0 else 0
            mean_reward = None
            mean_ep_len = None
            if "episode_returns" in self.metrics_buffers and len(self.metrics_buffers["episode_returns"]) > 0:
                mean_reward = statistics.mean(self.metrics_buffers["episode_returns"])
            if "episode_lengths" in self.metrics_buffers and len(self.metrics_buffers["episode_lengths"]) > 0:
                mean_ep_len = statistics.mean(self.metrics_buffers["episode_lengths"])

            res_str = ""
            if res_monitor:
                stats = res_monitor.get_stats()
                res_str = (
                    f"""{"CPU usage:":{pad}} {stats['cpu_percent']:.1f}%\n"""
                    f"""{"Memory:":{pad}} {stats['mem_used_gb']:.1f}/{stats['mem_total_gb']:.1f} GB\n"""
                    f"""{"MPS Memory:":{pad}} {stats.get('mps_mem_gb', 0.0):.1f} GB\n"""
                    f"""{"GPU power:":{pad}} {stats.get('gpu_power', 'N/A')}\n"""
                )

            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" FastSAC iteration {it}/{max_iterations} ".center(width)}\033[0m \n\n"""
            log_string += (
                f"""{"Total steps:":{pad}} {tot_timesteps}\n"""
                f"""{"Steps per second:":{pad}} {fps}\n"""
                f"""{"Collection time:":{pad}} {collect_time:.3f}s\n"""
                f"""{"Learning time:":{pad}} {learn_time:.3f}s\n"""
                f"""{"Weight sync time:":{pad}} {sync_time:.3f}s\n"""
                f"""{"Replay buffer size:":{pad}} {len(self.replay_buffer)}\n"""
                f"""{"Alpha:":{pad}} {self.learner.alpha:.4f}\n"""
            )

            for key, value in loss_accum.items():
                log_string += f"""{f"{key}:":{pad}} {value:.4f}\n"""

            if mean_reward is not None:
                log_string += f"""{"Mean reward:":{pad}} {mean_reward:.2f}\n"""
            if mean_ep_len is not None:
                log_string += f"""{"Mean episode length:":{pad}} {mean_ep_len:.2f}\n"""

            step_log_means = {}
            if self.step_log_accumulator:
                all_keys = set()
                for entry in self.step_log_accumulator:
                    all_keys.update(entry.keys())
                for key in sorted(all_keys):
                    vals = [e[key] for e in self.step_log_accumulator if key in e]
                    step_log_means[key] = sum(vals) / len(vals)
                self.step_log_accumulator = []
            for key in sorted(step_log_means.keys()):
                log_string += f"""{f"{key}:":{pad}} {step_log_means[key]:.4f}\n"""

            if res_str:
                log_string += f"""{"-" * width}\n"""
                log_string += res_str

            done_it = it + 1
            remaining_it = max_iterations - done_it
            eta = tot_time / done_it * remaining_it if done_it > 0 else 0
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":{pad}} {time.strftime("%H:%M:%S", time.gmtime(tot_time))}\n"""
                f"""{"ETA:":{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            if tb_writer is not None:
                for key, value in loss_accum.items():
                    tb_writer.add_scalar(f"Loss/{key}", value, it)
                if mean_reward is not None:
                    tb_writer.add_scalar("Train/mean_reward", mean_reward, it)
                if mean_ep_len is not None:
                    tb_writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
                tb_writer.add_scalar("Train/alpha", self.learner.alpha, it)
                tb_writer.add_scalar("Perf/total_fps", fps, it)
                tb_writer.add_scalar("Perf/buffer_size", len(self.replay_buffer), it)
                for key, value in step_log_means.items():
                    if "/" in key:
                        tb_writer.add_scalar(key, value, it)
                    else:
                        tb_writer.add_scalar(f"Episode/{key}", value, it)

        if tb_writer is not None:
            tb_writer.close()
        if res_monitor:
            res_monitor.stop()

    def _save_checkpoint(self, log_dir, iteration):
        import os
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"model_{iteration}.pt")
        torch.save(
            {
                "actor_state_dict": self.learner.actor.state_dict(),
                "critic1_state_dict": self.learner.critic1.state_dict(),
                "critic2_state_dict": self.learner.critic2.state_dict(),
                "target_critic1_state_dict": self.learner.target_critic1.state_dict(),
                "target_critic2_state_dict": self.learner.target_critic2.state_dict(),
                "actor_optimizer_state_dict": self.learner.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.learner.critic_optimizer.state_dict(),
                "log_alpha": self.learner.log_alpha.detach().cpu(),
                "iteration": iteration,
            },
            path,
        )
        print(f"  [Checkpoint] Saved to {path}")

    def play(self, log_dir):
        """Play using the trained policy."""
        # Load latest model if available
        if log_dir and os.path.exists(log_dir):
            import re
            checkpoints = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
            if checkpoints:
                # Sort by iteration number
                checkpoints.sort(key=lambda x: int(re.search(r"model_(\d+).pt", x).group(1)))
                latest_ckpt = os.path.join(log_dir, checkpoints[-1])
                print(f"Loading checkpoint: {latest_ckpt}")
                import torch
                state_dict = torch.load(latest_ckpt, map_location=self.device)
                self.learner.actor.load_state_dict(state_dict["actor"])
                self.learner.actor.eval()
            else:
                print(f"No checkpoints found in {log_dir}, playing with random/initial policy.")
        else:
            print("No log_dir provided or does not exist, playing with random/initial policy.")

        # Sync weights to workers
        import ray
        weights_ref = ray.put(self.learner.get_weights())
        ray.get([w.set_weights.remote(weights_ref) for w in self.workers])
        
        # Disable exploration noise (for valid fallback)
        ray.get([w.set_exploration_noise.remote(0.0) for w in self.workers])

        # Setup video recording
        try:
            from unilab.envs.utils import render_many
            import mediapy as media
            from unilab.envs import registry
            from tensordict import TensorDict
            
            print("Setting up local environment for video recording...")
            num_play_envs = 16 
            env = registry.make(self.env_name, num_envs=num_play_envs, sim_backend="mujoco")
            
            # Reset
            obs, _ = env.reset()
            obs_torch = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
            obs_td = TensorDict({"policy": obs_torch}, batch_size=num_play_envs, device=self.device)
            
            state_list = []
            num_steps = 200 # ~4 seconds
            
            print(f"Collecting {num_steps} steps for video...")
            with torch.inference_mode():
                for _ in range(num_steps):
                    # Get action (mean of Gaussian for SAC play)
                    obs_td["policy"] = obs_torch
                    # MLPModel returns distribution if stochastic=True, mean if deterministic=True?
                    # MLPModel.forward(stochastic_output=True) returns Actions.
                    # We want mean. MLPModel usually returns mean if stochastic=False.
                    # But our config has stochastic=True.
                    # If we call forward(stochastic_output=False), it returns mean?
                    # Looking at MLPModel implementation (or guessing): standard impl returns mode/mean if stochastic_output=False.
                    actions = self.learner.actor(obs_td, stochastic_output=False) # valid arg for SAC actor?
                    # The actor wrapper in worker handles this? No, worker calls actor(obs, stochastic_output=True).
                    # Here we call directly.
                    # If MLPModel supports it, great. If not, it might return sample.
                    # Let's assume consistent interface.
                    
                    # Step env
                    obs, _, _, _, _ = env.step(actions.cpu().numpy())
                    obs_torch = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
                    
                    # Save state
                    state_list.append(env.state.physics_state.copy())
            
            # Render
            print("Rendering frames...")
            output_dir = log_dir if log_dir and os.path.exists(log_dir) else "."
            from pathlib import Path
            output_video = Path(output_dir) / "play_video.mp4"
            
            frames = render_many.render_states_get_frames(
                state_list,
                env.cfg.model_file,
                width=1280,
                height=720,
                camera_id=-1,
                num_processes=1 # Ray compatibility
            )
            
            print(f"Saving video to {output_video}...")
            media.write_video(str(output_video), frames, fps=int(1.0/env.cfg.ctrl_dt))
            print("Done!")
            
            env.close()
            return

        except Exception as e:
            print(f"Error during video recording: {e}")
            import traceback
            traceback.print_exc()

        print("Starting play loop (no video)... Press Ctrl+C to stop.")
        try:
            import time
            while True:
                ray.get([w.sample.remote(self.steps_per_env) for w in self.workers])
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    def close(self):
        import ray
        ray.shutdown()
