import ray
import torch
import time
import numpy as np
from collections import defaultdict

from unilab.algo.boprl.worker import RolloutWorker
from unilab.algo.boprl.learner import BOPRLLearner
from unilab.utils.rsl_rl_compat import convert_config_v3_to_v4, is_rsl_rl_v4
from rsl_rl.utils import resolve_callable


class BOPRLRunner:
    def __init__(
        self,
        env_name,
        env_cfg_overrides,
        rl_cfg,
        device="cuda:0",
        num_workers=4,
        steps_per_env=24,
        num_envs_per_worker=1,
    ):
        self.device = device
        self.env_name = env_name
        self.num_workers = num_workers
        self.steps_per_env = steps_per_env
        self.num_envs_per_worker = num_envs_per_worker

        # 1. Prepare Config
        # Ensure compatibility
        if is_rsl_rl_v4():
            self.rl_cfg = convert_config_v3_to_v4(rl_cfg)
        else:
            self.rl_cfg = rl_cfg  # Fallback for local dev if rsl_rl 3.x

        # 2. Init Ray
        if not ray.is_initialized():
            ray.init()

        # 3. Create Workers
        print(f"Spawning {num_workers} workers with {num_envs_per_worker} envs each...")

        worker_env_cfg = env_cfg_overrides.copy()
        worker_env_cfg["num_envs"] = num_envs_per_worker

        self.workers = [
            RolloutWorker.remote(env_name=env_name, env_cfg_overrides=worker_env_cfg, device="cpu")
            for _ in range(num_workers)
        ]

        # 4. Initialize Policy on Workers
        # Get policy config from rl_cfg
        policy_cfg = {
            "actor": self.rl_cfg["actor"],
            "obs_groups": self.rl_cfg.get("obs_groups", {"default": ["policy"]}),
        }

        # Init policies
        print(f"DEBUG RUNNER: Policy Config sent to workers: {policy_cfg['actor']}")
        futures = [w.init_policy.remote(policy_cfg) for w in self.workers]
        ray.get(futures)
        print("Workers initialized.")

        # 5. Create Learner (Local GPU)
        # Determine shapes from a worker or environment
        # We need Env info to create learner models
        # Let's ask first worker for env info
        # Or instantiate a dummy env locally? No, heavy.
        # Worker has num_envs and num_actions.
        # We need observation shape.
        # Let's add get_env_info to worker

        # For now assume known or hardcode from cfg?
        # Better: ask worker.
        # But worker.sample requires initialized policy.
        # init_policy creates dummy obs.

        # We reconstruct Actor/Critic here.
        obs_dim = 48  # Hardcoded for Go2 Flat? No, bad practice.
        # TODO: Get obs_dim from worker
        # Let's create a temporary dummy env or read from config if possible?
        # unilab.envs.registry has specs?
        # Hack: just create one env instance locally to get dims, then close it.
        from unilab.envs import registry

        temp_env = registry.make(env_name, **env_cfg_overrides)
        obs_dim = temp_env.observation_space.shape[0]
        num_actions = temp_env.action_space.shape[0]
        # num_local_envs = temp_env.num_envs # This defaults to 1 if not overridden!
        temp_env.close()

        self.num_envs_per_worker = num_envs_per_worker
        self.total_envs = num_workers * num_envs_per_worker

        print(f"Obs Dim: {obs_dim}, Actions: {num_actions}, Total Envs: {self.total_envs}")

        # Create Actor/Critic models on GPU
        obs_example = torch.zeros((self.total_envs, obs_dim), device=device)
        from tensordict import TensorDict

        td_example = TensorDict({"policy": obs_example}, batch_size=self.total_envs)

        actor_cfg = self.rl_cfg["actor"].copy()
        critic_cfg = self.rl_cfg["critic"].copy()

        actor_cls = resolve_callable(actor_cfg.pop("class_name"))
        critic_cls = resolve_callable(critic_cfg.pop("class_name"))

        actor = actor_cls(td_example, self.rl_cfg["obs_groups"], "actor", num_actions, **actor_cfg)
        critic = critic_cls(td_example, self.rl_cfg["obs_groups"], "critic", 1, **critic_cfg)

        algo_cfg = self.rl_cfg["algorithm"].copy()
        if "class_name" in algo_cfg:
            del algo_cfg["class_name"]

        self.learner = BOPRLLearner(actor, critic, device=device, **algo_cfg)

    def _collate_results(self, results):
        """Collate worker results into a single batch dict on learner device."""
        batch_dict = {}
        device = self.learner.device

        # Separate time-series data (dim=1 cat) vs single-step data (dim=0 cat)
        time_series_keys = ["observations", "actions", "rewards", "dones", "truncated", "actions_log_prob"]
        single_step_keys = ["last_obs"]  # [N, D] per worker, cat along dim=0

        for k in time_series_keys:
            if k in results[0]:
                tensors = [r[k] for r in results]
                # NOTE: non_blocking=False to avoid MPS async transfer issues
                batch_dict[k] = torch.cat(tensors, dim=1).to(device, non_blocking=False)

        for k in single_step_keys:
            if k in results[0]:
                tensors = [r[k] for r in results]
                batch_dict[k] = torch.cat(tensors, dim=0).to(device, non_blocking=False)

        return batch_dict

    def _aggregate_metrics(self, results):
        """Aggregate episode metrics from worker results."""
        for r in results:
            if "metrics" in r:
                m = r["metrics"]
                for key, val_list in m.items():
                    self.metrics_buffers[key].extend(val_list)

    def learn(self, max_iterations=1000, save_interval=50, log_dir=None):
        """Main training loop with async double-buffered pipeline and rsl_rl-style logging.

        Pipeline design (true overlap):
          Iter 0: sync weights → collect batch_0 (cold start, no overlap)
          Iter n>0:
            1. Start async collection (workers use weights from previous sync)
            2. While workers collect, process & train on previous batch (GPU)
            3. Wait for collection to finish
            4. Sync updated weights to workers (fast, for next iteration)

        This maximizes CPU/GPU overlap: GPU trains while CPU collects.
        """
        import statistics
        from collections import deque

        # Metrics buffers (same as rsl_rl Logger)
        self.metrics_buffers = defaultdict(lambda: deque(maxlen=100))

        # TensorBoard writer
        tb_writer = None
        if log_dir:
            import os

            os.makedirs(log_dir, exist_ok=True)
            try:
                from torch.utils.tensorboard import SummaryWriter

                tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            except ImportError:
                print("  [Warning] tensorboard not installed, skipping TB logging")

        tot_timesteps = 0
        tot_time = 0.0
        collection_size = self.steps_per_env * self.total_envs
        width = 80
        pad = 40

        # CRITICAL: Set learner to training mode so EmpiricalNormalization.update() works
        self.learner.train_mode()

        # === Prefetch Async Pipeline ===
        # Pre-collect first batch (no overlap possible)
        weights = self.learner.get_weights()
        weights_ref = ray.put(weights)
        ray.get([w.set_weights.remote(weights_ref) for w in self.workers])

        precollect_start = time.time()
        results = ray.get([w.sample.remote(self.steps_per_env) for w in self.workers])
        precollect_time = time.time() - precollect_start
        current_batch = self._collate_results(results)
        self._aggregate_metrics(results)
        print(f"  [Prefetch] Initial collection: {precollect_time:.2f}s")

        for it in range(max_iterations):
            iter_start = time.time()

            # 1. Sync latest weights to workers for NEXT batch
            sync_start = time.time()
            weights = self.learner.get_weights()
            weights_ref = ray.put(weights)
            ray.get([w.set_weights.remote(weights_ref) for w in self.workers])
            sync_time = time.time() - sync_start

            # 2. Start async collection for NEXT batch (workers use updated weights)
            collect_start = time.time()
            sample_futures = [w.sample.remote(self.steps_per_env) for w in self.workers]

            # 3. While workers collect on CPU, process & train on CURRENT batch on GPU
            learn_start = time.time()
            current_batch = self.learner.process_batch(current_batch)
            loss_dict = self.learner.update(current_batch)
            learn_time = time.time() - learn_start

            # 4. Wait for collection to finish → becomes current_batch for next iter
            results = ray.get(sample_futures)
            collect_time = time.time() - collect_start

            # 5. Collate new data
            current_batch = self._collate_results(results)
            self._aggregate_metrics(results)

            iteration_time = time.time() - iter_start
            tot_timesteps += collection_size
            tot_time += iteration_time

            # Checkpoint saving
            if log_dir and save_interval > 0 and (it % save_interval == 0 or it == max_iterations - 1):
                self._save_checkpoint(log_dir, it)

            # === Logging (rsl_rl style) ===
            fps = int(collection_size / iteration_time)

            # Gather metrics
            mean_reward = None
            mean_ep_len = None
            if "episode_returns" in self.metrics_buffers and len(self.metrics_buffers["episode_returns"]) > 0:
                mean_reward = statistics.mean(self.metrics_buffers["episode_returns"])
            if "episode_lengths" in self.metrics_buffers and len(self.metrics_buffers["episode_lengths"]) > 0:
                mean_ep_len = statistics.mean(self.metrics_buffers["episode_lengths"])

            # Console output
            log_string = f"""{"#" * width}\n"""
            log_string += f"""\033[1m{f" Learning iteration {it}/{max_iterations} ".center(width)}\033[0m \n\n"""

            log_string += (
                f"""{"Total steps:":>{pad}} {tot_timesteps} \n"""
                f"""{"Steps per second:":>{pad}} {fps:.0f} \n"""
                f"""{"Collection time:":>{pad}} {collect_time:.3f}s \n"""
                f"""{"Learning time:":>{pad}} {learn_time:.3f}s \n"""
                f"""{"Weight sync time:":>{pad}} {sync_time:.3f}s \n"""
                f"""{"Learning rate:":>{pad}} {self.learner.learning_rate:.6f}\n"""
            )

            for key, value in loss_dict.items():
                log_string += f"""{f"{key}:":>{pad}} {value:.4f}\n"""

            if mean_reward is not None:
                log_string += f"""{"Mean reward:":>{pad}} {mean_reward:.2f}\n"""
            if mean_ep_len is not None:
                log_string += f"""{"Mean episode length:":>{pad}} {mean_ep_len:.2f}\n"""

            for key in sorted(self.metrics_buffers.keys()):
                if key in ["episode_returns", "episode_lengths"]:
                    continue
                if len(self.metrics_buffers[key]) > 0:
                    pretty_key = "Mean " + key.replace("_", " ")
                    log_string += f"""{pretty_key+":":>{pad}} {statistics.mean(self.metrics_buffers[key]):.4f}\n"""

            if hasattr(self.learner.actor, "output_std") and self.learner.actor.distribution is not None:
                log_string += (
                    f"""{"Mean action noise std:":>{pad}} {self.learner.actor.output_std.mean().item():.2f}\n"""
                )

            done_it = it + 1
            remaining_it = max_iterations - done_it
            eta = tot_time / done_it * remaining_it if done_it > 0 else 0
            log_string += (
                f"""{"-" * width}\n"""
                f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
                f"""{"Time elapsed:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(tot_time))}\n"""
                f"""{"ETA:":>{pad}} {time.strftime("%H:%M:%S", time.gmtime(eta))}\n"""
            )
            print(log_string)

            # TensorBoard logging
            if tb_writer is not None:
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(f"Loss/{key}", value, it)
                if mean_reward is not None:
                    tb_writer.add_scalar("Train/mean_reward", mean_reward, it)
                if mean_ep_len is not None:
                    tb_writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
                tb_writer.add_scalar("Train/learning_rate", self.learner.learning_rate, it)
                tb_writer.add_scalar("Perf/total_fps", fps, it)
                tb_writer.add_scalar("Perf/collection_time", collect_time, it)
                tb_writer.add_scalar("Perf/learning_time", learn_time, it)
                # Reward components
                for key in sorted(self.metrics_buffers.keys()):
                    if key in ["episode_returns", "episode_lengths"]:
                        continue
                    if len(self.metrics_buffers[key]) > 0:
                        tb_writer.add_scalar(f"Train/{key}", statistics.mean(self.metrics_buffers[key]), it)

        # No pending batch to process — the prefetch pipeline processes each batch
        # exactly once in the loop body. The last collected batch (current_batch)
        # is for the next iteration that won't happen, so we skip it.

        if tb_writer is not None:
            tb_writer.close()

    def _save_checkpoint(self, log_dir, iteration):
        """Save model checkpoint."""
        import os

        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"model_{iteration}.pt")
        torch.save(
            {
                "actor_state_dict": self.learner.actor.state_dict(),
                "critic_state_dict": self.learner.critic.state_dict(),
                "optimizer_state_dict": self.learner.optimizer.state_dict(),
                "iteration": iteration,
            },
            path,
        )
        print(f"  [Checkpoint] Saved to {path}")

    def close(self):
        for w in self.workers:
            w.close.remote()
        ray.shutdown()
