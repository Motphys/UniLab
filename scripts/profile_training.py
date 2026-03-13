"""Profile GPU utilization during training."""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from torch.profiler import profile, ProfilerActivity, schedule
from unilab.algos.torch.fast_sac.runner import FastSACRunner
from unilab.utils.algo_utils import ensure_registries

ensure_registries()

# Create runner
runner = FastSACRunner(
    env_name='Go1JoystickFlatTerrain',
    num_envs=4096,
    replay_buffer_n=1024,
    batch_size=8192,
    warmup_steps=0,
    updates_per_step=8,
)

# Profile 10 iterations
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=5, warmup=2, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    try:
        runner.learn(max_iterations=10, save_interval=0, logger_type='none')
    finally:
        runner.close()

print("Profile saved to ./logs/profile")
print("View with: tensorboard --logdir=./logs/profile")
