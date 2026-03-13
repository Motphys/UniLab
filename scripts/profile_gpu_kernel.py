"""Profile GPU kernel efficiency."""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import torch
from unilab.algos.torch.fast_sac.runner import FastSACRunner
from unilab.utils.algo_utils import ensure_registries

ensure_registries()

runner = FastSACRunner(
    env_name='G1WalkTaskMjSAC',
    num_envs=4096,
    warmup_steps=0,
)

# 运行几次迭代预热
print("Warming up...")
runner.learn(max_iterations=5, save_interval=0, logger_type='none')

# Profile 单次 update
print("\nProfiling single update...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
) as prof:
    # 手动执行一次 update
    batch = runner.learner._last_batch if hasattr(runner.learner, '_last_batch') else None
    if batch is None:
        # 从 buffer 采样
        batch = runner._shared_resources[0].sample(8192)

    runner.learner.update_critic(batch)
    runner.learner.update_actor(batch)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

runner.close()
