# FlashSAC 真实训练计时

`benchmark_flashsac_training.py` 会调用生产入口 `train_flashsac.py`，创建真实物理后端环境，并实时转发训练输出。训练结束后，它从同一次训练的 TensorBoard 日志中打印：

- 每轮 learner update 时间；
- 每轮 collector cycle 时间（环境步进、replay 写入和 bookkeeping）；
- 每轮总时间和 reward；
- learner、collector、总时间的 mean / median / p90 / p95。

## 运行

MuJoCo：

```bash
uv run scripts/benchmark/rl/benchmark_flashsac_training.py \
  --backend mujoco --iterations 20 --num-envs 256 \
  --output scripts/benchmark/outputs/flashsac_training/summary.json
```

Motrix：

```bash
uv run scripts/benchmark/rl/benchmark_flashsac_training.py \
  --backend motrix --iterations 20 --num-envs 256
```

默认开启之前 FlashSAC 优化使用的 full-objective compile。要测 eager 路径，使用 `--no-compile`。如果要明确使用本地的优化版 `unilab_rl` checkout：

```bash
uv run scripts/benchmark/rl/benchmark_flashsac_training.py \
  --uni-rl-src ../unilab_rl-flashsac-optimization/src \
  --backend mujoco --iterations 20 --num-envs 256
```

compile 首轮可能包含编译开销；要只看 steady-state，可排除前几条共同 timing 样本：

```bash
... --skip-first 4
```

JSON 同时保留未过滤的 `rows_all` 和用于统计的 `rows`。

实际训练日志会保存在 `scripts/benchmark/outputs/flashsac_training/<timestamp>/`，可以直接用 TensorBoard 查看：

```bash
uv run tensorboard --logdir scripts/benchmark/outputs/flashsac_training
```

`--task` 可替换为其他 FlashSAC owner task，只要对应的 `<task>/<backend>.yaml` 存在。`--extra-override key=value` 可以追加 Hydra 参数，例如：

```bash
--extra-override training.torch_threads.learner_num_threads=8
```

注意：这不是模型或 collector 的仿真 microbenchmark，而是完整的 collector→replay→learner 训练链路；collector 时间来自真实物理后端的 `env.step()`。
