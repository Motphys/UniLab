# FAQ

Quick answers to "it does not match", "it cannot be found", and "the behavior
changed" questions, grouped by topic. Each entry gives the direct answer and
one link to the page that owns the details.

For training-process problems that need a diagnosis chain (NaN localization,
reward decomposition, backend mismatch debugging, trace analysis), start from
{doc}`../6-how_to/3-training_diagnostics` instead — this page points there
rather than repeating it.

## Installation and platforms

### `sim=<backend> requires the <backend> extra` at launch

```
SystemExit: sim=mujoco requires the MuJoCo extra. Install it with
`pip install unilab[mujoco]` (or `uv sync --extra mujoco` in a source checkout).
```

The CLI checks each backend's Python dependencies before routing
(`src/unilab/cli.py`). Sync the matching extra — for example
`uv sync --extra mujoco --extra mjwarp` — and retry. `isaacsim` and `isaacgym`
additionally need their isolated external worker environments, installed with
`scripts/tools/setup_isaacsim_env.sh` / `scripts/tools/setup_isaacgym_env.sh`.
Backend extras and prerequisites: {doc}`2-installation`.

### `uv sync --extra mujoco` fails while building `mjbatch`

`fatal error: Python.h: No such file or directory` or
`c++: No such file or directory` mean the pinned `mjbatch` engine is being
built from source without a C++17 toolchain or Python development headers.
A uv-managed Python (`uv python install`) bundles the headers; system Pythons
need `python3-dev` (Debian/Ubuntu) or `python3-devel` (Fedora/RHEL). The full
error-to-fix table, including the `mjbatch was built against MuJoCo <version>`
watchdog, is in the "Install Error Signatures" section of
{doc}`2-installation`.

### Which platforms and Python versions are supported?

Python `>=3.10,<3.14` (`pyproject.toml`), with resolved dependency profiles
for Linux x86_64, Linux aarch64, and Apple Silicon macOS; `3.13` is the
recommended interpreter. On Windows the `mjbatch` engine ships no wheels, so
the MuJoCo physics backend stays Linux/macOS-only there. ROCm and Intel XPU
use the dedicated `make sync-rocm` / `make sync-xpu` profiles. Details:
{doc}`2-installation`.

### Asset or checkpoint downloads fail / Hugging Face is unreachable

Robot meshes, motion clips, scenes, and demo checkpoints download lazily from
Hugging Face dataset repos on first use. When `huggingface.co` is unreachable,
set `HF_ENDPOINT=https://hf-mirror.com` (the downloader also auto-retries the
official endpoint when a mirror fails — `src/unilab/assets/hub.py`). To
pre-fetch robot assets for CI or offline work, run
`uv run unilab-pull-assets` (`--robot all` pulls every registered robot).
Asset layout: {doc}`2-installation`; demo cache: {doc}`1-quick_demo`.

### httpx SOCKS proxy missing socksio

```
ImportError: Using SOCKS proxy, but the 'socksio' package is not installed.
```

`huggingface_hub` uses `httpx`, which requires `socksio` once it detects
`ALL_PROXY=socks5://...`. Install it into the project `.venv/` (not conda):

```bash
uv pip install httpx[socks] --python .venv/bin/python
```

Or switch to an HTTP proxy to avoid SOCKS entirely:

```bash
unset all_proxy ALL_PROXY
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080
```

Replace `proxy.example.com:8080` with your own proxy endpoint.

### conda install connection fails

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
```

Some conda setups do not use shell proxy variables. Configure conda's own proxy
settings when package resolution fails behind a proxy:

```bash
conda config --set proxy_servers.http http://proxy.example.com:8080
conda config --set proxy_servers.https http://proxy.example.com:8080
```

### Native H2D extension unavailable

```
Native H2D extension unavailable: CUDA_HOME not set and nvcc not found.
```

The off-policy CUDA replay pipeline may JIT-compile the optional
`unilab_native_h2d` extension. UniLab falls back to the pure-PyTorch async copy
path when the native extension is unavailable, so this message is usually
diagnostic rather than fatal. If you want the native extension path, check these
toolchain prerequisites:

**C++ compiler**:

```bash
sudo apt-get install build-essential
```

**CUDA Toolkit**:

```bash
conda install -c nvidia cuda-toolkit=12.8 -y
```

**CUDA path**:

```bash
export CUDA_HOME=$CONDA_PREFIX
```

UniLab checks both `$CUDA_HOME/include` and
`$CUDA_HOME/targets/x86_64-linux/include` for `cuda_runtime_api.h`.

## Configuration and CLI

### Hydra rejects an override: `Key '...' is not in struct`

```
Key 'not_a_key' is not in struct
    full_key: algo.not_a_key
```

The override key does not exist in the composed config — almost always a typo,
a stale key from an older release, or a field that lives under a different
group. Print the fully composed config and compare the key path:

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco --cfg job
```

Override mechanics and owner-YAML layout: {doc}`../2-user_guide/1-training/2-hydra_config`;
field-by-field tables: {doc}`../2-user_guide/2-algorithms/1-ppo` (and the
sibling APPO/SAC/FlashSAC pages).

### `No owner config exists for algo=..., task=..., sim=...`

That task/algorithm/backend combination has no owner YAML
(`src/unilab/conf/<algo>/task/<task>/<backend>.yaml`), so training refuses to
start. Pick a supported combination from the support matrix. `eval` is more
lenient: it falls back to a sibling backend owner of the same task and
re-applies the requested backend through the sim2sim contract check
(`src/unilab/cli.py`). Supported combinations: {doc}`../5-reference/5-support_matrix`;
the eval fallback and its guard: {doc}`../3-deployment/2-sim_to_sim/7-config_guard`.

### Setting `training.sim_backend=<backend>` did not switch the backend

It is not a backend switch: `training.sim_backend` is an identity field set by
the selected owner YAML, and the CLI treats it as a reserved override. Select
the backend with `--sim` (plus the matching `--algo`/`--task` owner), e.g.
`uv run train --algo ppo --task go2_joystick_flat --sim mjwarp`. Background:
{doc}`../2-user_guide/1-training/2-hydra_config`.

### `--load-run` rejects a checkpoint path

`--load-run` accepts only `-1` (latest run) or a run directory name under the
task's log root, not an arbitrary path (`src/unilab/cli.py`). Run directories
are named `YYYY-MM-DD_HH-MM-SS_<sim_backend>` under
`logs/<algo.algo_log_name>/<task>/`. Log layout and resume semantics:
{doc}`../2-user_guide/1-training/3-logging`.

## Training and debugging

### Training hit NaN — what do I look at first?

The NaN guard writes a state dump (`nan_dump_<timestamp>_step<N>.npz` under
`/tmp/unilab/nan_dumps` by default) naming the detection step and the NaN env
rows; replay it with `uv run unilab-viz-nan <dump> --env-index 0`, and read the
`reward/<term>` curves just before the detection step to find the diverging
term. The full diagnosis chain: {doc}`../6-how_to/3-training_diagnostics`.

### The policy does not learn / reward curves look wrong

Check the overall progress keys (`Train/mean_reward`,
`Train/mean_episode_length`) against the per-term `reward/<term>` curves, then
the hyperparameters against the algorithm reference tables. The symptom-to-check
map (reward explosion, flat learning, per-term attribution) lives in
{doc}`../6-how_to/3-training_diagnostics`; metric key semantics are in
{doc}`../2-user_guide/1-training/3-logging`.

### A checkpoint from backend A refuses to play on backend B

That is the sim2sim config guard working as designed: training snapshots the
policy-defining fields into `run_config.json`, and play validates the target
owner config against that snapshot before env construction. DENYLIST mismatches
(observations, actions, network shapes, `ctrl_dt`, ...) raise
`CrossBackendIncompatibleError` under the default `training.sim2sim_strict=true`;
`training.sim2sim_strict=false` downgrades them to warnings at your own risk.
Field tiers and resolution options: {doc}`../3-deployment/2-sim_to_sim/7-config_guard`;
debugging path: {doc}`../6-how_to/3-training_diagnostics`.

### How do I train on multiple GPUs?

`CUDA_VISIBLE_DEVICES` is the sole GPU topology source: omit it for one rank,
or list one entry per GPU to get one rank per entry (physical ordinals, UUIDs,
or MIG tokens all work). Each rank is a full learner+collector pair pinned to
its local `cuda:0`, and `algo.num_envs` is a **per-rank** count, not a global
budget. Topology, seeding, and aggregation semantics:
{doc}`../2-user_guide/2-algorithms/1-ppo`.

### Multi-GPU gradient sync looks slow on an NVLink host

UniLab's data-parallel launch sets `NCCL_P2P_DISABLE=1` and
`NCCL_SHM_DISABLE=1` by default (`os.environ.setdefault`, so your environment
wins) because peer transport hangs on some hosts. On a healthy NVLink machine,
export `NCCL_P2P_DISABLE=0` and `NCCL_SHM_DISABLE=0` before launching to
restore peer/SHM bandwidth
(`src/unilab/rl/runtime/distributed/sync.py`).

## Backends and compatibility

### Which task/algorithm/backend combinations are supported?

The support matrix is generated from registry entries, owner YAMLs, and
validation inventories — treat it as the support claim, not the `--sim` choice
list alone. {doc}`../5-reference/5-support_matrix`.

### IsaacSim record/interactive playback crashes in `createHydraEngine`

Known runtime blocker: on the validated IsaacSim 5.1.0 / Kit 107.3.3 hosts,
both render modes crash during `AppLauncher` initialization
([issue #2579](https://github.com/Motphys/UniLab/issues/2579)). Headless SAC
`g1_walk_flat` training is the supported cell; evaluate with
`--render-mode none`. Current evidence and the contract matrix:
{doc}`../2-user_guide/3-backends/4-isaacsim`.

### Two backends give different numbers for the same task

Different simulators are not bit-identical; UniLab's contract is that the
policy I/O fields match (enforced by the sim2sim config guard) while solver-level
dynamics can differ. Known per-backend capability differences:
{doc}`../3-deployment/2-sim_to_sim/6-capability_gaps`; cross-backend parity
methodology: {doc}`../3-deployment/2-sim_to_sim/4-reward_parity`.

## Version migration

### `NpEnv` / `np_env.py` no longer exist

Removed in 1.3.4: the Manager-Based runtime is tensor-only and `TorchEnv` is
the sole environment lifecycle base ({doc}`/adr/ADR-0011-torch-only-manager-based-runtime`).
Port NumPy-based code to the Torch contract. Release notes: {doc}`/changelog`.

### HIM-PPO / HORA / the Go2 arm and Sharpa tasks are gone

Moved out of UniLab in 1.2.0: Go2 arm manipulation and HIM-PPO live in
[legged-manipulation_unilab](https://github.com/unilabsim/legged-manipulation_unilab),
Sharpa in-hand and HORA in `sharpa_rl_unilab`; their configs, assets, and
scripts were removed from this repository. Release notes: {doc}`/changelog`.

### `import uni_rl` fails / the `unirl` extra is gone

The unilab-rl package is merged back into this repository as `unilab.rl`
({doc}`/adr/ADR-0016-rl-reintegration-into-unilab-rl`); the RL stack
ships in the base install and the legacy `uni_rl` extra was removed — do not
pass it to `uv sync` or `pip install`. Import mapping:
{doc}`../3-deployment/3-framework_migration/7-from_unilab_rl`.

### v1.4 reset code raises `TypeError` on `reset_mask`

1.5 restored the `env_ids` row-index reset contract and removed the v1.4
fixed-width bool mask form without a compatibility layer. Port guide and
signature table: {doc}`6-migration_1_4_to_1_5`.

### `Episode_Reward/*` is missing from new TensorBoard runs

Intentionally removed: it duplicated `reward/<term>` with a different meaning.
Use `reward/<term>` for per-term diagnostics and `Train/mean_reward` for
overall progress; historical event files still display the old key. Current
metric contract: {doc}`../2-user_guide/1-training/3-logging`.

## Quick reference

| Symptom | Fix |
|---------|-----|
| `sim=<backend> requires the <backend> extra` | `uv sync --extra <backend>`; Isaac owners need their external worker setup scripts |
| `Python.h: No such file or directory` building mjbatch | uv-managed Python, or `python3-dev` / `python3-devel` |
| Hugging Face unreachable | `export HF_ENDPOINT=https://hf-mirror.com`; pre-fetch with `uv run unilab-pull-assets` |
| socksio not installed | `uv pip install httpx[socks] --python .venv/bin/python` |
| conda HTTP 000 | configure `conda config --set proxy_servers.*` for your proxy |
| native H2D extension unavailable | install C++ / CUDA Toolkit prerequisites or use the fallback |
| `Key '...' is not in struct` | typo'd or removed override; inspect with `--cfg job` |
| `No owner config exists for ...` | pick a supported combination from the support matrix |
| backend did not change | use `--sim`, not `training.sim_backend=` |
| NaN during training | read the guard dump with `uv run unilab-viz-nan`, then the diagnostics page |
| `CrossBackendIncompatibleError` on play | align DENYLIST fields, or `training.sim2sim_strict=false` at your own risk |
| IsaacSim playback crashes | known blocker; use `--render-mode none` |
| `NpEnv` / HIM-PPO / HORA / `uni_rl` missing | removed or moved; see the migration entries above and {doc}`/changelog` |
