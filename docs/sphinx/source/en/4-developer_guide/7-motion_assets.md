# Motion Asset Migration (Hugging Face)

## Background

Motion assets (`.npz` / `.csv`) are no longer stored in the Git repository.
They are hosted on the Hugging Face dataset repo
[unilabsim/unilab-motions](https://huggingface.co/datasets/unilabsim/unilab-motions)
to keep the repo small and to improve clone and CI experience.

The local directory `src/unilab/assets/motions/g1/` is preserved as the
download target, so existing path references stay valid.

## First Use

1. Install dependencies (`huggingface_hub` is part of the core dependencies):

   ```bash
   uv sync
   ```

2. Run any training or evaluation command. Motion files are downloaded
   lazily when `MotionLoader` is initialized:

   ```bash
   uv run train --algo ppo --task g1_motion_tracking --sim mujoco
   ```

   On first download the log shows:

   ```
   INFO:unilab.assets.hub:Downloading motions/g1/dance1_subject2_part.npz from HF repo unilabsim/unilab-motions ...
   INFO:unilab.assets.hub:Downloaded to /path/to/src/unilab/assets/motions/g1/dance1_subject2_part.npz
   ```

3. Once downloaded, files are cached locally and later runs do not trigger
   another download.

## Offline Use

Set the environment variable to forbid network requests:

```bash
export HF_HUB_OFFLINE=1
```

The resolver then only looks up local files and raises if a file is missing.

To pre-download every asset in an environment that does have network access:

```bash
huggingface-cli download unilabsim/unilab-motions \
  --repo-type dataset \
  --local-dir src/unilab/assets
```

After this completes the assets are available for offline use.

## CI Caching

In CI, point `HF_HOME` at a persistent cache directory to avoid repeated
downloads:

```yaml
env:
  HF_HOME: /cache/huggingface
```

Alternatively, pre-download into the in-repo directory with `--local-dir`
(already excluded by `.gitignore`).

## Adding New Motion Files

1. Generate the `.npz` with the existing pipeline (see
   `scripts/motion/README.md`).
2. Upload to the HF repo, keeping the directory layout identical:

   ```bash
   huggingface-cli upload unilabsim/unilab-motions \
     src/unilab/assets/motions motions \
     --repo-type dataset
   ```

3. Reference the new file path in the env config.

## Robot Mesh Assets

Robot binary meshes (`.STL`) are externalized the same way, on the Hugging Face
dataset repo
[unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots).
X2 meshes download lazily on first use and land under their original path
`src/unilab/assets/robots/x2/meshes/`, so the XML `meshdir` references resolve
unchanged. Pre-fetch them without running a task:

```bash
uv run unilab-pull-assets --robot x2
```

To add a new robot's meshes:

1. Upload to the HF repo, keeping the directory layout identical:

   ```bash
   huggingface-cli upload unilabsim/unilab-robots \
     src/unilab/assets/robots/<robot>/meshes robots/<robot>/meshes \
     --repo-type dataset
   ```

2. Ignore the local `*.STL` in `.gitignore` (keep a `.gitkeep` so the directory
   persists).
3. Resolve the directory once on a cold path from the env, e.g.
   `resolve_robot_asset_dir("robots/<robot>/meshes", marker="<some>.STL")`.

## Architecture Notes

- Asset resolver module: `src/unilab/assets/hub.py`
  (`resolve_motion_files`).
- Single integration point: `MotionLoader.__init__` in
  `src/unilab/envs/motion_tracking/g1/motion_loader.py`, which calls the
  resolver once on a cold path.
- Hot paths (`step` / `reset`) never trigger any file download or parsing.
- `ASSETS_ROOT_PATH` is unchanged, so the download target matches the
  original local path exactly.
- Robot meshes use the same directory resolver (`resolve_robot_asset_dir`),
  integrated at `X2WallFlipTrackingEnv.__init__` in
  `src/unilab/envs/motion_tracking/x2/flip_tracking.py`, and exposed as the
  `unilab-pull-assets` CLI.
