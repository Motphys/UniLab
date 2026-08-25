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

## Robot Binary Assets

Robot binary meshes and textures (for example `.STL`, `.obj`, and `.png`) are
externalized the same way, on the Hugging Face dataset repo
[unilabsim/unilab-robots](https://huggingface.co/datasets/unilabsim/unilab-robots).
X2 meshes download lazily on first use and land under their original path
`src/unilab/assets/robots/x2/meshes/`. T800 OBJ files and textures land under
`robots/t800/assets/` and `robots/t800/textures/`, respectively, so the original
relative XML paths remain valid. Pre-fetch them without running a task:

```bash
uv run unilab-pull-assets --robot x2
uv run unilab-pull-assets --robot t800
```

To add a new robot's binary assets:

1. Upload each directory to the HF repo while keeping the directory layout
   identical. A robot with multiple asset directories requires one upload per
   directory. For example, T800 uses:

   ```bash
   uv run hf upload unilabsim/unilab-robots \
     src/unilab/assets/robots/t800/assets robots/t800/assets \
     --repo-type dataset
   uv run hf upload unilabsim/unilab-robots \
     src/unilab/assets/robots/t800/textures robots/t800/textures \
     --repo-type dataset
   ```

2. Ignore the downloaded directory contents in `.gitignore` and keep a
   `.gitkeep` so each directory persists.
3. Resolve every referenced directory on a cold path before the backend parses
   the XML. The current API resolves one directory per call, so a T800 task uses:

   ```python
   resolve_robot_asset_dir("robots/t800/assets", marker="LINK_BASE.obj")
   resolve_robot_asset_dir("robots/t800/textures", marker="LINK_BASE.png")
   ```

## Architecture Notes

- Asset resolver module: `src/unilab/assets/hub.py`
  (`resolve_motion_files`).
- Single integration point: `MotionLoader.__init__` in
  `src/unilab/tasks/motion_tracking/common/motion_loader.py`, which calls the
  resolver once on a cold path.
- Hot paths (`step` / `reset`) never trigger any file download or parsing.
- `ASSETS_ROOT_PATH` is unchanged, so the download target matches the
  original local path exactly.
- Robot binary assets use the same directory resolver
  (`resolve_robot_asset_dir`). The
  thin `make_x2_wall_flip_env` factory in
  `src/unilab/tasks/motion_tracking/x2/__init__.py` resolves them once before
  delegating to the shared manager environment factory. The resolver is also
  exposed through the `unilab-pull-assets` CLI.
