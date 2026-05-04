# Motion Preprocessing

This directory contains scripts for preprocessing motion data for motion tracking tasks.

## BONES-SEED CSV Replay

The `replay_bones_seed_csv.py` script replays local BONES-SEED G1 CSV clips
directly in the MuJoCo viewer.

### Input Format

The replay script expects a fixed 36-column layout:
- `Frame`
- `root_translateX/Y/Z`
- `root_rotateX/Y/Z`
- 29 `*_joint_dof` columns that map directly to G1 MuJoCo joint names

The script assumes:
- `root_translate*` is in centimeters and converts it to meters
- `root_rotate*` is in degrees
- `*_joint_dof` is in degrees

### Usage

```bash
# Replay the whole flip dataset
uv run scripts/motion/replay_bones_seed_csv.py

# Replay one clip
uv run scripts/motion/replay_bones_seed_csv.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv

# Validate parsing without opening the viewer
uv run scripts/motion/replay_bones_seed_csv.py --dry-run
```

### Controls

- `Space`: pause / resume
- `[`: previous CSV in playlist
- `]`: next CSV in playlist

## BONES-SEED CSV to NPZ

The `bones_seed_csv_to_npz.py` script converts local G1 flip CSV clips into NPZ
files with precomputed forward kinematics.

### Output Format

Generated NPZ files contain:
- `fps`
- `joint_pos`
- `joint_vel`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

### Usage

```bash
# Convert the whole flip dataset into src/unilab/assets/motions/g1/flip_npz
uv run scripts/motion/bones_seed_csv_to_npz.py

# Convert one clip next to a chosen output file
uv run scripts/motion/bones_seed_csv_to_npz.py \
  --input src/unilab/assets/motions/g1/flip/flip_090_001__A304.csv \
  --output temp/flip_090_001__A304.npz

# Validate inputs without exporting
uv run scripts/motion/bones_seed_csv_to_npz.py --dry-run
```

## CSV to NPZ Conversion

The `csv_to_npz.py` script converts motion data from CSV format to NPZ format with precomputed forward kinematics.

### Input Format

CSV files should contain motion data in Unitree's generalized coordinate convention:
- Columns 0-2: Base position (x, y, z)
- Columns 3-6: Base quaternion (x, y, z, w) - will be converted to wxyz internally
- Columns 7+: Joint angles (29 joints for G1)

### Output Format

NPZ files contain:
- `fps`: Frame rate (integer)
- `joint_pos`: Joint positions (N_frames × N_joints)
- `joint_vel`: Joint velocities (N_frames × N_joints)
- `body_pos_w`: Body positions in world frame (N_frames × N_bodies × 3)
- `body_quat_w`: Body quaternions in world frame (N_frames × N_bodies × 4, wxyz)
- `body_lin_vel_w`: Body linear velocities (N_frames × N_bodies × 3)
- `body_ang_vel_w`: Body angular velocities (N_frames × N_bodies × 3)

### Usage

```bash
# Basic usage
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50

# With custom model file
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --model_file path/to/model.xml

# Process specific line range
uv run scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --line_range 100 500
```

### Parameters

- `--input_file`: Path to input CSV file (required)
- `--output_file`: Path to output NPZ file (required)
- `--input_fps`: Frame rate of input CSV (default: 30)
- `--output_fps`: Desired output frame rate (default: 50)
- `--model_file`: MuJoCo model file (default: G1 flat scene)
- `--line_range`: Line range to process [start, end] (optional)

### Notes

- The script uses LERP for position interpolation and SLERP for quaternion interpolation
- Velocities are computed using numerical differentiation
- Forward kinematics is computed using MuJoCo for all bodies
- The output FPS should match the control frequency of your training environment (typically 50 Hz)
