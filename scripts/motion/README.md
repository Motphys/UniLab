# Motion Preprocessing

This directory contains scripts for preprocessing motion data for motion tracking tasks.

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
uv run python scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50

# With custom model file
uv run python scripts/motion/csv_to_npz.py \
  --input_file path/to/motion.csv \
  --output_file path/to/motion.npz \
  --input_fps 30 \
  --output_fps 50 \
  --model_file path/to/model.xml

# Process specific line range
uv run python scripts/motion/csv_to_npz.py \
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
