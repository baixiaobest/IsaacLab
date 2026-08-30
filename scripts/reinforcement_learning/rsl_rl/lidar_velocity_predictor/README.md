# Temporal-LiDAR Point Velocity Predictor

Collect scan-time labels with the dedicated fixed-coverage task. `--num_envs` must be a positive multiple of 40; every multiple adds one balanced replica of the ten levels and four terrain columns.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/rollout.py \
  --checkpoint /path/to/kp_policy.pt --num_envs 400 --max_episodes 1000 --headless
```

Audit before training:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/audit.py \
  --dataset_path datasets/lidar_point_velocity --output_dir datasets/lidar_point_velocity/audit
```

Train, evaluate, and export:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/train.py \
  --dataset_path datasets/lidar_point_velocity --run_name first_run
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/evaluate.py \
  --dataset_path datasets/lidar_point_velocity --checkpoint logs/lidar_velocity_predictor/first_run/best.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/export.py \
  --checkpoint logs/lidar_velocity_predictor/first_run/best.pt --output logs/lidar_velocity_predictor/first_run/model.pt
```

The exported model takes `(batch, 2, 4, 128)` and returns `(batch, 128, 2)` robot-body-frame XY velocities.
The rollout writes schema-v2 `point_velocity_b` labels. Archive or remove the earlier world-frame files before
collecting into the same dataset root. Each newly selected best model is also atomically published to
`logs/lidar_velocity_predictor/best_jit.pt` for the dynamic CBF PLAY task.

Training reports each epoch to Weights & Biases by default under the `lidar velocity predictor`
project. Authenticate once with `wandb login`, choose another project with `--wandb_project`, or
train without remote logging via `--logger none`.
`last.pt` is uploaded each epoch; retained checkpoints under `checkpoints/epoch_*.pt` are saved
and uploaded every 10 epochs by default (configure with `--checkpoint_save_interval`).
Whenever validation produces a new best checkpoint, both `best.pt` and its TorchScript export
`best_jit.pt` are saved and uploaded.
