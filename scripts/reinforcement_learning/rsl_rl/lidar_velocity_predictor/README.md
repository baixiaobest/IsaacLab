# Temporal-LiDAR Point Velocity Predictor

Collect evaluation-time labels with the dedicated fixed-terrain task. `--num_envs` must be a positive multiple of 40; every multiple adds one balanced replica of the ten levels and four terrain columns. Each environment cycles independently through the natural grouped-ray pattern (about 33%) and 40%, 47%, 54%, 60%, 67%, 73%, and 80% filled variants. One sample is saved at every observation step, including later poses while a scan is held.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/rollout.py \
  --checkpoint /path/to/kp_policy.pt --num_envs 400 --max_episodes 1000 --headless
```

Audit before training:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/audit.py \
  --dataset_path datasets/lidar_point_velocity --output_dir datasets/lidar_point_velocity/audit
```

The audit also writes four sampled static and four sampled dynamic labelled-scan plots by default to
`audit/scan_samples/`, with their source file, episode, capture index, and plot path recorded in
`audit/scan_samples.json`. Each plot overlays all collected predictor frames in the evaluation-yaw
coordinates (forward +X, left +Y), including the modeled yaw drift and XY projection noise. Pedestrian
returns progress from red to yellow and static returns from dark grey to light grey as frames age.
Velocity arrows show newest-frame pedestrian labels. Older dataset files lack per-frame return classes
and must be recollected for these plots. Adjust the
number, range, or arrow scale with `--num_scan_samples`, `--scan_plot_range_m`, and
`--velocity_arrow_seconds`.

Train, evaluate, and export:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/train.py \
  --dataset_path datasets/lidar_point_velocity --run_name first_run
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/evaluate.py \
  --dataset_path datasets/lidar_point_velocity --checkpoint logs/lidar_velocity_predictor/first_run/best.pt
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/export.py \
  --checkpoint logs/lidar_velocity_predictor/first_run/best.pt --output logs/lidar_velocity_predictor/first_run/model.pt
```

The navigation policy retains its four-frame scan. The velocity rollout saves eight frames, and training defaults to all available frames; pass `--num_frames 4` to train on the newest four instead. The exported model takes `(batch, 2, num_frames, 128)` and returns `(batch, 128, 2)` absolute obstacle velocities in the yaw-aligned frame of the **evaluation pose**. Only reflected bins have supervised outputs; ignore predictions in other bins. The rollout writes schema-v3 labels and evaluation-pose, capture-time, and measured-coverage metadata. Use a new dataset name for eight-frame collection. Each newly selected best model is also atomically published to
`logs/lidar_velocity_predictor/best_jit.pt` for the dynamic CBF PLAY task.

Temporal-LiDAR policy scans include capture-stable yaw drift in both mixed and non-mixed tasks: each new scan adds an independent Gaussian increment with 0.5° standard deviation, and older world-point scans rotate around their capture position by their accumulated error relative to the newest scan. The newest policy scan and clean critic scan remain aligned with the velocity labels. Existing XY projection noise remains enabled. The rollout records the yaw drift rate in file metadata and refuses to append to files collected with another rate. Collect a new dataset and retrain before using a predictor with this observation distribution.

`src/projection.py` defines the portable `temporal_lidar_v4` projection contract. A variable number of newest-first sparse world-point scans are projected around one evaluation-time position and yaw into 256 world angular bins, then the yaw-centred 128-bin forward arc is returned. Distance is divided by 20 m; validity is 1 for a sampled hit or no-return ray and 0 for an unavailable direction. A no-return ray contributes 20 m, never a close obstacle. `fixtures/projection_fixture_v3.json` remains the four-frame parity fixture. ROS should rebuild the tensor at its snapshot pose and rotate each valid output with that same yaw. The current ROS CBF does not yet run this predictor.

Evaluation reports first-capture and held-pose results and eight measured-coverage bands, each with a zero-velocity baseline. A new best model is published only if its dynamic-return error beats the zero baseline at both the natural sparse and 80% coverage endpoints. Run `evaluate.py` on the held-out test split before deploying the artifact.
TorchScript exports embed the contract version; the Isaac Lab dynamic CBF rejects older, untagged predictor files. Retrain and replace the configured JIT path before running that task.

Training reports each epoch to Weights & Biases by default under the `lidar velocity predictor`
project. Authenticate once with `wandb login`, choose another project with `--wandb_project`, or
train without remote logging via `--logger none`.
`last.pt` is uploaded each epoch; retained checkpoints under `checkpoints/epoch_*.pt` are saved
and uploaded every 10 epochs by default (configure with `--checkpoint_save_interval`).
Whenever validation produces a new best checkpoint, both `best.pt` and its TorchScript export
`best_jit.pt` are saved and uploaded.

To continue a stopped run, provide its trainable checkpoint. This restores the model, AdamW
moments, and the previous best validation score; `--epochs` is the total target epoch, so this
example continues an epoch-20 run through epoch 50:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/train.py \
  --dataset_path datasets/lidar_point_velocity --epochs 50 \
  --resume_checkpoint logs/lidar_velocity_predictor/first_run/last.pt
```

Use `--resume_model_only` with `--resume_checkpoint` to initialize from a previous model while
resetting AdamW and the epoch schedule. Resume from `last.pt` or `best.pt`, not `best_jit.pt`.
