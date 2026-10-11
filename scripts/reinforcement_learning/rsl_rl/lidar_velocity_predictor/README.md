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

Compare a trained checkpoint or exported TorchScript model with rollout labels:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/lidar_velocity_predictor/audit_model.py \
  --dataset_path datasets/lidar_point_velocity \
  --model logs/lidar_velocity_predictor/first_run/best.pt \
  --output_dir datasets/lidar_point_velocity/model_audit
```

`audit_model.py` samples up to four examples in each of four groups: static or
pedestrian, first capture or held scan. Each sample gets paired `lidar_noisy`
and `lidar_clean` inference plots under `samples/`, using its respective saved
scan tensor and the same velocity labels. The manifest reports both sets of
plotted-bin errors. This comparison changes all simulated input corruptions
together and does not isolate any one noise source. The tool uses the held-out test episodes from
the model's sibling `splits.json` when present, otherwise the dataset's seeded
episode split. Pass `--splits_path` to select a specific split file. Plots use
each saved input, overlay solid blue ground-truth and dashed orange
predicted arrows, and mark static bins with black rings and pedestrian bins with
red rings. Both arrows use the same `--velocity_arrow_seconds` scale, defaulting
to 1 m per 1 m/s. Only reflected newest-frame bins receive arrows. The
`manifest.json` records source samples and plotted-bin errors. Use
`--input_name lidar_noisy` or `--input_name lidar_clean` to render only one input, or adjust sampling with
`--num_samples_per_group`, `--seed`, and `--scan_plot_range_m`.

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

The policy and predictor add at most 5 cm of uniform noise to each reflected
range after normalizing by the 20 m limit (normalized amplitude 0.0025).
Unavailable and no-return distances, and the binary validity channel, remain
unchanged. Older rollouts used uniform amplitude 0.05 on both normalized
channels, equivalent to up to 1 m of range noise. New rollouts record the
range-noise setting and cannot append to those older files. Recollect data and
retrain before treating an existing predictor as validated with this input.

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
