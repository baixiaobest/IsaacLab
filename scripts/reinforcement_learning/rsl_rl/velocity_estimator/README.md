# Velocity Estimator Workflow

This folder contains the end-to-end workflow for training a velocity estimator from policy rollouts, evaluating it online, and exporting a black-box TorchScript module that combines the estimator with the policy.

The intended pipeline is:

1. Collect rollout data with `rollout.py`.
2. Train the estimator with `train_estimator.py`.
3. Evaluate the estimator online with `play_estimator.py`.
4. Export a combined TorchScript policy wrapper with `policy_estimator_jit_generator.py`.


## Folder Overview

- `rollout.py`
  Collects policy rollouts and writes episode-wise HDF5 datasets.
- `dataset.py`
  Loads rollout HDF5 files and creates sliding-window training samples.
- `model.py`
  Defines the `VelocityEstimator` MLP.
- `train_estimator.py`
  Trains the estimator and saves checkpoints plus metadata.
- `play_estimator.py`
  Runs the policy while replacing policy velocity inputs with estimator outputs.
- `policy_estimator_jit_generator.py`
  Exports a TorchScript module that runs estimator plus policy as one black-box model.
- `observation_utils.py`
  Shared utilities for reconstructing named observation terms from concatenated observation tensors.
- `checkpoint_utils.py`
  Shared utilities for resolving policy checkpoints and rebuilding estimator checkpoints.


## Required Environment Contract

This workflow assumes the task exposes:

- a `policy` observation group used by the RL policy
- a `ground_truth` observation group containing the target velocity terms

For the current GO2 locomotion setup, the important observation terms are:

- `policy/base_lin_vel`
- `policy/base_ang_vel`
- `ground_truth/base_lin_vel`
- `ground_truth/base_ang_vel`

The estimator is trained to predict the `ground_truth` velocity terms from the remaining `policy` terms.


## Rollout Data Generation

`rollout.py` runs a trained policy in Isaac Lab and writes episode data into HDF5 files.

### What gets stored

Each completed episode is written under the Isaac Lab dataset layout and contains:

- `observations/...`
  Named policy observations except `ground_truth`.
- `ground_truth/...`
  The target velocity terms used later for estimator training.
- `actions`
- `rewards`
- `dones`
- `time_outs`
- `step_index`
- `metadata/...`
  Episode length, source environment id, and termination flags.

Files are rotated after `--episodes_per_file` completed episodes.

### Important behavior

- Rollouts are stored per environment, per episode.
- Observation tensors are split back into named terms using the observation manager layout.
- `ground_truth` is written as a top-level target group, not mixed back into `observations`.
- Termination reasons are taken from `extras["log"]`, because termination flags are cleared during reset inside the env step path.

### Example

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/velocity_estimator/rollout.py \
  --task Isaac-Locomotion-Vel-Unitree-Go2-Rollout-v0 \
  --num_envs 50 \
  --checkpoint logs/rsl_rl/EncoderActorCriticGO2/Locomotion/model_1999.pt \
  --dataset_root datasets/rollout \
  --dataset_name go2_locomotion \
  --episodes_per_file 1000 \
  --max_episodes 3000 \
  --headless
```


## Dataset and Training

`dataset.py` converts the rollout HDF5 files into training samples for the estimator.

### Sliding-window dataset

For each time step in each episode:

- the input is a fixed horizon of past observations, including the current step
- the target is the current `ground_truth` velocity

If the episode is shorter than the requested history near the start of an episode, the first frame is repeated on the left so every sample still has exactly `horizon` frames.

### Input and target selection

The dataset automatically infers:

- `input_paths` from `observations/...`
- `target_paths` from `ground_truth/...`

Target term names are automatically excluded from the estimator inputs. That means if the target is `base_lin_vel` and `base_ang_vel`, those terms are removed from the model input even if they exist in `policy` observations.

### Train/validation split

Training and validation are split by episode, not by individual samples. This keeps windows from the same rollout episode from leaking across the split.

### Model

`model.py` defines `VelocityEstimator`, an MLP that expects input of shape:

```text
(batch, horizon, input_dim)
```

The model flattens the horizon dimension internally before applying the MLP.

### Training outputs

`train_estimator.py` saves:

- `best.pt`
- `last.pt`
- optional `checkpoints/epoch_XXXX.pt`
- `metadata.json`
- TensorBoard logs under `tensorboard/`
- optional WandB artifacts

Each checkpoint includes:

- model weights
- optimizer state
- `input_paths`
- `target_paths`
- `input_dim`
- `target_dim`
- `horizon`
- CLI args used for training

Those schema fields are what later runtime and export scripts use to stay consistent with training.

### Example

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/velocity_estimator/train_estimator.py \
  --dataset_path datasets/rollout \
  --output_dir logs/velocity_estimator \
  --run_name go2_locomotion \
  --horizon 10 \
  --batch_size 4096 \
  --epochs 50 \
  --checkpoint_save_interval 5 \
  --learning_rate 0.001 \
  --validation_fraction 0.1
```


## Online Estimator Usage

`play_estimator.py` evaluates the estimator online by inserting estimated velocity back into the policy observation before the action is computed.

### How it works

At every simulation step:

1. The script reads the full named observation dictionary from `extras["observations"]`.
2. It rebuilds the estimator input from the checkpoint `input_paths`.
3. It maintains a rolling history buffer of shape `(num_envs, horizon, input_dim)`.
4. The estimator predicts the current velocity.
5. The predicted `base_lin_vel` and `base_ang_vel` replace the corresponding slices in the flat `policy` observation tensor.
6. The policy acts using this modified observation.
7. The script compares estimator outputs with `ground_truth` and accumulates RMSE statistics.

At the start of an episode, the history buffer is initialized or reset by repeating the newest available feature frame.

### Output

The script periodically prints:

- step count
- completed episode count
- simulation-to-real-time ratio
- cumulative RMSE for each velocity term and the total

### Example

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/velocity_estimator/play_estimator.py \
  --task Isaac-Locomotion-Vel-Unitree-Go2-Play-v0 \
  --num_envs 20 \
  --checkpoint logs/rsl_rl/EncoderActorCriticGO2/Locomotion/model_1999.pt \
  --estimator_checkpoint logs/velocity_estimator/go2_locomotion/checkpoints/epoch_0010.pt
```


## TorchScript Export for Black-Box Usage

`policy_estimator_jit_generator.py` exports a single TorchScript module that combines:

- the trained velocity estimator
- the trained RL policy
- the policy observation normalizer
- the logic that injects estimated velocity into the policy observation

The exported module is intended for direct black-box inference without needing to reimplement the observation replacement logic externally.

### Why the exporter instantiates an environment

The exporter currently uses the task environment to recover the exact policy observation layout from the observation manager. That gives it the correct slices for:

- non-velocity policy terms copied directly from the input feature vector
- velocity terms that must be replaced with estimator outputs

It also uses the environment to instantiate `OnPolicyRunner` so the policy and observation normalizer are rebuilt exactly as the training stack expects.

### Exported module contract

The scripted module expects input of shape:

```text
(batch, horizon, input_dim)
```

where:

- `horizon` matches the estimator training horizon
- `input_dim` matches the flattened per-step estimator feature size
- per-step feature ordering matches `input_paths` saved in the estimator checkpoint

For the current GO2 run, the generator reported:

```text
Expected input shape: (batch, 10, 42)
Input terms: ['policy/actions', 'policy/joint_pos', 'policy/joint_vel', 'policy/projected_gravity', 'policy/velocity_commands']
Estimated velocity terms: ['base_ang_vel', 'base_lin_vel']
```

That means each step of the input horizon must contain the concatenation of those five policy terms in exactly that order.

### What the JIT does internally

At inference time the scripted module:

1. Runs the estimator on the full input horizon.
2. Takes the newest frame from the horizon as the current non-velocity feature vector.
3. Rebuilds the policy observation tensor.
4. Inserts estimator outputs into the `base_lin_vel` and `base_ang_vel` slices.
5. Applies the policy normalizer.
6. Runs the policy actor.
7. Returns the action tensor.

The scripted module also exports:

- `forward(inputs)` for actions
- `estimate_velocity(inputs)` for velocity-only prediction
- `reset()` for recurrent policy state handling

### Example

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/velocity_estimator/policy_estimator_jit_generator.py \
  --task Isaac-Locomotion-Vel-Unitree-Go2-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/EncoderActorCriticGO2/Locomotion/model_1999.pt \
  --estimator_checkpoint logs/velocity_estimator/go2_locomotion/checkpoints/epoch_0010.pt \
  --output logs/rsl_rl/EncoderActorCriticGO2/Locomotion/exported/policy_estimator.pt \
  --headless
```


## Typical Workflow

1. Add a `ground_truth` observation group to the task config with the target velocity terms.
2. Generate rollout HDF5 files with `rollout.py`.
3. Train the estimator with `train_estimator.py`.
4. Check online behavior and RMSE with `play_estimator.py`.
5. Export a black-box TorchScript module with `policy_estimator_jit_generator.py`.
6. In deployment, feed the scripted module a horizon of non-velocity policy observations in the checkpoint-defined order.


## Notes and Caveats

- The rollout and online scripts assume concatenated observation groups.
- The estimator/export path currently supports manager-based RL environments.
- The exported TorchScript module depends on the estimator checkpoint schema. If the observation layout changes, regenerate rollouts, retrain, and re-export.
- If HDF5 rollout files become unreadable, `dataset.py` will skip corrupted files when valid files are present and will fail with a detailed message if all files are invalid.