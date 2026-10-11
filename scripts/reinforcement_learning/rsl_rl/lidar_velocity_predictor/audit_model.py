"""Plot labeled rollout scans with predictor and ground-truth velocity arrows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from audit import _bin_positions, _frame_colors
from src.dataset import PointVelocityDataset
from src.model import TemporalLidarVelocityCNN


def load_model(path: Path, device: torch.device):
    """Load a predictor checkpoint or an exported TorchScript artifact."""
    extras = {"projection_contract.txt": b"", "num_frames.txt": b""}
    try:
        model = torch.jit.load(str(path), map_location=device, _extra_files=extras)
    except (RuntimeError, ValueError):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"{path} is not a predictor checkpoint or TorchScript model.")
        contract = checkpoint.get("projection_contract")
        frames = int(checkpoint.get("num_frames", 4))
        model = TemporalLidarVelocityCNN(frames).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        contract = extras["projection_contract.txt"].decode()
        try:
            frames = int(extras["num_frames.txt"])
        except ValueError as error:
            raise ValueError("TorchScript artifact lacks valid num_frames.txt metadata.") from error
    if contract not in ("temporal_lidar_v3", "temporal_lidar_v4") or frames < 1:
        raise ValueError(f"Unsupported predictor projection contract or frame count: {contract}, {frames}.")
    model.eval()
    return model, frames, contract


def split_indices(dataset: PointVelocityDataset, model_path: Path, splits_path: Path | None) -> tuple[list[int], str]:
    path = splits_path if splits_path is not None else model_path.parent / "splits.json"
    if path.is_file():
        indices = json.loads(path.read_text(encoding="utf-8"))["test_indices"]
        if not isinstance(indices, list) or any(type(i) is not int or i < 0 or i >= len(dataset) for i in indices):
            raise ValueError(f"Invalid test indices in {path}.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate test indices in {path}.")
        return indices, str(path)
    if splits_path is not None:
        raise FileNotFoundError(path)
    _, _, test = dataset.split()
    return list(test.indices), "generated_episode_split_seed_42"


def validate_dataset_contract(dataset: PointVelocityDataset, model_frames: int, model_contract: str) -> None:
    import h5py

    if model_contract == "temporal_lidar_v3" and model_frames != 4:
        raise ValueError("The temporal_lidar_v3 model contract supports four frames only.")
    for path in {entry.file_path for entry in dataset.entries}:
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle["data"].attrs["metadata"])
        contract = metadata.get("projection_contract", "temporal_lidar_v3")
        if contract not in ("temporal_lidar_v3", "temporal_lidar_v4"):
            raise ValueError(f"Unsupported dataset projection contract in {path}: {contract}.")
        if contract != model_contract and model_frames != 4:
            raise ValueError("Cross-version projection is supported only for equivalent four-frame inputs.")


def choose_samples(dataset: PointVelocityDataset, indices: list[int], per_group: int, seed: int):
    """Reservoir-sample static/dynamic and first/held observations independently."""
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    import h5py

    selected_indices = set(indices)
    for entry_id, entry in enumerate(dataset.entries):
        offset = dataset.offsets[entry_id]
        local_indices = [local for local in range(entry.length) if offset + local in selected_indices]
        if not local_indices:
            continue
        with h5py.File(entry.file_path, "r") as handle:
            group = handle["data"][entry.episode_name]
            reflected = np.asarray(group["reflection_mask"], dtype=bool)
            dynamic = np.asarray(group["dynamic_mask"], dtype=bool)
            first = np.asarray(group["first_after_capture"], dtype=bool)
            for local in local_indices:
                if not reflected[local].any():
                    continue
                category = ("dynamic" if (dynamic[local] & reflected[local]).any() else "static")
                category += "_first" if first[local] else "_held"
                counts[category] += 1
                selected = reservoirs[category]
                index = offset + local
                if len(selected) < per_group:
                    selected.append(index)
                elif per_group:
                    replacement = int(rng.integers(counts[category]))
                    if replacement < per_group:
                        selected[replacement] = index
    return {name: sorted(values) for name, values in reservoirs.items()}, dict(counts)


def plot_sample(path: Path, lidar: np.ndarray, target: np.ndarray, prediction: np.ndarray,
                reflection: np.ndarray, dynamic: np.ndarray, max_range_m: float,
                arrow_seconds: float, title: str, history_dynamic: np.ndarray | None = None) -> dict:
    """Render the actual model input and paired velocity arrows in evaluation-yaw axes."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    frames = lidar.shape[1]
    if lidar.shape != (2, frames, 128) or target.shape != prediction.shape != (128, 2):
        raise ValueError("Invalid input or velocity shape for model audit plot.")
    if reflection.shape != (128,) or dynamic.shape != (128,):
        raise ValueError("Invalid newest-frame label masks.")
    if not np.isfinite(lidar).all() or not np.isfinite(prediction[reflection]).all():
        raise ValueError("Nonfinite model input or supervised prediction.")
    # Labels can include a reflection at the 20 m saturation limit. It cannot
    # be distinguished visually from a no-return ray in the model tensor.
    visible = reflection & (lidar[1, 0] > 0.5) & (lidar[0, 0] < 1.0) & (lidar[0, 0] > 0.0)
    if np.any(reflection & (lidar[1, 0] <= 0.5)):
        raise ValueError("Newest predictor input marks a labeled reflection unavailable.")
    fig, axes = plt.subplots(figsize=(8, 8))
    static_colors, dynamic_colors = _frame_colors(frames)
    if history_dynamic is not None and history_dynamic.shape != (frames, 128):
        raise ValueError("Invalid historical dynamic mask shape.")
    for age in reversed(range(frames)):
        ranges = lidar[0, age] * 20.0
        returns = (lidar[1, age] > 0.5) & (lidar[0, age] > 0.0) & (lidar[0, age] < 1.0)
        returns &= ranges <= max_range_m
        points = _bin_positions(ranges)
        pedestrian = returns & (history_dynamic[age] if history_dynamic is not None else
                                (dynamic if age == 0 else np.zeros(128, dtype=bool)))
        for mask, color in ((returns & ~pedestrian, static_colors[age]), (pedestrian, dynamic_colors[age])):
            axes.scatter(points[mask, 0], points[mask, 1], s=12 if age else 28,
                         color=color, alpha=0.55 if age else 0.9, zorder=2 if age == 0 else 1)
    points = _bin_positions(lidar[0, 0] * 20.0)
    visible &= lidar[0, 0] * 20.0 <= max_range_m
    static = visible & ~dynamic
    moving = visible & dynamic
    axes.scatter(points[static, 0], points[static, 1], s=34, facecolors="none", edgecolors="black", zorder=4)
    axes.scatter(points[moving, 0], points[moving, 1], s=34, facecolors="none", edgecolors="red", zorder=4)
    for index in np.flatnonzero(visible):
        start = points[index]
        for vector, color, linestyle in ((target[index], "#1874bd", "solid"),
                                         (prediction[index], "#ec8a19", "dashed")):
            if np.linalg.norm(vector) < 1e-6:
                continue
            axes.annotate("", xy=start + arrow_seconds * vector, xytext=start,
                          arrowprops={"arrowstyle": "->", "color": color, "lw": 1.3,
                                      "linestyle": linestyle}, zorder=5)
    axes.scatter([0], [0], marker="^", c="tab:blue", s=70, zorder=6)
    axes.axhline(0, color="0.85", linewidth=0.8)
    axes.axvline(0, color="0.85", linewidth=0.8)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim(-0.5, max_range_m + 0.5)
    axes.set_ylim(-max_range_m - 0.5, max_range_m + 0.5)
    axes.set_xlabel("evaluation-yaw X: forward (m)")
    axes.set_ylabel("evaluation-yaw Y: left (m)")
    axes.set_title(title)
    axes.grid(alpha=0.2)
    axes.legend(handles=[Line2D([], [], color="#1874bd", label="ground truth"),
                         Line2D([], [], color="#ec8a19", linestyle="dashed", label="prediction"),
                         Line2D([], [], color="black", marker="o", markerfacecolor="none", linestyle="None", label="static"),
                         Line2D([], [], color="red", marker="o", markerfacecolor="none", linestyle="None", label="pedestrian")],
                loc="upper left", fontsize=8)
    from matplotlib.colors import ListedColormap

    inset = axes.inset_axes((0.64, 0.85, 0.32, 0.09))
    inset.imshow(np.arange(2 * frames).reshape(2, frames),
                 cmap=ListedColormap(list(dynamic_colors) + list(static_colors)),
                 aspect="auto", vmin=-0.5, vmax=2 * frames - 0.5, interpolation="nearest")
    inset.set_yticks([0, 1], ["pedestrian", "static"], fontsize=7)
    inset.set_xticks([0, frames - 1], ["now", f"−{frames - 1}"], fontsize=7)
    inset.tick_params(length=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    error = np.linalg.norm(prediction - target, axis=-1)
    speed = np.linalg.norm(prediction, axis=-1)
    return {"static_bins": int(static.sum()), "dynamic_bins": int(moving.sum()),
            "static_false_motion_mean_mps": float(speed[static].mean()) if static.any() else None,
            "dynamic_velocity_error_mean_mps": float(error[moving].mean()) if moving.any() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model", required=True, help="Training checkpoint or exported TorchScript .pt file.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits_path", default=None, help="Optional splits.json; otherwise use model sibling or episode split seed 42.")
    parser.add_argument("--input_name", choices=("both", "lidar_noisy", "lidar_clean"), default="both",
                        help="Render both saved inputs for each sample by default.")
    parser.add_argument("--num_samples_per_group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan_plot_range_m", type=float, default=10.0)
    parser.add_argument("--velocity_arrow_seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.num_samples_per_group < 0 or args.scan_plot_range_m <= 0 or args.velocity_arrow_seconds <= 0:
        parser.error("Sample count must be nonnegative, and plot range and arrow scale must be positive.")
    model_path = Path(args.model).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    device = torch.device(args.device)
    model, frames, contract = load_model(model_path, device)
    dataset = PointVelocityDataset(args.dataset_path, input_name="lidar_noisy", num_frames=frames)
    validate_dataset_contract(dataset, frames, contract)
    indices, split_source = split_indices(dataset, model_path,
                                           Path(args.splits_path).expanduser().resolve() if args.splits_path else None)
    selected, eligible_counts = choose_samples(dataset, indices, args.num_samples_per_group, args.seed)
    manifest = []
    import h5py

    for category, samples in sorted(selected.items()):
        for index in samples:
            entry_id = int(np.searchsorted(dataset.offsets, index, side="right") - 1)
            entry = dataset.entries[entry_id]
            local = index - dataset.offsets[entry_id]
            sample = dataset[index]
            with h5py.File(entry.file_path, "r") as handle:
                group = handle["data"][entry.episode_name]
                capture_index = int(group["capture_index"][local])
                capture_time = float(group["capture_time_s"][local])
                evaluation_time = float(group["evaluation_time_s"][local])
                history_dynamic = (np.asarray(group["history_dynamic_mask"][local], dtype=bool)[:frames]
                                   if "history_dynamic_mask" in group else None)
                inputs = {"lidar_noisy": sample["lidar"].numpy()}
                if args.input_name in ("both", "lidar_clean"):
                    inputs["lidar_clean"] = dataset._canonical_lidar(
                        group["lidar_clean"][local], f"{entry.file_path}:{entry.episode_name}/lidar_clean"
                    )
            name = f"{category}_{Path(entry.file_path).stem}_{entry.episode_name}_{local:06d}.png"
            selected_inputs = ("lidar_noisy", "lidar_clean") if args.input_name == "both" else (args.input_name,)
            plots = {}
            metrics = {}
            for input_name in selected_inputs:
                lidar = inputs[input_name]
                with torch.inference_mode():
                    predicted = model(torch.from_numpy(lidar).unsqueeze(0).to(device)).detach().cpu().numpy()
                if predicted.shape != (1, 128, 2):
                    raise ValueError(f"Predictor returned {predicted.shape}, expected (1, 128, 2).")
                plot_path = output / "samples" / input_name / name
                metrics[input_name] = plot_sample(
                    plot_path, lidar, sample["target"].numpy(), predicted[0],
                    sample["reflection_mask"].numpy(), sample["dynamic_mask"].numpy(),
                    args.scan_plot_range_m, args.velocity_arrow_seconds,
                    f"{input_name} | {category} | {entry.episode_name} | capture {capture_index}\n"
                    f"coverage {float(sample['ray_coverage']):.0%}, age {evaluation_time - capture_time:.3f} s",
                    history_dynamic,
                )
                plots[input_name] = str(plot_path)
            manifest.append({"category": category, "file_path": entry.file_path, "episode_name": entry.episode_name,
                             "sample_index": local, "capture_index": capture_index, "capture_time_s": capture_time,
                             "evaluation_time_s": evaluation_time, "ray_coverage": float(sample["ray_coverage"]),
                             "plots": plots, "metrics_by_input": metrics})
    output.mkdir(parents=True, exist_ok=True)
    def aggregate(input_name: str, metric: str, count: str) -> float | None:
        values = [item["metrics_by_input"][input_name] for item in manifest]
        total = sum(item[count] for item in values if item[metric] is not None)
        return (sum(item[metric] * item[count] for item in values if item[metric] is not None) / total
                if total else None)

    aggregate_metrics = {
        input_name: {
            "plotted_static_false_motion_mean_mps": aggregate(input_name, "static_false_motion_mean_mps", "static_bins"),
            "plotted_dynamic_velocity_error_mean_mps": aggregate(input_name, "dynamic_velocity_error_mean_mps", "dynamic_bins"),
        }
        for input_name in (("lidar_noisy", "lidar_clean") if args.input_name == "both" else (args.input_name,))
    }

    report = {"model": str(model_path), "projection_contract": contract, "num_frames": frames,
              "input_name": args.input_name, "split_source": split_source,
              "test_sample_count": len(indices), "eligible_counts": eligible_counts,
              "velocity_arrow_seconds": args.velocity_arrow_seconds, "plotted_metrics_by_input": aggregate_metrics,
              "samples": manifest}
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote {len(manifest)} samples with {len(aggregate_metrics)} input view(s) to {output / 'samples'}")


if __name__ == "__main__":
    main()
