"""Export the trained point-velocity predictor as TorchScript."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import TemporalLidarVelocityCNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    contract = checkpoint.get("projection_contract")
    if contract not in ("temporal_lidar_v3", "temporal_lidar_v4"):
        raise RuntimeError("Export requires a predictor trained with a supported evaluation-pose projection contract.")
    num_frames = int(checkpoint.get("num_frames", 4))
    model = TemporalLidarVelocityCNN(
        num_frames, bool(checkpoint.get("aux_classifier", False)), float(checkpoint.get("dynamic_threshold") or 0.3)
    ).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    scripted = torch.jit.script(model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_files = {
        "projection_contract.txt": contract.encode(), "num_frames.txt": str(num_frames).encode(),
        "aux_classifier.txt": str(int(model.aux_classifier)).encode(),
    }
    if model.aux_classifier:
        extra_files["dynamic_threshold.txt"] = str(model.dynamic_threshold).encode()
    if checkpoint.get("range_noise_m") is not None:
        extra_files["range_noise_m.txt"] = str(checkpoint["range_noise_m"]).encode()
        extra_files["range_noise_applies_to.txt"] = str(checkpoint["range_noise_applies_to"]).encode()
    torch.jit.save(scripted, str(output), _extra_files=extra_files)
    print(f"[INFO] Saved body-frame TorchScript predictor to {output}; input=(B,2,{num_frames},128), output=(B,128,2)")


if __name__ == "__main__":
    main()
