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
    model = TemporalLidarVelocityCNN().to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("projection_contract") != "temporal_lidar_v3":
        raise RuntimeError("Export requires a predictor trained with the temporal_lidar_v3 projection contract.")
    model.load_state_dict(checkpoint["model_state_dict"])
    scripted = torch.jit.script(model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(output), _extra_files={"projection_contract.txt": b"temporal_lidar_v3"})
    print(f"[INFO] Saved body-frame TorchScript predictor to {output}; input=(B,2,4,128), output=(B,128,2)")


if __name__ == "__main__":
    main()
