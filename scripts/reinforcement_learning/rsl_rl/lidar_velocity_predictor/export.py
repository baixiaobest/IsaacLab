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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    scripted = torch.jit.script(model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"[INFO] Saved TorchScript predictor to {output}; input=(B,2,4,128), output=(B,128,2)")


if __name__ == "__main__":
    main()
