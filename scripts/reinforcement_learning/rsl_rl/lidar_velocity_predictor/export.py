"""Export the trained point-velocity predictor as TorchScript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.model import TemporalLidarVelocityCNN


def _resolve_fov_bins(checkpoint: str, explicit: int | None) -> int:
    """Return ``explicit`` if given, else auto-detect from the checkpoint's sibling metadata.json."""
    if explicit is not None:
        return explicit
    metadata_file = Path(checkpoint).resolve().parent / "metadata.json"
    if metadata_file.exists():
        fov_bins = json.loads(metadata_file.read_text(encoding="utf-8")).get("args", {}).get("fov_bins")
        if fov_bins is not None:
            return int(fov_bins)
    return 128


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--fov_bins",
        type=int,
        default=None,
        help="Forward-arc bin count. Defaults to auto-detecting from the checkpoint's metadata.json, else 128.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    fov_bins = _resolve_fov_bins(args.checkpoint, args.fov_bins)
    model = TemporalLidarVelocityCNN(fov_bins=fov_bins).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model_state_dict"])
    scripted = torch.jit.script(model)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    print(f"[INFO] Saved body-frame TorchScript predictor to {output}; input=(B,2,4,{fov_bins}), output=(B,{fov_bins},2)")


if __name__ == "__main__":
    main()
