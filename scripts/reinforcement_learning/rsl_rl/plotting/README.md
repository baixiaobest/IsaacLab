# Crossing behavior analysis

`crossing_behavior_analysis.py` builds a bird's-eye-view grid from the saved
interaction replay clips. It reads each clip's existing `canonical_label` from
`interaction_event_cases.json` and uses only `assert` and `yield` cases. It does
not invoke the event detector or assign new labels.

Run it on an evaluation directory:

```bash
python repo/scripts/reinforcement_learning/rsl_rl/plotting/crossing_behavior_analysis.py \
  /path/to/evaluation/2026-09-07_07-56-18 \
  --r-min 1 --r-max 4 --grid-resolution 0.25 --min-samples 5
```

The output directory contains:

- `crossing_behavior_heatmap.png`: masked P(ASSERT) heatmap, with +X right and
  +Y forward in the robot frame.
- `crossing_behavior_grid.csv`: one row per cell with bounds, counts,
  probability, and mask status.
- `crossing_behavior_grid.npz`: the same grid as arrays, including raw and
  masked probabilities, edges, configuration, and source metadata.

Only active samples inside each captured event's `[start_time_s, end_time_s]`
interval are used; replay padding is excluded. The distance filter is inclusive
at both `r_min` and `r_max`. Custom `--x-limits` and `--y-limits` can crop the
otherwise square `[-r_max, r_max]` grid.

For a higher-resolution grid, reduce the cell size, for example
`--grid-resolution 0.05 --dpi 300`.

Structured telemetry uses labels from an existing immutable detector run and
does not reclassify them:

```bash
python repo/scripts/reinforcement_learning/rsl_rl/plotting/crossing_behavior_analysis.py \
  --database /srv/research-agent/experiments/research_agent.sqlite3 \
  --telemetry-root /srv/research-agent/experiments \
  --detector-run detector-... \
  --grid-resolution 0.05 --min-samples 5 --output-dir ./crossing-bev
```
