# rosa_main

Extract cross-section loops from a point cloud and fit them with the shared mini
network. This is the small, self-contained pipeline behind the seam and
orientation experiments; it is separate from `main.py` / `README.md`.

## What it needs

| file | role |
|---|---|
| `rosa_main.py` | command line and the run |
| `loop_extraction.py` | slicing, loop centres, PLY and figure writing |
| `mini_nn.py` | network, trainer, warm-up load, seam distillation |
| `warmup/cube_warmup_axis.pt` | warm-up checkpoint (initialisation and distill teacher) |
| an input `.ply` with normals | default `normalized_shapes/laurent_hand_normal.ply` |

Python packages: PyTorch (CUDA), NumPy, SciPy, trimesh, matplotlib. Use the
environment that has them, e.g. `/home/lyangiw/anaconda3/envs/seawall/bin/python`;
the plain `python` on this machine fails on `libcudnn.so.8`.

## Run

Extraction only (no fitting):

```bash
python rosa_main.py \
    --path normalized_shapes_cuboid/rocker-arm_points_cubic_lam3.ply \
    --samples 29
```

Fit:

```bash
python rosa_main.py \
    --path normalized_shapes_cuboid/rocker-arm_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane \
    --epochs 1000 --normal 0.01 --triplane-tv 0.01 \
    --snapshot-every 0 --out log/axis_tri_tv01_rocker-arm
```

Add seam distillation (teacher is the warm-up checkpoint):

```bash
python rosa_main.py \
    --path normalized_shapes_cuboid/rocker-arm_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane \
    --epochs 1000 --normal 0.01 --triplane-tv 0.01 \
    --snapshot-every 0 --distill 10 --out log/axis_seam10_rocker-arm
```

`--no-warmup` fits from a blank network; it cannot be combined with `--distill`.

## Flags that matter

| flag | meaning |
|---|---|
| `--path` | input cloud |
| `--samples` | planes per axis (29 in the experiments) |
| `--train` | fit; without it the run stops after extraction |
| `--feature triplane` | triplane conditioning; required by the warm-up and distillation |
| `--normal` | weight of the direction term (0.01 in the experiments) |
| `--triplane-tv` | total-variation weight on the three grids (0.01) |
| `--warmup` | warm-up checkpoint, default `warmup/cube_warmup_axis.pt` |
| `--distill W` | seam-distillation weight; 0 is off |
| `--distill-warmup` | teacher checkpoint, defaults to `--warmup` |
| `--epochs` | training epochs, one batch of all loops per epoch |
| `--snapshot-every` | checkpoint cadence; 0 disables snapshots |
| `--out` | run folder, default `log/<input stem>/` |

The warm-up checkpoint must match the run's network and grid: axis-conditioned,
2x64 MLP, 32x32x3 grids with 16 channels. A mismatch is refused instead of being
loaded in part.

## Outputs in `--out`

- `<stem>_slices.ply`, `<stem>_slices.png` — extraction, one colour per slice
- `<stem>_model.pt` — fitted network plus replay metadata
- `<stem>_good_loops.ply`, `<stem>_bad_loops.ply`, `<stem>_loop_split.txt` —
  predictions split at the worst-point error limit, and the per-loop flag
- `<stem>_fits.png` — target vs fitted curve for every loop
- `<stem>_epoch_terms.csv`, `<stem>_loop_winding.npz` — per-epoch losses and
  signed winding per loop
- `<stem>_triplane.npz`, `<stem>_feature_graph.csv`,
  `<stem>_feature_smoothness.txt` — field samples and the loop-pair graph
- `planes.txt`, `planes/` — per-plane readings and figures

`bad` in the final report means a loop with a point past `0.15` from its target.
The `bad_loops.txt` shortlist uses the separate `--max-err` mean-error rule.

## More

Design, measurements and the six-shape distillation table are in
[WORKFLOW.md](WORKFLOW.md) §2.3–2.4; the open items are in [todo.md](todo.md).