# rosa_main

Extract cross-section loops from a point cloud and fit them with the shared mini
network. This is the small, self-contained pipeline behind the seam and
orientation experiments; it is separate from `main.py` / `README.md`.

Project-wide status — current branch, latest results, in-progress work and open
items — is in [PROJECT_STATE.md](PROJECT_STATE.md).

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

Frozen start — hold the shared MLP for the first 100 epochs, then fit as above:

```bash
python rosa_main.py \
    --path normalized_shapes_cuboid/rocker-arm_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane \
    --epochs 1000 --normal 0.01 --triplane-tv 0.01 \
    --snapshot-every 0 --distill 10 --freeze-f-epochs 100 \
    --out log/axis_frz100_rocker-arm
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
| `--freeze-f-epochs N` | freeze the shared MLP `f` for the first N epochs; only the triplane grid trains (100 in the experiments) |
| `--epochs` | training epochs, one batch of all loops per epoch |
| `--snapshot-every` | checkpoint cadence; 0 disables snapshots |
| `--out` | run folder, default `log/<input stem>/` |

The warm-up checkpoint must match the run's network and grid: axis-conditioned,
2x64 MLP, 32x32x3 grids with 16 channels. A mismatch is refused instead of being
loaded in part.

## Frozen start

`--freeze-f-epochs N` holds the shared MLP `f` fixed for the first N epochs; only
the triplane grid trains, so the field's only way to reduce chamfer is to move
each loop's query onto its target points. Epoch lines carry `[f frozen]` and the
boundary prints `f map released`; the freeze is a placement initialiser, and the
rest of the run is the ordinary fit. Later refinement stages inherit the same
budget, so they do not refund the frozen epochs.

On the panel of eight shapes (29 planes per axis, 1000 epochs, controls are the
matched `--distill 10` runs without the freeze, 1239 loops), the freeze cuts
loops whose own-centre winding is reversed from 207 to 47:

| shape | loops | fit ctl -> frz | reversed ctl -> frz | tv ctl -> frz |
|---|---:|---|---|---|
| bunny | 80 | 0.0010 -> 0.0012 | 5 -> 0 | 0.00070 -> 0.00536 |
| armadillo | 162 | 0.0013 -> 0.0011 | 4 -> 1 | 0.00213 -> 0.01224 |
| horse | 139 | 0.0005 -> 0.0005 | 41 -> 8 | 0.00154 -> 0.00762 |
| lucy | 157 | 0.0007 -> 0.0007 | 33 -> 9 | 0.00238 -> 0.01216 |
| laurent_hand | 149 | 0.0006 -> 0.0011 | 41 -> 16 | 0.00195 -> 0.00687 |
| rocker-arm | 153 | 0.0007 -> 0.0007 | 59 -> 1 | 0.00149 -> 0.00446 |
| neptune | 189 | 0.0005 -> 0.0006 | 20 -> 1 | 0.00255 -> 0.01046 |
| dancing_children100K | 210 | 0.0009 -> 0.0009 | 4 -> 11 | 0.00308 -> 0.01361 |

Seven of eight shapes improve; dancing children is the exception and laurent
hand keeps its z-family reversal (y goes 22 -> 0, z 15 -> 14), so those two need
a seeded repeat. The fit cost is transient: at epoch 99 the frozen runs read
0.022-0.070 against the controls' 0.002-0.0035, and by epoch 999 only laurent
hand and bunny differ by more than 20%. The field is 3-8x rougher in tv. No seed
is fixed, so the discrete winding counts move between runs (horse x reads 8/43
here against 2/43 in `freeze100_full`) — read the table as direction and
magnitude, not exact counts. The 50-epoch freeze is not enough: the late z
turn-over returns.

## Clustering the loop codes

`code_clusters.py` reads a run's `<stem>_triplane.npz` and clusters the 48-D
triplane code at each loop query (cosine average linkage). k defaults to the
silhouette choice; `--k 10` makes shapes comparable.

```bash
# final state of a run
python code_clusters.py \
    log/axis_frz100_rocker-arm/rocker-arm_points_cubic_lam3_triplane.npz --k 10

# an early state, saved by a replay at epoch 100
python code_clusters.py \
    log/axis_frz100_rocker-arm/rocker-arm_points_cubic_lam3_triplane.npz \
    --state log/frz100_replay/rocker-arm/state_e100.pt --tag e0100 --k 10
```

It writes `<stem>_code_clusters[_tag].png/.csv/.ply` (a sphere PLY per cluster)
and prints silhouette plus per-cluster size, axis mix and intra/inter cosine.
`--kmax` bounds the auto search and `--sphere-radius` sizes the PLY markers.

At k = 10 the frozen arm separates its codes better on every shape (silhouette
+0.03..+0.13); the auto-k numbers move the same way. The epoch-100 partition is
already the run's final one, and the clusters are the axis families:

| shape | sil ctl | sil frz | sil frz e100 | ARI e100/final | Jaccard |
|---|---:|---:|---:|---:|---:|
| bunny | 0.487 | 0.590 | 0.628 | 0.703 | 0.843 |
| armadillo | 0.418 | 0.452 | 0.468 | 0.646 | 0.666 |
| horse | 0.434 | 0.561 | 0.578 | 0.897 | 0.795 |
| lucy | 0.350 | 0.459 | 0.509 | 0.931 | 0.888 |
| laurent_hand | 0.390 | 0.486 | 0.454 | 0.841 | 0.812 |
| rocker-arm | 0.410 | 0.550 | 0.597 | 0.819 | 0.752 |
| neptune | 0.380 | 0.432 | 0.505 | 0.794 | 0.785 |
| dancing_children100K | 0.207 | 0.336 | 0.381 | 0.638 | 0.620 |

The fit stage slightly mixes some groups: e100 silhouette is above the final one
on seven shapes, and only laurent hand rises during the fit (0.454 -> 0.486),
the same shape whose placement is still unfinished at e100. The two arms end on
different partitions (ARI 0.35-0.63), so the frozen start is a cleaner grouping,
not a reparametrisation of the same one. The epoch-100 states are replayed with
the same command and 1000-epoch schedule and kept as
`log/frz100_replay/<shape>/state_e100.pt`.

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

The runtime files above and `code_clusters.py` stay at the repository root.
Follow-up readings of a finished run — `seam_check.py`,
`loop_propagation.py`, the `plot_*` scripts and the probes — live in
[plc_utils](plc_utils/) and run as `python plc_utils/<script>.py` or
`python -m plc_utils.<script>`.

Design and measurements are in [WORKFLOW.md](WORKFLOW.md): §2.3–2.4 for the
loss terms, the cube warm-up, the six-shape distillation table and the
eight-shape frozen panel; §4 for feature clustering and the loop tools. The open
items, including the seeded repeats on dancing children and laurent hand and the
direction-weight floor, are in [todo.md](todo.md).