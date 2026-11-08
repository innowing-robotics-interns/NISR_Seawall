# Project state (shared memory)

This is the cross-device entry point for the project's latest progress. It is
tracked in git on purpose: pull the repository and read this file before
starting work. Details stay in the linked documents; this file is the summary
and the reading order.

- Last updated: 2026-11-08
- Branch: `polycube` (`origin` on GitHub); last code commit `a0b778f`, this
  file added on top (check `git log -1` for the current tip)
- Detailed experiment log: [todo.md](todo.md); method: [WORKFLOW.md](WORKFLOW.md)

## What the project is

`NISR_Seawall` is a neural implicit atlas reconstruction pipeline, documented in
[README.md](README.md) and driven by `main.py`. The loop/orientation work is a
separate, self-contained pipeline documented in [README_ROSA.md](README_ROSA.md):

```
point cloud with normals
  -> cross-section loops            loop_extraction.py
  -> one shared MLP fitted per loop mini_nn.py, rosa_main.py
  -> seam / orientation / codes     plc_utils/*, code_clusters.py
```

## Current committed state

- **Input data.** `normalized_shapes_cuboid/*_cubic_lam3.ply` (89 shapes) is
  `normalized_shapes/*.ply` passed through cubic ARAP stylization (Liu &
  Jacobson 2019), lambda 3, kNN k = 8, 60 iterations. The generator is restored
  at [plc_utils/cubic_stylize_points.py](plc_utils/cubic_stylize_points.py);
  parameters and per-shape metrics are in
  `normalized_shapes_cuboid/manifest.json`.
- **Cube warm-up.** `warmup/cube_warmup_axis.pt` is a unit cube (side 1), axis
  conditioned, 2x64 MLP, triplane 3x16 grids at 32x32. It fits the cube's 12
  loops to about 0.0050 with direction cos about +0.997, and is both the
  initialisation and the distillation teacher.
- **Triplane conditioning.** 48-D code read from three 16-channel grids mapped
  onto the cloud's box; `--feature triplane` is required by the warm-up and by
  distillation.
- **Seam distillation.** `--distill 10` holds the student under the teacher's
  per-pair seam distance `D`; it can lower `D` but never raise it, and it is
  orientation-blind by construction.
- **Frozen start.** `--freeze-f-epochs 100` freezes the shared MLP for the first
  100 epochs so only the triplane grid trains; the field then places each loop
  while `f` keeps the warm-up's map. This is a placement initialiser, not a
  different fit: the rest of the run is the ordinary objective.
- **Code clustering.** [code_clusters.py](code_clusters.py) clusters the 48-D
  loop codes with cosine average linkage; use `--k 10` for cross-shape
  comparison, `--state` to read a replayed early checkpoint.

## Results snapshot

Eight shapes, 29 planes per axis, 1000 epochs, `--normal 0.01 --distill 10`;
controls are the same runs without the freeze.

- Frozen start cuts loops whose own-centre winding is reversed from **207 to 47
  of 1239** (4.4x). Seven of eight shapes improve. Exceptions: dancing children
  4 -> 11, and laurent hand loses y (22 -> 0) but keeps z (15 -> 14).
- Fit is recovered by epoch 999; only laurent hand and bunny differ from the
  control by more than 20%. The freeze costs fit while it lasts (e99 reads
  0.022-0.070 against 0.002-0.0035).
- Cost of the freeze: the field is **3-8x rougher** in the tv term.
- Clustering: the frozen arm separates codes better on every shape (silhouette
  +0.03..+0.13 at k = 10); the epoch-100 partition is already 0.64-0.93 ARI from
  the final one, and e100 silhouette is above final on seven shapes (laurent the
  exception). Clusters are the axis families.
- Horse reference cost: 139 loops, 1000 epochs, 215.5 s (27 s extraction) and
  2.7 GB peak RSS.
- Training is not bit-reproducible (about 4e-5 in weights); discrete winding
  counts move run to run, so read direction and magnitude, not exact counts.

## In progress on the authoring device (not pushed as of 2026-11-08)

The working tree has uncommitted changes (about 805 inserted lines) adding an
`occupied_cells` footprint read of the triplane field and related flags:

- `--feature-pool {centroid,occupied_cells}`: pool the grid nodes a loop covers
  instead of one bilinear sample at its centroid, so adjacent loops in the same
  cell get different codes.
- `--match-angle`: gate chamfer pairs by the angle between the implied and
  target normals.
- `--save-plane-files`: write per-plane artefacts only when requested.

Files touched: `mini_nn.py`, `rosa_main.py`, `loop_extraction.py`,
`plc_utils/loop_propagation.py`, `warmup_cube.py`, `todo.md`, `WORKFLOW.md`.
Run `git status` and `git log` before relying on these; they are not in the
pushed state described above.

## Commands

```bash
# environment (plain python fails on libcudnn.so.8)
conda run -n seawall python rosa_main.py --help

# fit one shape, triplane + seam distillation
conda run -n seawall python rosa_main.py \
    --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane --epochs 1000 \
    --normal 0.01 --triplane-tv 0.01 --snapshot-every 0 \
    --distill 10 --out log/axis_seam10_horse

# frozen start
conda run -n seawall python rosa_main.py \
    --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane --epochs 1000 \
    --normal 0.01 --triplane-tv 0.01 --snapshot-every 0 \
    --distill 10 --freeze-f-epochs 100 --out log/axis_frz100_horse

# cluster a finished run's loop codes
conda run -n seawall python code_clusters.py \
    log/axis_frz100_horse/horse_points_cubic_lam3_triplane.npz --k 10

# rebuild a cubic target from a normalized cloud
conda run -n seawall python plc_utils/cubic_stylize_points.py \
    --points normalized_shapes/horse_points.ply \
    --out logs/cubic_fit/targets/horse_points \
    --lam 3 --k 8 --iters 60
```

## What is shared and what is not

- Shared through git: source, docs and this state file.
- **Not shared**: `log/` run folders, `normalized_shapes/` and
  `normalized_shapes_cuboid/` PLYs, and `warmup/cube_warmup_axis.pt`. `*.ply`,
  `*.pt` and `log/` are gitignored, so a new device must copy the data, the
  warm-up checkpoint and any run folders it needs separately. The warm-up can be
  regenerated with `warmup_cube.py`, but only such a checkpoint reproduces the
  warm-up exactly. Each run's outputs are listed in
  [README_ROSA.md](README_ROSA.md) and [WORKFLOW.md](WORKFLOW.md) §2.7.
- The full reading order for a new device: this file, `README_ROSA.md`,
  `WORKFLOW.md` §2.3-2.4 and §6, then the tail of `todo.md`.

## Open items

1. Seeded repeats on dancing children and laurent hand (the two shapes where the
   frozen start does not cleanly win) with the nondeterminism floor stated.
2. Direction-weight floor at the end of the run, where the surviving late
   turn-overs happen; test it against the same panel.
3. Field roughness: the frozen start costs 3-8x in tv. Try a tv anneal, or
   release `f` with the direction term held at a floor, at matched fit.
4. Evaluate the in-progress `occupied_cells` pooling on the same eight-shape
   panel with matched controls.

## How to update this file

After each experiment, in the same commit as the results:

1. Append the command, table, figures, caveats and next tests to
   [todo.md](todo.md) (do not replace older entries).
2. Update the date, branch/commit and the results snapshot here; keep this file
   short and move detail into `todo.md` / `WORKFLOW.md`.
3. Commit and push, so the other device gets the same state on `git pull`.
