# Project state (shared memory)

This is the cross-device entry point for the project's latest progress. It is
tracked in git on purpose: pull the repository and read this file before
starting work. Details stay in the linked documents; this file is the summary
and the reading order.

- Last updated: 2026-11-09
- Branch: `polycube` (`origin` on GitHub), synced at `b01a260`; the overhaul
  landed in `7661a95 overhaul; cleaning`
- Detailed experiment log: [todo.md](todo.md); method: [WORKFLOW.md](WORKFLOW.md)

## What the project is

`NISR_Seawall` is a neural implicit atlas reconstruction pipeline, documented in
[README.md](../README.md) and driven by `main.py`. The loop/orientation work is a
separate, self-contained pipeline documented in
[README_PLC.md](../README_PLC.md). Since the 2026-11-08 overhaul every file it
needs lives under a `plc_*` name:

```
point cloud with normals
  -> cross-section loops            plc_model/loop_extraction.py
  -> one shared MLP fitted per loop plc_model/mini_nn/, plc_main.py
  -> seam / orientation / codes     plc_utils/*, plc_utils/code_clusters.py
  -> documents and this file        plc_md/
```

Only `plc_main.py` stays at the repository root; `main.py`, `model/` and `utils/`
are the unrelated NISR atlas pipeline.

## Layout after the 2026-11-08 overhaul

Commit `7661a95` moved the old flat ROSA files under `plc_*` and split the
2059-line `mini_nn.py` into one class per module. Verified after the move:
`plc_main.py --help` runs, the package compiles, and every forward call site
unpacks the new tuple return.

| before | after |
|---|---|
| `rosa_main.py` | [plc_main.py](../plc_main.py), the only runtime file left at the root |
| `mini_nn.py` | [plc_model/mini_nn/](../plc_model/mini_nn): `network`, `trainer`, `chamfer`, `curves`, `cubes`, `square`, `triplane`, `data`, `warmup` |
| `loop_extraction.py` | [plc_model/loop_extraction.py](../plc_model/loop_extraction.py) |
| `warmup_cube.py` | [plc_model/warmup_cube.py](../plc_model/warmup_cube.py) |
| `code_clusters.py` and root analysis scripts | [plc_utils/](../plc_utils) |
| `README_ROSA.md`, `WORKFLOW.md`, `todo.md`, reports | [plc_md/](.) (this file included) |

The same commit changed the run defaults: random-cube warm-up
`warmup/cube_warmup_random_ranges.pt`, `--feature-pool occupied_cells`, axis off
and `--distill 0`. Axis-on/centroid runs must name the old checkpoint, pooling
and `--axis-condition` explicitly.

Doc drift left by the move was fixed in `b01a260`: `README.md`,
`README_PLC.md` and this file now point at `plc_main.py`, `plc_utils/` and
`plc_md/`. `WORKFLOW.md`, `CUBE_WARMUP_REPORT.md` and `LOOP_ANCHOR_PLAN.md`
still quote old module names and line numbers; treat those as historical
references and update them as each is touched.

## Current state

Committed through `7661a95` unless marked; the only uncommitted work is the
`--cycle` / `--repel` pair described under "In progress" below.

- **Input data.** `normalized_shapes_cuboid/*_cubic_lam3.ply` (89 shapes) is
  `normalized_shapes/*.ply` passed through cubic ARAP stylization (Liu &
  Jacobson 2019), lambda 3, kNN k = 8, 60 iterations. The generator is restored
  at [plc_utils/cubic_stylize_points.py](../plc_utils/cubic_stylize_points.py);
  parameters and per-shape metrics are in
  `normalized_shapes_cuboid/manifest.json`.
- **Warm-up.** The default checkpoint is `warmup/cube_warmup_random_ranges.pt`
  (2026-11-08): 8 random cubes per step, half-sides 0.05-0.45, redrawn every
  step, with `occupied_cells` features and no axis input; 2x64 MLP, triplane
  3x16 grids at 32x32. It fits the cubes to 0.211 mean (0.365 max) with
  direction cos +0.991 and seam `D` median 0.0002 / max 0.0060. It is the
  initialisation of a default run and the default distillation teacher when
  `--distill` is on (`--distill` itself defaults to 0). The earlier
  `warmup/cube_warmup_axis.pt` (one unit cube, axis conditioned, centroid
  features, fit 0.0050, cos +0.997) remains as the matched checkpoint for the
  axis-on/centroid recipe. Caveat: on bunny, armadillo, horse and laurent hand
  the aggregate plane error over all extracted loops is higher than the
  fixed-cube run at 1000 epochs, and laurent hand's bad-loop count reads 55
  against 20. That aggregate includes loops that are not intended fit targets,
  so it does not establish that either warm-up is better; the comparison
  remains open until scored on the intended loops.
- **Triplane conditioning.** 48-D code read from three 16-channel grids mapped
  onto the cloud's box. The default `--feature-pool occupied_cells` averages the
  unique grid nodes a loop's points occupy; `centroid` samples one point at the
  loop centre (the older code). `--feature triplane` is required by the warm-up
  and by distillation.
- **Seam distillation.** `--distill 10` holds the student under the teacher's
  per-pair seam distance `D`; it can lower `D` but never raise it, and it is
  orientation-blind by construction.
- **Frozen start.** `--freeze-f-epochs 100` freezes the shared MLP for the first
  100 epochs so only the triplane grid trains; the field then places each loop
  while `f` keeps the warm-up's map. This is a placement initialiser, not a
  different fit: the rest of the run is the ordinary objective.
- **Code clustering.** [plc_utils/code_clusters.py](../plc_utils/code_clusters.py) clusters the 48-D
  loop codes with cosine average linkage; use `--k 10` for cross-shape
  comparison, `--state` to read a replayed early checkpoint.

## Results snapshot

Eight shapes, 29 planes per axis, 1000 epochs, `--normal 0.01 --distill 10`;
controls are the same runs without the freeze. These are the **axis-on,
centroid** runs from before the 2026-11-08 default switch; the
`occupied_cells`/axis-off panel has not been run on all eight shapes yet (open
item 4); the four-shape aggregate did not rank the warm-ups.

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

## In progress on the authoring device (not pushed as of 2026-11-09)

The working tree adds two optional loss terms, both defaulting to 0, so
committed runs are unchanged by the code. Neither has a run or a `todo.md` entry
yet:

- `--cycle W`: the network now also returns `z_pred`, a reading of the
  parameter back from a point on the curve and its condition. The term is the
  mean squared error against `z`; the curve branch is detached inside `forward`,
  so it trains the inverse map and the condition, not the fit. A curve that
  folds two distant parameters onto one point cannot be inverted.
- `--repel W --repel-gap F --repel-samples K`: anchors spread around the loop
  are charged when a non-neighbour pair lands closer than `F` times the target's
  own RMS radius. Unlike `cycle` this acts on the points, so the fit feels it.

Supporting changes: `miniNN.forward` returns `(curve, z_pred)`, so every caller
unpacks `[0]`; checkpoints are loaded with `strict=False` so pre-inverse
checkpoints still load. The touched runtime files are `plc_main.py` and
`plc_model/mini_nn/{network,trainer,warmup}.py`; the `plc_utils` plotting and
probe scripts were updated for the tuple return. The package compiles and
`plc_main.py --help` lists the new flags.

Files touched: `plc_main.py`, `plc_model/mini_nn/network.py`,
`plc_model/mini_nn/trainer.py`, `plc_model/mini_nn/warmup.py`,
`plc_utils/plot_axis_epochs.py`, `plc_utils/plot_epoch_matching.py`,
`plc_utils/plot_first_iteration.py`, `plc_utils/plot_loop_matching.py`,
`plc_utils/reclassify_run.py`, `plc_utils/seam_check.py`,
`plc_utils/warmup_normal_probe.py`. Untracked and unrelated to PLC:
`model/parameterization.py`.
Run `git status` and `git log` before relying on these; they are not in the
pushed state described above.

## Commands

```bash
# environment (plain python fails on libcudnn.so.8)
conda run -n seawall python plc_main.py --help

# fit one shape, triplane + seam distillation
conda run -n seawall python plc_main.py \
    --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane --epochs 1000 \
    --normal 0.01 --triplane-tv 0.01 --snapshot-every 0 \
    --distill 10 --out log/axis_seam10_horse

# frozen start
conda run -n seawall python plc_main.py \
    --path normalized_shapes_cuboid/horse_points_cubic_lam3.ply \
    --samples 29 --train --feature triplane --epochs 1000 \
    --normal 0.01 --triplane-tv 0.01 --snapshot-every 0 \
    --distill 10 --freeze-f-epochs 100 --out log/axis_frz100_horse

# cluster a finished run's loop codes
conda run -n seawall python plc_utils/code_clusters.py \
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
  `normalized_shapes_cuboid/` PLYs, and the warm-up checkpoints under
  `warmup/` (`cube_warmup_random_ranges.pt` is the default; the axis-checkpoint
  pair is needed for the old recipe). `*.ply`,
  `*.pt` and `log/` are gitignored, so a new device must copy the data, the
  warm-up checkpoint and any run folders it needs separately. The warm-up can be
  regenerated with `plc_model/warmup_cube.py`, but only such a checkpoint
  reproduces the warm-up exactly. Each run's outputs are listed in
  [README_PLC.md](../README_PLC.md) and [WORKFLOW.md](WORKFLOW.md) §2.7.
- The full reading order for a new device: this file,
  `../README_PLC.md`, `WORKFLOW.md` §2.3-2.4 and §6, then the tail of `todo.md`.

## Open items

1. Seeded repeats on dancing children and laurent hand (the two shapes where the
   frozen start does not cleanly win) with the nondeterminism floor stated.
2. Direction-weight floor at the end of the run, where the surviving late
   turn-overs happen; test it against the same panel.
3. Field roughness: the frozen start costs 3-8x in tv. Try a tv anneal, or
   release `f` with the direction term held at a floor, at matched fit.
4. Re-evaluate the new defaults (`occupied_cells`, axis off, random-cube
   warm-up) on the eight-shape panel against matched fixed-cube/centroid/axis
   controls. The first four-shape comparison reported higher aggregate error
   and more bad loops for the random-cube arms, but that aggregate covers every
   extracted loop, including loops not intended as fit targets, so it does not
   rank the warm-ups; score the panel on the intended loop set.
5. Finish the `--cycle` / `--repel` work: one short fit per term to prove the
   terms are wired (loss printed, fit not degraded), then a matched A/B on
   shapes that fold or turn over, and a `todo.md` entry with the command and
   the numbers. Decide whether either default should move off 0.

## How to update this file

After each experiment, in the same commit as the results:

1. Append the command, table, figures, caveats and next tests to
   [todo.md](todo.md) (do not replace older entries).
2. Update the date, branch/commit and the results snapshot here; keep this file
   short and move detail into `todo.md` / `WORKFLOW.md`.
3. Commit and push, so the other device gets the same state on `git pull`.
