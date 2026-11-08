# Loop code from the loop's points

Status: implemented (2026-11-07); first horse result below says the pooled,
axis-free code fits and orients well but clusters worse than the centroid code
at the end of a full run.

## The problem

We want one code per loop that separates it from every other loop. The current
triplane code is sampled at a single point, the loop centroid. Two loops can
have centroids in the same triplane cell, and then their 48-D codes are nearly
the same even though the loops themselves differ. That is bad for the fit --
the network cannot tell the loops apart -- and worse for the later clustering,
where loops that should seed different tubes collapse into one group.

A code read at one point describes the field at that point, not the loop. What
the loop actually is -- its size, its shape, which part of the shape it cuts --
is in its point set. The proposal is to read the field at those points and pool
the samples into one code per loop.

## What the code is now

`TriplaneField.sample(xyz)` maps a world point to the unit cube, bilinearly
reads each of the XY, XZ and YZ grids, and concatenates the three 16-D samples
into a 48-D vector:

```
code(x) = [ T_XY(x) ; T_XZ(x) ; T_YZ(x) ]          # 48-D
```

In a fit, each loop gets one query point, its world centroid:

```
q_i = frame_i + mean(P_i)
c_i = code(q_i)                                    # 48-D, one per loop
```

Here `P_i` is the loop's target point set in its plane's local frame and
`frame_i` is that frame's origin; `rosa_main.py` builds exactly this at
`queries = frame[i] + targets[i].mean(0)`. The network then reads
`[sin z, cos z, c_i, axis_i]` for every sample of the loop, so the entire curve
is conditioned on one vector.

The npz export stores `queries` (the centroids) and `features` (the codes at
them), and `code_clusters.py` clusters those 48-D vectors.

## Evidence that centroid codes alias

Horse, frozen-start run, 139 loops, 127 adjacent same-slot loop pairs. A pair
"shares a cell" when both centroids fall in the same cell of a plane. At
resolution 32, 11 of the 127 adjacent pairs (8.7%) share *all three* cells; at
resolution 64 this falls to 0, but the plane that is invariant along a sweep
still aliases almost as often:

| adjacent pairs sharing a cell | R = 32 | R = 64 |
|---|---:|---:|
| all three planes | 11 / 127 | 0 / 127 |
| x-sweep, YZ (invariant plane) | 13 / 37 | 11 / 37 |
| y-sweep, XZ (invariant plane) | 11 / 28 | 7 / 28 |
| z-sweep, XY (invariant plane) | 18 / 62 | 17 / 62 |
| all 9,591 pairs sharing all three cells | 15 | 0 |

The consequence is in the codes themselves: adjacent same-family loops sit at
cosine distance 0.059 (R=32) and 0.065 (R=64), against 0.75 for random
same-axis pairs. The code is effectively about 10-dimensional (participation
ratio 9.5 of 48 at R=32, 9.9 at R=64), so it has little room to distinguish
loops even before aliasing.

Resolution is not the fix. The learned grid is smooth: 99.8% of the R=32 grid
energy is below 16 cycles/side, and R=64 puts only 2% above that; the extra
cells mostly interpolate the same signal (neighbour-difference / variance
0.24 -> 0.13). The res-64 run costs 4x the grid storage for +0.04 seed-averaged
silhouette at k=12 and no tube-readiness change (54-55/139 ready, longest chain
18). See `todo.md` and `log/axis_frz64_horse/`.

## Proposal

For each loop, sample the field at the loop's own points instead of at its
centroid, then pool:

```
c_i = Pool_{x in P_i} code(frame_i + x)            # one 48-D (or 96-D) code
```

The point set `P_i` is the loop's target slice points -- the observation the
fit is already given -- not the fitted curve. Using fitted points would make
the condition depend on the answer and the fit could ignore it; the targets are
fixed, outside the optimisation, and describe the loop we want.

Concretely, every target point belongs to one cell in each of the three
planes. Take the cells that contain the loop's target points -- the occupied
cells of the loop, not the single cell of its centroid -- sample the field in
those cells, and reduce the samples to one vector. The pooling can run over the
points themselves or over the unique occupied cells (see below); the important
change is that the loop contributes a *set* of cells, not one point.

Pooling makes the code a function of the whole point set. Two loops with the
same centroid cell but different extent, orientation, density or in-plane
offset now read different sets of cells and get different codes. Loops that
are genuinely identical still get identical codes, which is the correct
behaviour: uniqueness means distinguishable geometry, not a unique id.

The pooled code stays 48-D if each of the three plane samples is pooled
separately and then concatenated:

```
c_i = [ mean_{x in P_i} T_XY(x) ;
        mean_{x in P_i} T_XZ(x) ;
        mean_{x in P_i} T_YZ(x) ]
```

This keeps the network interface, the warm-up's first layer and the clustering
input unchanged. A mean+max version doubles it to 96-D and is a later variant,
not the first one.

## Pooling choices

| pool | dim | keeps | loses | notes |
|---|---:|---|---|---|
| mean over points | 48 | average field over the loop | within-loop variation, density-sensitive | simplest drop-in; first experiment |
| mean over occupied cells | 48 | each cell equally | density differences | remove point-density bias; one weight per occupied cell |
| max over points | 48 | the most active cell | average behaviour, outlier-sensitive | good complement to mean |
| mean + max | 96 | average and extremes | -- | needs a wider first layer, new warm-up |
| attention pool | 48 | learned weighting | interpretability, more parameters | risk of collapsing to one cell early |
| mean + centroid | 96 | footprint and location | -- | explicit location term; axis/offset prior by another route |

Recommended order: mean over occupied cells, then mean+max, then attention.
The cell-pooled mean is appealing because it is invariant to how many slice
points happen to fall in a cell, which the cloud density otherwise controls.

## Where it plugs in

- `TriplaneField.sample` does not change; it already takes arbitrary points.
- The trainer needs one point set per loop in world coordinates. `rosa_main`
  already has it: `frame[i] + targets[i]` (targets are in the plane-local
  frame), and the centroids are just their mean. Pass those alongside
  `query_points`.
- `Trainer.feature_values()` becomes the pooling site: for each target index,
  sample `triplane(points_i)` and reduce to one vector. The result keeps the
  shape `n_loops x 48`, so everything downstream -- the network's `feat`
  argument, `feature_pairs`, the seam distillation teacher and the npz export
  -- keeps working.
- During training the grid changes every step, so the pooled code must be
  recomputed each step (or each time the field is read); the point sets and
  their cell indices are fixed and can be precomputed once.
- The npz should store the pooled `features` as now, plus enough to reproduce
  them: the point counts (and optionally the unique-cell index arrays). The
  centroid `queries` can stay as metadata.

Sketch:

```python
# once, in the trainer: world points and occupied-cell ids per loop
pts_world[i] = frame[i] + targets[i]            # S_i x 3
cell_ids[i]  = occupied_cells(pts_world[i], triplane)  # unique (plane, row, col)

# every step, in feature_values()
for i in loops:
    samples = triplane(pts_world[i])                  # S_i x 48
    feats[i] = mean_pool(samples, cell_ids[i])        # 48
```

If full point sets are too slow, subsample K points per loop (e.g. 128-256) or
use one representative point per occupied cell. With 139 loops x 1,426 points
the full version is already only ~0.2M points per step on a 4090.

## What to measure

Run against the centroid baseline at fixed architecture and schedule:

1. **Aliasing:** adjacent same-family code distance, the ratio to random
   same-axis pairs, and the number of pairs whose code cosine similarity is
   above 0.999.
2. **Clustering:** silhouette in the code space, k chosen the same way, axis
   purity, and ARI against the centroid partition.
3. **Downstream:** tube readiness (loops in chains of >= 4, longest chain) with
   the same split rule used so far.
4. **Fit:** final chamfer, reversed loops, seam D and tv, to check that the
   richer condition does not change what the fit converges to.
5. **Cost:** wall-clock per epoch and npz size.

Acceptance is not a higher combined-space silhouette. It is fewer aliased
adjacent pairs and longer/cleaner tube-ready chains at the same fit quality.

## Risks and open questions

- **Conditioning on the answer?** The code is computed from the target points,
  which are also the fit target. This is not a label leak -- the targets are
  the input observation -- but it does make the fit easier. Control it by
  comparing a K-point subsample and a coarse cell-only pool; if either retains
  the clustering gain, the gain is not from fitting the points directly.
- **Position vs shape.** Mean pooling measures the field over the loop's
  footprint, so it mixes location and shape. If clusters become spatial
  patches rather than axis families, weight cells by occupancy (or subtract the
  loop mean before pooling) to make it more shape-sensitive.
- **The invariant third.** Along a pure translation sweep, the YZ projection
  of the point set is unchanged, so the YZ block still cannot say where along
  the sweep the loop sits. Pooling fixes aliasing and shape, not the geometric
  degeneracy of the three-plane construction; axis/offset must still come from
  the cut plane if that block is to carry sweep position.
- **Warm-up.** The cube warm-up also reads one query per loop. Either pool the
  cube loops' points too (the cube points are known) or keep the 48-D mean pool
  and accept a small train/test mismatch; the first layer width is unchanged in
  either case.
- **Density.** Slice point sets are not uniformly dense. Decide once whether
  pooling is per point or per occupied cell; report both on one shape before
  the eight-shape panel.
- **Should normals be pooled too?** A normal-weighted pool would emphasize
  curved parts, but it adds another field read. Defer.

## What was implemented

Pooling by occupied cell, as recommended, and run with the axis input removed:

- `TriplaneField.cell_ids` snaps a world point to its nearest grid node on each
  plane; `cell_pool_structure` takes the unique nodes of every loop's target
  points, pads them to one width and builds the gather masks once. A node counts
  once however many points fall in it, so the mean is per cell, not per point.
- `TriplaneField.pool_cells` averages the grid vectors of those nodes, plane by
  plane, and concatenates the three means. The code stays 48-D and gradients
  still reach the grid.
- `Trainer(feature_pool='occupied_cells', query_sets=...)` builds the structure
  from the loops' world points (`frame[i] + targets[i]`), and
  `feature_values()` pools instead of sampling the centroid. The distillation
  teacher pools the same points carried into its own frame.
- `warm_on_cube(feature_pool=...)` pools the cube loops' points, and
  `warmup_cube.py` grew `--feature-pool` and `--no-axis-condition`.
  `rosa_main.py` grew `--feature-pool`; the npz writes `feature_pool` and
  `cell_counts`.
- The epoch line now prints `pair cos`, the mean cosine distance of the
  neighbour-graph features, so a frozen-start run shows whether the field is
  separating adjacent loops.

## First horse result

Warm-up: pooled, no axis, 300 steps, 32^3 grids of 16 channels
(`warmup/cube_warmup_pool_noaxis.pt`); fit: the frz100 recipe with
`--feature-pool occupied_cells --no-axis-condition`, 1000 epochs, 139 loops.
The 20-epoch diagnostic is `log/pool20_horse/`, the full run
`log/pool_axisoff_frz100_horse/`.

| reading | centroid + axis (`axis_frz100_horse`) | pooled, no axis |
|---|---:|---:|
| final fit | 0.0005 | 0.0004 |
| reversed loops | 8 | 2 |
| silhouette (spectral, k=12) | 0.655 | 0.505 |
| axis purity (same labels) | 0.906 | 0.820 |
| adjacent pairs cos distance < 0.001 | 9 / 126 | 14 / 126 |
| adjacent median cos distance | 0.062 | 0.034 |
| participation ratio | 19.9 / 48 | 14.6 / 48 |
| tube readiness (chains >= 4) | 57 / 139 | 46 / 139 |
| longest ready chain | 18 | 17 |

The pooled code carries enough to fit and orient the loops *better* without an
axis input (fit 0.0004, 2 reversed), so the axis is no longer needed for the
fit. It does not replace the axis for clustering: at the end of the run the
codes are less separable on every reading, and more adjacent pairs are
effectively identical. The 20-epoch state looks better (silhouette 0.559,
purity 0.863) but its curves are still near the warm-up square (fit 0.26, every
loop over the error limit), so that is not a usable partition either. The
neighbour-graph cosine distance fell from 0.23 to 0.019 over the run: the field
fits with ever smaller feature differences and the code-space family structure
is what the fit spends.

Next, if the pooled route is to continue: keep the axis input while pooling (to
separate the effect of pooling from the effect of dropping the axis), or add a
term that holds the neighbour codes apart so the fit cannot collapse them.


## Eight-shape panel

The same pooled, axis-free recipe on the eight-shape panel: frz100 (1000
epochs, `--distill 10 --freeze-f-epochs 100`), `--feature-pool occupied_cells`,
`--no-axis-condition`, warm-up `warmup/cube_warmup_pool_noaxis.pt`. Runs are
`log/pool_axisoff_frz100_<shape>/`; each has a spectral k=12 group sheet
`pool_axisoff_frz100_<shape>_cluster_loops_spectral_groups.png`. The table is
`log/pool_axisoff_eight_summary.md`. Per-plane files are no longer written
(`--save-plane-files` is opt-in); `planes.txt` and `bad_loops.txt` still are.

| shape | loops | silhouette base/pool | purity base/pool | adj. cos base/pool | ready base/pool | longest base/pool | clockwise base/pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| armadillo | 162 | 0.509 / 0.439 | 0.809 / 0.728 | 0.533 / 0.270 | 51 / 38 | 11 / 6 | 1 / 42 |
| bunny | 80 | 0.575 / 0.689 | 0.863 / 0.988 | 0.037 / 0.022 | 56 / 50 | 21 / 17 | 1 / 23 |
| dancing_children100K | 210 | 0.358 / 0.424 | 0.752 / 0.624 | 0.569 / 0.338 | 73 / 73 | 13 / 7 | 10 / 15 |
| horse | 139 | 0.655 / 0.508 | 0.906 / 0.842 | 0.062 / 0.032 | 58 / 47 | 18 / 17 | 5 / 10 |
| laurent_hand | 149 | 0.533 / 0.555 | 0.872 / 0.980 | 0.227 / 0.068 | 67 / 62 | 14 / 14 | 12 / 12 |
| lucy | 157 | 0.457 / 0.409 | 0.917 / 0.707 | 0.126 / 0.044 | 49 / 54 | 13 / 10 | 5 / 24 |
| neptune | 189 | 0.511 / 0.478 | 0.804 / 0.640 | 0.362 / 0.222 | 48 / 67 | 8 / 10 | 4 / 7 |
| rocker-arm | 153 | 0.617 / 0.647 | 0.928 / 0.954 | 0.051 / 0.016 | 132 / 129 | 20 / 19 | 0 / 56 |

Base is `axis_frz100_<shape>` (centroid code + axis), pool is this run. Silhouette
and purity use spectral clustering on cosine affinity at k = 12, recomputed
from each run's npz. "adj. cos" is the median cosine distance of adjacent loops
in the same slot; "ready" counts loops in chains of >= 4 that pass the gap,
Hausdorff and radius-ratio tests. "clockwise" counts loops whose traversal is
clockwise seen from the +axis at the final epoch (summed over x/y/z) -- the
orientation convention the cube warm-up trains and the axis input carries.

What the panel says:

- Fits are fine everywhere (final fit 0.0004-0.0010), so the pooled code is
  enough to place, shape and close the curves without the axis.
- Clustering is a wash, not a gain: silhouette improves on four shapes
  (bunny, dancing, laurent, rocker) and drops on four (armadillo, horse, lucy,
  neptune); axis purity improves only on three and drops badly on lucy
  (0.917 -> 0.707) and neptune (0.804 -> 0.640). No shape-independent reason to
  drop the axis.
- Pooling does not keep neighbours apart. The adjacent median cosine distance
  is *lower* with pooling on all eight shapes, and near-identical pairs
  (distance < 0.001) remain or increase (horse 9 -> 13 of 128, neptune 2 -> 14
  of 178, rocker 2 -> 9 of 145). The code still collapses along a family as the
  fit converges; pooling the footprint changes the code's shape sensitivity
  but not this.
- Orientation is the clear regression. Winding consistency (clockwise count)
  degrades on seven of eight shapes and collapses on rocker-arm (0 -> 56 of
  153), armadillo (1 -> 42 of 162), bunny (1 -> 23 of 80) and lucy (5 -> 24 of
  157). Without the axis the field cannot recover the sign of the plane
  orientation, which is exactly what the axis was added to supply. The saved
  curves are still individually turned to agree with the points' normals
  (17-32 loops per run on armadillo/bunny/lucy/rocker), but a family lofted
  along the sweep needs one traversal direction, not per-loop corrections.
- Tube readiness drops on five shapes, ties on one and improves on two; the
  pooled gains (neptune 48 -> 67, lucy 49 -> 54) come with the large purity and
  winding losses, so they are not a clean win.

Conclusion: keep the axis. Pooling is worth keeping as a code, but the axis is
not replaceable by it -- the next test is pooling *with* the axis input, which
separates the pooling effect from the axis effect, and/or a term that holds
neighbour codes apart. The pooling implementation stays; only the axis-free
configuration is closed.

## Axis restored

Putting the axis back into f's input while keeping the occupied-cell pool
separates the two effects. Warm-up: `warmup/cube_warmup_pool_axis.pt`, trained
on the cube with `--feature-pool occupied_cells` and the axis condition on
(300 steps, fit 0.0045, direction +0.996 cos, seam D 0). Fits: the same frz100
recipe, `--feature-pool occupied_cells --axis-condition`, into
`log/pool_axison_frz100_<shape>/`; group sheets
`pool_axison_frz100_<shape>_cluster_loops_spectral_groups.png`. Three-arm table
`log/pool_axis_three_arm_summary.md`.

Mean over the eight shapes, in the order **centroid+axis / pooled, no axis /
pooled + axis**:

| reading | centroid+axis | pooled, no axis | pooled + axis |
|---|---:|---:|---:|
| silhouette, k=12 | 0.527 | 0.519 | 0.510 |
| axis purity | 0.857 | 0.808 | 0.792 |
| tube-ready fraction | 45.7% | 43.8% | **47.3%** |
| clockwise loops | 2.9% | 16.6% | 5.1% |

Per shape (base / no axis / + axis):

| shape | silhouette | purity | ready | clockwise |
|---|---:|---:|---:|---:|
| armadillo | .509 / .439 / .499 | .809 / .728 / .667 | 51 / 38 / 59 | 1 / 42 / 0 |
| bunny | .575 / .689 / .691 | .863 / .988 / .988 | 56 / 50 / 57 | 1 / 23 / 0 |
| dancing_children100K | .358 / .424 / .397 | .752 / .624 / .681 | 73 / 73 / 82 | 10 / 15 / 15 |
| horse | .655 / .508 / .618 | .906 / .842 / .863 | 58 / 47 / 56 | 5 / 10 / 7 |
| laurent_hand | .533 / .555 / .563 | .872 / .980 / .919 | 67 / 62 / 64 | 12 / 12 / 10 |
| lucy | .457 / .409 / .317 | .917 / .707 / .643 | 49 / 54 / 52 | 5 / 24 / 27 |
| neptune | .511 / .478 / .420 | .804 / .640 / .677 | 48 / 67 / 53 | 4 / 7 / 7 |
| rocker-arm | .617 / .647 / .575 | .928 / .954 / .895 | 132 / 129 / 133 | 0 / 56 / 2 |

What the axis restores:

- Orientation, nearly to the baseline: the clockwise fraction falls from 16.6%
  to 5.1% (baseline 2.9%), and the against-normal alignment returns to the
  baseline value on seven of eight shapes (armadillo 0.173 vs 0.172, bunny 0.089
  vs 0.081, laurent 0.221 vs 0.243, rocker 0.280 vs 0.251). The ambiguity the
  field alone could not resolve was the plane-orientation sign.
- Tube readiness: pooled+axis is the best of the three arms on average and
  beats the centroid baseline on six of eight shapes (armadillo 51 -> 59,
  dancing 73 -> 82, rocker 132 -> 133, lucy 49 -> 52, neptune 48 -> 53,
  bunny 56 -> 57), losing on horse (58 -> 56) and laurent (67 -> 64). The
  gain is mostly more short chains, not a longer one: the longest chain is
  shorter or equal on six shapes (armadillo 11 -> 10, bunny 21 -> 18, dancing
  13 -> 10, horse 18 -> 17, laurent 14 -> 14, rocker 20 -> 20) and longer only
  on lucy (13 -> 17).
- The seam distillation becomes easy: the student's mean D on the teacher's
  pairs falls to 0.0009-0.0029 on six shapes (0.0406 on lucy, 0.0691 on
  neptune), against 0.049-0.121 without the axis. With the direction supplied,
  the fit satisfies the teacher's seam arrangement almost exactly, so the seam
  term is no longer where the difficulty lies.

What it does not restore:

- Clustering. Pooled+axis is the weakest arm on mean silhouette (.510 vs .527
  base, .519 no-axis) and mean purity (.792 vs .857 base, .808 no-axis). It
  beats the base on bunny, dancing and laurent -- the same shapes where pooling
  helped without the axis -- and loses on horse, lucy, neptune and rocker. The
  pooling gains are real but shape-specific; they are not produced by the axis.
- Neighbour separation. Adjacent median cosine distance is still below the
  centroid baseline on seven of eight (bunny .010 vs .037, rocker .018 vs .051,
  horse .036 vs .062, lucy .054 vs .126, laurent .126 vs .227, neptune .261 vs
  .362, dancing .478 vs .569; armadillo is the exception at .326 vs .533). More
  global separation (all-pair cosine rises with the axis) without more local
  separation is the same collapse as before: codes of adjacent loops become
  more similar as the fit converges.
- Lucy. Pooled+axis is a poor fit on lucy in this seed: align 0.680 (baseline
  0.347, no-axis 0.536), final fit 0.0020, 27 of 157 loops clockwise at the end,
  and silhouette .317 vs .457. The teacher D is also large (0.0406 mean, 2.14
  max), so the seam term is not being met. A repeat
  (`log/pool_axison_frz100_lucy_b/`) improves the fit (fit 0.0007, align 0.463)
  but not the clustering (silhouette .273, purity .701) and still leaves
  17/157 clockwise (baseline 5), so the clustering deficit on lucy is stable;
  the fit quality varies between seeds, the orientation and cluster loss do
  not go away.

Conclusion: the axis should stay -- it is what makes orientation and readiness
match or beat the baseline -- but it is not the missing piece for clustering.
Pooling with the axis helps three shapes and hurts four, so the centroid+axis
code is still the safer default. The open problem is local code separation, not
the conditioning: a term that holds neighbour codes apart (or an explicit
shape/offset component in the distance) is the next thing to test.
