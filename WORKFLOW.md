# Workflow: point cloud -> loops -> fitted curves -> tracks -> tubes

This is the current implementation as of November 2026. It replaces the older
`CHANGELOG.md` / `TRAINING.md` descriptions, which no longer match the code.
Design notes for individual stages live in [loop_extraction.md](loop_extraction.md),
[loop_propagation.md](loop_propagation.md) and
[triplane_loop_clustering.md](triplane_loop_clustering.md); this file is the
run-to-run map. [LOOP_ANCHOR_PLAN.md](LOOP_ANCHOR_PLAN.md) is a proposal for
anchoring the `z = 0` seam and is **not implemented**.

The pipeline has four implemented stages and one planned stage:

```
point cloud (.ply with normals)
        |
        v
[1] extraction      loop_extraction.py        planes, slices, loop grouping
        |
        v
[2] fitting         mini_nn.py + rosa_main.py shared MLP, one curve per loop
        |
        v
    <run>/<stem>_model.pt                     weights + replay metadata
        |                    \
        |                     v
        |            [3] replay probe         loop_shift_probe.py
        v
[4] propagation     loop_propagation.py       steps 1 -> 2 -> [2b] -> 3 -> 4
        |
        v
[5] polycube        not implemented           see todo.md
```

Stages 1 and 2 are one command (`rosa_main.py`). Stages 3 and 4 are separate
scripts that read the run folder.

---

## 0. Environment and input

Use the `seawall` conda environment. Plain `python3` fails to import torch with
`libcudnn.so.8` on this machine.

```bash
conda run -n seawall python rosa_main.py --help
```

Inputs live under `normalized_shapes_cuboid/` and must carry point normals
(`nx`, `ny`, `nz`). Current shapes:

| name | file |
|---|---|
| dancing children | `normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply` |
| hand | `normalized_shapes_cuboid/laurent_hand_normal_cubic_lam3.ply` |
| bunny | `normalized_shapes_cuboid/bunny_points_cubic_lam3.ply` |
| horse | `normalized_shapes_cuboid/horse_points_cubic_lam3.ply` |
| armadillo | `normalized_shapes_cuboid/armadillo_normal_cubic_lam3.ply` |
| lucy | `normalized_shapes_cuboid/lucy_points_cubic_lam3.ply` |

The run folder defaults to `log/<input stem>/`; `--out` names another folder so a
second experiment on the same shape sits beside the first.

---

## 1. Extraction (`rosa_main.py` without `--train`)

```bash
conda run -n seawall python rosa_main.py \
    --path normalized_shapes_cuboid/bunny_points_cubic_lam3.ply \
    --samples 15 --field
```

`LoopExtractor.extract()` runs the chain in order:

1. **load** the cloud and its normals;
2. **make planes** per axis, inset from the bounding-box faces by `--margin`
   (default `0.1` of the axis extent), `--samples` planes per axis;
3. **cut** a slab around each plane, half-width `--band` (default `0.0075` of the
   longest extent), and leave out points whose normal is within `--max-align`
   (default `0.95`, about 18 degrees) of the plane normal, since a surface lying
   flat in the slice has no usable in-plane direction;
4. **find centres** from the Poisson / implicit field of each slice;
5. **group loops**: each slice's points are split into one or more closed loops,
   with an implicit-field fallback when point grouping is ambiguous;
6. **save** the slices.

Outputs of this stage:

| file | content |
|---|---|
| `<stem>_slices.ply` | every sliced point, coloured by slice |
| `<stem>_slices.png` | one projected panel per plane (unless `--no-image`) |
| `<stem>_field.png` | implicit field per panel (only with `--field`) |

Useful flags: `--arrows`, `--normal-arrows`, `--assign`.

Extraction is purely geometric: `LoopExtractor` never reads the triplane field.
The field is built at fit time and conditions the shared MLP; after fitting it
also serves as a loop similarity for ranking candidate tracks (§4, "Triplane
feature clustering"). It is not part of plane placement or loop grouping.

---

## 2. Fitting (`rosa_main.py --train`)

The fitting runs used for the shift experiments (they predate the TV default,
so `--triplane-tv 0` is explicit):

```bash
conda run -n seawall python rosa_main.py --train \
    --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --samples 29 --epochs 1000 \
    --feature triplane --code-smooth 0.01 --triplane-tv 0 --normal 0.01

# same experiment with the deeper/narrower network
conda run -n seawall python rosa_main.py --train \
    --path normalized_shapes_cuboid/bunny_points_cubic_lam3.ply \
    --samples 29 --epochs 1000 \
    --feature triplane --code-smooth 0.01 --triplane-tv 0 --normal 0.01 \
    --layers 4 --neurons 32
```

The long adaptive runs use refinement instead of a fixed dense plane set:

```bash
conda run -n seawall python rosa_main.py --train \
    --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --samples 5 --add-planes 2 --samples-end 15 --refine-every 100 \
    --epochs 1000 --normal 0.01
```

The same run with in-training seam phases: identical fit, and the exported
loops start from a seam their neighbours agree on (see §2.3):

```bash
conda run -n seawall python rosa_main.py --train \
    --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --samples 29 --epochs 1000 --normal 0.01 \
    --seam-align 1 --seam-every 10 --seam-steps 100
```

### 2.1 What conditions the shared network

`--feature` selects how a loop is identified to the one MLP that draws every
curve.

| mode | flag | condition vector | behaviour |
|---|---|---|---|
| embedding | `--feature embedding` | one learned row per loop, keyed by `(axis, offset, slot)`, width `feat_dim` = 32 | rows are independent; neighbouring loops only become similar if `--code-smooth` pulls them |
| triplane (default) | `--feature triplane` | 48 numbers sampled from three `R x R` grids | the feature is a continuous function of world position, so neighbouring loops start similar by construction |

The triplane field (`mini_nn.TriplaneField`) holds three orthogonal grids:

- XY, XZ, YZ, each `--triplane-channels` (default 16) channels and
  `--triplane-res` (default 32) cells per side;
- the grids span the cloud bounding box plus a 2% margin, so a loop centre on the
  bounding face still falls inside the field;
- a query is the loop's world centroid; each plane is bilinearly sampled with
  `padding_mode='border'` and the three vectors are concatenated to 48-D.

`--triplane-tv` adds a total-variation penalty on the grids (default `0.01`,
so it is on for every triplane run; pass `--triplane-tv 0` to leave the grids
free).
The grids are exported to `<stem>_triplane.npz` (`grid`, `lo`, `hi`, `queries`,
`features`, `keys`) so a later script can inspect the field without retraining.

The current default arm is therefore triplane + TV `0.01`. The conditioning
comparison in [run_conditioning_compare.sh](run_conditioning_compare.sh) runs
three arms on four shapes at 29 planes, 1000 epochs and a 4x32 net:
`emb_cs01` (embedding + code smoothness), `tri_cs01` (triplane + feature
smoothness) and `tri_tv01` (triplane + grid TV, the default). Outputs are
`log/cmp_<arm>_<shape>/` with logs beside them.

### 2.2 The network

`miniNN` is one MLP shared by all loops. The loop parameter `z` is first
encoded as `[sin(2 pi z), cos(2 pi z)]` (the raw scalar is dropped; the two ends
of `[0, 1)` then map to the same input so the loop closes without a seam). In
triplane mode the input is

```
[ sin(2 pi z), cos(2 pi z), feature (48) ]
```

and it outputs a 3-D point in the loop's plane-local frame. In embedding mode
the feature is the loop's row instead of the triplane sample.

Architecture flags:

| flag | default | effect |
|---|---|---|
| `--neurons` | 64 | width of every hidden layer |
| `--layers` | 2 | number of hidden layers |

So the baseline is 2 hidden layers x 64 neurons; `--layers 4 --neurons 32` is
the deeper/narrower variant (about 4.9k weights versus about 7.6k). Hidden
activations are `Softplus(beta=200)`.

The loop parameter `z` is sampled on `[0, 1)` at `M` points, where `M` defaults
to the median number of points in a loop. The fitted PLY stores `M` world points
per loop.

### 2.3 Loss terms

The total loss per batch is:

```text
chamfer  +  smooth * bend  +  normal_weight(t) * normal  +  code_smooth * code  +  triplane_tv * tv  +  distill * seam_distill
```

| term | flag | formula | role |
|---|---|---|---|
| Chamfer | always | symmetric squared Chamfer between the predicted curve and the loop points | position |
| bend | `--smooth` | mean squared second derivative over mean squared speed | keeps the curve smooth; a circle reads 39.5 |
| normal | `--normal` | mean `1 - cos(tangent x plane_normal, target normal projected in plane)` | fixes which way round the loop runs; 0 aligned, 1 square, 2 reversed |
| code / feature smoothness | `--code-smooth` | weighted mean squared distance between neighbouring loops' codes or sampled features | makes adjacent cross-sections vary continuously |
| triplane TV | `--triplane-tv` | mean squared difference of neighbouring grid cells | regularises the field; on by default (`0.01`) in triplane mode |
| seam distillation | `--distill` | weighted mean over neighbour pairs of `scale * max(D_student - D_teacher, 0)` | holds the seam distance under the cube warm-up's per-pair ceiling; can lower `D`, never raise it; off by default |

`rosa_main.py --normal` is the **normal-orientation** loss. It is unrelated to
the loop-parameter seam alignment in `loop_propagation.py` step 3.

The normal term is annealed inside every `Trainer.train()` call: its weight is
`--normal` on the first batch and falls to zero by half a cosine on the last. The
printed `align` number is the unweighted term, so it stays comparable across
epochs; the printed `(w ...)` is the weight actually applied. Runs in this repo
use `--normal 0.01`, and their logs end at `w 0.0000`.

The warm-up now settles the sense before the fit (see §2.4), but the fine-tune
can still flip loops while the cube shrinks onto the loop: on the horse at
`--normal 0.01` two identical runs ended with 7 and 12 of 139 reversed. Holding
the fine-tune term at `--normal 0.3` ends with 0/139 on two seeds at the same fit
loss (TV 0.027–0.031 against 0.019–0.021), while `--normal 1.0` also reaches zero
but roughly doubles the fit error. The default stays 0.01; use 0.3 when zero
reversals matter more than the last of the field's smoothness.

`--code-smooth` connects mutual-nearest loops on adjacent planes, gated by
`--code-gate` (in-plane separation in plane spacings) and `--code-min-radius`
(ratio of smaller to larger loop radius). In triplane mode the term acts on the
sampled features, so gradients reach the grid.

**Seam phase** (`--seam-align`, off by default) is a solve, not a loss term.
Where `z = 0` falls on a closed loop is a free gauge: Chamfer, bend and the
normal term are all invariant to `f(z) -> f(z + p_i)`, so nothing in the fit
can place it. With `--seam-align 1` the trainer carries one phase per loop
(`Trainer.phase`, a parameter the fit's own optimiser never sees) and solves it
every `--seam-every` epochs with `--seam-steps` Adam steps at 2e-2 turns. Each
solve reads every curve once, detached, and samples it periodically at
`z + p_i`; only the phases receive a gradient, so the printed fit, bend and
`align` numbers are the run's own — checked against `--seam-align 0`: 30 epoch
lines identical to four decimals, weights within the `3.7e-5` run-to-run noise
of two identical runs, and the point set of every curve unchanged (max nearest
distance `0.0014`, the resampling error of a `0.6`-radius loop).

Pairs are the `--code-smooth` neighbour graph. Each pair is first matched at its
best circular shift, and votes `lam / (lam + D_best)` (`--seam-lam`, default
`0.02`) — two loops of one cross-section vote nearly in full, two different
loops barely at all — while the energy itself is the seam distance `D`, so a
badly placed seam is pulled hard instead of sinking into a saturating switch.
The zero-mean prior on the phases fixes the gauge the pairs cannot. On the
trained 29-plane bunny checkpoint (51 pairs) the solve moves every pair to its
best achievable shift, none worse, and the 90th-percentile `D` falls
`0.979 -> 0.0385`; on a 5-plane, 30-epoch bunny run, `0.78 -> 0.196` with 10 of
10 pairs improved. The residual is the shape disagreement of the two loops,
which a phase cannot change. Phases are written to the checkpoint as `phases`
and applied by `Trainer.predict`, so figures, exported loops and replay all
start from the aligned seam; `loop_shift_probe.py` adds them back when it
replays a `t = 0` curve.

The printed `D median a -> b` is the solve's plateau for the curves as they
stood at that epoch. The checkpoint is written after the last fit epoch, which
can move the curves again without a new solve: on the 13-loop smoke run the
last solve printed `0.0417`, while the exported checkpoint measures `0.1962`
with [seam_check.py](seam_check.py). Quote the checkpoint number when comparing
runs.

### 2.4 The cube warm-up (`--warmup`)

Before the first fit, `rosa_main` loads a cube-warm-up checkpoint
(`warmup/cube_warmup.pt` by default) into the network and the triplane field.
The checkpoint comes from its own training script, `warmup_cube.py`, which fits
a network on a unit cube and nothing else; [WARMUP.md](WARMUP.md) is the
specification of that task. The checkpoint carries the shared MLP weights and
the triplane grid, and nothing else — no per-loop rows, no phases, no plane set.
The shape's extraction, planes, loops, targets and phases are all its own, and
the fit keeps its own triplane frame: the grid is a function of unit-cube
position and the frame is where that cube is laid on the cloud's box.

`--no-warmup` fits from a blank network. `--feature embedding` prints a note and
skips the load: embedding rows name cube planes and have no meaning on a shape's
plane set. The checkpoint's `--neurons`, `--layers`, `--triplane-res` and
`--triplane-channels` must match the run's; a mismatch is refused rather than
loaded in part.

The warm-up only applies once, before the first fit; planes added by refinement
are seeded as usual (or, in triplane mode, have no row to seed).

Measured on bunny, 5 planes, 13 loops, 2×64 net, 300 epochs, `--normal 0.01`:

| run | median plane error | turned round | D median / mean / p90 / max |
|---|---:|---:|---|
| cube warm-up | 0.0161 | 5 / 13 | 0.0716 / 0.1204 / 0.2727 / 0.2975 |
| blank network | 0.0195 | 10 / 13 | 0.0947 / 0.1311 / 0.2417 / 0.3804 |

Runs: [log/fit300n_warm](log/fit300n_warm), [log/fit300n_blank](log/fit300n_blank).
`D` is the distance the seam solve minimises; [seam_check.py](seam_check.py)
measures it on a checkpoint at `z = 0` (or at its own phases), and `--against`
compares per pair with another run on the same plane set. The fit's own losses
are seam-invariant, so it can still slide the seam after the warm-up; holding it
is `--seam-align`'s job (§2.3).

**Seam distillation** (`--distill W`, off by default) is the follow-up that
holds `D` through the fit. The warm-up checkpoint (`--distill-warmup`, defaulting
to `--warmup`) is loaded again as a frozen teacher; its field lives on the unit
cube, so each shape query is carried into the teacher's own frame before
sampling. The teacher is read once at this shape's queries and axes, and its
per-pair `D` is the **ceiling** the student's `D` is held under: the loss charges
`W * scale * max(D_s - D_t, 0)`, so it can lower `D` but never raise it. Only
`D` is matched, never the curves, so a common deformation is free and shape and
orientation stay with the fit; `D` is also invariant to orientation, so the term
has no direction of its own and any change in the against-normal count is the
fit's, not a correction the term makes. `D` is scale-free, so each side reads
**its own** curve size — the teacher draws at the warm-up's cube scale, the
student at the shape's — and the pair's fixed target radius only carries the
dimensionless excess into chamfer units. Using one denominator for both is
wrong: on armadillo the teacher's curves are cube-scale while some shape loops
are tiny, and a shape denominator inflated the teacher profile from
`D ≈ 0.0011` to `mean 0.32 / max 13`, enough to pull those seams the wrong way.
At step 0 student and teacher are the same network, so both `D` values are equal
and the term is exactly zero. It needs `--feature triplane`.

At `W = 10`, 1000 epochs, on six shapes (matched arms, same loops and pairs;
each cell baseline → distillation):

| shape | pairs | fit | against normals | D med | D mean | D p90 | D max | teacher D mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bunny | 61 | 0.0007 → 0.0010 | 0 → 0 | 0.000845 → 0.000042 | 0.009591 → 0.000168 | 0.021680 → 0.000519 | 0.170108 → 0.001383 | 0.000192 / 0.002480 |
| armadillo | 108 | 0.0007 → 0.0013 | 4 → 1 | 0.024032 → 0.000518 | 0.108036 → 0.002483 | 0.271146 → 0.005144 | 1.275502 → 0.032433 | 0.001146 / 0.022824 |
| horse | 99 | 0.0003 → 0.0005 | 11 → 3 | 0.011446 → 0.000507 | 0.022065 → 0.001094 | 0.061962 → 0.002535 | 0.151364 → 0.008623 | 0.000643 / 0.007110 |
| lucy | 83 | 0.0004 → 0.0007 | 13 → 14 | 0.056359 → 0.000470 | 0.289455 → 0.025304 | 0.883988 → 0.011167 | 2.798740 → 1.480760 | 0.036220 / 2.141922 |
| laurent hand | 90 | 0.0004 → 0.0006 | 0 → 0 | 0.004851 → 0.000214 | 0.067148 → 0.000826 | 0.074943 → 0.002397 | 3.228513 → 0.011227 | 0.000237 / 0.004099 |
| rocker arm | 119 | 0.0005 → 0.0007 | 1 → 0 | 0.001296 → 0.000136 | 0.045203 → 0.000734 | 0.171421 → 0.001625 | 0.650649 → 0.014472 | 0.000749 / 0.016231 |

The median falls 9.5–120×, the mean 11–81×, the p90 24–105×, the max 18–288×
(lucy 1.9×); fit costs at most `0.0006` absolute, and no shape changes its loop
count or pair graph. "Against normals" is the per-loop direction vote (mean
cosine below `-0.1`) taken before the export flip; `D` is orientation-blind, so
the shifts are the fit's own outcome, not a correction the term makes —
armadillo and horse improve, bunny and laurent stay at zero, rocker arm goes
`1 → 0`, lucy gains one. The mean direction penalty moves with it (`1 - cos`:
bunny `0.104 → 0.103`, armadillo `0.314 → 0.265`, horse `0.377 → 0.321`,
lucy `0.462 → 0.531`, laurent `0.266 → 0.345`, rocker arm `0.400 → 0.470`),
again the fit's outcome rather than a target of the term. (The summary line
used to hard-code "0 aligned and 2 reversed" next to this vote; it now prints
the measured count.) Two tails are kept visible rather than smoothed: lucy's
high mean/max come from a pair that the teacher itself cannot align (teacher
max `2.14`), so the hinge stops at `1.48` rather than pull the loop off its
points — a junction candidate; and on laurent hand 5 more loops cross the
bad-fit threshold (rocker arm gains 3, `0 → 3`). The student does not always
reach the ceiling (laurent D mean `0.000826` vs teacher `0.000237`), so the
end-of-run `distill` residual (`4e-6`–`1.4e-5`) is real, not noise. The ceiling
is approached, not attained: at `W = 10` 45/61 bunny, 92/108 armadillo, 71/99
horse, 54/83 lucy, 77/90 laurent and 83/119 rocker arm pairs still sit above
their teacher `D`, by a median of `1.5e-5` to `2.0e-4`; only the worst pair in
each shape pays a visible excess.

Rocker arm is the sixth shape and a second instance of a whole-family
reversal. The per-loop cosine vote reads `1 → 0` of 153, but the signed
winding (`curve_winding`, centre at the loop's target centroid) says the y
family is mostly clockwise: 33/57 → 41/57 of its loops, mean winding
`0.000 → -0.439`, while x stays `1/59 → 0/59` and z `21/37 → 21/37`. The y
loops are also folded — median foldover `1.569 → 1.281` — so their forward
and backward stretches pair with the same target points and the local cosine
averages to about zero; that is why the vote cannot see the family. `D` is
orientation-blind, so the term did not cause this and cannot remove it; the
change is the fit's own outcome, and it reproduces the horse y-family
reversal. Read the winding next to the cosine vote, not instead of it.

Runs:
[log/axis_tri_tv01_bunny](log/axis_tri_tv01_bunny) vs
[log/axis_seam10_bunny](log/axis_seam10_bunny), and the same pair of folders for
`armadillo`, `horse`, `lucy`, `laurent_hand`, `rocker-arm`.

### 2.5 Schedule and refinement

- **Cube warm-up** is on by default via `--warmup warmup/cube_warmup.pt`, a
  separate artifact trainable with `python warmup_cube.py` (WARMUP.md).
  `--no-warmup` fits from a blank network. The warm-up is tied to the network
  and grid size, not to the shape.
- **Learning rate** is cosine-annealed across each `train()` call; the optimiser
  rate is reset at the start of every stage.
- **Refinement**: with `--refine-every N`, every `N` epochs the widest gaps on
  each axis are bisected, the cloud is cut again, and the fit is carried over.
  `--add-planes` sets how many planes are added per axis per stage;
  `--samples-end` caps the final density. Planes already cut keep their offsets
  and their loops, so no loop is re-matched by position.
  In embedding mode new rows are seeded by `--code-init blend` (default) or
  `random`; in triplane mode there is no row to seed.
- **Triplane** is a feature map on the unit cube: three `--triplane-res` grids,
  uniformly subdivided, the same cube for every shape. The cube is laid over the
  cloud's bounding box by an affine frame (`centre`, `half`) and stored with the
  grid, so a query in world coordinates is carried into it by
  `(x - centre) / half`. Stretching that frame is a change of coordinates on a
  feature map, not a different field; nothing about the triplane changes when the
  shape's box is a different size.
- **Snapshots**: `--snapshot-every 50` writes the full fit grid into
  `<out>/snapshots/` every 50 epochs, named by absolute epoch.

### 2.6 Checkpoint and replay metadata

At the end of a trained run, `mini_nn.save_model` writes
`<out>/<stem>_model.pt` containing:

- the model state dict (network, triplane field or embedding table);
- architecture metadata: `num_neurons`, `num_layers`, `pe`, `embedding_dim`,
  `feature_mode`, `latent_points`;
- triplane configuration and bounds, when used;
- `phases`: the seam phase of every loop in checkpoint order, or `None` when
  the run did not use `--seam-align`.
  A replay must add it to `z` to draw the same starting point the exported
  loops do;
- replay metadata: `keys` (`axis, offset, slot` per loop), `plane_normal`,
  `plane_point`, the loop centroid `queries`, and `samples`.

`mini_nn.load_model` reconstructs the network from that metadata. Old
checkpoints without `num_layers` load as the 2-layer default. This checkpoint
is what makes `loop_shift_probe.py` possible without retraining.

### 2.7 Fitting outputs

| file | content |
|---|---|
| `<stem>_good_loops.ply` | loops at or under the `0.15` bad-point limit, `N_loops x M` vertices, coloured by cutting axis (x blue, y green, z red), with predicted in-plane normals in `nx/ny/nz` |
| `<stem>_bad_loops.ply` | loops over the `0.15` export limit, all red. Both sides together hold every loop; no combined all-loops PLY is written |
| `<stem>_loop_split.txt` | one `g`/`b` per loop in checkpoint order, used to interleave the two split PLYs back |
| `<stem>_fits.png` | grid of loops and fitted curves |
| `planes/plane_SS_<axis><offset>_{slice,loops,fit}.ply` and `.png` | per-plane files |
| `planes.txt` | per-plane counts and errors, `!` marks an outlier |
| `bad_loops.txt` | one line per loop over `--max-err` |
| `snapshots/` | periodic fit grids (if enabled) |
| `<stem>_model.pt` | checkpoint + replay metadata |
| `<stem>_triplane.npz` | triplane grids and sampled features (triplane mode) |
| `<stem>_feature_graph.csv`, `<stem>_feature_smoothness.txt` | neighbour graph and smoothness report (triplane) |
| `<stem>_code_graph.csv`, `<stem>_code_smoothness.txt` | the same for embedding rows |

There are two different limits, and they do different jobs:

- `--max-err` (default `0.03`, a fraction of the longest extent) is the
  **highlighting** limit. It turns a loop red in the figures, writes the `!` in
  `planes.txt`, and fills `bad_loops.txt`. It does not split the PLY export.
- `BAD_POINT_ERR = 0.15` (fixed, in the plane frame) is the **export** limit. A
  loop with any single fitted point past it goes to `_bad_loops.ply`; every other
  loop goes to `_good_loops.ply`. The two files plus `_loop_split.txt` hold every
  loop, and no combined `_loops.ply` is written by current runs.

A loop whose traversal disagrees with the majority of its paired point normals is
reversed before export (curve, normals, errors and tangents together), so the
saved block always runs the way the surface does. The printed `align` value is
measured before that reversal.

---

## 3. Replay probe (`loop_shift_probe.py`)

This tests the hypothesis behind the triplane field: shifting a loop's centroid
query along its plane normal should generate something close to the loop that
was actually fitted on the neighbouring plane.

```bash
conda run -n seawall python loop_shift_probe.py --run log/triplane_shift_s29
conda run -n seawall python loop_shift_probe.py \
    --run log/triplane_shift_s29 --loops 3,17,42
```

For every loop it:

1. finds the nearest loop on each adjacent plane by shifting the loop's centroid
   along its plane normal (matching by `queries`, not by the shared plane origin);
2. decodes the network at the shifted query;
3. reports two variants: query shifted but the curve still placed in the original
   plane, and query plus plane origin shifted together (the generated curve is
   placed on the shifted plane);
4. compares the generated curve with the real neighbour and with the original
   loop.

The script first checks `t = 0` replay against the stored fit; a non-trivial
error there means the checkpoint and the loops PLY disagree.

Metrics written to `loop_shift/shift_pairs.csv`:

| column | meaning |
|---|---|
| `gen_nb` | Chamfer, generated shifted curve -> real neighbour |
| `gen_orig` | Chamfer, generated shifted curve -> original loop |
| `nb_orig` | Chamfer, real neighbour -> original loop |
| `shape` | shape change from moving the query alone, same plane |
| `cov_gen`, `cov_nb` | coverage of one curve by the other at `--radius` |
| `step` | signed axial move, `+/-` one plane spacing |
| `axial` | measured centroid move along the normal |
| `axial_res` | `abs(axial - step)`; side-aware, because raw `+/-` shifts cancel |
| `drift` | in-plane movement of the centroid |
| `r0`, `r1`, `a0`, `a1` | mean radius and signed area before/after |

Outputs: `shift_pairs.csv`, `loop_shift.png` (2-D panels), `loop_shift_3d.png`
(three 3-D views), and `loop_shift.ply` (cloud of generated curves).

The probe requires a triplane checkpoint with stored queries. It reads
`<stem>_loops.ply` when present; otherwise it interleaves
`<stem>_good_loops.ply` and `<stem>_bad_loops.ply` using the per-loop flags in
`<stem>_loop_split.txt`. `--loop-ply` overrides the path.

### Current five-shape result (2 x 64, `--samples 29`, 1000 epochs, TV 0, normal 0.01)

Median Chamfer:

| shape | pairs | gen -> neighbour | real neighbour -> original | gen closer than neighbour | within 0.02 |
|---|---:|---:|---:|---:|---:|
| dancing children | 405 | 0.0768 | 0.0451 | 52% | 29% |
| bunny | 154 | 0.0925 | 0.0498 | 35% | 18% |
| horse | 268 | 0.0176 | 0.0463 | 70% | 53% |
| armadillo | 314 | 0.0827 | 0.0548 | 50% | 23% |
| lucy | 302 | 0.0585 | 0.0505 | 52% | 23% |

Interpretation: the generated curve is better than the real-neighbour baseline
on horse and roughly on par at the median on dancing, armadillo and lucy, but
not on bunny. The network absorbs part of the axial shift, so the generated loop
often stays closer to the original than to the neighbour. The limiting factor is
the query, not MLP capacity: moving the query to the same in-plane position on
the next plane samples a feature that is off the manifold the field learned.
This is worst where the cross-section changes fastest.

### 4 x 32 versus 2 x 64

Same experiment, `--layers 4 --neurons 32` (the deep/narrow arm):

| shape | 2x64 gen -> nbr | 4x32 gen -> nbr | 2x64 closer | 4x32 closer |
|---|---:|---:|---:|---:|
| dancing | 0.0768 | 0.0691 | 52% | 51% |
| bunny | 0.0925 | 0.0679 | 35% | 48% |
| horse | 0.0176 | 0.0131 | 70% | 71% |
| armadillo | 0.0827 | 0.0652 | 50% | 50% |
| lucy | 0.0585 | 0.0582 | 52% | 54% |

The deeper/narrower net lowers the median on every shape, with the largest gain
on bunny, and lifts bunny's success rate most. Fit loss is essentially unchanged
except on armadillo (`0.0010 -> 0.0013`) and lucy (`0.0006 -> 0.0010`). The
improvement is architectural, not extra capacity: 4 x 32 has fewer weights than
2 x 64.

---

## 4. Propagation, alignment and tubes (`loop_propagation.py`)

This stage reads a finished run's per-plane files and `planes.txt`; it imports
only `align_loops.py`. Nothing here is imported by extraction or fitting.

Always pass `--run` explicitly: the script default is the old hand run
(`log/laurent_hand_normal_cubic_lam3`), not the run you just trained.

```bash
conda run -n seawall python loop_propagation.py \
    --run log/triplane_shift_s29 --out log/loop_propagation_dancing_triplane --step 1

conda run -n seawall python loop_propagation.py \
    --run log/triplane_shift_s29 --out log/loop_propagation_dancing_triplane --step 2

# optional deformation-budgeted variant
conda run -n seawall python loop_propagation.py \
    --run log/triplane_shift_s29 --out log/loop_propagation_dancing_triplane --step 2b

conda run -n seawall python loop_propagation.py \
    --run log/triplane_shift_s29 --out log/loop_propagation_dancing_triplane --step 3

conda run -n seawall python loop_propagation.py \
    --run log/triplane_shift_s29 --out log/loop_propagation_dancing_triplane --step 4
```

`--step all` runs 1, 2, 3, 4 (not 2b). Step 1 reads the run; steps 2-4 read the
previous step's JSON, and each writes `stepN/` under `--out`.

### Step 1: stable loops

Per-loop in-plane RMS of the fitted curve against its own labelled slice points.
Loops under `--rms` (default `0.025`) are stable, and that cut is what step 2
propagates from. The adjacent-plane switch reported here is diagnostic only.

With `--good-only`, `load_run` first drops every loop on the bad side of the
run's `<stem>_good_loops.ply` split, so tracks and tubes are built from good
loops only. It expects exactly one `*_good_loops.ply` in the run folder.

Outputs: `step1/loops.csv`, `loops.json`, `stable_montage.png`,
`rejected_montage.png`, `rms_distribution.png`, `rms_vs_s.png`,
`thresholds.png`, `summary.txt`.

### Step 2: rigid translation transfer

Two-sided translation transfer from every stable loop to the components on
adjacent planes. Each candidate translation is calibrated against a null
distribution, ranked, and grown into tracks. `--tau` (default `0.02`) is the
in-plane tolerance, `--accept` (default `0.5`) the acceptance ratio. Loops are
sorted into kept, adopted, discarded and orphan.

Outputs: `step2/tracks.json`, `attempts.csv`, `track_loops.csv`, `null.png`,
`decisions.png`, `tracks.png`, `tracks_strip.png`, `summary.txt`.

### Step 2b: deformation-budgeted transfer (optional)

The same propagation, but the seed may also move along its in-plane normal with
a constant taper mode plus `--deform-modes` Fourier modes (default 8), fitted by
robust ICP inside a fixed budget `--deform-budget` (default 0.15 of the smaller
section radius). An edge is admitted when rigid passes, or when deformation
passes with an RMS offset under the budget. The target's own fitted loop still
continues the track, so the deformed seed is evidence only.

Outputs mirror step 2 under `step2b/`. Select these tracks downstream with
`--tracks step2b`; `--tracks auto` (default) prefers step 2b when it exists.

### Step 3: loop-parameter alignment

This is the only place loop-parameter (seam/phase) alignment happens. It calls
`align_loops.synchronize` on every track:

1. `pair_residual` estimates the circular shift between two loops by FFT
   cross-correlation with parabolic refinement;
2. `switch` decides from that residual `D` whether a pair is genuinely shifted,
   with closed-form switch `s = lambda / (lambda + D)`;
3. `solve_offsets` solves the seam offsets `alpha` by weighted least squares,
   with a gauge prior `mu`;
4. `circular_shift` applies the Fourier shift in turns.

Tracks are split at pairs whose switch falls below `--min-switch` (default
`0.3`). Parameters: `--lam 0.02`, `--k 256`.

Outputs: `step3/align.json` (chains, per-pair fits, offsets, diagnostics),
`pairs.csv`, `tracks.csv`, `pairs.png`, `seams.png`, `aligned_N.png` panels,
`summary.txt`.

Alignment is **not** run by `rosa_main.py` or `loop_extraction.py`; it exists
only as this step. The new `triplane_shift*` and `triplane_shift4x32_*` runs do
not have a `step3/` yet, so their loops are not parameter-aligned.

### Step 4: tubes

Loft every aligned sub-chain of two or more loops and export the tubes.

Outputs: `step4/tubes.ply`, `tubes_cloud.ply`, `tubes.csv`, `tubes.json`,
`tubes.png`, `tubes_views.png`, `tubes_context.png`, `tubes_check.png`,
`sections.png`, `discarded.csv`, `summary.txt`. The loader checks the expected
vertex count on reload.

The top-level `summary.md` is assembled from every `stepN/summary.txt` present,
so rerunning a single step does not truncate it.

### Propagation CLI reference

| flag | default | meaning |
|---|---|---|
| `--run` | `log/laurent_hand_normal_cubic_lam3` | finished fitting run |
| `--out` | `log/loop_propagation` | output directory |
| `--step` | `all` | `1`, `2`, `2b`, `3`, `4`, `all` |
| `--tracks` | `auto` | `auto`, `step2`, `step2b` |
| `--rms` | `0.025` | stable-loop cut |
| `--tau` | `0.02` | in-plane transfer tolerance |
| `--accept` | `0.5` | acceptance ratio |
| `--lam` | `0.02` | switch prior |
| `--min-switch` | `0.3` | split threshold |
| `--k` | `256` | loop samples for alignment |
| `--deform-modes` | `8` | Fourier modes in step 2b |
| `--deform-budget` | `0.15` | deformation budget in step 2b |
| `--good-only` | off | drop loops on the bad side of the good/bad split before propagation |

### Triplane feature clustering (diagnostic)

A separate question from the shift probe: can the distance between two loops'
triplane features replace, or guide, the geometric propagation rule? The current
answer, measured on lucy in [triplane_loop_clustering.md](triplane_loop_clustering.md),
is **feature proposes, geometry disposes**:

- accepted step-2 edges have median feature distance `0.486`, rejected ones
  `2.053`; a feature threshold ranks them with AUC `0.967` (centroid distance is
  marginally better at `0.977`);
- a mutual feature nearest-neighbour graph recovers `93%` of the geometric edges,
  but only `71%` of its own edges are geometric;
- step-3 seam cuts are **not** explained by feature distance (AUC `0.601`);
- pure feature tracking gives 10 tubes / 47 loops, against the geometric
  11 tubes / 51 loops. The conservative rule (geometric tracks plus a 0.40
  coverage rescue and the seam/drift checks) keeps all 11 tubes and adds one
  loop, for 11 tubes / 52 loops.

This is a diagnostic join, not a committed pipeline stage; `loop_propagation.py`
still uses geometry.

### Seam and loop inspection scripts

| script | purpose |
|---|---|
| `seam_z0_view.py` | write `_good_loops_z0.ply`/`.png` with the first 1% of each loop green and an octahedron at every `z = 0` seam; optional multi-run contact sheet |
| `seam_check.py` | print the pair seam distance `D` a checkpoint exports, at `z = 0` and at its own phases, and compare per pair (`--against`) with another run on the same plane set |
| `seam_sphere_ply.py` | append a red icosphere at each loop's seam to a copy of a loop PLY, leaving loop vertices and colours untouched |
| `loop_sweep_movie.py` | movie revealing the loops along x, y and z one plane at a time, then the union |
| `axis_color_loops.py` | colour older loop PLYs by cutting axis; current `rosa_main` already colours good loops by axis, so this is for runs made before that change |

### Anchored seam and learned alignment

- [LOOP_ANCHOR_PLAN.md](LOOP_ANCHOR_PLAN.md) proposes making `z = 0` lie on a
  fixed in-plane reference direction per axis, by exact post-hoc
  reparameterization. It changes neither geometry nor fit error, and its optional
  seam-synchronisation stage is off by default. Status: proposed, not implemented.
  The gauge-only solve that is implemented (`--seam-align`, §2.3) aligns the
  seams to each other; it does not tie them to a direction of the plane.
- The gauge-only phase buffer — no change to the objective — is implemented as
  `--seam-align` in `mini_nn.Trainer` and `rosa_main.py`. The cube warm-up
  (§2.4) is a separate prior on the network itself, not a phase carrier.

---

## 5. End-to-end recipe for a new shape

```bash
SHAPE=normalized_shapes_cuboid/my_shape.ply
STEM=my_shape
RUN=log/my_shape_triplane

# 1+2. extract and fit, with checkpoint. The cube warm-up
#      (warmup/cube_warmup.pt) is loaded by default; --no-warmup ablates it.
#      Add --seam-align 1 --seam-every 10 --seam-steps 100 to solve the seam.
conda run -n seawall python rosa_main.py --train --path "$SHAPE" --out "$RUN" \
    --samples 29 --epochs 1000 \
    --feature triplane --triplane-tv 0.01 --normal 0.01

# check the seam the checkpoint exports
conda run -n seawall python seam_check.py "$RUN/${STEM}_model.pt"

# 3. check replay and axial-shift behaviour
conda run -n seawall python loop_shift_probe.py --run "$RUN"

# 4. propagate, align, loft
conda run -n seawall python loop_propagation.py \
    --run "$RUN" --out "${RUN}_propagation" --step all

# optional: use only the loops on the good side of the export split
conda run -n seawall python loop_propagation.py \
    --run "$RUN" --out "${RUN}_propagation_good" --step all --good-only
```

For a quick low-resolution pass, drop `--samples` to `5` and `--epochs` to
`300`. For the adaptive density used in the long runs, use
`--samples 5 --add-planes 2 --samples-end 15 --refine-every 100 --epochs 1000`.

---

## 6. Current state and known gaps

What is verified:

- The triplane field trains and fits all sampled shapes to a median loop error
  around `0.001` (squared Chamfer).
- Checkpoints replay exactly (`t = 0` probe error is numerical noise), and the
  shift probe runs on five shapes.
- `align_loops.py` self-test passes; step 3 produces non-zero seam offsets and
  splits tracks at rejected pairs.
- The normal-orientation loss is implemented and active when `--normal > 0`; all
  runs here used `0.01` and annealed it to zero.
- The phase-only seam solve (`--seam-align`) leaves the fit untouched (identical
  epoch lines, weights at the run-to-run noise floor) and moves every
  neighbour-graph pair to its best achievable shift on the trained bunny
  checkpoint. See §2.3.

Known gaps:

- **Shift generation is shape-dependent.** It is reliable on smooth, slowly
  varying cross-sections (horse) and weak on bunny and on fast-changing slices.
  The cause is the query feature, not network width.
- **Topological changes are not handled.** A section that splits into detached
  digits cannot be produced by deforming one closed loop. Step 2b stops at the
  deformation budget; cutting the seed at a junction and closing each arc
  against its own component is still open.
- **Parameter alignment coverage.** Runs trained with `--seam-align` carry their
  phases in the checkpoint and are seam-aligned at export. Runs without it (the
  `triplane_shift*` batch) are only aligned where `loop_propagation.py --step 3`
  has run; the newer `log/cmp_tri_tv01_*_propagation/` runs do. Step 3 still
  decides which pairs are the same cross-section and where tracks are cut —
  aligning the gauge does not replace it.
- **Feature clustering is diagnostic only.** The triplane feature graph is not
  used by `loop_propagation.py`; its role and numbers are in
  [triplane_loop_clustering.md](triplane_loop_clustering.md).
- **The seam is aligned, not anchored.** `--seam-align` puts neighbouring seams
  in the same place up to one global turn per run; it does not pin them to a
  fixed in-plane direction. That is what
  [LOOP_ANCHOR_PLAN.md](LOOP_ANCHOR_PLAN.md) proposes, and it is still proposed.
- **No polycube extraction yet.** The next planned stage is
  `polycube_from_loops.py`; routes, decisions and evaluation are listed in the
  "Polycube from the loops" section of [todo.md](todo.md).
