# Loop propagation test plan

Status: implemented and run on 2026-11-02. `loop_propagation.py` is standalone:
nothing here is wired into `loop_extraction.py` or `rosa_main.py`; it reads a
finished run's PLYs and `planes.txt` and imports `align_loops.py`.

Run: `log/laurent_hand_normal_cubic_lam3` — 45 planes, 78 loops, longest extent
2.058.

Steps are 1, 2, 2b (optional deformation-budgeted propagation), 3 and 4.
`--tracks {auto,step2,step2b}` selects which propagation result steps 3–4
consume; `auto` takes the most recently written one.

## Outcome

- 48 loops stable (step 1), 37 accepted propagation edges forming 13 tracks
  (step 2), 20 of 37 pairs kept forming 9 aligned sub-chains (step 3), and
  **8 tubes using 26 loops** (step 4).
- One of the 9 sub-chains (c7, z, 3 loops) was rejected at step 4 because its
  section centres slide sideways 4.06× faster than along its section normal:
  it is a sheared ribbon, not a tube. Removing it also removes the one
  questionable surface from the export.
- 52 loops are discarded: 49 that no tube-chain reached plus the 3 loops of
  the sheared chain. Compared with the earlier single-loop whole-hand baseline
  (3 tubes, 11 loops), this is 8 tubes and 26 loops.
- All four reload checks pass: 52 components (8 tubes + 26 loops + 18
  connectors), bbox margin 0.0175, unique loop accounting, and all slice
  vertices preserved in `tubes_cloud.ply`.

The two method corrections discovered by running it are documented in step 2
(physical placement instead of free translation; the far-plane null is
diagnostic only) and step 3 (orientation canonicalization; axis drift).

### Deformation-budgeted propagation (step 2b)

Step 2b allows the seed to deform when the rigid transfer nearly fits a
neighbour, without turning the edge test into a free re-fit. The seed is still
placed physically and may shift only within the step-2 cap. Its offset along
the loop normal is expanded in a constant mode (taper/dilation) plus
`--deform-modes` Fourier modes and fitted to the target's points by robust
ICP: the discovery radius starts at the seed's current 90th-percentile
distance (capped at half the section radius) and shrinks to `tau`, and a Tukey
weight stops far points from pulling. An edge is admissible when either the
rigid transfer passes, or the deformed one passes and its RMS normal offset is
at most `--deform-budget` of the smaller section radius. In both cases the
track continues with the target's **own** fitted loop; the deformed seed is
evidence only.

Result on the hand at budget 0.15:

| | rigid (step 2) | deformed (step 2b) |
|---|---|---|
| edges tested / accepted | 138 / 37 | 149 / 52 |
| tracks | 13 | 13 (merged) |
| stable kept / discarded | 39 / 9 | 42 / 6 |
| adopted / orphan | 11 / 19 | 23 / 7 |
| accepted with a deformed seed | — | 13 |
| rejected only by the budget | — | 2 |
| tubes / loops used (steps 3–4) | 8 / 26 | 9 / 30 |

The extra tubes are the palm `y` sleeve growing from 5 to 6 loops
(`y -0.83 … -0.36`, v 0.607) and a new `x` tube of 3 loops
(`x +0.23 … +0.33`, s 0.63, drift 0.53). The new x chain that reaches the wrist
(`x -0.34 … -0.28`) is still cut by the switch and the remaining x chain is
excluded as sheared, so the x gain is the distal one only. All five step-4
reload checks pass (60 components = 9 + 30 + 21, bbox margin 0.0175).

The budget sweep on this run is 37 / 43 / 52 / 52 accepted edges at 0.05 /
0.10 / 0.15 / 0.20. 0.05 is the rigid baseline, 0.15 already captures every
pair that passes the coverage test, and 0.20 adds none. The accepted rescues
need at most 0.143 of the seed radius; the next rejected pair needs 0.264, so
0.15 sits in a clear gap rather than on a threshold. These numbers are from
one shape; the budget is still a design parameter, not a calibrated constant.

The remaining failure is topological, as before: a plane whose section splits
into detached digits cannot be explained by deforming one closed loop. The
next step is to cut the seed at the junction and close each arc against its
own component.

### Density test: 29 planes per axis

The same pipeline on a 29-plane/axis fit of the same hand
(`log/laurent_hand_normal_cubic_lam3_s29`, 87 planes, 149 loops, 1000 epochs
flat from the start) gives:

| | 15/axis | 29/axis |
|---|---|---|
| planes / loops | 45 / 78 | 87 / 149 |
| stable | 48 (61%) | 123 (83%) |
| median RMS | 0.0218 | 0.0080 |
| accepted edges / tracks | 37 / 13 | 113 / 17 |
| aligned sub-chains | 9 | 20 |
| tubes / loops used | 8 / 26 | 10 / 70 |
| y / z tubes | 6 / 2 | 7 / 2 |
| sheared chains excluded | 1 (3 loops) | 10 (45 loops) |
| reload components | 52 | 140 |

The palm gap closes: the longest `y` tube grows from 5 to 14 loops (offset
-0.83 to -0.06), and one `z` tube reaches 11 loops. Two caveats belong with
these numbers. The dense run is also a better fit (median plane error 0.0098
vs 0.0225) and trained 1000 epochs at full density, so part of the stable-loop
gain is fit quality, not only spacing. And the axis-drift gate is a ratio whose
denominator is the axial plane step, which halves here; the same physical shear
therefore reads twice as large. Normalising drift by the spacing would accept
four more `z` chains (15 loops) and one `x` chain (6 loops, a thin palm section
that may still be a sheet): 15 tubes / 91 loops before any further shape check.
So 10 tubes is a conservative count, not the density limit. A flat 15-plane,
1000-epoch control would separate fit quality from spacing; it has not been run.

All reload checks pass on the dense export: 140 components (10 + 70 + 60),
bbox margin 0.0133, unique loops, used set equals aligned set, 384,929 slice
vertices plus 263,080 tube/loop vertices. Discard reasons are written one row
per loop to `step4/discarded.csv` (45 sheared chain, 15 seam cut, 4 no accepted
edge, 15 not reached), so every one of the 79 discarded loops has a logged
reason.

The loader reads the per-loop block size from each `_fit.ply` instead of
assuming 2383 points, and finds `*_slices.ply` by suffix instead of the run
folder name; both were hard-coded to the 15-plane run and blocked this one.

### Density test: dancing children

The same pipeline on the dancing-children shape. The existing run is 10 planes
per axis (`log/dancing_children100K_cubic_lam3`, produced by the 5->10
refinement recipe); the dense run is 29 per axis, flat, 1000 epochs
(`log/dancing_children100K_normal_cubic_lam3_s29`).

| | 10/axis | 29/axis |
|---|---|---|
| planes / loops | 30 / 73 | 87 / 210 |
| stable | 55 (75%) | 161 (77%) |
| median loop RMS | 0.0110 | 0.0109 |
| median plane fit error | 0.0088 | 0.0083 |
| accepted edges / tracks | 18 / 159 → 14 | 115 / 571 → 35 |
| aligned sub-chains | 12 | 34 |
| tubes / loops used | 12 / 24 | 33 / 121 |
| x / y / z tubes | 1 / 8 / 3 | 6 / 19 / 8 |
| sheared chains excluded | none | 1 (3 loops) |
| reload components | 48 | 242 |

Unlike the hand, fit quality is essentially unchanged (0.0088 vs 0.0083), so
this gain is spacing, not a better upstream fit. All reload checks pass: 242
components (33 + 121 + 88), bbox margin 0.0323 past the slice cloud (loop fit
deviation), unique loops, used set equals aligned set, 285,726 slice vertices
plus 453,904 tube/loop vertices (739,630 total), and 121 used + 89 discarded =
210 with no overlap. Every discard is logged in `step4/discarded.csv`
(3 sheared chain, 26 seam cut, 25 no accepted edge, 35 not reached).

The drift caveat is larger here: the median plane spacing falls from 0.152 /
0.119 / 0.079 to 0.049 / 0.038 / 0.026, a factor of about 3.1 on every axis.
The one excluded `z` chain has drift 3.94, which is 1.27 in baseline-spacing
units, and three included `z` tubes at 1.60-1.98 are 0.52-0.64. So a
spacing-normalised gate would keep 34 tubes / 124 loops. The reported 33 is the
conservative, fixed-ratio count.

### Code smoothness across adjacent planes

The loop fit conditions a shared MLP on a 32-D code per `(axis, offset,
slot)`. Nothing ties codes of neighbouring loops together, so equal shape has
no reason to give equal codes. `rosa_main.py --code-smooth` adds an opt-in
penalty on the codes of loops that are geometric neighbours:

- Pair loops on adjacent planes whose in-plane centroids are mutual nearest
  neighbours.
- Reject a pair when the separation exceeds `--code-gate` times that axis's
  median plane spacing, or when the two in-plane radii differ by more than
  `--code-min-radius` (default ratio 0.4), so a torso loop is not pulled
  towards a small limb loop.
- Weight each pair by `1 - d/limit`, and add
  `lambda * mean(w * ||c_i - c_j||^2)` to the Chamfer loss, in the same units.
  The graph is rebuilt at every refinement stage and covers the codes that are
  still alive, so inherited rows are constrained too.

The fit writes `*_code_graph.csv` (one row per pair: keys, separation, code
distance, weight) and `*_code_smoothness.txt` (graph vs all-pair medians,
ratio, weighted mean square). With `--code-smooth 0` the same diagnostics are
still written, so baselines are comparable.

1000-epoch A/B on the dense dancing-children fit (`--normal 0.01`,
`--snapshot-every 0`; 87 planes, 210 loops, 123 graph pairs):

| | no term | λ=1e-3 | λ=1e-2 |
|---|---|---|---|
| graph code distance (median) | 0.4346 | 0.0883 | 0.0277 |
| all-pair code distance (median) | 1.8771 | 1.6352 | 1.5870 |
| graph/all ratio | 0.232 | 0.054 | 0.017 |
| weighted mean squared distance | 0.8544 | 0.0679 | 0.0038 |
| corr(code distance, in-plane separation) | 0.226 | 0.504 | 0.607 |
| plane fit error (median) | 0.0095 | 0.0102 | 0.0101 |
| planes over 1.5× median | 25 | 28 | 17 |
| step-1 stable loops | 161 | — | 159 |
| accepted edges / tracks | 115 / 35 | — | 117 / 32 |
| seam cuts (step 3) | 25 | — | 14 |
| tubes / loops used | 33 / 121 | — | 34 / 133 |

At `λ=1e-2` the median graph code distance falls 16× (0.4346 → 0.0277) and
the weighted mean square falls 225×, but the all-pair median only moves from
1.88 to 1.59. So the term compresses neighbours without collapsing the whole
code space: distances now follow spatial separation (correlation 0.61 rather
than 0.23), and the code becomes a local descriptor rather than an arbitrary
identifier. Fit is essentially neutral — the median plane error
moves by 0.0006 — while the number of planes worse than 1.5× the median falls
from 25 to 17. The three worst `x` planes in the baseline (0.016, 0.021, 0.026)
come back as 0.014, 0.013, 0.015, which is why the constrained fit has no
flagged planes. Downstream, step 3 cuts 14 instead of 25 pairs and step 4
keeps one more tube and 12 more loops (77 discarded instead of 89). The
trade-off is one extra sheared chain: two chains exceed the fixed drift gate
instead of one (z drift 3.04 and 2.43), so six loops are dropped as sheared
rather than three. All reload checks pass for `log/loop_propagation_dancing_cs001`
(266 components = 34 tubes + 133 loops + 99 connectors, bbox margin 0.0193,
unique loops, used set equals aligned set, slice vertices preserved).

The constraint does not manufacture matches. The six previously inspected
rejected pairs are still rejected at `λ=1e-2`, including the false friend
`82:0 -> 83:0` (point coverage 0.99, loop support 0.43) and the near miss
`22:0 -> 23:0` (0.62 / 0.37).

`log/code_smooth_sweep/code_smoothness_1000ep.png` plots the per-edge code
distance against separation, the endpoint penalty, and the downstream counts;
`log/code_smooth_sweep/compare_code_smoothing_1000ep.py` regenerates it.

The codes themselves are still not exported (only pairwise distances are), so
step 2 cannot yet score an edge by code agreement. That is the natural next
use: a seed and target whose codes differ sharply are unlikely to be the same
feature even when their shapes almost overlap.

### Triplane feature field instead of one row per loop

A row table makes the code an identifier: two loops only share it if the fit
happens to write similar numbers, and a loop that appears in a later stage gets
a fresh row. `--feature triplane` replaces the table with three orthogonal 2D
grids (XY, XZ, YZ), `C` channels each and `R` cells per side, spanning the cloud
bounding box plus 2% padding. Every loop samples the field at its own centre in
3D (bilinear, border padding), and the three samples are concatenated into
`3C` values that condition the same MLP. The code is now a continuous function
of world position, so neighbouring loops start similar by construction and a
loop added at a later stage simply reads the field where it is instead of
getting an untrained row.

```bash
python rosa_main.py ... --feature triplane --triplane-res 32 \
  --triplane-channels 16 --triplane-tv 0 --code-smooth 1e-2
```

| Flag | Meaning |
|---|---|
| `--feature {embedding,triplane}` | what conditions the MLP [`triplane`] |
| `--triplane-res` | cells per side of each grid [`32`] |
| `--triplane-channels` | channels per grid; latent width is `3x` [`16`] |
| `--triplane-tv` | total-variation weight on the grids [`0.01`; only with triplane] |

The graph and the smoothness term are unchanged; they now act on the sampled
`3C`-D features rather than on rows. Diagnostics are written as
`*_feature_graph.csv` / `*_feature_smoothness.txt`, and the learned field is
exported to `*_triplane.npz` (grids, bounds, loop queries, sampled features).
Inheritance between stages copies the grids together with the MLP, and nothing
is held or re-warmed after stage 0.

Note on the zero start: the grid is initialised to zero, and a run loads the
cube-warm-up checkpoint (`--warmup`, [WARMUP.md](WARMUP.md)) before its first
fit by default. The numbers below were taken with the older per-loop square
warm-up, before the cube checkpoint existed; read them as the historical series
they are.

1000-epoch run on the same dense dancing-children fit as the table above (87
planes, 210 loops, 123 graph pairs, `--normal 0.01`). "median plane fit error"
is the median over planes, as in `planes.txt`; the figure pools the 210 loop
errors, which are smaller because a multi-loop plane averages them.

| | emb. no term | emb. λ=1e-2 | tri 32 no term | tri 32 λ=1e-2 | tri 32 λ=1e-2, 60 warm | tri 64 λ=1e-2 |
|---|---|---|---|---|---|---|
| median plane fit error | 0.0095 | 0.0101 | 0.0115 | 0.0124 | 0.0098 | 0.0233 |
| bend (a circle is 39.5) | 2076 | 2324 | 2439 | 2115 | 1672 | 2791 |
| graph feature distance (median) | 0.4346 | 0.0277 | 0.4851 | 0.0421 | 0.0596 | 0.0312 |
| all-pair feature distance (median) | 1.8771 | 1.5870 | 1.6700 | 1.5904 | 1.5261 | 1.6522 |
| graph/all ratio | 0.232 | 0.017 | 0.290 | 0.026 | 0.039 | 0.019 |
| weighted mean squared distance | 0.8544 | 0.0038 | 0.4155 | 0.0048 | 0.0113 | 0.0024 |
| step-1 stable loops | 161 | 159 | 156 | 159 | 154 | 131 |
| tracks | 35 | 32 | 32 | 33 | 33 | 25 |
| tubes / loops used | 33 / 121 | 34 / 133 | 32 / 117 | 35 / 138 | 35 / 131 | 28 / 116 |

What the runs say:

- The triplane is continuous but not smooth by construction. At `λ=0` its
  graph/all ratio is 0.290 against 0.232 for free rows — if anything slightly
  worse — because the MLP is free to read the field in any direction. The
  explicit smoothness term is still what brings it to 0.026. Continuity only
  guarantees that neighbours share a cell, not that the loss asks them to use
  it alike.
- At 32 cells the triplane is at parity with the row table on fit (0.0124 vs
  0.0101 median plane error) and slightly ahead downstream (35 tubes / 138
  loops vs 34 / 133), with feature locality a little worse (0.026 vs 0.017).
  On one shape that is a tie, not a win.
- Resolution matters more than expected. 64 cells puts a cell at 0.0277 world
  units while adjacent planes are 0.026–0.049 apart, so most loops sit in a
  cell of their own: the sharing that made the triplane attractive disappears,
  fit degrades to 0.0233 and downstream falls to 28 tubes / 116 loops. 32 cells
  (cell ≈ 0.055) is the right scale for this sampling.
- A 60-step warm-up is free and fixes the early-fit penalty: median plane
  error 0.0098 and bend 1672, both best of the seven; downstream 35 tubes but
  131 loops, and locality 0.039. Whether the extra loop or the better fit is
  worth more is a downstream question this run does not settle.
- Run-to-run noise on feature distances is 5–8% (two identical r32 runs gave
  graph 0.0390 vs 0.0421 and weighted mean square 0.0045 vs 0.0048), so
  differences smaller than that in the table are not real. The fit
  distributions use all 210 loops and are less sensitive.

Visual: [compare_triplane.png](log/triplane_compare/compare_triplane.png)
(fit, locality, downstream), [triplane_field.png](log/triplane_compare/triplane_field.png)
(the three learned grids and the loop features),
[fit_compare_triplane.png](log/triplane_compare/fit_compare_triplane.png)
(planes 10, 30, 60 side by side); `log/triplane_compare/compare_triplane.py`
regenerates all three. Field: [triplane.npz](log/dancing_children100K_triplane_s29/dancing_children100K_cubic_lam3_triplane.npz).
Propagation: [triplane](log/loop_propagation_dancing_triplane/summary.md),
[60 warm-up](log/loop_propagation_dancing_triplane_p60/summary.md),
[64 cells](log/loop_propagation_dancing_triplane_r64/summary.md),
[no term](log/loop_propagation_dancing_triplane_lam0/summary.md). All four
reload checks pass (e.g. the λ run: 276 components = 35 tubes + 138 loops + 103
connectors, bbox margin 0.0208, unique loops, used set equals aligned set).

The sampled features in `*_triplane.npz` are the triplane's equivalent of the
row table, so the code-agreement use for step 2 is now open for both modes.
`todo.md`'s triplane item is closed by this section.

The question this plan answers, step by step:

1. Which fitted loops are stable enough to be seeds?
2. Which of those seeds can be propagated to a neighbouring plane, and which
   loops therefore stay or are discarded?
3. How do the kept loops align along each track?
4. What tubes do the aligned tracks produce?

Every step writes its own table, figure and JSON under
`log/loop_propagation/stepN/`, with a per-step `summary.txt`; a top-level
`summary.md` is assembled from the four after the last step. Later steps read
the earlier JSON, so each step can be inspected before the next runs.

```bash
python loop_propagation.py --step 1     # stable-loop inventory
python loop_propagation.py --step 2     # propagation, keep/discard
python loop_propagation.py --step 2b    # same with a bounded deformation
python loop_propagation.py --step 3     # alignment per track
python loop_propagation.py --step 4     # tubes
python loop_propagation.py --step all   # steps 1-4 (rigid path)
python loop_propagation.py --step all --good-only   # good loops only
```

Steps 3–4 read `--tracks` (default `auto`, the newest of `step2` and `step2b`).
Run the deformable chain as `--step 2b`, then `--step 3 --tracks step2b`, then
`--step 4 --tracks step2b`.

Parameters (defaults in brackets):

| Flag | Meaning |
|---|---|
| `--run` | finished run folder [`log/laurent_hand_normal_cubic_lam3`] |
| `--out` | output root [`log/loop_propagation`] |
| `--rms` | stable-loop RMS limit, in world units [`0.025`] |
| `--tau` | point-to-curve tolerance for coverage [`0.02`] |
| `--accept` | two-sided coverage needed to propagate [`0.5`] |
| `--lam` | switch penalty in `align_loops` [`0.02`] |
| `--min-switch` | pairs below this `s` are cuts [`0.3`] |
| `--k` | common loop samples [`256`] |
| `--tracks` | propagation result for steps 3–4: `auto` (newest), `step2`, `step2b` [`auto`] |
| `--deform-modes` | Fourier modes for the normal offset, on top of the constant taper mode [`8`] |
| `--deform-budget` | max RMS normal offset as a fraction of the smaller section radius [`0.15`] |
| `--good-only` | drop the loops on the bad side of `<stem>_good_loops.ply` before propagation, so only good loops enter tracks and tubes [`off`] |

Conventions, fixed for the whole test:

- A loop is one fixed-size block of a `_fit.ply` (2383 points at 15
  planes/axis, 2354 at 29: the block is the trainer's latent resolution, so the
  loader reads its size per run); the block's colour is the same colour as its
  labelled component in `_loops.ply`, so a loop's own slice points are selected
  by colour, never by list position.
- Steps 1–2 measure shape in 2D. Each loop is projected into its own best-fit
  plane, with the normal signed along its cut axis and `e1` the projection of a
  fixed world axis, so adjacent planes share an in-plane orientation and no
  loop is mirrored. Distances are point-to-polyline, not point-to-point.
- Steps 3–4 use the frame the existing tube builder uses: the 3D fitted loop,
  centred at its centroid, with the cut-axis coordinate zeroed, and the centre
  path interpolated in 3D. This is the frame of the verified whole-hand PLY.

---

## Step 1 — identify stable loops by fit RMS

### Input

All 78 loops of the run, each with its own labelled slice points.

### Score

For loop `i`:

```
rms_i = sqrt( mean over its slice points q of  dist(q, fitted curve)^2 )
```

measured in the loop's own projected plane. Also record:

- `q90_i`, `cov_i` = fraction of its points within `tau` of its curve;
- `radius_i` = RMS radius of the curve, and `rms_i / radius_i` (a size-fair
  second reading, reported but not used for the cut);
- `s_i` = the best switch against the nearest loop on each adjacent plane of the
  same axis. This is a **diagnostic lower bound** on connectivity, not part of
  the stable test: on a multi-loop plane the nearest centre may be the wrong
  component, and step 2 searches all components properly.

### Selection rule

```
stable  if rms_i <= --rms   (default 0.025)
```

0.025 is 1.2% of the hand's longest extent and about the thickness of the slice
band the loops were fitted to: a stable loop is one whose typical point is
within the band it came from. The plan also writes the counts for 0.015, 0.020,
0.022 and 0.030 so the cut can be moved without rerunning anything else.

Current measurement with the default cut (see Appendix):

- 48 stable: x 5, y 25, z 18;
- 30 rejected: x 11, y 5, z 14.

The rejected x loops are the large palm cross-sections, whose RMS is 0.026–0.082;
the accepted x loops are `x -0.28, -0.18, -0.08, -0.03, +0.38`. This is an
expected and reportable consequence: the x family is seeded thinly.

### Artifacts

| File | Shows |
|---|---|
| `step1/loops.csv` | one row per loop: plane, axis, offset, component, n, rms, q90, cov, radius, rms/radius, s, stable |
| `step1/loops.json` | the same with the exact per-loop numbers |
| `step1/rms_distribution.png` | histogram of all 78 RMS values with the cut, plus a per-axis strip plot (x = offset, y = RMS, one marker per loop) |
| `step1/thresholds.png` | stable count and rejected count against the cut, 0.010–0.040 |
| `step1/stable_montage.png` | every stable loop drawn over its own slice points, in plane coordinates, titled with RMS; 8×6 grid |
| `step1/rejected_montage.png` | the same for every rejected loop, with the reason (RMS above the cut) |
| `step1/rms_vs_s.png` | RMS against the diagnostic `s`, with the stable cut and the `s = 0.3` line |

The two montages are the "did the screen pick the right loops" check. The
`rms_vs_s` panel is expected to show the tension already measured on this run:
the smallest digit loops have the best RMS, while the large palm loops that
transfer best have merely moderate RMS. That is why step 2, not step 1, decides
what survives.

### Expected result

The band `y -0.59 … -0.36` is stable at 0.020–0.024 and seeds the palm track.
`y -0.71, -0.24, -0.12` fall just above the cut and can still be adopted in
step 2 if a seed reaches them. Most fingertip components are stable.

---

## Step 2 — propagate, keep or discard

### Graph

Nodes are `(plane, component)` — all 78, stable or not. Edges are possible only
between adjacent plane offsets of the same axis (the nearest offset above and
below), and only while at least one endpoint is active. Active starts as the
stable loops from step 1 and grows as targets are adopted.

This makes every track start from a stable loop, while allowing a stable loop
to adopt a neighbour that failed the step-1 screen.

### Is the seed's shape transferable? (one edge)

The seed is not copied onto the target; the test asks whether its **shape**
explains the target's points at the seed's **actual physical position**. Each
active node acts as the seed for its untried side, so a loop adopted in one
step becomes the origin of the next:

1. project the seed and the target's labelled points into the target loop's
   plane frame (signed normal, shared `e1`);
2. place the seed at its own 3D position projected into that frame — no free
   re-centring — and search only a small local correction, coarse then fine,
   with `|shift| <= cap = clip(0.18 * min(radius_seed, radius_target), 0.01,
   0.06)`;
3. score two directions at `tau = 0.02`:
   - `cov_pt` = fraction of the target's points within `tau` of the shifted seed;
   - `cov_loop` = fraction of the seed's samples within `tau` of the target's points.

`cov_pt` alone would accept a small fragment inside a big loop; `cov_loop`
alone would accept a partial arc that the seed overruns. Both must pass.

This replaced the first implementation, which searched a coarse ±0.2 / fine
±0.03 translation. That was wrong on this hand: a free translation can re-centre
a loop onto any similarly sized component, and the far-plane null then looked
"better" than the adjacent plane. Physical placement plus a bounded local shift
is the actual propagation question — does this loop continue *here* — and the
accepted shifts are almost all below 0.02.

### Far-plane null: diagnostic only

For every seed, the script still repeats the search against components on
planes at least three offsets away, same axis, and writes `step2/null.png` and
`null.q95` into `tracks.json`. On this run those nulls are all self-similar
(`q95` 0.52–0.84 in point coverage, 0.67–0.84 in loop support), because far
planes in a tubular hand really do contain the same section: the palm repeats
along `y`, the digits along `z`. A null built by translating a loop onto far
planes therefore measures the hand's self-similarity, not chance, so it cannot
set the acceptance threshold. It is kept as a diagnostic only.

Acceptance is consequently a fixed two-sided threshold of `0.5`, chosen with
the bounded shift as a conservative "more than half of both directions must be
explained" rule. It is a design parameter, not a calibrated p-value. The check
that the accepted edges are real is visual and geometric: every accepted edge
is between adjacent planes, every kept loop has a logged edge, and the step-4
surfaces are reloaded and measured.

### Assignment and tracks

- Accept an edge when `min(cov_pt, cov_loop) >= --accept` (default 0.5); log
  the shift, scores and the far-plane null.
- A node may keep at most one accepted successor on each side; if several pass,
  keep the highest `min(cov_pt, cov_loop)` and log the others as `superseded`.
- Grow breadth-first from all stable seeds until no edge is added.
- Tracks are the connected components of the accepted graph. A node never has
  more than two edges, so each component is a chain.
- A track must contain at least one stable seed (true by construction) and at
  least two loops to continue.

Discard categories, reported separately:

| Category | Meaning |
|---|---|
| `stable_kept` | stable loop adopted into a track |
| `stable_discarded` | stable loop with no accepted edge — cannot be propagated |
| `adopted` | unstable loop reached by a track |
| `orphan` | no accepted edge and not stable |

The propagated loop that stays is the target's **own fitted loop**, not the seed
translated. Copying the seed would make step 3 trivial and would force a
constant cross-section onto a hand that tapers. The seed's translated curve is
evidence for the link; the local fit is the geometry that continues.

### Artifacts

| File | Shows |
|---|---|
| `step2/attempts.csv` | every tested edge: seed, target plane/component, shift vector and norm, cov_pt, cov_loop, pass/fail and reason |
| `step2/tracks.json` | accepted edges, track ids, node lists, discard categories, per-track length, axis and offset span, far-plane `null.q95` |
| `step2/null.png` | null distributions against the accepted scores, per axis (diagnostic) |
| `step2/decisions.png` | montage: accepted edge, rejected edge (point coverage), rejected edge (loop support), superseded edge — target points, shifted seed, scores |
| `step2/tracks.png` | 3D loops coloured by track, stable seeds marked, orphans grey |
| `step2/tracks_strip.png` | offset on x, track id on y, one marker per loop; a track is a contiguous row of markers |
| `step2/summary.txt` | counts by category and axis, plus the diagnostic null |

### Result

- 138 edges tested, 37 accepted, 13 tracks.
- Stable loops: 39 kept, 9 discarded. Unstable loops: 11 adopted, 19 orphan.
- The tracks are contiguous in offset and every track contains a stable seed.
  The `y` palm band is track 3 (`y -0.709 … -0.239`, 5 loops); the main slab
  is track 4 (`y -0.004 … +0.232`, 3 loops). Several distal `y` tracks
  (t6–t9) and `z` tracks survive that the earlier free-translation probe
  predicted would stop; with physical placement they are genuine repeated
  sections, not re-centred matches.
- The two `x` tracks (t1, 2 loops; t2, 6 loops) are propagated here but every
  one of their pairs is cut in step 3, so they produce no tube.

---

## Step 2b — propagate with a bounded deformation

### Method

Step 2 asks whether the seed explains the neighbour without changing shape.
Step 2b asks whether it explains it after a *small* change. The difference is
the acceptance budget, which is what keeps it from becoming the free Fourier
re-fit that explains everything.

For one seed/target pair:

1. project the seed into the target plane and search the same bounded local
   shift as step 2;
2. sample the seed's in-plane normal `n(u)` and expand the normal offset as

   ```
   d(u) = c0 + sum_{k=1..M} (a_k cos 2*pi*k*u + b_k sin 2*pi*k*u)
   ```

   with `M = --deform-modes`. The constant `c0` is a taper/dilation; without
   it the fit can bend a section but not shrink it;
3. run robust ICP: match each deformed seed sample to its nearest target
   point, weight the match by a Tukey kernel, and solve the linear
   least-squares system for `(c, shift)`. The discovery radius starts at
   `1.5 * q90` of the seed's current distance to the target, capped at half the
   section radius, and shrinks to `tau`; the normals stay those of the seed, so
   the model is linear in the coefficients. Higher Fourier modes carry a light
   smoothness penalty;
4. measure `deform_rms` (RMS normal offset) and `deform_rel = deform_rms /
   min(r_src, r_tgt)`;
5. admit the edge when the **rigid** solution passes the two-sided coverage
   test, **or** when the deformed solution passes it and
   `deform_rel <= --deform-budget`. If the rigid solution passes, it is used
   and the deformation is reported only as a candidate; the deformation is a
   rescue for pairs rigid cannot explain.

The target's own fitted loop always continues the track. The deformed seed is
never copied into the track, so step 3 still receives real fitted geometry.

### Artifacts

The same tables and figures as step 2, written under `step2b/`, with these
extra columns in `attempts.csv`:

| Column | Meaning |
|---|---|
| `rigid_cov_pt`, `rigid_cov_loop` | the zero-deformation scores |
| `deform_rel` | RMS normal offset of the deformed candidate, as a fraction of the smaller section radius |
| `used_rel` | the offset actually used (0 when the rigid solution was used) |
| `deform_used` | `rigid` or `deformed` |
| `admissible` | rigid passed, or the deformed candidate passed within the budget |
| `n_match` | matched seed samples in the last ICP iteration |

`decisions.png` now draws the used curve (rigid or deformed) and prints the
used deformation, so an accepted edge can never show a deformation above the
budget.

### Result

At the default budget 0.15: 149 edges tested, 52 accepted, 13 tracks, 42 stable
kept, 23 adopted, 7 orphan; 13 accepted edges needed a deformed seed and 2
were rejected only by the budget. The added edges are local continuations that
rigid narrowly missed, e.g. `y -0.83 -> -0.71` has rigid point coverage 0.48
and loop support 0.60, deformed 0.84 / 1.00 at 0.095. The budget-rejected
candidates are 0.264 and 0.419, far from the accepted 0.143 maximum. Steps 3–4
then give 9 tubes / 30 loops instead of 8 / 26.

---

## Step 3 — align the kept loops

### Input

Each track from step 2, ordered by offset. Minimum two loops.

### Method

Reuse the existing alignment module unchanged:

```python
x_i   = resample(loop_i, K=256), centred at its centroid, cut axis zeroed
fits  = candidate_pairs(centers, offsets, neighbors=1)
delta, D, sigma2 = pair_residual(x_i, x_j)
s, E  = switch(D, lam=0.02)
alpha, diag = solve_offsets(...)          # via synchronize(...)
```

Before any of that, every loop is canonicalized: its 2D signed area in the
plane frame is forced positive (reverse the block if it is negative). Without
this, a chain can contain loops traversed in opposite directions; linear
interpolation between them then collapses the section (observed: a tube whose
area fell to 0.005 between loops with areas -0.097 and +0.108). After the fix
the same pair keeps area 0.014–0.031.

`synchronize(loops, centers, offsets, lam=0.02, neighbors=1, min_switch=0.3)`
does the three steps at once. `s >= 0.3` keeps an edge; below that it is a cut.
After solving, split a track at every cut pair: alignment failure is a boundary,
not a reason to throw away the rest of the track.

Report per pair: `delta`, `D`, `s`, kept/cut. Report per track: seam residual
RMS, `alpha` range, number of cuts, `v` span from `loop_coordinates(centers)`,
mean and max centre step per unit offset, and **axis drift**:

```
drift_k = |centre_{k+1} - centre_k|_in-plane / |axis component|
```

This is the tube-eligibility test. A generalized cylinder's section centre
should advance along the section normal; all chains here have drift ≤ 0.47
except c7 at 4.06, which is a sliver whose centre slides in-plane. The drift is
recorded in step 3 and enforced at step 4.

### Artifacts

| File | Shows |
|---|---|
| `step3/pairs.csv` | one row per consecutive pair: track, loops, delta, D, s, kept/cut |
| `step3/tracks.csv` | one row per aligned sub-chain: loops, axes/offsets, seam RMS, alpha range, v span, min s, axis drift, cuts |
| `step3/pairs.png` | left: `D` against `s` for every pair, with the 0.3 line; right: `s` along each track |
| `step3/seams.png` | `alpha` against offset per track, and the kept/cut switch distribution |
| `step3/aligned_1.png`, `aligned_2.png` | per sub-chain: every loop in a common 2D frame with a **seam dot at sample 0**, before and after the cyclic shift, plus the mean section. The dots are what the cyclic shift moves; the closed curve itself is unchanged by a cyclic shift, so the mean section is the real alignment proof. |
| `step3/summary.txt` | kept pairs, cuts, tracks, drift, sheared chains |

### Result

- 37 pairs, 20 kept, 17 cuts; 9 aligned sub-chains from 13 tracks.
- The `x` tracks are cut entirely (their pair switches sit below 0.3), so the
  `x` family contributes no tube.
- The palm chain c1 (5 loops) keeps all pairs, `s_min` 0.62, seam spread 0.038.
  c8 is a 2-loop `z` chain with only 0.037 span (`short`); c2 has the largest
  seam spread (0.083) but is a large 0.15-area section, so it is relatively
  small.
- c7 (z, loops 34:0, 35:0, 36:1) has drift 4.06 and is marked sheared.

### Does triplane continuity already align the seams?

The fit is invariant to `f(z) ↦ f(z + p_i)` on each loop, so nothing in the
loss ties the network's `z = 0` across loops. A triplane does make adjacent
loops' conditioning features continuous, and that raises a fair question: does
the resulting output vary smoothly enough in `z` that the seams arrive already
aligned? `align_after_training.py` reads this straight off the trained runs.
Every fitted loop's sample 0 is the network's `z = 0`, so for each consecutive
pair inside the step-2 tracks the script measures the best circular shift
`|delta|`, the normalized pair residual with the seams left where the network
put them (`D0`) against the residual after the best shift (`D`), and the offset
spread and post-solve jitter (`phase_rms`, turns) of each chain. The pairing is
the triplane run's step-2 `tracks.json`, so both conditionings measure the same
edges. Runs: `log/cmp_tri_tv01_*` and `log/cmp_emb_cs01_*`; output in
`log/align_after_training_*/`.

| shape | conditioning | pairs | median `|delta|` | pairs ≤ 0.02 | `D0 / D` | median `alpha` spread | `phase_rms` |
|---|---|---|---|---|---|---|---|
| horse | triplane | 105 | 0.0054 | 80% | 1.3 | 0.034 | 0.0042 |
| horse | per-loop code | 105 | 0.0027 | 92% | 1.3 | 0.030 | 0.0022 |
| bunny | triplane | 65 | 0.0428 | 46% | 25.0 | 0.089 | 0.0052 |
| bunny | per-loop code | 65 | 0.0043 | 60% | 2.1 | 0.0037 | 0.0007 |
| armadillo | triplane | 91 | 0.0099 | 76% | 1.3 | 0.026 | 0.0010 |
| armadillo | per-loop code | 91 | 0.0037 | 86% | 1.2 | 0.013 | 0.0006 |
| lucy | triplane | 89 | 0.0143 | 58% | 1.6 | 0.022 | 0.0005 |
| lucy | per-loop code | 89 | 0.0076 | 76% | 1.4 | 0.013 | 0.0005 |

Continuity does not pin the seam. On horse it usually starts close — 80% of
pairs within 0.02 turns — but the tail is real: a z chain reaches 0.083 turns
between adjacent loops. On bunny the median pair is already 0.043 turns apart
and the x chain 7..27 rotates steadily: consecutive sections match to
`D ≈ 0.0005–0.0009` after a shift, yet the network seams sit 0.16–0.19 turns
apart and the solved `alpha` runs from 0.371 to 0.051 along the chain. The
zero-shift residual there is 25× the post-shift residual (median), so a loft
that trusts sample 0 connects a corner to a point 65° around the perimeter.
[seam_jump.png](log/align_after_training_bunny/seam_jump.png) shows one such
pair: same section, seams 0.157 turns apart, matching to `D = 0.00085` once
shifted. A per-loop code is no worse — on all four shapes its median shift is
smaller than the triplane's — so this is a property of conditioning, not of
continuity.

Alignment is therefore still needed, for two reasons. It removes the offset
and the drift (otherwise every loft segment twists by `delta` and the chain by
the accumulated `alpha`; `phase_rms` ≤ 0.0052 after the solve says most of that
is a per-loop offset, not noise). And the same `D`/`s` solve is the geometric
compatibility test that cuts a track where the sections genuinely differ —
feature distance does not separate those cuts (AUC 0.601, "feature proposes,
geometry disposes"). Triplane continuity earns its keep upstream, ranking
propagation edges (AUC 0.967), not at the seam gauge.

The gauge part of that can now be solved during training instead of after it:
`rosa_main.py --seam-align` carries one phase per loop, solves it against the
same neighbour graph on detached curves, and exports the loops already at the
aligned seam ([WORKFLOW.md](WORKFLOW.md) §2.3). The fit is untouched — the
epoch lines are identical to the run without it — and on the trained bunny
checkpoint every neighbour pair reaches its best achievable shift. Step 3 is
still what decides which pairs are the same cross-section and where to cut.

---

## Step 4 — make tubes

### Input

Aligned sub-chains from step 3 with at least two loops.

### Method

Eligibility first: a sub-chain is lofted only when its axis drift is at most 2
(step 3). A sheared sliver is reported in `tubes.json.excluded` with its drift
and does not become a tube.

The verified loft, unchanged in spirit from the whole-hand run:

- `v = loop_coordinates(centers)`, strictly increasing along the chain;
- each loop shifted by `-alpha_i` (its solved seam);
- section at `v` = linear interpolation between the two neighbouring shifted
  loops; centre at `v` = linear interpolation of the two loop centres;
- 3D vertex = centre + in-plane section, with the section added on the two
  non-cut axes;
- `nv = max(8, 6 * n_loops)` sections and `nz = 192` around the loop.

The loop rings and seam connectors are exported in the **same world frame** as
the tube surface: the shifted section plus its own centre. An earlier version
used the centred local section, which piled every ring at the origin (combined
bbox margin 0.0314); the world-coordinate fix drops the margin to 0.0175.

A two-loop chain is a straight loft between the two sections; it is allowed but
flagged `short` when its `v` span is under 0.05.

### Artifacts

| File | Shows |
|---|---|
| `step4/tubes.json` | per tube: track id, loops, offsets, v span, min/mean s, drift, seam RMS, section area min/max/mean, centre-step ratio, bbox; plus `excluded` sheared chains |
| `step4/tubes.csv` | the same as a table |
| `step4/tubes.png` | 3D slice cloud with every tube sampled, loops drawn on the tubes, one colour per tube |
| `step4/sections.png` | cross-sections at 5 positions along each tube, in the tube's own frame, to show taper and twist |
| `step4/tubes.ply` | combined tube surface, one colour per tube, plus the used loops and the seam connectors |
| `step4/tubes_cloud.ply` | all slice points and the tube mesh in one file for overlay |
| `step4/tubes_check.png` | the exported PLY reloaded and drawn from two views |
| `step4/tubes_views.png` | lit six-view sheet of the exported mesh (±x, ±y, ±z) |
| `step4/tubes_context.png` | the mesh with the slice points in +z and +x, the "does it cover the hand" view |
| `step4/summary.txt` | tubes, loops used/discarded, exclusion, reload checks |

### Verification (must pass before reporting)

1. Reload `tubes.ply`; component count equals the number of tubes plus the
   number of used loops plus the number of inter-loop connectors.
2. Each tube's bbox lies inside the slice cloud's bbox expanded by the local
   section radius.
3. Recompute each tube's `v` span and min `s` from the reloaded geometry's
   source JSON; they match `tubes.csv`.
4. Every loop used by a tube appears in exactly one tube; every loop in
   `stable_discarded`, `orphan`, or an excluded sheared chain appears in none.
5. `tubes_cloud.ply` keeps every slice vertex: vertex count equals the slice
   count plus the tube/loop vertex count.

### Result

8 tubes, 26 loops used, 52 discarded (49 unreached + 3 in the excluded sheared
chain). All five checks pass:

- 52 components = 8 tubes + 26 loops + 18 connectors;
- bbox margin 0.0175 past the slice cloud (loop fit deviation, not tube error);
- unique loop accounting, used set equals the aligned set of eligible chains;
- `tubes_cloud.ply` = 196,712 slice vertices + 97,484 tube/loop vertices.

| tube | axis | loops | v span | min s | drift | section area |
|---|---|---|---|---|---|---|
| 1 | y | 5 | 0.483 | 0.62 | 0.19 | 0.281–0.304 |
| 2 | y | 3 | 0.253 | 0.41 | 0.47 | 0.116–0.184 |
| 3 | y | 2 | 0.126 | 0.40 | 0.09 | 0.021–0.025 |
| 4 | y | 2 | 0.114 | 0.35 | 0.10 | 0.012–0.017 |
| 5 | y | 4 | 0.352 | 0.37 | 0.19 | 0.024–0.041 |
| 6 | y | 5 | 0.463 | 0.32 | 0.06 | 0.014–0.031 |
| 7 | z | 2 | 0.037 | 0.33 | 0.47 | 0.024–0.030 (`short`) |
| 8 | z | 3 | 0.068 | 0.31 | 0.30 | 0.027–0.031 |

Tube 1 is the palm sleeve; tube 2 the main slab; tubes 3–6 are `y` sections
through the digits; tubes 7–8 are short `z` sections. There is no `x` tube:
both `x` tracks were cut in step 3.

---

## Reporting and decision points

After each step, the summary is read in this order:

1. **Step 1** — is the stable set believable? If the chosen cut admits loops
   that visibly do not follow their points, raise `--rms`; if it rejects the
   palm band entirely, lower it. The threshold is a parameter, not a result.
2. **Step 2** — are the accepted edges local continuations? The far-plane null
   is expected to be self-similar on this hand and is not a chance model; read
   `tracks_strip.png` for contiguity, `decisions.png` for what the scores mean,
   and `attempts.csv` for every shift. If the accepted tracks jump between
   unrelated components, lower `--accept` or tighten the shift cap.
2b. **Step 2b** — is the deformation budget doing its job? Accepted edges with
   `deform_used = rigid` must have `used_rel = 0`; accepted `deformed` edges
   must have `used_rel <= --deform-budget`. Read the gap between the largest
   accepted `used_rel` and the smallest budget-rejected `deform_rel`. If that
   gap is not clear, lower the budget; if no edge needs the deformation, the
   budget is not buying anything.
3. **Step 3** — are the seams stable? A track with cuts is shortened, not
   rescued. Check `pairs.png` for the gap around `s = 0.3` and `axis_drift` for
   sliver chains.
4. **Step 4** — do the reloaded tubes pass the five checks, and does
   `tubes_context.png` show the surfaces on the hand?

The test is a success if every kept loop has a logged propagation edge, every
discarded loop has a logged reason, every tube chain advances along its section
normal, and every exported tube passes the reload checks. It fails if a tube
uses a loop that no accepted edge reached, or if an accepted tube-chain is a
sheared sheet.

## Storage

Everything is written under `log/loop_propagation/`; the largest files are the
two PLYs and the montages, expected under 20 MB. Nothing in `log/` is removed by
this test.

---

## Appendix — current measurements used to set the defaults

Measured today from the run's PLYs:

- 78 loops; in-plane RMS (loop's own plane, slice points to curve):
  min 0.0054, 25th 0.0117, median 0.0218, 75th 0.0346, 90th 0.0528, max 0.0904.
- Counts at candidate cuts: <=0.012: 22, <=0.015: 31, <=0.020: 38,
  <=0.022: 39, <=0.025: 48, <=0.030: 52.
- Stable at 0.025: 48 (x 5, y 25, z 18); rejected 30 (x 11, y 5, z 14).
- Palm band RMS: `y -0.71` 0.0265, `-0.59` 0.0200, `-0.47` 0.0231, `-0.36`
  0.0238, `-0.24` 0.0370, `-0.12` 0.0324.
- Diagnostic best switch `s >= 0.3` for 20 of 78 loops; among the 48 stable
  loops, 10. This is the reason step 1 does not use `s` as a filter and
  step 2 searches all components rather than the nearest centre.
- Earlier transfer probe (leave-one-out, adjacent/short gaps, free translation,
  tau 0.02): point coverage 0.59–0.66 and loop support 0.69–0.75 for
  `y 16–21` (offsets -0.71 … -0.12); 0.58/0.58 at `y -0.00` (the main slab);
  0.45/0.52 at `+0.11`; 0.34/0.40 at `+0.23`; 0.32/0.30 at `+0.35`; loop
  support 0.13–0.21 at `+0.47 … +0.82`. This probe assumed free translation
  and is retained only as history; the implemented test places the seed
  physically, and with that correction the distal `y` tracks do propagate.
- Whole-hand single-loop baseline: 3 tubes, 11 loops, 12 single-loop planes
  outside (10 failed a pair switch, 2 had no chain partner). This test should
  be compared against that baseline, not against zero.

Prior artifacts: `log/align_loops/hand_tubes.png` (baseline tubes),
`log/align_loops/loop_explain_neighbors.png` (transfer evidence),
`log/align_loops/loop_stability.json` (per-loop RMS in the loop's own plane,
plus the diagnostic best switch used above). `loop_rms.json` is the earlier
slice-frame variant and its numbers are not the ones quoted here.
