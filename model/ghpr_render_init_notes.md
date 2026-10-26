# GHPR inside-out initialization for adaptive render-loss runs

Date: 2026-10-26. Branch: `renderloss`. Related: `model/ngf_rendering_loss_notes.md`.

## Why try a GHPR start

Adaptive render runs start from a box pretrained against a reference box. With
Chamfer removed, the unweighted render L1 has an **empty-frame basin**: if the
initial sheet leaves the 120-view raster, the render loss returns a constant
(target-only cost), its gradient vanishes, and the delayed tangent/SVD terms
cannot pull the sheet back. Measured symptom: Chamfer explodes to 4-18 within
100 epochs while R locks to a repeating constant (≈0.069 on armadillo), then
every render is blank.

`--ghpr_init` starts inside the object instead:

1. pick an interior viewpoint (centroid when the inside test passes, otherwise
   step inward from the nearest surface point);
2. place the pretrain box at that viewpoint, sized to the distance of the
   nearest visible point (the inscribed radius);
3. optionally fit GHPR-visible subsets for `--visible_epochs` before the full
   cloud.

## What the switch actually changes under render supervision

Render targets are built from the **full** cloud, and these runs use
`--gamma 0` (normal loss off). Therefore:

- The GHPR-visible stages do **not** change the surface fit; they only change
  the logged Chamfer target and the `stage=` bookkeeping.
- The real effect is the **box placement and size**. Measured start boxes:

  | shape | interior visible | start radius | AABB half-extents |
  |---|---|---|---|
  | hand | 29.4% | 0.00503 | 0.60 / 1.00 / 0.50 |
  | armadillo | 41.4% | 0.11891 | 0.84 / 1.00 / 0.76 |
  | isidore | 55.1% | 0.21114 | 0.99 / 1.00 / 0.65 |
  | elephant | 49.1% | 0.13382 | 0.72 / 1.00 / 0.60 |

  The hand value was verified against the cloud: its centroid is 0.0037 from
  the nearest surface point, which is why the box starts so small.

  On thin or concave shapes the centroid can sit close to the surface, so the
  start box is very small (hand). It still grows under render supervision, but
  more slowly.

## Phase 9: matched pairs with the validated objective

6000 epochs, `--render_laplacian_weight 0.0`, `--lambda_ext_area 0.0`,
conformal SVD delayed to epoch 3000, μ delayed to 1000, AABB vs GHPR with the
same seed. Checks via `render_check.py` (16 views; metrics on the exported
sheet).

| run | seed | patches | L1 | normal / position | Chamfer | coverage | normals med/p95 | UV p95 |
|---|---|---|---|---|---|---|---|---|
| hand_base | 41 | 246 | 0.0165 | 0.0249 / 0.0082 | 0.0160 | 0.989 | 3.1° / 22.4° | 3.78 |
| hand_ghpr | 41 | 225 | 0.0159 | 0.0238 / 0.0080 | 0.0173 | 0.991 | 2.6° / 16.9° | 3.21 |
| isidore_base | 41 | 87 | 0.0755 | 0.0968 / 0.0543 | 0.738 | **0.000** | 89.8° / 90.0° | 11.17 |
| isidore_ghpr | 41 | 219 | 0.0148 | 0.0226 / 0.0071 | 0.0130 | 0.985 | 5.0° / 34.1° | 4.96 |
| elephant_ghpr | 31 | 174 | 0.0129 | 0.0195 / 0.0063 | 0.0145 | 0.964 | 3.6° / 30.0° | 5.82 |
| armadillo_ghpr | 41 | 306 | 0.0178 | 0.0262 / 0.0094 | 0.0180 | 0.979 | 6.2° / 30.8° | 4.58 |

Reference points:

- `elephant_base` (phase 8, seed 31): L1 0.0134, Chamfer 0.0132, coverage 0.979,
  UV p95 5.07.
- `armadillo_svdfast` (phase 8 rescue, SVD from epoch 0): L1 0.0186,
  Chamfer 0.0170, coverage 0.978, UV p95 3.78, 195 patches.
- Standard-schedule armadillo collapsed on 9/9 attempts (unseeded, seed 7,
  seed 32, seeds 41-48); seed screen at epoch 300 gave Chamfer 4.1-17.8.

Findings:

1. **GHPR prevents the empty-frame collapse.** Armadillo now passes on the
   validated schedule, so the SVD-rescue is no longer required for it, and
   isidore seed 41 passes where the matched AABB run collapses to zero
   coverage.
2. **On shapes that already work, GHPR is neutral on L1** (hand 0.0159 vs
   0.0165) and slightly better on normals and UV spread.
3. Elephant is unchanged within seed noise.
4. Caveat: one seed per pair. The hand/isidore seeds are arbitrary, and Chamfer
   is logged only, not optimised.

## Phase 10: GHPR with the conformal term removed

Question: was the rescue due to the inside-out start alone, or does it still
need the conformal regularizer? Command is phase 9 minus the SVD term
(`--lam_svd 0.0`, same seeds, same everything else); launcher:
`scripts/run_ghpr_nosvd_tests.sh`.

All four runs finished (6000 epochs, SVD term confirmed off: `SVD=0.00000`,
`λsvd_eff=0.000`). Final check metrics:

| run | patches | L1 | Chamfer | coverage | normals med/p95 | UV p95 |
|---|---|---|---|---|---|---|
| armadillo_ghpr_nosvd | 273 | 0.0182 | 0.0144 | 0.989 | 5.2 / 31.5° | 0.042 |
| isidore_ghpr_nosvd | 216 | 0.0147 | 0.0120 | 0.988 | 3.9 / 28.0° | 0.041 |
| elephant_ghpr_nosvd | 189 | 0.0135 | 0.0119 | 0.988 | 2.9 / 28.4° | 0.041 |
| hand_ghpr_nosvd | 216 | 0.0162 | 0.0162 | 0.991 | 2.4 / 71.9° | 0.041 |

No collapse on any shape. Removing the conformal term does not hurt the
surface fit: L1 and Chamfer stay within seed noise of the phase-9 GHPR runs
(e.g. armadillo 0.0182/0.0144 vs 0.0178/0.0180; isidore 0.0147/0.0120 vs
0.0148/0.0130; elephant 0.0135/0.0119 vs 0.0129/0.0145). The hand's normal p95
rises to 71.9° (16.9° with SVD), concentrated at the open wrist rim rather than
the fingers, which is the one visible cost of dropping the term.

The term does show up in the exported UV geometry. Mean conformal distortion
over the final sheets, measured from the checkpoint (`patch_vis`), roughly
doubles without SVD training: armadillo 0.0277 vs 0.0121, isidore 0.0237 vs
0.0125, elephant 0.0247 vs 0.0109, hand 0.0154 vs 0.0050. So the conformal
penalty still buys flatter patches; it is simply not needed to prevent the
empty-frame collapse once the start box is inside the shape.

## Phase 11: robust clipping of the render loss

Hypothesis: the same UV sheet is supervised by several orbit views, and a few
views (or view regions) with large losses pull the sheet into a fold.  Tested
on elephant, the worst healthy shape in the foldover audit (phase 10:
4.53% of samples at delta=0.004).

View-level hypothesis does not hold.  On the phase-10 final mesh the per-view
full six-channel L1 spans 0.0106-0.0160 (max/median 1.18); at epochs 1000 and
3000 the spread is 1.28 and 1.36.  No view ever approaches a 2x median outlier,
so a per-view cap at 2/3/5x is inactive: the view-clip control logged
`vclip=0%` at every logged epoch from 100 to 1700.

The heavy tail is spatial, not per-view.  On covered pixels of the final mesh
the pixel error has median 0.0069, mean 0.0631, p90 0.170, p99 0.847; 33% of
covered pixels exceed 3x the covered-pixel median and that tail carries 94%
of the total render loss.

Two opt-in flags were added; both default to 0 and leave the original NGF L1
untouched:

- `--render_view_clip F`: cap each view mean at F x the batch median view
  error.  Inactive on this shape (see above).
- `--render_pixel_clip F`: cap each covered pixel's six-channel mean error at
  F x the median covered-pixel error; background stays zero.

Elephant, 6000 epochs, GHPR init, `--lam_svd 0.0`, seed 31, pixel clipping:

| run | px clipped | patches | fold@.002 | fold@.004 | L1 | Chamfer | coverage | normals p95 |
|---|---|---|---|---|---|---|---|---|
| phase 10 baseline | 0% | 189 | 1.37% | 4.53% | 0.0135 | 0.0119 | 0.988 | 28.4 deg |
| pixel clip x2 | 35-42% | 138 | 0.00% | 0.46% | 0.0219 | 0.0352 | 0.801 | 60.9 deg |
| pixel clip x3 | 27-38% | 153 | 0.05% | 1.56% | 0.0201 | 0.0334 | 0.812 | 55.3 deg |
| pixel clip x5 | 0-29% | 177 | 1.09% | 4.86% | 0.0180 | 0.0230 | 0.887 | 53.5 deg |

The dose response is real and already visible at epoch 2000 (baseline 4.25%
at delta=0.004; clip x2/x3/x5: 0.89/1.74/2.50%).  The x2 and x3 runs keep folds
down to the final checkpoint, but they pay for it: coverage drops from 0.988 to
0.80-0.81, cloud->sheet error rises from 0.0066 to 0.029-0.030, and the
rendered normal p95 doubles.  Clip x5 is mild enough to keep coverage 0.887
but no longer removes folds (4.86%).

Why clipping is not a free win: the folds are interior, same-orientation double
layers.  At delta=0.004 the twin normals on the worst patches have cosine
+1.00 (patch 60/61/62/134), so this is not a thin two-sided sheet and not a
patch-rim flap (hit samples have median UV-boundary distance 0.174).  Once two
coplanar layers coincide, the z-buffer shows one and the hidden layer receives
no gradient; flattening the loss tail removes the pull that creates the
duplicate, but also removes the gradient that refines coverage.  Hard clipping
therefore trades folds for holes.

Recommendation: keep the flags off by default.  If fold suppression is needed,
prefer a soft cap (or clip only the early epochs) and combine it with a
topology/coverage term; the plain hard cap is too blunt.

## Reproduction

```bash
scripts/run_ghpr_tests_nolap.sh     # phase 9 matched pairs
scripts/run_ghpr_nosvd_tests.sh     # phase 10, conformal term removed
scripts/screen_armadillo_seeds.sh 41 42 43 44 45 46 47 48   # seed screen
scripts/probe_armadillo_stabilizers.sh                     # stabilizer probe
scripts/run_armadillo_svdfast_nolap.sh                     # SVD-from-0 rescue
scripts/run_elephant_armadillo_retry_nolap.sh              # plain rerun
scripts/run_viewclip_tests.sh                             # phase 11 view cap
scripts/run_pixelclip_tests.sh                            # phase 11 pixel cap
scripts/check_foldover.py logs/...                        # fold audit
```

## Practical guidance

- For a collapse-prone shape, try `--ghpr_init` before changing the loss
  schedule: it is an initialization change, not a new objective term.
- Watch the printed start radius. Very small radii (thin shapes) grow slowly;
  check R at epoch 100-300 rather than waiting for the final loss.
- Do not attribute behaviour to the GHPR stages under render supervision. Only
  the box placement matters until a per-stage target is wired into the render
  loss.
