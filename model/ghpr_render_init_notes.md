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

Status at this commit: runs started 15:24, armadillo and isidore in progress,
hand and elephant queued. **Provisional** readings (not final):

| run | epoch | R | Chamfer | patches |
|---|---|---|---|---|
| armadillo_ghpr_nosvd | 2200 | 0.0281 | 0.0343 | 225 |
| isidore_ghpr_nosvd | 2200 | 0.0162 | 0.0286 | 198 |

Neither shows the collapse signature (no frozen R, no exploding Chamfer, SVD
term confirmed off in the log). Final metrics will be added after the run
finishes.

## Reproduction

```bash
scripts/run_ghpr_tests_nolap.sh     # phase 9 matched pairs
scripts/run_ghpr_nosvd_tests.sh     # phase 10, conformal term removed
scripts/screen_armadillo_seeds.sh 41 42 43 44 45 46 47 48   # seed screen
scripts/probe_armadillo_stabilizers.sh                     # stabilizer probe
scripts/run_armadillo_svdfast_nolap.sh                     # SVD-from-0 rescue
scripts/run_elephant_armadillo_retry_nolap.sh              # plain rerun
```

## Practical guidance

- For a collapse-prone shape, try `--ghpr_init` before changing the loss
  schedule: it is an initialization change, not a new objective term.
- Watch the printed start radius. Very small radii (thin shapes) grow slowly;
  check R at epoch 100-300 rather than waiting for the final loss.
- Do not attribute behaviour to the GHPR stages under render supervision. Only
  the box placement matters until a per-stage target is wired into the render
  loss.
