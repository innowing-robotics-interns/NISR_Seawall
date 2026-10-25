# Render-loss follow-ups

State: the four-shape conformal batch is finished and checked
(`logs/render_phase4_conformal/check/render4_conformal_summary.md`). The two
open questions below are the next tasks. Numbers are from the final checkpoints
unless noted; render L1 is on the training orbit views, Chamfer is held out.

## Task 1 — Why the teapot handle is missing

### What is measured

- Handle ROI (`x < -0.55`, 17,715 points = 8.9% of the cloud):
  **39.9%** within 2% of the sheet with collapse SVD, **39.1%** with conformal.
  Median distance 0.083 → 0.092, p95 0.312 → 0.311.
- The sheet's most negative x is **-0.642** (collapse) / **-0.614**
  (conformal). The handle tip is at **x = -0.900**. The sheet reaches the
  handle base/attachments only.
- Spout ROI (`x > 0.55`): **87.2% → 87.7%** within 2%, median 0.006.
  Body: **95.8%** both runs.
- Over-time sweep (collapse checkpoint series): handle coverage never exceeds
  ~39%. The sheet's minimum x is already -0.587 at epoch 1000 and stays near
  -0.64. The spout, by contrast, retracts to x ≈ 0.60 at epoch 1000 and grows
  back to ~1.0 by epoch 3000. The handle is lost in the box → body transition
  and never regrows.

### What is ruled out

- **Supervision.** The handle is front-most in **100/120** training views
  (1,220 px/view, **8.94%** of foreground pixels); the spout is 106/120
  (8.10%). It is not a view-coverage blind spot.
- **Leaf allocation.** In the collapse run, 37 of 471 leaves touch the handle
  (depths 3 and 4), so the quadtree does spend capacity there.
- **Conformal.** The handle result is unchanged by the conformal SVD term
  (39.9% → 39.1%). Conformal energy `(E+G) - 2*sqrt(EG-F^2) = (s1-s2)^2`
  penalizes shear, not missing surface, and is exactly zero at a collapsed
  Jacobian.
- **Extension masking.** The image-space render L1 cannot be masked; only
  per-sample tangential/normal terms honor `no_fit`. Extension masking covers
  at most 5.3% of target samples (epoch 2500) and 0.03% by epoch 4500.

### Working explanation (hypothesis, not yet proven)

The teapot's handle is a **thin tubular loop detached from the body**, so a
single box-initialized sheet must leave the wall, turn, and come back around a
hole. The render L1 does supply a signal there, but:

1. the miss costs only ~9% of the foreground, so the body term dominates;
2. producing the loop is a high-curvature, high-area deformation, while the
   Laplacian weight 1.0 and the box prior favor a smooth convex body;
3. nothing in the current objective rewards creating new sheet boundary or
   wrapping a hole. Subdivision only refines an existing patch; it cannot
   create a protrusion.

The spout is a wider, simpler protrusion and is reconstructed (87%), and the
botijo (overall coverage 0.996) reconstructs its broad strap handle, which is
consistent with "thin loop" rather than "handles in general" being the failure.

### Next checks

1. Instrument the render-L1 gradient on the handle ROI in the first 1000
   epochs: is there a net outward pull at the attachment, or only a weak
   tangential signal?
2. Ablate `--render_laplacian_weight` (1.0 → 0.0 → 0.3) on teapot only. If the
   handle still fails, smoothness is not the blocker.
3. Test a mechanism that can create surface: extension-membrane reward, a
   second initialized sheet/leaf aimed at the handle, or a box initialized with
   a prong into the handle region. Keep the published loss terms unchanged and
   make any added term an explicit, opt-in ablation.
4. Map the extension openings at epochs 1000–4000 to the handle; current
   diagnostics do not show whether the handle ever forms an unpaired opening
   the sheet could wrap.

Acceptance: handle coverage > 80% at the 2% threshold with body ≥ 95% and full
render L1 ≤ 0.02, plus a gallery where the loop is visible from the side.

## Task 2 — Why distortion is clearly visible on the rocker arm

### The metric

`aspect` is the ratio of the largest to the smallest singular value of the
surface Jacobian in each checker cell: `s1/s2` from the first fundamental form.
1.0 means an isometric (conformal) map; large values mean one UV direction is
compressed in 3D. It is not a geometric error metric. `checker edge p95` is the
overall cell size and is a different quantity.

### What is measured

| rocker arm | collapse SVD | conformal SVD |
|---|---|---|
| global UV aspect p95 | **152.2** | **15.8** |
| worst per-patch aspect p95 | 2168 (patch 382, depth 3) | 1767 (patch 137, depth 3) |
| active leaves | 849 | 330 |
| checker edge p95 | 0.034 | 0.033 |
| full render L1 | 0.0118 | 0.0119 |
| held-out CD | 0.0278 | 0.0148 |

The global max is dominated by degenerate cells where `s2 → 0` (2.7e9
collapse, 4.4e9 conformal) and should not be read alone; p95 is the robust
number.

### Why it is most obvious on the rocker arm

1. **The shape is the most elongated/thin of the four.** The rocker has a long
   lever arm, a thin shell and small hole rims. Quadtree leaves are **square in
   UV**. Covering a long, thin 3D band with square UV domains forces one
   direction to compress, which is exactly large `s1/s2`.
2. **Nothing penalized shear in the collapse run.** `--svd_mode collapse`
   returns exactly zero energy, so the earlier four runs had no SVD term at all.
   The Laplacian penalizes bending, not anisotropy; the render loss only cares
   about the image. Distortion-driven subdivision reacted by splitting to 849
   leaves, but square leaves cannot remove the anisotropy, so p95 stayed at
   152.
3. **Conformal penalizes exactly this quantity.** Its energy is
   `(s1 - s2)^2`, so it directly pulls the singular values together. That
   explains the 10× drop (152 → 15.8) and the drop in leaves (849 → 330):
   fewer splits are triggered because the distortion that triggered them is
   removed. Fit is unchanged (L1 0.0118 → 0.0119) and CD actually halves.
4. **The checkerboard makes it visible.** Checker squares stretch into long
   thin slivers in the collapse run, most clearly in view 0 (upper lever) and
   view 10 (dome/hole rim). In the conformal run the cells are near-square in
   the same views. See the two `checkerboard_all_views.png` links below.

### Open item

A few patches still have extremely anisotropic cells even with conformal
(per-patch p95 up to 1767), while the global p95 is 15.8. Check whether those
cells sit on patch seams, hole rims or near-collapsed folds, and whether they
should be excluded as degenerate. If they are real, the next lever is a
per-cell conformal weight or a seam-aware parameterization, not a larger λ.

Acceptance: global aspect p95 < 10, no checker cell with `s2` below a stated
floor, and a visible checkerboard with square cells on the lever and dome.

## Artifacts

- Conformal summary and table:
  `logs/render_phase4_conformal/check/render4_conformal_summary.md`
- Teapot ROI numbers:
  `logs/render_phase4_conformal/check/diag_teapot_handle/roi_compare.json`
- Teapot uncovered points:
  `logs/render_phase4_conformal/check/diag_teapot_handle/uncovered_only.png`
- Collapse rocker checkerboard:
  `logs/render_phase4/check/rocker_adaptive_render/rocker_adaptive_render/checkerboard/checkerboard_all_views.png`
- Conformal rocker checkerboard:
  `logs/render_phase4_conformal/check/rocker_adaptive_render/rocker_adaptive_render/checkerboard/checkerboard_all_views.png`
- Rocker metrics:
  `logs/render_phase4/check/rocker_adaptive_render/rocker_adaptive_render/metrics/metrics.json`
  and `logs/render_phase4_conformal/check/rocker_adaptive_render/rocker_adaptive_render/metrics/metrics.json`
