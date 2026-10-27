# NGF rendering loss — NISR integration notes

Module: [`ngf_rendering_loss.py`](ngf_rendering_loss.py)  
Branch: `renderloss`  
Status: **wired into `main.py`** for fixed-grid (`single_sheet`, `six_sheet`
with `--no_presplit`) and adaptive cube atlases.

## What is implemented

- `utils/render_mesh.py` samples the atlas into a live-autograd indexed mesh
  (`vertices`, `faces`, area-weighted `vertex_normals`). No CPU copy.
- `utils/render_targets.py` builds NGF-convention orbit cameras and a
  z-buffered point-splat reference of `[normal, position, alpha]` from the
  input cloud in NISR's own normalized frame.
- `model/ngf_rendering_loss.py` rasterizes the mesh with nvdiffrast and returns
  NGF's **unweighted** L1 over all pixels and the six normal/position
  channels. Alpha is ignored. An optional uniform Laplacian is added.
- `main.py` exposes `--lambda_render`, `--render_views`, `--render_batch`,
  `--render_resolution`, `--render_mesh_res`, `--render_laplacian_weight`,
  `--render_min_coverage`, `--render_start_epoch`, `--render_warmup_epochs`
  and `--render_preview`.

The renderer is single-layer. NGF's original mesh loss peels three layers; a
single front layer supervises only the visible surface. For closed shells the
back side gets no render gradient (regularizers still apply).

## Fixed-grid behaviour

The render term is **added** to whatever surface term is configured:

- `--lambda_chamfer 0 --lambda_render 1` is pure NGF (used for the six-sheet
  `g06b` run).
- With both positive, Chamfer and render are optimized together.

Validated on bunny six-sheet (`g06b_box_ngf_120v`, 120 views, batch 10, 256²,
Laplacian 1.0, 2000-epoch box pretrain, 1000 render epochs): render L1
0.392 → 0.053 at 100 epochs → **0.023 at 1000 epochs**.

## Adaptive behaviour

With `--adaptive --lambda_render > 0` the render term **replaces Chamfer** as
the surface-fitting term. Every other term of the previous adaptive design is
kept: μ Dirichlet tangent, λ_svd, γ normal, extension-area, subdivision and
GHPR staging. The flow is the original one:

1. `pretrain_adaptive_box` fits the 24-leaf atlas to the reference box
   (default 500 epochs; box loss reached 2e-6 on Lucy).
2. Render targets are built from the **full** cloud (not a GHPR visible
   subset). `--surface_loss_type chamfer` remains the stored value but CD is
   computed only for logging.
3. The render mesh is resampled from the live atlas every step and rebuilt
   after every subdivision. Under `--render_mesh_res 0` the per-patch grid is
   automatically reduced to stay near 30k vertices.

Because the reference set is the full cloud, GHPR visible-only stages no
longer affect the surface fit; they only change the (optional) normal pool and
the logged CD. Full-cloud AABB initialization (`--box_init aabb`, no
`--ghpr_init`) is the recommended start for render-loss runs.

### The delayed-μ requirement

A plain Dirichlet tangent energy `0.5(|t_u|² + |t_v|²)` is minimized at **zero
scale**. With Chamfer present it is balanced; with render-only supervision and
μ active from the first post-pretrain epoch it wins before the render loss
leaves the box basin. Controlled evidence on bunny:

| run | μ schedule | outcome |
|---|---|---|
| `smoke2` | 0.08 from epoch 1 | collapsed: R→trivial, CD→1.03, mesh span→0.001 |
| `smoke3` | μ=0 | fitted: R 0.266→0.050, CD 0.60→0.15 |
| `smoke5` | 0.08 resumed from fitted `smoke3` | stable: R 0.050→0.038, CD 0.135→0.103 |
| `smoke6` | 0.08 delayed 500, ramp 500 | fitted + 3 subdivisions: R 0.2705→0.0283, CD 0.0266 |

`train_adaptive` now prints a warning when μ would be nonzero at the first
render epoch. Recommended: `--mu_warmup_delay 500 --mu_warmup_epochs 500` at
minimum; the Lucy/dancing runs use `1000/1000`.

### Learning-rate stability (sparse silhouettes)

The empty-frame basin is also reachable by taking too-large optimizer steps.
Controlled Lucy runs from the same 500-epoch box pretrain (24 leaves, 400
epochs, no regularizers, Laplacian 1.0):

| lr | render L1 | Chamfer | coverage |
|---|---|---|---|
| 1e-3 | 0.0393 (floor) | 0.465 | 0.004 |
| 5e-4 | 0.0419 (floor) | 0.562 | 0.000 |
| 2e-4 | 0.0230 | 0.053 | 0.230 |
| 1e-4 | 0.0250 | 0.070 | 0.170 |

With subdivision, 1e-4 for 2500 epochs reached coverage 0.706, Chamfer 0.030,
render L1 0.019 (24→165 leaves, μ delayed 1000, SVD delayed 1500). NGF itself
starts from a decimated version of the object, not a bounding box; the low LR
plays the role of a gentle initialization for the box. `train_adaptive` warns
when `--lr >= 5e-4` is used with a render-only adaptive objective. The full
Lucy/dancing runs use **2e-4**.

### Full box-initialized runs (10,000 epochs)

Two full runs replace Chamfer with render L1 on the adaptive atlas. Both use
the original 500-epoch box pretrain, 24 root patches, 120 views at 256°, batch
10, Laplacian 1.0, the old μ/SVD/γ/extension schedules, and `lr 2e-4` with
μ delayed 1000/ramped 1000 (the only schedule change). Chamfer is logged only.

| run | leaves | render L1 ep1 → final | logged CD | render-check full L1 | held-out CD | coverage | rendered normal med/p95 | visible-layer normal med/p95 | checker p95 |
|---|---|---|---|---|---|---|---|---|---|
| Lucy render | 249 | 0.1449 → 0.0154 | 0.0200 | 0.0170 | 0.0160 | 0.927 | 10.64 / 55.19° | 8.64 / 56.22° | 0.033 |
| Lucy old Chamfer | 150 | n/a | 0.0147 | 0.0180 | 0.0080* | 0.999 | 8.93 / 64.02° | 5.08 / 41.42° | 0.039 |
| dancing render | 369 | 0.2411 → 0.0261 | 0.0308 | 0.0228 | 0.0209 | 0.931 | 5.09 / 41.95° | 3.25 / 23.43° | 0.045 |
| dancing old Chamfer | 726 | n/a | 0.0155 | 0.0270 | 0.0080* | 1.000 | 4.72 / 59.52° | 2.22 / 12.50° | 0.047 |

`*` The old runs optimized Chamfer, so their CD is a fit diagnostic, not a
held-out score. The render runs never optimize it, so 0.0160 / 0.0209 are
genuine held-out values. Image L1 is also not held out for the render runs:
the 12/16 deterministic gallery views are a subset of the 120 training views.
The pictures show a smooth, complete approximation; the old Chamfer runs are
closer in CD and coverage, while the render runs are slightly cleaner on the
training views. At 256² Lucy covers only ~11% of each frame, so an L1-only
objective leaves thin limbs under-resolved even when the image loss is low.

- Lucy check: `logs/render_phase4/check/lucy_render_vs_CD/inspection.md`
- dancing check: `logs/render_phase4/check/dancing_render_vs_CD/inspection.md`

### Adaptive smoke result (bunny, 1500 epochs total)

500 box-pretrain epochs followed by 1000 render epochs, 120→94 views above
min coverage, batch 10, 256², Laplacian 1.0, μ 0.08 delayed 500, λ_svd 0.08
delayed 500, subdivision 24→96→165→192 leaves (auto mesh 32²→17²→13²→12²).

- render L1 0.2705 → 0.0343 at epoch 500 → **0.0283** at 1500
- Chamfer (held out) 0.0215; cloud coverage 0.910; rendered-normal median
  4.71°, p95 31.41°; visible-layer normal median 2.72°
- checker edge p95 0.041, UV aspect p95 3.95
- render check: `logs/render_phase4/check/smoke6_bunny_full/inspection.md`
- gallery predictions are smooth and complete but visibly puffier than the
  target at this budget (1500 epochs).

## Caveats

1. **No alpha mask.** Per the requested original objective, the L1 covers the
   whole frame. Sparse targets therefore penalize geometry where the target is
   empty. Keep cloud coverage high, or use a dense scan.
2. **Render-trivial basin.** Hiding the mesh lowers the render L1 toward the
   empty-frame floor (roughly the mean |target| over all pixels). A low L1 is
   not sufficient evidence; always inspect coverage and the gallery.
3. **One layer only.** The back of a closed shell receives no render gradient.
4. **Seams are duplicated.** Each patch/leaf is sampled as its own grid; seam
   vertices are not welded and neighboring adaptive leaves of different depth
   meet in T-junctions. At 256² the cracks are subpixel in the smoke run, and
   the seam positions themselves stay continuous (`max gap 1.4e-7`), but the
   Laplacian never smooths across a seam.
5. **Laplacian differs from NGF.** NGF smooths a quad patch graph; this port
   applies a uniform triangle-edge Laplacian to every mesh vertex.
6. **Camera/image layout.** nvdiffrast images are bottom-up and the projection
   flips x. Targets are generated by the matching code path, so no external
   flip is needed.
7. **Normalization.** Targets and model use NISR's `center = mean(points)`,
   `scale = max |p − center|`; NGF's bbox-frame maps would misalign.

## Reproduction: adaptive render loss

```bash
PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH \
python main.py --adaptive \
  --file normalized_shapes/bunny_points.ply \
  --result_dir logs/render_phase4/smoke6_bunny_full \
  --N 30000 --M_per_patch 128 --epochs 1500 --pretrain_epochs 500 \
  --lambda_render 1.0 --render_views 120 --render_batch 10 \
  --render_resolution 256 --render_min_coverage 0.05 \
  --render_laplacian_weight 1.0 --render_preview 2 \
  --mu 0.08 --mu_warmup_delay 500 --mu_warmup_epochs 500 \
  --lam_svd 0.08 --svd_mode conformal \
  --svd_warmup_delay 500 --svd_warmup_epochs 500 \
  --gamma 0 --subdiv_threshold 0.1 --subdiv_every 500 \
  --subdiv_start 500 --subdiv_max_depth 4 --mesh_res 16 --box_init aabb
```

The renderer JIT-compiles on first use and needs `ninja` on `PATH`; the
seawall environment provides it.

## Depth-peeled far-layer supervision (opt-in, phase 12)

`--render_layers 2` peels both sides of the supervision:

- The point-cloud reference is splatted in layers: layer 0 keeps the original
  front-surface target; layer 1 is z-buffered only from points the front layer
  did not consume (`utils/render_targets.py`, `_splat_view`).
- The model mesh is peeled per view (`NGFRenderer.render_layers`): after a pass,
  the triangles that won that view's z-test are removed and the remaining faces
  are rasterized again. Peeling per view matters: a batch-wide removal deletes
  triangles that are front-facing in one view but a legitimate back surface in
  another, which halves the usable far-layer coverage.
- Layer 0 keeps NGF's full-frame six-channel L1 and weight 1.0. Deeper layers
  are compared only where the target has a surface (alpha > 0.05) and the model
  rendered one, and the error sum is divided by `B*H*W`, so the term has the
  same units as the front term. Pixels the model leaves empty are not rewarded;
  a full-frame far-layer L1 would instead push triangles to grow into peeling
  holes.
- The term is opt-in (`--render_far_weight`, default 0.5) and every layer is
  logged as `render_layer_N`, with the matched-pixel fraction as
  `render_layer_N_match`. The single-layer phase-10 path is unchanged.

Armadillo baseline (phase 10) at 8 views: target far layer covers 16.1% of the
frame, 59% of its normals oppose the front normal; the peeled mesh reaches
18.6% coverage on layer 2, 15.4% matched. Far-layer pixels on the worst fold
patches (26.5% of layer 2) carry 0.098 error against a 0.089 matched average.
