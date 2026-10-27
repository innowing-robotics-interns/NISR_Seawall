# NGF rendering loss — integration plan

Status: Phases 0–3 are implemented and verified on branch `renderloss`.
Rendering is opt-in (`--lambda_render 0` by default). Phase 3 smoke and A/B
results are below; remaining Phase 4 work is at the end.

Verification artifacts:

- Phase 1–2: [`experiments/render_phase2_verify/`](../experiments/render_phase2_verify/)
  (`summary.json`, target previews, sphere alignment, fitted-model comparison).
- Phase 3: [`logs/render_phase3/`](../logs/render_phase3/) (`ab/summary.json`,
  per-run target previews and final comparisons, `phase4_checks.json`,
  TensorBoard check, overhead runs).

Related:

- [`ngf_rendering_loss.py`](ngf_rendering_loss.py) — the loss and renderer.
- [`ngf_rendering_loss_notes.md`](ngf_rendering_loss_notes.md) — interface and
  compatibility gaps found before integration.
- [`utils/render_mesh.py`](../utils/render_mesh.py) — Phase 1 sampler.
- [`utils/render_targets.py`](../utils/render_targets.py) — Phase 2 targets.
- [`scripts/verify_render_integration.py`](../scripts/verify_render_integration.py)
  — end-to-end check used for the results below.

## Scope

The target is `train_multi_patch` in `main.py`. The same injection point exists
on `main` (`main.py:135`, total loss at `main.py:998`) and on the
`area_min`/`renderloss` checkout (`main.py:207`, total loss at `main.py:1065`).
Only the TensorBoard/extension-area plumbing differs upstream of the loss.

Version 1 is intentionally narrow:

- fixed-grid `MultiPatchForwardMap`, `atlas_mode='single_sheet'`;
- one front layer, matching a point cloud;
- unweighted L1 over the six `[normal, position]` channels — NGF's original
  render term;
- input cloud and reference maps in NISR's own normalized frame;
- rendering disabled by default (`--lambda_render 0`).

Deferred: adaptive quadtree atlases, two-/six-sheet atlases, depth peeling,
alpha-masked loss, and NGF's quad-graph Laplacian.

## Why this shape

NISR optimizes a neural atlas from unstructured per-patch UV samples. The loss
needs an indexed triangle mesh, so the atlas must be sampled into one. The
existing exporter `utils.sample_multi_patch_grid` has the right topology but is
export-only: it calls `.cpu().numpy()` (`utils/utils.py:660-661`), which detaches
the graph, and returns no normals. Phases 1–2 add a GPU sampler and reference
generator without changing export.

Generating both the reference and the mesh from `pts3n` also avoids the NGF vs
NISR normalization mismatch documented in `ngf_rendering_loss_notes.md` §4, and
using NGF's camera convention on both sides avoids the image-layout mismatch in
§10.

## Phase 0 — prerequisites and branch hygiene

- Add `ninja` to `environment.yaml`; keep the editable `nvdiffrast` install
  documented. `nvdiffrast` compiles its rasterizer on first use and needs the
  `ninja` executable on `PATH`. `run_seawall.sh` invokes the environment's
  Python directly, so either activate the environment or prepend its `bin`
  directory before rendering.
- Keep `nvdiffrast` out of the critical import path. The loss module already
  imports it lazily; target generation and mesh sampling are pure torch.
- When rendering is enabled, run one tiny 64² render before training so a
  missing JIT/ninja setup fails immediately with an actionable message.
- The current `renderloss` branch is `main` plus the `area_min` commit plus the
  loss module. For a clean PR against `main`, cut a branch from `main` and
  cherry-pick the loss-module commit; merging `area_min` wholesale is a
  separate decision because it also changes README/TRAINING/run scripts.

## Phase 1 — differentiable mesh sampler

New file: `utils/render_mesh.py`.

### API

```python
layout = get_patch_mesh_layout(patch_ids, resolution, device)
vertices, faces = sample_patch_mesh(F, resolution, patch_ids=active_ids)
normals = vertex_normals(vertices, faces)
```

- `PatchMeshLayout` caches the UV grid, the `(F, 3)` int32 faces, and per-patch
  vertex offsets. Cache key is `(patch_ids, resolution, device, dtype)`;
  topology is built once per training run.
- `sample_patch_mesh` evaluates `F(patch_id, uv_chunk)` in chunks of 4096 on
  the GPU and concatenates. It never calls `.cpu()`, `.numpy()`, or `detach()`.
- Face winding is `[00, 10, 11]`, `[00, 11, 01]`, consistent with NISR's
  `n = t_u × t_v` convention (`model/losses.py:472`).
- `vertex_normals` accumulates area-weighted face normals with `index_add_`
  and normalizes. This is cheap, differentiable, and matches the winding. Seam
  copies get one-sided normals; acceptable for the front-layer render term.
- `auto_mesh_resolution` chooses the largest `R` with `P·R² <= 30000` vertices,
  clamped to `[8, 32]`. For 16 patches this is `R=32`; for 64 patches, `R=21`.

### Boundaries

- Works for `MultiPatchForwardMap` (`model/model.py:661`); technically works
  for any module with `forward(patch_idx, uv)` and `n_patches`.
- Adaptive leaves are not handled: sampling each leaf on its own grid creates
  T-junctions and cracks. Keep the exporter for those atlases.
- Only `active_ids` are sampled during training; inactive patches have no
  target support and should not receive render gradients.

## Phase 2 — orbit cameras and reference targets

New file: `utils/render_targets.py`. Ported from
`ngf_yl/source/pointcloud_render.py` (torch/numpy/scipy only, no cross-repo
import).

### Inputs

Use NISR's `pts3n` and `meta['normals']` (`main.py:3182-3190`) directly. They
are already in the same normalized frame as the model, and normals are already
unit length. Do not re-normalize positions with NGF's bbox normalizer.

### Components

- `PointCloud` + `prepare_point_cloud`: validate shapes and normals, compute
  local spacing with a one-time KD-tree.
- `make_orbit_views`: object-centric Fibonacci cameras, fov 45°, radius factor
  1.08, world-to-camera `(V, 4, 4)` with columns `[right, up, look, eye]`.
- `render_pointcloud`: z-buffered screen-space splat of
  `[normal (3), position (3), alpha (1)]` in NGF's bottom-up image order.
- `build_render_targets`: render, compute per-view coverage, drop views below
  `min_coverage`, split into batches, cache `(V, H, W, 7)` on CPU, and
  optionally save previews.
- `save_preview` / `save_comparison`: normal-shaded target and prediction
  montages with silhouette, for visual frame checks.

### Frame and layout checks

- Both target and mesh render use the same cameras and the same `pts3n` frame.
- A 256² target for 16 views is about 29 MB in float32 on CPU.
- `min_coverage` default 0.1; fail if no view survives. Coverage is the mean
  splat alpha, i.e. a silhouette fraction, so it is shape-dependent. A thin
  shape can legitimately cover ~12% of the frame (`120477`), while the default
  0.5 rejected every valid view of it. The default is therefore a near-empty
  guard, not a "well framed" test; previews remain mandatory.
- Keep alpha out of the loss. It is used only to filter empty views and to draw
  previews, preserving NGF's unweighted objective.

## Phase 1–2 verification results

Run on 2026-10-23 with `seawall` (RTX 4090, torch 2.5.1+cu124, nvdiffrast
0.3.3):

```bash
PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH \
/home/lyangiw/anaconda3/envs/seawall/bin/python \
  scripts/verify_render_integration.py \
  --ply normalized_shapes/stanford-bunny_rotate.ply \
  --out experiments/render_phase2_verify \
  --views 12 --resolution 128 --points 20000 \
  --pretrain-epochs 500 --fit-epochs 2500 --n-patches 25 --mu 0.05
```

**Camera and image-order check (synthetic sphere, 128²).** The target is a
splat of 30k points sampled from an icosphere; the prediction is a mesh render
of the same icosphere. No model fitting is involved, so this measures the
renderer conventions directly.

| metric | value |
|---|---:|
| six-channel L1, all pixels | 0.0243 |
| six-channel L1, shared pixels | 0.0071 |
| mean normal cosine, shared pixels | 0.995 |
| silhouette IoU | 0.954 |

The residual is splat footprint and antialiased edges, not a flip or offset.

**Real-cloud targets.** Bunny cloud, 20k points, 12 views at 128²: coverage
0.239 mean (0.162–0.280), build 0.2 s, 12/12 views kept. A dense build (200k
points, 256²) takes 0.9 s and visibly removes the splat holes. The thin
`120477` shape covers only 0.063–0.145 per view; this is what motivated the
0.1 default.

**Fitted-model render comparison.** The script does a *test-harness* fit, not a
training-recipe proposal: 25 patches (22 active), 500 flat-sheet pretrain
epochs, 2500 Chamfer epochs with `mu=0.05`; about 53 s of GPU work (10 s
pretrain + 43 s fit). It then samples a 32²-per-patch mesh (22,528 vertices,
42,284 faces) and renders the 12 cached views.

| metric | value |
|---|---:|
| mesh sampling | 0.020 s |
| 12-view render | 0.001 s |
| six-channel L1, all pixels | 0.0398 |
| six-channel L1, shared pixels | 0.1004 |
| shared normal L1 | 0.162 |
| mean normal cosine, shared pixels | 0.818 |
| shared position L1 | 0.0390 |
| silhouette IoU | 0.833 |
| target / prediction coverage | 0.239 / 0.215 |

The remaining error includes the rough quick fit, so this is an upper bound on
the renderer's own alignment error. Position error is ~2% of the object extent;
the overlap montage shows the predicted and target silhouettes in the same
pose. The render loss backward reaches both parameter groups:
`vertex_features` grad norm 0.027, decoder grad norm 1.46 across 14 tensors.

Artifacts: `sphere/sphere_montage.png` (conventions),
`targets_dense/targets/montage_normals.png` (200k-point targets),
`compare/fit_montage.png` (target / prediction / overlap),
`fit/patch_assignments.png`, `summary.json` (all numbers above).

## Phase 3 — wiring into `train_multi_patch` (implemented)

Add parameters and CLI flags:

| Flag | Default | Purpose |
|---|---:|---|
| `--lambda_render` | `0.0` | Render term weight; `0` disables everything |
| `--render_start_epoch` | `500` | Delay before the term turns on |
| `--render_warmup_epochs` | `500` | Ramp via existing `mu_warmup_schedule` |
| `--render_every` | `1` | Compute every N epochs |
| `--render_views` / `--render_batch` | `16` / `2` | Orbit cameras / views per step |
| `--render_resolution` | `256` | Target and renderer resolution |
| `--render_mesh_res` | `0` | `0` = auto from Phase 1 |
| `--render_min_coverage` | `0.1` | Drop empty/partial views |
| `--render_preview` | `4` | Target/prediction preview count |

Before the loop: build `NGFRenderer`, cache targets, precompute topology, and
warm up the rasterizer. Inside the epoch loop, after `Q_flat`:

```python
if render_on(epoch):
    verts, faces = sample_patch_mesh(F, render_res, active_ids)
    normals_v = vertex_normals(verts, faces)
    vb, rb = next_view_batch()
    render_loss, parts = ngf_rendering_loss(
        verts, normals_v, faces, vb, rb,
        renderer=renderer, laplacian_weight=0.0, return_parts=True)
    render_lambda_eff = mu_warmup_schedule(
        epoch, render_warmup_epochs, lambda_render,
        schedule=schedule, delay_epochs=render_start_epoch)
else:
    render_loss = zero
    render_lambda_eff = 0.0

loss = (... existing terms ...) + render_lambda_eff * render_loss
```

- Use `laplacian_weight=0`. NISR already has tangent, SVD, extension-area, and
  boundary regularizers. A triangle-mesh Laplacian over duplicate seams is not
  NGF's quad-graph operator (notes §5–6). If full fidelity is wanted later,
  apply a quad-grid Laplacian to area-uniform UV samples.
- No alpha mask and no silhouette term; the render objective stays the
  original six-channel L1.
- Log `history['render']`, TensorBoard `train/render` and
  `render_lambda_eff`, and add the render configuration to checkpoint args.
- Start with `lambda_render=0.1`, not 1.0: a rough normal error of 0.1
  contributes ~0.05 to the six-channel mean, while CD is often ~0.005 in
  normalized units. Calibrate from logged parts so the render term is an
  auxiliary constraint, not the dominant one.
- Cycle deterministically through cached view batches; do not resample cameras
  per step. Resume rebuilds the same targets from the same cloud and config.

### Implementation notes

- All flags are in a dedicated `NGF rendering loss` argparse group. Validation
  rejects `--adaptive`, `atlas_mode != 'single_sheet'`, missing/mismatched
  normals, and out-of-range values before any training starts.
- Targets are built once in `train_multi_patch` before the optimizer loop,
  cached on CPU, and coverage-filtered. `--render_preview` writes the target
  normal/alpha montages to `<result_dir>/render_targets/`.
- The nvdiffrast rasterizer is warmed up with a full-resolution render of the
  current mesh (not a 64² probe), so a missing `ninja`/JIT fails immediately.
- `render_start_epoch` is exclusive: the first active epoch is
  `render_start_epoch + 1`. The weight uses the existing
  `mu_warmup_schedule` with `delay_epochs=render_start_epoch`.
- `history['render']` / `history['render_eff']`, TensorBoard
  `train/render` / `train/render_eff`, the epoch line, and checkpoint args all
  carry the render state. `metadata.json` records the render hyperparameters,
  effective mesh resolution, target-view count, and the render loss/weight
  history. `plot_history` and the final multi-patch figure plot `render` when
  present (the per-batch sawtooth is expected; see Risks).
- Branch housekeeping: `--pretrain_then_train` passed `six_sheet_face_*` to
  `pretrain_multi_patch_flat_sheet`, which does not accept them. That
  pre-existing mismatch (also on `main`) made every flat-pretrain CLI run fail;
  the two unsupported kwargs were removed. `--seed` was added for reproducible
  A/B runs (default unseeded, so existing behavior is unchanged).

### Phase 3 verification (2026-10-23)

Harness: `normalized_shapes/bunny_points.ply` (30k points with normals), 16
patches (15 active), `d_features=64`, `M_per_patch=512`, 300-epoch flat-sheet
pretrain plus 1000 training epochs, `--seed 0`, 16 orbit views at 256²,
2 views/step, auto `R=32` (15,360 vertices). Render runs used
`lambda_render=0.1`, start epoch 100, 100-epoch warmup.

Metrics are post-hoc: rebuild the same reference targets, render the final
mesh over all 16 views, and compare. `render full L1` averages all six channels
over all pixels (background included); `cos` is the normal cosine on pixels
covered by both; `mesh→cloud` is the mean nearest-point distance from the
sampled mesh to the cloud.

| run | γ | λ_render | CD (last) | render full L1 | shared L1 | cos | IoU | mesh→cloud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 0.0571 | 0.1136 | 0.4313 | −0.502 | 0.8166 | 0.0152 |
| render only | 0 | 0.1 | 0.0569 | 0.1129 | 0.4287 | −0.495 | 0.8171 | 0.0153 |
| γ only | 0.5 | 0 | 0.0580 | 0.0550 | 0.1601 | 0.657 | 0.8233 | 0.0113 |
| γ + render | 0.5 | 0.1 | 0.0583 | 0.0539 | 0.1566 | 0.667 | 0.8279 | 0.0112 |

What this says:

- CD is unchanged within 0.5% across the four runs. At this short horizon the
  render term is an auxiliary constraint, not a replacement for Chamfer.
- With γ=0.5, adding the render term improves every render metric slightly:
  full L1 −2.0%, shared L1 −2.2%, normal cosine +0.010, IoU +0.5%. The render
  loss itself falls 10.3% between the first and last full view-batch cycles
  (0.0602 → 0.0540). This is a mechanics/trend result, not a quality claim.
- Without γ, the sheet's winding normals stay anti-aligned with the cloud
  normals (cos ≈ −0.50) and 900 active epochs at λ=0.1 barely move them
  (0.1147 → 0.1130). Run the render term with normal consistency; a warning
  now prints for `--lambda_render > 0 --gamma 0`.

Overhead (300 epochs from random init, 16 patches, `M_per_patch=4096`,
γ=0.5, same seed):

| setting | wall | ms/epoch | overhead |
|---|---:|---:|---:|
| no render | 31.0 s | 103 | — |
| every epoch | 42.0 s | 140 | +36% |
| every 2 epochs | 36.9 s | 123 | +19% |

The fixed cost is ~35–37 ms per active epoch, dominated by evaluating `F` on
the ~15k-vertex render mesh and backpropagating, not by rasterization
(~1 ms/view). At the smoke's small `M_per_patch=512` the same fixed cost makes
the total epoch 3.2× the baseline, which is why the relative overhead depends
strongly on the training batch size. `--render_every 2` meets the 20% budget at production
settings; lowering `--render_mesh_res` is the other lever.

Additional checks (see `logs/render_phase3/phase4_checks.json`):

- Self-render target: loss `1.9e-13`.
- Jittering the mesh by 0.01 / 0.05 raises the loss to 0.0214 / 0.0853.
- 61% of mesh vertices receive nonzero render gradients; `vertex_features`
  and all 14 decoder parameters receive gradients.
- `min_coverage=0.99` raises instead of returning an empty cache.
- TensorBoard exposes `train/render` and `train/render_eff`.

The final figure with the render curve is
`logs/render_phase3/ab/g05_r1/result_with_render.png`; target previews are in
each run's `render_targets/` directory.

## Phase 4 — verification and rollout

`scripts/verify_render_integration.py` currently checks:

1. Sampler: vertex/face counts, finite unit normals, gradients reach
   `F.complex.vertex_features` and the decoder (22,528 vertices / 42,284 faces
   for 22 patches at R=32).
2. Camera/layout: synthetic-sphere target vs mesh render (table above).
3. Real targets: build, coverage filter, previews, dense-cloud variant.
4. Fitted-model comparison: target/prediction L1, silhouette IoU, montages.
5. Render-loss backward on cached views.

Completed during Phase 3:

- Self-render, perturbation, coverage-guard, gradient and TensorBoard checks
  all pass (numbers above).
- 1000-epoch smoke with the render term enabled; finite losses and a
  decreasing render term.
- Seeded 2×2 A/B (γ ∈ {0, 0.5}, λ_render ∈ {0, 0.1}). CD unchanged within
  0.5%; with γ, render metrics improve slightly and the render loss falls
  10.3%; previews align (IoU 0.83). Epoch overhead is +36% at every=1 and
  +19% at every=2 for `M_per_patch=4096`; the 20% budget requires
  `--render_every 2` or a lower mesh resolution.

Remaining before rollout:

- Full-length run with production settings (`M_per_patch=4096`, 5000 epochs)
  on one shape, comparing final CD/render metrics against a seeded baseline.
  The 1000-epoch A/B is a trend check, not a quality verdict.
- A second shape (`120477` or `dancing_children100K`) to exercise partial
  coverage and sparser normals.
- Decide whether any recipe turns rendering on by default (currently every
  recipe keeps `--lambda_render 0`).
- Clean PR branch from `main`: the current `renderloss` branch also carries
  the `area_min` commit and the unrelated pretrain-call fix.

## Phase 5 — deferred work

- Adaptive quadtree: crack-free triangulation or weld plus constrained
  triangulation; hanging vertices need stitching.
- Two-/six-sheet: depth peeling and a matching multi-layer reference; one
  front layer cannot supervise the hidden sheet.
- Alpha-masked loss: opt-in deviation from the original objective for partial
  scans and holey targets.
- NGF-style quad Laplacian over area-uniform samples.
- Learned or coverage-adaptive view sampling.

## Risks

- **Partial point clouds.** The unweighted original loss assumes the target
  observes every surface the mesh renders. A one-sided scan penalizes the
  unseen side toward zero. Filter views, or use masked mode as an explicit
  deviation.
- **Normal orientation.** Wrongly oriented input normals corrupt the target
  directly. Separately, the fitted sheet's winding can stay anti-aligned with
  the cloud: with `--gamma 0` the smoke ended at normal cosine −0.50 and the
  render term barely moved it in 900 epochs. Use `--gamma > 0` (and previews);
  if input normals are suspect, orient them once against the coarse model
  mesh, analogous to NGF's `_orient_normals_to_mesh`.
- **Logging noise.** `train/render` is the loss for the current view batch, and
  batches cycle deterministically, so consecutive logs are not comparable.
  Compare per-batch curves across a full cycle (or average a cycle) rather than
  reading adjacent epochs; `train/render_eff` shows the weight separately.
- **Overhead.** The render term adds a roughly constant ~35 ms per active epoch
  (mesh evaluation + backward). It is +36% at `M_per_patch=4096` every epoch
  and +19% at `--render_every 2`; at small `M_per_patch` the relative cost is
  much larger. Measure on the target recipe before enabling it broadly.
- **Seam normals.** Unwelded boundaries get one-sided normals; check shading
  previews. Weld only if visible.
- **Loss scale.** Render and CD have different natural scales; ramp and log
  both terms separately.
- **Splat density and resolution.** At 20k points / 128² the target shows
  speckle holes; use 100–200k points or raise `radius_scale`. The dense build
  is cheap (0.9 s for 200k points, 12 views at 256²), so prefer dense targets
  when memory allows.
- **Cache invalidation.** Targets depend on the points, normals, camera and
  splat settings. Rebuild them if any of those change; resuming with a
  different `pts3n` normalization invalidates the cache.
- **Partial coverage.** Whole-image coverage cannot detect a scan that is
  missing a back side while the front is fully covered. Check previews (and
  normal orientation) before trusting an A/B run.
- **Clipping.** Cameras are framed from the input cloud. If the model expands
  far outside it, use a larger radius factor and check per-view coverage.
