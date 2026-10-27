# Changelog

Notable changes to NISR_Seawall. Dates are local time (YYYY-MM-DD).

## 2026-10-27

### CD-driven subdivision

- `model/subdivision.py`: new `compute_patch_cd` scores every leaf on a fixed
  `G x G` UV grid against the full point cloud. `fwd` is the mean distance from
  a leaf's grid points to the nearest target point; `bwd` lets every target
  point vote for the leaf that owns its nearest grid point and records the mean
  vote distance, which grows when a leaf is under-resolved. The split score is
  `fwd + bwd`. Extension-masked samples are excluded and leaves whose valid
  fraction is below 0.25 are skipped.
- `model/subdivision.py`: new `subdivide_by_cd` splits leaves above an absolute
  threshold, largest score first, with an optional per-round cap.
- `main.py`: `--subdiv_metric {distortion,cd}` with `--subdiv_cd_threshold`
  (0 disables), `--subdiv_cd_grid_res` (default 8) and
  `--subdiv_cd_min_valid_frac` (default 0.25). The CD branch reuses the CD
  sampling grid, records metric/threshold/grid in metadata and preserves Adam
  moments across the optimizer rebuild.
- Threshold 0.03 was calibrated on phase-14 checkpoints: per-leaf score p50/p90
  was 0.028/0.043 at epoch 1000 (84 leaves) and 0.013/0.018 at epoch 6000
  (312 leaves).
- Phase 15 (conformal SVD from epoch 3000, threshold 0.03, 48-split cap) grew
  24 -> 81 -> 159 -> 303 -> 369 -> 396 -> 483 -> 582 -> 636 leaves. Render L1
  stayed at 0.0180, but the depth-6/7 leaves added after epoch 3000 folded:
  17.1% intra-patch foldover at delta 0.002 versus 0.99% in phase 14.

### Phase 17: NGF Laplacian regularizer instead of SVD

Armadillo, 6000 epochs, seed 41, same GHPR init, two-layer render loss (far
weight 0.5), CD subdivision at 0.03 and tangent term as phase 15/16, with
`--render_laplacian_weight 1.0 --lam_svd 0.0`.

| Metric | phase 15 conformal | phase 16 arap_si | phase 17 Laplacian |
|---|---:|---:|---:|
| Leaves | 636 | 711 | 297 |
| Full L1 | 0.0180 | 0.0180 | 0.0180 |
| Chamfer symmetric | 0.0123 | 0.0144 | 0.0126 |
| Coverage (<0.02) | 0.997 | 0.987 | 0.990 |
| Rendered normal p95 (deg) | 31.07 | 25.44 | 23.70 |
| UV aspect p95 | 15.6 | 24.3 | 9.9 |
| Foldover delta=0.002 | 17.10% | 26.77% | 6.44% |
| Foldover delta=0.008 | 58.08% | 57.78% | 34.56% |

- The Laplacian term ran at 0.0019-0.0021 against a render term of 0.024-0.028
  (about 7% of the objective), so it is not a dominating smoother.
- It also suppresses CD splitting: the `bwd` coverage radius drops and the run
  stops at 297 leaves instead of 636-711, avoiding most depth-6/7 leaves.
  Folding falls below both phase-15 draws (17.1% and 25.5%).
- Folding is reduced, not removed: 16 patches remain above 50% folded, mostly
  at depth 5-6. Cloud->sheet Chamfer is slightly worse (0.0072 versus 0.0056),
  consistent with mild Laplacian shrink.
- Single draw. CD-driven topology still varies between runs, so a second seed
  is needed to separate the regularizer effect from the topology draw.

### Training speed: batched render path, cheaper tracking, regularizer cadence

The render-loss step was dominated by per-leaf Python/GPU dispatch, not by the
NGF objective itself.

- `utils/render_mesh.py`: `sample_patch_mesh` now evaluates all patches in one
  vectorized forward call (chunked over flattened points) instead of one call
  per leaf. This removes the GPU sync inside adaptive feature interpolation.
- `model/ngf_rendering_loss.py`: multi-layer rendering now uses a batched
  `dr.DepthPeeler`, matching NGF's `render.py`, instead of a per-view,
  per-layer Python loop with manual triangle removal. The single-layer path is
  unchanged.
- `utils/utils.py`: `extract_adaptive_vertex_global_positions` evaluates all
  leaf corners in one call instead of one call per leaf.
- `main.py`: `--reg_every` now applies to adaptive training. The shared
  Jacobian and every term that depends on it (tangent, SVD, normal,
  extension-area) follow the cadence. Default remains 1.
- `main.py`: `--save_vertex_pos_every` now defaults to 0. The tracker JSON
  grows quadratically and rewrites the whole file on every save
  (1.58 s per save at 45 MB); opt in explicitly when tracking is needed.

### Equivalence checks

- Batched mesh sampling: max vertex difference 5.4e-7, gradient difference
  2.4e-7 relative, identical faces. Checked with patch subsets, chunks that
  cross patch boundaries, and the fixed-grid `MultiPatchForwardMap` (3.4e-8).
- DepthPeeler: layer 0 is bit-identical (3.0e-8). Layer 1 differs on 0.8% of
  covered pixels (frame-mean 2e-4) because true per-pixel depth peeling and
  whole-triangle removal pick different back surfaces. `DepthPeeler` is the
  NGF reference path, and the far layer is weighted 0.5 on matched pixels.
- Vertex tracker: identical ids and values (2.4e-7).

### Component timings

Armadillo checkpoint, 630 leaves, 8x8 mesh grid per patch, 10 views at 256x256:

| Component | Before | After |
|---|---:|---:|
| Render mesh forward | 437 ms | 27 ms |
| Full render block, fwd+bwd | 687 ms | 50 ms |
| Tracker extraction | 394 ms | 5 ms |
| Simulated epoch, reg every epoch | 0.857 s | 0.220 s |
| Simulated epoch, reg every 2 epochs | - | 0.149 s |

Smoke tests covered 10 epochs with tracker saves, subdivision from 24 to 72
leaves with atlas rebuilds (32 to 25 to 20 grid per patch), and a 40-epoch
resumed run at 636 leaves (0.223 s/epoch).

### Phase 15 rerun comparison

Armadillo, 6000 epochs, identical recipe and seed 41. Wall clock went from
63m23s to 16m22s (3.87x); the in-loop timer went from 3774.7 s to 953.9 s
(3.96x). Cumulative speedup: 2.2x at epoch 1000, 3.7x at 3000, 4.0x at 6000.

| Metric | Original | Rerun |
|---|---:|---:|
| Full L1 | 0.0180 | 0.0180 |
| Normal L1 | 0.0259 | 0.0259 |
| Position L1 | 0.0101 | 0.0102 |
| Chamfer symmetric | 0.0123 | 0.0133 |
| Coverage (<0.02) | 0.997 | 0.995 |
| Rendered normal median / p95 | 5.10 / 31.07 deg | 5.24 / 30.49 deg |
| Cloud normal median / p95 | 2.86 / 17.58 deg | 3.02 / 17.17 deg |
| Checker edge p95 | 0.035 | 0.035 |
| UV aspect p95 | 15.6 | 22.5 |
| Final leaves | 636 | 750 |

Notes:

- The rerun is not bit-identical. DepthPeeler's layer-1 supervision shifts
  CD-driven split scores near the 0.03 threshold, so subdivision decisions
  diverge (750 vs 636 leaves). Seeding covers Python/NumPy/torch but not
  deterministic cuDNN/nvdiffrast kernels, and split ordering amplifies the
  remaining numeric differences. Per-leaf fit and normals are equivalent; the
  extra leaves explain the slightly higher Chamfer and UV aspect.
- Final seam gap: 3.998e-7.
- Both runs carry the known UV-aspect warning (one compressed UV direction in
  many patches); geometry metrics are unaffected.
- `--reg_every 2` was not used for the rerun. It is expected to save roughly
  another 30%.

Artifacts:

- Comparison sheet and report:
  `logs/render_phase15_conformal_cdgrid/check/compare_old_vs_rerun/`
  (`comparison.png`, `inspection.md`)
- Rerun log:
  `logs/render_phase15_conformal_cdgrid/armadillo_layers2_conformal_cdgrid_rerun.log`
- Rerun output and final checkpoint:
  `logs/render_phase15_conformal_cdgrid/armadillo_layers2_conformal_cdgrid_rerun/`

The original run directory, log, and check outputs were left untouched.
