# Training Losses and Hyperparameters

This is the companion reference to [README.md](README.md). It explains what the
training objectives are, how the weights are scheduled, and what every important
flag does. Every formula below matches the implementation in
[main.py](main.py) and [model/losses.py](model/losses.py).

Contents:

1. [Notation and model](#1-notation-and-model)
2. [Complete objective per pipeline](#2-complete-objective-per-pipeline)
3. [Surface-fitting losses](#3-surface-fitting-losses)
4. [Regularizers](#4-regularizers)
5. [Pretraining objectives](#5-pretraining-objectives)
6. [Weight schedules and optimizer](#6-weight-schedules-and-optimizer)
7. [Adaptive subdivision and GHPR stages](#7-adaptive-subdivision-and-ghpr-stages)
8. [Hyperparameter reference](#8-hyperparameter-reference)
9. [Tuning guide and caveats](#9-tuning-guide-and-caveats)

---

## 1. Notation and model

Input point clouds are normalized to `[-1, 1]^3` with `center = mean(points)` and
`scale = max |points - center|`, so all distances (and therefore all loss values)
are expressed in normalized units. `--init_ckpt_reuse_normalization` reuses the
previous frame's transform for sequences.

| Symbol | Meaning |
|---|---|
| `F` | Forward map `(patch_id, uv) -> xyz`, the learnable surface |
| `G` | Optional inverse map `xyz -> uv` (only used by cycle losses) |
| `P` | Target point cloud |
| `Q` | Points sampled from `F` for the current step |
| `t_u, t_v` | Surface tangents `dF/du`, `dF/dv`, computed by autograd |
| `J` | Jacobian `[t_u, t_v]` (3×2); its singular values are `S = (s1, s2)` |
| `E, F_, G_` | First fundamental form `t_u·t_u`, `t_u·t_v`, `t_v·t_v` |

`F` is not a per-patch MLP. Each atlas carries a learnable feature matrix
(`d_features` per vertex); a query is the mean-value-coordinate interpolation of
the leaf's corner features (shared across patch borders), optionally concatenated
with a positional encoding of the global UV (fixed-grid) or reference-cube
position (adaptive), and decoded by a shared skip-MLP of width `W`, depth `D`,
Softplus(`beta`). This is why `d_features`, `W`, `D`, `L`, and `beta` are
architectural in nature, independent of the losses below.

The optimizer is Adam at `--lr`, cosine-annealed to `1e-6` over the run, with
global gradient-norm clipping at `1.0`, in both pipelines.

---

## 2. Complete objective per pipeline

### 2.1 Fixed-grid multi-patch (`--multi_patch`, default)

```
L = L_surf
    + λ₁ · L_cycle
    + λ₂ · L_param
    + μ_eff · L_tangent
    + λ_svd_eff · L_svd
    + γ_eff · L_normal
    + λ_outer · L_outer
```

`L_surf` depends on the training phase:

| Phase | `L_surf` |
|---|---|
| Start of run | Chamfer distance (`--surface_loss_type chamfer`, default) |
| DDF window | `λ_ddf · DDF` (`ddf`) or `λ_chamfer · CD + λ_ddf · DDF` (`cd_ddf`) |
| After `--corr_switch_epoch` / with `--corr_from_init` | `mean ||F(u_i) - p_i||₂` against a frozen correspondence table |

### 2.2 Adaptive cube atlas (`--adaptive`)

```
L = L_surf + μ_eff · L_tangent + λ_svd_eff · L_svd + γ_eff · L_normal
    + λ_ext_area · L_ext_area
```

with the same `L_surf` options, but there is no inverse map, no cycle term, and no
outer-boundary term. `L_surf` switches between Chamfer and DDF on the
`--ddf_start_epoch` / `--chamfer_resume_epoch` schedule.

### 2.3 Pretraining (initialization only, not part of the objective above)

| Mode | Loss |
|---|---|
| Flat sheet | Pointwise `L1` (default) or `MSE` to the `z = 0` plane, optionally `CD + 0.001 · Dirichlet` |
| Closed shape | Chamfer to a synthetic box/sphere (+ `μ_eff · Dirichlet`) |
| Adaptive box | Pointwise `MSE` or `L1` to the reference cube |

Pretraining produces `pretrain_checkpoint.pt` and the weights are then fine-tuned
with the main objective. `--init_ckpt` replaces it with the previous frame's
checkpoint (sequential tracking).

---

## 3. Surface-fitting losses

### 3.1 Chamfer distance (default)

```text
L_CD(P, Q) = mean_{p∈P} min_{q∈Q} ||p - q||₂  +  mean_{q∈Q} min_{p∈P} ||q - p||₂
```

Symmetric Chamfer as the **sum** of both one-sided means (not the average), so
its magnitude is roughly twice the usual symmetric CD average. Predictions are
`K · M_per_patch` UV samples per step (`K` = active patches). Targets come from
the full cloud (`--no_presplit`) or from per-patch pre-segmented subsets
(fixed-grid default). Large batches are evaluated in chunks (`--ddf_chunk_size`
where relevant; Chamfer chunks at 2048).

Related flags: `--M_per_patch`, `--N` (cloud size), `--no_presplit`,
`--ext_cd_weight` (see §4.7).

### 3.2 Pointwise correspondence regression

When the UV↔point correspondence is frozen, the surface term is a direct
regression instead of a set-to-set distance:

```text
L_MSE-like = mean_i ||F(patch_i, uv_i) - p_i||₂
```

It is the **mean Euclidean norm** (not squared), which keeps gradient magnitude
from vanishing near convergence and keeps the units comparable to Chamfer.
Building the table means projecting every target point onto the current surface
(dense UV search at `--corr_search_resolution`, then `--corr_refine_steps`
gradient steps at `--corr_refine_lr`) and freezing the result.

| Flag | Effect |
|---|---|
| `--corr_switch_epoch N` | Build the table at epoch `N` and use regression afterwards (0 = off) |
| `--corr_from_init` | Build it once at epoch 0 against `--init_ckpt`; regression for the whole run |
| `--corr_search_resolution` | Per-patch UV grid for the initial nearest-point search (default 64) |
| `--corr_refine_steps` / `--corr_refine_lr` | Per-point UV refinement (default 200 / 1e-3) |

Constraints: requires `--no_presplit`; incompatible with `--lam/--lam2` (cycle),
`--gamma` (normal), and DDF; implemented for the fixed-grid pipeline.
`--corr_from_init` additionally requires `--init_ckpt`.

### 3.3 Directional Distance Field (DDF) loss

DDF represents a surface by, for each query point, its distance to the surface and
the vector toward the closest point: `[d, closest - query]` (4 values; the
direction is not normalized).

Implementation:

1. **Reference points** `p_j` are sampled as target-cloud points plus Gaussian
   noise: `p_j = x_j + σ·ε`, `ε ~ N(0, I)` (`--ddf_sigma`, default 0.005).
2. **Ground-truth DDF** at those points is computed once against the target cloud.
   The closest point is approximated by the inverse-distance-weighted average of
   the `k` nearest target points (`--ddf_k_neighbors`, default 5).
3. **Predicted DDF** is evaluated at the same reference points against the
   predicted cloud `Q`.
4. The per-point residual and confidence-weighted loss are

```text
r_j = |Δdistance_j| + Σ_{c∈{x,y,z}} |Δdirection_j[c]|
s_j = exp(-β · r_j.detach())                  # confidence, β = --ddf_beta
L_DDF = Σ_j s_j r_j / Σ_j s_j                 # plain mean if β = 0
```

The residual is an L1 norm over the 4-vector. The weight is detached so the model
cannot reduce the loss by inflating the residual. `--ddf_beta 0` disables the
confidence weighting.

Phase control:

```text
--surface_loss_type ddf      : Chamfer until epoch ddf_start_epoch-1,
                               DDF from ddf_start_epoch (and optional switch-back)
--surface_loss_type cd_ddf   : L_surf = λ_chamfer · CD + λ_ddf · DDF during the window

--ddf_start_epoch E          : first epoch of the DDF window (0 = from epoch 1)
--chamfer_resume_epoch E     : return to Chamfer at E (0 = never)
--ddf_mu_decay k             : multiply μ_eff by k while DDF is active
```

Reference points and their ground truth are cached; `--ddf_resample_every N`
refreshes them every `N` epochs (0 = sample once). The adaptive pipeline also
refreshes them whenever the GHPR target subset changes.

Constraints: in the fixed-grid pipeline DDF requires `--no_presplit` (multi-sheet
atlasses only). The adaptive pipeline supports it directly. Because the k-NN
closest-point approximation can blend across nearby sheets, the recommended
recipe is Chamfer first for coarse geometry, then DDF for fine detail.

Flags: `--ddf_n_reference_points` (75000), `--ddf_k_neighbors` (5),
`--ddf_sigma` (0.005), `--ddf_beta` (0), `--ddf_chunk_size` (2048),
`--save_ddf_reference_every` (1000, debug dumps).

---

## 4. Regularizers

### 4.1 Tangent / Dirichlet energy (`--mu`)

```text
L_tangent = mean [ 0.5 · ( ||t_u||² + ||t_v||² ) ] = 0.5 · mean(E + G)
```

This is the Dirichlet (stretch) energy. It penalizes area distortion but is
scale-dependent: a patch that grows physically pays more. The fixed-grid pipeline
always uses this mode. The adaptive pipeline uses `--tangent_mode dirichlet` by
default and can switch to symmetric Dirichlet at `--sym_dirichlet_epoch`.

Weight: `μ_eff` (see §6). Typical values are `0.08` (run scripts) to `1.0`
(adaptive default); it is one or two orders of magnitude larger than the Chamfer
residual at convergence, so it is the dominant shape-regularity term.

### 4.2 Symmetric Dirichlet (`--tangent_mode symmetric_dirichlet`)

With singular values `S` of `J`, target `k = --svd_target`, and floor `ε`:

```text
Sc = clamp(S / k, ε, ∞)
L_sym = mean Σ_i ( Sc_i² - 2 ln Sc_i )
```

The log barrier prevents singular values from collapsing to zero, and the target
places the energy minimum at `S = k` instead of `S = 1`. Use
`--sym_dirichlet_epoch N` to start with plain Dirichlet and switch mid-run.

### 4.3 SVD / ARAP regularization (`--lam_svd`)

`--svd_mode` selects the energy, all evaluated per sampled point and averaged:

| Mode | Energy | Notes |
|---|---|---|
| `arap` | `Σ (S_i - 0.25)²` | Target is **hard-coded 0.25** in the current implementation |
| `arap_si` (default) | `Σ (S_i - mean(S))²` | Scale-invariant near-isometry; `mean(S)` detached |
| `one_sided_arap` | `Σ relu(S_i)² = Σ S_i²` | Currently equals a stretch penalty; `--svd_target` not used |
| `conformal` | `E + G - 2√(E·G - F_²)` | Equals `(s1 - s2)²`; penalizes anisotropy only |
| `collapse` | returns `0` | Placeholder; contributes nothing |

Caveats worth knowing:

- The code path only reads `--svd_target` in `symmetric_dirichlet`; `arap` uses
  the fixed 0.25 and `one_sided_arap` uses a zero threshold. The default
  `--svd_target 0.25` matches `arap`, so this only matters if you change it.
- `--svd_eps` is the clamp floor used by `symmetric_dirichlet` (default 1e-4).
- `--lam_svd` defaults to `0`, so SVD regularization is off unless enabled.
- Independent schedule: `--svd_warmup_epochs` / `--svd_warmup_delay`.

### 4.4 Normal consistency (`--gamma`)

The surface normal is `n = normalize(t_u × t_v)`. Two directions are matched:

```text
fwd = mean over predicted points:   φ( n(pred) · N(target_nearest) )
bwd = mean over target points:      φ( n(pred_nearest) · N(target) )

L_normal = w_fwd · fwd + w_bwd · bwd
```

`φ` is one of:

| Option | `φ(cos)` | Flag |
|---|---|---|
| Default | `1 - cos` | — |
| Unsigned | `1 - |cos|` | `--normal_unsigned` |
| Hinge | `relu(τ - cos)` | `--normal_hinge_tau τ` |

The backward/coverage term (`w_bwd`, default 1.0) applies the target's normal to
the nearest predicted point; the forward term (`w_fwd`, default 0.1) applies the
nearest target normal to each predicted point. Nearest neighbours come from a CPU
KD-tree over the full cloud (`--normal_nn kdtree`, default) or from the sampled
target batch (`batch`).

Constraints: the input file must carry normals. In the fixed-grid pipeline the
normal loss requires per-patch batches (`--no_presplit` must be off) and is not
available in the correspondence/MSE phase. In the adaptive pipeline it works in
either mode.

Scheduling: normally `γ_eff` follows the delay/warmup schedule. With multiple GHPR
stages the loss instead stays off for the first
`--normal_off_frac × --visible_epochs` epochs of every phase, then switches on at
full γ (hard switch); see §7.

### 4.5 Cycle consistency (`--lam`, `--lam2`)

Only active with an inverse map `G`:

```text
L_cycle = Σ_patches CD( F(G(P_patch)), P_patch )
L_param = Σ_patches CD( G(Q_patch), UV_grid_patch )
```

`L_cycle` (weight `λ₁ = --lam`) requires the reconstructed surface to map back to
the target; `L_param` (weight `λ₂ = --lam2`) requires the UVs recovered from the
predicted surface to match the sampled grid.

Caveats:

- `G` is only constructed for `--atlas_mode single_sheet`; other atlases raise
  `NotImplementedError`.
- Both terms require per-patch grouped batches (`--no_presplit` off) and are
  incompatible with the correspondence/MSE phase.
- The inverse-map implementation is flagged deprecated under the current MVC
  encoder in [model/model.py](model/model.py). The shipped run scripts therefore
  set `--lam 0 --lam2 0`.

`--L_inv` sets positional-encoding frequencies for `G` (default 0).

### 4.6 Outer-boundary rectangle (`--lambda_outer_boundary`)

Matches the **global outer border** of a single-sheet atlas to the rectangle
`[-1,1]² × {0}` in the normalized plane. For every patch edge that lies on the
outer grid boundary (internal shared edges are excluded), `--outer_boundary_samples`
points are sampled and compared to that patch's own segment of the rectangle
perimeter with `L1` (default) or `MSE`; the edge terms are averaged.

This keeps the atlas from shrinking away from its initial flat sheet. It is
designed for `single_sheet`; set `--lambda_outer_boundary 0` for `two_sheet` /
`six_sheet` (the run scripts do). Defaults: weight 1.0, 512 samples per edge, L1.

### 4.7 Extension mask (adaptive only)

At `--ext_mask_epoch`, unpaired openings in the learned UV atlas are detected
(`utils/uv_hole_mask.py` → `utils/opening_labels.py` → `utils/pair_opening.py`)
and their samples are treated differently in the loss:

| `--ext_mode` | Effect |
|---|---|
| `no_reg` (default) | Samples inside extensions are removed from `L_tangent` and `L_svd`, so Chamfer alone shapes them |
| `no_fit` | Samples are removed from the fitting losses (Chamfer/DDF/normal, both directions), so only the regularizers apply |

With `no_reg`, `--ext_cd_weight` scales both Chamfer directions for extension
samples (1 = same as the rest). `--ext_mask_every` re-detects every N epochs
(0 = detect once). Other knobs: `--ext_max_area_frac` (0.05),
`--ext_resolution` (64), `--ext_nn_scale` (10), `--ext_min_opening_vertices` (10),
`--ext_weld_tol` (1e-5), `--ext_face_res` (1024).

`--lambda_ext_area` (default `0` = off) adds an explicit penalty that shrinks
the extension membrane:

```text
L_ext_area = mean over extension samples [ || ∂F/∂u × ∂F/∂v || ]
```

The cross-product norm is the local 3D area element, so this directly minimizes
the extension's area. It requires `--ext_mask_epoch > 0`, is independent of
`--ext_mode`, and is logged as `train/ext_area` in TensorBoard and on the
history plot.

---

## 5. Pretraining objectives

### 5.1 Flat-sheet pretraining (fixed-grid)

Every patch is supervised to map its local UV square onto its own cell of the
`z = 0` plane inside `[-1,1]²`. For global coordinates
`(gu, gv) = ((row + u)/n_rows, (col + v)/n_cols)`, the target is
`(2gu - 1, 2gv - 1, 0)`.

| `--pretrain_loss` | Loss |
|---|---|
| `l1` (default) | Pointwise L1 |
| `mse` | Pointwise MSE |
| `cd` | Per-patch Chamfer + `0.001 × Dirichlet` (regularizer fixed internally) |

### 5.2 Closed-shape pretraining (fixed-grid)

Fits a synthetic shape (`--pretrain_shape`, default `box`,
`--pretrain_shape_samples` default 200,000) before seeing the real cloud:

- `no_presplit_pretrain` off: per-patch Chamfer to pre-segmented synthetic points;
- `no_presplit_pretrain` on: global Chamfer to the whole synthetic cloud;
- six-sheet + `--face_aware_box_supervision` + `--pretrain_shape box`: direct
  per-face, per-patch Chamfer against cube-face targets.

A `μ_eff · Dirichlet` term (μ from `--mu`, same `--mu_warmup_*` schedule) is added
throughout. Note that `six_sheet` pretraining without face-aware supervision
currently fails in pre-segmentation — use `--face_aware_box_supervision` (as in
`run.sh`) or `--pretrain_mode flat`.

### 5.3 Adaptive box pretraining

Fits every leaf directly to the reference cube/cuboid (no subdivision):

```text
L_box = MSE or L1( F(leaf, uv), center + radius · cube_xyz(leaf, uv) )
```

`center`/`radius` come from GHPR (`--ghpr_init`), the cloud's padded AABB
(`--box_init aabb`, default), the canonical `[-1,1]³` cube (`--box_init cube`),
or explicit `--box_center` / `--box_radius`. `--pretrain_loss` is restricted to
`mse`/`l1` here. The seam check after pretraining should report gaps around
`1e-6` or smaller.

### 5.4 Warm start from a checkpoint

`--init_ckpt PATH` loads `F` from a previous run and skips synthetic pretraining.
This is the frame-to-frame tracking mode: each frame starts from the previous
frame's converged surface. Combine with `--corr_from_init` to also inherit the UV
parameterization. `--init_ckpt_reuse_normalization` keeps the previous frame's
center/scale so all frames live in one coordinate frame.

---

## 6. Weight schedules and optimizer

### 6.1 Warmup / delay

`μ_eff`, `λ_svd_eff`, and `γ_eff` each use the same ramp with their own target,
delay, and length. With `μ` as the example:

```text
μ_eff(e) = 0                                      if e <= mu_warmup_delay
         = μ · ramp((e - delay) / warmup_epochs)  if 0 < e - delay < warmup_epochs
         = μ                                      otherwise
```

If `warmup_epochs = 0`, the target value applies immediately after the delay.
Epochs are 1-based, so `--mu_warmup_delay 300` keeps μ at 0 through epoch 300 and
starts the ramp at 301.

`--schedule` selects the ramp shape:

| Schedule | `ramp(t)` |
|---|---|
| `linear` | `t` |
| `cosine` (default) | `0.5 (1 - cos(πt))` |
| `exponential` | `1 - exp(-5t)` (≈99.3% at `t=1`) |
| `sigmoid` | `1 / (1 + exp(-12(t - 0.5)))` |

Additional rules:

- During the DDF window, `μ_eff *= --ddf_mu_decay` (adaptive and fixed-grid).
- In the adaptive pipeline, if `--gamma > 0` and multiple GHPR stages are used,
  `γ_eff` ignores delay/warmup and uses the per-phase hard switch described in §7.
- The reported `μ_eff`, `λsvd_eff`, `γ_eff` in the console and in TensorBoard are
  the effective values after scheduling.

### 6.2 Optimizer and epoch structure

- Adam (`--lr`, default 1e-3), cosine annealing to `1e-6` over `--epochs`.
- Gradient clipping at norm 1.0 every step.
- Every epoch samples fresh UVs; targets are resampled each step (or each epoch
  in pretraining).
- `--reg_every N` evaluates tangent/SVD/normal only every N epochs (fixed-grid
  only; 1 = every epoch). Intermediate epochs still apply the surface loss.
- `--checkpoint_every` controls `checkpoint_<epoch>.pt` frequency (0 disables).

---

## 7. Adaptive subdivision and GHPR stages

These mechanisms change *where* the model has capacity and *which* points it fits;
they are not extra loss terms.

### 7.1 Distortion-driven subdivision

Every `--subdiv_every` epochs between `--subdiv_start` and `--subdiv_stop`
(0 = never stop), each leaf's distortion is estimated on
`--distortion_samples` random UV samples using `--distortion_mode`:

| Mode | Distortion |
|---|---|
| `area` | `mean √(E·G - F_²)` (3D area per unit UV) |
| `dirichlet` (default) | `mean 0.5(E + G)` (stretch) |
| `symmetric_dirichlet` | `mean [E + G - ln(E·G - F_²)]` |
| `conformal` | `mean [(E-G)² + 4F_²] / (E+G)²` (anisotropy in [0,1]) |

Leaves above `--subdiv_threshold` are split, up to `--subdiv_max_depth`
(`--ghpr_subdiv_max_depth` caps depth during GHPR stages) and at most
`--max_splits_per_round` per check (0 = unlimited). Splitting is
surface-preserving (new vertex features are interpolated), and the Adam moments
are carried over while the cosine schedule restarts over the remaining epochs.

### 7.2 GHPR staged warm start

`--ghpr_init` picks an interior viewpoint, places the pretrain box there, and fits
increasingly complete subsets of the cloud:

- Stage `i` uses `gamma = --ghpr_gamma × 0.1^i` (or explicit `--ghpr_gammas`);
- each stage lasts `--visible_epochs` epochs; the rest of training uses the full
  cloud;
- `--ghpr_keep_largest` keeps only the connected group containing the previous
  stage's points (link radius `--ghpr_connect_radius` × median NN spacing);
- with more than one stage, the normal loss is off for the first
  `--normal_off_frac` of every phase, then switches on.

---

## 8. Hyperparameter reference

Defaults are the argparse defaults unless marked otherwise. "run.sh" means the
value used by the annotated six-sheet example script.

### 8.1 Data, architecture, and sampling

| Flag | Default | Meaning |
|---|---|---|
| `--file` | none | Input `.ply`/`.xyz`/`.npy`; omit for a synthetic shape |
| `--shape` | `box` | Synthetic shape when `--file` is absent |
| `--N` | 100000 | Points used (downsamples); `-1` keeps all |
| `--mesh_res` | 100 (adaptive 40) | Per-patch grid resolution for exports |
| `--result_dir` | `results_sheet` (adaptive `logs/adaptive`) | Output directory (auto-incremented) |
| `--log_every` | 200 (adaptive 100) | Console/TensorBoard logging cadence |
| `--tensorboard` | off | Log scalars, figures, and the final mesh |
| `--device` | CUDA if available | Torch device |
| `--epochs` | 5000 (adaptive 1000) | Main training length |
| `--lr` | 1e-3 | Adam learning rate |
| `--W` | 256 (adaptive 512) | Decoder hidden width |
| `--D` | 6 | Decoder hidden layers |
| `--beta` | 100 | Softplus sharpness (not DDF β) |
| `--L` | 0 | PE frequencies (fixed-grid: global UV; adaptive: reference cube) |
| `--d_features` | 88 | Latent features per atlas vertex |
| `--M` | 4096 | Single-patch samples/step; unused by the multi-patch path |
| `--M_per_patch` | 4096 (adaptive 128) | UV samples per active patch per step |
| `--n_patches` | 4 | Single-sheet patch count (grid is factored automatically) |
| `--reg_every` | 1 | Fixed-grid: evaluate regularizers every N epochs |
| `--checkpoint_every` | 50 (adaptive 100) | Intermediate checkpoint cadence (0 = off) |

### 8.2 Atlas configuration

| Flag | Default | Meaning |
|---|---|---|
| `--atlas_mode` | `single_sheet` | `single_sheet`, `two_sheet`, or `six_sheet` |
| `--two_sheet_side_rows/cols` | 2 / 2 | Patches per side of a two-sheet atlas |
| `--two_sheet_split_axis` | 2 | Axis used to split the cloud into two sides |
| `--two_sheet_side_axes` | `0 1` | Axes for in-side segmentation |
| `--six_sheet_face_rows/cols` | 2 / 2 | Patches per cube face |
| `--face_aware_box_supervision` | off | Direct per-face box pretraining for six-sheet |
| `--no_presplit` | off | Train against the full cloud (two/six-sheet only) |
| `--no_presplit_pretrain` | off | Same, but only during pretraining |

`--n_patches` applies to `single_sheet`; two-sheet and six-sheet counts follow
`2 × rows × cols` and `6 × rows × cols`. `run.sh` uses 2×2 per face = 24 patches.

### 8.3 Surface-fitting loss

| Flag | Default | Meaning |
|---|---|---|
| `--surface_loss_type` | `chamfer` | `chamfer`, `ddf`, or `cd_ddf` |
| `--lambda_chamfer` | 1.0 | CD weight in `cd_ddf` |
| `--lambda_ddf` | 1.0 | DDF weight in `ddf` / `cd_ddf` |
| `--ddf_start_epoch` | 0 | First DDF epoch (0 = epoch 1) |
| `--chamfer_resume_epoch` | 0 | Return to Chamfer at this epoch (0 = never) |
| `--ddf_mu_decay` | 1.0 | μ multiplier during the DDF window |
| `--ddf_n_reference_points` | 75000 | Reference points per DDF shell |
| `--ddf_k_neighbors` | 5 | k-NN used for the closest-point estimate |
| `--ddf_sigma` | 0.005 | Gaussian noise displacing reference points |
| `--ddf_beta` | 0.0 | Confidence sharpness `exp(-βr)` (0 = off) |
| `--ddf_resample_every` | 0 | Refresh reference set every N epochs (0 = once) |
| `--ddf_chunk_size` | 2048 | Chunk size for distance computations |
| `--save_ddf_reference_every` | 1000 | DDF debug dump cadence (0 = off) |
| `--corr_switch_epoch` | 0 | Epoch to switch to pointwise regression |
| `--corr_from_init` | off | Freeze correspondence at epoch 0 (needs `--init_ckpt`) |
| `--corr_search_resolution` | 64 | UV grid for the correspondence search |
| `--corr_refine_steps` | 200 | UV refinement iterations |
| `--corr_refine_lr` | 1e-3 | UV refinement step size |

### 8.4 Regularization weights and schedules

| Flag | Default | Meaning |
|---|---|---|
| `--mu` | 0.5 (adaptive 1.0; run.sh 0.08) | Tangent/Dirichlet weight |
| `--mu_warmup_epochs` | 0 | Ramp length |
| `--mu_warmup_delay` | 0 | Delay before the ramp |
| `--schedule` | `cosine` | Ramp shape for μ/λ_svd/γ |
| `--tangent_mode` | `dirichlet` | Adaptive tangent energy (`dirichlet` / `symmetric_dirichlet`) |
| `--sym_dirichlet_epoch` | 0 | Switch to symmetric Dirichlet at this epoch (0 = never) |
| `--lam_svd` | 0 | SVD/ARAP weight (0 = off) |
| `--svd_mode` | `arap_si` | `arap`, `arap_si`, `one_sided_arap`, `conformal`, `collapse` |
| `--svd_target` | 0.25 | Target singular value (read by `symmetric_dirichlet`) |
| `--svd_eps` | 1e-4 | Singular-value floor (`symmetric_dirichlet`) |
| `--svd_warmup_epochs` | 0 | SVD ramp length |
| `--svd_warmup_delay` | 0 | SVD ramp delay |
| `--gamma` | 0 | Normal-consistency weight (requires normals) |
| `--gamma_warmup_epochs` | 0 | Normal ramp length |
| `--gamma_warmup_delay` | 0 | Normal ramp delay |
| `--normal_nn` | `kdtree` | Nearest-neighbour source: full cloud or batch |
| `--normal_unsigned` | off | Use `1 - |cos|` |
| `--normal_w_fwd` | 0.1 | Forward (pred→target) weight |
| `--normal_w_bwd` | 1.0 | Backward (target→pred) weight |
| `--normal_hinge_tau` | none | Hinge threshold `relu(τ - cos)` |
| `--normal_off_frac` | 1 (adaptive, staged) | Fraction of each GHPR phase with normals off |
| `--lam` | 1.0 (run.sh 0) | Cycle-consistency weight λ₁ |
| `--lam2` | 1.0 (run.sh 0) | Inverse-cycle weight λ₂ |
| `--L_inv` | 0 | PE frequencies for the inverse encoder |
| `--lambda_outer_boundary` | 1.0 (run.sh 0) | Rectangle boundary weight |
| `--outer_boundary_samples` | 512 | Samples per outer edge |
| `--outer_boundary_loss_type` | `l1` | `l1` or `mse` |
| `--lambda_bcd` | 0 | Parsed/stored but currently unused |

### 8.5 Pretraining

| Flag | Default | Meaning |
|---|---|---|
| `--pretrain_init` | off | Run initialization pretraining only |
| `--pretrain_then_train` | off | Pretrain, then continue into main training |
| `--pretrain_epochs` | 2000 | Pretraining length |
| `--pretrain_loss` | `l1` (adaptive `mse`) | `l1`, `mse`, or fixed-grid `cd` |
| `--pretrain_mode` | `closed_shape` | `auto`, `flat`, or `closed_shape` |
| `--pretrain_shape` | `box` | Synthetic shape (`sphere` for two-sheet spheres) |
| `--pretrain_shape_samples` | 200000 | Synthetic cloud size |
| `--init_ckpt` | none | Warm start instead of synthetic pretraining |
| `--init_ckpt_reuse_normalization` | off | Reuse the checkpoint's center/scale |

### 8.6 Adaptive pipeline (`--adaptive`)

| Flag | Default | Meaning |
|---|---|---|
| `--base_subdivisions` | 1 | Uniform splits at construction (1 → 24 leaves) |
| `--pretrain_only` | off | Stop after box pretraining |
| `--skip_pretrain` | off | Start main training immediately |
| `--load_ckpt` | none | Resume topology + weights from an adaptive checkpoint |
| `--save_vertex_pos_every` | 10 | Tracked-vertex dump cadence (0 = off) |
| `--subdiv_threshold` | 0.3 | Distortion above which a leaf splits (0 = no subdivision) |
| `--subdiv_max_depth` | 5 | Absolute quadtree depth cap |
| `--ghpr_subdiv_max_depth` | none | Depth cap during GHPR stages |
| `--subdiv_every` | 500 | Subdivision check interval |
| `--subdiv_start` | 1000 | First check epoch |
| `--subdiv_stop` | 0 | Last check epoch (0 = never stop) |
| `--distortion_mode` | `dirichlet` | `area`, `dirichlet`, `symmetric_dirichlet`, `conformal` |
| `--distortion_samples` | 128 | UV samples per leaf for the estimate |
| `--max_splits_per_round` | 0 | Cap on splits per check (0 = unlimited) |
| `--ghpr_init` | off | GHPR interior viewpoint + staged visible targets |
| `--ghpr_gamma` | -0.01 | Base GHPR kernel exponent |
| `--ghpr_k` | 15 | Neighbours for the interior test |
| `--visible_epochs` | 2000 | Length of each GHPR stage |
| `--ghpr_stages` | 1 | Number of visible-only stages |
| `--ghpr_gammas` | none | Explicit per-stage gammas (overrides `--ghpr_gamma`) |
| `--ghpr_keep_largest` | off | Keep one connected visible group per stage |
| `--ghpr_connect_radius` | 5.0 | Link radius in units of median NN spacing |
| `--box_init` | `aabb` | Pretrain box: padded AABB or canonical cube |
| `--box_margin` | 0.05 | AABB padding fraction |
| `--box_min_frac` | 0.1 | Floor on each half-extent |
| `--box_center` | none | Explicit pretrain box center (normalized frame) |
| `--box_radius` | 0 | Explicit box half-width (0 = auto) |
| `--ext_mask_epoch` | 0 | Detect UV extensions at this epoch (0 = off) |
| `--ext_mode` | `no_reg` | `no_reg` (mask μ/λ_svd) or `no_fit` (mask fitting losses) |
| `--ext_cd_weight` | 1.0 | Chamfer weight for extension samples (`no_reg`) |
| `--lambda_ext_area` | 0 | Weight of `mean ||∂F/∂u × ∂F/∂v||` on extension samples (0 = off) |
| `--ext_mask_every` | 0 | Re-detect every N epochs (0 = once) |
| `--ext_max_area_frac` | 0.05 | Max opening area relative to the surface |
| `--ext_resolution` | 64 | UV grid per leaf per side for detection |
| `--ext_nn_scale` | 10 | Hole threshold × median NN spacing |
| `--ext_min_opening_vertices` | 10 | Drop smaller opening components |
| `--ext_weld_tol` | 1e-5 | Vertex welding tolerance |
| `--ext_face_res` | 1024 | Stored mask resolution per cube face |

### 8.7 Adaptive defaults vs the shared defaults

With `--adaptive`, these defaults are replaced before parsing (explicit CLI
values still win):

| Flag | Shared default | Adaptive default |
|---|---|---|
| `--shape` | `box` | `box` |
| `--result_dir` | `results_sheet` | `logs/adaptive` |
| `--N` | 100000 | 100000 |
| `--mesh_res` | 100 | 40 |
| `--log_every` | 200 | 100 |
| `--checkpoint_every` | 50 | 100 |
| `--epochs` | 5000 | 1000 |
| `--W` | 256 | 512 |
| `--mu` | 0.5 | 1.0 |
| `--M_per_patch` | 4096 | 128 |
| `--pretrain_loss` | `l1` | `mse` |
| `--gamma` | 0 | 0 |

### 8.8 Typical shipped configurations

| | `run.sh` (six-sheet) | `run_all_3d_models.sh` | `adaptive.sh` |
|---|---|---|---|
| Atlas | 6 faces × 2×2 = 24 patches | same | adaptive, 1 base subdivision = 24 leaves |
| Init | box pretrain, 1000 epochs, face-aware | same | box pretrain, 500 epochs |
| Samples | `M_per_patch 500`, `N 100000` | `M_per_patch 1100` | `M_per_patch 128` |
| Surface loss | Chamfer | Chamfer | Chamfer |
| Regularizers | `mu 0.08`, no normal, no SVD, no cycle | same | `mu 0.08`, `svd_mode arap`, `lam_svd 0` |
| Boundary/cycle | all off (`lambda_outer 0`, `lam 0`, `lam2 0`) | same | n/a |
| Duration | 5000 | 5000 | 5000 |

---

## 9. Tuning guide and caveats

**Where to start.** Use the `run.sh` configuration for a closed object with a
box-like topology. Keep `--mu` small (≈0.05–0.1) when Chamfer is the fitting
loss; raise it if the atlas folds or collapses during pretraining, lower it if
the surface cannot reach details.

**Understanding the balance.** Chamfer/DDF values shrink toward zero as the fit
converges; regularizers do not, so their effective weight usually needs to be
lower than the initial loss values suggest. Watch `train/total`,
`train/surface`, and the effective weights together in TensorBoard.

**DDF.** Start with Chamfer and switch to DDF once coarse shape is resolved
(`--ddf_start_epoch`), optionally returning to Chamfer
(`--chamfer_resume_epoch`). A large `--ddf_sigma` samples farther from the
surface; `--ddf_beta > 0` focuses the loss on already-well-fit regions. In the
fixed-grid pipeline DDF requires `--no_presplit`.

**Correspondence training.** Use `--corr_switch_epoch` for a fixed final
refinement phase with a stable parameterization, or `--corr_from_init` +
`--init_ckpt` for frame-to-frame tracking. Both require `--no_presplit` and
exclude cycle/normal terms.

**Normal consistency.** Requires normals in the input file. In the fixed-grid
pipeline it requires per-patch batches (no `--no_presplit`) and is unavailable in
the correspondence phase. Use `--normal_unsigned` when the input normals' sign is
unreliable, and `--normal_hinge_tau` to stop penalizing already-aligned normals.

**Boundary and cycle terms.** The outer-boundary rectangle and cycle losses are
single-sheet tools; the shipped multi-sheet configurations disable them. Leaving
`--lam/--lam2` at their 1.0 defaults with a non-single-sheet atlas raises an error;
set both to 0.

**Atlas sizing.** Total per-step samples are `active_patches × M_per_patch`.
Approximate `M_per_patch` by keeping the product near 24k for 24 patches (1100),
or lower it when memory-bound. Larger `d_features` and `W` increase capacity and
cost; `L` is the main high-frequency detail knob (`L = 0` starts purely
feature-driven).

**Adaptive runs.** Subdivision is the capacity controller: a low threshold or
deep `--subdiv_max_depth` grows leaves quickly. Prefer a coarse-to-fine run
(Chamfer first, then optional DDF) and keep `--subdiv_start` after the coarse
shape has formed. GHPR stages require care: every stage must be shorter than the
remaining run, and `--normal_off_frac` only applies when there is more than one
stage.

**Known implementation quirks.**

- `--lambda_bcd` is accepted but unused.
- `--M` is ignored by the multi-patch path; use `--M_per_patch`.
- `arap` uses a hard-coded target of 0.25 and `one_sided_arap` a zero threshold;
  only `symmetric_dirichlet` reads `--svd_target`.
- `collapse` currently returns zero energy.
- `G` / cycle consistency is flagged deprecated under the MVC forward map.
- Six-sheet closed-shape pretraining without `--face_aware_box_supervision` fails
  in pre-segmentation.
- The outer-boundary loss assumes a single connected sheet grid.

**Diagnostics.** Console logs and TensorBoard report every term separately
(`train/cd`, `train/ddf`, `train/tangent`, `train/svd`, `train/normal`,
`train/total`, effective weights, LR, leaves). `history.png`, `patch_config_*.png`,
and the correspondence/DDF/boundary debug folders under `--result_dir` show what
the losses are actually doing.
