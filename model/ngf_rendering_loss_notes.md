# NGF rendering loss — NISR integration notes

Module: [`ngf_rendering_loss.py`](ngf_rendering_loss.py)  
Branch: `renderloss`  
Status: renderer and loss verified; deliberately **not wired into `main.py`**.

## Bottom line

The loss consumes a connected triangle mesh. NISR optimizes a neural atlas from
unstructured per-patch UV samples, so the arrays only line up after the atlas
has been sampled into an indexed mesh. The existing exporter
`utils.sample_multi_patch_grid` produces compatible vertex/face shapes, but it
is export-only: it copies every patch to CPU and calls `.numpy()`
(`utils/utils.py:660-661`), which detaches the graph. It also keeps duplicate
vertices at patch seams and returns no normals. Cameras and reference maps do
not exist in the repository yet.

## Interface expected by the loss

| Argument | Shape / type | Notes |
|---|---|---|
| `vertices` | `(V, 3)` torch float32, CUDA | normalized frame |
| `normals` | `(V, 3)` torch float32, CUDA | unit length, same indexing as vertices |
| `faces` | `(F, 3)` torch int32, CUDA | triangle indices into the shared vertex array |
| `views` | `(B, 4, 4)` torch float32 | world-to-camera, columns `[right, up, look, eye]` |
| `reference` | `(B, H, W, 6)` or `(B, H, W, 7)` | `[normal, world position]`; a 7th alpha channel is ignored |

The renderer is single-layer and returns `(B, H, W, 6)`. Images use
nvdiffrast's bottom-up order and NGF's x-negating projection. Reference and
prediction must be in the same normalized frame and use the same cameras.

## What NISR provides today

| Data | Shape / form | Source |
|---|---|---|
| Surface samples | `(M, 3)` from `F(patch_id, uv)`, `uv` is `(M, 2)` | training loop |
| Analytic tangents | `t_u, t_v` per sample; `n = t_u × t_v` | `model/losses.py:344`, `model/losses.py:472` |
| Export mesh | `(P·R², 3)` NumPy float32, `(P·2·(R−1)², 3)` NumPy int32 | `utils/utils.py:615` |
| Mesh normals | not returned by the exporter | — |
| Cameras / reference maps | none in the training pipeline | — |
| Adaptive mesh | no triangle mesh in the training loop; quadtree leaves with hanging vertices | `model/model.py:1037`, `model/model.py:1301` |

## Blockers

### 1. The existing mesh is detached

`sample_multi_patch_grid` evaluates each patch in a loop and appends
`F(...).cpu()`, then converts the concatenation with `.numpy()`
(`utils/utils.py:660-661`). It is used only for export and visualization
(e.g., `main.py:2677`). A rendering loss needs a GPU sampler that keeps the
graph alive; otherwise the render loss cannot reach the decoder or vertex
features.
A differentiable sampler also needs to be cheap enough for every optimization
step, using a much lower resolution than `--mesh_res 100`.

### 2. No vertex normals in the exported mesh

The exporter returns only `verts, faces`. NISR does have the right normal
convention: `n = cross(t_u, t_v)` (`model/losses.py:472`,
`main.py:1035`), and the exporter's triangle winding is consistent with it
(`[00, 10, 11]` and `[00, 11, 01]`). Normals must still be produced per mesh
vertex, either by evaluating analytic tangents on the UV grid or by computing
area-weighted face normals and accumulating them. The former is cleaner at
seams and avoids an extra normalization pass.

### 3. No orbit cameras or reference renderer

There is no camera/view-matrix utility in the training pipeline; the pyrender
and Open3D scripts are offline visualizers. NISR needs:

1. NGF-convention `(B, 4, 4)` orbit views;
2. a point-cloud splatter that produces `(B, H, W, 6)` targets;
3. cached targets, batched like the NGF trainer (120 views, 10 per step).

The point-cloud splatter is **not** on this branch. It lives in
`ngf_yl/source/pointcloud_render.py`; only the loss and the NGF-compatible
renderer were ported here.

## Semantic mismatches

### 4. Normalization frames differ

NISR uses `center = mean(points)` and `scale = max |p − center|`
(`utils/utils.py:221-233`). NGF uses the **bbox center** and half the maximum
bbox extent. The loss itself is frame-agnostic, but reusing target maps
rendered in NGF's frame will misalign them with NISR's mesh. Generate the
targets from NISR's normalized cloud (or apply NISR's `meta['center']` /
`meta['scale']` explicitly).

### 5. Duplicate seam vertices

`sample_multi_patch_grid` deliberately keeps per-patch copies of boundary
vertices; its docstring says they can be merged in post-processing
(`utils/utils.py:624`). The positions coincide, so rasterization is
geometrically seamless, but:

- `UniformLaplacian` sees truncated one-rings at each patch boundary and never
  smooths across a seam;
- seam copies receive independent gradients and can drift apart;
- normals computed from one-sided faces can differ across the seam.

The render term can be evaluated as-is, but antialiasing and per-vertex normals
at a duplicated seam should be checked visually; the Laplacian definitely
needs welding or cross-seam accumulation.

### 6. The Laplacian is not exactly NGF's

NGF smooths a **quad patch graph** over uniformly sampled patch vertices. Here,
`UniformLaplacian` smooths triangle edges, which adds the grid diagonals, and
it is applied to every mesh vertex rather than to NGF's uniform samples. Expect
different magnitude and behavior. For a faithful port, build the 4-neighbor
quad adjacency on the UV grid and apply it to the same sampled vertices.

### 7. Adaptive atlases need a different mesh builder

`sample_multi_patch_grid` assumes a uniform `n_rows × n_cols` grid. Adaptive
leaves have different dyadic sizes and hanging vertices, so sampling each leaf
on its own grid creates T-junctions and cracks. A dedicated triangulation (or a
weld plus constrained triangulation) is needed. Fixed-grid single-sheet is the
right first target.

### 8. One layer cannot supervise multiple sheets

The loss rasterizes only the front layer. For `two_sheet` / `six_sheet`, the
back sheet receives no render gradient. NGF's original mesh loss peels three
layers for exactly this reason. Either restrict the render term to the
single-sheet atlas, or extend the renderer to depth peeling and render a
matching multi-layer reference. The Laplacian still applies to the hidden
sheet.

### 9. No alpha mask

Per the requested original objective, the render term is an unweighted L1 over
all pixels and channels. Sparse or holey point-cloud targets will penalize the
mesh where the target is empty. Use dense targets, or add a foreground mask as
a later, explicitly opt-in change.

### 10. Image and camera layout

nvdiffrast writes bottom-up images and the projection flips x. References made
with pyrender/OpenGL (typically top-down, different view convention) must be
flipped and converted before use. A syntactically valid but differently
oriented target will produce a finite loss that cannot go to zero.

## What is already compatible

- For fixed-grid patches, triangle winding matches NISR's
  `n = t_u × t_v` normal convention.
- Once converted to torch and moved to CUDA, the exporter's `(V, 3)` /
  `(F, 3)` arrays fit the loss signature directly.
- 7-channel NGF point-cloud targets are accepted; alpha is ignored, so the
  target layout itself is reusable if the frame is fixed.
- The renderer was checked against NGF's front-layer renderer on BIMBA
  (4 views, 256²): maximum output difference `0.0`.
- The actual `sample_multi_patch_grid` output was fed through the loss
  (2×2 atlas, `R=16`: 1024 vertices, 1800 triangles, 63 duplicated seam
  copies). The reference was non-empty, the self-render term was `0.0`, a
  0.01 perturbation gave a render term of `0.0099`, and gradients were finite.
- Laplacian smoothing matches a naive one-ring mean, and the loss backpropagates
  finite gradients.

## Installation record

Installed into the `seawall` environment (the interpreter used by
`run_seawall.sh`):

- `ninja 1.13.2`
- `nvdiffrast 0.3.3`, editable from `/home/lyangiw/codes/nvdiffrast`
  (commit `729261d`)

CUDA JIT compilation and a 64² render were verified. `nvdiffrast` compiles its
rasterizer on first use and requires the `ninja` executable on `PATH`.
`run_seawall.sh` invokes the environment's Python directly rather than
activating conda, so either activate the environment first or prepend its bin
directory to `PATH` before using the renderer. Neither package is listed in
`environment.yaml` yet.

## Suggested first integration

1. Add a differentiable `sample_patch_mesh(model, resolution, patch_ids)`
   returning `V, F, N` as CUDA tensors, with a modest per-patch resolution
   (start around 24–32) and optional vertex welding.
2. Build `UniformLaplacian` once per topology, not per step.
3. Port the orbit-view and point-cloud splatter utilities, generating targets
   from the input cloud in NISR's own normalization frame.
4. Start with the fixed-grid single-sheet atlas, 8–16 views, and the render
   term as an auxiliary loss; verify that a reference rendered from the current
   mesh gives a render term of zero **and is non-empty** (a clipped camera also
   yields zero).
