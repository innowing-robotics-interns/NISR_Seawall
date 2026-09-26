# Hole cutting and stitching: giving the reconstruction genus

This is the working record for adding genus (handles) to the adaptive cube
atlas (`main.py --adaptive`). It covers:
- what the method does and why;
- the code that implements it;
- how to run it;
- its limitations;
- a log of bugs;
- a list of tasks deferred to the future plan.

**Status (proof of concept).** Detection, cutting, stitching and retraining
are implemented for **one handle**. They are tested on a synthetic cube and end
to end on a torus point cloud (§9), where the result has genus 1 and fits
better than the genus-0 model. They have not been run on real scan data yet.

**Working assumption.** The input has genus 1, and the two membranes that
cover the hole cross each other, so `patch_intersection.py` finds a crossing
curve. This is not guaranteed in general. Making detection robust is left for
later.

---

## 1. The problem

The adaptive model represents the surface as a quadtree over the six faces of
a cube (`AdaptiveCubeComplex`, `model/model.py`). Every leaf is a polygon of
vertex ids (corners plus hanging vertices), and each id is one row of the
learnable matrix `vertex_features` (`n_vertices × d_features`). A query
`(leaf, uv)` is decoded as

```
feature = MVC(vertex_features[polygon ids], polygon uvs, uv)
xyz     = decoder(feature)
```

Neighbouring leaves list the same vertex ids along their shared edge, and on
an edge MVC reduces to linear interpolation between the two endpoints. Seams
are therefore exactly C0.

A subdivided cube is a sphere, so the fitted surface always has genus 0. When
the target has a handle (a torus, a mug handle, a hole through a wall), the
chamfer loss can only wrap it. It stretches a **membrane** over each side of
the hole and pulls the two membranes into the hole until they pass through each
other.

## 2. Detection

Detection reuses the three existing analysis scripts, now called
**in-process** by `model/cutting_hole.py::detect_crossing()` instead of being
run by hand:

| Step | Functions used | Result |
|---|---|---|
| Hole mask | `uv_hole_mask.median_nn_spacing` | each leaf sampled on an R×R grid; a vertex is black (hole) when its nearest target point is farther than `nn_scale × median NN spacing` |
| Openings | `opening_labels.weld_boundary_vertices`, `label_openings` | connected black components across welded leaves, one ID each |
| Crossings | `patch_intersection.build_patch_meshes`, `black_triangle_ids`, `build_seam_tree`, `make_ray_backend`, `detect_intersections`, `assign_openings`, `group_by_opening` | edges of black triangles ray-cast against every other leaf; hits grouped by opening pair |

Each crossing hit is a correspondence: `(patch_a, uv_a)` on opening A and
`(patch_b, uv_b)` on opening B are the same 3D point. Together the hits sample
the **crossing curve** `C = A ∩ B`.

**Choosing the pair (one handle assumed).** Crossings are not only found at
the hole. Where the surface folds through itself inside the object's body, the
folds make openings that cross too. The trained torus in §9 had three crossing
pairs: the hole pair and two folds. The fold pairs had *more* crossings, so
"most crossings" picks the wrong one. The hole's two membranes are the largest
no-data regions, so the pair whose **smaller opening has the largest 3D area**
is taken. `--hole_openings a,b` overrides the choice. Each pair's area,
crossing count and `loop_closedness` are printed and stored in
`hole_summary.json`.

The standalone scripts still work exactly as before. The only change to them
is an import fallback: bare imports when run from `utils/`, package-relative
when imported as `utils.*` by `main.py`.

## 3. Cutting and stitching

The original plan was to cut along the crossing curve `C`. If `C` is a closed
curve on each membrane, it splits each opening into:

- an **inner disk**: the part that has passed through the other membrane.
  It covers the hole and has no target points.
- an **outer annulus**: the part between `C` and the real surface.

```
before                                   after
  surface ── A annulus ── A disk ─┐        surface ── A annulus ──┐
                                  ╳ C                             │ tube
  surface ── B annulus ── B disk ─┘        surface ── B annulus ──┘
```

**Cutting** removes both inner disks and leaves two boundary loops.
**Stitching** glues a tube between the loops.

Removing two disks lowers the Euler characteristic χ by 2. The tube (an
annulus, χ = 0) glued along two circles leaves it there. Since χ = 2 − 2g, the
genus goes up by one. After retraining, the annuli and the tube together become
the inner wall of the hole.

**What the torus test showed.** On a real trained model, `C` was an **open
arc**: the membranes pass through each other only partway, and the ends of `C`
leave the hole region. An open arc encloses nothing, so cutting along `C`
failed. Each *opening*, however, is a disk: a connected no-data region bounded
by the rim of the hole. So there are two cut modes:

| Mode | Removed | Tube joins | Needs |
|---|---|---|---|
| **`opening`** (default) | each whole opening (the membrane) | the two openings' rims | A and B to be disks |
| `crossing` | the ring of leaves `C` crosses plus what it encloses | two loops just outside `C` | `C` to be a closed loop on each membrane |

Both modes make the same topological change (two disks out, one tube in, genus
+1). In `opening` mode the crossing only decides *which* two openings form the
handle.

## 4. Cutting (`model/cutting_hole.py`)

All geometry is handled in **face coordinates** `(face, fu, fv)`, with
`fu, fv ∈ [0,1]` on one cube face, not in leaf indices. Every subdivision
renumbers leaf indices; face coordinates never change.

1. **Consistency check.** Every directed sub-edge `(a, b)` of every leaf
   polygon must have exactly one reversed partner `(b, a)`. This holds on the
   untouched cube complex, including across hanging vertices and cube edges,
   and the later steps rely on it.
2. **Refine** (surface-preserving subdivision). `max_depth` caps the quadtree
   depth, and a floor keeps leaves at least a few cells of the detection grid
   wide: `opening_min_leaf_cells` in `opening` mode, `min_leaf_cells` in
   `crossing` mode.
   - `opening` mode (`refine_openings`): split the leaves an opening touches
     until at least `min_disk_leaves` leaves lie mostly (≥ 50 %) inside it.
     How much of a leaf an opening covers (`opening_fraction`) is measured from
     the opening's own detection-grid vertices, each placed in the leaf that
     contains it. Sampling each leaf instead can miss an opening that is small
     compared with its leaf (bug #5).
   - `crossing` mode (`refine_along_curves`): split the leaves `C` passes
     through until `C` touches `min_loop_leaves` leaves. Here `min_leaf_cells`
     also keeps leaves larger than the gaps the seam guard leaves in `C`
     (about 3 grid cells), otherwise step 3 leaks.
3. **Select what to remove**, once per opening.

   `opening` mode (`select_opening_disk`):
   - take the leaves at least 50 % covered by the opening;
   - keep the largest connected piece;
   - fill in any islands of other leaves it encloses, so there is one boundary
     loop.

   `crossing` mode (`select_inner_disk`):
   - **Ring**: the leaves that contain a crossing point (grown by `cut_dilate`
     rings if asked).
   - **Seeds**: every leaf with no sample labelled with this opening.
   - **Flood fill** from the seeds over edge-adjacent leaves, never entering
     the ring. The leaves the fill cannot reach are the inner disk.
   - **Removed set** = ring ∪ inner disk.

   Removing the ring too puts each loop just outside `C`, so the tube starts
   with a finite width.

   Loop lengths: in `opening` mode the two rims usually differ (28 and 20
   vertices on the torus). §5.1 explains why that needs no new loop vertices.
4. **Validate:**
   - the two removed sets are disjoint;
   - they share no vertex;
   - each has χ = 1, i.e. it is a disk.
5. **Remove** (`AdaptiveCubeComplex.remove_patches`). The leaves are flagged
   `removed` and dropped from `leaf_patches`. Their vertex rows stay in
   `vertex_features`, because ids are never renumbered.
6. **Extract the loops** (`boundary_loops`). Boundary sub-edges are directed
   edges of kept leaves with no reversed partner; chaining them gives exactly
   two closed loops. Each loop is ordered in the kept leaves' direction.
7. **Freeze** the kept leaves that own a loop edge. Subdividing one would add
   a vertex to the loop that the tube does not know, which opens a crack.

After the cut, χ goes from 2 to 0 (a sphere with two holes).

## 5. Stitching (`model/stitching_hole.py`)

The tube is an N (columns, around the loops) × R (rows, from A to B) grid of
cells. Each cell is a `_TubePatch`: a polygon of vertex ids and uvs on the unit
square's boundary. It sits in `leaf_patches` after the quadtree leaves and
decodes through the same MVC path, so its seams are exactly C0 for the same
reason quadtree seams are.

1. **Orientation** (no choice involved). Each shared edge must be walked in
   opposite directions by its two cells. The kept leaves walk the loops in
   record order, so the tube's bottom edge walks loop A **reversed** and its
   top edge walks loop B **forward**.
2. **Phase** (`align_loops`). Each crossing point is mapped to its nearest
   vertex on loop A and on loop B. The offset between their arc-length
   positions (a circular mean) aligns B to A, so the tube doesn't twist.
   - `phase_concentration` in the record measures how well the crossing points
     agree: 1 = perfectly, 0 = not at all.
   - If running the loops opposite ways fits much better, a warning is printed.
3. **Corners.** N corner vertices are chosen on each loop:
   - evenly by index on A;
   - on B, the vertex nearest in arc length to each A corner, forced to be
     strictly increasing.
   - `N = min(len A, len B, 64)` unless `tube_columns` is set.
4. **Cells.**
   - **Row 0:** the bottom edge of column k is the whole chain of loop-A
     vertices between corners k and k+1, placed as hanging vertices along
     `v = 0` by arc length.
   - **Row R−1:** the top edge carries the loop-B chain along `v = 1`.
   - **Middle rows:** plain quads.

### 5.1 The vertex matrix after stitching

- **Loop vertices are reused, not duplicated, averaged or moved.** The tube
  cells list the same ids the cut surface uses, so both read the same rows of
  `vertex_features`. That shared row is what makes the seam exact. Averaging
  or overwriting it would move the surface on the kept side.
- **Different loop lengths need no new loop vertices.** If loop A has 40
  vertices and loop B has 28, the end cells absorb the difference: one column
  might carry 2 A-edges and 1 B-edge. This is the same hanging-vertex
  mechanism quadtree leaves use. Inserting new vertices into a loop would
  require splitting the frozen kept leaves' edges instead.
- **New rows.** `(R − 1) × N` rows are appended after the existing ones, one
  per interior ring vertex. Ring r of column k is initialised as

  ```
  z_new[r, k] = (1 − r/R) · z[A corner k] + (r/R) · z[B corner k]
  ```

  This is a straight blend of the two matched loop corners.
- **Welding alternative (not used).** Loops A and B could be welded directly,
  with no tube, by giving each matched pair of vertices one shared row.
  - The natural value for that row would be the mean `(z_a + z_b) / 2`,
    because both decode to nearby points.
  - This isn't done for three reasons. The loops lie on opposite sides of `C`,
    not at the same point. Averaging would move both surfaces. And welding
    would mean rewriting the ids of frozen quadtree leaves, which the complex
    doesn't support.
- **Orphaned rows.** The rows of removed leaves stay in the matrix. They get
  no gradient.

After stitching:
- the surface is closed (no edge without a partner);
- χ = 0, i.e. genus 1;
- `seam_gaps` measures the 3D gap across every tube edge.

## 6. Training after the stitch

`main.py --adaptive --cut_hole` runs three phases:

1. **Box pretraining, then `train_adaptive` for `--epochs`.** Unchanged.
2. **Detect, cut and stitch** (`utils/cutting_hole.py::cut_and_stitch_hole`).
   - `checkpoint_before_hole.pt` is saved first.
   - `checkpoint_hole_stitched.pt` is saved after stitching.
   - PLYs and `hole_summary.json` go to `cutting_hole/`.
3. **`train_adaptive` again for `--hole_epochs`** (−1 = `--epochs`), with the
   **same data sampling and loss** (chamfer + μ tangent + γ normal + SVD).
   - It is a new call, so it builds a **fresh Adam**, the LR restarts at
     `--lr` with a new cosine schedule, and the μ/γ/SVD warmups and delays
     restart from epoch 1.
   - Subdivision restarts too; tube cells and loop leaves are frozen.
   - Its files carry a `hole_` prefix: `checkpoint_hole_<epoch>.pt`,
     `patch_config_hole_<epoch>.png`, `history_hole.png`,
     `vertex_positions_normalized_hole.json`.

If detection finds no crossing, a message is printed, the hole phase is
skipped, and the run finishes normally. Any other cutting error stops the run
after `checkpoint_before_hole.pt` is saved. You can then retry with the
standalone script and different `--hole_*` settings.

## 7. Usage

```bash
# all in one run
python main.py --adaptive --file data/x.ply --epochs 5000 \
    --cut_hole --hole_epochs 5000

# or on an existing checkpoint
python utils/cutting_hole.py --ckpt logs/.../checkpoint.pt      # detect + cut + stitch
python main.py --adaptive --load_ckpt logs/.../cutting_hole/checkpoint/checkpoint_hole.pt \
    --epochs 5000 ...                                           # retrain (fresh optimizer)
```

Settings (`HoleConfig`, passed as `--hole_<name>` to both `main.py` and the
script):

| Option | Default | Meaning |
|---|---|---|
| `resolution` | 128 | per-leaf grid for the mask and the intersection meshes |
| `nn_scale` | 5.0 | hole threshold, in multiples of the cloud's median NN spacing |
| `weld_tol`, `min_opening_vertices` | 1e-5, 10 | as in `opening_labels.py` |
| `triangle_rule`, `seam_eps_scale`, `eps_t`, `backend` | any, 1.5, 1e-4, auto | as in `patch_intersection.py` |
| `openings` | '' (auto) | force the pair, e.g. `0,3` |
| `cut_mode` | opening | `opening` or `crossing` (§3) |
| `min_disk_leaves` | 16 | [opening] refine until each opening covers this many leaves |
| `min_loop_leaves` | 24 | [crossing] refine until `C` spans this many leaves |
| `min_closedness` | 0.75 | [crossing] warn when `C` is less loop-like |
| `max_refine_rounds`, `max_depth` | 8, 12 | refinement limits |
| `opening_min_leaf_cells` | 2 | [opening] leaf-size floor, in mask-grid cells |
| `min_leaf_cells` | 8 | [crossing] leaf-size floor, in mask-grid cells (above the seam-guard gaps) |
| `cut_dilate` | 0 | [crossing] grow the ring to close gaps |
| `allow_non_disk` | off | keep going when a removed region is not a disk |
| `tube_columns` | 0 (auto) | N |
| `tube_rows` | 4 | R |

`main.py` also takes `--hole_cut_only` and the script takes `--no_stitch`. Both
cut without stitching, for debugging.

Outputs in `cutting_hole/`:

| File | Content |
|---|---|
| `crossing.ply` | the crossing curve C |
| `loops.ply` | loop A red, loop B blue |
| `cut_surface.ply` | after the cut; loop leaves yellow |
| `stitched_surface.ply` | tube cells orange |
| `hole_summary.json` | detection stats, cut diagnostics, stitch info, seam gaps, χ and genus |

## 8. Code map

| File | Role |
|---|---|
| `model/cutting_hole.py` | `HoleConfig`, `detect_crossing`, `cut_hole` and helpers (point location, opening lookup, leaf graph, refinement, disk selection, loops, Euler characteristic) |
| `model/stitching_hole.py` | `build_tube`, `stitch_hole`, `align_loops`, `seam_gaps` |
| `utils/cutting_hole.py` | `cut_and_stitch_hole` (detect → cut → stitch → checks → PLYs), and the CLI |
| `model/model.py` | `_Patch.removed/frozen`; `_TubePatch`; `add_tube`, `remove_patches`, `freeze_patches`, `n_quad_leaves`; `cube_xyz` returns 0 for tube cells; `serialize`/`replay` store removed/frozen leaves and tubes |
| `model/subdivision.py` | frozen leaves are never split; the cube-key seam check covers quadtree leaves only |
| `main.py` | `--cut_hole` phase in `run_adaptive`; `train_adaptive(tag=...)` |
| `utils/{patch_vis,uv_hole_mask,opening_labels,patch_intersection}.py` | import fallback so `main.py` can import them |

**Checkpoint topology.** `topology` gains `removed`, `frozen` and `tubes`. Each
tube stores:
- its cells;
- its new vertex keys and first id;
- `subdiv_log_len`, the number of subdivisions done before it was added.

`replay()` inserts the tube at that point in the subdivision log, so vertex ids
come out in the original order even when training subdivided after the stitch.
Old checkpoints load unchanged. The hole record is stored under the `hole` key
of the checkpoint.

## 9. Testing

**Synthetic cube** (`base_subdivisions=2`, openings on +Z/−Z, a circular
crossing curve), in both cut modes:
- Cut, `crossing` mode: 80 leaves removed, loops of 32/32 vertices, χ 2 → 0.
- Cut, `opening` mode: 48 leaves removed, loops of 24/24 vertices, χ 2 → 0.
- Stitch: 0 open edges, χ = 0 (genus 1), max tube seam gap 2e-8 in both modes.
- Training after the stitch: 20 epochs with two subdivision rounds. Tube and
  loop leaves were never split, the seams stayed at 3e-8, and the surface
  stayed closed.
- Save and reload reproduces the model exactly, including the tube inserted
  between subdivisions.

**End-to-end on a torus cloud** (20k points, R = 1, r = 0.35; a small CPU
config: `W 128`, `D 4`, `d_features 32`, `base_subdivisions 2` = 96 leaves,
300 pretraining + 800 training epochs):
- **Detection** (mask resolution 96) found 8 openings and 3 crossing pairs:
  - (0,3): the hole membranes, above and below the hole, inside its radius;
  - (4,5) and (6,7): folds of the surface inside the torus body.

  The folds had more crossings, and every crossing curve, including the hole's,
  was an **open arc**, not a loop (bug log #3, #4).
- **Cut** in `opening` mode, selecting pairs by opening area:
  - openings 0 and 3 are chosen (area 0.66–0.77 vs 0.11–0.15 for the folds);
  - after refinement, 27 + 16 leaves are removed;
  - the rim loops have 24 and 18 vertices.
- **Stitch:** an 18 × 3 tube with 36 new vertices, phase fit 0.98, closed,
  χ = 0 (genus 1), seam gap 1.6e-7.
- **Retraining** (`--epochs 0 --load_ckpt <trained> --cut_hole --hole_epochs 300`):
  - μ_eff restarted at 0 and warmed up again;
  - the chamfer distance fell from 0.067 right after the stitch to 0.036 at
    epoch 300. Before the cut the genus-0 model had plateaued at 0.051;
  - the tube seams stayed at 2e-7 and the surface stayed closed.
- **Hole check** (dense samples of both models, against the target):

  | Model | Surface inside the hole (r < 0.8 × inner radius) | Mean distance | p99 distance |
  |---|---|---|---|
  | before (genus 0) | 0.47 % (membranes) | 0.0175 | 0.125 |
  | after (genus 1) | 0.00 % | 0.0104 | 0.032 |

  In a top view, the membranes inside the hole are gone and the tube has
  become the hole's inner wall.
- **Not yet run** on a real scanned input, or with larger settings on GPU.

## 10. Limitations

**Detection**
- **One handle, and the membranes must cross.** Handles whose membranes never
  meet are not found.
- **Pair selection is a heuristic.** The crossing pair whose smaller opening
  has the largest 3D area is cut. That fits one hole with fold artefacts
  elsewhere (the torus), but a large fold could win. Use `--hole_openings a,b`
  to override.
- **Mask threshold.** `nn_scale` decides what counts as a hole: too tight and
  the openings fragment, too loose and they miss the rim.

**Cutting**
- **Adaptive atlas only.** The fixed-grid atlases (`single_sheet`,
  `two_sheet`, `six_sheet`, used by `run.sh`) are not supported.
- **`opening` mode needs each opening to be a disk** and the two to be
  separate. An opening with a hole in it, or two openings that touch, stops the
  run.
- **Opening shape follows the mask.** The removed region is only as good as
  the `nn_scale` threshold.
- **`crossing` mode needs `C` to be a closed loop** on each membrane. On the
  trained torus it wasn't (§3). An open `C` stops the run with an
  open-ring error.
- **The loop is a staircase** along leaf edges. Its accuracy is limited by
  leaf size, which `min_leaf_cells` ties to the detection grid.
- **Real surface can be cut** (reported as `n_removed_outside_opening`). In
  `opening` mode these are enclosed islands filled in; in `crossing` mode,
  leaves inside the ring.

**Stitching**
- **Fixed tube resolution.** Tube cells are never subdivided, so the tube's
  resolution is set by N × R at stitch time.
- **Frozen loop leaves** cannot gain resolution after the cut.
- **Initial tube shape.** It is a straight feature blend between the two
  loops, not a geometric fit. Retraining has to open it up.
- **Twists.** A phase error twists the tube. `phase_concentration` reports the
  fit but doesn't correct it.
- **Positional encoding.** Tube cells have no cube position, so with `L > 0`
  the positional encoding sees 0 on the tube; use `L = 0` (the adaptive
  default).
- **Diagnostics coverage.** `uv_hole_mask.py` and `opening_labels.py`
  atlas figures place leaves by cube face, and tube cells (face −1) are left
  out. The distortion subdivision still scores tube cells but never splits them.

## 11. Bug log

Bugs found or reported, with their status. New reports are added here.

| # | Date | Reported by | Bug | Status |
|---|---|---|---|---|
| 1 | 2026-09-24 | found in review | `main.py` passed `checkpoint_extra=` twice to `train_adaptive` in `run_adaptive`: `SyntaxError`, `main.py` could not run | fixed |
| 2 | 2026-09-26 | found in review | `utils.load_point_cloud` sets `center = None; scale = None` right before using them, so the `center`/`scale` arguments are ignored. `uv_hole_mask.py` passes the checkpoint's normalization and silently gets a per-file one; with a different downsample the frame shifts slightly. `utils/cutting_hole.py` works around it by normalizing with the checkpoint's values itself. | open, not changed (the comment above it suggests it may be intended) |
| 3 | 2026-09-26 | found in torus test | Cutting along the crossing curve failed ("ring does not enclose any leaf"). On a trained model the membranes cross only partway, so `C` is an open arc, and an arc can't bound a region. | fixed: added `opening` mode (remove whole openings, stitch rim to rim) and made it the default; `crossing` mode kept |
| 4 | 2026-09-26 | found in torus test | The pair with the most crossings was a fold inside the torus body, not the hole. A closed-loop score (`loop_closedness`) couldn't tell them apart either (arc curves score 0.6–0.8). | fixed: pick the pair whose smaller opening has the largest 3D area; `--hole_openings` override |
| 5 | 2026-09-26 | user (`run.sh`, rocker-arm.ply) | `opening` mode stopped with "opening 1: no leaf lies mostly inside it". After 5000 epochs the model had only 48 coarse leaves, and opening 1 was small. Coverage was measured with 4×4 samples per leaf, which fell between the opening's cells, so every leaf read 0 % and refinement never split the leaves around it. | fixed: coverage now comes from the opening's own grid vertices (`OpeningLookup.opening_points`); the leaf-size floor is 2 grid cells in `opening` mode (`opening_min_leaf_cells`, was the 8 meant for `crossing` mode); `max_refine_rounds` 4 → 8. Reproduced with a tiny opening in a coarse leaf (old measure: 0 leaves); now cut and stitched, genus 1. |

## 12. Future plan (to be implemented)

Tasks deferred on request are recorded here.

| # | Date | Task | Notes |
|---|---|---|---|
| – | – | (none yet) | – |
