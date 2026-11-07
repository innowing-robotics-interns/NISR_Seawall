# `rosa_main.py` + `loop_extraction.py`: pipeline and pseudocode

## Purpose

`rosa_main.py` is the only entry point: the command line, and with `--train` the fitting of every loop with `mini_nn`. `loop_extraction.py` has no command line of its own; it holds the geometry and `LoopExtractor`, one object per point cloud that runs the extraction and keeps each stage's result on itself.

```bash
# extraction only
python rosa_main.py --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --field --samples 15

# extraction, then mini_nn fitting
python rosa_main.py --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --field --train --samples 15 --epochs 1000 --normal 0.01

# the same density, reached as the fit goes: 5 planes per axis to start with, two
# more per axis every 100 epochs, capped at the 15 above by epoch 500
python rosa_main.py --path normalized_shapes_cuboid/dancing_children100K_cubic_lam3.ply \
    --field --train --samples 5 --add-planes 2 --samples-end 15 --refine-every 100 \
    --epochs 1000 --normal 0.01
```

The pipeline has two largely independent stages:

1. **Geometry extraction:** point cloud -> slices -> point groups -> implicit fields -> loop centres and assigned loop points.
2. **Optional curve fitting:** each extracted loop becomes one target for `mini_nn.Trainer`.

The main idea is to use point normals to determine which regions are enclosed by a slice. A positive connected component of the solved implicit field represents a candidate loop interior.

---

## 1. End-to-end overview

```text
load_points(path)
    |
    v
Create equally spaced planes along x, y, and z
    |
    v
For every plane:
    Select points in a thin slab around the plane
    Remove points whose surface lies flat in the plane
    Project the remaining points and normals into 2D
    |
    v
    Group nearby projected points with slice_loops()
    |
    v
    For every sufficiently large group:
        Build a Poisson implicit field from points and normals
        Find positive connected components
        Reject only components below the minimum area thresholds
        Recursively re-solve each accepted component from its own points
        Keep all newly discovered subcomponents
        If no component survives, use the normal-line least-squares fallback
    |
    v
Assign slice points to the final centres and form 3D loop point sets
    |
    v
Save slice/field visualizations and PLY files
    |
    v
Optional: train one shared miniNN on all loops and save fitted curves
```

After `LoopExtractor.extract()`, the object carries:

```text
points    N x 3 input points
normals   N x 3 input normals, or None
slices    one input-point index array per plane
centres   per plane: (centres, RMS residuals, point labels)
loops     one (slice index, centre, points, normals) tuple per extracted loop
p_normal  one unit normal per plane
p_point   one point on each plane
```

The stages are also callable one at a time, each returning the object: `load()`, `make_planes()`, `cut()`, `find_centres(assign=False)`, `group_loops()`, `save(figures, n_arrows, field, assign)`. `extract()` is those six in order.

---

## 2. Loading the point cloud

Function: `load_points(path)`

```text
Read the PLY without reordering or processing its vertices
points = mesh.vertices

if raw PLY vertex data contains nx, ny, nz:
    read normals from those fields
else if the loaded mesh exposes vertex_normals:
    use mesh.vertex_normals
else:
    normals = None
```

The raw PLY fields matter because a faceless PLY loads as a `trimesh.PointCloud`, which does not expose `vertex_normals` in the same way as a mesh.

Normals are essential for centre detection. Without them, slicing still works, but Poisson fields and normal-derived loop centres cannot be computed.

---

## 3. Creating slices

Functions: `LoopExtractor.make_planes()`, `LoopExtractor.cut()` and `slice_point_cloud()`

### 3.1 Plane placement

For `--samples S`, the code creates `S` planes along each axis, giving `3S` planes in total.

```text
lo, hi = input bounding box
margin = margin_frac * bounding_box_extent

for axis in [x, y, z]:
    offsets = evenly spaced values from lo + margin to hi - margin
    create one plane at each offset
```

The margin prevents planes from merely grazing the extreme ends of the shape.

`make_planes(offsets=…)` takes one sorted array of positions per axis instead of `--samples` even ones. That is what a refinement stage passes: it keeps every plane already cut and bisects some of the gaps, which a count cannot say, and passing the positions is what makes a plane, its slice and its loops the same across the stages.

### 3.2 Selecting points for each plane

The slab half-width is:

```text
threshold = band * longest_bounding_box_side
```

The default `band` is `0.0075`, or 0.75% of the longest side.

```text
for every input point p and plane (normal n, point q):
    signed_distance = dot(p - q, n)
    keep the point if abs(signed_distance) < threshold
```

### 3.3 Removing flat-surface points

A point whose normal is almost parallel to the plane normal lies on a surface running flat through the slice. Its in-plane normal contains little useful contour information.

```text
if normals are available:
    normalize every input normal
    remove the point when abs(dot(point_normal, plane_normal)) >= max_align
```

The default `max_align=0.95` removes points within approximately 18 degrees of the plane normal. This filtering is performed independently for every plane.

---

## 4. Projecting a slice into 2D

Function: `plane_basis()`

For a plane with unit normal `n`, the code constructs two perpendicular unit vectors `u` and `v` spanning the plane.

```text
relative_points = slice_points - plane_point

uv[:, 0] = dot(relative_points, u)
uv[:, 1] = dot(relative_points, v)

nv[:, 0] = dot(slice_normals, u)
nv[:, 1] = dot(slice_normals, v)

normalize each 2D normal nv[i]
```

After this step, centre detection is a 2D problem. The original 3D coordinates are restored only after the 2D centres have been found.

---

## 5. Initial point grouping

Function: `slice_loops(uv, k=8, c=2.0)`

This is a geometric grouping step; it does not use normals.

```text
Build a KD-tree for the 2D slice points
For each point i:
    spacing[i] = distance to its k-th nearest neighbour

For candidate point pairs (i, j):
    connect i and j when:
        distance(i, j) < c * max(spacing[i], spacing[j])

Find connected components of the resulting graph
Return one group label per point
```

The adaptive distance is important because sampling density can vary greatly along one contour. Groups containing fewer than `min_points=100` points are ignored by centre detection.

A group is not necessarily one loop. Two nearby loops can merge into one group, which is why the normal-derived field is needed next.

---

## 6. Building the implicit field

Function: `poisson_field(uv, nv)`

The outward normals of a closed contour describe the boundary of an interior region. The code turns these normals into a 2D indicator-like field `chi`.

```text
Create a regular grid around the group's 2D bounding box
Bilinearly splat each point's 2D normal onto the grid
Blur the normal vector field V
Compute divergence(V)
Solve: Laplacian(chi) = -divergence(V), using an FFT
Remove the constant mode during the solve
Blur chi slightly

return chi, grid_origin, grid_cell_size
```

Interpretation:

- `chi > 0`: candidate interior.
- `chi < 0`: candidate exterior.
- The zero level separates the two sides.
- The field is not a calibrated signed-distance function.
- Its amplitude can vary greatly between groups, so rendering normalizes each field by its own peak.

---

## 7. Finding and validating positive components

Function: `field_parts(uv, nv, ...)`

This function performs the first two stages of the centre algorithm: build the field, then find positive connected components.

```text
function field_parts(points, normals):
    if fewer than 3 points or all points coincide:
        return no components, all labels = -1

    chi, origin, h = poisson_field(points, normals)
    raw_components = connected_components(chi > level * max(chi))

    accepted = []
    for each raw component:
        mask = cells belonging to this component
        mass = chi inside mask, zero elsewhere

        if total mass <= 0:
            reject

        Compute the chi-weighted centroid
        Snap the centroid to the nearest positive cell in this component
        Store (component id, snapped centre, area)

    Remove components that are too small relative to:
        the largest component, and
        the group's own bounding-box area

    Assign every input point to a retained component:
        use the component at its own grid cell when possible
        otherwise use the nearest retained component

    Remap component ids to consecutive labels 0, 1, 2, ...
    return accepted centres and point labels
```

Important defaults:

| Parameter | Default | Meaning |
|---|---:|---|
| `level` | `0.0` | Positive field means `chi > 0` |
| `min_frac` | `0.02` | Drop components smaller than 2% of the largest |
| `min_area` | `0.01` | Drop components below 1% of the group box area |

There is no rim-support rejection. Consequently, an open arc that produces a sufficiently large positive component is retained as a loop candidate.

“Connected component” here means a connected component on the grid. The code does not separately test the topological property of being simply connected, such as whether the component contains holes.

### Why the centre is snapped

A weighted centroid of a concave component can lie in a hole or outside the component. Therefore, the code selects the positive cell nearest that centroid. For accepted components, this guarantees that the returned centre samples a positive field value.

---

## 8. Recursive normal-based splitting

Function: `field_loops(uv, nv)`

After the initial positive components are found, each component is checked again using only its assigned points and normals.

```text
function field_loops(points, normals):
    initial_parts, initial_labels = field_parts(points, normals)

    if initial_parts is empty:
        return least_squares_normal_line_centre(points, normals),
               all point labels = 0

    final_centres = []
    final_labels = array of -1

    function split(point_ids, current_centre):
        if fewer than 3 points:
            keep current_centre for these points
            return

        subparts, sublabels = field_parts(points[point_ids],
                                          normals[point_ids])
        Discard subparts containing no assigned points

        if zero or one nonempty subpart remains:
            keep the re-solved centre when available
            otherwise keep current_centre
            return

        for every nonempty subpart:
            split(its point ids, its centre)

    for every initial component:
        split(points assigned to it, its centre)

    return final_centres, final_labels
```

This recursion is what allows one initially connected positive region to separate into several loops when its own point normals support several smaller enclosed regions.

The previous implementation kept only the first re-solved centre. The current implementation retains all nonempty subcomponents and remaps the point labels to the final centres.

---

## 9. The least-squares fallback

Function: `fit_centre(uv, nv)`

When `field_parts()` finds no acceptable enclosed component, `field_loops()` falls back to a least-squares intersection of normal lines.

For a candidate centre `c`, each point-normal pair contributes the projection:

\[
P_i = I - n_i n_i^T.
\]

The solve is approximately:

\[
\min_c \sum_i \|P_i(c-p_i)\|^2.
\]

```text
Build the sum of projection matrices
Build the corresponding right-hand side
Solve the small 2 x 2 least-squares system
Return the resulting centre
Assign every point in the group to that centre
```

**Important:** this fallback is not constrained to lie in a positive field component. It is intended for open arcs, where no reliable enclosed region exists. Consequently, a fallback centre can appear outside the red region in a field plot.

---

## 10. Collecting centres and loop points

Functions: `loop_centres()` and `assigned_loops()`

For every sufficiently large point group:

```text
centres_2d, labels = field_loops(group_points_2d, group_normals_2d)

for every final centre:
    own_points = points whose label names this centre
    if own_points is empty:
        skip this centre

    RMS = root_mean_square(distance from centre to each point's normal line)
    centre_3d = plane_point + centre_u * u + centre_v * v
```

The RMS value measures how well the normals agree on the centre. It is not the same as the later neural curve-fitting error.

`assigned_loops()` then gathers the original 3D points and normals for every centre, producing the training targets.

---

## 11. What the field image shows

Functions: `slice_fields()` and `save_field_grid()`

The field image is generated by recomputing and drawing the raw Poisson field for every sufficiently large point group.

```text
for each group:
    chi = poisson_field(group points, group normals)
    normalize chi by its maximum
    draw cells whose absolute normalized value is at least FIELD_FLOOR
    draw slice points and final centres over the field
```

The renderer does **not** apply the `field_parts()` area filters before drawing a positive region. A very small red region can therefore be drawn even when it is too small to become an accepted loop component. Larger positive regions are retained without a rim-support test.

A centre can still appear outside every red region when no sufficiently large positive component exists and the least-squares fallback is used.

---

## 12. Concrete example: `z = -0.23`

For `dancing_children100K_cubic_lam3`, the affected plane is exactly:

```text
z = -0.2281449965
```

The affected point group contains 3,433 points.

Before rim-support rejection was removed, approximately 29% of this component's rim cells were unsupported. That exceeded the old 15% limit, so the component was rejected and the normal-line fallback produced a centre outside the red region.

With the simplified positive-component rule, the sequence is now:

```text
Build Poisson field
    -> one positive component is found

Apply only the area filters
    -> component is large enough and is retained

Compute and snap its centre
    -> final 2D centre = (-0.28329492, -0.49570365)
    -> normalized field at the centre = 0.8693, inside the red region
```

The original problem was therefore the rim-support rejection followed by the unconstrained fallback, not a coordinate-projection error.

---

## 13. Optional neural curve fitting

This stage runs only with `--train`.

```text
for every extracted loop:
    target = loop_points - point_on_its_plane
    target_normals = corresponding input normals

row_of = {}      # (axis, offset, slot) -> embedding row, handed out once, never moved

# --refine-every a > 0 splits the run into stages, one plane set each (see below)
plan = refine_plan(--samples, epochs, a, --add-planes, --samples-end)
for stage, (planes, stage_epochs) in enumerate(plan):
    if stage:
        offsets, inserts = bisect_offsets_trace(each axis, to `planes` positions)
                      # inserts: (inserted, left, right) per added plane
        found.restage(offsets)      # only the added planes are cut
        previous = the stage before's network
    codes = [row_of.setdefault(key, len(row_of))            # a row per plane, stable
             for key in plane_keys(p_normal, p_point, loops)]
    trainer = mini_nn.Trainer(loops of this plane set,
                              num_models=len(row_of), codes=codes)
    if stage:
        trainer.inherit(previous trainer)   # shared network + the rows already handed out
        blend_added_codes(trainer, inserts, previous row count)
        # --code-init random calls random_added_codes instead; nothing is re-warmed
    elif warmup is not None:
        mini_nn.load_warmup(warmup, trainer.model)  # cube checkpoint, stage 0 only
    trainer.train(stage_epochs, every=1, on_epoch=watch)
    # --normal is the weight the direction term starts at on each stage
    # every --snapshot-every epochs: predict, measure, and write the fit grid to snapshots/
found.resave()      # the artefacts are written once, for the plane set it ended on

# predict once per loop; everything reported comes from these
for every loop i:
    prediction = trainer.predict(i)
    tangent = trainer.predict_tangent(i)
    predicted_normal = tangent x plane_normal       # kept at tangent length
    prediction_3d = prediction + point_on_its_plane

# measure from those same predictions, one nearest pass per loop
for every loop i:
    distances, indices = nearest target point of every predicted point
    point_errors = distances
    paired_normals = target_normals[indices]
# the normal the fit implies, tangent x plane normal, against the given one
align error = mean(1 - cos(tangent x plane normal, paired normal))   # when --normal > 0

# turn every loop the fit ran against its points' normals (see "Predicted normals"):
# the mean cosine against the paired normal, unit-normalized, below -0.1
for every loop i where mean cos(tangent x plane_normal, paired_normals) < -0.1:
    reverse prediction, tangent, predicted_normal and point_errors together

# report bend / align / error range and how many loops were turned, then write the
# PLYs and the figures
```

The network uses a periodic parameter \(z\): it is mapped through sine and cosine so the two ends of a fitted loop meet.

The direction term's weight is **annealed** rather than held: `--normal` is the weight on the first batch, and `cosine_anneal(step, epochs * batches)` takes it down to nothing on the last, along half a cosine. The term settles which way round each loop runs, and that is decided while the fit is still coarse — early, two stretches of a contour passing close are told apart by the way they run rather than by where they are — while later it can only pull a fit that already sits on its points. The schedule counts batches rather than epochs, so it is smooth inside an epoch and a one-epoch run reaches its end just as a fifty-epoch one does. The `align` column of the epoch line is the term itself, unweighted, so it stays comparable across epochs; the weight it was taken at is printed beside it as `(w …)`. There is no bend term in the fit: the run prints the curve's bend at the end as a
reading (a circle is 39.5) and nothing optimises it.

The run can also be watched while it is going. `--snapshot-every a` (50 by default, `0` for none) hands `Trainer.train` a callback that fires after epoch 50, 100, … and writes the fit grid of that moment to `snapshots/<stem>_fits_eNNNN.png`. It is the grid the run writes at its end — the same panels, the same colours, the same red `--max-err` limit — drawn from the network as the epoch line above it left it, measured by the very pass the end of the run is measured by, so a snapshot and the final figure can be laid side by side. A prediction pass per snapshot is what it costs; the alternative, cutting the run into 50-epoch calls, would restart the learning-rate and direction-term anneals at every cut.

### Refining the planes during the run

`--refine-every a` (0 by default, which fits the one plane set the extraction cut) makes the run grow its own sampling: every `a` epochs it bisects the widest gaps of each axis, adding `--add-planes` planes per axis (`2` by default), and carries the fit over to that plane set. `--samples-end` caps the count, so a long run reaches the density it wants early and spends the rest of its epochs on the fit at that density; without it the last stage ends on `--samples` plus `--add-planes` per stage, which is `--samples 3 --epochs 1000 --refine-every 100` ending on 21 planes per axis.

The refinement is **nested**, and that is the whole of why it can carry a fit over. Bisection splits a gap, so every plane of the stage before is still there, at exactly the same offset: its slice holds the same points, its loops are the same loops, and its targets are unchanged. `LoopExtractor.restage()` re-cuts only the planes that are new, naming a plane by axis and offset rather than by index — a stage inserts planes, so index 5 names a different plane afterwards. That is also what keeps a staged run affordable: cutting is the expensive half of this pipeline (on the hand, 27 planes take 14 s to cut against 2.3 s per 100 epochs of fitting), so re-cutting every plane would cost more than the training it is refining.

### A plane's feature code

An embedding row belongs to a **plane**, not to a position in a list. `plane_keys(p_normal, p_point, loops)` gives every loop the key `(axis, offset, slot)` — which axis it was cut across, where along that axis, and which of that plane's loops it is (a plane crossing two rings needs two rows, so slot 0 is the plane's own code) — and `rosa_main` keeps one dict, `row_of`, from key to row. A key seen before gets its old row; a key seen for the first time is given `len(row_of)`, at the end of the table. Nothing is ever renumbered or reused, so the assignment is fixed for the whole run and no stage has to decide which loop continues which.

This is what makes the hand-over a copy rather than a guess: because every plane the extractor cut is still cut in every later stage, a stage's rows are exactly the rows it inherited plus one per added plane, i.e. the earlier table is a **prefix** of the later one. `Trainer.inherit(previous)` therefore copies the shared MLP whole and the embedding table row-for-row up to the last row it was given (`source` remains for a caller that does not hand out rows this way). A loop is new when its code is past the previous table's row count; those rows are the ones the stage seeds before it trains.

The same codes are what `predict(i)`, `predict_tangent(i)`, `bend()` and the training loop index the table with; `Trainer.codes` holds one per point set, and `axes` (the plane normal the direction term turns about) stays one per point set, indexed by the point set rather than by its row.

This subsection describes embedding mode. The default, `--feature triplane`, has no row per loop at all: a loop is identified by where it is. See "The triplane feature map" below.

The cube warm-up is loaded **once**, into stage 0, and never again. A later stage inherits the network and field, seeds only the rows of its added planes (embedding mode), and then fits the whole table together; nothing is re-warmed. Re-warming a stage was measured to damage the inherited fit rather than protect it: on the hand, a stage handed 8 fits of its own and 4 new loops stepped off from a fit error of 0.006 to 0.18 in those 30 steps, and the 10 epochs of fitting that followed could not put them back. Removing the per-stage warm-up and seeding instead halved the 200-epoch median (0.0178 → 0.0085) and nearly halved the 1000-epoch one (0.0087 → 0.0047), and the retained planes improved too (0.0085 → 0.0051 at 1000 epochs).

#### Seeding an added plane's row (`--code-init`)

A row added by a stage starts from one of two places, chosen by `--code-init` (default `blend`):

- **`blend`** — `bisect_offsets_trace` records, for every inserted plane, the two planes it was inserted between. The row is written `(1-t)·w_L + t·w_R`, with `t` the insertion's position across the gap and rows matched by slot; a side without that slot is left out of the blend. If neither neighbour holds the slot, the row is drawn at random instead (at the same σ as below), so no added row is left at the zero feature. Insertions are visited in the order bisection made them, so a plane added into a gap an earlier insertion already split reads rows that are already seeded. This is the default because a new loop then starts as the shape its two neighbours agree on.
- **`random`** — each added row is drawn independently from `N(0, σ)`, with σ the standard deviation of every inherited row, so a new loop starts at the scale the learned rows have rather than at an extra constant someone has to pick. The draw uses the run's own random state, so a fixed model seed reproduces it.

The table starts at zero — that is `Trainer`'s constructor, and in embedding mode no warm-up replaces it — but every later addition is blended or drawn, so no added row starts at zero. There is no zero-init option and no every-stage warm-up switch.

Each stage is its own `train` call, so the learning rate and the direction term's anneal restart with every stage; new planes get a full direction term again. The seeded rows are learned inside that call, together with the inherited ones. And a stage at a count that has stopped growing is not a stage: once the `--samples-end` cap is reached the remaining epochs are one stage on that plane set, so nothing is re-cut for the same loops and the anneal spans the whole tail.

What it costs in fit, measured on `laurent_hand` against a flat run on the same final plane set (9 planes per axis, 47 loops, seed 0; median/mean of the per-loop error in `planes.txt`):

| run | epochs | median | mean | p90 |
|---|---|---|---|---|
| flat `--samples 9` | 200 | 0.0144 | 0.0170 | 0.0311 |
| `--samples 2 --add-planes 1 --refine-every 10`, every-stage warm-up | 200 | 0.0178 | 0.0199 | 0.0316 |
| `--samples 2 --add-planes 1 --refine-every 10`, one warm-up + blend | 200 | **0.0085** | **0.0101** | **0.0165** |
| flat `--samples 9` | 1000 | 0.0064 | 0.0083 | 0.0172 |
| `--samples 2 --add-planes 1 --refine-every 10`, every-stage warm-up | 1000 | 0.0087 | 0.0099 | 0.0167 |
| `--samples 2 --add-planes 1 --refine-every 10`, one warm-up + blend | 1000 | **0.0047** | **0.0053** | **0.0082** |

An epoch spent on a coarse plane set is still not an epoch spent on the final one, so at equal total epochs a staged run arrives with a tail — at 200 epochs, 2→9 with 10-epoch stages spends 70 epochs growing the set and 130 on the final one. But with the warm-up taken out, the staged run now beats the flat fit at both budgets on this shape, so the tail was not the main cost: most of the earlier gap came from re-warming, not from the plane set changing. A run whose stages are each a handful of epochs still needs a long final stage to reach the same fit. The staged numbers are seed 0, which the CLI does not expose: the runs set it through a wrapper before importing `rosa_main`.

### The triplane feature map

Extraction does not read the triplane. Every stage from the planes to the grouped loops in `loop_extraction.py` works on the cloud and the slab alone — the module has no reference to `triplane` or to a feature vector — and `rosa_main` builds the field only once the loops exist, inside the fitting call. What the field does is replace the row table above: under `--feature triplane` (the default) a loop is not identified by a key into a table but by where it is.

The field (`mini_nn.TriplaneField`) holds three orthogonal grids — XY, XZ, YZ — each `--triplane-channels` (16) values deep and `--triplane-res` (32) cells per side, spanning the cloud bounding box plus a 2% margin so a loop centre on the bounding face still falls inside. A query is the loop's world centroid: each grid is bilinearly sampled (`padding_mode='border'`) and the three 16-vectors concatenated, so the code is 48 numbers and, crucially, a continuous function of position. Two loops near each other read overlapping cells and start similar by construction. There is no row to seed when `--refine-every` inserts a plane, because the copied field already has a value there — `--code-init` is read only in embedding mode. The `(axis, offset, slot)` keys are still recorded for replay and export, but they no longer index a learned row.

`rosa_main` exports the map to `<stem>_triplane.npz`: `grid` (`3 x C x R x R`), `lo`/`hi` (the sampling bounds), `queries` (`N x 3`, the world centroid each loop was decoded at), `features` (`N x 48`) and `keys`. A loop's query is the mean of its target points, so a feature row joins to a block of the loop PLY by centroid (median match `1e-8` on lucy), in checkpoint order.

That continuity is what makes the field usable as a loop similarity, not only as conditioning. The question propagation asks — which loop on the next plane continues this one — can also be asked in feature space: take the Euclidean distance between two loops' `features`, restricted to adjacent planes of the same axis, and it is a shape-and-place descriptor. On lucy (157 good loops, `log/cmp_tri_tv01_lucy`) accepted step-2 edges have median feature distance `0.486` against `2.053` for rejected edges (AUC `0.967`), and a mutual feature nearest-neighbour graph recovers `93%` of the geometric edges, though only `71%` of its own are geometric. It is not a gate: step-3 seam cuts are not separated by feature distance (AUC `0.601`), and pure feature tracking gives 10 tubes / 47 loops against the geometric 11 / 51. The measured conclusion is "feature proposes, geometry disposes": feature distance is a candidate ranker and a tie-break, and neither `loop_extraction.py` nor `loop_propagation.py` reads it. The full comparison is [triplane_loop_clustering.md](triplane_loop_clustering.md).

Two limits matter before trusting a threshold. The scale is run-specific — 48-D distances depend on the field magnitude (lucy's all-pair median is `2.024`), so thresholds must be recalibrated or the features whitened per run. And resolution changes the field. On dancing children at 29 planes, 32 cells is the right scale: fit is close to the row table (median plane error `0.0124` against `0.0101`) and downstream is slightly ahead (35 tubes / 138 loops against 34 / 133), while 64 cells puts most loops in a cell of their own, degrading fit to `0.0233` and downstream to 28 tubes / 116 loops. The cube warm-up also differentiates the field at each loop query, which helps the fit; if feature smoothness matters between loops, pair it with `--code-smooth 0.01`, since continuity only shares cells, it does not make neighbours use them alike.

### The cube warm-up

Before its first fit a run loads a cube-warm-up checkpoint (`--warmup`,
`warmup/cube_warmup.pt` by default). The checkpoint is trained by its own script,
`warmup_cube.py`, on a unit cube and nothing else: its loops are the cube's
cross-sections, and the network learns to draw them on the cube, all circulating
the same way and all putting \(z = 0\) where their neighbouring loops do. The
checkpoint holds the shared MLP and the triplane grid, and nothing else — no
rows, no phases, no plane set — so the shape's loops, planes, targets and phases
are its own, and the fit keeps its own triplane frame. [WARMUP.md](WARMUP.md) is
the specification of the task; `--no-warmup` fits from a blank network.

The transfer is through the triplane field: the grid is a function of position
on the unit cube, so a field warmed on the cube reads the same way once the cube
is laid over the cloud's box. Embedding mode has no such carrier — a row names a
cube plane, not a shape plane — so it prints a note and skips the load.

### Predicted normals

The network predicts points, not normals. For each fitted point:

```text
tangent = analytic d(prediction)/dz from the network
normal  = tangent x plane_normal
```

The normal is calculated inside `Trainer.predict_with_normals()` from the analytic network tangent. The curve lies in its slice, which leaves one in-plane direction square to its tangent, and the order of the cross product is the whole choice between the two: crossing the tangent with the plane normal is one quarter turn, crossing the plane normal with the tangent is the same turn the other way. A quarter turn applied twice is a half turn, `plane_normal x (plane_normal x t) = -t`, so the surface normal is `t x plane_normal`; the other order would be that normal negated, the tangent turned the wrong way for a surface the curve is a contour of. The direction term is written the same way round: it takes `t x plane_normal` as the normal the fit implies and measures `1 - cos` against the point normal the curve was matched to, both read in the loop's plane. The arrow is therefore the surface normal the curve implies — the one the cloud would give — and on a fit the direction term is happy with, it agrees with the target normals; a loop running against them draws inward with it. That is a state the term can only prevent, not repair: at a reversed closed loop its penalty is at its maximum and its gradient is exactly zero, and the weight is annealed to zero besides. On the warm-up square, where the parameterisation orders the walk, it already matches the square's own normals to about 0.96 with the normal term switched off.

Its length is left as the turn gives it rather than scaled to one: because the fitted curve runs inside its slice, that length is the tangent's own, i.e. how far the fit travels per step of the parameter. On the fit panels the projected normal is drawn at that length, so the arrows keep their proportions to each other and to the loops instead of all reading as unit vectors. It is written into the loop PLYs as `nx/ny/nz` and drawn on the fit panels; `--normal-arrows` controls the count per loop.

**The written traversal is turned to agree with the points.** Because a reversed loop is a stationary point of the direction term, the fit's choice of direction is canonicalized once, after the fit, by `rosa_main.fitted()`, from the same nearest-neighbour pairing as the error: for each loop, the mean cosine between the unit predicted normal and the unit paired target normal (unit-normalized, so a fast stretch does not outvote a slow one). If that mean is below −0.1, the loop is turned round — prediction, tangent, predicted normal and per-point error reversed together, so the block a consumer reads is still one loop written in one direction. The margin leaves a folded or side-on loop alone, since such a loop has no one direction to take. `align` is measured before the turn, so it still reports what the fit did, and the run prints how many loops were turned. On the 30-step horse run that was 28 of 30 Y loops with align 0.490; on the 300-step run none were turned (align 0.193) because the fit itself was already right. The PLY `nx/ny/nz`, the fit figures and the per-plane files carry the turned normals; `loop_propagation.load_run` still canonicalizes traversal independently by signed area.

### Reading the parameter colours

The network is evaluated at one uniform walk of the parameter, so the rows of a prediction are already in parameter order and each curve is drawn as segments coloured by where in that walk they sit. The map is cyclic (`hsv`), because the parameter is periodic: the colour at one full turn is the colour at zero, which is the seam ring. Red is therefore the start, and the walk runs red → yellow → green → cyan → blue → magenta → red. Two things read straight off it. The **direction** is the order the colours come in, and a loop running the other way round shows the reversed sequence. The **pace** is how much colour a stretch of arc carries: a long stretch of nearly one colour is crossed in few parameter steps, so the fit is moving fast there, while a short stretch holding the whole rainbow is where the fit spends most of its walk. A curve that doubles back on itself repeats the same colours on top of each other, which is how a folded fit shows up before its error is read. Under the colours a thin casing keeps the loop's own assignment colour, red and heavier past `--max-err`, so the colour scheme does not cost the bad-loop mark. The fit grid and every per-plane fit figure carry the same colour bar.

### Two different error rules

There are currently two separate error classifications:

1. **PLY export:** if the maximum point error of a loop is greater than `0.15`, every predicted point in that loop is coloured red.
2. **Fit figures and tables:** `--max-err` defines a fraction of the longest bounding-box side, and is applied to the loop's mean error.

These thresholds are independent.

---

## 14. Output files

By default, outputs go to `log/<input_stem>/`.

| Output | Contents |
|---|---|
| `<stem>_slices.ply` | Points selected for all slices |
| `<stem>_slices.png` | Projected slices, normals, centres, and optional fields |
| `<stem>_field.png` | Raw implicit fields with centres, when `--field` is used |
| `<stem>_good_loops.ply` | Fitted loops whose maximum point error is at most `0.15`, vertex-coloured by cutting axis (x blue, y green, z red), with predicted normals in `nx/ny/nz`, turned to run with the target normals when the raw fit ran against them |
| `<stem>_bad_loops.ply` | Fitted loops whose maximum point error exceeds `0.15`, all red, with normals, turned the same way. The two sides together hold every loop; no combined all-loops PLY is written |
| `<stem>_loop_split.txt` | One `g`/`b` per loop in checkpoint order, so the two split PLYs interleave back into the order of `_model.pt` |
| `<stem>_fits.png` | Target loops and fitted curves with predicted-normal arrows; its subtitle carries the mean out-of-plane distance and the alignment penalty (`0` runs with the target normals, `2` against), the latter measured before the output turn. Each curve is coloured by the loop parameter, one cyclic turn from the seam ring back to it, with a colour bar under the figure |
| `snapshots/<stem>_fits_eNNNN.png` | The same grid taken during the run, every `--snapshot-every` epochs (50 by default, `0` for none), drawn from the network as it stood on that epoch and against the same red limit, so a run can be watched taking shape rather than only judged at its end. Flipping through the files shows the curves arriving on their loops; the last one is identical to `<stem>_fits.png` above |
| `planes/` | Per-plane slice, assignment, fit PLYs with fit normals, and fit figures |
| `planes.txt` | Per-plane counts and fitting-error summary |
| `bad_loops.txt` | Loops exceeding the figure's mean-error threshold |
| `<stem>_model.pt` | The fit, replayable without the cloud: network weights, architecture, embedding table or triplane field, and one `(axis, offset, slot)` key, plane origin and normal, and query centroid per loop |
| `<stem>_triplane.npz` | Triplane mode only: `grid` (`3 x C x R x R`), sampling bounds `lo`/`hi`, query centroids, the 48-D feature sampled at each, and the keys |

An empty good or bad category still produces an empty PLY file.

A run that refines its planes writes the extraction artefacts (`_slices.ply`, `_slices.png`, `_field.png`) once, at the end, for the plane set it finished on: the intermediate plane sets are cut to train on, not to be kept. The fit artefacts are written for that same final plane set.

---

## 15. Key distinctions to remember

- A **point group** from `slice_loops()` is only a geometric connected group; it can contain several loops.
- A **raw positive component** is a candidate interior shown by the field renderer.
- An **accepted component** is a positive component large enough to pass the area filters in `field_parts()`; no rim-support test is applied.
- A **final component** results from recursively re-solving accepted components with their own points and normals.
- A **fallback centre** comes from normal-line least squares when no component is accepted, and is not guaranteed to be inside a positive field.
- **Centre RMS** measures normal-line agreement; **curve-fitting error** measures neural predictions against target points. They are different quantities.
- **`align` on the report line measures the fit before the output turn.** A loop that the raw fit ran backwards is turned for the PLYs and figures, and the run prints how many were turned; a nonzero count with a high `align` is the fit's own direction, not the written one.
- The **triplane feature map** conditions the fit and can measure loop similarity after it; it is not read by extraction or by propagation. Feature distance ranks candidate continuations well but does not decide them ("feature proposes, geometry disposes").
