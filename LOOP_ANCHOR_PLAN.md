# Anchoring the loop parameter origin by slice axis, with optional seam synchronization

**Status:** proposed, not implemented. Scope: `rosa_main.py`, `mini_nn.py`,
`loop_extraction.py`, the two existing docs (`loop_extraction.md`,
`log/refine_test/REPORT.md`), and a new optional `loop_sync.py` for §5b. §5b
adds an optional seam-synchronization stage taken from `todo.md`; it is off by
default and does not alter the anchor.

## 1. Goal

For every fitted loop `i` cut on a plane whose normal is along axis `a`, let

- `c_i` be the loop's centre,
- `n_a` the plane's unit normal,
- `f_i(z)` the fitted curve (`N x 3`, in the plane frame), and
- `r_a` one fixed in-plane direction per axis.

Make the point at `z = 0` lie on the ray from the centre in direction `r_a`:

```text
                  proj_{n_a}( f_i(0) - c_i )
r_a  =  -----------------------------------------       for every loop i of axis a
        || proj_{n_a}( f_i(0) - c_i ) ||
```

So the direction from the loop's centre to the `z = 0` point is the same for all
loops of the same axis — across planes, slots, refinement stages and runs.

Non-goals:

- no change to the fitted geometry, the fit error, or the bend;
- no change to the traversal direction; `--normal` and the square warm-up own that;
- no single 3D direction shared by *different* axes — the loops lie in different
  planes, so the direction is necessarily per axis;
- not a full angular reparameterization: only `z = 0` is pinned, the speed around
  the loop stays whatever the fit learned;
- seam synchronization and tube reconstruction are an optional follow-up (§5b),
  not part of the anchor; the `z = 0` contract holds when it is off.

## 2. Why the current `z = 0` is not anchored

- The periodic network maps `z` through `sin/cos`, so `z = 0` and `z = 1` are the
  same point (the seam, drawn red in the fit grids). `Trainer.z` is a uniform
  parameter grid (`mini_nn.py:511-514`).
- The fit is correspondence-free: chamfer (`CD_Loss`, `mini_nn.py:58-82`), the
  bend term (`mini_nn.py:129-154`) and the optional normal term
  (`mini_nn.py:158-204`) see the curve, not the parameter's origin. A shift
  `f(z) -> f(z + phi)` changes neither the curve nor, in the continuum, any of
  those losses. The phase is a free gauge; nothing in training fixes it.
- The square warm-up is the only phase cue. `Square.from_points`
  (`mini_nn.py:288-312`) fixes the plane-normal sign (largest component positive)
  and takes `u` as the world axis least aligned with the normal, `v = axis x u`;
  `Square.at(0)` is the corner `(u - v) / sqrt(2)`
  (`mini_nn.py:253-336`). That corner is the same direction for all loops of an
  axis, but it is measured from the loop's **point mean**, not its field centre,
  and fine-tuning is free to drift away from it.
- In practice the drift is large. Measured on the four `*_blend_e1000` fit PLYs
  (the first point of each predicted curve is `z = 0`; centre approximated by
  the predicted curve's centroid; `u` from the table below; counts match the
  reported loop counts exactly):

  | run | x: median / p90 abs angle | y | z |
  |---|---|---|---|
  | `four_hand_blend_e1000` (77) | 5.1 / 18.9 (16) | 103.7 / 150.9 (30) | 64.0 / 92.8 (31) |
  | `four_dance_blend_e1000` (114) | 44.9 / 97.5 (26) | 83.4 / 145.4 (50) | 49.8 / 108.9 (38) |
  | `four_dragon_blend_e1000` (104) | 33.2 / 104.1 (31) | 70.8 / 144.5 (45) | 31.3 / 73.9 (28) |
  | `four_rocker_blend_e1000` (80) | 73.9 / 141.8 (30) | 98.8 / 127.4 (31) | 31.9 / 69.1 (19) |

  The centroid is only a stand-in for the field centre, so these are indicative,
  not the final numbers; they are large enough that the conclusion is not in
  doubt. The implementation will report before/after angles using the field
  centre (see §8).

### A documentation gotcha

`LoopExtractor.loops` is `(slice, slot, points, normals)`; the second field is
the loop's **label/slot**, not a centre (`assigned_loops`,
`loop_extraction.py:465-492`). The field centre is

```python
c_i = found.centres[s][0][slot]        # world 3D, one row per loop
```

`loop_centres` stores it at `loop_extraction.py:449-459`. The docstrings and
`loop_extraction.md` currently call the loop tuple's second field "centre"; that
wording should be fixed while adding the anchor.

## 3. Canonical frame per axis

The planes are axis-aligned: `p_normal = np.repeat(np.eye(3), samples, axis=0)`
(`loop_extraction.py:1610-1612`), so the axis index is
`a = argmax(|p_normal[s]|)` and the normal is `+e_a`. Choose `r_a` exactly as
`Square.from_points` chooses `u` (first world axis least aligned with the
normal):

| slice axis `a` | `n_a` | `r_a = u_a` | `v_a = n_a x u_a` |
|---|---|---|---|
| x | +x | +y | +z |
| y | +y | +x | -z |
| z | +z | +x | +y |

This is deterministic, matches the square's own frame, and is independent of the
loop's current phase. `plane_basis` in `loop_extraction.py:95-99` uses a
different in-plane pair; do **not** mix the two. The anchor works in 3D
(predictions are local to `p_point[s]`; add it back), projects out `n_a`, and
only then compares with `r_a`, so the extraction frame is irrelevant.

Loops on the same axis share `r_a`. Loops with a reversed winding still satisfy
the radial condition; if tangent consistency at `z = 0` is also wanted, that is a
separate constraint (with a convex loop and CCW travel, the tangent at the `+u`
crossing is `+v`; reversed loops give `-v`).

## 4. Approach: exact post-hoc reparameterization

The phase is a gauge choice, so choose it **after** fitting rather than asking a
loss to find it:

1. Fit the loops exactly as today (no new loss, no new weight, no retraining
   recipe).
2. For each loop, find the parameter `t_i*` where the fitted curve crosses the
   `+r_a` ray from `c_i`; when the ray meets the curve more than once, take the
   **farthest** crossing from `c_i` (see §5).
3. Define the anchored evaluation `g_i(z) = f_i((z + t_i*) mod 1)`. Then
   `g_i(0) = f_i(t_i*)` lies on the ray by construction.
4. Use `g_i` everywhere the curve is evaluated: fit figures, snapshots, error
   measurement, normals, and PLY export.

Why this is safe:

- it is a reparameterization of the same curve; the underlying geometry is
  unchanged;
- the only change in the reported fit error is that the `N` uniform samples land
  at different positions along the curve (the curve itself is the same), so with
  the existing dense `N` the error moves by well under 1%;
- no weight to tune and no way for the anchor to distort the shape.

Optional companion change (not required): start the square warm-up at the `+u`
edge midpoint instead of the `(u - v)` corner, by shifting `Square.at` by
`1/8` turn for loop squares. This makes the raw, pre-anchor `z = 0` closer to the
target and reduces the size of the correction. It does not replace the post-hoc
anchor, because the fit can still drift.

Rejected alternatives:

- **A training loss on `f(0)` only.** A single point can be moved by a local
  bulge rather than by a global phase shift, and the term needs a weight and
  interacts with chamfer/bend; it is weaker than the exact construction.
- **Azimuth-based target correspondence.** Fails for non-star-shaped loops
  (multiple radii per direction) and changes the fitting paradigm.
- **Square initialization only.** Only sets the initial phase; §2 shows it drifts.
- **Learnable per-loop phase.** Exact in principle, but adds state that must be
  inherited across refinement stages; unnecessary because the phase can be
  chosen after fitting.

If a future loss depends on `z = 0` during training (e.g. cross-loop
correspondence), use the stored phase `phi_i` to shift that loss's inputs, or
add a weak anchor loss on top of this construction; the post-hoc phase remains
the reference.

## 5. Finding the anchor parameter

For one loop:

```text
input:  f_i(z)          N x 3 local predictions + p_point[s] => world q_i(z)
        c_i              world centre (field centre)
        n_a, r_a         unit plane normal and reference direction
        v_a = n_a x r_a

sample z_k = k / K, k = 0..K-1 (K = 2048 by default), one batched forward
w_k     = q_i(z_k) - c_i
w_k     = w_k - (w_k . n_a) n_a          # remove numerical out-of-plane part
cos_k   = (w_k . r_a) / ||w_k||
sin_k   = (w_k . v_a) / ||w_k||
theta_k = atan2(sin_k, cos_k)

find brackets where sin(theta) changes sign and cos(theta) > 0
    (theta near 0 mod 2*pi; include the periodic wrap by appending z = 1)
for every bracket:
    refine t* by bisection on atan2(sin, cos) = 0, 25-30 iterations
    radius* = ||w(t*)||                    # in-plane distance from c_i
choose the candidate with the largest radius*   # farthest intersection
if there is no crossing:
    t* = z_k maximizing cos_k
    status = "no_crossing"
phi_i = t* mod 1
```

- The rule is **farthest from the centre**: among all `+r_a` candidates, take the
  one with the largest in-plane radius. A non-convex or self-intersecting fit can
  meet the same ray several times; the nearest candidate may sit on an inner
  fold, while the farthest is the boundary through which the ray leaves the
  loop. Every candidate lies on the same ray, so farthest along the ray and
  farthest in Euclidean distance are the same test.
- The choice is deterministic and does not depend on the current phase or on the
  order the curve was walked. For a convex loop there is exactly one `+r_a`
  crossing; for a concave or self-intersecting fit there can be several; report
  the count and the chosen radius so the choice is auditable. Break a tie
  within `1e-6 * max_radius` by the smaller `t*`.
- If the centre lies outside the loop (the normal-line fallback case), the ray
  may miss the curve; the best-azimuth fallback still anchors as closely as the
  geometry allows and is flagged.
- If nearly all radii are zero, mark `degenerate` and leave the phase at 0.
- The search needs evaluations at arbitrary `z`. It runs once per loop per
  snapshot and once at the end; with a tiny MLP and `K ~ 2048`, the cost is
  negligible against training.

## 5b. Synchronized seams: the switchable pair objective from `todo.md`

The anchor is absolute and per-loop: `G_i(0)` lies on `+r_a` for every loop of
an axis. It does not claim that `z` runs in step between neighbouring loops:
consecutive cross-sections can be rotated or shaped differently, so the point at
a given `z` on one loop need not correspond to the point at the same `z` on the
next. `todo.md` ("A smooth switch with a closed-form solution", "Synchronized
loops define the tube coordinates") proposes the next stage. It is optional and
sits after the anchor:

```text
G_i(z) = p_point[s_i] + f_i((z + phi_i) mod 1)   # anchored world curve
H_i(z) = G_i((z + alpha_i) mod 1)                # tube cross-section
F(z, v_i) = H_i(z),   F(0, v_i) = G_i(alpha_i)   # synchronized seam
```

`G_i` is the world version of the §4 `g_i` (local curve plus its plane point).
`alpha_i = 0` puts the tube seam at the anchor. Enabling this stage never moves
`G_i(0)`, `anchors.csv`, or the per-axis angle check; it only selects the
parameterization used to build the tube. The farthest-crossing rule of §5 is
unaffected.

### Pair residual and closed-form switch

For loops `i, j` on adjacent planes of the same axis, both planes share `n_a`,
`r_a`, `v_a`. Remove the out-of-plane part and centre each loop at its field
centre:

```text
x_i(z) = P_a (G_i(z) - c_i),          P_a = I - n_a n_a^T
sigma_ij^2 = (mean_k ||x_i(z_k)||^2 + mean_k ||x_j(z_k)||^2) / 2
D_ij(delta) = (1/N) sum_k ||x_i(z_k) - x_j(z_k + delta)||^2 / sigma_ij^2
```

`D_ij` is the squared alignment error of `todo.md`, normalized by the pair's own
mean squared radius so that `lambda` below is dimensionless. If `sigma_ij` is
zero, the pair is degenerate and is skipped. Search `delta` on a circular grid
(`--sync-samples`, default 512), take the global grid minimum, and refine it
locally by parabolic interpolation over the three samples around the minimum:

```text
delta*_ij = argmin_delta D_ij(delta)
D*_ij     = D_ij(delta*_ij)
```

The sign convention is fixed: `x_i(z) ~ x_j(z + delta*_ij)`; hence the global
offsets below satisfy `alpha_j - alpha_i = delta*_ij`.

For a fixed `D`, the switch energy from `todo.md` is

```text
E_ij(s) = s^2 D*_ij + lambda (1 - s)^2
dE/ds   = 2 s D*_ij - 2 lambda (1 - s) = 0
s*_ij   = lambda / (lambda + D*_ij)
E*_ij   = lambda D*_ij / (lambda + D*_ij)
```

The second derivative is positive, so `s*` is the unique minimiser. Since
`dE*/dD = lambda^2 / (lambda + D)^2 > 0`, minimising `E*` is exactly minimising
`D*`: the joint `(delta, s)` problem has the closed-form solution
`(delta*_ij, s*_ij)` and needs no alternating optimisation. `s* = 1` for an
exact pair, `s* = 1/2` when `D* = lambda`, and `s* -> 0` for a pair that cannot
be aligned; `E*` is bounded by `lambda`, so a bad pair contributes at most
`lambda` instead of an unbounded squared error. Pairs with
`s* < --sync-min-switch` (default 0.3) are dropped; kept pairs are weighted by
`w_ij = (s*_ij)^2`, matching the `s^2 D` term of the energy.

### Candidate pairs

Only loops on **adjacent plane offsets of the same axis** are compared. For each
loop, take the `--sync-neighbors` (default 2) nearest loop centres on each
adjacent plane as candidates; when a slot exists on both planes it is also a
candidate. The switch, not the candidate rule, decides which pairs survive.
This deliberately avoids requiring a one-to-one assignment between loops of
different planes, which is exactly the "align if they can be aligned, drop those
that cannot" backlog item.

### Global seam offsets

Pair constraints need not agree around cycles, so solve for all `alpha_i` at
once, per axis and per connected component of the kept-pair graph:

```text
min_alpha  sum_{(i,j) in E} w_ij * wrap(alpha_j - alpha_i - delta*_ij)^2
           + mu * sum_i alpha_i^2
```

`wrap(.)` takes the representative nearest zero in turns. With the anchor as
the initial guess the corrections are small, so the system is solved as a sparse
linear least-squares problem (graph Laplacian plus a diagonal prior); re-wrap
and re-solve if a residual exceeds half a turn. The prior `mu` (recommended
`1e-3 * mean(w_ij)`) fixes the otherwise free global gauge and keeps the tube
seam near the per-axis anchor; it does not pin any individual loop. Initialise
`alpha = 0` and solve each component directly. Because `D_ij` is
already minimised over the relative shift, a global gauge change does not alter
the switch values, so one pass is enough for the first version. Isolated loops
keep `alpha_i = 0`. If a component's edges conflict, the result is a
least-squares compromise; the pair residuals are reported so the conflict is
visible rather than hidden.

### What this stage does and does not do

- It produces a seam offset `alpha_i` per loop and a kept/dropped verdict per
  adjacent pair; the anchor's `z = 0` remains on `+r_a`.
- It does not change the fitted curves, the fit error, the bend, the align
  term, or the embedding table; it is a post-hoc parameterization of the same
  curves.
- It does not yet build the tube: the loft `(1-t) H_i(z) + t H_{i+1}(z)` and the
  shared neural surface `F_theta(PE(z), v)` from `todo.md` are a separate
  follow-up. This stage only supplies the synchronized `H_i` and the kept-pair
  graph; `v_i` (cumulative distance between centres) is assigned later along
  each synchronized chain, since one axis can hold several tubes.
- It does not revisit the farthest-crossing choice of §5: every pair is compared
  in the anchored parameterization, and if that choice makes a pair unalignable
  the switch drops the pair rather than changing the anchor branch.
- It only searches a relative phase shift; it does not reparameterize speed
  along the loop. A pair that would need stretching to align keeps a high `D*`
  and is dropped.
- It does not handle reversed winding; a reversed neighbour will usually give a
  poor `D*` and be dropped. Winding remains `--normal`'s job.

## 6. Code changes

### `mini_nn.py`

Add optional `z` arguments; training methods stay untouched.

```python
def predict(self, i=None, z=None):                 # z: N x 1 or 1D, default self.z
    ...
def predict_tangent(self, i=None, z=None):         # same coords as predict
    ...
def predict_with_normals(self, i, axis=None, z=None):   # forwards z to both
```

- Convert `z` with `torch.as_tensor(..., dtype=torch.float32, device=self.device)`
  and reshape to `N x 1`.
- `z = None` reproduces today's behavior exactly.
- `bend()` can stay on the training grid: it is shift-invariant up to sampling,
  and it is a reported diagnostic, not an exported curve. If exact before/after
  comparison is wanted, give it the same `z` treatment.

### `loop_extraction.py`

- Add `axis_reference(p_normal, s) -> (n_a, r_a, v_a)` implementing the table in
  §3. It should agree with `Square.from_points` for axis-aligned loops.
- Add a `loop_centre(found, loop)` helper (or document the indexing once) so the
  field centre is not confused with the tuple's slot.
- Add an `anchors.csv` writer, or keep the writer in `rosa_main.py` next to
  `planes.txt`; the file format is in §8.
- Fix the `assigned_loops` / `loop_extraction.md` wording about the second tuple
  field.

### `rosa_main.py`

CLI:

```text
--anchor {axis,none}      default: axis; none is the regression path
--anchor-samples K        samples for the crossing search (default 2048)
```

`fit_loops(..., anchor="axis", anchor_samples=2048)`:

- In `fitted()` (`rosa_main.py:441-485`), before predicting:
  - for every loop `i`, read `s, slot = loops[i][0], loops[i][1]`, then
    `c_i = found.centres[s][0][slot]`;
  - get `n_a, r_a, v_a` from `axis_reference`;
  - call `anchor_phase(...)` and keep `(phi_i, info_i)`.
- Evaluate the curve with

  ```python
  z = (trainer.z + phi_i) % 1.0
  local   = trainer.predict(i, z=z)
  tangent = trainer.predict_tangent(i, z=z)
  ```

  and use the same `z` for points and tangents; the normals are still
  `tangent x plane_normal`, so they rotate with the curve as before.
- `fitted()` is already the function used by `watch()` for snapshots
  (`rosa_main.py:498-507`) and by the final report (`rosa_main.py:587`), so this
  one change anchors snapshots and final outputs together.
- Compute the before/after angles with the **same** centre:
  - before: the unphased `trainer.predict(i)`'s `z = 0` point;
  - after: the anchored `z = 0` point (expected ~0 for `ok` loops).
- Keep `preds` in the same loop order; they are concatenated into
  `<stem>_loops.ply` (`rosa_main.py:607-609`), so each loop's first vertex
  becomes its anchor. `N` is the same for every loop (`latent_points =
  max_points`), so loop `i`'s anchor vertex is `i * N` in the combined PLY.
- Write the final `anchors.csv` after the last `fitted()` call; snapshots only
  update the console/figures, not the CSV.
- `--anchor none` skips the search and uses `trainer.z`; the rest of the code
  path is unchanged, so old runs remain reproducible.

### `loop_sync.py` (optional stage only)

New pure module; no training code touched.

```python
candidate_pairs(...)                  -> [(i, j), ...]
pair_switch(G_i, G_j, ...)            -> (delta, D, s, E)
sync_offsets(n, edges, deltas, switches, mu) -> alpha
```

- `pair_switch` samples both curves on the common `z` grid (`--sync-samples`),
  centres and projects them into the plane, and evaluates the circular shift
  correlation (FFT or direct, `K` is small), followed by local refinement. It
  returns the closed-form `s` and `E` from §5b.
- `sync_offsets` solves the sparse system with wrapped residuals; a component
  with one loop yields `alpha = 0`, and `mu` fixes the gauge.
- `candidate_pairs` uses `plane_keys` ordering and the loop centres; it does not
  force a one-to-one assignment.
- Unit tests are pure NumPy, with no cloud, network, or trainer.

### `rosa_main.py` (optional stage)

CLI:

```text
--sync-seams {off,switch}   default: off
--sync-lambda L             default: 0.01 (dimensionless)
--sync-min-switch S         default: 0.3
--sync-samples K            default: 512
--sync-neighbors N          default: 2
--sync-prior-mu MU          default coefficient 1e-3, scaled by mean(w_ij)
```

- After the final `fitted()` call and the `anchors.csv` write, if
  `--sync-seams switch`:
  - build `G_i` from each loop's anchored phase and final prediction;
  - form candidate pairs on adjacent offsets of the same axis;
  - compute `delta*`, `D*`, `s*`, `E*`, drop pairs below `--sync-min-switch`;
  - solve `alpha` per connected component and write the two CSV files (§8);
  - print the pair/switch/alpha summary.
- Leave `preds`, PLYs, fit figures, and snapshots untouched; do not apply
  `alpha` to `trainer.z` or to `anchors.csv`.
- `--sync-seams off` writes no new files and reproduces the anchor output
  exactly.
- The sync pass runs once at the final output, not per snapshot and not per
  refinement stage; it has no state to inherit.

### Docs

- `loop_extraction.md` §13: add "Anchoring the parameter origin" with the
  definition, the frame table, the search, and the output columns; document the
  optional `--sync-seams` stage and its CSVs, keeping the anchor and the tube
  seam distinct; correct the loop-tuple wording.
- `log/refine_test/REPORT.md`: add the four-shape before/after anchor table, the
  fit-impact check (§9), and the optional switch/seam statistics.

## 7. Refinement stages

- The phase is a property of the final fitted curve, not of a plane or a code
  row. It is recomputed from the current network at every snapshot and at the
  end; nothing needs to be inherited across stages.
- Loop rows are still keyed by `(axis, offset, slot)` (`plane_keys`,
  `loop_extraction.py:1449-1479`); the anchor does not touch the embedding
  table, `inherit`, `blend_added_codes`, or the plane plan.
- Retained loops keep their fit across stages as today; each gets its phase from
  the network as it stands. Added loops get their own phase from the same pass.
- Because snapshots are anchored independently, their red seam is at `+u` at
  every saved epoch, which makes the animation comparable frame to frame even
  though the underlying unphased training coordinates drift.
- The optional seam sync of §5b reads only the final network: it is not run per
  snapshot or per stage, and it has no state to inherit.

## 8. Outputs and reporting

New file `<out>/anchors.csv`, one row per loop:

```text
loop,slice,axis,offset,slot,centre_x,centre_y,centre_z,
phase_turns,angle_before_deg,angle_after_deg,crossings,radius,status,vertex_offset
```

- `axis` is `x|y|z`, `offset = p_point[s] . p_normal[s]`, `slot` is the loop's
  label (`loops[i][1]`), `phase_turns` is `phi_i`.
- `angle_before_deg`/`angle_after_deg`: the signed angle of the radial direction
  from `r_a`, measured from the field centre.
- `status`: `ok`, `no_crossing`, or `degenerate`.
- `radius`: the chosen crossing's in-plane distance from the centre.
- `vertex_offset = i * N` into `<stem>_loops.ply` (the main PLY; the
  good/bad PLYs filter vertices and therefore renumber).

### Optional seam-synchronization outputs

Only written with `--sync-seams switch`; `anchors.csv` is unchanged.

`<out>/sync_pairs.csv`, one row per candidate pair:

```text
axis,offset_i,slot_i,offset_j,slot_j,i,j,
delta_turns,delta_deg,D,switch,E,residual_turns,kept
```

- `delta_turns` is wrapped to `[-0.5, 0.5)`; `delta_deg = 360 * delta_turns`.
- `D` is the normalized squared alignment error, `switch = s*`, `E = E*`.
- `residual_turns = wrap(alpha_j - alpha_i - delta_turns)` after the global
  solve; blank for a dropped pair.
- `kept` is `1` when `s* >= --sync-min-switch`, `0` otherwise.

`<out>/seams.csv`, one row per loop:

```text
loop,axis,offset,slot,alpha_turns,alpha_deg,sync_component,kept_pairs,
anchor_angle_after_deg
```

- `alpha` is the tube seam offset `H_i(z) = G_i(z + alpha_i)`; it is not
  applied to the anchor.
- `anchor_angle_after_deg` repeats the §5 check, so the files show that enabling
  the sync did not move `z = 0`.
- `sync_component` names the connected component of the kept-pair graph;
  isolated loops have their own component and `alpha = 0`.
- `v_i` is deliberately absent: it belongs to a tube/path and is assigned by the
  tube follow-up, not by the per-loop sync.

Console, after the fit:

```text
anchor: 77 loops | before |angle| median 84.2 p90 152.9 deg
        after  median 0.000 p90 0.001 deg | 0 no-crossing, 0 degenerate
  x: 16 loops, before 5.1 / 18.9, after 0.000 / 0.001
  y: 30 loops, ...
  z: 31 loops, ...
```

With `--sync-seams switch`, one more summary:

```text
sync: 172 candidate pairs | 140 kept, 32 dropped (switch < 0.3)
      switch median 0.88 p10 0.21 | D median 0.004
      alpha -0.021 .. 0.019 turns | 6 components | 4 isolated loops
      seam residual (weighted) 0.031 -> 0.002 turns^2
```

Optional `anchors.ply`: one point per loop at its anchored `z = 0`, coloured by
axis, for a quick visual check in the PLY viewer.

Visible effects:

- the red seam of every fit grid is at `+u` for its axis;
- the first vertex of each fitted loop PLY is the anchor;
- the fit figures and snapshots show the same starting point for every loop of
  an axis.

## 9. Validation

Unit tests (synthetic, no cloud):

1. Default square (`u = x`, `v = y`), reference `+x`: the `+x` crossing is at
   `z = 1/8` (the middle of the right edge, from `Square.at`); the search must
   return `0.125` within `1e-6`.
2. A circle parameterized with an arbitrary starting angle: the returned phase
   must make the post-anchor angle zero within tolerance.
3. A concave "peanut" with two `+x` crossings: the **farther** crossing is
   chosen and `crossings == 2`.
4. A shape whose `+x` ray crosses three times: the farthest of the three is
   chosen and `crossings == 3`; the nearest is deliberately not chosen.
5. A centre outside / ray-miss case: `status == "no_crossing"` and the reported
   angle is the best achievable azimuth.
6. Two copies of the same rounded square (not a circle, whose shift is
   unidentifiable) with a known relative shift `delta0`: the pair search returns
   `delta*` within tolerance, `s* ~ 1`, and `E* ~ 0`.
7. The same pair with one copy scaled by 1.5 or with noisy samples: `D*` rises,
   `s*` falls below the default 0.3, and the pair is dropped.
8. A three-loop chain with consistent shifts: with `mu = 0` and one loop fixed,
   the global solve recovers them; with the prior, the whole chain moves toward
   zero by the expected compromise. A cycle with inconsistent shifts yields a
   nonzero `residual_turns`, which the pair CSV reports.
9. `--sync-seams switch` leaves `anchors.csv`, `G_i(0)`, and
   `anchor_angle_after_deg` identical to `--sync-seams off`; only the two new
   CSV files differ.

Integration (same seed, `--normal 0.01`):

- `laurent_hand`, `--samples 9 --epochs 200` and the refined
  `--samples 2 --add-planes 1 --samples-end 9 --refine-every 10 --epochs 1000`;
- the four production runs (`5 -> 15`, `1000`, seed 0, `--code-init blend`)
  used in `REPORT.md`;
- compare `--anchor none` vs `--anchor axis`: the per-loop fit error must move
  by less than 1% (same seed; only the sampled positions shift), the align error
  by less than 1%, and the runtime by less than 1–2% (the search runs only at
  snapshots and at the end);
- for the sync stage, run `--sync-seams switch` on the hand run and the four
  production shapes; report candidate/kept/dropped counts, switch quartiles,
  alpha range, and the weighted seam residual before/after; the fit error, PLYs,
  and anchor angles must be unchanged.

Acceptance:

- for `ok` loops: `angle_after` median <= 0.01 deg, p90 <= 0.05 deg,
  max <= 0.5 deg (bisection plus float32 model output);
- `no_crossing` + `degenerate` <= 1% of loops on the four shapes, each listed in
  the CSV, not silently counted as anchored;
- `--anchor none` reproduces the pre-change outputs exactly;
- visual: all loops of an axis have the red seam on the same side in the fit
  grids, including snapshots;
- sync: the synthetic cases pass; disconnected components and degenerate pairs
  do not crash; the anchor checks above are unchanged when it is enabled.

## 10. Risks and edge cases

- **Multiple crossings** (concave or self-intersecting fits): "farthest from
  the centre" is deterministic but may pick a different lobe than a human would;
  the `crossings`/`radius` columns make it auditable. Do not substitute the
  nearest crossing, which can sit on an inner fold and jump between candidates;
  an alternative rule can be added later without changing the interface.
- **Centre outside the loop** (normal-line fallback): no crossing in some
  directions; the best-azimuth fallback is reported, not hidden.
- **Tangency / plateau**: the ray grazes the curve; the bisection may be
  ill-conditioned. Detect a small `|cos|` over a run of samples and fall back to
  the grid maximum.
- **Self-intersections / topology changes between snapshots**: the chosen
  crossing (or its radius) can jump; the snapshot-to-snapshot seam may rotate.
  This is a property of the fit, not of the anchor; the final CSV records the
  chosen crossing.
- **Centre definition**: the field centre is used; the square's point mean is a
  different point, so the pre-anchor warm-up corner is only approximately
  aligned. If the point mean is preferred, switch the centre argument; keep one
  definition in the CSV and the validation.
- **Switch scale**: `D` is normalized by the pair's mean squared radius, so
  loops of different sizes are comparable, but a fixed `lambda` may not suit
  every dataset. The mean is outlier-sensitive; fall back to a median radius if
  spikes dominate. Report `D` and `s` and calibrate `lambda` only with a
  recorded protocol.
- **Candidate graph**: nearest-centre candidates can miss the true tube
  neighbour or include a false one; the switch filters only the pairs it is
  given. The pair CSV and kept counts expose this; a one-to-one assignment is
  out of scope.
- **Out-of-plane deviation**: `D` projects out `n_a`, so a pair that agrees
  in-plane but separates along the normal is not penalized; the tube follow-up
  should add a normal-direction term if that matters.
- **Tube seam vs anchor**: `alpha` is a tube offset; applying it to `G_i(0)`
  would break the per-axis invariant. Keep the two separate and report both.
- **Cycle conflicts**: the global solve returns a least-squares compromise;
  optionally drop low-switch edges and re-solve. The pair residuals show the
  conflict instead of hiding it.
- **Multiple shift minima**: a nonconvex pair can have several local minima of
  `D_ij(delta)`; take the global minimum on the grid. A near-tie between two
  distinct minima makes the chosen shift ambiguous even when `s*` is high, so
  flag the margin when it is below a few grid cells.
- **Longitudinal coordinate**: one axis can hold several tubes, so `v_i` is not
  a global per-loop value; assign it per synchronized chain in the tube
  follow-up.
- **Reversed winding**: the radial anchor still works; only the tangent at
  `z = 0` differs. Use `--normal` for winding consistency.
- **Resampling changes the fit error slightly**: the curve is the same but the
  `N` samples move. If exact bitwise preservation of the sampled point set is
  required, add an index-snap mode that rolls the existing samples by
  `round(phi * N)`; the cost is an angle quantization of `360/N` degrees. The
  default should stay continuous.
- **PLY vertex offsets**: the combined `_loops.ply` preserves loop order, but
  the good/bad PLYs filter vertices; use `vertex_offset` only for the main PLY.

## 11. Rollout

1. Add `axis_reference` + the unit tests; verify the table against
   `Square.from_points` on axis-aligned loops.
2. Add the optional `z` arguments and the phase search; wire `fitted()` and
   `anchors.csv`; keep `--anchor none` as the default until the numbers are in.
2b. (Optional, separate change) Add `loop_sync.py`, the switch tests, and the
    `--sync-seams switch` output path; keep it off by default. Validate on the
    hand run and the four production shapes before feeding `H_i`/`v_i` into any
    tube reconstruction.
3. Validate on the hand (flat + refined) and one production shape; check the
   fit-error delta and the visual seam.
4. Switch the default to `--anchor axis`; re-run the four-shape set and add the
   new table to `REPORT.md`.
5. Optionally, shift the square's parameter origin by `1/8` and/or align its
   centre with the field centre, then re-check the warm-up mapping error.
6. Update both docs and keep the plan next to them for reference.

## 12. Open choices (recommendations in bold)

| choice | options | recommendation |
|---|---|---|
| reference direction | `+u` per axis vs current square corner `(u-v)/sqrt(2)` | **`+u`**: simplest, matches the frame table |
| centre | field centre vs point mean used by `Square.from_points` | **field centre**: it is the pipeline's loop centre; report it explicitly |
| crossing | farthest from centre vs first in parameter order vs nearest to centre | **farthest from centre**: deterministic, phase-independent, outer boundary of the loop along the ray |
| phase resolution | continuous reparameterization vs integer index roll | **continuous**: exact angle; note the <1% resampling change |
| default | `axis` on vs `none` | **`axis`** after validation; `none` kept for reproduction |
| tangent at `z=0` | radial only vs radial + traversal sense | **radial only**; winding is `--normal`'s job |
| square warm-up | keep corner start vs shift by `1/8` | optional; shift only if raw pre-anchor phases matter |
| seam sync | off vs on by default | **off**: the anchor is the deliverable; enable after switch validation |
| alpha scope | tube seam only vs also the anchor | **tube seam only**: preserves the per-axis `z = 0` invariant |
| pair cost | normalized in-plane L2 vs 3D/chamfer | **normalized in-plane L2**: matches `todo.md`'s squared alignment error and factors over shifts |
| switch `lambda` | fixed vs adaptive | **fixed dimensionless**; log `D`/`s`; calibrate with a recorded protocol |
| pair candidates | nearest centres vs same slot | **nearest centres on adjacent offsets**, plus same slot; the switch filters |
| global gauge | weak prior vs fix one loop | **weak prior** toward `alpha = 0`; report `mu` |

## 13. Files and line references

- `mini_nn.py:511-514` — uniform parameter grid.
- `mini_nn.py:58-82` — chamfer; `129-154` — bend; `158-204` — normal term.
- `mini_nn.py:253-336` — `Square` edges and `at`; `288-312` — `from_points`.
- `mini_nn.py:517-616` — `Trainer.train`; `618-733` — `pretrain`;
  `806-838` — `predict`/`predict_tangent`; `839-875` — normals; `877-899` — bend.
- `rosa_main.py:422-430` — targets in the plane frame; `441-485` — `fitted()`;
  `498-507` — snapshot watcher; `539-584` — stages; `599-630` — export.
- `loop_extraction.py:95-99` — `plane_basis` (a different frame; do not mix);
  `449-459` — 3D centres; `465-492` — `assigned_loops` (slot, not centre);
  `1449-1479` — `plane_keys`; `1610-1612` — axis-aligned normals.
- `todo.md:12-20` — switch energy and closed form; `22-51` — synchronized
  seams, `v_i`, loft and shared-neural-surface proposals.
- `loop_sync.py` — new optional pure module (§5b): pair switch and global
  offsets; to be added, no training code touched.
- `loop_extraction.md` §13 — fitting and parameterization.
- `log/refine_test/REPORT.md` — the four-shape production numbers to extend.
