# The warm-up: pretraining on a cube

Scope: **only** the warm-up. The shape fit is described in `WORKFLOW.md` and
`loop_extraction.md`.

Each decision has an id (`W1`, `W2`, ...). To revise, edit the decision itself and
say which id changed; the code will then be rewritten from scratch against the
revised text, not patched.

---

## The idea

We are not given a shape. We are given **a cube**, and nothing else.

Pretrain the network on that cube so that, when it draws the cube's
cross-section loops:

1. every loop runs the **same way round** along an axis — consistent orientation;
2. every loop puts **`z = 0` where its neighbours put theirs** — consistent seam;
3. every loop **lies on the cube** — the curves fit the 3D object.

That trained state is then handed to a real shape's fit as its initialisation.
The cube is the same object for every shape and for every run, so the warm-up is
identical everywhere: it is a fixed task, not a task derived from the shape.

What the cube is for: it is the simplest 3D object whose cross-sections are
loops, whose loops have exact closed-form parameterisations, and which has
neighbouring loops along each axis to be consistent with. The bunny (or any other
shape) plays no part in it.

---

## Running it

```
python warmup_cube.py                  # train the cube, write warmup/cube_warmup.pt
python rosa_main.py ... --train        # loads that checkpoint by default
python rosa_main.py ... --no-warmup    # ablation: fit from a blank network
```

`warmup_cube.py`'s `--neurons`, `--layers`, `--resolution` and `--channels` must
match the fit's, or `rosa_main` refuses the checkpoint rather than loading it in
part.

---

## Decisions

### W1 — The object is a canonical unit cube

The warm-up's object is the unit cube, side 1: `[-0.5, 0.5]^3`, always the same
numbers. It is a definition — no cloud, no bounding box, no extent is ever read
to size or place it.

Its surface is sampled deterministically (a fixed seam-free sample of the six
faces) and carries outward normals. If a sample is ever needed on a face, the
face's own normal is the normal there — no normals are estimated.

### W2 — The loops are the cube's cross-sections, and the cube *is* cut

Planes cut the cube along x, y and z at fixed offsets inside it. Each plane cuts
exactly one loop: the square `cube ∩ {axis = d}`. It has side 1, it is centred
at `d · axis`, and its parameterisation is exact.

This is the answer to the earlier question about "cutting the cube": here the
plane really is at `d`, and the loop really is the cross-section there. The cube
is not re-centred per loop and no offset is discarded.

Because a cube is convex and its faces are axis-aligned, every cut by an
axis-aligned plane is exactly a square — so the loop is a `Square` object at
`centre = d · axis`, not an approximation of a rectangle or hexagon.

**No plane through the exact centre.** The three centre cuts, `x = 0`, `y = 0`
and `z = 0`, all have their centroid at the origin. Since the network is
conditioned on the field at the loop's centroid (W5), one feature would be
asked to produce three different squares, and the fit can only answer with an
average that is wrong on all three. A real shape does not repeat that
coincidence — its three centre cuts have different centroids — so the cube skips
the centre. The number of planes per axis is therefore even.

### W3 — Winding and parameterisation

Each loop is walked counter-clockwise seen from the plane's normal `+axis`:

- `u` in-plane, `v = axis × u`, so `(u, v, axis)` is right-handed;
- four edges walked from a corner, parameter `z ∈ [0, 1)` wrapped;
- `z = 0` is that same corner on **every** loop of the cube;
- the outward normal at `z` is `tangent × axis`;
- `at(t)` is exact at any `t`, no nearest-sample.

`u` is fixed per axis by a deterministic rule (world axis least aligned with the
normal, ties by lower index), never by a float comparison, so the frame cannot
flip by a quarter turn between loops.

### W4 — Neighbouring loops are the ones that must agree

Along each axis the loops are ordered by their offset. The consistency asked for
is between loops adjacent in that order — that is the cube's analogue of two
adjacent cut planes on a shape, and it is what "along an axis" means in the
pretraining task.

The measure is the same `D` the seam solve uses: the two curves read at the same
parameter and separated by their own mean squared radius. A phase per loop is
what moves `D`; the warm-up's job is to leave the network's own `z = 0` at a
placement where `D` is already small without a phase buffer.

### W5 — What the network is conditioned on

One network for the whole cube, conditioned on the field sampled at the loop's
centroid — the same form the real run uses. The triplane grid is the carrier
that transfers: it is a function of position on the unit cube (W9), so a field
pretrained on the cube means the same thing on any shape's box. The centre cut
is excluded precisely so every loop's query is its own (W2).

Per-loop embedding rows are *not* usable for transferring a cube warm-up: a row
indexed by a cube plane has no meaning on a shape's plane set. (Choice C4.)

### W6 — Objective: three terms, all three shape-independent

For the cube's loops, with the network's parameters and the field's grid being
optimised:

1. **Fit** — each curve must lie on the cube: the fit term the real run uses,
   against the cube's surface samples (or the exact cross-section, which is the
   same set of points on the cube). This is "fits to the 3D cube".
2. **Direction** — each curve must run the way the cube's outward normals say;
   this is what makes all loops along an axis circulate the same way.
3. **Seam** — `D` between neighbouring loops along an axis, with the phase held
   at zero. Crucially this term is differentiated through the **network**, not
   into a phase buffer: the point is that the trained network *itself* puts
   `z = 0` consistently, so the property survives transfer.

There is no shape and no correspondence search: the cube's loops are known, so
every loop's target is exact. (Choice C2: whether a pure chamfer fit is used
instead of the exact target, to make the warm-up identical in form to the real
fit.)

### W7 — Optimisation

A fixed schedule, same for every run: every loop of the cube in every step,
cosine-annealed learning rate, 300 steps (default).

The warm-up prints, per run: fit error, mean direction cos, and `D` between
neighbouring loops at `z = 0` — the three things the task asks for.

### W8 — Transfer: the warm-up is a checkpoint

The warm-up is its own training script, `warmup_cube.py`, and its product is a
checkpoint (`warmup/cube_warmup.pt` by default). The checkpoint holds:

- the trained network weights and the triplane field, unchanged;
- **nothing else** — no per-loop rows, no phases, no plane set.

A shape's fit loads it by default (`--warmup`), keeping its own triplane frame
(W9) and its own extraction, plane set, targets and phases. The load is refused
if the checkpoint's network or grid shape does not match the run's, so a warm-up
cannot be applied in part. In triplane mode there is nothing to re-index: the
field is a function of position on the unit cube, so it reads the same way after
the cube is mapped onto the shape's box. (Choice C4 covers embedding mode.)

The real fit then runs as it does today, including its own `--seam-align` if
asked.

### W9 — The triplane: a grid on the unit cube, mapped to the shape's box

The grids live on the unit cube, uniformly subdivided, the same cube for every
shape — the warm-up's cube and the triplane's domain are the same object.

At fit time the cube is laid over the shape's point-cloud bounding box by an
affine frame (`box_centre`, `box_half`): a query is carried in by
`(x - centre) / half` and clamped. The box is *only* the frame — it is not the
triplane's definition, and changing it is a change of coordinates on a feature
map, not a different field (verified: max feature difference 0.0 between the old
`lo/hi` form and this one).

So the same field means the same thing on the cube warm-up and on any shape.

### W10 — What the real fit inherits, and what can still drift

It inherits a network that already produces on-surface, consistently oriented,
consistently seeded loops. It does **not** inherit a guarantee: the real fit's
losses are invariant to where `z = 0` sits on a loop, so the fit can still slide
the seam unless its own `--seam-align` term holds it. The warm-up lowers the
starting `D`; holding it during the fit is the seam term's job.

### W11 — The warm-up runs once per network, not once per shape

The cube warm-up is a fixed task, so its checkpoint is trained once for a given
network and grid shape and reused by every run. A run that changes `--neurons`,
`--layers`, `--triplane-res` or `--triplane-channels` needs a warm-up trained
with the same sizes; the load refuses a mismatch.

Planes added later by refinement are not warmed again; their rows are seeded, or
in triplane mode there is no row to seed.

### W12 — Report

Per run:

- warm-up fit error on the cube (same units as the fit),
- direction: mean `cos` between predicted normal and the cube's outward normal,
- seam: `D` between neighbouring loops at `z = 0`, median and max.

None of these is a shape-fit error, and they are printed separately from it.

---

## Choices (settled)

- **C1 — Plane set.** Four planes per axis, at `±0.4` and `±0.1333` (evenly
  spread over `±0.8 · half`, `half = 0.5`), so no plane cuts the exact centre
  (W2). Twelve loops, nine neighbouring pairs. The centre plane was tried
  first; it cannot be represented (three different squares share one query), so
  the run left three averaged loops and a degraded field nearby.
- **C2 — Fit term.** The exact cross-section target, `|f(z) − p(z)|`. The cube
  supplies the correspondence the shape cannot, so no chamfer or sampling is
  used.
- **C3 — Seam term.** Constant weight `1.0`, `D` read on the network output at
  `z = 0` and differentiated through the network (no phase buffer). Its
  denominator is the pair's size on the cube (`pair_scale`, detached), not the
  predicted pair's own radius: with a predicted denominator the term can fall by
  growing one curve towards the other, which is not the placement being trained.
- **C4 — Transfer in embedding mode.** Triplane only. Embedding rows name cube
  planes and have no meaning on a shape's plane set, so `rosa_main` skips the
  warm-up (with a printed note) when `--feature embedding`.
- **C5 — The field after transfer.** The shape's fit keeps training the field and
  the network as usual; the warm-up is only the initial state.

---

## Measured

The cube itself, 4 planes per axis (12 loops), 2×64 network, 32×32×3 grid of 16
channels, 300 steps, seed 0:

| reading | value |
|---|---|
| fit to the cube's cross-sections | 0.0039 mean, 0.0043 max |
| direction cos against the cube's outward normals | +0.997 |
| `D` between neighbouring loops at `z = 0` | median 0.0000, max 0.0000 |

The shape fit that loads it, bunny at 5 planes/axis, 13 loops, same network,
300 epochs, `--normal 0.01` (the standard launcher's setting):

| run | median plane error | loops turned round | D median / mean / p90 / max | feature graph ratio |
|---|---|---|---|---|
| cube warm-up | 0.0161 | 5 | 0.0716 / 0.1204 / 0.2727 / 0.2975 | 0.58 |
| blank network | 0.0195 | 10 | 0.0947 / 0.1311 / 0.2417 / 0.3804 | 0.61 |

With the normal term off (`--normal 0`, the CLI default) the same pair at 30
epochs reaches fit 0.0165 warm against 0.0391 blank.

Before any fit, the warmed network's `D` on the bunny is median 0.0336. The
blank network's `D` reads 0.0000, but that zero is degenerate: a zero field is
one feature for every loop, so all loops draw the same curve — the same curve
everywhere has no seam disagreement while fitting nothing (fit 0.1245 against
0.2724). `D` only means something once the loops are distinct.

Two caveats: the fit's own losses are seam-invariant, so it can and does slide
`z = 0` after the warm-up (W10); holding it is `--seam-align`'s job. And the
cube's sharp corners leave more bend in the warmed fit (2190 against 778), which
`--smooth` addresses if it matters.

Runs: `log/fit300n_warm`, `log/fit300n_blank` (normal term on), `log/fit300_warm`,
`log/fit300_blank` (normal term off).

---

## Previous scheme (for contrast)

The scheme this replaced warmed each loop on an independently placed unit square
read at that loop's plane, one loop at a time, with a position + direction loss
and no cube, no loop set, and no seam term. It is a curve-shape prior, not a task
prior: it teaches "draw a square", not "lay loops on an object consistently".
Its measured numbers, on the bunny at 5 planes/axis, 13 loops, 2×16, 300 epochs:

| run | warm-up error | fit bad loops | D median / mean / p90 / max |
|---|---|---|---|
| teacher, half 1.0 | 0.0695 | 4 | 0.0375 / 0.1052 / 0.2275 / 0.3472 |
| teacher, unit cube | 0.0348 | 6 | 0.0320 / 0.0994 / 0.2888 / 0.3099 |
| student, half 1.0 | 0.0562 | 4 | 0.1268 / 0.1497 / 0.2949 / 0.3621 |
| student, unit cube | 0.0487 | 4 | 0.0339 / 0.0709 / 0.2033 / 0.2190 |

Runs: `log/seam_s5_300_unitcube`, `log/refit_s5_300_unitcube` and the older
`..._cutplane` pair.

---

## Code map

| what | where |
|---|---|
| cube and its loops | `mini_nn.cube_loops` — the unit cube cut at even offsets, no centre |
| square geometry | `mini_nn.Square`, at `centre = d · axis` |
| warm-up training | `mini_nn.warm_on_cube` — fit + direction + seam, all three through the network |
| direction term | `mini_nn.normal_penalty`, against the cube's outward normals |
| seam `D` | `D` between adjacent loops, denominator the cube pair's own scale |
| checkpoint | `mini_nn.save_warmup` / `load_warmup`, `kind='cube_warmup'` |
| training script | `warmup_cube.py` → `warmup/cube_warmup.pt` |
| fit that loads it | `rosa_main.py --warmup` (default), `--no-warmup` to skip |
| triplane | `mini_nn.TriplaneField`, reused unchanged |

---

## How to revise

Edit any decision in place and mark it, e.g.:

> **W6 (changed):** the seam term is only a diagnostic; the warm-up trains fit
> and direction, and the seam is left to the real run's `--seam-align`.

or delete a decision that should not exist. Then the implementation is rewritten
from this document alone, and this file is updated to match.
