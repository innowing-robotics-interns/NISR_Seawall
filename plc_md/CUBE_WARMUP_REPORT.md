# Cube warm-up: result on four shapes

Date: 2026-11-07

> Historical: this report measures the unit-cube, axis-conditioned checkpoint
> `warmup/cube_warmup.pt`. Since 2026-11-08 the default is the random-cube,
> occupied-cell, axis-free `warmup/cube_warmup_random_ranges.pt`; see
> [WARMUP.md](WARMUP.md) W13 and the random-cube measurements there. The
> conclusions below describe the earlier checkpoint, not the current default.

This report measures what the cube warm-up (`warmup_cube.py` →
`warmup/cube_warmup.pt`) does when it initialises a real shape's loop fit. It
covers two things separately: the warm-up artifact itself, and four 29-plane
fits that load it, compared against identical runs that start from a blank
network (`--no-warmup`).

The fits are the `log/warm_tri_tv01_<shape>/` runs already on disk. The controls
are `log/blank_tri_tv01_<shape>/`, launched for this report with the same shape,
plane set, epochs, network and seed-free setup (the only flag removed is the
warm-up).

## Short version

- The checkpoint does its three cube tasks exactly: 12 cube loops, fit 0.0039
  (0.7 % of the 0.574 loop radius), direction cos +0.997, seam D 0.
- With the warm-up, every shape's loops sit on their slices: loop-mean error
  median 0.004–0.013, worst 0.031, and no loop passes the 3 %-of-extent red
  limit. 14 of the 538 loops hold one point beyond 0.15 and are marked bad.
- The warm-up is a **starting point**, not a preserved property. At load it is
  still a cube fit — loop error 0.25–0.45, §2.1 — and 1000 epochs of shape
  fitting lower that 19–103×. Final position error is then a wash: ahead on
  `bunny` and `horse`, level on `armadillo`, a little behind on `lucy`'s loop
  means. Seam D has no consistent direction: warm is better on `bunny`/`lucy`,
  blank on `armadillo`/`horse` mean.
- Its robust effect is **orientation**, but it acts through the fit: at load the
  warmed network is not aligned (orientation median 0.06–0.26, 7–39 reversed).
  Warm runs end with 1–6 reversed loops on every shape (5, 3, 1, 6), and
  `bunny` gives the same 3 in both repeats. Blank runs end with 27 and 43 on
  `bunny`, 31 on `horse`, 17 on `lucy`, and 0 then 27 on `armadillo` in two
  draws — whole families flipped. The one blank 0 is a lucky draw, not a
  property of the shape.
- The blank arm is a lottery while the warm arm reproduces: loading the
  checkpoint replaces the random MLP init, so repeated warm runs start from the
  same fit (0.3755 twice on `bunny`) and keep the same circulation (3 reversed
  twice). Blank repeats differ in initial fit (0.553–0.606), reversed count
  (27 vs 43) and which family flips.
- Its visible cost is bend: warm loops are 1.35–2.1× more bent than blank loops
  on all four shapes, because the cube's square corners train the network to
  make corners. The fit has no bend term, so nothing pushes back against that.

## 1. What was run

Warm-up artifact:

```bash
python warmup_cube.py            # 300 steps, 4 planes/axis, 12 loops
                                 # -> warmup/cube_warmup.pt
```

Each fit:

```bash
python rosa_main.py \
  --path normalized_shapes_cuboid/<shape>_cubic_lam3.ply \
  --field --train --samples 29 --epochs 1000 --normal 0.01 \
  --out log/warm_tri_tv01_<shape>
```

and the control, identical except for `--no-warmup`. Defaults in both arms:
`--feature triplane`, 32×32×3 grids, 16 channels, TV 0.01, 2×64 MLP, no code
smoothness, no `--seam-align`. The parameter grid is 362–1737 points per loop
(chosen per shape), 87 planes and 80–162 loops per shape.

`--seam-align` is **off**, so the seam reported below is the one the fit itself
ended on. Nothing in the reported runs enforced D.

## 2. The warm-up, measured again from the checkpoint

Decoded independently (`mini_nn` loaded from `warmup/cube_warmup.pt`, frame set
to the unit cube), not read from the training log:

| quantity | value |
|---|---|
| cube loops | 12 (4 planes per axis) |
| fit, mean / max | 0.0039 / 0.0043 |
| loop radius (target) | 0.574 |
| fit relative to radius | 0.68 % |
| direction cos | +0.9972 |
| seam D, median / max | 1.5e-14 / 2.6e-14 |

The exact-centre cross-sections of the cube are deliberately absent (even
planes per axis). They share the centroid (0, 0, 0), so one centroid-conditioned
field feature would be asked for three different squares; skipping them is what
takes the cube fit from ~0.09–0.21 to 0.004. See `WARMUP.md` W2.

### 2.1 The same checkpoint on each shape: before and after the fit

The checkpoint is shape-agnostic, but what it delivers is not: it is sampled at
each shape's loop centroids and then fitted. "After warm-up" below is the
checkpoint loaded into the shape's own triplane frame with no shape
optimisation — the fit's epoch 0, rebuilt from the saved network. The rebuild is
validated against the run: its `fit` equals the log's epoch-0 value and its
mean `1 − cos` matches the log's epoch-0 `align` within 0.005 on all four
shapes. "After shape fit" is epoch 999 of the same run. Both rows come from the
same probe, so they compare like with like; `fit` is the trainer's own loss.

| shape | stage | fit | loop err med | p90 | orientation med cos | reversed | D med | D mean | bend |
|---|---|---|---|---|---|---|---|---|---|
| armadillo | after warm-up | 0.563 | 0.448 | 0.954 | 0.157 | 21 | 0.0000 | 0.0116 | 1049 |
| armadillo | after shape fit | 0.0007 | 0.0064 | 0.0146 | 0.834 | 5 | 0.0225 | 0.0985 | 1651 |
| bunny | after warm-up | 0.376 | 0.252 | 0.354 | 0.055 | 7 | 0.0000 | 0.0094 | 1071 |
| bunny | after shape fit | 0.0010 | 0.0130 | 0.0182 | 0.890 | 3 | 0.0015 | 0.0093 | 4953 |
| horse | after warm-up | 0.558 | 0.411 | 0.895 | 0.260 | 8 | 0.0000 | 0.0168 | 1058 |
| horse | after shape fit | 0.0004 | 0.0040 | 0.0099 | 0.899 | 1 | 0.0113 | 0.0239 | 4500 |
| lucy | after warm-up | 0.606 | 0.395 | 0.941 | 0.099 | 39 | 0.0000 | 0.0186 | 1045 |
| lucy | after shape fit | 0.0004 | 0.0057 | 0.0094 | 0.675 | 6 | 0.0463 | 0.2370 | 2712 |

What the two stages say:

- **The checkpoint is a cube, not a shape.** Loaded onto a shape it draws one
  square-ish curve (bend ≈ 1050, against 39.5 for a circle) rather than the
  shape's loops: loop error 0.25–0.45, orientation median 0.06–0.26, and 7–39
  loops already past the reversal cut. The fit, not the warm-up, is what puts
  the loops on their slices: loop error falls 19–103× and orientation rises to
  0.67–0.90.
- **The orientation result is produced by the fit.** At load the warm state is
  poorly aligned on every shape; after 1000 epochs it has 1–6 reversed loops.
  The warm-up supplies the starting map, not a correct circulation. What it
  changes is which way the fit settles: the blank arm, starting from a random
  map, locks in whole reversed families — 27 and 43 on `bunny`, 31 on `horse`,
  17 on `lucy`, and 0 then 27 on `armadillo` (§7).
- **D = 0 at load is degenerate, not a seam.** On every shape the loaded
  checkpoint reads a median pair distance of exactly 0 because only 22–36 % of
  loops read a non-zero triplane feature: the rest read zero, land on the same
  curve, and make a neighbouring pair coincide. The counts are 43/162
  (armadillo), 21/80 (bunny), 31/139 (horse) and 56/157 (lucy) loops with a
  non-zero feature. The non-zero `D` mean (0.009–0.019) comes from the pairs
  where at least one loop does read a non-zero feature. The seam that matters
  is the one after fitting (0.0015–0.046 median), which is the fit's.
- **The grid reaches a minority of loops.** Since the cube warm-up only
  trained the grid near its 12 cross-sections, most shape loops start from the
  zero feature and inherit the warm-up through the trained MLP alone. That is
  why the effect is a starting convention rather than a per-loop
  initialisation.
- **Bend rises with fitting** (1049–1071 → 1651–4953): the one shared
  square-ish curve becomes many loops of different size and curvature. The
  warm arm's final bend is still above the blank arm's on every shape (§7).

## 3. Fit quality with the warm-up

Errors are in the cloud's own (normalised) units. "loop-mean" is the mean
distance from the predicted curve to the nearest target point; "worst point" is
the single largest per-point error in the run. The red limit is 3 % of the
longest extent (0.0515–0.0599 depending on shape).

| shape | loops | loop-mean med | p90 | max | worst point | loops > 0.15 | red loops | planes > 1.5× median | bend |
|---|---|---|---|---|---|---|---|---|---|
| armadillo | 162 | 0.0064 | 0.0146 | 0.0266 | 0.179 | 2 | 0 | 26 / 87 | 1651 |
| bunny | 80 | 0.0130 | 0.0182 | 0.0311 | 0.310 | 3 | 0 | 7 / 87 | 4953 |
| horse | 139 | 0.0040 | 0.0100 | 0.0211 | 0.522 | 9 | 0 | 19 / 87 | 4500 |
| lucy | 157 | 0.0057 | 0.0094 | 0.0140 | 0.105 | 0 | 0 | 7 / 87 | 2712 |

What the table says:

- Every loop is well inside the red limit on its mean error. The 14 "bad" loops
  are loops with one stray point beyond 0.15, concentrated on `horse` (9). They
  are a per-point failure on a thin or noisy stretch, not a wrong curve.
- `armadillo` has 26 planes over 1.5× its 0.0074 median. That flag is relative
  to the run's own median; in absolute terms the worst plane mean is 0.0266,
  under half the red limit. It is the many small limb loops that raise the
  spread, not a failing region.
- **Bend is high everywhere** (a circle is 39.5). The fit has no bend term, so
  nothing penalised it, and the cube-warmed network keeps a square-like
  tendency; see §7 where the blank controls show the warm-up raises it
  1.35–2.1×.

Fit grids: [armadillo](log/warm_tri_tv01_armadillo/armadillo_normal_cubic_lam3_fits.png),
[bunny](log/warm_tri_tv01_bunny/bunny_points_cubic_lam3_fits.png),
[horse](log/warm_tri_tv01_horse/horse_points_cubic_lam3_fits.png),
[lucy](log/warm_tri_tv01_lucy/lucy_points_cubic_lam3_fits.png). Slice targets:
the `*_slices.png` beside each.

## 4. Seam agreement D

D is the phase-independent distance `seam_check.py` reports on the run's own
neighbour graph, at `z = 0`:

    D = mean |v_i - v_j|² / ( 0.5 (mean |v_i|² + mean |v_j|²) )

so 0 is the same point round both loops; on a circle ~2 is a quarter loop off
and ~4 is half a loop off. These runs have no phases (no `--seam-align`), so
this is the seam the fit ended on.

| shape | pairs | median | mean | p90 | max | x median | y median | z median |
|---|---|---|---|---|---|---|---|---|
| armadillo | 108 | 0.0225 | 0.0985 | 0.2447 | 1.2197 | 0.0403 | 0.0077 | 0.0311 |
| bunny | 61 | 0.0015 | 0.0093 | 0.0201 | 0.1923 | 0.0024 | 0.0007 | 0.0017 |
| horse | 99 | 0.0113 | 0.0239 | 0.0785 | 0.1451 | 0.0029 | 0.0051 | 0.0146 |
| lucy | 83 | 0.0463 | 0.2370 | 0.6390 | 2.5698 | 0.0565 | 0.0045 | 0.0447 |

- The medians are small: for `bunny` and `horse` the typical neighbouring pair
  reads almost the same parameter, and `armadillo`/`lucy` are a small fraction
  of a quarter turn off.
- The tails are the interesting part. The worst `lucy` pairs are exactly where
  the section changes topology (x ≈ 0.25–0.31, slots 2→1, 1→1, 0→1) and where
  a big loop meets a small one (D 2.57 ≈ a quarter loop). The worst `armadillo`
  pairs are on z planes where the same slot persists but the loop is small
  (z ≈ −0.10 → −0.05, slot 2 → 2, D 1.22).
- `lucy`'s y family is its best (0.0045) and its x family its worst (0.0565);
  the same pattern appears in `armadillo` (x 0.0403, y 0.0077).
- The warm-up starts with D ≈ 0 on the cube but does not freeze it, and on a
  shape the loaded checkpoint's D = 0 is degenerate rather than good (§2.1:
  most loops read a zero feature and start on the same curve). Nothing in the
  fit holds `z = 0` in place: Chamfer, bend and normal are all invariant to a
  per-loop phase shift. That is what `--seam-align` exists for, and it was off
  here.

## 5. Orientation

For every loop the probe rebuilds the target normals, pairs them with the
predicted points, and takes the mean cosine between the normal the curve
implies (`tangent × plane normal`) and the target normal projected into the
plane — the same quantity the fit's direction term trains.

| shape | loops | median cos | mean | min | loops with cos < 0 | reversed (< −0.1) | x median | y median | z median |
|---|---|---|---|---|---|---|---|---|---|
| armadillo | 162 | 0.834 | 0.757 | −0.559 | 5 | 5 | 0.812 | 0.937 | 0.768 |
| bunny | 80 | 0.890 | 0.622 | −0.280 | 6 | 3 | 0.244 | 0.949 | 0.894 |
| horse | 139 | 0.899 | 0.845 | −0.431 | 1 | 1 | 0.865 | 0.941 | 0.895 |
| lucy | 157 | 0.675 | 0.584 | −0.514 | 6 | 6 | 0.684 | 0.621 | 0.711 |

Most loops agree with the surface normals; a handful per shape are reversed.
`bunny`'s x family is the weak one (median 0.244, 5 of 32 negative) — its
x-planes cut the ears and face where the contour is small and the normals are
less clear. A closed loop walked backwards is a stationary point of the
`1 - cos` term, so a loop that settles fully reversed keeps its direction and
the fit cannot flip it afterwards. Which loops settle that way is decided
early: §2.1 shows the warm state itself is poorly aligned, so the low final
counts come from fitting out of that start, not from the start already
pointing the right way.

### 5.1 Why the remaining loops are still reversed

The warm-up is not leaving them reversed. Comparing every loop's mean cosine
at load and after the fit: of the 15 loops still reversed in the four warm
runs, 10 were not reversed at load, and the fit repaired 70 of the 75 loops
that were reversed at load (19/21 armadillo, 7/7 bunny, 8/8 horse, 36/39
lucy). The fit both repairs and creates reversals; the survivors are the loops
whose direction it never settled.

| shape | reversed after fit | also reversed at load | reversed after, not at load |
|---|---|---|---|
| armadillo | 5 | 2 | 3 |
| bunny | 3 | 0 | 3 |
| horse | 1 | 0 | 1 |
| lucy | 6 | 3 | 3 |

Two properties of the loss explain why an unsettled loop stays wrong:

- **The direction weight is gone at the end.** `--normal` is the starting
  weight, not the kept one: `weight = align * cosine_anneal(step, total)`
  (`mini_nn.Trainer.train`), which reaches exactly 0 on the last step — the
  logs print `w 0.0000` at epoch 999. From then on the loss is exactly
  invariant to reversing a loop: Chamfer sees the same point set, bend uses
  squared derivatives, and TV only sees the features. Whatever sign a loop has
  when the weight vanishes is kept.
- **Reversal is a stationary point of `1 − cos`.** Its gradient is proportional
  to sin θ, zero at cos = −1 and weak near it, so a loop that turns round while
  the weight is small has almost nothing pulling it back across the 90° barrier
  (which costs 1 per point). The anneal is deliberate — the term settles
  direction while the fit is coarse and is given back later — but it also means
  a late reversal is permanent.

The survivors are the weakly constrained loops, not random ones:

- `bunny`'s 3 sit on the outer x planes (±0.70) and z = −0.26, where the
  in-plane part of the target normal is weakest (mean projection 0.65–0.79
  against 0.95 typical), and their fits double back (bend 2.6k–11.4k against
  39.5 for a circle); one is a 233-point ear loop.
- `lucy`'s 6 include two tiny x loops (37 and 333 points, radius 0.007/0.023,
  bend 6.9k–7.6k) where the direction is ill-conditioned, and four large loops
  with well-defined normals (0.85–0.93) that simply settled backwards.
- `armadillo` and `horse` survivors have well-defined normals (0.85–0.94) but
  double back: mean cos −0.17 to −0.56, bend 800–2,100.
- 2 of armadillo's 5 and 3 of lucy's 6 came in reversed and stayed; the other
  10 are new. The blank arm's 17–43 are the same effect with a worse starting
  map.

The exported files are not affected: `rosa_main.fitted` turns any loop with
mean cos < −0.1 round before saving, so `*_good_loops.ply` and the figures
agree with the cloud's normals. The counts above describe the network's own
traversal.

## 6. Feature field

The triplane field is diagnostic here; nothing downstream reads it.

| shape | graph feature distance | all-pair median | ratio | weighted mean square |
|---|---|---|---|---|
| armadillo | 0.2943 | 1.0046 | 0.29 | 0.1081 |
| bunny | 0.2156 | 0.8928 | 0.24 | 0.0911 |
| horse | 0.1205 | 0.9803 | 0.12 | 0.0602 |
| lucy | 0.2931 | 0.8742 | 0.34 | 0.1215 |

Neighbouring loops read features 2.9–8.3× closer than a random pair, so the
field does organise by position. `horse` is the most coherent (0.12), `lucy`
the least (0.34) — consistent with its seam tail.

## 7. Warm versus blank

Both arms: same shape, 87 planes, 1000 epochs, same net and losses. The only
difference is `--no-warmup`. The trainer has no `--seed` flag, so the same
command does not repeat exactly. To separate the warm-up's effect from that
noise, `bunny` was run twice per arm (`log/warm2_tri_tv01_bunny/`,
`log/blank2_tri_tv01_bunny/`) and `armadillo` got a second blank control; the
repeat spread is quoted in §7.1 and used as the bar for every comparison.

| shape | arm | fit epoch 0 | fit epoch 25 | fit epoch 999 | loop-mean med | loops > 0.15 | bend | align | loops reversed | D median | D mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| armadillo | warm | 0.563 | 0.0430 | 0.0007 | 0.0064 | 2 | 1651 | 0.243 | 5 | 0.0225 | 0.0985 |
| armadillo | blank | 0.706 | 0.0671 | 0.0010 | 0.0055 | 7 | 1042 | 0.188 | 0 | 0.0201 | 0.0927 |
| bunny | warm | 0.376 | 0.0194 | 0.0010 | 0.0130 | 3 | 4953 | 0.378 | 3 | 0.0015 | 0.0093 |
| bunny | blank | 0.606 | 0.0552 | 0.0017 | 0.0176 | 3 | 2994 | 0.853 | 27 | 0.0020 | 0.0132 |
| horse | warm | 0.558 | 0.0292 | 0.0004 | 0.0040 | 9 | 4500 | 0.155 | 1 | 0.0113 | 0.0239 |
| horse | blank | 0.608 | 0.0447 | 0.0005 | 0.0047 | 7 | 2132 | 0.437 | 31 | 0.0127 | 0.0206 |
| lucy | warm | 0.606 | 0.0319 | 0.0004 | 0.0057 | 0 | 2712 | 0.406 | 6 | 0.0463 | 0.2370 |
| lucy | blank | 0.564 | 0.0296 | 0.0005 | 0.0053 | 0 | 1991 | 0.458 | 17 | 0.0481 | 0.2467 |

`align` is the fit's own reading: mean `1 − cos`, 0 aligned and 2 reversed.
`loops reversed` is the count with mean cos < −0.1, i.e. the loops that run
against the points' normals. They are reported but no longer turned round at
export.

Blank-run orientation, rebuilt the same way as §5:

| shape | median cos | mean | min | loops < 0 | reversed | x median | y median | z median |
|---|---|---|---|---|---|---|---|---|
| armadillo | 0.869 | 0.812 | +0.149 | 0 | 0 | 0.865 | 0.917 | 0.813 |
| bunny | 0.078 | 0.146 | −0.943 | 30 | 27 | 0.050 | 0.948 | −0.895 |
| horse | 0.855 | 0.564 | −0.813 | 31 | 31 | −0.311 | 0.934 | 0.876 |
| lucy | 0.674 | 0.533 | −0.648 | 24 | 17 | 0.779 | 0.029 | 0.703 |

### 7.1 Repeat runs

Same command, same machine. `#1` is the run in the table above, `#2` the repeat
where one was run.

| shape | arm | run | fit 0 | fit 999 | loop-mean med | bend | align | reversed | median cos | D median | D mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bunny | warm | #1 | 0.376 | 0.0010 | 0.0130 | 4953 | 0.378 | 3 | 0.890 | 0.0015 | 0.0093 |
| bunny | warm | #2 | 0.376 | 0.0010 | 0.0122 | 5150 | 0.414 | 3 | 0.789 | 0.0016 | 0.0083 |
| bunny | blank | #1 | 0.606 | 0.0017 | 0.0176 | 2994 | 0.853 | 27 | 0.078 | 0.0020 | 0.0132 |
| bunny | blank | #2 | 0.553 | 0.0015 | 0.0150 | 3663 | 1.006 | 43 | −0.129 | 0.0028 | 0.0112 |
| armadillo | warm | #1 | 0.563 | 0.0007 | 0.0064 | 1651 | 0.243 | 5 | 0.834 | 0.0225 | 0.0985 |
| armadillo | blank | #1 | 0.706 | 0.0010 | 0.0055 | 1042 | 0.188 | 0 | 0.869 | 0.0201 | 0.0927 |
| armadillo | blank | #2 | 0.685 | 0.0010 | 0.0064 | 1176 | 0.431 | 27 | 0.834 | 0.0218 | 0.1150 |
| lucy | warm | #1 | 0.606 | 0.0004 | 0.0057 | 2712 | 0.406 | 6 | 0.675 | 0.0463 | 0.2370 |
| lucy | blank | #1 | 0.564 | 0.0005 | 0.0053 | 1991 | 0.458 | 17 | 0.674 | 0.0481 | 0.2467 |

The two warm bunny runs start from exactly the same epoch-0 fit (0.3755) and
end with the same 3 reversed loops: loading the checkpoint replaces the random
MLP init, so only the data order and the optimisation path vary, and they move
little (bend 4953 vs 5150, D median 0.0015 vs 0.0016). The blank runs are a
lottery. On `bunny` epoch-0 fit is 0.606 vs 0.553, reversed loops 27 vs 43, and
a different family flips: #1 points the whole z family backwards (19/19) and
11/32 x loops; #2 points 23/32 x loops and the whole y family (29/29)
backwards. After the −0.1 reversal cut that is 27 and 43 loops. On `armadillo`
the two blank draws give 0 and 27 reversed, the second concentrated in z (29/68
point backwards). `lucy`'s blank draw reverses 17, with y the weak family (22/48
point backwards). On `bunny` the warm runs point 6 loops backwards both times
(5 in x, 1 in z; 3 past the cut), median x cos 0.24 and 0.18.

What the comparison shows:

1. **Starting point.** The warm-up cuts epoch-0 fit on three shapes: 0.376 vs
   0.553–0.606 (`bunny`), 0.563 vs 0.685–0.706 (`armadillo`), 0.558 vs 0.608
   (`horse`). `lucy` is the exception — its warm start is slightly worse (0.606
   vs 0.564). By epoch 999 the difference is mostly gone: `bunny` still favours
   warm (0.0010 in both repeats vs 0.0015–0.0017; loop-mean 0.0122–0.0130 vs
   0.0150–0.0176), `horse` slightly (0.0040 vs 0.0047), `armadillo` is level
   (loop-mean 0.0064 vs 0.0055/0.0064) and `lucy` is a little behind (0.0057 vs
   0.0053); both gaps are ≤0.001, under the 0.0026 loop-mean spread between the
   two blank `bunny` runs. The initial advantage is repeatable; the final one is
   small and shape-dependent.
2. **Orientation is where the warm-up changes the outcome, and it acts through
   the fit.** Most loops in both arms start on one shared curve: the blank grid
   is all zeros, and at load 64–78 % of warm loops also read a zero feature
   (§2.1). The difference is the map that draws that curve. In the blank arm
   the random map lets the fit settle into different per-family circulations —
   z and x in #1, x and y in #2 — giving 27 and 43 reversed loops; the warmed
   map ends with 3 both times. `horse` shows the same lottery with one draw (31
   reversed, its x family), `lucy` with 17 (mostly y). `armadillo` shows the
   point of the repeat: its first blank draw came up 0 reversed, the second 27,
   so that clean draw was luck, not a property of the shape. Warm runs end with
   1–6 reversed on all four shapes. The warm-up does not align every loop — at
   load its own median cos is 0.06–0.26, and bunny still has 5 of its 32 x
   loops pointing backwards at the end — but it replaces the lottery with a
   convention.
3. **Seam.** There is no consistent direction. On `bunny` warm is lower in both
   repeats for the median (0.0015/0.0016 vs 0.0020/0.0028) and the mean
   (0.0093/0.0083 vs 0.0132/0.0112), but the blank median spread (0.0008) is
   larger than the warm–blank gap (≈0.0005). On `armadillo` the blank median is
   better in both draws (0.0201/0.0218 vs 0.0225) while the mean is mixed
   (0.0927/0.1150 vs 0.0985); on `horse` the median favours warm and the mean
   favours blank. The fit is seam-invariant — Chamfer, bend and the direction
   term do not see a per-loop phase — and the warm start's D = 0 is degenerate
   (§2.1), so the fitted D is the fit's own.
4. **Bend.** The cost is consistent across shapes and repeats: 1651 vs
   1042/1176 on `armadillo` (1.4–1.6×), 4953/5150 vs 2994/3663 on `bunny`
   (1.35–1.7×), 4500 vs 2132 on `horse` (2.1×), 2712 vs 1991 on `lucy` (1.36×).
   The cube warm-up trains on squares, so it teaches corners; the fit has no
   bend term, so nothing removes them once the position term is satisfied. This is the
   concrete cost of §2's exact cube fit.

### 7.2 Recreated controls: fresh draws (2026-11-07)

The blank arm and the repeats were run again from scratch with the commands in
"Reproduction". The trainer still has no `--seed`, so the blank draws are new;
the warm runs are the same ones and reproduce §3/§5 exactly. The folders on disk
are now these draws — §7 records the first batch, this table the second.

| shape | arm | fit 0 | fit 25 | fit 999 | loop-mean med | worst point | bad | bend | align | reversed | D median | D mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| armadillo | warm | 0.563 | 0.043 | 0.0007 | 0.0063 | 0.179 | 2 | 1651 | 0.243 | 5 | 0.0225 | 0.0985 |
| armadillo | blank | 0.724 | 0.071 | 0.0010 | 0.0058 | 0.209 | 7 | 1317 | 0.342 | 20 | 0.0286 | 0.0999 |
| armadillo | blank #2 | 0.684 | 0.060 | 0.0011 | 0.0068 | 0.209 | 8 | 965 | 0.499 | 35 | 0.0246 | 0.1137 |
| bunny | warm | 0.376 | 0.019 | 0.0010 | 0.0130 | 0.310 | 3 | 4953 | 0.378 | 3 | 0.0015 | 0.0093 |
| bunny | warm #2 | 0.376 | 0.019 | 0.0017 | 0.0176 | 0.312 | 4 | 5308 | 0.405 | 1 | 0.0013 | 0.0101 |
| bunny | blank | 0.543 | 0.057 | 0.0013 | 0.0152 | 0.320 | 3 | 2770 | 0.537 | 18 | 0.0026 | 0.0091 |
| bunny | blank #2 | 0.535 | 0.050 | 0.0013 | 0.0125 | 0.318 | 4 | 1277 | 0.100 | 0 | 0.0013 | 0.0069 |
| horse | warm | 0.558 | 0.029 | 0.0004 | 0.0040 | 0.522 | 9 | 4500 | 0.155 | 1 | 0.0113 | 0.0239 |
| horse | blank | 0.656 | 0.049 | 0.0005 | 0.0075 | 0.208 | 5 | 1583 | 0.427 | 27 | 0.0112 | 0.0328 |
| lucy | warm | 0.606 | 0.032 | 0.0004 | 0.0057 | 0.105 | 0 | 2712 | 0.406 | 6 | 0.0463 | 0.2370 |
| lucy | blank | 0.567 | 0.030 | 0.0005 | 0.0055 | 0.126 | 0 | 618 | 0.309 | 1 | 0.0297 | 0.1577 |

`reversed` is the rebuilt orientation probe (mean cos below −0.1), the §5/§7
reading; `bad` is loops with a point past 0.15. The blank #2 rows are the
repeats (`log/blank2_tri_tv01_*`); bunny's warm #2 is `log/warm2_tri_tv01_bunny`.

Orientation, fresh draws (median cos by family):

| shape | arm | median cos | x / y / z med | reversed (x / y / z) |
|---|---|---|---|---|
| armadillo | warm | 0.834 | 0.812 / 0.937 / 0.768 | 3 / 2 / 0 |
| armadillo | blank | 0.875 | 0.855 / 0.915 / 0.808 | 0 / 0 / 20 |
| armadillo | blank #2 | 0.826 | 0.827 / 0.917 / 0.365 | 6 / 0 / 29 |
| bunny | warm | 0.890 | 0.244 / 0.949 / 0.894 | 2 / 0 / 1 |
| bunny | warm #2 | 0.783 | 0.222 / 0.948 / 0.901 | 0 / 0 / 1 |
| bunny | blank | 0.833 | −0.110 / 0.946 / 0.901 | 17 / 0 / 0 |
| bunny | blank #2 | 0.932 | 0.932 / 0.948 / 0.900 | 0 / 0 / 0 |
| horse | warm | 0.899 | 0.866 / 0.941 / 0.895 | 1 / 0 / 0 |
| horse | blank | 0.843 | −0.260 / 0.924 / 0.894 | 27 / 0 / 0 |
| lucy | warm | 0.675 | 0.684 / 0.621 / 0.711 | 4 / 2 / 0 |
| lucy | blank | 0.709 | 0.748 / 0.667 / 0.660 | 1 / 0 / 0 |

What survives, and what does not:

1. **Start and early convergence reproduce.** Epoch-0 fit is cut on the same
   three shapes — 0.563 vs 0.684–0.724 (`armadillo`), 0.376 vs 0.535–0.543
   (`bunny`), 0.558 vs 0.656 (`horse`) — and `lucy` is again the exception
   (0.606 vs 0.567). By epoch 25 warm is 0.043 / 0.019 / 0.029 / 0.032 against
   blank 0.071 / 0.057 / 0.049 / 0.030. The early lead is the repeatable part.
2. **The final fit is still a wash.** Warm ends at 0.0007/0.0010/0.0004/0.0004
   against blank 0.0010–0.0011/0.0013/0.0005/0.0005, and loop-mean medians
   differ by ≤0.002 in either direction (`bunny`'s blank #2 is better than its
   warm #2 on both). `horse` is the one shape where warm is ahead on loop mean
   (0.0040 vs 0.0075). The warm start buys speed, not a better minimum.
3. **Orientation is still the warm-up's effect, and the blank lottery is now
   wider.** Warm ends at 5 / 3 / 1 / 6 reversed (bunny repeat 1); blank draws
   span 0–35: `armadillo` 20 then 35 (both z, 29/68 in the second),
   `bunny` 18 (x, 17/32) then 0, `horse` 27 (x, 27/43), `lucy` 1 (x). The two
   bunny draws bracket the report's 27 and 43, and the 0 is the good tail: that
   run reached align 0.100 with no loop turned, the best orientation of the
   eleven. A blank fit can come out aligned, but which family flips (or whether
   any does) is still decided per draw, while warm keeps to a small count.
   Warm's own failures stay in the weak families — `bunny` x (median 0.22–0.24)
   and `lucy` x/y — as in §5.
4. **Seam still has no direction.** `armadillo` warm is a little better on the
   median (0.0225 vs 0.0246/0.0286) and level on the mean; `bunny` is level on
   the median and mixed on the mean (blank #2's 0.0069 is the best of the
   three); `horse` is level on the median and warm-better on the mean (0.0239
   vs 0.0328); `lucy`'s blank draw is better on both (0.0297/0.1577 vs
   0.0463/0.2370). The fit stays seam-invariant, so nothing holds the warm
   start's D ≈ 0 through 1000 epochs.
5. **Bend is still the cost, now draw-dependent in size.** Warm 1651 vs
   1317/965 on `armadillo` (1.25–1.71×), 4953/5308 vs 2770/1277 on `bunny`
   (1.79–4.2×), 4500 vs 1583 on `horse` (2.84×), 2712 vs 618 on `lucy`
   (4.39×). Every comparison has warm more bent; the multiple is as variable as
   the blank draw, and the blank `lucy` and blank #2 `bunny` are unusually
   smooth runs rather than a different rule.
6. **The output turn holds on every run.** Re-reading the exported fit PLYs of
   all eleven, no loop votes below −0.1 (mean vote +0.60 to +0.88), so the
   files on disk all run with their target normals regardless of which way the
   raw fit settled.

## 8. Limits of this evidence

- **No `--seed`, and the blank arm is not reproducible.** `rosa_main` exposes no
  seed, so repeats were run instead. Warm repeats collapse to the same start
  because the checkpoint replaces the random init (`bunny` fit 0.3755 twice,
  3 reversed twice, D median 0.0015 vs 0.0016). Blank repeats do not: `bunny`
  fit 0 0.553–0.606, reversed 27 vs 43, D median 0.0020 vs 0.0028, bend 2994 vs
  3663; `armadillo` reversed 0 then 27. Every warm–blank comparison above is
  read against those spreads. The seam gap on bunny (median ≈0.0005) is below
  the blank spread. Orientation is different in kind: warm stays at 1–6
  reversed on every run, while blank draws span 0–43 and move 0 → 27 between
  repeats of one shape.
- **One repeat pair, one shape.** Only `bunny` has two runs per arm; `armadillo`
  has two blank draws, `horse` and `lucy` one each. The `armadillo` 0 → 27
  swing is the clearest warning that a single blank control says little about a
  shape.
- **Warm-up and fit are separated by reconstruction.** The "after warm-up" rows
  in §2.1 are the saved checkpoint reloaded into the shape's frame, not an
  epoch-0 file (the runs save only the final state). Its fit equals the log's
  epoch-0 value and its alignment matches within 0.005; D and bend at load are
  computed by the same code as the final row. The warm grid is zero at most
  loop centroids, so on 64–78 % of loops the checkpoint acts through the MLP
  alone — the grid and MLP contributions are not separated here.
- **No `--seam-align`.** D here is whatever the fit left; the number that
  matters for export with the phase solve on is a different experiment.
- **No bend term.** The fit does not optimise bend; the readings are the
  curves' own.
- **Blank start is degenerate.** With a zero grid every loop starts on the same
  curve. That is what `--no-warmup` means, but it makes the blank's initial D
  trivially 0 and its orientation arbitrary — the final numbers are the only
  fair comparison.
- **Shape set.** Four shapes, one 29-plane setting, 2×64, TV 0.01. The
  orientation weakness on bunny's x family and lucy's topology changes are
  visible in these runs but not explained by them.

## Reproduction

```bash
# artifact (once)
python warmup_cube.py

# warm arm
python rosa_main.py --path normalized_shapes_cuboid/bunny_points_cubic_lam3.ply \
  --field --train --samples 29 --epochs 1000 --normal 0.01 \
  --out log/warm_tri_tv01_bunny

# control
python rosa_main.py --path normalized_shapes_cuboid/bunny_points_cubic_lam3.ply \
  --field --train --no-warmup --samples 29 --epochs 1000 --normal 0.01 \
  --out log/blank_tri_tv01_bunny

# repeats (same commands, new output dirs)
#   bunny warm -> log/warm2_tri_tv01_bunny
#   bunny blank -> log/blank2_tri_tv01_bunny
#   armadillo blank -> log/blank2_tri_tv01_armadillo

# seam on any saved checkpoint (writes the D table above)
python plc_utils/seam_check.py log/warm_tri_tv01_bunny/bunny_points_cubic_lam3_model.pt
```

Files: `*_model.pt` (network, grid, queries, keys, plane frame), `*_triplane.npz`
(grid and sampled features), `*_feature_graph.csv` (the D pairs), `planes.txt`
(per-plane and per-loop errors), `*_fits.png` / `*_slices.png`.

Code notes: `rosa_main` used to print a hardcoded "0 runs the way the normals
say, 2 runs the other way round" on its align line; it now prints the actual
reversed count, which is the number quoted in §3/§7.
