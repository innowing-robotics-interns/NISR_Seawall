# TODO

## Loop orientation: a learned axis condition, not a flip

The loops must come out of the network already oriented. No per-loop sign buffer,
no calibration pass, no flip at export: if a loop disagrees with its normals,
that is a result to explain, not a correction to apply.

- [x] Feed the slice axis to the network as an observation

  The warm-up trains against outward normals (+0.997 cos on the cube), but
  `miniNN.forward` received only `z` and the triplane feature. On a shape,
  64–78% of loop queries read a zero feature at load (§2.1 of
  `CUBE_WARMUP_REPORT.md`), so loops from different plane families received the
  same condition and the network could not know which way round each should
  run. The axis was in the loss, not in the input: the missing observation, not
  the training.

  Built: `miniNN(axis_condition=True)` concatenates the unit plane normal after
  the feature, so the first layer reads `[sin z, cos z, feature, axis]` (53 wide
  for a 48-D field). `Trainer` reads the axes off the targets with
  `Square.from_points`, which is the same +x/+y/+z the direction term uses, and
  builds them whenever `axis_condition` is on — not only when `--normal` is.
  The training pass, `predict`, `predict_tangent`, `bend`, the seam solve and
  the warm-up all pass the same vector; checkpoints store the flag and the
  per-loop axes, and `load_warmup` refuses a checkpoint warmed in the other
  mode. `--no-axis-condition` is the ablation; `warmup/cube_warmup_axis.pt` is
  the axis warm-up and the default checkpoint.

  > Later (2026-11-08): the defaults were inverted. `--no-axis-condition` and
  > `--feature-pool occupied_cells` are now the defaults, and the default
  > checkpoint is `warmup/cube_warmup_random_ranges.pt` (W13 in
  > [WARMUP.md](WARMUP.md)). This entry records the earlier comparison; the
  > axis-on/centroid recipe now needs its flags stated explicitly.

  Epoch 0 on the same loops, zero feature everywhere (the grid stays zero-init):

  | shape | arm | zero features | cos med | reversed | D mean | bend |
  |---|---|---|---|---|---|---|
  | armadillo | off | 119/162 | 0.157 | 21 | 0.0116 | 1049 |
  | armadillo | axis | 119/162 | 0.478 | **0** | 0.0011 | 714 |
  | bunny | off | 59/80 | 0.055 | 7 | 0.0094 | 1071 |
  | bunny | axis | 59/80 | 0.543 | **1** | 0.0002 | 703 |
  | horse | off | 108/139 | 0.260 | 8 | 0.0168 | 1058 |
  | horse | axis | 108/139 | 0.543 | **1** | 0.0006 | 738 |
  | lucy | off | 101/157 | 0.099 | 39 | 0.0186 | 1045 |
  | lucy | axis | 101/157 | 0.443 | **5** | 0.0362 | 703 |

  The zero-feature counts are unchanged, so the improvement comes from the axis
  alone: a loop whose field is the shared zero still gets its own plane. The
  load-time fit error is a little higher (bunny 0.252 vs 0.255, horse 0.411 vs
  0.574) because the axis warm-up draws each family's own square rather than one
  shared square; D and bend are lower.

- [x] Triplane initialisation: zero was right, Xavier was not

  The grid starts at zero, so every cell the warm-up never reads stays zero and
  most shape loops share one condition. Xavier removes that degeneracy but
  hands the warm-up MLP codes it never trained on: at 1000 epochs on bunny it
  ends with 8 reversed loops against 3 from the zero start, D mean 0.031 against
  0.009, and bend 6765 against 4953, for a fit within 0.0006 of the same. The
  init therefore trades a known degeneracy for an out-of-distribution one, and
  the axis observation is what actually supplies the missing information.
  Reverted to zeros; the Xavier checkpoint is left on disk as the ablation.

- [ ] Keep the direction term; treat persistent disagreement as a junction signal

  Do not release the direction term near the points and do not turn loops round
  by hand. Keep the orientation term through the fit (at least at its nominal
  weight; a far-from-target boost may be added, but it never falls to zero), and
  report every loop whose direction stays ambiguous.

  A loop that cannot be made consistent with its target normals is evidence that
  the slice contains a junction, a branch, or several loops — a natural place to
  segment the shape. The orientation residual is therefore a segmentation cue,
  not a failure to hide. Flag loops below a cosine margin and write them with
  the loop key, axis, cosine and fit error for the segmentation step.

  Measured on bunny with two warm-ups that differ only at epoch 0 (zero vs
  Xavier, both 200 epochs, `--normal 0.01`): the end is the same — fit 0.0170 vs
  0.0192, cos median 0.809 vs 0.793, reversed 34 vs 33 — so the start does not
  decide the orientation. The schedule does: the same shape and weight at 1000
  epochs ends with 3 reversed (zero start) and 8 (Xavier start), because the
  anneal is on for five times as long in absolute epochs (it is
  `cosine_anneal(step, total)`). At 200 epochs the weight reaches 0 early and
  the ~34 reversed loops are frozen while the fit error still converges. So the
  budget changes the count by an order of magnitude; the init does not. Keeping
  the term is therefore not optional; the comparison at equal budget is part of
  the acceptance test.

  Measured again on horse with the f map frozen for the first 50 epochs
  (`--freeze-f-epochs 50`, `log/reverse_fix_horse/freeze50_full`): z holds 0
  reversed from epoch 75 to 950, then reads 0 -> 22 clockwise as the direction
  weight falls from 1e-4 to 0 between the run's printed lines 951 and 974, and
  the z fit (pred -> target, per axis) rises 0.00010 -> 0.00044 across that
  flip. The family was upright for 875 epochs under the term and turned over in
  the 23 epochs it was released. A floor on the weight is not a cosmetic
  change; it is what holds the orientation.

- [x] Remove the manual export flip in `rosa_main.fitted()`

  The function currently reverses any loop with mean cos < -0.1 before writing
  the PLYs and figures. Once the axis is an input and orientation is learned,
  that flip hides exactly the signal the previous item wants to keep. Report the
  direction as the network produced it; if a file must follow the cloud's
  normals for a consumer, that is the consumer's rendering choice, not a silent
  change to the loop.

  Done: `fitted()` still counts loops below the −0.1 cosine and prints them, but
  no longer reverses prediction, tangent, normal or per-point error. The loop
  PLYs, fit grids, snapshots and per-plane figures now carry the fitted
  direction, and the summary line says the loops keep it. No measured number
  changes: `align_err` was already computed before the old flip, point errors
  are order-invariant, and winding, fit and the 48-D codes never used the
  turned arrays. Only downstream consumers of the exported orientation change —
  existing runs on disk keep their previously turned PLYs, while new runs and
  `loop_propagation.load_run`'s signed-area canonicalisation see the raw fit.

- [x] Reverse fix: start each loop at its target's scale and place, not at the
  cube's

  Cause. The warm-up was then trained on the unit cube (now the default draws
  random-size cubes; see WARMUP.md W13), so the loop a fit starts from
  is a square the size of the cube's cross-section, mapped by the triplane into
  the cloud's box. A slice's points often occupy a small part of that box and
  sit away from its centre -- on horse y = -0.66 the points span x -0.10..0.36
  and z -0.89..0.29, while the warm-up square spans roughly -0.55..0.59 by
  -0.52..0.46. Reaching the points therefore takes a long translation and
  scaling in the first updates: the loop is dragged onto them and the shared
  field `f` is rewritten at every query, not just bent into shape.

  The early snapshots show that drag and the reversal that follows it
  (`log/axis_seam10_horse_repro`, all loops counter-clockwise through epoch 2).
  At epoch 3, 60 of 66 z loops read clockwise (median foldover 1.39, so they are
  retracing mid-drag); the y family follows at epochs 3-5 (14/30 clockwise at
  epoch 4, 28/30 at epoch 5). Most z loops then turn back (9/66 by epoch 20)
  while the y loops stay turned. Where the section is thin, the dragged curve
  pairs to the opposite sheet of the slice, the cosine reads negative, and the
  winding follows the pairing. The direction term is at its full weight (0.01)
  throughout these epochs, so the flip happens despite it -- the fit, not the
  anneal, is what moves the loop.

  Fix to test: put each loop's start where its targets are -- give the warm-up
  square the slice's own in-plane extent and centroid, either by conditioning
  it on those two numbers or by an initial rigid scale/translation fitted to
  the targets -- so the first updates refine shape instead of moving the whole
  loop. Acceptance: fewer reversed loops at equal fit across the shape panel,
  with the direction term and the exported orientation unchanged.

  First test -- hold `f` fixed for 50 epochs (built). `--freeze-f-epochs N`
  freezes the shared MLP (`Trainer.set_f_trainable`) and trains only the
  triplane grid for the first N epochs. `plc_utils/replay_epochs.py
  --freeze-f-epochs` reproduces such a run, `plc_utils/reverse_fix_report.py`
  measures a saved state (per-axis winding, the direction vote, full-target
  chamfer) and `plc_utils/plot_reverse_fix.py` draws it. Horse, the same
  command and 1000-epoch schedule as `log/axis_seam10_horse_repro`
  (`--normal 0.01 --distill 10`): replay `log/reverse_fix_horse/replay_freeze50`,
  full run `log/reverse_fix_horse/freeze50_full`. The replay's first 50 epoch
  lines are identical to the full run's.

  At the end of the frozen phase (`state_e50.pt`, after 50 updates; the run's
  printed line 49; reversed is own-centre winding < 0, against is the per-loop
  mean direction cosine < -0.1, fit is the full-target squared chamfer averaged
  over loops):

  | arm | x rev | y rev | z rev | against | fit | line-49 fit |
  |---|---:|---:|---:|---:|---:|---:|
  | baseline | 1/43 | 27/30 | 9/66 | 38/139 | 0.00602 | 0.0062 |
  | f frozen | 0/43 | 1/30 | 0/66 | 1/139 | 0.11842 | 0.1206 |

  The baseline really does rewrite `f`: its MLP moves 0.498 relative L2 in the
  first 50 epochs while the field moves 9.06x its own norm. Frozen, `f` moves
  0.000 and the field moves 16.66x. The loops are still placed, but by the
  field, and the reversal is gone in all three families -- at a 20x fit cost.

  The cost is transient. By epoch 200 the two arms have the same overall fit
  (0.0008 frozen, 0.0009 baseline), and y stays upright through 1000 epochs
  (0/30 clockwise, fold 1.05, mean cos +0.81, 0 against; baseline 27/30, fold
  1.15, cos +0.40). The end is not clean: z turns over when the direction
  weight reaches zero (the item above), ending 26/66 reversed (own centre)
  against the baseline's 11/66, and x ends 5/43 against 6/43. The run's own
  final diagnostic reads 7/0/20; the gap to the own-centre 5/0/26 is the same
  target-centroid effect. Final total 31/139 reversed against 44/139, but fit
  0.00125 against 0.00046 and the field is 2.8x rougher (triplane tv 0.0042 vs
  0.0015; relative field change 24.8x vs 12.8x). So the frozen start fixes the
  drag-induced y reversal; it does not hold orientation once the direction term
  is gone.

  The run's own epoch-line diagnostic uses the target centroid and reads
  5/2/19 at the end of the frozen phase even though the loops themselves are
  upright (own-centre 0/1/0): a displaced curve's target centroid is not
  inside it. Read own-centre winding at a mid-fit epoch, not that diagnostic.

  Figures: `log/reverse_fix_horse/matching_y_e50_compare.png` (warm-up,
  baseline e50, frozen e50, six y loops), `matching_z_e50_compare.png`, and
  every loop at each state in
  `log/axis_seam10_horse_repro/matching_{x,y,z}_{baseline,frozen}_e50.png`.

  Second test -- freeze to 100 epochs (`log/reverse_fix_horse/freeze100_full`).
  The same command with `--freeze-f-epochs 100`, 1000 epochs. The bend term is
  gone from the fit outright (it was already at weight 0 in every triplane run
  here, so this run's objective is unchanged); the only smoothness left is the
  triplane tv (weight 0.01). Wall clock 3 min 35.5 s (215.51 s, max RSS
  2.70 GB); plane extraction alone is 26.9 s, so the 1000 epochs and the export
  take about 189 s, 0.19 s an epoch.

  At the end of the freeze (`state_e100`, own-centre winding, full-target
  squared chamfer):

  | arm | x rev | y rev | z rev | against | fit |
  |---|---:|---:|---:|---:|---:|
  | baseline e100 | 1/43 | 27/30 | 9/66 | 33/139 | 0.00194 |
  | f frozen 100 | 1/43 | 1/30 | 0/66 | 4/139 | 0.05658 |

  (the run's own epoch line at 99 reads 5/2/12, the target-centroid effect
  again). At the end (`e999`; own-centre winding, fit as above, bend is the
  trainer's reading over all loops, tv the field's):

  | arm | x rev | y rev | z rev | against | fit | bend | tv |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | baseline | 6/43 | 27/30 | 11/66 | 0/139 | 0.000461 | 4189 | 0.0015 |
  | f frozen 50 | 5/43 | 0/30 | 26/66 | 19/139 | 0.001247 | 3370 | 0.0042 |
  | f frozen 100 | 2/43 | 0/30 | 0/66 | 0/139 | 0.000476 | 2134 | 0.00766 |

  The longer freeze keeps y and z upright to the end (y winding +1.00, fold
  1.01; z +1.00, fold 1.01): the late z flip of the 50-epoch arm never happens,
  and the final fit is within 3% of the baseline, 0.000476 against 0.000461,
  at 2/139 reversed against 44/139. The run's own final diagnostic reads
  6/0/0 against the own-centre 2/0/0. The cost moves to the field: tv 0.00766,
  5.1x the baseline's 0.0015 and 1.8x the 50-epoch arm's. The freeze is also
  expensive mid-run, fit 0.0566 at e100 against the baseline's 0.00194, but it
  catches up by about e200 (0.0014 against 0.0009) and ends slightly ahead.

  Figures: `log/axis_seam10_horse_repro/matching_y_baseline_final.png` and
  `matching_y_frozen100_final.png`, and the same pair for z.

  The code partition confirms the freeze timing and shows what the freeze is
  for. Sampling each loop's 48-D triplane code at the freeze-end state
  (`code_clusters.py --state`, k = 10 on cosine similarity) gives silhouette
  0.578 at e100 and 0.514 at e50. The e100 partition is already the final one
  (ARI 0.907 against `final100`, mean best-match Jaccard 0.814; the x-family
  loops are the tightest clusters, intra cos 0.96); e50 is coarser (ARI 0.650
  against `final50`, one 62-loop front/rear body cluster). So the frozen
  training is a **placement initialiser**: with `f` fixed, the field's only way
  to reduce chamfer (0.0566 at e100) is to move each loop's query onto its
  target points, and by the end of the 100-epoch freeze that move is done and
  the code groups are already the ones the run ends with. This is why the
  orientation survives the unfrozen stage. What follows is a **fit stage**:
  releasing `f` lets the remaining updates spend themselves on shape variation
  -- the per-loop curve deviation and the loop-vs-cloud residual -- not on
  re-deciding the grouping, and fit drops 119x (0.000476) with the partition
  nearly unchanged. The 50-epoch arm stops the initialiser early, its grouping
  is still coarse, and it pays for that with the late z turnover. Figures and
  sphere PLYs:
  `freeze100_full/snapshots/horse_points_cubic_lam3_codes_e0100.{png,csv,ply}`,
  `freeze50_full/snapshots/horse_points_cubic_lam3_codes_e0050.{png,csv,ply}`.

  Next: the field roughness is now the cost (5x the baseline), so test lowering
  it without losing the orientation -- a tv anneal, or releasing f with the
  direction term held at a floor -- and repeat at matched fit on the shape
  panel. A shorter freeze is no longer the direction: the 50-epoch arm brought
  the late z flip back.

  Third test -- the panel of eight shapes, `--freeze-f-epochs 100`, the same
  1000 epochs and `--normal 0.01 --distill 10` (the distillation arm is the
  matched control, so `--feature triplane --triplane-tv 0.01 --snapshot-every
  0` too). Eight runs, 29 planes per axis, output `log/axis_frz100_<shape>`;
  controls are the existing `log/axis_seam10_<shape>` runs (bend weight 0 in
  both arms) plus a new `log/axis_seam10_dancing_children100K` made for this
  pair. The four genus-0 shapes here are bunny, armadillo, horse and lucy.
  Total 1239 loops. Own-centre winding < 0, `against` is the run's per-loop
  direction vote (mean cos < -0.1), fit is the last epoch's mean squared
  chamfer, tv is the last epoch's field term, D is the exported seam median on
  the run's own pair graph:

  | shape | loops | fit ctl -> frz | reversed ctl -> frz | against | bad | tv ctl -> frz | D med ctl -> frz |
  |---|---:|---|---|---:|---:|---|---|
  | bunny | 80 | 0.0010 -> 0.0012 | 5 -> 0 (3/2/0 -> 0/0/0) | 0 -> 0 | 4 -> 3 | 0.00070 -> 0.00536 | 42e-6 -> 42e-6 |
  | armadillo | 162 | 0.0013 -> 0.0011 | 4 -> 1 (1/1/2 -> 1/0/0) | 1 -> 0 | 9 -> 1 | 0.00213 -> 0.01224 | 518e-6 -> 562e-6 |
  | horse | 139 | 0.0005 -> 0.0005 | 41 -> 8 (4/26/11 -> 8/0/0) | 3 -> 0 | 5 -> 5 | 0.00154 -> 0.00762 | 507e-6 -> 660e-6 |
  | lucy | 157 | 0.0007 -> 0.0007 | 33 -> 9 (5/28/0 -> 6/3/0) | 14 -> 2 | 0 -> 0 | 0.00238 -> 0.01216 | 470e-6 -> 422e-6 |
  | laurent_hand | 149 | 0.0006 -> 0.0011 | 41 -> 16 (4/22/15 -> 2/0/14) | 0 -> 0 | 38 -> 34 | 0.00195 -> 0.00687 | 214e-6 -> 212e-6 |
  | rocker-arm | 153 | 0.0007 -> 0.0007 | 59 -> 1 (1/37/21 -> 1/0/0) | 0 -> 0 | 3 -> 1 | 0.00149 -> 0.00446 | 136e-6 -> 163e-6 |
  | neptune | 189 | 0.0005 -> 0.0006 | 20 -> 1 (5/1/14 -> 1/0/0) | 10 -> 6 | 0 -> 0 | 0.00255 -> 0.01046 | 1119e-6 -> 1122e-6 |
  | dancing_children100K | 210 | 0.0009 -> 0.0009 | 4 -> 11 (0/3/1 -> 1/8/2) | 0 -> 1 | 6 -> 9 | 0.00308 -> 0.01361 | 191e-6 -> 154e-6 |

  The frozen start cuts the reversed total 207 -> 47 of 1239 (4.4x). Seven of
  the eight shapes improve; dancing children is the exception (4 -> 11, its y
  family 3 -> 8) and laurent hand loses its y reversal (22 -> 0) but keeps the
  z family (15 -> 14), so the two hard shapes need a seeded repeat. The fit
  cost is transient as on horse -- frozen e99 reads 0.022-0.070 against the
  controls' 0.002-0.0035 -- and by e999 only laurent hand (0.0011 against
  0.0006) and bunny (0.0012 against 0.0010) differ by more than 20%. Bad loops
  fall on five shapes (armadillo 9 -> 1, rocker 3 -> 1), rise on dancing
  children (6 -> 9); laurent hand stays high in both arms (38/34). The field is
  3-8x rougher in tv, and rocker arm's x family pays an extra foldover
  (1.127 -> 1.435 against 1.006/1.007 for y/z). The bend reading falls on six
  shapes (bunny 6000 -> 4083, horse 3983 -> 1962, laurent 9545 -> 4837) and
  rises slightly on neptune (1415 -> 1600) and dancing children (2606 -> 2984).
  Distill D is essentially unchanged at the median (it is the term doing that
  job); lucy's tail improves (mean 0.0253 -> 0.0130, max 1.48 -> 0.405).

  Caveat: no seed was fixed. Discrete winding counts move run to run at this
  radius -- this horse frozen arm reads x 8/43 where `freeze100_full` read
  2/43, while its control reads 4/43 against 6/43 -- so read the panel as
  direction and magnitude, not as exact counts. Next: seeded repeats on
  dancing children and laurent hand, and the direction-weight floor at the end
  of the run, which is where the surviving late turn-overs happen.

  Clustering on the 48-D triplane codes (cosine average linkage, fixed k = 10
  so the shapes are comparable; the e100 state replays the same schedule for
  100 epochs with `--freeze-f-epochs 100` and is kept in
  `log/frz100_replay/<shape>/state_e100.pt`). The replay reproduces the run's
  epoch-99 line to every printed digit on bunny, horse, lucy and dancing
  children.

  | shape | sil ctl final | sil frz final | sil frz e100 | ARI e100/final | Jaccard | ARI ctl/frz final |
  |---|---:|---:|---:|---:|---:|---:|
  | bunny | 0.487 | 0.590 | 0.628 | 0.703 | 0.843 | 0.464 |
  | armadillo | 0.418 | 0.452 | 0.468 | 0.646 | 0.666 | 0.435 |
  | horse | 0.434 | 0.561 | 0.578 | 0.897 | 0.795 | 0.626 |
  | lucy | 0.350 | 0.459 | 0.509 | 0.931 | 0.888 | 0.441 |
  | laurent_hand | 0.390 | 0.486 | 0.454 | 0.841 | 0.812 | 0.345 |
  | rocker-arm | 0.410 | 0.550 | 0.597 | 0.819 | 0.752 | 0.539 |
  | neptune | 0.380 | 0.432 | 0.505 | 0.794 | 0.785 | 0.457 |
  | dancing_children100K | 0.207 | 0.336 | 0.381 | 0.638 | 0.620 | 0.361 |

  The frozen arm separates its codes better on every shape (silhouette
  +0.03..+0.13 at k = 10; the auto-k numbers move the same way, e.g. rocker
  0.550 -> 0.619, dancing 0.264 -> 0.389). At e100 the partition is already
  0.64-0.93 ARI from the run's final one, and the horse numbers reproduce the
  earlier result (silhouette 0.578 at k = 10; ARI 0.897 against the earlier
  0.907 at auto k). The clusters are the axis families: rocker arm's largest is
  37 x loops, bunny's are 26 x and 21 y plus 2 x, lucy has a 21 y cluster,
  laurent hand 47 y. The fit stage then slightly mixes some of them -- e100
  silhouette is above the final one on seven shapes, and only laurent hand
  rises during the fit (0.454 -> 0.486), the same shape whose placement is
  still unfinished at e100 (fit 0.0369). The two arms end on different
  partitions (ARI 0.35-0.63), so the frozen start is not a reparametrisation
  of the same grouping: it is a cleaner one. Figures and per-loop labels:
  `<run>/<stem>_code_clusters.png/.csv/.ply` in each `log/axis_*_<shape>/`.

  Resolved by the frozen start above, not by target-scale initialisation:
  `--freeze-f-epochs 100` holds the shared MLP for the first 100 epochs and
  trains only the triplane field, so the field places each loop while `f`
  keeps the warm-up's map and cannot be rewritten by the drag. On horse it
  ends 2/139 reversed at full-target fit 0.000476 against the baseline's
  44/139 at 0.000461 (z 0/66, y 0/30), direction term unchanged at
  `--normal 0.01`. The field is 5.1x rougher (tv 0.00766 against 0.00151) and
  the freeze costs fit while it lasts; 50 epochs is not enough, because z
  turns over late. The command, tables and figures are in the second test
  above; `WORKFLOW.md` §2.4 documents the flag. The cross-shape panel is still
  open.

- [x] Sample the triplane code over the loop, not only at its centroid

  The exported code is one bilinear sample at the loop centroid (the `queries`
  in `<stem>_triplane.npz`). Centroids of adjacent same-family loops are close
  enough to land in the same triplane cell, so their 48-D codes are nearly
  identical: the code weakly encodes where the loop is and does not encode its
  shape, which is exactly what lofting and clustering need to separate. Horse,
  139 loops: of the 127 adjacent same-slot pairs, 11 (8.7%) share a cell in all
  three planes at R=32; at R=64 that falls to 0, but the sweep-invariant plane
  still aliases almost as often -- x-sweep YZ 13 -> 11 of 37, y-sweep XZ 11 -> 7
  of 28, z-sweep XY 18 -> 17 of 62. Of all 9,591 pairs, 15 share all three
  cells at R=32 and 0 at R=64, but 188 still share the XY cell at R=64 because
  z-sweep centroids barely move in x,y. The invariant block is 16 of the 48
  dims, so a third of the code carries no information about the position along
  the sweep, at any resolution.

  Resolution does not fix it. Adjacent-loop cosine distance rises only
  0.059 -> 0.065 from R=32 to R=64, the participation ratio 9.5 -> 9.9 of 48
  dims, and 99.8% of the R=32 grid energy is below 16 cycles/side; R=64 puts
  only 2% above that, with a *smoother* cell-to-cell field (neighbour-diff /
  variance 0.24 -> 0.13). The res-64 run costs 4x the grid storage for a
  seed-averaged +0.04 silhouette at k=12 (single seed 0.655 -> 0.661) and no
  tube-readiness change (54-55/139 ready, longest chain 18). Run
  `log/axis_frz64_horse/`, group sheet
  `axis_frz64_horse_cluster_loops_spectral_groups.png`, clustering helper
  `utils/spectral_code_clusters.py`.

  Next: sample K points around each loop (mean, or concatenate a few) instead
  of one centroid, and/or widen `--triplane-channels`, then measure
  adjacent-loop code distance, silhouette and tube readiness against the
  centroid baseline on the eight-shape panel; the design is in `loop_code.md`.
  Adding axis/offset to the
  distance was tested separately: it raises axis purity (horse 0.906 -> 0.993
  at k=12) but moves only 18-21/139 labels and leaves readiness unchanged
  (55/139, longest 18), so it is not the fix either.

  Tested: mean over occupied cells, axis input removed, frz100 recipe, all
  eight shapes. The pooled code fits everywhere (final fit 0.0004-0.0010) and
  clusters about as well as the centroid+axis baseline on some shapes, but not
  better: spectral silhouette improves on 4/8 and drops on 4/8, axis purity
  improves on 3/8 and drops badly on lucy (0.917 -> 0.707) and neptune
  (0.804 -> 0.640). Adjacent-loop codes are closer on every shape (horse
  median cosine distance 0.062 -> 0.032, rocker 0.051 -> 0.016; neptune
  near-identical pairs 2 -> 14 of 178), so pooling does not hold neighbours
  apart. Winding consistency, the convention the axis input supplies, degrades
  on 7/8 and collapses on rocker-arm (0 -> 56 of 153 clockwise), armadillo
  (1 -> 42 of 162), bunny (1 -> 23 of 80) and lucy (5 -> 24 of 157). Tube
  readiness is mixed (improves on neptune and lucy, ties on dancing, drops on
  five). Conclusion: keep the axis; pooling is not a substitute for it. Runs
  `log/pool_axisoff_frz100_<shape>/`, warm-up
  `warmup/cube_warmup_pool_noaxis.pt`, summary
  `log/pool_axisoff_eight_summary.md`, full table and reading in
  `loop_code.md`.

  Tested next: the same pool with the axis kept, eight shapes, frz100
  (`log/pool_axison_frz100_<shape>/`, warm-up
  `warmup/cube_warmup_pool_axis.pt`). Axis restores orientation (clockwise
  fraction 16.6% -> 5.1% mean, against-normal alignment back to the baseline on
  7/8) and gives the best tube readiness of the three arms (47.3% vs 45.7%
  centroid+axis and 43.8% no-axis; better on 6/8). It does not restore
  clustering: mean silhouette 0.510 vs 0.527 base, mean purity 0.792 vs 0.857;
  it beats the base on bunny, dancing and laurent only. Adjacent codes are
  still closer than the baseline on 7/8 (bunny 0.010 vs 0.037, rocker 0.018 vs
  0.051), so local code separation remains the open problem. Lucy is an outlier
  (`align` 0.680, 27/157 clockwise, silhouette 0.317); a repeat
  (`log/pool_axison_frz100_lucy_b/`) fits better (align 0.463) but still
  clusters worse (silhouette 0.273, 17/157 clockwise vs 5 baseline), so the
  loss there is stable. Summary `log/pool_axis_three_arm_summary.md`, reading
  in `loop_code.md`. Next: keep centroid+axis as the default, and test a
  neighbour-code separation term or a distance with an explicit shape/offset
  component. (Later, 2026-11-08: this recommendation was not taken; the defaults
  are now `occupied_cells` and axis off, WARMUP.md W13.) Per-plane output is now
  opt-in (`--save-plane-files`), so these
  runs keep only `planes.txt`/`bad_loops.txt`.

## Teacher-student seam distillation (`--distill`)

The cube warm-up sets shape and orientation; the follow-up fit is what has to
make the seam distance `D` small. The warm-up is therefore loaded a second time
as a frozen teacher, read once at the shape's own queries and axes, and its
per-pair `D` is the **ceiling** the student is held under: only the excess over
the teacher is charged, so the term can lower `D` but never raise it. The curves
themselves are never matched pointwise: only `D` is, so a common deformation is
free and shape and orientation stay with the fit. `D` is invariant to
orientation, so the term has no direction of its own: it cannot flip a loop
directly, and any change in the against-normal count is the fit's own outcome —
that division of labour is deliberate.

- [x] Implement the follow-up in `mini_nn.Trainer`

  `Trainer._build_distill` validates the checkpoint kind, axis flag and feature
  width, rebuilds the frozen teacher, maps each shape query through the
  teacher's unit-cube frame (`teacher.centre + (q - centre)/half * teacher.half`),
  and records `D` per neighbour pair. `D` is scale-free, so each side reads
  **its own** curve size — the teacher draws at cube scale (the sizes the
  warm-up trained on), the
  student at the shape's — and the two relative disagreements are compared. The
  pair's fixed shape-target radius only carries the dimensionless excess into
  chamfer units: `weight * scale * max(D_student - D_teacher, 0)`. At step 0 the
  student is the teacher, both `D` values are equal, and the term is exactly
  zero. Flags: `--distill W` and `--distill-warmup CKPT` (defaults to
  `--warmup`); off by default.

  Two wrong versions were measured and replaced. Dividing the teacher's `D` by
  the shape target's radius inflated its profile wherever a shape loop is small:
  on armadillo the teacher's mean/max read `0.32 / 13.0` instead of
  `0.0011 / 0.023`, and matching it raised the student's own-scale `D` mean
  `0.108 -> 0.132` and max `1.28 -> 3.59`. Matching the profile symmetrically
  (with the denominator fixed) still pulled lucy's worst pair up to the
  teacher's `2.14` from the baseline's `1.53`. Hence the own-scale comparison
  and the one-sided ceiling.

- [x] Measure at 1000 epochs on six shapes (`W = 10`, 29 planes)

  Matched arms: `log/axis_tri_tv01_<shape>` (baseline) and
  `log/axis_seam10_<shape>` (distillation), both axis-conditioned triplane at
  `--normal 0.01 --triplane-tv 0.01`, same loops and pairs. Each cell is
  baseline → distillation.

  | shape | loops / pairs | fit | bend | against normals | bad | D med | D mean | D p90 | D max |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | bunny | 80 / 61 | 0.0007 → 0.0010 | 4968 → 6000 | 0 → 0 | 3 → 4 | 0.000845 → 0.000042 | 0.009591 → 0.000168 | 0.021680 → 0.000519 | 0.170108 → 0.001383 |
  | armadillo | 162 / 108 | 0.0007 → 0.0013 | 3484 → 2256 | 4 → 1 | 8 → 9 | 0.024032 → 0.000518 | 0.108036 → 0.002483 | 0.271146 → 0.005144 | 1.275502 → 0.032433 |
  | horse | 139 / 99 | 0.0003 → 0.0005 | 5320 → 3983 | 11 → 3 | 4 → 5 | 0.011446 → 0.000507 | 0.022065 → 0.001094 | 0.061962 → 0.002535 | 0.151364 → 0.008623 |
  | lucy | 157 / 83 | 0.0004 → 0.0007 | 2280 → 3969 | 13 → 14 | 0 → 0 | 0.056359 → 0.000470 | 0.289455 → 0.025304 | 0.883988 → 0.011167 | 2.798740 → 1.480760 |
  | laurent hand | 149 / 90 | 0.0004 → 0.0006 | 8791 → 9545 | 0 → 0 | 33 → 38 | 0.004851 → 0.000214 | 0.067148 → 0.000826 | 0.074943 → 0.002397 | 3.228513 → 0.011227 |
  | rocker arm | 153 / 119 | 0.0005 → 0.0007 | 14080 → 6530 | 1 → 0 | 0 → 3 | 0.001296 → 0.000136 | 0.045203 → 0.000734 | 0.171421 → 0.001625 | 0.650649 → 0.014472 |

  The term works on all six: the median falls 9.5–120×, the mean 11–81×, the
  p90 24–105×, and the max 18–288× (`lucy` is the exception at 1.9×). Fit
  costs at most `0.0006` in absolute terms (`0.0003 → 0.0005` on horse, `0.0007 →
  0.0013` on armadillo, `0.0005 → 0.0007` on rocker arm); no shape changes its
  loop count or its pair graph. `bend` is printed with weight 0 in these runs,
  so its move is curve roughness, not a loss cost; rocker arm's roughness falls
  the most (`14080 → 6530`) while its bad-loop count rises.

  "Against normals" is the per-loop direction vote (mean cosine below `-0.1`),
  measured before the export flip. `D` is orientation-blind, so the term does
  not target this count; what changes is the fit's own outcome: armadillo
  `4 → 1` and horse `11 → 3` improve, rocker arm goes `1 → 0`, bunny and
  laurent stay at zero, lucy goes `13 → 14`. The mean direction penalty moves
  with it (`1 - cos`: bunny `0.104 → 0.103`, armadillo `0.314 → 0.265`, horse
  `0.377 → 0.321`, lucy `0.462 → 0.531`, laurent `0.266 → 0.345`, rocker arm
  `0.400 → 0.470`). The reporting line used to hard-code "0 aligned and 2
  reversed" beside this vote; it now prints the measured count, so these are
  read from the `N of M loops ran against the points' normals` line rather than
  the stale prefix.

  The cosine vote undercounts a folded family. Rocker arm's signed winding
  (`curve_winding`, centre at the loop's target centroid) shows its y family
  mostly clockwise, `33/57 → 41/57`, mean winding `0.000 → -0.439`, with x at
  `1/59 → 0/59` and z at `21/37 → 21/37`. Those y loops have median foldover
  `1.569 → 1.281`: forward and backward stretches pair with the same target
  points, so the local cosine averages out and the vote reads almost nothing.
  `D` is orientation-blind, so distillation neither caused nor removed this;
  it is the fit's own outcome and reproduces the horse y-family reversal. Read
  the signed winding next to the cosine vote, not instead of it.

  Two honest tails. `lucy`'s mean and max stay high because the teacher's own
  profile is high there (mean `0.0362`, max `2.1419`): some pairs in lucy have
  no consistent seam even for the warm-up, and a `1.48` max under a `2.14`
  ceiling is the hinge declining to move a loop far from its points. That pair
  is a junction candidate, not a distillation failure. `laurent hand` pays +5
  bad loops (`33 → 38`), the largest quality cost of the six; rocker arm pays
  +3 (`0 → 3`). The student does not always reach the ceiling (`laurent` D mean
  `0.000826` vs teacher `0.000237`), so the reported residual `distill`
  (`4e-6`–`1.4e-5`) at the end is real: fit and seam still trade against each
  other. The ceiling is approached, not attained: 45/61 bunny, 92/108
  armadillo, 71/99 horse, 54/83 lucy, 77/90 laurent and 83/119 rocker arm pairs
  still sit above their teacher `D`, by a median excess of `1.5e-5` to
  `2.0e-4`.

## Backlog

- [x] Can we build a triplane to do the feature?

  Built: `--feature triplane` conditions the shared MLP on three sampled 2D grids
  instead of a row per loop (`mini_nn.TriplaneField`, exported as
  `*_triplane.npz`). On 29-plane dancing children at 32 cells it matches the row
  table on fit and is slightly ahead downstream (35 tubes / 138 loops vs
  34 / 133); continuity alone does not smooth the features, the `--code-smooth`
  term is still needed. 64 cells is worse; a 60-step warm-up gives the best fit.
  Details, table and figures: [loop_propagation.md](loop_propagation.md#triplane-feature-field-instead-of-one-row-per-loop).

## Loop alignment and generalized cylinders

Currently, we are using linear interpolated features along the axis.

**Implementation:** [align_loops.py](plc_utils/align_loops.py) — standalone module (numpy + optional scipy only). It implements the pair residual `D_ij`, the closed-form switch `s* = λ/(λ + D)`, the seam-offset solve, `v_i`, and the loft; `python plc_utils/align_loops.py` runs a synthetic self-test. It is not imported by `loop_extraction.py` or `rosa_main.py` yet.

- [x] How to align two adjacent loops if they can be aligned, and drop those pairs that cannot be aligned

  Implemented as `pair_residual` (FFT circular shift + parabolic refinement) → `switch(D; λ)` → keep when `s ≥ min_switch`; seam offsets come from the weighted least-squares solve. On hand planes y = −0.71 → −0.36 all 3 adjacent pairs are kept (`s = 0.857/0.767/0.806`, `D = 0.0033/0.0061/0.0048`), seam spread 0.030, residual RMS 7.5e−4. Visual: [hand_alignment.png](log/align_loops/hand_alignment.png), numbers: [hand_alignment.json](log/align_loops/hand_alignment.json).

- [x] Whole-hand pass (λ = 0.02, min switch 0.3): 45 planes = 23 single-loop + 22 multi-loop (55 loops). The 23 single-loop planes form an x chain (14 planes) and a y chain (7 planes); 8/19 adjacent pairs pass. That gives **3 tubes covering 11 loops** — x planes 1–2, x planes 11–13, y planes 16–21 (6 loops, v span 0.607, s 0.623–0.742, seam 0.069). The other 12 single-loop planes stay outside (10 fail a pair switch, 2 z planes have no chain partner). The y tube is solid; the x 1–2 tube is shape-aligned but its centres jump 4.1× the plane spacing (`Δc/Δoff`), and y 15–16 is marginal (s = 0.2998). Multi-loop planes cannot join a tube with the current one-loop-per-plane API, so most of the hand's 78 loops remain uncovered. Visual: [hand_tubes.png](log/align_loops/hand_tubes.png), numbers: [hand_tubes.json](log/align_loops/hand_tubes.json), meshes: [hand_tubes.ply](log/align_loops/hand_tubes.ply) and [hand_tubes_cloud.ply](log/align_loops/hand_tubes_cloud.ply).

- [ ] Can we make generalized cylinders from the loops

### A smooth switch with a closed-form solution

Use the switchable least-squares objective

$$
E_{ij} = s_{ij}^2\, D_{ij} + \lambda\,(1 - s_{ij})^2, \qquad \lambda > 0
$$

Here $D_{ij}$ is already a squared alignment error, so $s_{ij}^2 D_{ij}$ means multiplying the underlying residual by $s_{ij}$ before squaring. This is the standard structure of a switchable least-squares constraint.

### Synchronized loops define the tube coordinates

- [x] Assign a longitudinal coordinate $v_i$ to each loop — for example, cumulative distance between centers:

$$
v_0 = 0, \qquad v_{i+1} = v_i + \lVert c_{i+1} - c_i \rVert
$$

- [x] Now the data are samples of a two-parameter surface:

$$
F(z, v_i) = g_i(z + \alpha_i)
$$

Where:

- $z$ runs around the tube
- $v$ runs along the tube
- $F(0, v)$ is the seam formed by the synchronized starting points

- [x] First reconstruction: interpolate between successive loops with a simple loft (this is proposed, not a guarantee of staying on the original surface):

$$
F(z, v) = (1 - t)\, x_i(z) + t\, x_{i+1}(z),
\qquad
t = \frac{v - v_i}{v_{i+1} - v_i}
$$

  `loop_coordinates`, `synchronize`, and `sample_tube` implement this; the lofted hand tube is the bottom-right panel of [hand_alignment.png](log/align_loops/hand_alignment.png) (`v` span 0.362). The loft passes through the four synchronized loops but is not fitted to the cloud, so it is not guaranteed to stay on the surface.

- [ ] Alternative: fit a shared neural surface $F_\theta(\mathrm{PE}(z), v)$ to the synchronized loops while regularizing smoothness between them; generalized-cylinder models similarly represent surfaces through a longitudinal path and varying cross-sections

### Wiring alignment into training: two meanings, staged

The seam is a free gauge in the current objective: Chamfer, bend and the normal
term are all invariant to $f(z) \mapsto f(z + p)$, so "wiring alignment into
training" can mean two different things. They should be staged.

1. **Gauge alignment, no change to the objective.** Give every loop a phase
   buffer $p_i$ and evaluate $z = (z + p_i) \bmod 1$ wherever the network is
   sampled. Solve $p$ against the `code_neighbour_graph` pairs with the periodic
   closed-form residual already in [align_loops.py](plc_utils/align_loops.py)
   ($D_{ij}$ over circular shifts, sparse weighted least squares, switch
   $s = \lambda/(\lambda + D)$). The solve can run after or between stages,
   any number of times, on any checkpoint. Persist $p$ in `*_model.pt` and
   write `phases.csv`, so predictions, snapshots and exports report the same
   seam. This only changes where $z = 0$ is placed; the fitted curves and every
   loss value stay the same. It is the prerequisite for tubes and does not
   depend on the conditioning mode.

   **Done** — `--seam-align` (see [WORKFLOW.md](WORKFLOW.md) §2.3). Two
   departures from the plan above: the phases are solved by Adam on a
   periodic linear interpolation of the detached curves rather than by the
   closed-form least-squares solve, so the trainer needs no `align_loops`
   import; and only `*_model.pt` is written (`phases`), no `phases.csv`.
   Each pair is matched at its best circular shift first and votes
   `lam/(lam + D_best)`, so a pair of two different loops cannot drag the
   phases; the energy is the seam distance `D` itself, because the switched
   value saturates and leaves a far seam with almost no gradient. Verified on
   the trained 29-plane bunny: every one of 51 pairs moves to its best
   achievable shift, none worse, p90 `D` `0.979 -> 0.0385`; a 30-epoch run
   has identical epoch lines with the solve on and off.
2. **Carry the phase on the warm-up, so the checkpoint needs no buffer. Done**
   — `--seam-teacher` in [rosa_main.py](rosa_main.py) and
   `Trainer.pretrain` in [mini_nn.py](mini_nn.py). The phase is not distilled
   from the fitted checkpoint; the checkpoint supplies one number per loop, and
   a fresh fit learns the seam from the warm-up target.

   **Recipe.** Pass 1 is a normal `--seam-align` run (above); its checkpoint
   stores `phases`. Pass 2 runs the same shape and the same first-stage plane
   set with `--seam-teacher <pass1_model.pt>`, keeps `--pretrain` on, leaves
   `--seam-align` off, and fits from scratch. Each loop's phase is looked up by
   `(axis, offset, slot)`; the universal square's points *and* edge normals are
   read at `z + p_i`:

   $$Q_i(z) = Q(z + p_i), \qquad n_i(z) = n_Q(z + p_i)$$

   Nothing else changes: no teacher weights, grid or fitted curve is copied,
   the published objective is untouched, and the refit writes `phases = None`.
   The warm-up runs on stage 0 only; planes added by refinement are seeded as
   usual.

   **Why the warm-up and not the fitted curves.** Chamfer, bend and the normal
   term are invariant to `f(z) -> f(z + p_i)`, so the fit cannot hold a seam;
   the only place to put the phase without a seam term is a target the fit
   starts from. The square is shape-independent and already the pretraining
   target, so the phase enters as one extra turn on that target.

   **Measured** on bunny, 5 planes, 13 loops, 2×16 net, 300 epochs, seam solve
   every 10 epochs (100 steps), `--normal 0.01`, TV `0.01`:

   | run | fit | exported D | p90 | max |
   |---|---:|---:|---:|---:|
   | teacher, unshifted seam | 0.0099 | 0.5121 | 2.0458 | 2.2619 |
   | teacher, aligned seam | 0.0099 | 0.1470 | 0.5137 | 0.8178 |
   | refit, 300 warm-up steps | 0.0092 | 0.3407 | 0.8285 | 1.0249 |
   | refit, 40 warm-up steps | 0.0152 | 0.2912 | 0.7824 | 0.8591 |

   The refit removes about a third of the unshifted seam distance at equal fit
   error, but it does not reach the teacher's aligned `0.15`: 3 of 10 pairs
   (40-step) and 5 of 10 (300-step) are more than `0.02` worse than the
   teacher. Short 30-epoch smoke runs can look better (median `0.06`) because
   the curves have not moved far yet.

   **Why it drifts.** `p_i` is an offset on the teacher's fitted
   parameterisation, but the warm-up applies it to the square, whose arc-length
   distribution differs; sample 0 therefore lands only approximately at the
   teacher's geometric direction, and the free fit can move it further. Carrying
   the phase exactly would need the teacher's fitted curves (a post-fit
   distillation, not this path) or a live seam term (which changes the
   objective).

   **Acceptance** — [seam_check.py](plc_utils/seam_check.py) reads the checkpoint's own
   `*_feature_graph.csv`, rebuilds the curves on its `z` grid and prints the
   pair `D` at the exported seam (and at `z = 0` when the checkpoint has
   phases), plus a per-pair comparison with `--teacher`. Verified:

   - both refit checkpoints have `phases = None`;
   - `D` at `z = 0` improves over the teacher's unshifted seam (`0.51 ->
     0.29–0.34`) at the same or better fit;
   - the teacher's own exported `D` is `0.147`, so the refit leaves a gap;
   - 5 of 10 pairs exceed teacher + `0.02`; the tail is not hidden by the median.

Both stages are now implemented: (1) gives aligned checkpoints with identical
geometry; (2) moves that alignment into the warm-up target so the checkpoint
needs no phase buffer, at the measured drift above.

## Seed from confident loops, then propagate

Right now the tubes may be wrong because of the loop-extraction algorithm. Start from the most confident extracted loops and propagate the construction to nearby planes.

Probe of the propagation idea (rigid transfer, no refit): take the stable y loops 16–21, transfer one to each neighbouring plane, and measure two-sided coverage at 0.02 (1% of the hand extent). On their own planes the loops cover only 0.64 of the slice points on average (they are smooth envelopes of the blocky cubic slices), so that is the ceiling. Leave-one-out inside the band with the extrapolated centre gives 0.49–0.59 of the points and 0.59–0.69 of the loop for planes 17–20. If the loop is registered to the target cloud by translation only (shape fixed, placement from the cloud), the explanation is much better: plane 15 0.63/0.60, 16 0.74/0.82, 17–20 0.59–0.66/0.69–0.75, 22's main slab 0.58/0.58 (its detached thumb 0.09), 23 0.45/0.52, 24 0.34/0.40, 25 0.32/0.30. Beyond 24 point coverage can look like 0.37–0.42 but loop-side support is 0.13–0.21, so those are partial-arc coincidences, not explanations. Two conclusions: the stable loop's shape transfers well and the extrapolated centre is the weak link (e.g. plane 15 is 0.34/0.33 with the tube centre but 0.63/0.60 when registered); and the failure after 24 is topological, because the detached digits are separate loops and the stable loop's upper arc is exactly the part that becomes them. A free deformable fit (Fourier template) covers 100% of 15–21, but it is just a re-fit and no longer tests the seed. Numbers: [loop_explain_neighbors.json](log/align_loops/loop_explain_neighbors.json); panels and coverage curve: [loop_explain_neighbors.png](log/align_loops/loop_explain_neighbors.png).

- [x] Define a confidence score for each extracted loop
- [x] Use the highest-confidence loops as seeds
- [x] Propagate the construction (alignment → synchronization → tube coordinates) to nearby planes
- [x] Rebuild or replace tubes whose loops fall below the confidence threshold
- [x] Allow a bounded deformation to fit the neighbouring point cloud

  Implemented as `loop_propagation.py --step 2b` (see [loop_propagation.md](loop_propagation.md) §Step 2b). The seed may shift within the step-2 cap and deform along its in-plane normal by a constant taper mode plus 8 Fourier modes, fitted by robust ICP. An edge is admitted when rigid passes **or** deformation passes with an RMS offset ≤ 0.15 of the smaller section radius; the target's own fitted loop still continues the track, so the deformed seed is evidence only. On the hand: 52 accepted edges vs 37 rigid, 13 accepted edges need a deformed seed, 2 rejected only by the budget; 9 tubes / 30 loops vs 8 / 26 (a new distal `x` tube and a 6-loop palm `y` sleeve). The budget sweep 0.05/0.10/0.15/0.20 gives 37/43/52/52 edges, and accepted rescues need ≤ 0.143 while the next rejected pair needs 0.264. The remaining failure is still topological: a section that splits into detached digits cannot be explained by deforming one closed loop. Next step: cut the seed at that junction and close each arc against its own component.

## Polycube from the loops

We now have, per shape, fitted loops on 87 planes (`*_loops.ply`, and the same
fit replayable from `*_model.pt`), each stored in its plane's local frame with
the plane's origin and normal, plus tube chains and synchronized loops from
[align_loops.py](plc_utils/align_loops.py). The question is how to turn these 1-D contours
into a polycube: a coarse, axis-aligned cuboid complex approximating the same
solid.

Why the loops are a reasonable starting point: a polycube's faces are
axis-aligned, so cutting it on an axis plane leaves an orthogonal polygon, and
each extracted loop is already the smooth outline of one such slice. The three
plane families (x, y, z) give redundant cross-sections of the same solid.

- [ ] Extract a polycube from the loops we have

  Candidate routes, cheapest first:

  1. **Slice rasterisation** — rasterise every loop into a 2-D occupancy grid in
     its plane, stack the grids along each axis with interpolation between
     planes, threshold to a 3-D volume, then run a voxel-to-polycube pass
     (dominant-normal snapping plus cube merging). The three axis reconstructions
     can be intersected to remove family-specific artefacts.
  2. **Loft and quantise** — reuse the tubes from [align_loops.py](plc_utils/align_loops.py):
     loft between synchronized adjacent sections, snap the lofted surface
     normals to ±x/±y/±z, and rasterise the snapped surface. This reuses the
     seam alignment and the `v` coordinate instead of re-deriving them.
  3. **Half-space carving** — treat every loop as the boundary of the solid in
     its plane, carve the bounding box by the inside/outside test of all loops
     and their holes, then quantise the remaining volume.

  Decisions the first pass has to settle:

  - cube size (cubes across the longest axis) and regular grid vs octree;
  - how to combine the three axis families without double-counting, and how to
    reconcile a cell that one family calls solid and another calls empty;
  - multi-loop planes: holes and detached components are separate loops, so a
    plane must not be filled as one polygon;
  - closure and splits: a polycube is a closed manifold, so end caps and
    junctions where one loop becomes several need explicit handling;
  - evaluation: volume IoU against a voxelised input cloud, Chamfer to the loops
    and to the cloud, cube count, and a manifold/self-intersection check.

  Deliverable: a standalone `polycube_from_loops.py` producing the occupancy
  volume, the cube mesh, a per-family comparison figure, and those metrics.
  Run it first on horse (clean and elongated) and dancing children
  (multi-component), then on bunny and armadillo, whose cross-sections changed
  fastest along the axis.

### Cubic domain from the code clusters

`code_clusters.py` already labels every loop from its 48-D triplane code and
writes `<run>_code_clusters.csv` (`axis, offset, slot, x, y, z, cluster`), plus
the matching PNG and sphere PLY. That gives a part-level partition of the loops;
the open question is how to turn it into the cubic domain the polycube is built
on — one axis-aligned cube grid, or one box per cluster merged afterwards.

- [ ] Build a cubic domain from the loop cluster labels

  Inputs per run: `<run>_triplane.npz` (queries, keys, features),
  `<run>_code_clusters.csv` (labels), the exported loop PLYs, and the source
  cloud. Candidate routes, cheapest first:

  1. **Cluster boxes on a common grid.** For each cluster take the bounding box
     of its loop points (optionally including the in-band cloud points), snap
     the faces to a shared cube size, and fill the cells inside each loop. The
     domain is the union of the per-cluster boxes; overlaps and gaps are
     resolved by the loops' inside/outside tests.
  2. **Cluster label field.** Interpolate a per-voxel cluster label from the
     loop centroids (nearest loop, or a classifier on the same triplane
     features), fill each label's region up to its loop boundaries, and
     quantise the result to cubes.
  3. **Cluster-wise lofts.** Keep only clusters whose loops form tube chains,
     loft within each cluster reusing `align_loops.py` synchronisation, snap the
     lofted normals, and rasterise. Clusters without a chain fall back to
     route 1 or 2.

  Decisions:

  - one global grid vs per-cluster boxes; if per-cluster, how to merge the boxes
    into a single closed manifold without gaps or double-covered cells;
  - spatial connectivity: cosine clustering can join loops from disjoint parts
    (e.g. left and right legs of the same code family), so a cluster must be
    split by the loop adjacency graph or by position before it can seed a box;
  - how many clusters: silhouette `k` answers a clustering question, not a
    domain-complexity one; the domain route may need a coarser `k` or an
    explicit merge/split pass;
  - holes and multi-loop planes: detached components stay separate, and a plane
    must not be filled as one polygon;
  - cube size, and the same evaluation as the polycube route (volume IoU,
    Chamfer to loops and cloud, cube count, manifold/self-intersection check).

  Deliverable: a `cluster_domain.py` (or a `--domain` mode of
  `code_clusters.py`) that writes `*_cluster_domain.npz` with labels, per-cluster
  boxes and occupancy plus a cube mesh, tested first on horse and dancing
  children.

## Pretrain f, then freeze it for the shape fit

- [ ] Freeze the pretrained `f` for the whole fit and train only the feature

  For cubes the map from the warm-up square to the shape should be fixed: the
  cube warm-up already gives `f` the shape-independent square, and the existing
  `--freeze-f-epochs 100` shows that a frozen start places the loops with the
  triplane field alone, keeps the code partition (`ARI` 0.64–0.93 at e100) and
  cuts reversed loops 207 → 47 of 1239. Now hold `f` frozen past the placement
  phase — the entire run — so the only trainable quantity is the triplane
  feature, and the loops become a read-out of a fixed cube-to-shape map.

  Acceptance: on the eight-shape panel at the frz100 recipe, compare the
  fully-frozen arm against the 100-epoch freeze and the joint baseline on fit,
  reversed/against counts, code silhouette and axis purity, tube readiness, and
  field tv. The known risk is the field roughness the shorter freezes already
  show (tv 3–8× the baseline) and a fit floor once `f` cannot adapt; report
  both rather than only the orientation gain, and check that the code clusters
  do not collapse when the feature is the only degree of freedom.

  First test: rocker-arm and laurent hand, 1000 epochs, `--freeze-f-epochs
  1000`, axis off, `occupied_cells`, `--normal 0.01 --distill 10`; the control
  is the same path with the 100-epoch freeze
  (`log/pool_axisoff_frz100_<shape>`). Cells are control -> full freeze.

  | shape | fit | align | reversed | bad | sil | purity | ready | longest | adj cos |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | rocker-arm | 0.0008 -> 0.0174 | 0.369 -> 0.505 | 56 -> 59 of 153 | 1 -> 28 | 0.647 -> 0.548 | 0.954 -> 0.922 | 129 -> 130 | 19 -> 14 | 0.0161 -> 0.0303 |
  | laurent_hand | 0.0006 -> 0.0103 | 0.194 -> 0.475 | 12 -> 10 of 149 | 35 -> 43 | 0.555 -> 0.501 | 0.980 -> 0.946 | 62 -> 63 | 14 -> 14 | 0.0675 -> 0.1230 |

  The fit floor is real: 17–22× the control, with field tv ending at 0.103
  (rocker) and 0.119 (laurent) against a 0.01 regulariser — the feature alone
  cannot hold the fit once `f` is fixed. Orientation does not improve either
  (laurent 12 -> 10 is the only move; rocker's x family stays turned, 59/59
  against 56/59), and code clustering degrades on both (silhouette -0.05 to
  -0.10, adjacent cosine distance almost doubles). Conclusion: the whole-run
  freeze is not the route. The 100-epoch freeze is a placement initialiser and
  `f` must be released for the fit stage; a longer freeze would need a
  different `f` (trained on cubes with the shape variation left to the feature)
  or a field-side capacity/regularisation fix, not just a longer hold. Runs:
  `log/freeze1000_rocker-arm/`, `log/freeze1000_laurent_hand/`.

## Cross-attention from loops to cubes

- [ ] Consider cross-attention to get a cube from the loops

  Let a set of cube tokens query the loop tokens instead of rasterising and
  unioning the three plane families by hand. Loop tokens carry the 48-D triplane
  code plus axis/offset (optionally a few sampled curve points), cube tokens
  carry their grid position; cross-attention aggregates the loops that pass
  through or near each cube, and a head predicts occupancy or the six face
  labels. This is a learned replacement for the intersection/merge step in
  "Slice rasterisation" and could consume multi-loop planes and holes as
  separate tokens rather than as one polygon.

  Open questions: cube resolution and locality of the attention, permutation
  invariance over loops, how to enforce a closed manifold (quantised occupancy
  vs face logits plus a snapping/merge pass), and supervision — voxelised input
  cloud, the loops' own inside/outside tests, or both. Deliverable: a prototype
  that takes a run's loop PLYs plus `*_triplane.npz`, predicts a cube occupancy
  grid, and is scored with the same volume IoU / Chamfer / manifold checks as the
  raster route; test on horse and dancing children first.