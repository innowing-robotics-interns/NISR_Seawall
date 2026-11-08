# Triplane features for loop clustering

Status: first comparison, 2026-11-03. Nothing in `loop_propagation.py` was
changed. The numbers below come from the lucy run
`log/cmp_tri_tv01_lucy` and its propagation `log/cmp_tri_tv01_lucy_propagation`.

## Question

Every loop is decoded from one triplane feature vector sampled at its centroid.
If two nearby cross-sections change shape only a little, their features should
also be close. Can that feature distance replace, or at least guide, the
geometric propagation rule?

Short answer: it is a strong **candidate-ranking** signal, not a replacement
for the geometric rule.

- Accepted step-2 edges have median feature distance `0.486`; rejected ones
  `2.053`. A feature threshold ranks them with AUC `0.967`.
- A mutual feature nearest-neighbour graph on adjacent planes recovers `93%`
  of the geometric edges, but only `71%` of its own edges are geometric edges.
- The step-3 seam cuts are not explained by feature distance (AUC `0.601`);
  features would bridge many cuts, including thin slivers, but cannot tell a
  sliver from a genuine continuation on their own.
- The step-4 shear gate is independent of shape: sheared and used chains have
  overlapping feature-step distributions.

The useful role is **feature proposes, geometry disposes**.

## 1. What is stored

`rosa_main.py:851-862` writes `<stem>_triplane.npz` after fitting:

| array | shape | meaning |
|---|---|---|
| `grid` | `3 x C x R x R` | the three feature planes |
| `lo` / `hi` | `3` | sampling bounds, cloud frame |
| `queries` | `N x 3` | the world centroid each loop was decoded at |
| `features` | `N x 48` | trilinear feature at that query (`C = 16`, `R = 32`) |
| `keys` | `N x 3` | `(axis, offset, slot)` in the same loop order |

The loop order is the checkpoint's order, which is also the order of
`preds`, `loop_errors` and `_loop_split.txt` in `rosa_main`. The query of a loop
is exactly the mean of its target points, so a loop PLY joins to its feature row
by centroid (median match error `1e-8` on lucy).

This field was trained with reconstruction plus grid TV
(`--triplane-tv 0.01`), not with an explicit pairwise feature-smoothness term.
`code_smooth` is `0` in this run. The run's own diagnostic already reports that
mutual-nearest geometric neighbours land at median feature distance `0.480`
(ratio `0.237` of the all-pair median `2.024`), so the field does concentrate
neighbouring sections without being told to.

Scale for lucy (157 loops, good-only):

| quantity | median | q10 / q90 |
|---|---|---|
| all loop pairs | 2.024 | 1.289 / 2.870 |
| nearest feature neighbour, any loop | 0.458 | — / 1.022 |

The absolute scale is run-specific: 48-D distances depend on the field
magnitude. Any threshold must be calibrated per run (or whitened) rather than
copied between shapes.

## 2. The rule it is compared with

`loop_propagation.py` decides continuations from geometry and placement only;
the feature field is never read.

- **Step 1, stability** (`loop_propagation.py:344-356`): a loop is stable when
  its sampled points are within RMS `0.025` of the fitted curve in its own PCA
  plane.
- **Step 2, propagation** (`loop_propagation.py:712-841`): only loops on
  adjacent planes of the same axis are candidates. The source loop is projected
  into the target plane at the position it physically occupies, may shift by
  at most `clip(0.18 * min(radius), 0.01, 0.06)`, and the edge is accepted when
  both point coverage and loop support are at least `0.5`
  (`loop_propagation.py:554-569, 814-815`). The graph grows from stable seeds.
- **Step 2b, optional deformation** (`loop_propagation.py:593+`): a bounded
  Fourier deformation of the seed may rescue a rigid miss, but the deformation
  budget is a fraction of the section radius and the target's own loop still
  continues the track.
- **Step 3, seam alignment** (`loop_propagation.py:1121-1201`): consecutive
  loops are opened at a common seam using the switch
  `s = lam / (lam + D)` with `lam = 0.02`. A pair is cut when `s < 0.3`, i.e.
  `D > 0.0467`; a track is split into sub-chains at every cut. Thin slivers
  therefore die here even when step 2 passed them.
- **Step 4, tubes** (`loop_propagation.py:1428-1442`): a chain is rejected when
  its section centre drifts sideways more than `2x` the axial plane step
  (`axis_drift > 2`). This removes sheared sheets that still look like tubes
  locally.

Two properties matter for the comparison. First, adjacency is local and
axis-specific: the rule never asks whether two loops *look* alike, only whether
one explains the other's points at its physical position. Second, the failure
modes on lucy are mostly metric: thin loops are cut in step 3, and wing chains
are rejected in step 4 as sheared. Feature similarity can speak to the first
kind of failure, not the second.

## 3. Feature-clustering rule under test

The natural graph is over loops, not pixels:

- node `i`: feature `f_i` (optionally L2-normalised or whitened per run);
- edge `i-j`: `d_f(i, j) = ||f_i - f_j||`, restricted to loops on adjacent
  planes of the same axis for a propagation comparison;
- a mutual-nearest or thresholded kNN graph is clustered into connected
  components, or agglomerated at a distance gate.

For propagation, the feature graph can enter in three places:

1. **Candidate prefilter.** Rank the loops of the next plane by `d_f` and test
   only the top `k` with the geometric transfer. This is a speed win, not a
   correctness change.
2. **Tie-break / soft gate.** When coverage is marginal, let a small `d_f`
   accept and a large `d_f` reject, with geometry still bounding placement and
   shift.
3. **Cut bridging.** Reconnect a step-3 cut when `d_f` is small and the bridge
   obeys the same physical placement and drift limits. This is the only route
   that could recover the thin wing loops.

It should not replace the drift gate: shear is a spatial property of the chain,
and two sheared sections can have nearly identical shapes.

## 4. Comparison protocol

Join `step2/attempts.csv`, `step3/pairs.csv` and `step4/tubes.csv` to the
`features` array through the query centroids, then measure:

- **Step 2 separability.** Distribution of `d_f` for accepted vs rejected
  edges, per axis, and AUC of `-d_f` for predicting `passed`. Compare with the
  trivial in-plane centroid distance as a baseline.
- **Retrieval.** For each source loop and adjacent plane, rank candidate
  targets by `d_f`; count how often the best accepted target is top-1.
- **Graph agreement.** Mutual feature nearest neighbours on adjacent planes
  vs the accepted geometric edges: precision and recall.
- **Threshold cost.** Pick a `d_f` threshold at a quantile of accepted edges and
  count rejected edges below it (false accepts) and accepted edges above it
  (misses).
- **Step 3 / step 4.** Feature distance for kept vs cut seam pairs, and mean
  consecutive feature step for used vs sheared chains.

The geometric labels are a proxy, not ground truth: a step-2 rejection can be a
true miss, and a step-3 cut can be a genuine thin-loop failure. The comparison
therefore reports agreement, not accuracy.

## 5. Measured comparison (lucy, 157 loops)

### Step 2: accepted vs rejected

359 adjacent-plane attempts, 91 accepted.

| axis | accepted / rejected | median `d_f` accepted | median `d_f` rejected | AUC(`-d_f`) | AUC(`-centroid`) |
|---|---|---|---|---|---|
| x | 39 / 178 | 0.468 | 2.108 | 0.974 | 0.980 |
| y | 22 / 57 | 0.460 | 2.175 | 0.971 | 0.978 |
| z | 30 / 33 | 0.579 | 1.751 | 0.944 | 0.981 |
| all | 91 / 268 | 0.486 | 2.053 | 0.967 | 0.977 |

The accepted-edge median `0.486` is almost exactly the nearest-neighbour median
`0.458`, and the rejected median is the all-pair median. In this run the
geometric rule is effectively selecting feature-nearest pairs.

Centroid distance is marginally better (`0.977` vs `0.967`) because the
geometric rule is local by construction; feature space also contains repeated
shapes elsewhere on the body, which a purely local rule never sees.

### Retrieval and graph agreement

- 90 source/adjacent-plane groups contain at least one accepted edge. Top-1
  retrieval of the best accepted target: feature `97%`, centroid `100%`; mean
  rank `1.03` vs `1.00`.
- Mutual feature nearest neighbours on adjacent planes: 119 edges vs 91
  geometric edges, precision `71%`, recall `93%`.
- A threshold at the 90th percentile of accepted distances (`1.204`) leaves
  `26/268` rejected edges below it (`10%` false accepts) and misses `9/91`
  accepted edges. At the 95th percentile (`1.322`) it is `13%` false accepts
  and `5` misses. At the accepted maximum (`2.027`) half the rejected edges
  fall below it (`49%` false accepts).

So the feature graph is a high-recall, medium-precision candidate generator.
The 29% non-geometric feature edges are worth inspecting: some may be the thin
loops the geometric rule loses, but the comparison cannot separate those from
shape repeats without a manual label.

### Step 3: seam cuts

89 pairs, 60 kept, 29 cut.

| | median `d_f` | AUC(`-d_f`) |
|---|---|---|
| kept | 0.468 | — |
| cut | 0.590 | 0.601 |

Feature distance barely separates kept from cut pairs. The cuts are dominated
by alignment residuals of thin or asymmetric loops, not by shape change. Eight
of the 29 cuts have `d_f <= 0.4`, including pairs at `0.112`, `0.131`, `0.204`
and `0.210`; a feature-based bridge would reconnect these, which is exactly the
kind of intervention that might recover lucy's wing slivers. A looser bridge
threshold would additionally reconnect cuts with `d_f` near `1.2-1.9`, which are
probably genuine discontinuities. A bridge rule therefore needs the physical
placement and drift checks around the proposed reconnection.

### Step 4: shear

| chain group | median consecutive `d_f` | range |
|---|---|---|
| accepted tubes (11 chains) | 0.342 | 0.048 - 0.803 |
| sheared, `axis_drift > 2` (6 chains) | 0.581 | 0.249 - 1.497 |

The distributions overlap. Two sheared chains (`c10` at `0.440`, `c14` at
`0.249`) look as smooth in feature space as accepted tubes. Feature similarity
cannot replace `axis_drift`; at best it is an extra check inside a chain that
has already passed placement.

## 6. Interpretation

- The triplane field encodes cross-section shape and local context well enough
  that an accepted continuation is usually the feature-nearest loop. That is a
  useful prior and a cheap ranking signal.
- It does not encode *where* the loop is along the surface strongly enough to
  decide physical continuation, and it does not encode lateral centre drift at
  all. The geometric placement, coverage, seam and shear checks remain
  necessary.
- The two rules fail on different pairs. Step 2 misses thin loops that step 3
  then cuts; these often have small `d_f` and are the best candidates for
  bridging. Step 4 removes sheared chains that are feature-smooth; no feature
  threshold should override it.
- Feature distance at step 2 is close to centroid distance in ranking power.
  Its advantage is not raw accuracy but that it is defined for *any* pair of
  loops, including pairs on non-adjacent planes or different axes, so it can
  support stitching and clustering where the centroid rule has no candidates.

## 7. Proposed use in the pipeline

A conservative hybrid, in order of increasing risk:

1. **Ranking only (safe).** In step 2, sort each adjacent group by `d_f` and
   test the top `k`. Report the coverage of the geometric graph retained as a
   function of `k`. No accepted edge changes.
2. **Marginal-edge tie-break (testable).** When `cov_pt` is in `[0.4, 0.5]` and
   `cov_loop >= 0.5`, accept if `d_f` is below the run's accepted-edge 90th
   percentile, reject otherwise. Measure the extra tubes and their downstream
   reload checks.
3. **Cut stitching (risky).** After step 3, propose reconnections for cut pairs
   with small `d_f` and a physical placement / drift check; run step 4 on the
   stitched chain. This is the candidate route for the wing loops.
4. **Feature clustering as a diagnostic.** Cluster the whole loop set by `d_f`
   and compare clusters to geometric tracks. Report a confusion matrix
   (cluster x track) rather than changing propagation. This would show whether
   shape families (wings, body, base) cut across the axis tracks. Section 9 is
   the first run of this diagnostic.

Do not add a feature term to the drift gate, and do not replace step 2's
coverage with a pure feature threshold: both lose the placement guarantee that
keeps repeated sections apart.

## 8. Open questions

- Is the 71% precision of the mutual feature graph mostly true shape repeats, or
  the thin-loop misses? This needs a manual label on lucy's wing region, or a
  mesh-based ground truth.
- Does `--code-smooth 0.01` (feature-pair smoothness, `mini_nn.py:780`)
  improve the step-2 AUC or the cut bridging? The present run has `code_smooth
  = 0`; only grid TV was active.
- Does whitening or cosine distance change the ranking? Raw 48-D Euclidean is
  sensitive to channel scale.
- The comparison is one shape. Repeat on armadillo, bunny, horse and the hand,
  and on the 15- vs 29-plane runs, before choosing a threshold. The 29-plane
  run halves the plane spacing and changes the geometric baseline.
- If a feature bridge is used, what is the right unit: a per-run quantile of
  accepted edges, or a distance to the same loop's nearest neighbour?

## 9. Clustering results (lucy)

Status: first run of the diagnostic in section 7 item 4, on the same lucy run
as section 5 (157 good loops, 24 step-2 tracks, 11 tubes). The numbers below
are recomputed from `log/cmp_tri_tv01_lucy` and
`log/cmp_tri_tv01_lucy_propagation`.

### The adjacent-plane graph

Nodes are the 157 good loops; edges are mutual nearest neighbours in raw 48-D
feature space, restricted to loops on adjacent planes of the same axis (152
undirected adjacent-plane pairs, each scored once).

- 119 mutual-NN edges, 38 connected components.
- Against the 91 accepted geometric edges: precision `71%`, recall `93%`
  (85/119 and 85/91).
- 21 of the 24 geometric tracks lie entirely in one component. Three are
  split by the mutual rule: t16 (`31:2` | `32:2`), t18 (`44:1` | `45:2` |
  `46:1`) and t22 (`82:1`, `83:1`, `84:2` | `85:1`, `86:0`). A split means
  consecutive sections of one geometric track are each closer to a different
  loop, so the chain breaks even though the sections are a valid tube.

Four components merge more than one track:

| component | loops | tracks | orphans | axis |
|---|---|---|---|---|
| c20 | 29 | t17 + t19 | 12 | y |
| c33 | 29 | t20 + t21 + t24 | 3 | z |
| c6 | 19 | t4 + t8 + t9 | 7 | x |
| c0 | 9 | t1 + t7 | 0 | x |

Edges are same-axis only, so a component never mixes axes. These merges are
shape families within one axis, and the largest (c20) is exactly the two y
wing tracks that the geometric rule cuts, plus 12 loops that never reach a
track.

### Orphan loops

44 loops are in no track. The feature graph places them as:

| where the orphan falls | loops | reading |
|---|---|---|
| singleton component | 16 | no shape neighbour at all |
| component with one track | 6 | unambiguous adoption candidate |
| component with 2-3 tracks | 22 | belongs to a family, track is ambiguous |

The 22 ambiguous ones are the 12 y loops in c20, the 7 x loops in c6 and the
3 z loops in c33. Feature clustering therefore recovers *coverage* (these loops
have a clear shape family) but not *assignment*. Adopting them still needs a
rule beyond nearest feature: nearest section centre, monotone offset, or the
drift test.

### Threshold sweep

Thresholding instead of mutual-NN trades recall for merges. "Tubes whole"
counts the 11 step-4 tubes whose loops all fall in one component; "sheared
merged" counts the 6 step-3 chains with `axis_drift > 2` that are also whole.

| `d_f <=` | edges | components | precision | recall | tubes whole | sheared merged |
|---|---|---|---|---|---|---|
| 0.4 | 35 | 122 | 0.89 | 0.34 | 7/11 | 0/6 |
| 0.6 | 67 | 90 | 0.85 | 0.63 | 8/11 | 2/6 |
| 0.8 | 92 | 65 | 0.79 | 0.80 | 9/11 | 3/6 |
| 1.0 | 100 | 57 | 0.78 | 0.86 | 10/11 | 4/6 |
| 1.2 | 106 | 51 | 0.75 | 0.87 | 10/11 | 4/6 |

The two right-hand columns decide the question: every threshold that keeps 10
of 11 tubes also merges 4 of the 6 chains the drift gate rejects. Shape
similarity cannot separate a tube from a sheared ribbon, as section 5 already
found at chain level. Lower thresholds avoid the shear merges but shatter the
tubes (7/11 at 0.4).

### What the feature graph adds to geometry

The 34 mutual edges the geometric rule does not accept were re-tested with the
step-2 rigid transfer, then the step-3 seam and step-4 drift formulas.

- At the current `accept = 0.5`, 2/34 pass: `31:0-32:0` at coverage
  `0.58/0.87` and `32:0-33:0` at `0.55/0.84`. Neither was ever tested, because
  `31:0` never became a reached track endpoint. This is a candidate-generation
  gap: the features find a valid edge that the growth order misses.
- Relaxing to `accept = 0.40` passes 13/34 (22 at 0.35, 8 at 0.45). Ten of the
  13 survive the seam switch, forming five pieces:

| piece | axis | max drift | step-4 outcome |
|---|---|---|---|
| `16:0-17:1` | x | 4.73 | sheared |
| `31:0-32:0-33:0` | y | 2.45 | sheared |
| `36:0-37:0-38:0-39:0-40:0-41:0` | y | 2.91 | sheared |
| `47:0-48:0` | y | 3.08 | sheared (would join t17 and t19) |
| `79:0-80:0` | z | 0.15 | tube-eligible |

- The z piece extends tube 20 by one loop (`79:0` is its end, `80:0` is an
  orphan). Step 2 did test `79:0-80:0` and rejected it at `cov_pt = 0.494`,
  0.006 under the gate, so this is purely a coverage-threshold miss.
- The y pieces are the wing: two of them reach t17 at `41:0`, and `47-48`
  would join t17 and t19. They are seam-consistent and have small feature
  distance, but their section centres slide laterally. Step 4 rejects them for
  the right reason. The two edges that pass at the current gate (`31-32`,
  `32-33`) are inside the first y piece, so they too end as a rejected sheet.

Net under the current rules: **one loop recovered** (`80:0` on the z tube),
no new tube. The wing is recovered as a chain but rejected as a tube -- a
representation limitation, not a correspondence failure.

### Unrestricted clustering

Removing the adjacent-plane restriction:

- Mutual NN over all loops: 40 edges, 117 clusters (40 pairs + 77 singletons).
  Cross-body shape repeats at mutual-NN strength are rare.
- Full threshold graph:

| `d_f <=` | components | singletons | largest | largest spatial diameter |
|---|---|---|---|---|
| 0.5 | 99 | 67 | 12 | 0.77 |
| 0.7 | 64 | 34 | 17 | 0.78 |
| 0.9 | 40 | 21 | 30 | 1.26 |

The object extent is `2.0`, so at 0.9 the largest cluster spans more than half
the height. A pure feature threshold merges distinct body parts. Clustering
whole loops by shape is a diagnostic (which loops are shape-similar), not a
propagation rule. The adjacency restriction is what makes the feature graph
useful: `71%` precision and 38 interpretable components.

### Reading

- Use the feature graph as a **candidate generator and orphan prior**, not as a
  gate. It finds valid continuations the growth order never tested and says
  which orphans belong to a shape family, but it cannot choose the track.
- The wing failure is representational. The feature chain is seam-consistent
  but drifts laterally; step 4 is right to reject it as a tube. Covering it
  needs a sheet/ribbon primitive or a cut-and-close step, not a lower feature
  threshold.
- The only actionable gain on lucy is small: rank and test feature candidates
  (including the two that pass at 0.5), and allow the 0.40 coverage rescue
  only with seam and drift checks. Expected gain: one z loop.
- The threshold is not a free parameter: at 1.0 it recovers 10/11 tubes but
  also merges 4/6 sheared chains.

### Artifacts

| file | content |
|---|---|
| `log/cmp_tri_tv01_lucy/triplane_clusters.png` | 2x2 3D views, loops coloured by component; legend `cID axis (count)` for the 16 largest |
| `log/cmp_tri_tv01_lucy/lucy_triplane_clusters.ply` | 10,048 points (157 loops x 64 samples) with per-component colours |
| `log/cmp_tri_tv01_lucy/triplane_clusters.csv` | one row per component: loops, geometric tracks, orphans, axes |

Component IDs are arbitrary but shared between the PNG, PLY and CSV.

### Final feature-cluster tube (`fc`)

Running the 19 feature tracks (137 loops, component c25 excluded because it
holds no stable loop) through the same seam and drift stages gives
**10 tubes / 47 loops** (`log/cmp_tri_tv01_lucy_fc`): 16 aligned sub-chains,
6 rejected as sheared. The z tube grows from 21 to 23 loops
(`59:0` to `81:0`), but the merged x component (t4 + t8 + t9) becomes an
11-loop chain with drift `4.73`, so the valid 6-loop x tube `10:0..15:0` is
lost. Pure feature tracking is therefore a net loss of four loops against the
geometric 11 tubes / 51 loops. The conservative rule in section 9 -- geometric
tracks plus the 0.40 coverage rescue and the seam/drift checks -- keeps all 11
tubes and adds only `80:0`, for **11 tubes / 52 loops**
(`log/cmp_tri_tv01_lucy_fc_plus`, z tube `59:0..80:0`).

| file | content |
|---|---|
| `log/cmp_tri_tv01_lucy_fc/fc.ply` | pure feature-cluster tube surface (10 tubes, 47 loops), one colour per tube, plus used loops and seam connectors |
| `log/cmp_tri_tv01_lucy_fc/fc_cloud.ply` | all slice points plus the pure feature-cluster tube mesh |
| `log/cmp_tri_tv01_lucy_fc/fc.png` | lit six-view sheet of the pure feature-cluster tube surface |
| `log/cmp_tri_tv01_lucy_fc/fc_context.png` | slices and tubes in `+z` and `+x` views |
| `log/cmp_tri_tv01_lucy_fc_plus/fc_plus.ply` | conservative final tube: geometric 11 tubes plus feature-recovered `80:0` (52 loops) |
| `log/cmp_tri_tv01_lucy_fc_plus/fc_plus_cloud.ply` | all slice points plus the conservative tube mesh |
| `log/cmp_tri_tv01_lucy_fc_plus/fc_plus.png` | lit six-view sheet of the conservative tube surface |

## Reproduce

The comparison was a diagnostic join, not a committed script:

1. Fit a triplane run (`rosa_main.py --feature triplane --triplane-tv 0.01`);
   that writes `<stem>_triplane.npz`.
2. Run `loop_propagation.py` on the same run (`--good-only`), producing
   `step2/attempts.csv`, `step3/pairs.csv`, `step3/tracks.csv` and
   `step4/tubes.csv`.
3. Load the loops with `loop_propagation.load_run(run, good_only=True)`, load
   `features` and `queries` from the npz, and join each loop to a feature row
   by `cKDTree(queries).query(L['pts'].mean(0))`.
4. Compute the per-pair feature distance and the metrics in section 4.
5. For section 9, build the mutual-NN graph over adjacent same-axis plane pairs
   and label connected components with
   `scipy.sparse.csgraph.connected_components`; the threshold sweep uses the
   same 152-pair set. Join `step2/tracks.json`, `step3/align.json` and
   `step4/tubes.csv` for the track, shear and tube counts. For the feature-only
   edges, re-run `transfer` and `synchronize` on each pair and compute drift
   from `resample3(L['block'], K).mean(0)` as in step 3.

The join is exact on lucy (median `1e-8`); a run whose loops were reloaded with
different point subsampling should be checked the same way before trusting the
comparison.
