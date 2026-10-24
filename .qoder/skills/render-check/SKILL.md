---
name: render-check
description: Render and evaluate trained NISR_Seawall checkpoints for visual inspection, producing a target-vs-prediction gallery over deterministic orbit views, per-patch checkerboard renders, geometry diagnostics (Chamfer, coverage, normals, UV stretch), and a multi-run comparison sheet. Works with fixed-grid single/two/six-sheet atlases and adaptive quad-tree cube checkpoints. Use when asked to inspect or look at renders of a checkpoint or run, check a model visually after training, compare two or more runs, verify a box-initialized six-sheet result, inspect an adaptive dancing_children-style run, or diagnose patch stretching and normals.
---

# Render Check

## Overview

Builds one self-contained folder that answers "does this checkpoint look
right?" for a NISR_Seawall checkpoint. It runs the three inspection scripts in
`scripts/` (gallery, checkerboard, metrics), then assembles a multi-run
comparison sheet and a Markdown report. Everything is read-only with respect
to the checkpoint.

The checkerboard is part of every render, including quick and interim checks
of a run that is still training. It is the only view that exposes per-patch UV
behavior, seams, and stretch; a gallery alone is not a complete render check.
Do not call `scripts/render_gallery.py` directly for inspection. That skips
the checkerboard.

Both atlas families are supported. Fixed-grid checkpoints rebuild their atlas
from stored args; adaptive quad-tree checkpoints rebuild their topology and
treat every leaf as one patch (dancing_children runs have hundreds of leaves).
Adaptive runs use automatic sampling budgets and write a compact palette image
instead of one texture per leaf.

## Quick start

From the repository root:

```bash
/home/lyangiw/anaconda3/envs/seawall/bin/python \
  .qoder/skills/render-check/scripts/render_check.py \
  --checkpoint logs/render_phase3/ab/g06b_box_ngf_120v \
  --compare logs/render_phase3/ab/g05_r1 logs/render_phase3/ab/g05_r3_ngf_ft \
  --out logs/render_phase3/check/g06b_box_ngf_120v
```

`--checkpoint` and every `--compare` value may be a run directory (the newest
`checkpoint*.pt` is used) or an explicit `.pt` file. The first checkpoint is
the primary: its targets label the comparison sheet and its name titles the
folder contents. Later checkpoints appear as extra columns.

For a single run without comparisons:

```bash
... render_check.py \
  --checkpoint logs/render_phase3/ab/g06b_box_ngf_120v/checkpoint.pt \
  --out logs/render_phase3/check/g06b_box_ngf_120v
```

The script defaults to the seawall interpreter when it exists; override with
`--python PATH` or the `SEAWALL_PYTHON` environment variable. It falls back to
CPU automatically when CUDA is unavailable.

Gallery, checkerboard, and metrics run by default. Keep the checkerboard in
every reported check, including interim renders. Use `--no-checkerboard` only
for a deliberately reduced debugging run, and say so when reporting.

## Interim renders while training

When a run is still training, point `--checkpoint` at the run directory; the
newest `checkpoint_N.pt` is picked. Skip the KD-tree metrics to stay quick but
keep the checkerboard:

```bash
/home/lyangiw/anaconda3/envs/seawall/bin/python \
  .qoder/skills/render-check/scripts/render_check.py \
  --checkpoint logs/render_phase4/botijo_adaptive_render \
  --out logs/render_phase4/check/interim_botijo \
  --checker-views 2 --no-metrics
```

Use `--checker-views 4` (the default) when there is time. Re-running later with
the final checkpoint replaces the interim folder contents.

## Outputs

```text
<out>/
  comparison.png                  target + one column per run
  inspection.md                   metrics tables and image links
  check_summary.json              every parsed number
  render_check.log                all stage output
  <run>/gallery/
    target_vs_pred_16.png         pairs, eight columns
    target_16.png, prediction_16.png
    views/target_XXXX.png, views/pred_XXXX.png
    summary.json                  per-view L1, per-channel L1, rendered normals
  <run>/checkerboard/
    checkerboard_all_views.png
    checkerboard_atlas.png        fixed-grid key: swatch per face/row/col
    checkerboard_palette.png      adaptive key: swatch per leaf (id)
    checkerboard_view_XX.png
    textures/SlideN.jpg           fixed-grid only (for patch_vis.py)
    summary.json
  <run>/metrics/
    metrics.json                  Chamfer, coverage, normals, UV stretch
```

Run labels come from the checkpoint's parent directory, sanitized to
`[A-Za-z0-9._-]` and deduplicated. All paths in `inspection.md` are relative
to the check folder.

## Options

| Option | Meaning |
|---|---|
| `--checkpoint PATH` | Primary checkpoint file or run directory (required) |
| `--compare PATH...` | Extra checkpoints or run directories for columns |
| `--out DIR` | Check folder (required) |
| `--device DEVICE` | Passed to the stage scripts (default `cuda`) |
| `--resolution N` | Gallery render size (default 256) |
| `--rows N` | Views in the comparison sheet (default 4) |
| `--checker-views N` | Checkerboard orbit views (default 4) |
| `--checker-resolution N` | Checkerboard render size (default 512) |
| `--metrics-mesh-res N` | UV samples per patch side for metrics (default 0 = auto: 96 fixed-grid, 16 adaptive) |
| `--no-gallery` | Skip the target-vs-prediction gallery |
| `--no-checkerboard` | Skip checkerboard renders; debugging only, not for reported checks |
| `--no-metrics` | Skip geometry metrics |
| `--python PATH` | Interpreter with torch/nvdiffrast |
| `--force` | Re-render even when stage summaries are current |

`--compare` requires galleries. The comparison sheet picks evenly spaced
views, ending at the last view, and draws 48 px column headers.

## Reuse and freshness

Each stage is skipped when its `summary.json`/`metrics.json` exists, names the
same checkpoint, has current settings (resolution, view count, mesh
resolution), contains the expected image files, and is newer than the
checkpoint. Use `--force` after changing a stage script or when the summary was
produced by an older format. A folder with only a gallery is incomplete: rerun
without `--no-checkerboard` and the checkerboard stage fills in.

## Reading the results

- **Full L1** is the six-channel `|prediction - target|` mean over the whole
  frame, matching the NGF training objective. It mixes normals (first three
  channels) and positions (last three). `l1_normal` and `l1_position` separate
  them; position error shrinks with better geometry, normal error with better
  orientation. The 16 orbit cameras are deterministic: when `lambda_render > 0`
  they are the same views the render loss trained on, so this is not a
  held-out test for those runs.
- **Empty-render floor**: the targets have roughly 25% foreground, so hiding
  all geometry still leaves a mean |target| near 0.106. A low L1 alone is not
  evidence of a good surface; check the gallery and coverage.
- **Chamfer** is fully held out only when `lambda_chamfer` is 0. When it is 1
  the run optimized Chamfer, so read it as a fit diagnostic (the dancing runs
  are in this group). A closed multi-face shell wraps the scan, so its unseen
  back layer is counted as error. For six-sheet runs, prefer cloud coverage and
  the visible-layer normal.
- **Rendered normal** compares the predicted and target normal channels on
  pixels covered by both. This is the angle visible in the gallery and the
  direct quality check; it is independent of which mesh layer is nearest in
  3-D.
- **Cloud normal** takes the best-oriented mesh vertex within the search radius
  for each cloud point instead of only the nearest vertex. This helps at thin
  features where a closed shell's back layer is closer than the visible layer,
  but it can still pick a different layer around ears and legs. Use it with
  the rendered angle, not instead of it.
- **Coverage** is the fraction of cloud points within the near threshold
  (default 0.02 in normalized coordinates). Missing ears or legs show up here
  before Chamfer moves much.
- **Checker edge p95** is the longest 3-D edge of an 8x8 UV checker cell,
  per patch. Large values or a lopsided atlas cell mean UV stretch; the
  checkerboard shows where.
- **UV aspect** is the ratio of long to short checker-cell edge. Adaptive
  quad-tree leaves can be strongly anisotropic (dancing_children: aspect p95
  33), which shows up as streaks in the checkerboard even though Chamfer and
  coverage stay good. Treat it as a texture/parameterization diagnostic, not a
  geometry failure.
- **Adaptive leaves** are enumerated from the checkpoint topology; the report
  lists the depth histogram (dancing_children: 726 leaves, depths 1-7).
  `checkerboard_palette.png` is the key for leaf ids and colors.
- **Comparison sheet columns** show the same orbit views and targets. Use it
  to distinguish a globally smooth but incomplete surface from a complete one
  with local artifacts.

## What to report

Give the user clickable local links to `comparison.png`, `inspection.md`, each
run's `target_vs_pred_16.png`, and the checkerboard images
(`checkerboard_all_views.png` plus the palette or atlas key). Always render and
report the checkerboard alongside the gallery; never deliver gallery-only
renders as a finished check. Quote the full L1,
normal/position split, Chamfer, cloud coverage, rendered-normal median/p95,
cloud-normal median/p95, and checker p95. State the atlas mode, patch count,
leaf-depth histogram for adaptive runs, and number of orbit views. Report every
warning (different clouds, mismatched view counts, closed-shell Chamfer bias,
high UV aspect). If a stage was reused, say so.

## Common issues

- **No GPU / nvdiffrast errors**: pass `--device cpu` for the gallery and
  metrics; the checkerboard stage needs nvdiffrast and will fail without it.
- **Missing checkpoint after choosing a run directory**: training may have
  deleted or not yet written the numbered checkpoint; pass the explicit file.
- **Different clouds**: targets are built per checkpoint from its stored cloud.
  The report warns, but the comparison is not meaningful.
- **Old stage output**: rerun with `--force`. Stages are intentionally cached.
- **Long runtime**: the checkerboard samples every active patch and metrics build
  KD-trees over the cloud and sheet. Adaptive runs use smaller automatic grids
  (checkerboard R=12, metrics R=16 for ~700 leaves) so they stay in the same
  range as fixed-grid runs. Run in the background for many runs and report the
  output folder when finished.
- **Adaptive checkerboard is wavy**: expected when leaves stretch one UV
  direction; the per-leaf colors still identify patches. Check `UV aspect` and
  `checker edge p95` before reading it as a rendering bug.
