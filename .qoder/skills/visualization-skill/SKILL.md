---
name: visualization-skill
description: Export the latest checkpoint of a NISR_Seawall training run as checkerboard-textured per-patch meshes. Use when asked to visualize the latest checkpoint, render the newest run folder for a model (horse_rotate, xyzrgb_dragon, bunny, etc.), inspect the final reconstruction after training, or produce patch PLY/SVD files from a checkpoint.
---

# Visualization Skill

## Overview

Finds the newest run folder under `logs/` for a model, selects its latest checkpoint
(final `checkpoint.pt` or the most recently written `checkpoint_N.pt`), and exports
checkerboard patch meshes by calling `utils/patch_vis.py`.

## Quick start

Run from the repository root:

```bash
/home/lyangiw/anaconda3/envs/seawall/bin/python \
  .qoder/skills/visualization-skill/scripts/visualize_latest.py xyzrgb_dragon
```

Omit the model name to pick the newest run across all of `logs/`:

```bash
/home/lyangiw/anaconda3/envs/seawall/bin/python \
  .qoder/skills/visualization-skill/scripts/visualize_latest.py
```

Pass an explicit run folder or checkpoint instead of a name:

```bash
... visualize_latest.py logs/adaptive/horse_rotate_..._extArea1
... visualize_latest.py logs/adaptive/horse_rotate_..._extArea1/checkpoint_8000.pt
```

## What the script does

1. Scans `<repo>/logs` recursively for folders containing `checkpoint*.pt`.
2. If a model substring is given, keeps folders whose path contains it; then picks
   the run whose newest checkpoint has the latest modification time.
3. Picks the latest checkpoint in that run: `checkpoint.pt` (final) competes by
   mtime with numbered `checkpoint_N.pt` files.
4. Reports whether training is still running (a newer checkpoint may appear later).
5. Runs `utils/patch_vis.py` with `--n_images 4 --subdivision_depth -1 --no_obj`
   (OBJ export disabled) and writes exports to `<out_dir>/<checkpoint_stem>/`.
6. Prints the combined PLY, SVD PLY, patch count, depth histogram, and SVD
   statistics. Full `patch_vis.py` output goes to
   `<out_dir>/patch_vis_<checkpoint_stem>.log`.

If an export for that checkpoint already exists and is newer than the checkpoint,
the script skips re-exporting. Use `--force` to redo it.

## Options

| Option | Meaning |
|---|---|
| `--logs-root DIR` | Root to search (default `<repo>/logs`) |
| `--out-dir DIR` | Export root (default: the run folder) |
| `--n-images N` | Checkerboard textures to load (default 4) |
| `--resolution N` | Per-patch UV resolution passed to `patch_vis.py` |
| `--subdivision-depth N` | `-1` = all UVs (default), `0` = active patches only |
| `--no-ply` | Skip vertex-colored PLY exports (smaller output) |
| `--device DEVICE` | Device passed to `patch_vis.py`; use `cpu` if training is occupying the GPU |
| `--force` | Re-export even when outputs are current |
| `--dry-run` | Show the selected run/checkpoint and command only |
| `--python PATH` | Interpreter with torch/trimesh (default: seawall env, then current) |

## Reporting results

- Give the user absolute paths to the combined PLY and SVD PLY, patch count, and
  the SVD summary line. Mention the depth histogram when useful.
- OBJ files are not exported; the combined PLY plus per-patch PLYs and textures
  are the deliverables.
- Dense runs produce large files (e.g., dragon final: 6.0M faces, combined
  PLY ~122 MB). Run the script in the background on such runs and report when
  it completes.
- If the run is still training, say so: only the current latest checkpoint was
  exported, and the final one can be exported later.
