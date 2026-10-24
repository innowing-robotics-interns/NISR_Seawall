---
name: train-shape
description: Launch and monitor NISR_Seawall point-cloud training through run.sh. Use when asked to train a shape or point cloud (bunny, lucy, 120477, dancing_children100K, custom .ply), start a new adaptive cube-atlas run, change the training shape or epoch count, resume from a checkpoint, or check on and stop a running training job.
---

# Train Shape

## Overview

Trains one point cloud with the adaptive 24-patch cube-atlas configuration in
`run.sh`. `run.sh` now reads environment overrides for shape, output directory,
epochs, checkpoint cadence, and resume checkpoint. The bundled launcher resolves
shape names, guards against duplicate runs, and starts `run.sh` detached with a
log and PID file.

## Quick start

From the repository root:

```bash
.qoder/skills/train-shape/scripts/train_shape.sh --shape bunny_points
```

This resolves `normalized_shapes/bunny_points.ply`, writes to
`logs/adaptive/bunny_points_train`, and returns immediately with a PID and log
path. Follow it with:

```bash
tail -n 40 logs/bunny_points_train_*.log
```

Use `--dry-run` first to inspect the resolved shape, result directory, and
command.

## Default configuration

The active block in `run.sh` is the adaptive quadtree cube atlas used for
`dancing_children100K`: 24 root patches, `M_per_patch 64`, 10,000 epochs,
checkpoint every 500 epochs, Dirichlet tangents, SVD collapse regularization,
extension masking from epoch 1000, and TensorBoard. Changing only `--shape`
keeps these hyperparameters.

## Shape resolution

`--shape` accepts a path or a bare name. Bare names are tried as `<name>.ply`
under `normalized_shapes/`, `3d_test_models/`, `data/`, then the repository root.
Prefer `normalized_shapes/` for comparable runs. Files elsewhere are normalized
per-file by `main.py` (`center = mean(points)`, `scale = max |p - center|`).
Use `--input-file PATH` to bypass resolution.

## Options

| Option | Meaning |
|---|---|
| `--shape NAME\|PATH` | Shape to train (default: keep `run.sh`'s `FILE`) |
| `--input-file PATH` | Explicit input PLY, overrides `--shape` |
| `--result-dir DIR` | Base output dir (default `logs/adaptive/<stem>_train`) |
| `--epochs N` | Override `--epochs` from `run.sh` |
| `--checkpoint-every N` | Override checkpoint cadence |
| `--init-ckpt PATH` | Initialize/resume from a checkpoint |
| `--reuse-normalization` | Add `--init_ckpt_reuse_normalization` |
| `--patch-vis` | Run `run.sh`'s optional post-training export |
| `--foreground` | Run attached instead of detached |
| `--force` | Launch even if another `main.py` appears to be running |
| `--log PATH` | Log file (default `logs/<stem>_train_<timestamp>.log`) |
| `--dry-run` | Print resolved settings and command only |

The launcher prepends the seawall interpreter to `PATH` (override with
`SEAWALL_PYTHON`) and sets `PYTHONUNBUFFERED=1` so the log updates live.

## Monitoring

1. Find the actual run directory (a `_1`, `_2` suffix is appended when the base
   directory already exists):

   ```bash
   log=logs/<stem>_train_*.log
   run_dir=$(grep -m1 "Output directory:" $log | sed 's/.*Output directory: *//')
   ```

2. Check progress and hardware:

   ```bash
   tail -n 40 "$log"
   ls -lh "$run_dir"/checkpoint_*.pt
   nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
   ```

3. `metadata.json` is written at startup with `"status": "running"` and updated
   to `"completed"` at the end. Checkpoints appear every
   `--checkpoint-every` epochs; `checkpoint.pt` is the final one.

4. TensorBoard (the config enables it):

   ```bash
   tensorboard --logdir "$run_dir/tensorboard" --port 6006
   ```

## Stopping a run

For detached runs, the launcher stores the process-group leader PID in
`<log>.pid`:

```bash
pid=$(cat "$log.pid")
kill -TERM -- -"$pid"
sleep 3
pgrep -af "python.*main\.py" || echo "stopped"
```

First confirm the PID belongs to this run (`ps -o pid,pgid,cmd -p "$pid"`).
If the process group survives 10 seconds, `kill -KILL -- -"$pid"`.

## Resuming and hyperparameters

Resume or initialize from a checkpoint without editing `run.sh`:

```bash
.qoder/skills/train-shape/scripts/train_shape.sh \
  --shape dancing_children100K \
  --init-ckpt logs/adaptive/<run>/checkpoint_5000.pt \
  --result-dir logs/adaptive/dancing_children100K_resume \
  --reuse-normalization
```

`--epochs` counts epochs for the new run; the optimizer/schedules restart.
Use `--reuse-normalization` only when the input should stay in the previous
run's coordinate frame (same frame/sequence).

For other hyperparameter changes, edit the active `python main.py` block in
`run.sh`. Keep backslash continuations intact: never put a comment after a
trailing `\`, because it swallows the remaining arguments (this previously
disabled `--tensorboard`).

Supported `run.sh` environment overrides: `FILE`, `INPUT_FILE`, `OUTPUT_DIR`,
`EPOCHS`, `CHECKPOINT_EVERY`, `INIT_CKPT`,
`INIT_CKPT_REUSE_NORMALIZATION`, `PATCH_VIS`.

## Batch training

Loop with `--foreground` so runs execute sequentially, and launch the whole loop
as one background task:

```bash
for shape in bunny_points lucy_points 120477; do
  .qoder/skills/train-shape/scripts/train_shape.sh --shape "$shape" --foreground
done
```

Each shape gets its own log at `logs/<stem>_train_<timestamp>.log`.

## Common issues

- **Duplicate runs:** the launcher refuses to start when `python ... main.py` is
  already running. Verify with `pgrep -af "python.*main\.py"` and stop the old
  run, or pass `--force`.
- **CUDA OOM:** free the GPU or lower `M_per_patch` / `N` in `run.sh`.
- **Result directory suffix:** `_get_unique_folder` appends `_1`, `_2`, so always
  read the `Output directory:` line instead of assuming `--result-dir`.
- **No TensorBoard data:** confirm the launch log shows the flag was parsed and
  `<run_dir>/tensorboard/events.out.tfevents.*` exists; check `metadata.json`'s
  stored args for `"tensorboard": true`.
- **No visualization after training:** `PATCH_VIS` defaults to `0`. Export with
  the visualization skill instead, or pass `--patch-vis`.

## Reporting

After launch, report the shape, actual run directory (once known), log path, PID,
epochs/checkpoint cadence, and GPU status. On completion, report
`checkpoint.pt`, `learned_sheet.ply`, and `metadata.json`, then offer the
visualization skill for a checkerboard patch export.
