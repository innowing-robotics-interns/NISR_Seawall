# NISR_Seawall

Neural implicit surface reconstruction from a 3D point cloud. The method learns a
multi-patch atlas: a set of coordinate-MLP "forward maps" from a 2D UV atlas into 3D,
trained to wrap onto the input points with Chamfer / directional-distance-field (DDF)
losses plus Dirichlet-style distortion regularization ("seawall"). Training can start
from a simple primitive (box or sphere) via a pretrain phase, and supports fixed
atlasses (single/two/six-sheet) or an adaptive quadtree cube atlas.

## Setup

```bash
conda env create -f environment.yaml   # creates the "NISR_Seawall" env
conda activate NISR_Seawall
```

The wrapper scripts assume an env named `seawall`; either rename the env, create it
with `conda env create -f environment.yaml -n seawall`, or point `SEAWALL_PYTHON` at
your interpreter.

Dependencies: Python 3.11, PyTorch (CUDA), numpy, scipy, matplotlib, trimesh,
scikit-learn, open3d, opencv-python. See [environment.yaml](environment.yaml).

The wrapper [run_seawall.sh](run_seawall.sh) runs `main.py` with the local
`seawall` conda interpreter. Point `SEAWALL_PYTHON` at another interpreter if needed:

```bash
./run_seawall.sh --help
```

## Training

All training goes through [main.py](main.py). Input is a `.ply` point cloud
(e.g. [3d_test_models/bunny_points.ply](3d_test_models/bunny_points.ply); synthetic
boxes/spheres are provided in the repo root for quick smoke tests).

### Quick start (six-sheet box atlas, recommended config)

This is the configuration in [run.sh](run.sh) and [run_all_3d_models.sh](run_all_3d_models.sh)
— a 6-face box atlas with 2x2 patches per face (24 patches), box-pretrained, with
Chamfer + DDF surface loss:

```bash
./run_seawall.sh \
    --multi_patch \
    --atlas_mode six_sheet \
    --file 3d_test_models/bunny_points.ply \
    --result_dir logs/bunny_run \
    --epochs 5000 \
    --d_features 88 --M_per_patch 1100 --W 512 --D 6 --L 0 \
    --beta 100 --mu 0.08 \
    --six_sheet_face_rows 2 --six_sheet_face_cols 2 \
    --face_aware_box_supervision \
    --pretrain_then_train --pretrain_epochs 1000 \
    --pretrain_shape box --pretrain_mode closed_shape \
    --no_presplit --correspondence_line_segment q_to_t \
    --surface_loss_type chamfer \
    --ddf_start_epoch 0 --ddf_sigma 0.05 --lambda_chamfer 1 --lambda_ddf 1 \
    --N 100000 --log_every 100 --checkpoint_every 1000 \
    --mu_warmup_epochs 1000 --mu_warmup_delay 300 --schedule cosine
```

Rule of thumb for `--M_per_patch` (samples per patch, ~24k total):
4x4 patches/sheet → 1100, 3x3 → 2000, 2x2 → 4096. For `six_sheet`, scale by
patches-per-face: 2x2 per face → 1100, 3x3 → ~700.

### Two-sheet sphere atlas

See the commented block in [run.sh](run.sh): `--atlas_mode two_sheet --shape sphere`
with `--two_sheet_side_rows/cols`, `--two_sheet_split_axis`, `--two_sheet_side_axes`.

### Adaptive quadtree atlas

[adaptive.sh](adaptive.sh) shows the `--adaptive` mode, which subdivides patches by
distortion during training:

```bash
./run_seawall.sh \
    --adaptive \
    --file 3d_test_models/bunny_points.ply \
    --result_dir logs/bunny_adaptive \
    --subdiv_threshold 0.1 --subdiv_max_depth 7 \
    --lam_svd 0 --mu 0.08 --svd_mode arap \
    --svd_warmup_delay 500 --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 --epochs 5000 \
    --M_per_patch 128 --L 0 --save_vertex_pos_every 10
```

### Batch over many models

[run_all_3d_models.sh](run_all_3d_models.sh) trains on every `.ply` in
`3d_test_models/` and exports visualization after each run.

### Useful flags

| Flag | Meaning |
|---|---|
| `--result_dir` | where checkpoints, meshes, and plots are written |
| `--N` | number of input point samples per epoch |
| `--M_per_patch` | UV samples per patch |
| `--epochs` / `--pretrain_epochs` | training / pretrain length |
| `--checkpoint_every` | save `checkpoint_<epoch>.pt` |
| `--init_ckpt` | resume/initialize from a checkpoint |
| `--device` | defaults to CUDA if available |
| `--surface_loss_type` | `chamfer`, `ddf`, or `cd_ddf` |
| `--mu` / `--tangent_mode` | distortion regularization weight / mode (`dirichlet`) |
| `--save_correspondence_every` | dump correspondence line debug output |
| `--tensorboard` | log scalars, figures, and the final mesh for TensorBoard |

Full list: `./run_seawall.sh --help`.

## Outputs

Everything lands in `--result_dir`:

- `checkpoint_<epoch>.pt`, `checkpoint.pt` (latest), `pretrain_checkpoint.pt` — full training state
- `learned_sheet.ply` / `.obj` (plus `_normalized` variants in [-1,1]^3) — reconstructed surface mesh
- `history.png` — loss curves; `patch_config_pretrain.png` / `patch_config_final.png` — atlas layout
- `patch_assignments.png`, `boundary_debug/`, `ddf_debug/`, `correspondence_lines*.ply` — periodic debug dumps
- `metadata.json` — final loss values and config
- `tensorboard/` — TensorBoard event files when `--tensorboard` is set
- adaptive mode additionally: `vertex_positions_{normalized,denormalized}.json`, `patch_config_<epoch>.png`

## TensorBoard

Pass `--tensorboard` to log training scalars (loss terms, effective weights, LR,
leaf count), pretrain scalars, final figures (`history.png`, patch layout,
`result.png`), and the learned mesh as a 3D mesh for the Mesh view. Logs are
written to `<result_dir>/tensorboard`:

```bash
./run_seawall.sh --tensorboard [other training flags...]
tensorboard --logdir logs/bunny_run/tensorboard
```

## Visualization

### Textured mesh export (main tool)

[utils/patch_vis.py](utils/patch_vis.py) loads a checkpoint and exports the learned
patches as textured OBJ/PLY files (checkerboard UV texture from
[utils/texture/](utils/texture)):

```bash
python utils/patch_vis.py \
    --ckpt logs/bunny_run/checkpoint_5000.pt \
    --out_dir logs/bunny_run \
    --n_images 4 \
    --subdivision_depth -1
```

- `--ckpt` accepts a checkpoint file or a directory containing `checkpoint*.pt`
  (latest is picked).
- `--subdivision_depth -1` disables UV occupancy filtering; `--input_file <ply>`
  enables point-based masking of active UV cells.
- `--resolution` sets the per-patch UV grid (default 100; adaptive runs use 64).
- `--adaptive yes/no/auto` tells it which atlas type the checkpoint holds.
- Outputs go to `--out_dir` (`checkerboard_export/` by default): `.obj` + `.mtl` +
  texture images, plus a vertex-colored `.ply` (disable with `--no_ply`).

### Viewing results

- Meshes: open `learned_sheet.ply` or the `patch_vis.py` exports in
  [MeshLab](https://www.meshlab.net/) or Open3D.
- Adaptive vertex trajectories:
  `python utils/plot_vertex_trajectory.py --json logs/run/vertex_positions_denormalized.json --vertices v2 --output_dir logs/run`
- Correspondence and GHPR utilities live in [utils/correspondence_vis.py](utils/correspondence_vis.py)
  and [utils/ghpr.py](utils/ghpr.py).

## Repo layout

- [main.py](main.py) — training entry point (atlas construction, losses, pretrain, subdivision)
- [model/](model/) — `model.py` (forward/inverse maps), `losses.py`, `correspondence.py`, `subdivision.py`
- [utils/](utils/) — visualization and helper scripts (`patch_vis.py`, `adaptive_vis.py`, `ghpr.py`, ...)
- [run.sh](run.sh) / [adaptive.sh](adaptive.sh) — annotated example configs
