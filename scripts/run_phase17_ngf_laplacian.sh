#!/bin/bash
# Phase 17: NGF Laplacian smoothing instead of SVD regularization.
#
# Everything matches phase 16 (logs/render_phase16_arapsi_cdgrid) and phase 15:
# GHPR init, 2-layer render loss with far weight 0.5, CD-driven subdivision at
# 0.03 on an 8x8 per-leaf grid, tangent energy mu=0.08 after a 1000-epoch delay,
# 6000 epochs, seed 41.  The single change is the regularizer:
#   phase 15/16: --lam_svd 0.08 with an SVD mode ramped in at epoch 3000
#   phase 17:    --lam_svd 0.0 and NGF's uniform-Laplacian term on the render
#                mesh at weight 1.0 (NGF's own training value)
# The Laplacian acts directly on the rendered mesh vertices, so unlike the
# scale-invariant SVD terms it also penalizes the local vertex spikes that a
# folded deep patch produces.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase17_ngf_laplacian
OUT="$OUT_ROOT/armadillo_layers2_ngf_laplacian"
LOG="$OUT_ROOT/armadillo_layers2_ngf_laplacian.log"
CHECK=.qoder/skills/render-check/scripts/render_check.py
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
EPOCHS="${EPOCHS:-6000}"
CD_THRESHOLD="${CD_THRESHOLD:-0.03}"
LAP_WEIGHT="${LAP_WEIGHT:-1.0}"
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs "$EPOCHS" --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight "$LAP_WEIGHT"
  --render_layers 2 --render_far_weight 0.5
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.0
  --gamma 0
  --subdiv_metric cd --subdiv_cd_threshold "$CD_THRESHOLD"
  --subdiv_cd_grid_res 8 --subdiv_cd_min_valid_frac 0.25
  --subdiv_threshold 0 --subdiv_every 500 --subdiv_start 1000
  --subdiv_max_depth 7 --max_splits_per_round 48
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every 1000
  --ghpr_init --seed 41
  --cd_sampling grid --cd_grid_res 8
)

echo "=== START $(date '+%F %T') lap=$LAP_WEIGHT lam_svd=0 cd_threshold=$CD_THRESHOLD epochs=$EPOCHS" >> "$OUT_ROOT/queue.log"
"$PY" main.py "${COMMON[@]}" \
  --file normalized_shapes/armadillo_normal.ply \
  --result_dir "$OUT" > "$LOG" 2>&1
rc=$?
echo "=== EXIT rc=$rc $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
[ "$rc" -ne 0 ] && exit "$rc"

(
  flock -w 7200 9 || exit 1
  "$PY" "$CHECK" --checkpoint "$OUT" \
    --out "$OUT_ROOT/check/armadillo_layers2_ngf_laplacian" \
    --resolution 256 --rows 4 --checker-views 4
) 9>"$OUT_ROOT/.check.lock" > "$OUT_ROOT/check_armadillo_layers2_ngf_laplacian.log" 2>&1
echo "=== CHECK rc=$? $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
