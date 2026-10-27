#!/bin/bash
# Phase 16: phase-15 CD-grid recipe with the scale-invariant ARAP regularizer.
#
# Everything matches logs/render_phase15_conformal_cdgrid/
# armadillo_layers2_conformal_cdgrid_rerun (GHPR init, 2-layer render loss with
# far weight 0.5, CD-driven subdivision at 0.03, SVD ramp at epoch 3000 with a
# 500-epoch warmup, 6000 epochs, seed 41) except the regularizer:
#   phase 15: --svd_mode conformal
#   phase 16: --svd_mode arap_si
# This isolates the effect of the SVD mode on the same atlas and targets.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase16_arapsi_cdgrid
OUT="$OUT_ROOT/armadillo_layers2_arapsi_cdgrid"
LOG="$OUT_ROOT/armadillo_layers2_arapsi_cdgrid.log"
CHECK=.qoder/skills/render-check/scripts/render_check.py
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
EPOCHS="${EPOCHS:-6000}"
CD_THRESHOLD="${CD_THRESHOLD:-0.03}"
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs "$EPOCHS" --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --render_layers 2 --render_far_weight 0.5
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode arap_si --svd_warmup_delay 3000 --svd_warmup_epochs 500
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

echo "=== START $(date '+%F %T') arap_si delay=3000 warmup=500 lam_svd=0.08 cd_threshold=$CD_THRESHOLD epochs=$EPOCHS" >> "$OUT_ROOT/queue.log"
"$PY" main.py "${COMMON[@]}" \
  --file normalized_shapes/armadillo_normal.ply \
  --result_dir "$OUT" > "$LOG" 2>&1
rc=$?
echo "=== EXIT rc=$rc $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
[ "$rc" -ne 0 ] && exit "$rc"

(
  flock -w 7200 9 || exit 1
  "$PY" "$CHECK" --checkpoint "$OUT" \
    --out "$OUT_ROOT/check/armadillo_layers2_arapsi_cdgrid" \
    --resolution 256 --rows 4 --checker-views 4
) 9>"$OUT_ROOT/.check.lock" > "$OUT_ROOT/check_armadillo_layers2_arapsi_cdgrid.log" 2>&1
echo "=== CHECK rc=$? $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
