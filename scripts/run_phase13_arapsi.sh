#!/bin/bash
# Phase 13: phase-12 two-layer far-point supervision plus the scale-invariant
# ARAP Jacobian regularizer (--svd_mode arap_si), active from epoch 1.
#
# Everything else matches logs/render_phase12_layers2/armadillo_layers2_w05:
# GHPR init, no Laplacian, no extension-area penalty, 6000 epochs, seed 41.
# The difference is --lam_svd 0.08 --svd_mode arap_si --svd_warmup_delay 0
# (the phase-12 run had SVD off).  Reference: phase-10 no-SVD baseline
# logs/render_phase10_ghpr_nosvd/armadillo_ghpr_nosvd.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase13_arapsi
OUT="$OUT_ROOT/armadillo_layers2_arapsi"
LOG="$OUT_ROOT/armadillo_layers2_arapsi.log"
CHECK=.qoder/skills/render-check/scripts/render_check.py
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
EPOCHS="${EPOCHS:-6000}"
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs "$EPOCHS" --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --render_layers 2 --render_far_weight 0.5
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode arap_si --svd_warmup_delay 0 --svd_warmup_epochs 0
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every 1000
  --ghpr_init --seed 41
)

echo "=== START $(date '+%F %T') arap_si lam_svd=0.08 epochs=$EPOCHS" >> "$OUT_ROOT/queue.log"
"$PY" main.py "${COMMON[@]}" \
  --file normalized_shapes/armadillo_normal.ply \
  --result_dir "$OUT" > "$LOG" 2>&1
rc=$?
echo "=== EXIT rc=$rc $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
[ "$rc" -ne 0 ] && exit "$rc"

(
  flock -w 7200 9 || exit 1
  "$PY" "$CHECK" --checkpoint "$OUT" \
    --out "$OUT_ROOT/check/armadillo_layers2_arapsi" \
    --resolution 256 --rows 4 --checker-views 4
) 9>"$OUT_ROOT/.check.lock" > "$OUT_ROOT/check_armadillo_layers2_arapsi.log" 2>&1
echo "=== CHECK rc=$? $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
