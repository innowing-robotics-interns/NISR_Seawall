#!/bin/bash
# Armadillo rescue run.  The validated schedule (SVD delayed to epoch 3000,
# tangent delayed to 1000) lets this shape's sheet fly out of frame within the
# first 100 epochs under every seed tried (unseeded, 7, 32, 41-48).  Enabling
# the existing conformal SVD term from epoch 0 holds it: CD 0.067 at epoch 1500
# versus 4-18 for the standard schedule.  No new loss terms, no weight changes;
# only the SVD warmup schedule differs.  Everything else matches phase 7.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase8_retry_nolap
OUT="${OUT_ROOT}/armadillo_svdfast_adaptive_render"
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
CHECK=.qoder/skills/render-check/scripts/render_check.py

"$PY" main.py \
  --adaptive --N 100000 --M_per_patch 64 --epochs 6000 --pretrain_epochs 500 \
  --lambda_render 1.0 --render_views 120 --render_batch 10 \
  --render_resolution 256 --render_min_coverage 0.05 --render_laplacian_weight 0.0 \
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000 \
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 0 --svd_warmup_epochs 0 \
  --gamma 0 \
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7 \
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0 \
  --box_init aabb --mesh_res 40 --lr 0.0002 --checkpoint_every 1000 --seed 41 \
  --file normalized_shapes/armadillo_normal.ply --result_dir "$OUT" \
  > "${OUT_ROOT}/armadillo_svdfast_adaptive_render.log" 2>&1
rc=$?
echo "=== TRAIN rc=$rc $(date '+%T')" >> "${OUT_ROOT}/queue.log"
if [ "$rc" -eq 0 ] && [ -f "${OUT}/checkpoint.pt" ]; then
  (
    flock -w 7200 9 || exit 1
    "$PY" "$CHECK" --checkpoint "$OUT" \
      --out "${OUT_ROOT}/check/armadillo_svdfast_adaptive_render" \
      --resolution 256 --rows 4 --checker-views 4
  ) 9>"${OUT_ROOT}/.check.lock" > "${OUT_ROOT}/check_armadillo_svdfast.log" 2>&1
  echo "=== CHECK rc=$? $(date '+%T')" >> "${OUT_ROOT}/queue.log"
fi
