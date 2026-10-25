#!/bin/bash
# Retry the gargoyle conformal run after the first attempt collapsed at
# epoch <100 (the sheet translated to y=-8 and left every camera before the
# conformal term was active at epoch 3000).  Same configuration as
# scripts/run_render4_conformal.sh, new output directory.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase4_conformal
OUT="$OUT_ROOT/gargoyle_retry_adaptive_render"
LOG="$OUT_ROOT/gargoyle_retry_adaptive_render.log"
CHECK=.qoder/skills/render-check/scripts/render_check.py
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python

# Wait until the rocker run in this batch has written its final checkpoint,
# so only two trainings occupy the GPU at any time.
for i in $(seq 1 720); do
  [ -f "$OUT_ROOT/rocker_adaptive_render/checkpoint.pt" ] && break
  sleep 30
done

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs 10000 --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 1.0
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 1.0
  --box_init aabb --mesh_res 40 --lr 0.0002
)

echo "=== RETRY START $(date '+%F %T')" >> "$OUT_ROOT/master.log"
"$PY" main.py "${COMMON[@]}" \
  --file normalized_shapes/gargoyle_normal.ply \
  --result_dir "$OUT" > "$LOG" 2>&1
rc=$?
echo "=== RETRY EXIT rc=$rc $(date '+%F %T')" >> "$OUT_ROOT/master.log"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

exec 9>"$OUT_ROOT/.check.lock"
flock -w 7200 9
"$PY" "$CHECK" --checkpoint "$OUT" \
  --out "$OUT_ROOT/check/gargoyle_retry_adaptive_render" \
  --resolution 256 --rows 4 --checker-views 4 > "$OUT_ROOT/check_gargoyle_retry.log" 2>&1
echo "=== RETRY CHECK rc=$? $(date '+%F %T')" >> "$OUT_ROOT/master.log"
