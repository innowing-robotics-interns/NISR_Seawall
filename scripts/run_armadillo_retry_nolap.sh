#!/bin/bash
# Retry armadillo after the first phase-6 attempt collapsed in phase 2:
# prediction coverage fell to 0.0005 (target 0.185), the render loss froze,
# and Tangent decayed to zero, so the sheet was effectively degenerate.
# Same corrected objective as scripts/run_bench4_nolap.sh, with a fixed seed
# so this attempt is reproducible.  Waits for the dancing run (queue B) to
# free the second GPU slot.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase6_nolap
OUT="$OUT_ROOT/armadillo_retry_adaptive_render"
LOG="$OUT_ROOT/armadillo_retry_adaptive_render.log"
CHECK=.qoder/skills/render-check/scripts/render_check.py
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python

for i in $(seq 1 480); do
  [ -f "$OUT_ROOT/dancing_adaptive_render/checkpoint.pt" ] && break
  sleep 30
done

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs 6000 --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every 1000 --seed 7
)

echo "=== ARMADILLO RETRY START $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
"$PY" main.py "${COMMON[@]}" \
  --file normalized_shapes/armadillo_normal.ply \
  --result_dir "$OUT" > "$LOG" 2>&1
rc=$?
echo "=== ARMADILLO RETRY EXIT rc=$rc $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
if [ "$rc" -ne 0 ]; then exit "$rc"; fi

(
  flock -w 7200 9 || exit 1
  "$PY" "$CHECK" --checkpoint "$OUT" \
    --out "$OUT_ROOT/check/armadillo_retry_adaptive_render" \
    --resolution 256 --rows 4 --checker-views 4
) 9>"$OUT_ROOT/.check.lock" > "$OUT_ROOT/check_armadillo_retry.log" 2>&1
echo "=== ARMADILLO RETRY CHECK rc=$? $(date '+%F %T')" >> "$OUT_ROOT/queue.log"
