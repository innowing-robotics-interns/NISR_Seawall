#!/bin/bash
# Retry elephant and armadillo sequentially with the validated no-Lap objective.
# Same configuration as phase 7 (6000 epochs, no Laplacian, lambda_ext_area 0,
# conformal SVD, AABB box init, 120 views at 256^2).  New seeds and new output
# directory so the phase-6/7 collapsed runs stay untouched.
#
# Expected failure signature (phase 6/7): Chamfer explodes within the first 100
# epochs while R freezes at a repeating empty-render constant, then Tangent and
# SVD never recover the sheet.  If that repeats, the run is stopped early.
#
# Usage:
#   scripts/run_elephant_armadillo_retry_nolap.sh
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase8_retry_nolap
EPOCHS="${EPOCHS:-6000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
CHECK=.qoder/skills/render-check/scripts/render_check.py
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs "$EPOCHS" --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every "$CKPT_EVERY"
)

run_check() {
  local name=$1 out=$2
  local check_out="${OUT_ROOT}/check/${name}_adaptive_render"
  echo "=== CHECK $name $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
  (
    flock -w 7200 9 || exit 1
    "$PY" "$CHECK" --checkpoint "$out" --out "$check_out" \
      --resolution 256 --rows 4 --checker-views 4
  ) 9>"${OUT_ROOT}/.check.lock" > "${OUT_ROOT}/check_${name}.log" 2>&1
  echo "=== CHECK $name rc=$? $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
}

run_one() {
  local name=$1 file=$2 out=$3 seed=$4
  local log="${OUT_ROOT}/${name}_adaptive_render.log"
  if [ -f "${out}/checkpoint.pt" ]; then
    echo "=== SKIP $name (checkpoint.pt already exists)" | tee -a "${OUT_ROOT}/queue.log"
    return 0
  fi
  echo "=== START $name seed=$seed $(date '+%F %T')  $file -> $out" \
    | tee -a "${OUT_ROOT}/queue.log"
  "$PY" main.py "${COMMON[@]}" --seed "$seed" --file "$file" --result_dir "$out" \
    > "$log" 2>&1
  local rc=$?
  echo "=== EXIT  $name rc=$rc $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
  if [ "$rc" -eq 0 ] && [ -f "${out}/checkpoint.pt" ]; then
    run_check "$name" "$out"
  fi
  return "$rc"
}

for i in $(seq 1 240); do
  pgrep -f "[m]ain\.py" >/dev/null || break
  sleep 30
done

echo "=== RETRY START $(date '+%F %T')  out=$OUT_ROOT" | tee -a "${OUT_ROOT}/queue.log"
run_one elephant  normalized_shapes/elephant_normal.ply \
  "${OUT_ROOT}/elephant_adaptive_render" 31
run_one armadillo normalized_shapes/armadillo_normal.ply \
  "${OUT_ROOT}/armadillo_adaptive_render" 32
echo "=== RETRY STOP $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
