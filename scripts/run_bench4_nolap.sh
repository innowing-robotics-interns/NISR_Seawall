#!/bin/bash
# Benchmark set with the corrected objective validated on Lucy/120477:
# no Laplacian term, no extension-area loss, conformal SVD, 6000 epochs,
# M_per_patch=64, AABB box init.  Checkpoints are written every 1000 epochs
# only (plus the final checkpoint at stage end).
#
# Shapes: bunny, armadillo, horse, dancing_children100K (200k points each).
# Two runs at a time, each followed by a render check under the shared lock.
#
# Usage:
#   scripts/run_bench4_nolap.sh
#   EPOCHS=10000 bash scripts/run_bench4_nolap.sh
#   OUT_ROOT=logs/... bash scripts/run_bench4_nolap.sh
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-logs/render_phase6_nolap}"
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
  local name=$1 file=$2 out=$3
  local log="${OUT_ROOT}/${name}_adaptive_render.log"
  if [ -f "${out}/checkpoint.pt" ]; then
    echo "=== SKIP $name (checkpoint.pt already exists)" | tee -a "${OUT_ROOT}/queue.log"
    return 0
  fi
  echo "=== START $name $(date '+%F %T')  $file -> $out" | tee -a "${OUT_ROOT}/queue.log"
  "$PY" main.py "${COMMON[@]}" --file "$file" --result_dir "$out" > "$log" 2>&1
  local rc=$?
  echo "=== EXIT  $name rc=$rc $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
  if [ "$rc" -eq 0 ] && [ -f "${out}/checkpoint.pt" ]; then
    run_check "$name" "$out"
  fi
  return "$rc"
}

queue_a() {
  run_one bunny    normalized_shapes/stanford_bunny_normal.ply "${OUT_ROOT}/bunny_adaptive_render"
  run_one horse    normalized_shapes/horse_points.ply           "${OUT_ROOT}/horse_adaptive_render"
}

queue_b() {
  run_one armadillo normalized_shapes/armadillo_normal.ply      "${OUT_ROOT}/armadillo_adaptive_render"
  run_one dancing   normalized_shapes/dancing_children100K.ply  "${OUT_ROOT}/dancing_adaptive_render"
}

# Do not start on top of another training.
for i in $(seq 1 240); do
  pgrep -f "python.*main\.py" >/dev/null || break
  sleep 30
done

echo "=== BATCH START $(date '+%F %T')  out=$OUT_ROOT  epochs=$EPOCHS  ckpt_every=$CKPT_EVERY" \
  | tee -a "${OUT_ROOT}/queue.log"
queue_a &
qa=$!
queue_b &
qb=$!
wait "$qa"
wait "$qb"
echo "=== ALL FOUR DONE $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
