#!/bin/bash
# Eight-hour exploration batch with the validated no-Lap objective:
# 6000 epochs, --render_laplacian_weight 0.0, --lambda_ext_area 0.0,
# conformal SVD, M_per_patch=64, AABB box init, checkpoints every 1000 epochs.
#
# Shapes run strictly one after another (single GPU slot).  A deadline guard
# stops launching new shapes when less than MIN_REMAIN seconds are left, so the
# batch never overruns the budget by more than one run.
#
# Shapes were chosen for variety (busts, animals, articulated thin parts) and
# exclude everything already trained in phases 4-6 (botijo, gargoyle, teapot,
# rocker, lucy, 120477, bunny, dancing, horse) and armadillo, which collapses
# reproducibly to zero rendered coverage.  Ordering interleaves shape types so
# that a deadline cut still leaves a diverse set.
#
# Usage:
#   scripts/run_more8h_nolap.sh
#   HOURS=4 bash scripts/run_more8h_nolap.sh
#   OUT_ROOT=logs/... bash scripts/run_more8h_nolap.sh
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-logs/render_phase7_nolap}"
HOURS="${HOURS:-8}"
EPOCHS="${EPOCHS:-6000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
MIN_REMAIN="${MIN_REMAIN:-2400}"
DEADLINE_TS="${DEADLINE_TS:-$(( $(date +%s) + HOURS * 3600 ))}"
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

can_start() {
  [ $(( DEADLINE_TS - $(date +%s) )) -ge "$MIN_REMAIN" ]
}

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
  if ! can_start; then
    echo "=== SKIP $name (deadline, $(( DEADLINE_TS - $(date +%s) ))s left)" \
      | tee -a "${OUT_ROOT}/queue.log"
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

queue_main() {
  run_one beethoven normalized_shapes/beethoven_normal.ply      "${OUT_ROOT}/beethoven_adaptive_render"
  run_one buste     normalized_shapes/buste_normal.ply          "${OUT_ROOT}/buste_adaptive_render"
  run_one elephant  normalized_shapes/elephant_normal.ply       "${OUT_ROOT}/elephant_adaptive_render"
  run_one isidore   normalized_shapes/isidore_horse_normal.ply  "${OUT_ROOT}/isidore_adaptive_render"
  run_one hand      normalized_shapes/laurent_hand_normal.ply   "${OUT_ROOT}/hand_adaptive_render"
  run_one maxplanck normalized_shapes/max-planck_points.ply     "${OUT_ROOT}/maxplanck_adaptive_render"
  run_one lion      normalized_shapes/chinese_lion_normal.ply   "${OUT_ROOT}/lion_adaptive_render"
  run_one gardet    normalized_shapes/georges_gardet_normal.ply "${OUT_ROOT}/gardet_adaptive_render"
  run_one goat      normalized_shapes/goat_normal.ply           "${OUT_ROOT}/goat_adaptive_render"
  run_one margaret  normalized_shapes/queen_margaret_normal.ply "${OUT_ROOT}/margaret_adaptive_render"
  run_one nefertiti normalized_shapes/nefertiti_points.ply      "${OUT_ROOT}/nefertiti_adaptive_render"
  run_one homer     normalized_shapes/homer_points.ply          "${OUT_ROOT}/homer_adaptive_render"
  run_one moai      normalized_shapes/moai.ply                  "${OUT_ROOT}/moai_adaptive_render"
  run_one neptune   normalized_shapes/neptune_normal.ply        "${OUT_ROOT}/neptune_adaptive_render"
}

# Do not start on top of another training.  The [m] keeps the pattern from
# matching the shell command that launched this script.
for i in $(seq 1 240); do
  pgrep -f "[m]ain\.py" >/dev/null || break
  sleep 30
done

echo "=== SEQ BATCH START $(date '+%F %T')  out=$OUT_ROOT  hours=$HOURS  deadline=$(date -d @$DEADLINE_TS '+%F %T')" \
  | tee -a "${OUT_ROOT}/queue.log"
queue_main
echo "=== SEQ BATCH STOP $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
