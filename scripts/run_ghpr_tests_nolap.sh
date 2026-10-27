#!/bin/bash
# Phase 9: test --ghpr_init (inside-out start) against the validated render
# objective.  Two parallel queues, everything identical to phase 7 except the
# GHPR switch and the fixed seed.
#
# Matched pairs (same seed, GHPR off vs on):
#   hand    seed 41  base / ghpr
#   isidore seed 41  base / ghpr
#   elephant seed 31 ghpr   (baseline already exists in phase 8, seed 31)
#   armadillo seed 41 ghpr  (standard schedule; every seed collapsed before)
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase9_ghpr
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
  local check_out="${OUT_ROOT}/check/${name}"
  (
    flock -w 7200 9 || exit 1
    "$PY" "$CHECK" --checkpoint "$out" --out "$check_out" \
      --resolution 256 --rows 4 --checker-views 4
  ) 9>"${OUT_ROOT}/.check.lock" > "${OUT_ROOT}/check_${name}.log" 2>&1
  echo "=== CHECK $name rc=$? $(date '+%T')" >> "${OUT_ROOT}/queue_${QUEUE}.log"
}

run_one() {
  local name=$1 file=$2 seed=$3 ghpr=$4
  local out="${OUT_ROOT}/${name}"
  local log="${OUT_ROOT}/${name}.log"
  if [ -f "${out}/checkpoint.pt" ]; then
    echo "=== SKIP $name" >> "${OUT_ROOT}/queue_${QUEUE}.log"
    return 0
  fi
  local extra=()
  if [ "$ghpr" = "on" ]; then extra+=(--ghpr_init); fi
  echo "=== START $name seed=$seed ghpr=$ghpr $(date '+%T')" \
    >> "${OUT_ROOT}/queue_${QUEUE}.log"
  "$PY" main.py "${COMMON[@]}" "${extra[@]}" --seed "$seed" \
    --file "$file" --result_dir "$out" > "$log" 2>&1
  local rc=$?
  echo "=== EXIT  $name rc=$rc $(date '+%T')" >> "${OUT_ROOT}/queue_${QUEUE}.log"
  if [ "$rc" -eq 0 ] && [ -f "${out}/checkpoint.pt" ]; then
    run_check "$name" "$out"
  fi
}

queue_a() {
  QUEUE=A
  run_one hand_base     normalized_shapes/laurent_hand_normal.ply   41 off
  run_one isidore_base  normalized_shapes/isidore_horse_normal.ply  41 off
  run_one elephant_ghpr normalized_shapes/elephant_normal.ply       31 on
}

queue_b() {
  QUEUE=B
  run_one hand_ghpr     normalized_shapes/laurent_hand_normal.ply   41 on
  run_one isidore_ghpr  normalized_shapes/isidore_horse_normal.ply  41 on
  run_one armadillo_ghpr normalized_shapes/armadillo_normal.ply     41 on
}

for i in $(seq 1 240); do
  pgrep -f "[m]ain\.py" >/dev/null || break
  sleep 30
done

echo "=== PHASE9 START $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
queue_a &
qa=$!
queue_b &
qb=$!
wait "$qa" "$qb"
echo "=== PHASE9 STOP $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
