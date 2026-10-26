#!/bin/bash
# Phase 10: GHPR inside-out start with the conformal SVD term removed
# (--lam_svd 0.0).  Everything else matches phase 9, including seeds, so the
# only difference against 2026-10-26 phase-9 GHPR runs is the SVD term.
#
# Question: was armadillo/isidore rescued by the inside-out init alone, or does
# it still need the conformal regularizer?
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase10_ghpr_nosvd
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
  --lam_svd 0.0 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every "$CKPT_EVERY"
  --ghpr_init
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
  local name=$1 file=$2 seed=$3
  local out="${OUT_ROOT}/${name}"
  local log="${OUT_ROOT}/${name}.log"
  if [ -f "${out}/checkpoint.pt" ]; then
    echo "=== SKIP $name" >> "${OUT_ROOT}/queue_${QUEUE}.log"
    return 0
  fi
  echo "=== START $name seed=$seed $(date '+%T')" >> "${OUT_ROOT}/queue_${QUEUE}.log"
  "$PY" main.py "${COMMON[@]}" --seed "$seed" \
    --file "$file" --result_dir "$out" > "$log" 2>&1
  local rc=$?
  echo "=== EXIT  $name rc=$rc $(date '+%T')" >> "${OUT_ROOT}/queue_${QUEUE}.log"
  if [ "$rc" -eq 0 ] && [ -f "${out}/checkpoint.pt" ]; then
    run_check "$name" "$out"
  fi
}

queue_a() {
  QUEUE=A
  run_one armadillo_ghpr_nosvd normalized_shapes/armadillo_normal.ply 41
  run_one hand_ghpr_nosvd      normalized_shapes/laurent_hand_normal.ply 41
}

queue_b() {
  QUEUE=B
  run_one isidore_ghpr_nosvd  normalized_shapes/isidore_horse_normal.ply 41
  run_one elephant_ghpr_nosvd normalized_shapes/elephant_normal.ply 31
}

for i in $(seq 1 240); do
  pgrep -f "[m]ain\.py" >/dev/null || break
  sleep 30
done

echo "=== PHASE10 START $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
queue_a &
qa=$!
queue_b &
qb=$!
wait "$qa" "$qb"
echo "=== PHASE10 STOP $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
