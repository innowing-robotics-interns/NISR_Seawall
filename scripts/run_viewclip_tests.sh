#!/bin/bash
# Phase 11: robust per-view clipping of the NGF render loss.
# Elephant is the worst healthy shape in the foldover audit (4.5% at delta=0.004
# in phase 10).  Everything matches phase 10 (GHPR init, lam_svd 0) except
# --render_view_clip: each view's mean error is capped at factor x the batch
# median view error before averaging.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase11_viewclip
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

run_one() {
  local name=$1 factor=$2 seed=$3
  local out="${OUT_ROOT}/${name}"
  local log="${OUT_ROOT}/${name}.log"
  if [ -f "${out}/checkpoint.pt" ]; then
    echo "=== SKIP $name" >> "${OUT_ROOT}/queue.log"
    return 0
  fi
  echo "=== START $name clip=$factor seed=$seed $(date '+%T')" >> "${OUT_ROOT}/queue.log"
  "$PY" main.py "${COMMON[@]}" --render_view_clip "$factor" --seed "$seed" \
    --file normalized_shapes/elephant_normal.ply --result_dir "$out" > "$log" 2>&1
  local rc=$?
  echo "=== EXIT  $name rc=$rc $(date '+%T')" >> "${OUT_ROOT}/queue.log"
  if [ "$rc" -eq 0 ] && [ -f "${out}/checkpoint.pt" ]; then
    (
      flock -w 7200 9 || exit 1
      "$PY" "$CHECK" --checkpoint "$out" --out "${OUT_ROOT}/check/${name}" \
        --resolution 256 --rows 4 --checker-views 4
    ) 9>"${OUT_ROOT}/.check.lock" > "${OUT_ROOT}/check_${name}.log" 2>&1
    echo "=== CHECK $name rc=$? $(date '+%T')" >> "${OUT_ROOT}/queue.log"
  fi
}

for spec in "elephant_clip2 2.0 31" "elephant_clip3 3.0 31" "elephant_clip5 5.0 31"; do
  set -- $spec
  run_one "$1" "$2" "$3" &
done
wait
echo "=== PHASE11 STOP $(date '+%F %T')" >> "${OUT_ROOT}/queue.log"
