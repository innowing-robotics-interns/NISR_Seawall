#!/bin/bash
# Four adaptive render-loss runs with no render-mesh Laplacian and no
# extension area-minimizing loss.
#
# Baseline for the render term is NGF's six-channel [normal, position] L1.
# The earlier launchers also set --render_laplacian_weight 1.0, which adds a
# uniform Laplacian on the sampled patch mesh to the optimized loss (logged as
# "Lap").  This launcher sets it to 0.0, so the render branch is pure L1.
#
# Everything else matches scripts/run_render4_noextarea.sh: conformal SVD term,
# extension mask with --ext_mode no_fit, and --lambda_ext_area 0.0.  Outputs go
# to a separate root so the completed Lap-on batch is preserved.
#
# Usage:
#   scripts/run_render4_noextarea_nolap.sh
#   OUT_ROOT=logs/my_comparison bash scripts/run_render4_noextarea_nolap.sh
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-logs/render_phase4_noextarea_nolap}"
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
CHECK=.qoder/skills/render-check/scripts/render_check.py
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs 10000 --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
)

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

queue_a() {
  run_one botijo   normalized_shapes/botijo_normal.ply    "${OUT_ROOT}/botijo_adaptive_render"
  run_one teapot   normalized_shapes/teapot_points.ply    "${OUT_ROOT}/teapot_adaptive_render"
}

queue_b() {
  run_one gargoyle normalized_shapes/gargoyle_normal.ply  "${OUT_ROOT}/gargoyle_adaptive_render"
  run_one rocker   normalized_shapes/rocker-arm_points.ply "${OUT_ROOT}/rocker_adaptive_render"
}

if pgrep -af "python.*main\.py" | grep -v pgrep >/dev/null; then
  echo "another main.py is already running; stop it first" >&2
  exit 1
fi

echo "=== BATCH START $(date '+%F %T')  out=$OUT_ROOT  svd=conformal  ext_area=0  render_lap=0" \
  | tee -a "${OUT_ROOT}/queue.log"
queue_a &
qa=$!
queue_b &
qb=$!
wait "$qa"
wait "$qb"
echo "=== ALL FOUR DONE $(date '+%F %T')" | tee -a "${OUT_ROOT}/queue.log"
