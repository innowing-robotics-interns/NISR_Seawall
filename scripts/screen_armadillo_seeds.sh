#!/bin/bash
# Screen initialization seeds for armadillo.  Phase 6/7 collapses are visible by
# epoch 100-300 as Chamfer exploding to 6-15 while R locks to an empty-render
# constant, so 300 epochs is enough to reject a seed.  Objective is unchanged;
# only --seed varies.
#
# Results: one TSV line per seed with the epoch-300 R / CD / tangent values.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1

OUT_ROOT=logs/render_phase8_retry_nolap/screen
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
mkdir -p "$OUT_ROOT"

COMMON=(
  --adaptive --N 100000 --M_per_patch 64 --epochs 300 --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
  --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002
  --checkpoint_every 1000
)

: > "${OUT_ROOT}/results.tsv"
for seed in "$@"; do
  log="${OUT_ROOT}/seed_${seed}.log"
  echo "=== SEED $seed $(date '+%T')" | tee -a "${OUT_ROOT}/queue.log"
  "$PY" main.py "${COMMON[@]}" --seed "$seed" \
    --file normalized_shapes/armadillo_normal.ply \
    --result_dir "${OUT_ROOT}/seed_${seed}" > "$log" 2>&1
  rc=$?
  line=$(grep "Epoch   300/300" "$log" | tail -1)
  printf '%s\t%s\t%s\n' "$seed" "$rc" "$line" >> "${OUT_ROOT}/results.tsv"
done
echo "=== SCREEN DONE $(date '+%T')" | tee -a "${OUT_ROOT}/queue.log"
