#!/bin/bash
# Diagnostic only: find which existing regularizer keeps armadillo's sheet in
# frame during the first 300 epochs.  Render objective is untouched; only the
# warmup schedules of terms already in the validated config change.
set -u
cd /home/lyangiw/codes/qoderhome/NISR_Seawall || exit 1
export PATH=/home/lyangiw/anaconda3/envs/seawall/bin:$PATH
export PYTHONUNBUFFERED=1
OUT_ROOT=logs/render_phase8_retry_nolap/probe
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
mkdir -p "$OUT_ROOT"
BASE=(
  --adaptive --N 100000 --M_per_patch 64 --epochs 300 --pretrain_epochs 500
  --lambda_render 1.0 --render_views 120 --render_batch 10
  --render_resolution 256 --render_min_coverage 0.05
  --render_laplacian_weight 0.0
  --gamma 0
  --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000 --subdiv_max_depth 7
  --ext_mask_epoch 1000 --ext_mask_every 500 --ext_mode no_fit --lambda_ext_area 0.0
  --box_init aabb --mesh_res 40 --lr 0.0002 --checkpoint_every 1000
  --mu 0.08 --lam_svd 0.08 --svd_mode conformal
  --file normalized_shapes/armadillo_normal.ply
)

run_probe() {
  local name=$1; shift
  "$PY" main.py "${BASE[@]}" --seed 41 --result_dir "${OUT_ROOT}/${name}" "$@" \
    > "${OUT_ROOT}/${name}.log" 2>&1
  printf '%s\t%s\n' "$name" "$(grep 'Epoch   300/300' "${OUT_ROOT}/${name}.log" | tail -1)" \
    >> "${OUT_ROOT}/results.tsv"
}

: > "${OUT_ROOT}/results.tsv"
run_probe A_mu_from0   --mu_warmup_delay 0 --mu_warmup_epochs 1000 --svd_warmup_delay 3000 --svd_warmup_epochs 500
run_probe B_mu_full    --mu_warmup_delay 0 --mu_warmup_epochs 0    --svd_warmup_delay 3000 --svd_warmup_epochs 500
run_probe C_svd_from0  --mu_warmup_delay 1000 --mu_warmup_epochs 1000 --svd_warmup_delay 0 --svd_warmup_epochs 0
run_probe D_both       --mu_warmup_delay 0 --mu_warmup_epochs 0    --svd_warmup_delay 0 --svd_warmup_epochs 0
echo done
