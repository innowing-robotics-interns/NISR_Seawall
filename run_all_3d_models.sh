#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="3d_test_models"
LOG_ROOT="logs/log_3dTestNoCollapse"

shopt -s nullglob
ply_files=("$MODELS_DIR"/*.ply)

if [ ${#ply_files[@]} -eq 0 ]; then
    echo "No .ply files found in $MODELS_DIR"
    exit 1
fi

for input_file in "${ply_files[@]}"; do
    file_name="$(basename "$input_file")"
    file_stem="${file_name%.*}"
    output_dir="$LOG_ROOT/${file_stem}_CD_24Patches_M1100_6Atlas_5k_mu0.08_dirichlet_refP25k_ddfBeta0_ddfSigma0.05_lamDDF1_muDecay1_noCollapse"

    echo "============================================================"
    echo "Processing $file_name"
    echo "Output: $output_dir"
    echo "============================================================"
    
    python main.py \
        --multi_patch \
        --atlas_mode six_sheet \
        --file ${input_file} \
        --result_dir ${output_dir} \
        --epochs 5000 \
        --d_features 88 \
        --M_per_patch 1100 \
        --W 512 \
        --D 6 \
        --L 0 \
        --beta 100 \
        --mu 0.08 \
        --gamma 0 \
        --lam 0 \
        --lam2 0 \
        --lambda_outer_boundary 0 \
        --log_every 100 \
        --mu_warmup_epochs 1000 \
        --mu_warmup_delay 300 \
        --schedule cosine \
        --N 100000 \
        --checkpoint_every 1000 \
        --six_sheet_face_rows 2 \
        --six_sheet_face_cols 2 \
        --face_aware_box_supervision \
        --pretrain_then_train \
        --pretrain_epochs 1000 \
        --pretrain_shape box \
        --pretrain_mode closed_shape \
        --no_presplit \
        --correspondence_line_segment q_to_t \
        --corr_switch_epoch 11000 \
        --save_boundary_debug 0 \
        --save_correspondence_every 5000 \
        --surface_loss_type chamfer \
        --ddf_start_epoch 0 \
        --ddf_sigma 0.05 \
        --ddf_mu_decay 1 \
        --chamfer_resume_epoch 0 \
        --lambda_chamfer 1 \
        --lambda_ddf 1 \


    python utils/patch_vis.py \
        --ckpt "$output_dir/checkpoint_5000.pt" \
        --out_dir "$output_dir" \
        --n_images 4 \
        --subdivision_depth "-1"
done

echo "All .ply files processed."
