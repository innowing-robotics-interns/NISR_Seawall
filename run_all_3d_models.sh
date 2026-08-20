#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODELS_DIR="3d_test_models"
LOG_ROOT="logs/log_3dModelsMSE"

shopt -s nullglob
ply_files=("$MODELS_DIR"/*.ply)

if [ ${#ply_files[@]} -eq 0 ]; then
    echo "No .ply files found in $MODELS_DIR"
    exit 1
fi

for input_file in "${ply_files[@]}"; do
    file_name="$(basename "$input_file")"
    file_stem="${file_name%.*}"
    output_dir="$LOG_ROOT/${file_stem}_boxInit_ARAP0.25_noPresplit_24Patches_M1500_6Atlas_pretrain2k_MED_FreezeUV_10k"

    echo "============================================================"
    echo "Processing $file_name"
    echo "Output: $output_dir"
    echo "============================================================"

    python main.py \
        --multi_patch \
        --atlas_mode six_sheet \
        --file "$input_file" \
        --result_dir "$output_dir" \
        --epochs 10000 \
        --d_features 88 \
        --M_per_patch 1500 \
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
        --N 5000 \
        --checkpoint_every 5000 \
        --six_sheet_face_rows 2 \
        --six_sheet_face_cols 2 \
        --face_aware_box_supervision \
        --pretrain_then_train \
        --pretrain_epochs 1000 \
        --pretrain_shape box \
        --pretrain_mode closed_shape \
        --no_presplit \
        --correspondence_line_segment q_to_t \
        --corr_switch_epoch 7500

    python utils/patch_vis.py \
        --ckpt "$output_dir/checkpoint.pt" \
        --out_dir "$output_dir" \
        --n_images 1 \
        --subdivision_depth "-1"
done

echo "All .ply files processed."
