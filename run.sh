#!/bin/bash

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR"

# FILE="noisy_data_4_trimmed.ply"
# INPUT_FILE="data/${FILE}"
# OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1"

FILE="bimba.ply"
INPUT_FILE="3d_test_models/${FILE}"
OUTPUT_DIR="logs/log_3dModelsBoxInit/${FILE%.*}_boxInit_ARAP0.25_noPresplit_24Patches_M1500_6Atlas_pretrain2k"


# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --pretrain_epochs 2000 \
#     --epochs 5000 \
#     --n_patches 16 \
#     --d_features 128 \
#     --M_per_patch 4096 \
#     --W 512 \
#     --N 5000000 \
#     --mesh_res 200 \
#     --file ${INPUT_FILE} \
#     --D 3 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --gamma 0 \
#     --lambda_outer_boundary 1 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 200 \
#     --pretrain_loss l1 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --checkpoint_every  5000\

# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --shape sphere \
#     --pretrain_epochs 2000 \
#     --epochs 5000 \
#     --n_patches 4 \
#     --d_features 88 \
#     --M_per_patch 4096 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --lambda_outer_boundary 0 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 100 \
#     --pretrain_loss l1 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --checkpoint_every  5000\

# Initialize the model as a sphere 
# If using 4x4 atlas then 2 sheets (32 Patches), set M_per_patch to 1100
# If using 3x3 atlas then 2 sheets (18 Patches), set M_per_patch to 2000
# if using 2x2 atlas then 2 sheets (8 Patches), set M_per_patch to 4096
# python main.py \
#     --multi_patch \
#     --atlas_mode two_sheet \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --shape sphere \
#     --epochs 5000 \
#     --d_features 88 \
#     --M_per_patch 2000 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --lambda_outer_boundary 0 \
#     --log_every 100 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --N 5000 \
#     --checkpoint_every 5000 \
#     --two_sheet_side_rows 3 \
#     --two_sheet_side_cols 3 \
#     --two_sheet_split_axis 2 \
#     --two_sheet_side_axes 0 1 \
#     --pretrain_then_train \
#     --pretrain_epochs 2000 \
#     --pretrain_shape sphere \
#     --pretrain_mode closed_shape \
#     --no_presplit \
#     --correspondence_line_segment q_to_t \


# initialize the model as a box
# 2x2 patches per sheet, 6 sheets (24 patches), M = 1500
# 3x3 patches per sheet, 6 sheets (54 patches), M = 700
python main.py \
    --multi_patch \
    --atlas_mode six_sheet \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --epochs 5000 \
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
    --pretrain_epochs 2000 \
    --pretrain_shape box \
    --pretrain_mode closed_shape \
    --no_presplit \
    --correspondence_line_segment q_to_t \

python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 1 \
    --subdivision_depth "-1" \
    # --input_file ${INPUT_FILE} \