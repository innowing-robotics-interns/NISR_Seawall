#!/bin/bash

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR"

FILE="noisy_data_1_normals.xyz"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/log_init+svd/${FILE%.*}_init_PCA_mu0.01"

python main.py \
    --multi_patch \
    --pretrain_then_train \
    --result_dir ${OUTPUT_DIR} \
    --pretrain_epochs 2000 \
    --epochs 10000 \
    --n_patches 4 \
    --d_features 88 \
    --M_per_patch 8192 \
    --W 512 \
    --N 5000000 \
    --mesh_res 200 \
    --file ${INPUT_FILE} \
    --D 6 \
    --L 0 \
    --beta 100 \
    --mu 0.01 \
    --gamma 0.07 \
    --lam 0 \
    --lam2 0 \
    --log_every 200 \
    --pretrain_loss l1 \

# python main.py \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --epochs 10000 \
#     --N 8000000 \
#     --n_patches 4 \
#     --L 1 \
#     --d_features 88 \
#     --M_per_patch 4096 \
#     --M 4096 \
#     --mu 0.001 \
#     --gamma 0.07 \
#     --lam 0.4 \
#     --lam2 0.4 \
#     --beta 100 \
#     --mesh_res 200 \
#     --W 512


# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --shape flat_sheet \
#     --pretrain_epochs 2000 \
#     --epochs 10000 \
#     --n_patches 4 \
#     --d_features 88 \
#     --M_per_patch 4096 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.1 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 200 \
#     --pretrain_loss l1 \

python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 1 \