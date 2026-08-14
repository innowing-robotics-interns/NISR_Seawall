#!/bin/bash

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR"

FILE="noisy_data_2_normals.xyz"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/quadtree/${FILE%.*}"

# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --pretrain_epochs 100 \
#     --epochs 2000 \
#     --n_patches 16 \
#     --d_features 88 \
#     --M_per_patch 200 \
#     --W 512 \
#     --N 5000000 \
#     --mesh_res 200 \
#     --file ${INPUT_FILE} \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.0 \
#     --gamma 0 \
#     --lambda_outer_boundary 1 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 200 \
#     --pretrain_loss l1 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --checkpoint_every 1000 \
#     --quadtree_max_points 2000 \
#     --quadtree_max_depth 4

REAL_OUTPUT_DIR="${OUTPUT_DIR}_10"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --input_file ${INPUT_FILE} \
    --subdivision_depth "-1" \
    --quadtree \
    --no_save_each_patches \
    --model_path model/model.py

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
