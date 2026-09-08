#!/bin/bash

FILE="stanford-bunny.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet_svd"

# python main_adaptive.py \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --subdiv_threshold 0.1 \
#     --subdiv_max_depth 7 \
#     --lam_svd 0.001 \
#     --mu 0.08 \
#     --svd_mode arap \
#     --svd_warmup_delay 500 \
#     --svd_warmup_epochs 1000 \
#     --pretrain_epochs 500 \
#     --epochs 5000 \
#     --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}_1"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint_4000.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 128 \