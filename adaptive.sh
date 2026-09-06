#!/bin/bash

FILE="stanford-bunny.ply"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_new_face_param"

python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.3 \
    --subdiv_max_depth 3 \
    --lam_svd 0.1 \
    --mu 0 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \

# REAL_OUTPUT_DIR="${OUTPUT_DIR}_1"

# python utils/patch_vis.py \
#     --ckpt ${REAL_OUTPUT_DIR}/checkpoint_500.pt \
#     --n_images 1 \
