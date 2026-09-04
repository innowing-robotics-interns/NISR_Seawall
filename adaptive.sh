#!/bin/bash

FILE="stanford-bunny.ply"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}"

# python main_adaptive.py \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --subdiv_threshold 0.3 \
#     --subdiv_max_depth 5 \
#     --lam_svd 0.1 \
#     --svd_mode arap \
#     --svd_warmup_delay 500 \
#     --svd_warmup_epochs 1000 \
#     --pretrain_epochs 500 \
#     --epochs 1000 \

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
