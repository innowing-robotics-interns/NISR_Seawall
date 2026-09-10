#!/bin/bash
# adaptive.sh — quadtree cube-atlas run (main.py --adaptive)

FILE="bimba_pc.ply"
INPUT_FILE="3d_test_models/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_ddf_beta0"

# python main.py \
#     --adaptive \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --subdiv_threshold 0.1 \
#     --subdiv_max_depth 7 \
#     --lam_svd 0 \
#     --mu 0.08 \
#     --svd_mode arap \
#     --svd_warmup_delay 500 \
#     --svd_warmup_epochs 1000 \
#     --pretrain_epochs 500 \
#     --epochs 8000 \
#     --M_per_patch 128 \
#     --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint_5500.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 4 \
    --resolution 64