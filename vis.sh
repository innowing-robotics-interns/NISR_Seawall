#!/bin/bash

FILE="noisy_data_6.ply"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1_1"

python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint_5000.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 1 \
    --subdivision_depth "-1" \
    --input_file ${INPUT_FILE} \