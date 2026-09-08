#!/bin/bash

# FILE="noisy_data_6.ply"
# INPUT_FILE="data/${FILE}"
# OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1_1"

# FILE="bimba_pc.ply"
# INPUT_FILE="3d_test_models/${FILE}"
# OUTPUT_DIR="logs/log_3dDirichletMSE/${FILE%.*}_boxInit_24Patches_M900_6Atlas_pretrain2k_MED_FreezeUV_dirichletCollapse0.08_10k"



python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint_2000.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 4 \
    --subdivision_depth "-1" \
    # --input_file ${INPUT_FILE} \