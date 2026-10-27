#!/bin/bash

# FILE="noisy_data_6.ply"
# INPUT_FILE="data/${FILE}"
# OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1_1"

# FILE="bimba_pc.ply"
# INPUT_FILE="3d_test_models/${FILE}"
# OUTPUT_DIR="logs/log_3dDirichletMSE/${FILE%.*}_boxInit_24Patches_M900_6Atlas_pretrain2k_MED_FreezeUV_dirichletCollapse0.08_10k"

OUTPUT_DIR="logs/adaptive/horse_rotate_CD_24Patches_M1100_6Atlas_5k_mu0.08_dirichlet_refP25k_ddfBeta0_ddfSigma0.05_lamDDF1_muDecay1_noCollapse"

python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint_4000.pt \
    --out_dir ${OUTPUT_DIR} \
    --no_obj \
    --n_images 4 \
    --resolution 32
    # --input_file ${INPUT_FILE} \