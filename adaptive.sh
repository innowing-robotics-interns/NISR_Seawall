#!/bin/bash
# adaptive.sh — quadtree cube-atlas run (main.py --adaptive)

FILE="laurent_hand_normal.ply"
INPUT_FILE="3d_input_models/${FILE}"
OUTPUT_DIR="logs/3dModels_normals/${FILE%.*}_mu0.08_dirichlet_conformal0.08_8k_confDelay6.5k_gamma0.08_gammaDelay4k"
# OUTPUT_DIR="logs/symmetric_dirichlet/${FILE%.*}_symDir_depth7"

python main.py \
    --adaptive \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.08 \
    --mu 0.08 \
    --svd_mode conformal \
    --svd_warmup_delay 6500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 8000 \
    --M_per_patch 128 \
    --L 0 \
    --sym_dirichlet_epoch 0 \
    --log_every 100 \
    --checkpoint_every 500 \
    --tangent_mode dirichlet \
    --gamma 0.08 \
    --gamma_warmup_delay 4000 \
    --gamma_warmup_epochs 1000 \

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint_4000.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 4 \
    --resolution 64 \