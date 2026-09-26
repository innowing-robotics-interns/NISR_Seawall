#!/bin/bash
# run.sh

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR"

# FILE="noisy_data_4_trimmed.ply"
# INPUT_FILE="data/${FILE}"
# OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1"

FILE="rocker-arm.ply"
INPUT_FILE="3d_test_models/${FILE}"
OUTPUT_DIR="logs/stitching/${FILE%.*}"


# Adaptive cube atlas + hole cutting/stitching (docs/documentation.md).
# Phase 1: box pretraining. Phase 2: --epochs of training. Then detect the
# crossing membranes, cut both openings, stitch a tube (genus +1), and train
# again for --hole_epochs with a fresh optimizer / LR / mu warmup.
python main.py \
    --adaptive \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --N 100000 \
    --pretrain_epochs 1000 \
    --epochs 1000 \
    --base_subdivisions 1 \
    --d_features 88 \
    --M_per_patch 128 \
    --W 512 \
    --D 6 \
    --L 0 \
    --beta 100 \
    --mu 0.08 \
    --mu_warmup_epochs 1000 \
    --mu_warmup_delay 300 \
    --schedule cosine \
    --gamma 0 \
    --log_every 100 \
    --checkpoint_every 1000 \
    --cut_hole \
    --hole_epochs 5000 \
    --hole_resolution 128 \
    --hole_tube_rows 4
    # --hole_openings 0,3      # force the opening pair if the auto choice is wrong
    # --hole_cut_mode crossing # cut along the crossing curve instead (needs a closed loop)

# checkpoint.pt = final model after the hole phase
# (checkpoint_before_hole.pt = genus-0 model, checkpoint_hole_<epoch>.pt = hole phase)
python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 4 \
    --subdivision_depth "-1" \
    # --no_unnormalize \
    # --input_file ${INPUT_FILE} \