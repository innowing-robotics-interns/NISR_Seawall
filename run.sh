#!/bin/bash
# run.sh

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# cd "$SCRIPT_DIR"

# FILE="noisy_data_4_trimmed.ply"
# INPUT_FILE="data/${FILE}"
# OUTPUT_DIR="logs/log_bestConfig/${FILE%.*}_D3_W512_M4096_d128_16Patches_bound1"

# FILE="max-planck.ply"
# FILE="bunny_points.ply"
# FILE="xyzrgb_dragon_points.ply"
# FILE="lucy_points.ply"
# FILE="120477.ply"
FILE="${FILE:-dancing_children100K.ply}"
INPUT_FILE="${INPUT_FILE:-normalized_shapes/${FILE}}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/adaptive/${FILE%.*}_CD_noCollapse_extArea1_noFit_allUnpaired}"


# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --pretrain_epochs 2000 \
#     --epochs 5000 \
#     --n_patches 16 \
#     --d_features 128 \
#     --M_per_patch 4096 \
#     --W 512 \
#     --N 5000000 \
#     --mesh_res 200 \
#     --file ${INPUT_FILE} \
#     --D 3 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --gamma 0 \
#     --lambda_outer_boundary 1 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 200 \
#     --pretrain_loss l1 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --checkpoint_every  5000

# python main.py \
#     --multi_patch \
#     --pretrain_then_train \
#     --result_dir ${OUTPUT_DIR} \
#     --shape sphere \
#     --pretrain_epochs 2000 \
#     --epochs 5000 \
#     --n_patches 4 \
#     --d_features 88 \
#     --M_per_patch 4096 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --lambda_outer_boundary 0 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --log_every 100 \
#     --pretrain_loss l1 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --checkpoint_every  5000\

# Initialize the model as a sphere 
# If using 4x4 atlas then 2 sheets (32 Patches), set M_per_patch to 1100
# If using 3x3 atlas then 2 sheets (18 Patches), set M_per_patch to 2000
# if using 2x2 atlas then 2 sheets (8 Patches), set M_per_patch to 4096
# python main.py \
#     --multi_patch \
#     --atlas_mode two_sheet \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --shape sphere \
#     --epochs 5000 \
#     --d_features 88 \
#     --M_per_patch 2000 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --lambda_outer_boundary 0 \
#     --log_every 100 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --N 5000 \
#     --checkpoint_every 5000 \
#     --two_sheet_side_rows 3 \
#     --two_sheet_side_cols 3 \
#     --two_sheet_split_axis 2 \
#     --two_sheet_side_axes 0 1 \
#     --pretrain_then_train \
#     --pretrain_epochs 2000 \
#     --pretrain_shape sphere \
#     --pretrain_mode closed_shape \
#     --no_presplit \
#     --correspondence_line_segment q_to_t \


# # initialize the model as a box
# # 2x2 patches per sheet, 6 sheets (24 patches), M = 1500
# # 3x3 patches per sheet, 6 sheets (54 patches), M = 700
# python main.py \
#     --multi_patch \
#     --atlas_mode six_sheet \
#     --file ${INPUT_FILE} \
#     --result_dir ${OUTPUT_DIR} \
#     --epochs 5000 \
#     --d_features 88 \
#     --M_per_patch 500 \
#     --W 512 \
#     --D 6 \
#     --L 0 \
#     --beta 100 \
#     --mu 0.08 \
#     --gamma 0 \
#     --lam 0 \
#     --lam2 0 \
#     --lambda_outer_boundary 0 \
#     --log_every 100 \
#     --mu_warmup_epochs 1000 \
#     --mu_warmup_delay 300 \
#     --schedule cosine \
#     --N 100000 \
#     --checkpoint_every 1000 \
#     --six_sheet_face_rows 2 \
#     --six_sheet_face_cols 2 \
#     --face_aware_box_supervision \
#     --pretrain_then_train \
#     --pretrain_epochs 1000 \
#     --pretrain_shape box \
#     --pretrain_mode closed_shape \
#     --no_presplit \
#     --correspondence_line_segment q_to_t \
#     --corr_switch_epoch 0 \
#     --save_boundary_debug 0 \
#     --save_correspondence_every 5000 \
#     --surface_loss_type chamfer \
#     --ddf_start_epoch 0 \
#     --ddf_sigma 0.05 \
#     --ddf_mu_decay 1 \
#     --chamfer_resume_epoch 0 \
#     --lambda_chamfer 1 \
#     --lambda_ddf 1 \

EPOCHS="${EPOCHS:-10000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
PATCH_VIS="${PATCH_VIS:-0}"
INIT_CKPT="${INIT_CKPT:-}"
INIT_CKPT_REUSE_NORMALIZATION="${INIT_CKPT_REUSE_NORMALIZATION:-0}"

INIT_ARGS=()
if [[ -n "${INIT_CKPT}" ]]; then
    INIT_ARGS+=(--init_ckpt "${INIT_CKPT}")
fi
if [[ "${INIT_CKPT_REUSE_NORMALIZATION}" == "1" ]]; then
    INIT_ARGS+=(--init_ckpt_reuse_normalization)
fi

# Start from the inside
python main.py \
    --adaptive \
    --visible_epochs 1000 \
    --epochs ${EPOCHS} \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.08 \
    --mu 0.08 \
    --svd_mode collapse \
    --svd_warmup_delay 3000 \
    --svd_warmup_epochs 500 \
    --pretrain_epochs 500 \
    --M_per_patch 64 \
    --L 0 \
    --log_every 500 \
    --checkpoint_every ${CHECKPOINT_EVERY} \
    --tangent_mode dirichlet \
    --sym_dirichlet_epoch 0 \
    --gamma 0.0 \
    --gamma_warmup_delay 5000 \
    --gamma_warmup_epochs 1000 \
    --normal_hinge_tau 0.3 \
    --surface_loss_type chamfer \
    --normal_off_frac 1 \
    --ghpr_init \
    --ghpr_keep_largest \
    --ghpr_stages 5 \
    --ghpr_subdiv_max_depth 3 \
    --ext_mask_epoch 5000 \
    --ext_mask_every 2000 \
    --lambda_ext_area 1.0


# python utils/patch_vis.py \
#     --ckpt ${OUTPUT_DIR}/checkpoint_100_before_subdivision.pt \
#     --out_dir ${OUTPUT_DIR} \
#     --n_images 1 \
#     --subdivision_depth "-1" \
#     # --input_file ${INPUT_FILE} \

python utils/patch_vis.py \
    --ckpt ${OUTPUT_DIR}/checkpoint_5000.pt \
    --out_dir ${OUTPUT_DIR} \
    --n_images 4 \
    --subdivision_depth "-1" \
    # --no_unnormalize \
    # --input_file ${INPUT_FILE} \