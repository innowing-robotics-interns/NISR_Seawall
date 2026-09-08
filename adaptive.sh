#!/bin/bash

FILE="max-planck.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"

python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="stanford-bunny.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"

python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}_2"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="bimba_pc.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"
python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="horse.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"
python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="dress.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"
python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="dancing_children.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"
python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \

FILE="rocker-arm.ply"
INPUT_FILE="3dModel/${FILE}"
OUTPUT_DIR="logs/adaptive/${FILE%.*}_dirichlet"
python main_adaptive.py \
    --file ${INPUT_FILE} \
    --result_dir ${OUTPUT_DIR} \
    --subdiv_threshold 0.1 \
    --subdiv_max_depth 7 \
    --lam_svd 0.005 \
    --mu 0.08 \
    --svd_mode arap \
    --svd_warmup_delay 500 \
    --svd_warmup_epochs 1000 \
    --pretrain_epochs 500 \
    --epochs 5000 \
    --L 0

REAL_OUTPUT_DIR="${OUTPUT_DIR}"

python utils/patch_vis.py \
    --ckpt ${REAL_OUTPUT_DIR}/checkpoint.pt \
    --out_dir ${REAL_OUTPUT_DIR} \
    --n_images 1 \
    --resolution 64 \