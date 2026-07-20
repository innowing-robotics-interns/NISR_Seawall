#!/bin/bash

FILE="noisy_data_1_normals.xyz"
INPUT_FILE="data/${FILE}"
OUTPUT_DIR="logs/${FILE%.*}"

for file in "${OUTPUT_DIR}/checkpoint"/*.pt; 
do 
    python utils/patch_vis.py \
        --ckpt "$file" \
        --out_dir ${OUTPUT_DIR} \
        --n_images 1 \
        --main_path main.py 
        
done; wait


