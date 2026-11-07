#!/bin/bash
#
# Cloth-deformation sequence on the adaptive cube atlas (main.py --adaptive).
#
#   frame 0  : trained from scratch (FRAME0_ARGS), or an existing run reused
#              via FRAME0_CKPT
#   frame k  : warm-started from frame k-1 (--load_ckpt), quadtree frozen,
#              plus two anchors that keep each (leaf, u, v) on the same cloth
#              point:
#                (A) --lambda_iso  isometry vs. FRAME 0  (bending keeps the metric)
#                (B) --lambda_tan  tangential drift vs. FRAME k-1
#
# Every frame is normalized on its own to [-1,1]^3 (utils.load_point_cloud);
# both anchors compare frames in world units, so this is accounted for.
#
# Surface fit: SURFACE=render (default, NGF rendering loss as in the
# "Rendering Loss" block of adaptive.sh; needs file normals + nvdiffrast) or
# SURFACE=cd (Chamfer).
#
# Edit the USER SETTINGS block below, then run:  ./run_cloth_sequence.sh
# Any setting can also be overridden from the terminal, e.g.
#   NUM_FRAMES=5 LAMBDA_ISO=0.5 ./run_cloth_sequence.sh
# Baseline (plain warm start, no anchors):
#   LAMBDA_ISO=0 LAMBDA_TAN=0 OUTPUT_DIR=logs/.../noAnchor ./run_cloth_sequence.sh
# Resuming at frame 12 (frames 0..11 already in OUTPUT_DIR):
#   START_INDEX=12 ./run_cloth_sequence.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ════════════════════════════════════════════════════════════════════════
#  USER SETTINGS — edit here (an environment variable of the same name
#  still overrides each value)
# ════════════════════════════════════════════════════════════════════════
# Folder holding the frame point clouds (*.ply, processed in sorted order)
DATA_DIR="${DATA_DIR:-t-shirt_datasets/shape00/108_22}"
# Output folder; one subfolder per frame (001_0001_garment, 002_...) is made inside
OUTPUT_DIR="${OUTPUT_DIR:-logs/clothDeformation/t-shirt/shape00/108_22/sequence/render_iso0.1_tan0.1_noSubdiv}"
# Trained frame-0 checkpoint to reuse ("" = train frame 0 from scratch first)
FRAME0_CKPT="${FRAME0_CKPT:-logs/clothDeformation/t-shirt/shape00/108_22/0000_garment_inside_noSubdiv/checkpoint.pt}"
# FRAME0_CKPT=""
# Frames to run: 0-based start index and how many (0 = all remaining)
START_INDEX="${START_INDEX:-0}"
NUM_FRAMES="${NUM_FRAMES:-0}"
# Surface fit: 'render' (NGF rendering loss) or 'cd' (Chamfer)
SURFACE="${SURFACE:-render}"
# Anchor weights: (A) isometry vs. frame 0, (B) tangential drift vs. previous frame
LAMBDA_ISO="${LAMBDA_ISO:-0.1}"
LAMBDA_TAN="${LAMBDA_TAN:-0.1}"
# Epochs and learning rate for every frame after frame 0
WARM_EPOCHS="${WARM_EPOCHS:-1000}"
WARM_LR="${WARM_LR:-1e-4}"
# ════════════════════════════════════════════════════════════════════════

case "$SURFACE" in render|cd) ;; * ) echo "SURFACE must be 'render' or 'cd'" >&2; exit 1 ;; esac
OUT_ROOT="$OUTPUT_DIR"

# Render-loss settings (the "Rendering Loss" block of adaptive.sh).  The
# reference maps are rebuilt for every frame from that frame's own normalized
# cloud and file normals.  Chamfer is then only a monitor (8x8 grid per leaf).
RENDER_ARGS=(
    --lambda_render 1.0 --render_views "${RENDER_VIEWS:-120}"
    --render_batch "${RENDER_BATCH:-10}" --render_layers "${RENDER_LAYERS:-2}"
    --render_resolution "${RENDER_RES:-256}" --render_min_coverage 0.05
    --render_laplacian_weight 0.0
    --cd_sampling grid --cd_grid_res 8
)

# ── frame 0 ─────────────────────────────────────────────────────────────
# FRAME0_CKPT (above) should have been trained with the same SURFACE fit as
# the warm frames.
FRAME0_EPOCHS="${FRAME0_EPOCHS:-10000}"
if [ "$SURFACE" = "render" ]; then
    # Same configuration as the "Rendering Loss" block of adaptive.sh.
    FRAME0_ARGS=(
        "${RENDER_ARGS[@]}"
        --epochs "$FRAME0_EPOCHS"
        --M_per_patch "${FRAME0_M_PER_PATCH:-900}"
        --pretrain_epochs 500
        --mu 0.08 --mu_warmup_delay 1000 --mu_warmup_epochs 1000
        --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
        --gamma 0
        --subdiv_threshold 0.1 --subdiv_every 500 --subdiv_start 1000
        --subdiv_max_depth 1 --max_splits_per_round 48
        --ext_mask_epoch 5000 --ext_mask_every 10000 --ext_mode no_fit --lambda_ext_area 0.0
        --ghpr_init --mesh_res 40
        --lr 0.0002
    )
else
    # Same configuration as the "Start from the inside" block of adaptive.sh.
    FRAME0_ARGS=(
        --visible_epochs 1000
        --epochs "$FRAME0_EPOCHS"
        --subdiv_threshold 0.1 --subdiv_max_depth 5 --max_splits_per_round 48
        --cd_sampling grid --cd_grid_res 0
        --mu 0.08 --tangent_mode dirichlet --sym_dirichlet_epoch 0
        --lam_svd 0.08 --svd_mode conformal --svd_warmup_delay 3000 --svd_warmup_epochs 500
        --pretrain_epochs 500
        --gamma 0
        --surface_loss_type chamfer
        --ghpr_init --ghpr_keep_largest --ghpr_stages 5 --ghpr_subdiv_max_depth 3
        --M_per_patch "${FRAME0_M_PER_PATCH:-121}"
    )
fi

# ── frames >= 1 ─────────────────────────────────────────────────────────
# μ (Dirichlet) shrinks the surface and would fight the isometry term, so the
# warm frames keep only the anchors as regularizers by default.
WARM_MU="${WARM_MU:-0}"
WARM_LAM_SVD="${WARM_LAM_SVD:-0}"
WARM_ARGS=(
    --epochs "$WARM_EPOCHS"
    # Per LEAF: a fitted frame 0 has hundreds of leaves, and the two frozen
    # reference models add ~0.12 GB per 1k samples (249 leaves x 64 ≈ 4.7 GB peak).
    --M_per_patch "${WARM_M_PER_PATCH:-$([ "$SURFACE" = render ] && echo 64 || echo 128)}"
    --lr "$WARM_LR"
    --freeze_topology
    --mu "$WARM_MU"
    --lam_svd "$WARM_LAM_SVD" --svd_mode conformal
    --gamma 0
    --surface_loss_type chamfer
    --lambda_iso "$LAMBDA_ISO"
    --lambda_tan "$LAMBDA_TAN"
)
# Same surface fit as frame 0.  The anchors are per-sample terms on the
# --M_per_patch samples and work unchanged with either fit.
if [ "$SURFACE" = "render" ]; then WARM_ARGS+=( "${RENDER_ARGS[@]}" ); fi

# ── shared by every frame ───────────────────────────────────────────────
COMMON_ARGS=(
    --adaptive
    --N "${N_POINTS:-100000}"
    --L 0
    --log_every 100
    --checkpoint_every 500
)
if [ -n "${SEED:-}" ]; then COMMON_ARGS+=( --seed "$SEED" ); fi

RUN_VIS="${RUN_VIS:-1}"             # checkerboard export after each frame
VIS_RES="${VIS_RES:-64}"

# ── frames ──────────────────────────────────────────────────────────────
mapfile -t FILES < <(find "$DATA_DIR" -maxdepth 1 -name '*.ply' | sort)
TOTAL=${#FILES[@]}
if [ "$TOTAL" -eq 0 ]; then echo "No .ply files in $DATA_DIR" >&2; exit 1; fi
END=$TOTAL
if [ "$NUM_FRAMES" -gt 0 ]; then END=$(( START_INDEX + NUM_FRAMES )); fi
if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi
mkdir -p "$OUT_ROOT"

frame_dir() {   # output directory of frame index $1
    local f="${FILES[$1]}"
    echo "${OUT_ROOT}/$(printf '%03d' "$1")_$(basename "${f%.*}")"
}

# frame-0 checkpoint: given, or the one this script trained earlier
if [ -z "$FRAME0_CKPT" ] && [ "$START_INDEX" -gt 0 ]; then
    FRAME0_CKPT="$(frame_dir 0)/checkpoint.pt"
fi
prev_ckpt=""
if [ "$START_INDEX" -gt 0 ]; then
    prev_ckpt="$(frame_dir $(( START_INDEX - 1 )))/checkpoint.pt"
    [ -f "$prev_ckpt" ] || { echo "Missing previous frame checkpoint: $prev_ckpt" >&2; exit 1; }
fi

echo "Sequence: $DATA_DIR  frames ${START_INDEX}..$(( END - 1 )) of $TOTAL  → $OUT_ROOT"
echo "  λ_iso=$LAMBDA_ISO (vs. frame 0)  λ_tan=$LAMBDA_TAN (vs. previous frame)"

for (( i = START_INDEX; i < END; i++ )); do
    file="${FILES[$i]}"
    out="$(frame_dir "$i")"
    echo
    echo "════════ frame $i / $(( TOTAL - 1 )): $file ════════"

    if [ "$i" -eq 0 ] && [ -n "$FRAME0_CKPT" ]; then
        [ -f "$FRAME0_CKPT" ] || { echo "FRAME0_CKPT not found: $FRAME0_CKPT" >&2; exit 1; }
        echo "  reusing frame-0 checkpoint $FRAME0_CKPT"
        prev_ckpt="$FRAME0_CKPT"
        continue
    fi

    # main.py appends _1, _2 ... to an existing result_dir, which would break
    # the chaining below, so refuse instead of silently writing elsewhere.
    if [ -e "$out" ]; then
        echo "  $out already exists; remove it or change OUTPUT_DIR / START_INDEX" >&2
        exit 1
    fi

    if [ "$i" -eq 0 ]; then
        python main.py "${COMMON_ARGS[@]}" "${FRAME0_ARGS[@]}" \
            --file "$file" --result_dir "$out"
        FRAME0_CKPT="$out/checkpoint.pt"
    else
        python main.py "${COMMON_ARGS[@]}" "${WARM_ARGS[@]}" \
            --file "$file" --result_dir "$out" \
            --load_ckpt "$prev_ckpt" \
            --prev_ckpt "$prev_ckpt" \
            --ref_ckpt "$FRAME0_CKPT"
    fi
    prev_ckpt="$out/checkpoint.pt"
    [ -f "$prev_ckpt" ] || { echo "Frame $i produced no checkpoint.pt" >&2; exit 1; }

    if [ "$RUN_VIS" = "1" ]; then
        python utils/patch_vis.py --ckpt "$prev_ckpt" --out_dir "$out" \
            --n_images 4 --resolution "$VIS_RES" --no_svd
    fi
done

echo
echo "Done → $OUT_ROOT"
