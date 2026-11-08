#!/bin/bash
# Compare warm-up checkpoints on the same real fits.
#
# Usage:
#   bash run_plc_test.sh fixed  warmup/cube_warmup_pool_noaxis.pt
#   bash run_plc_test.sh random warmup/cube_warmup_random.pt
#
# Every run uses the match-angle 150 / axis-off fit the eight-shape panel uses,
# changing only the warm-up checkpoint, and writes to
# log/<stamp>_<tag>_<shape>/, where stamp is the launch datetime. The shapes span the loop-size range measured
# in the panel: bunny's loops are large, armadillo's small, horse and laurent_hand
# mixed. The distillation teacher defaults to --warmup, so each arm is distilled
# by the checkpoint it started from.
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1

if (($# < 2)); then
  echo "usage: bash run_plc_test.sh TAG WARMUP_CKPT" >&2
  exit 2
fi

TAG=$1
WARMUP=$2
PY=/home/lyangiw/anaconda3/envs/seawall/bin/python
SHAPES=(bunny armadillo horse laurent_hand)
declare -A STEM=(
  [bunny]=bunny_points_cubic_lam3
  [armadillo]=armadillo_normal_cubic_lam3
  [horse]=horse_points_cubic_lam3
  [laurent_hand]=laurent_hand_normal_cubic_lam3
)

STAMP=$(date +%Y%m%d_%H%M%S)

for SHAPE in "${SHAPES[@]}"; do
  OUT="log/${STAMP}_${TAG}_${SHAPE}"
  LOG="${OUT}/run.txt"
  mkdir -p "$OUT"
  echo "=== ${TAG} ${SHAPE} start $(date -Iseconds)" > "$LOG"
  ## the command is built once and both written to the log and run, so the line
  ## at the top of run.txt is the run itself, quoted to be pasted back as-is
  CMD=(
    "$PY" plc_main.py
    --path "normalized_shapes_cuboid/${STEM[$SHAPE]}.ply"
    --train --samples 29 --epochs 1000
    --feature triplane --feature-pool occupied_cells --triplane-tv 0.01
    --warmup "$WARMUP"
    --match-angle 150 --normal 0.01
    --freeze-f-epochs 100 --distill 10
    --snapshot-every 0
    --out "$OUT"
  )
  printf '%q ' "${CMD[@]}" >> "$LOG"
  printf '\n\n' >> "$LOG"
  "${CMD[@]}" >> "$LOG" 2>&1
  code=$?
  echo "=== ${TAG} ${SHAPE} fit exit $code $(date -Iseconds)" >> "$LOG"
  if ((code != 0)); then
    echo "$SHAPE failed; see $LOG" >&2
    exit $code
  fi
  echo "$SHAPE done"
done
