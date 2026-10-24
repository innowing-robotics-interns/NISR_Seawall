#!/usr/bin/env bash
# Launch NISR_Seawall training through run.sh for one shape.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${SEAWALL_PYTHON:-/home/lyangiw/anaconda3/envs/seawall/bin/python}"
PYTHON_BIN_DIR="$(dirname "${PYTHON_BIN}")"
RUNNER="${RUNNER:-run.sh}"

SHAPE=""
INPUT_FILE_ARG=""
RESULT_DIR=""
EPOCHS=""
CHECKPOINT_EVERY=""
INIT_CKPT=""
REUSE_NORMALIZATION=0
PATCH_VIS=0
LOG=""
FOREGROUND=0
FORCE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Train one shape with run.sh.

Usage: train_shape.sh [options]

  --shape NAME|PATH        Shape to train; bare names resolve under
                           normalized_shapes/, 3d_test_models/, data/
  --input-file PATH        Explicit input .ply (overrides --shape)
  --result-dir DIR         Base output dir (default logs/adaptive/<stem>_train)
  --epochs N               Override --epochs from run.sh
  --checkpoint-every N     Override checkpoint cadence from run.sh
  --init-ckpt PATH         Initialize/resume from a checkpoint
  --reuse-normalization    Add --init_ckpt_reuse_normalization
  --patch-vis              Run run.sh's optional post-training export
  --foreground             Run in the foreground instead of detached
  --force                  Launch even if another main.py is running
  --log PATH               Log file (default logs/<stem>_train_<timestamp>.log)
  --dry-run                Print the resolved settings and command only
  -h, --help               Show this help

Environment:
  SEAWALL_PYTHON           Interpreter to put on PATH (default: seawall env)
  RUNNER                   Script to launch (default: run.sh)
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "missing value for $1"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shape)              need_value "$@"; SHAPE="$2"; shift 2 ;;
        --input-file)         need_value "$@"; INPUT_FILE_ARG="$2"; shift 2 ;;
        --result-dir)         need_value "$@"; RESULT_DIR="$2"; shift 2 ;;
        --epochs)             need_value "$@"; EPOCHS="$2"; shift 2 ;;
        --checkpoint-every)   need_value "$@"; CHECKPOINT_EVERY="$2"; shift 2 ;;
        --init-ckpt)          need_value "$@"; INIT_CKPT="$2"; shift 2 ;;
        --reuse-normalization) REUSE_NORMALIZATION=1; shift ;;
        --patch-vis)          PATCH_VIS=1; shift ;;
        --foreground)         FOREGROUND=1; shift ;;
        --force)              FORCE=1; shift ;;
        --log)                need_value "$@"; LOG="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=1; shift ;;
        -h|--help)            usage; exit 0 ;;
        *)                    die "unknown option: $1" ;;
    esac
done

[[ -x "${PYTHON_BIN}" ]] || die "interpreter not found: ${PYTHON_BIN} (set SEAWALL_PYTHON)"
[[ -f "${RUNNER}" ]] || die "runner not found: ${RUNNER} (run from the repo root)"
[[ -z "${SHAPE}" || -z "${INPUT_FILE_ARG}" ]] || die "use either --shape or --input-file, not both"

resolve_input() {
    local arg="$1" name dir
    if [[ -f "${arg}" ]]; then
        printf '%s\n' "${arg}"
        return 0
    fi
    name="${arg}"
    [[ "${name}" == *.ply ]] || name="${name}.ply"
    for dir in normalized_shapes 3d_test_models data .; do
        if [[ -f "${dir}/${name}" ]]; then
            if [[ "${dir}" == "." ]]; then
                printf '%s\n' "${name}"
            else
                printf '%s\n' "${dir}/${name}"
            fi
            return 0
        fi
    done
    return 1
}

INPUT_FILE=""
FILE_NAME=""
STEM=""
if [[ -n "${INPUT_FILE_ARG}" ]]; then
    INPUT_FILE="$(resolve_input "${INPUT_FILE_ARG}")" \
        || die "input file not found: ${INPUT_FILE_ARG}"
elif [[ -n "${SHAPE}" ]]; then
    INPUT_FILE="$(resolve_input "${SHAPE}")" \
        || die "shape not found: ${SHAPE} (tried normalized_shapes/, 3d_test_models/, data/, .)"
fi

if [[ -n "${INPUT_FILE}" ]]; then
    FILE_NAME="$(basename "${INPUT_FILE}")"
    STEM="${FILE_NAME%.*}"
fi

if [[ -n "${INIT_CKPT}" ]]; then
    [[ -f "${INIT_CKPT}" ]] || die "checkpoint not found: ${INIT_CKPT}"
fi

if [[ -z "${RESULT_DIR}" && -n "${STEM}" ]]; then
    RESULT_DIR="logs/adaptive/${STEM}_train"
fi
if [[ -z "${LOG}" ]]; then
    LOG="logs/${STEM:-run}_train_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "${LOG}")"

if [[ "${FORCE}" -ne 1 && "${DRY_RUN}" -ne 1 ]]; then
    RUNNING="$(pgrep -af '^(/[^ ]*/)?python[0-9.]* +main\.py( |$)' || true)"
    if [[ -n "${RUNNING}" ]]; then
        echo "ERROR: another training process appears to be running:" >&2
        echo "${RUNNING}" >&2
        echo "Stop it first, or pass --force to launch anyway." >&2
        exit 1
    fi
fi

GPU_LINE=""
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LINE="$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -n 1 || true)"
fi

ENV_ARGS=(
    "PATH=${PYTHON_BIN_DIR}:${PATH}"
    "PYTHONUNBUFFERED=1"
    "PATCH_VIS=${PATCH_VIS}"
)
if [[ -n "${INPUT_FILE}" ]]; then
    ENV_ARGS+=("FILE=${FILE_NAME}" "INPUT_FILE=${INPUT_FILE}")
fi
if [[ -n "${RESULT_DIR}" ]]; then
    ENV_ARGS+=("OUTPUT_DIR=${RESULT_DIR}")
fi
if [[ -n "${EPOCHS}" ]]; then
    ENV_ARGS+=("EPOCHS=${EPOCHS}")
fi
if [[ -n "${CHECKPOINT_EVERY}" ]]; then
    ENV_ARGS+=("CHECKPOINT_EVERY=${CHECKPOINT_EVERY}")
fi
if [[ -n "${INIT_CKPT}" ]]; then
    ENV_ARGS+=("INIT_CKPT=${INIT_CKPT}")
fi
if [[ "${REUSE_NORMALIZATION}" -eq 1 ]]; then
    ENV_ARGS+=("INIT_CKPT_REUSE_NORMALIZATION=1")
fi

echo "Shape      : ${INPUT_FILE:-${RUNNER} default}"
echo "Result dir : ${RESULT_DIR:-${RUNNER} default}"
echo "Epochs     : ${EPOCHS:-${RUNNER} default}"
echo "Checkpoints: ${CHECKPOINT_EVERY:-${RUNNER} default}"
[[ -n "${GPU_LINE}" ]] && echo "GPU        : ${GPU_LINE}"
echo "Log        : ${LOG}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf 'Command    :'
    printf ' %q' "${ENV_ARGS[@]}" bash "${RUNNER}"
    printf '\n'
    exit 0
fi

if [[ "${FOREGROUND}" -eq 1 ]]; then
    echo "Launching in foreground (Ctrl-C stops the shell, not necessarily python)."
    env "${ENV_ARGS[@]}" bash "${RUNNER}" 2>&1 | tee "${LOG}"
    exit "${PIPESTATUS[0]}"
fi

command -v setsid >/dev/null 2>&1 || die "setsid not found; use --foreground"
PID_FILE="${LOG}.pid"
setsid bash -c 'pid_file=$1; shift; echo "$$" > "${pid_file}"; exec env "$@"' \
    _ "${PID_FILE}" "${ENV_ARGS[@]}" bash "${RUNNER}" >"${LOG}" 2>&1 &

for _ in $(seq 1 50); do
    [[ -s "${PID_FILE}" ]] && break
    sleep 0.1
done
TRAIN_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -z "${TRAIN_PID}" ]] || ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "ERROR: training failed to start; see ${LOG}" >&2
    tail -n 20 "${LOG}" >&2 || true
    exit 1
fi

echo "PID        : ${TRAIN_PID} (process group; stop with: kill -TERM -- -${TRAIN_PID})"
echo "Started. Follow with: tail -n 40 ${LOG}"
