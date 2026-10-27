#!/usr/bin/env bash
# Launch NISR_Seawall with the local "seawall" conda environment.
# All arguments are passed straight through to main.py.
#
# Examples:
#   ./run_seawall.sh --file 3d_test_models/bunny_points.ply --N 20000 --epochs 100
#   ./run_seawall.sh --file /path/to/scan.ply --result_dir logs/my_scan
#   ./run_seawall.sh --adaptive --file /path/to/scan.ply --epochs 1000
#   ./run_seawall.sh --help
#
# Override the interpreter with SEAWALL_PYTHON=/path/to/python if needed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SEAWALL_PYTHON:-/home/lyangiw/anaconda3/envs/seawall/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    echo "Set SEAWALL_PYTHON or recreate the 'seawall' conda env." >&2
    exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON_BIN" main.py "$@"
