#!/bin/bash
# ============================================================================
# run_pipeline.sh — Run the complete thunderstorm prediction pipeline
# ============================================================================
# Usage:
#   cd /Storage03/nverma1/lightning_project/src
#   bash run_pipeline.sh          # run all steps
#   bash run_pipeline.sh 1 2 3    # run only steps 1, 2, and 3
# ============================================================================

set -e  # exit on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/Storage03/nverma1/lightning_project/scripts/venv"

# Activate venv
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "Activated venv: $VENV"
else
    echo "WARNING: venv not found at $VENV — using system Python"
fi

cd "$SCRIPT_DIR"

# Which steps to run
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
    STEPS=(1 2 3 4 5)
fi

run_step() {
    local n=$1
    echo ""
    echo "======================================================================"
    echo "  PIPELINE STEP $n"
    echo "======================================================================"
    case $n in
        1) python 01_extract_features.py ;;
        2) python 02_label_lightning.py  ;;
        3) python 03_build_dataset.py    ;;
        4) python 04_train_evaluate.py   ;;
        5) python 05_visualize.py        ;;
        *) echo "Unknown step: $n" && exit 1 ;;
    esac
}

echo ""
echo "======================================================================"
echo "  DEEP LEARNING THUNDERSTORM MODEL — FULL PIPELINE"
echo "  Steps: ${STEPS[*]}"
echo "======================================================================"

for step in "${STEPS[@]}"; do
    run_step $step
done

echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE"
echo "  Results: /Storage03/nverma1/lightning_project/results/"
echo "  Models:  /Storage03/nverma1/lightning_project/models/"
echo "======================================================================"
