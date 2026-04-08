#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="/Storage03/nverma1/lightning_project/src_v3/venv"

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate" && echo "Activated venv"
cd "$SCRIPT_DIR"

# 🔥 FIX: define python here (outside case)
PY="$VENV/bin/python"

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(1 2 3 4 5 6 7)

for n in "${STEPS[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  STEP $n  |  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════"

    case $n in
        1) $PY 01_extract_features.py ;;
        2) $PY 02_label_lightning.py ;;
        3) $PY 03_build_dataset.py ;;
        4) $PY 04_train_evaluate.py ;;
        5) $PY 05_visualize.py ;;
        6) $PY 06_train_deep_learning.py --model all ;;
        7) $PY 07_master_comparison.py ;;
        *) echo "Unknown step $n (valid: 1-7)" && exit 1 ;;
    esac
done

echo ""
echo "══════════════════════════════════════════════"
echo "  PIPELINE COMPLETE  |  $(date '+%Y-%m-%d %H:%M:%S')"
echo "══════════════════════════════════════════════"