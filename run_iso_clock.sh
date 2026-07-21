#!/bin/bash
# ISO-CLOCK MEASUREMENT: Softmax at 16,290 steps × 3 seeds
# Matched to SBA V3 kernel wall-clock (SBA 15k steps at 1.086× overhead)
# 
# IMPORTANT: Nothing else should run on this GPU during these ~11 hours.
# These numbers are the headline for the writeup.

set -e

PYTHON=/venv/main/bin/python
SCRIPT=sba_standalone/run_scratch_tokenized_compare.py
BASE_DIR=sba_standalone/results/iso_v3

echo "========================================================================"
echo "ISO-CLOCK MEASUREMENT — Softmax at 16,290 steps"
echo "Started at: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================================"

for SEED in 42 43 44; do
    OUT_DIR="${BASE_DIR}/seed${SEED}"
    echo ""
    echo "================================================================"
    echo "SEED ${SEED} — Starting at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Output: ${OUT_DIR}"
    echo "================================================================"

    $PYTHON $SCRIPT \
        --arms softmax \
        --batch-size 16 \
        --block-size 1024 \
        --data-files data/open_reason_code_mix.jsonl \
        --dropout 0.0 \
        --dtype bf16 \
        --eval-limit-per-file 4000 \
        --grad-accum-steps 4 \
        --grad-clip 1.0 \
        --log-every 50 \
        --lr 0.0006 \
        --n-embd 768 \
        --n-head 12 \
        --n-layer 12 \
        --output-dir "${OUT_DIR}" \
        --seed ${SEED} \
        --steps 16290 \
        --train-limit-per-file 135000 \
        --warmup-steps 1000 \
        --weight-decay 0.01

    echo "SEED ${SEED} — Finished at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

    # Print result
    if [ -f "${OUT_DIR}/softmax/result.json" ]; then
        echo "Result:"
        cat "${OUT_DIR}/softmax/result.json"
    fi
done

echo ""
echo "========================================================================"
echo "ALL SEEDS COMPLETE — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================================"

# Summary
echo ""
echo "RESULTS SUMMARY:"
for SEED in 42 43 44; do
    OUT_DIR="${BASE_DIR}/seed${SEED}"
    if [ -f "${OUT_DIR}/softmax/result.json" ]; then
        VAL=$(python3 -c "import json; print(json.load(open('${OUT_DIR}/softmax/result.json'))['final_val_loss'])")
        ELAPSED=$(python3 -c "import json; print(f\"{json.load(open('${OUT_DIR}/softmax/result.json'))['elapsed_seconds']:.0f}\")")
        echo "  Seed ${SEED}: val_loss=${VAL}, elapsed=${ELAPSED}s"
    fi
done

echo ""
echo "SBA baselines (from overnight sweep at 15k steps):"
echo "  Seed 42: val_loss=1.346876"
echo "  Seed 43: val_loss=1.350833"
echo "  Seed 44: val_loss=1.347314"
echo "  Average: 1.348341"
