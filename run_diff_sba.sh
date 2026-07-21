#!/bin/bash
set -e

# ==========================================
# 🚀 OPERATION: BREAK ZERO (Diff-SBA Arm)
# ==========================================

OUT_DIR="sba_standalone/results/blackwell_125m_overnight"
DATA_FILE="data/open_reason_code_mix.jsonl"
STEPS=15000
BLOCK_SIZE=1024

echo "Starting 125M Parameter Training Run (Diff-SBA Arm)..."

/venv/main/bin/python sba_standalone/run_scratch_tokenized_compare.py \
    --output-dir "$OUT_DIR" \
    --data-files "$DATA_FILE" \
    --train-limit-per-file 135000 \
    --eval-limit-per-file 4000 \
    --n-layer 12 \
    --n-head 12 \
    --n-embd 768 \
    --block-size $BLOCK_SIZE \
    --batch-size 16 \
    --grad-accum-steps 4 \
    --steps $STEPS \
    --warmup-steps 1000 \
    --lr 6e-4 \
    --weight-decay 0.01 \
    --dtype bf16 \
    --arms diff_sba \
    --save-checkpoints \
    --checkpoint-every 3000

echo "✅ Training Complete. Commencing Automated Task Evaluation Sweep for diff_sba..."

for CKPT in "$OUT_DIR/diff_sba/checkpoints/"*.pt; do
    if [ -f "$CKPT" ]; then
        STEP_NUM=$(echo $CKPT | grep -o -E 'step[0-9]+' | tail -n 1)
        EVAL_OUT="$OUT_DIR/diff_sba/task_eval_${STEP_NUM}.json"
        
        echo "   -> Evaluating $CKPT"
        /venv/main/bin/python sba_standalone/evaluate_scratch_tokenized_checkpoint.py \
            --checkpoint "$CKPT" \
            --output "$EVAL_OUT" \
            --max-new-tokens 384 \
            --temperature 0.0 \
            --device cuda
    fi
done

echo "🎉 Diff-SBA training run and evaluations finished successfully!"
