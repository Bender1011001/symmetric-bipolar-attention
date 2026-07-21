#!/bin/bash
set -e

# ==========================================
# 🚀 RIGOROUS BENCHMARK SWEEP (SBA vs Softmax)
# ==========================================

ROOT_DIR="sba_standalone/results/blackwell_125m_overnight"
DATA_FILE="data/open_reason_code_mix.jsonl"
STEPS=15000
ISO_STEPS=22760
BLOCK_SIZE=1024

# Helper function to run a single sweep configuration
run_config() {
    local arm=$1
    local seed=$2
    local steps=$3
    local folder=$4
    
    echo "=========================================================="
    echo "▶ STARTING: $folder (Arm: $arm, Seed: $seed, Steps: $steps)"
    echo "=========================================================="
    
    # 1. Run the training command
    /venv/main/bin/python sba_standalone/run_scratch_tokenized_compare.py \
        --output-dir "${ROOT_DIR}/${folder}_temp" \
        --data-files "$DATA_FILE" \
        --train-limit-per-file 135000 \
        --eval-limit-per-file 4000 \
        --n-layer 12 \
        --n-head 12 \
        --n-embd 768 \
        --block-size $BLOCK_SIZE \
        --batch-size 16 \
        --grad-accum-steps 4 \
        --steps $steps \
        --warmup-steps 1000 \
        --lr 6e-4 \
        --weight-decay 0.01 \
        --dtype bf16 \
        --arms $arm \
        --sba-impl sdpa \
        --seed $seed \
        --save-checkpoints \
        --checkpoint-every 3000
        
    # 2. Flatten directory structure
    mv "${ROOT_DIR}/${folder}_temp/${arm}" "${ROOT_DIR}/${folder}"
    rm -rf "${ROOT_DIR}/${folder}_temp"
    
    # 3. Evaluate checkpoints
    echo "🔍 Evaluating checkpoints for $folder..."
    for CKPT in "${ROOT_DIR}/${folder}/checkpoints/"*.pt; do
        if [ -f "$CKPT" ]; then
            STEP_NUM=$(echo $CKPT | grep -o -E 'step[0-9]+' | tail -n 1)
            EVAL_OUT="${ROOT_DIR}/${folder}/task_eval_${STEP_NUM}.json"
            
            echo "   -> Evaluating $CKPT"
            /venv/main/bin/python sba_standalone/evaluate_scratch_tokenized_checkpoint.py \
                --checkpoint "$CKPT" \
                --output "$EVAL_OUT" \
                --max-new-tokens 384 \
                --temperature 0.0 \
                --device cuda
        fi
    done
    
    echo "✅ Completed: $folder"
    echo "=========================================================="
}

# Queue runs sequentially
# 1. Softmax (Seed 43) - ~3.36h
run_config "softmax" 43 $STEPS "softmax_seed43"

# 2. Softmax (Seed 44) - ~3.36h
run_config "softmax" 44 $STEPS "softmax_seed44"

# 3. SBA (Seed 43) - ~5.11h
run_config "sba" 43 $STEPS "sba_seed43"

# 4. SBA (Seed 44) - ~5.11h
run_config "sba" 44 $STEPS "sba_seed44"

# 5. Softmax Iso-Clock (Seed 42) - ~5.08h
run_config "softmax" 42 $ISO_STEPS "softmax_iso"

# Run collation summary
echo "📊 Run queue complete! Collation summary:"
/venv/main/bin/python sba_standalone/scratch/collate_results.py
/venv/main/bin/python sba_standalone/scratch/collate_evals.py

echo "🎉 Rigorous benchmark sweep finished successfully!"
