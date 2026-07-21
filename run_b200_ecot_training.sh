#!/bin/bash
# run_b200_ecot_training.sh
# Optimization script for running Gemma 4 SBA Bridge Conversion on B200 (180GB VRAM)

set -e

# Output folder for results
OUT_DIR="/root/sba_standalone/results/gemma4_e4b_sba_ecot_b200"
mkdir -p "$OUT_DIR"

cd /root/sba_standalone

echo "================================================================="
# B200 Optimization parameters:
# - Batch size increased to 16 per device (due to 180GB HBM3e VRAM)
# - Max sequence length set to 1024 (full reasoning traces without truncation)
# - Total training steps set to 5000 (approx. 3.2 epochs on 50k examples)
# - Homotopy Stages:
#     - Stage 0 (Preservation): 500 steps (freeze softmax reference)
#     - Stage 1 (Transition): 1000 steps (homotopy ramp-up of SBA row losses)
#     - Stage 2 (Recovery): 3500 steps (loss recovery and alignment tuning)
# =================================================================

/venv/main/bin/accelerate launch run_softmax_bridge_conversion.py \
    --model_name_or_path google/gemma-4-E4B \
    --train_file data/ecot_train.jsonl \
    --text_column text \
    --output_dir "$OUT_DIR" \
    --max_seq_length 1024 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2.0e-4 \
    --bridge_learning_rate 5.0e-5 \
    --stage0_steps 500 \
    --stage1_steps 1000 \
    --stage2_steps 3500 \
    --loss_chunk_size 128 \
    --save_steps 500 \
    --logging_steps 10 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --seed 17 \
    2>&1 | tee "$OUT_DIR/gemma4_ecot_b200_training.log"

echo "================================================================="
echo "✅ Training Completed Successfully! Checkpoints saved to $OUT_DIR"
echo "================================================================="
