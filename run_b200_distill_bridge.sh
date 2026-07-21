#!/bin/bash
# run_b200_distill_bridge.sh
# Optimization script for running Gemma 4 SBA Bridge Conversion on B200 using self-distillation dataset
# Copyright (c) 2026.
# SPDX-License-Identifier: Apache-2.0

set -e

# Output folder for results
OUT_DIR="/root/sba_standalone/results/gemma4_e4b_sba_distill_bridge_b200"
mkdir -p "$OUT_DIR"

cd /root/sba_standalone

echo "================================================================="
# B200 Optimization parameters:
# - Batch size set to 16 per device (due to 180GB HBM3e VRAM)
# - Max sequence length set to 512 (context length of self-distillation data)
# - Total training steps set to 10000 (stage0 + stage1 + stage2)
# - Homotopy Stages:
#     - Stage 0 (Preservation): 500 steps (freeze softmax reference)
#     - Stage 1 (Transition): 1500 steps (homotopy ramp-up of SBA row losses)
#     - Stage 2 (Recovery): 8000 steps (loss recovery and alignment tuning)
# =================================================================

/venv/main/bin/accelerate launch run_softmax_bridge_conversion.py \
    --model_name_or_path google/gemma-4-E4B \
    --train_file data/sba_self_distill.jsonl \
    --text_column text \
    --output_dir "$OUT_DIR" \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2.0e-4 \
    --bridge_learning_rate 5.0e-5 \
    --stage0_steps 300 \
    --stage1_steps 1000 \
    --stage2_steps 3700 \
    --loss_chunk_size 32 \
    --save_steps 500 \
    --logging_steps 10 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --seed 17 \
    --orthogonal_projection \
    2>&1 | tee "$OUT_DIR/gemma4_distill_bridge_b200_training.log"

echo "================================================================="
echo "✅ Training Completed Successfully! Checkpoints saved to $OUT_DIR"
echo "================================================================="
