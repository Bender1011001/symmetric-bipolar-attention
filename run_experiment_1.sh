#!/bin/bash
# run_experiment_1.sh
# Experiment 1: Softmax-to-SBA Bridge Conversion with Orthogonal Projection and Disabled EAR
# Copyright (c) 2026.
# SPDX-License-Identifier: Apache-2.0

set -e

OUT_DIR="/root/sba_workspace/results/gemma4_e4b_sba_distill_bridge_ablation_ear"
mkdir -p "$OUT_DIR"

cd /root/sba_workspace

echo "================================================================="
echo "Starting Experiment 1: EAR Ablation (Orthogonal Projection + Disable EAR)"
echo "Resuming from checkpoint-1500 (Step 1500 to 5000)"
echo "Effective batch size: 32 (batch_size=4, accumulation=8)"
echo "================================================================="

python3 run_softmax_bridge_conversion.py \
    --model_name_or_path google/gemma-4-E4B \
    --train_file data/sba_self_distill.jsonl \
    --text_column text \
    --output_dir "$OUT_DIR" \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
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
    --disable_ear \
    --resume_from_checkpoint results/gemma4_e4b_sba_distill_bridge_b200/checkpoint-1500 \
    2>&1 | tee "$OUT_DIR/gemma4_distill_bridge_ablation_ear.log"

echo "================================================================="
echo "✅ Experiment 1 Completed Successfully!"
echo "================================================================="
