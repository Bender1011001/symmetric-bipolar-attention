#!/bin/bash
# run_training.sh
# Kicks off the 3-stage softmax-bridge conversion for Gemma-4-E4B on a single GPU.
# Memory optimized to fit within 32 GB VRAM (caps sequence length to 512 and loss chunk size to 32).

echo "Launching optimized Gemma 4 SBA Conversion training..."

nohup accelerate launch --num_processes=1 run_softmax_bridge_conversion.py \
  --model_name_or_path google/gemma-4-E4B \
  --dataset_name wikitext \
  --dataset_config wikitext-103-raw-v1 \
  --text_column text \
  --output_dir ./results/gemma4_e4b_sba_bridge \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --logging_steps 10 \
  --save_steps 500 \
  --mixed_precision bf16 \
  --max_seq_length 512 \
  --loss_chunk_size 32 \
  > gemma4_conversion.log 2>&1 &

echo "Training launched in the background!"
echo "To monitor progress, run: tail -f gemma4_conversion.log"
